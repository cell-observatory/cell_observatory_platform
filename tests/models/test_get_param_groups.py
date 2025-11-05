import re
from types import SimpleNamespace as NS

import torch
import torch.nn as nn
import pytest

from training.schedulers import get_param_groups  


# ------------------------- dummy models -------------------------


class DummyStack(nn.Module):
    """Has names that exercise MAE path rules."""
    def __init__(self, L=2, dim=4):
        super().__init__()
        self._L = L
        self.patch_embedding = nn.Linear(dim, dim, bias=False)  # -> lid=0
        self.pos_embedding = nn.Parameter(torch.zeros(dim))   # no-wd by suffix/1D
        self.cls_token = nn.Parameter(torch.zeros(dim))   # no-wd
        self.token_param = nn.Parameter(torch.zeros(dim))   # no-wd
        self.transformer_blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(L)])  # -> lid=i+1
        self.norm = nn.LayerNorm(dim) # -> lid=L
        self.output_projection = nn.Linear(dim, dim, bias=False) # -> lid=L

    def get_num_layers(self):
        return self._L


class SimpleEncoder(nn.Module):
    """Used by vjepa / vjepa2 branches."""
    def __init__(self, din=4, dout=3):
        super().__init__()
        self.fc = nn.Linear(din, dout)  # 2D weight (decay), 1D bias (no-wd)
        self.ln = nn.LayerNorm(dout)    # 1D weight/bias (no-wd)


class DummyModelJepa(nn.Module):
    def __init__(self, enc_L=2, dec_L=1):
        super().__init__()
        self.input_encoder = SimpleEncoder()
        self.target_predictor = SimpleEncoder()


class DummyModelMAE(nn.Module):
    def __init__(self, enc_L=2, dec_L=1):
        super().__init__()
        self.masked_encoder = DummyStack(enc_L)
        self.masked_decoder = DummyStack(dec_L)


# ------------------------- helpers -------------------------


def _cfg_default_disabled():
    return NS(optimizers=NS(param_group_split_mode=False))

def _cfg_mae(layer_decay=0.8, decoder_layer_decay=0.9, weight_decay=0.05, no_wd_list=()):
    return NS(optimizers=NS(
        param_group_split_mode="mae",
        layer_decay=layer_decay,
        decoder_layer_decay=decoder_layer_decay,
        wd=weight_decay,
        no_weight_decay_list=list(no_wd_list),
    ))

def _cfg_vjepa():
    return NS(optimizers=NS(param_group_split_mode="vjepa"))

def _cfg_vjepa2(zero_init_bias_wd=True):
    return NS(optimizers=NS(param_group_split_mode="vjepa2", zero_init_bias_wd=zero_init_bias_wd))


def _all_params_named(module: nn.Module):
    return dict(module.named_parameters())

def _flatten_params_from_groups(groups):
    out = []
    for g in groups:
        ps = g["params"]
        out.extend(list(ps))
    return out

def _find_group_for_param(groups, p):
    for g in groups:
        if any(pp is p for pp in g["params"]):
            return g
    return None


# ------------------------- tests -------------------------


def test_get_param_groups_disabled_returns_model_parameters():
    model = DummyModelJepa()
    cfg = _cfg_default_disabled()

    got = list(get_param_groups(cfg, model))
    expected = list(model.parameters())

    assert len(got) == len(expected), "When disabled, should return model.parameters()"
    # identity check (order may differ between versions, so compare by object id)
    assert {id(p) for p in got} == {id(p) for p in expected}


def test_get_param_groups_mae_minimal_grouping_and_scales():
    model = DummyModelMAE(enc_L=2, dec_L=1)
    cfg = _cfg_mae(layer_decay=0.8, decoder_layer_decay=0.9, weight_decay=0.05)

    groups = get_param_groups(cfg, model)  # list of dicts with 'lr_scale','weight_decay','params'
    assert isinstance(groups, list) and len(groups) > 0

    # every trainable param under masked_(encoder|decoder) should appear exactly once
    all_named = {n: p for n, p in model.named_parameters()
                 if n.startswith("masked_encoder.") or n.startswith("masked_decoder.")}
    flat = _flatten_params_from_groups(groups)
    assert len({id(p) for p in flat}) == len(all_named), "Each param should appear exactly once"
    assert all(any(p is q for q in flat) for p in all_named.values())

    # quick spot checks on a few specific params
    name_map = _all_params_named(model)

    # encoder layer/scale expectations
    enc_L = model.masked_encoder.get_num_layers()
    enc_scales = [cfg.optimizers.layer_decay ** (enc_L - i) for i in range(enc_L + 1)]

    # decoder layer/scale expectations
    dec_L = model.masked_decoder.get_num_layers()
    dec_scales = [cfg.optimizers.decoder_layer_decay ** (dec_L - i) for i in range(dec_L + 1)]

    # 1) encoder.patch_embedding.weight -> lid=0, decay
    p1 = name_map["masked_encoder.patch_embedding.weight"]
    g1 = _find_group_for_param(groups, p1)
    assert g1 is not None
    assert pytest.approx(g1["lr_scale"]) == enc_scales[0]
    assert pytest.approx(g1["weight_decay"]) == cfg.optimizers.wd

    # 2) encoder.transformer_blocks.0.weight -> lid=1, decay
    p2 = name_map["masked_encoder.transformer_blocks.0.weight"]
    g2 = _find_group_for_param(groups, p2)
    assert g2 is not None
    assert pytest.approx(g2["lr_scale"]) == enc_scales[1]
    assert pytest.approx(g2["weight_decay"]) == cfg.optimizers.wd

    # 3) encoder.norm.weight (1D) -> lid=L, NO decay
    p3 = name_map["masked_encoder.norm.weight"]
    g3 = _find_group_for_param(groups, p3)
    assert g3 is not None
    assert pytest.approx(g3["lr_scale"]) == enc_scales[enc_L]
    assert g3["weight_decay"] == 0.0

    # 4) decoder.output_projection.weight -> lid=dec_L, decay
    p4 = name_map["masked_decoder.output_projection.weight"]
    g4 = _find_group_for_param(groups, p4)
    assert g4 is not None
    assert pytest.approx(g4["lr_scale"]) == dec_scales[dec_L]
    assert pytest.approx(g4["weight_decay"]) == cfg.optimizers.wd

    # 5) encoder.pos_embedding (ALWAYS_NO_WD_SUFFIX) -> no decay
    p5 = name_map["masked_encoder.pos_embedding"]
    g5 = _find_group_for_param(groups, p5)
    assert g5 is not None and g5["weight_decay"] == 0.0


def test_get_param_groups_vjepa_simple_split():
    model = DummyModelJepa()
    cfg = _cfg_vjepa()

    groups = get_param_groups(cfg, model)
    assert isinstance(groups, list) and len(groups) == 4  # exactly as defined

    # materialize counts
    counts = [len(list(g["params"])) for g in groups]
    # For our SimpleEncoder: one 2D weight (decay) + three 1D/bias (no-wd)
    # So per encoder: decay=1, no-wd=3. Across input+target: [1, 1, 3, 3]
    assert counts[0] == 1  # input_encoder decay
    assert counts[1] == 1  # target_predictor decay
    assert counts[2] == 3 and groups[2].get("weight_decay", None) == 0  # input no-wd
    assert counts[3] == 3 and groups[3].get("weight_decay", None) == 0  # target no-wd

    # Sanity: no parameter duplication
    flat = _flatten_params_from_groups(groups)
    assert len({id(p) for p in flat}) == len(flat)


@pytest.mark.parametrize("zero_init_bias_wd", [True, False])
def test_get_param_groups_vjepa2_zero_init_flag(zero_init_bias_wd):
    model = DummyModelJepa()
    cfg = _cfg_vjepa2(zero_init_bias_wd=zero_init_bias_wd)

    groups = get_param_groups(cfg, model)
    assert isinstance(groups, list) and len(groups) == 4

    # groups[2] and groups[3] are the no-wd groups; check WD_exclude flag echoes config
    assert groups[2]["weight_decay"] == 0 and groups[2].get("WD_exclude", None) == zero_init_bias_wd
    assert groups[3]["weight_decay"] == 0 and groups[3].get("WD_exclude", None) == zero_init_bias_wd

    # quick count sanity like above
    counts = [len(list(g["params"])) for g in groups]
    assert counts[0] == 1 and counts[1] == 1 and counts[2] == 3 and counts[3] == 3

    # no duplication
    flat = _flatten_params_from_groups(groups)
    assert len({id(p) for p in flat}) == len(flat)