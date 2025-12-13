import pytest

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from omegaconf import open_dict

from cell_observatory_platform.tests.conftest import config
from cell_observatory_platform.training.helpers import (
    apply_activation_checkpointing,
    _apply_ac_to_module,
)


# TODO: add tests to test with all the models we currently support
#       and all permutations of activation checkpointing options


class MLPBlock(nn.Module):
    def __init__(self, d_model=32, hidden=48):
        super().__init__()
        self.proj1 = nn.Linear(d_model, hidden, bias=False)
        self.proj2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        h = x.reshape(B * T, D)
        h = h @ self.proj1.weight.t()
        h = torch.nn.functional.relu(h)
        h = h @ self.proj2.weight.t()
        return h.view(B, T, -1)


class TinyModel(nn.Module):
    """
    Tiny model with a stack of blocks under encoder.layers so that
    apply_activation_checkpointing can discover them via:
    modules = ["encoder"], block_names = "layers".
    """

    def __init__(self, d_model=32):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([MLPBlock(d_model) for _ in range(4)])
        self.out = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):
        for blk in self.encoder.layers:
            x = blk(x)
        return self.out(x).mean()


# reference: https://dev-discuss.pytorch.org/t/torchdispatchmode-for-debugging-testing-and-more/717
class MMCallCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Count only aten.mm.default calls
        if func is torch.ops.aten.mm.default:
            self.count += 1
        return func(*args, **kwargs)


def run_and_count_mm_calls(block: nn.Module, x: torch.Tensor) -> int:
    x_ = x.detach().requires_grad_(True)
    with MMCallCounter() as mode:
        y = block(x_)
        y.sum().backward()
        return mode.count


# helper to count autograd saved tensors
def count_saved_tensors(run_fn):
    saved = []
    def pack(t):
        saved.append(t)
        return t
    def unpack(t):
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        run_fn()
    return len(saved)


def _get_input(B=2, T=8, D=32):
    torch.manual_seed(0)
    return torch.randn(B, T, D, requires_grad=True)


def make_config(
    config=config,
    ac_enabled: bool = True,
    mode: str = "full",
    selective_ac_option=None,
    fqn_filters=None,
    mm_recompute_frac: int = 2,
):
    with open_dict(config):
        if "models" not in config.optimizations:
            config.optimizations.models = {}
        if "activation_checkpoint" not in config.optimizations.models:
            config.optimizations.models.activation_checkpoint = {}

        ac_cfg = config.optimizations.models.activation_checkpoint
        ac_cfg.enabled = ac_enabled
        ac_cfg.mode = mode
        ac_cfg.selective_ac_option = selective_ac_option
        ac_cfg.per_op_sac_force_recompute_mm_shapes_by_fqns = fqn_filters
        ac_cfg.mm_recompute_frac = mm_recompute_frac

        ac_cfg.modules = ["encoder"]
        ac_cfg.block_names = "layers"

    return config


def block_saved_tensors(block: nn.Module, x: torch.Tensor) -> int:
    def run():
        y = block(x.detach().requires_grad_(True))
        y.sum().backward()
    return count_saved_tensors(run)


def test_full_wraps_all_layers_and_effect_per_block(config):
    model = TinyModel()
    cfg = make_config(config, mode="full")
    apply_activation_checkpointing(cfg, model)

    wrapped_flags = [isinstance(m, CheckpointWrapper) for m in model.encoder.layers]
    assert all(wrapped_flags), f"full mode: expected all layers wrapped, got {wrapped_flags}"

    # effect: each wrapped block should save fewer tensors than its unwrapped counterpart
    x = _get_input()
    # make a fresh, unwrapped reference block with identical shapes
    ref_block = MLPBlock()
    # compare per-block (shape-identical) saved tensors
    for i, blk in enumerate(model.encoder.layers):
        wrapped_cnt = block_saved_tensors(blk, x)
        ref_cnt = block_saved_tensors(ref_block, x)
        assert wrapped_cnt < ref_cnt, f"layer {i}: wrapped should save fewer ({wrapped_cnt} < {ref_cnt})"


def test_selective_op_wraps_all_layers_and_targets_fqn_layer(config):
    model = TinyModel()
    cfg = make_config(
        config,
        mode="selective",
        selective_ac_option="op",
        fqn_filters=["encoder.layers.0.proj1"],
        mm_recompute_frac=8,
    )
    apply_activation_checkpointing(cfg, model)

    # all wrapped structurally
    wrapped_flags = [isinstance(m, CheckpointWrapper) for m in model.encoder.layers]
    assert all(wrapped_flags), f"op-selective: expected all layers wrapped, got {wrapped_flags}"

    # targeted layer should recompute its first mm in backward -> more mm calls overall
    x = _get_input()
    mm0 = run_and_count_mm_calls(model.encoder.layers[0], x)
    mm1 = run_and_count_mm_calls(model.encoder.layers[1], x)
    assert mm0 > mm1, f"layer 0 should execute more mm ops due to recompute (got {mm0} vs {mm1})"


def test_selective_layer_frequency_wraps_every_kth_layer_exact(config):
    model = TinyModel()
    # every 2nd module (visit order) is wrapped by layer-frequency path
    cfg = make_config(config, mode="selective", selective_ac_option="2")
    apply_activation_checkpointing(cfg, model)

    wrapped = [isinstance(m, CheckpointWrapper) for m in model.encoder.layers]
    # with global counter starting at 0 and incrementing per visitation:
    # layer indices 1 and 3 should be wrapped for ac_freq=2 (since count%2==0 wraps 2nd, 4th, ...)
    assert wrapped == [False, True, False, True], f"expected [F,T,F,T], got {wrapped}"

    # effect: wrapped blocks (1,3) save fewer tensors than unwrapped neighbors (0,2)
    x = _get_input()
    cnts = [block_saved_tensors(blk, x) for blk in model.encoder.layers]
    assert cnts[1] < cnts[0], f"layer 1 should save fewer than layer 0 ({cnts[1]} < {cnts[0]})"
    assert cnts[3] < cnts[2], f"layer 3 should save fewer than layer 2 ({cnts[3]} < {cnts[2]})"


# --- for testing mm_recompute_frac which is hard to debug with model_tree_with_opt print ---


def _wrap_block_op_selective(block: nn.Module, mm_recompute_frac: int):
    """
    Wrap a single block with op-selective AC and the given mm_recompute_frac.
    We pass a base_fqn so policy gets deterministic per-op indexing.
    """
    return _apply_ac_to_module(
        module=block,
        act_ckpt_mode="selective",
        base_fqn="blk",
        selective_ac_option="op",
        per_op_sac_force_recompute_mm_shapes_by_fqns=None,
        mm_recompute_frac=mm_recompute_frac,
    )


def _mm_calls_for_mm_frac(mm_frac: int) -> int:
    """
    Returns the number of aten.mm.default calls during fwd+bwd for a wrapped block.
    More recomputation => more mm calls.
    """
    x = _get_input()
    blk = MLPBlock()
    wrapped = _wrap_block_op_selective(blk, mm_frac)
    return run_and_count_mm_calls(wrapped, x)


def test_op_mm_frac_respected_monotonic_mm_calls():
    """
    Expectation:
      Lower mm_frac => more recompute => more aten.mm calls across fwd+bwd.
    So: mm_calls(1) >= mm_calls(2) >= mm_calls(8)
    """
    c1 = _mm_calls_for_mm_frac(1)
    c2 = _mm_calls_for_mm_frac(2)
    c8 = _mm_calls_for_mm_frac(8)

    assert c1 >= c2, f"Expected mm_calls(frac=1) >= mm_calls(frac=2), got {c1} < {c2}"
    assert c2 >= c8, f"Expected mm_calls(frac=2) >= mm_calls(frac=8), got {c2} < {c8}"