import pytest
import torch
import torch.nn as nn

from cell_observatory_platform.training.helpers import init_weights
import cell_observatory_platform.training.helpers as M


# ---- Tiny scaffold model that satisfies attribute expectations ----


class _DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)


class _DummyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(4, 4, bias=False)


class _DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = _DummyAttention()
        self.mlp = _DummyMLP()


class _EncoderWithBlocks(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_DummyBlock() for _ in range(n_layers)])


class _MaskedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # patch_embedding.proj.weight used in MAE path (Conv2d expected)
        self.patch_embedding = nn.Module()
        self.patch_embedding.proj = nn.Conv2d(3, 4, kernel_size=2, bias=False)


class _MaskedDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # token_param touched in MAE/VJEPA paths
        self.token_param = nn.Parameter(torch.zeros(1, 4))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_std = 0.02
        self.masked_encoder = _MaskedEncoder()
        self.masked_decoder = _MaskedDecoder()
        # include convs so Conv2d/Conv3d branches are visited by .apply()
        self.some_conv2d = nn.Conv2d(3, 3, kernel_size=1)
        self.some_conv3d = nn.Conv3d(1, 1, kernel_size=1)
        # objects that the rescale helpers expect:
        self.input_encoder = nn.Module()
        self.input_encoder.encoder = _EncoderWithBlocks()
        self.target_predictor = nn.Module()
        self.target_predictor.encoder = _EncoderWithBlocks()


# ---- Helpers ----


def _fake_trunc_normal_(tensor, std=0.02, **kwargs):
    with torch.no_grad():
        return tensor.normal_(mean=0.0, std=std)


# ---- Tests ----


def test_init_weights_mae_no_throw():
    model = TinyModel()
    init_weights(model, "mae")  # should not raise

def test_init_weights_vjepa_no_throw(monkeypatch):
    # ensure trunc_normal_ is present in the module under test
    monkeypatch.setattr(M, "trunc_normal_", _fake_trunc_normal_, raising=False)
    model = TinyModel()
    init_weights(model, "vjepa")  # should not raise

def test_init_weights_vjepa2_no_throw(monkeypatch):
    monkeypatch.setattr(M, "trunc_normal_", _fake_trunc_normal_, raising=False)
    model = TinyModel()
    init_weights(model, "vjepa2")  # should not raise

def test_init_weights_unknown_raises():
    model = TinyModel()
    with pytest.raises(ValueError):
        init_weights(model, "not-a-real-init")