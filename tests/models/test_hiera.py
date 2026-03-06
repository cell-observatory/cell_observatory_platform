import math

import pytest

import torch

from cell_observatory_platform.models.backbones.hiera import Hiera


ZYXC_CFG = dict(
    input_fmt="ZYXC",
    input_shape=(16, 32, 32, 1),   # Z=16, Y=32, X=32, C=1
    patch_shape=(4, 8, 8, None),   # Z_p=4, Y_p=8, X_p=8  => tokens (4,4,4)
    embed_dim=32,
    num_heads=2,
    q_pool=1,
    q_stride=(2, 2, 2),
    stages=(1, 1, 1, 1),
    mask_unit_size=(2, 2, 2),
)

TZYXC_CFG = dict(
    input_fmt="TZYXC",
    input_shape=(2, 32, 64, 64, 1),  # T=2, Z=32, Y=64, X=64, C=1 => tokens (1,8,8,8)
    patch_shape=(2, 4, 8, 8, None),  # T_p=2, Z_p=4, Y_p=8, X_p=8
    embed_dim=32,
    num_heads=2,
    q_pool=1,
    q_stride=(1, 2, 2, 2),
    stages=(1, 1, 1, 1),
    mask_unit_size=(1, 2, 2, 2),
)


def _num_tokens(cfg):
    """Initial patch count (before q-pooling)."""
    fmt = cfg["input_fmt"]
    shape = cfg["input_shape"]
    ps = cfg["patch_shape"]
    axis_to_val = dict(zip(fmt, shape))
    axis_to_ps = dict(zip(fmt, ps))
    t = (axis_to_val.get("T", 1) // axis_to_ps.get("T", 1)) if "T" in fmt else 1
    z = axis_to_val["Z"] // axis_to_ps["Z"]
    y = axis_to_val["Y"] // axis_to_ps["Y"]
    x = axis_to_val["X"] // axis_to_ps["X"]
    return t * z * y * x


def _num_output_tokens(cfg):
    """Token count after q-pooling (final model output)."""
    N = _num_tokens(cfg)
    q_stride = cfg.get("q_stride", (1, 1, 1))
    q_pool = cfg.get("q_pool", 0)
    if isinstance(q_stride, int):
        num_spatial = 4 if cfg.get("input_fmt") == "TZYXC" else 3
        q_stride_flat = q_stride**num_spatial
    else:
        q_stride_flat = int(math.prod(q_stride))
    return N // (q_stride_flat**q_pool)


@pytest.mark.parametrize("cfg", [ZYXC_CFG, TZYXC_CFG], ids=["ZYXC", "TZYXC"])
def test_hiera_forward_shape(cfg):
    model = Hiera(**cfg)
    B = 2
    input_shape = (B,) + cfg["input_shape"]
    x = torch.randn(input_shape)

    out, patches = model(x)
    N = _num_tokens(cfg)
    N_out = _num_output_tokens(cfg)
    final_dim = model.blocks[-1].dim_out
    assert out.shape[0] == B
    assert out.shape[1] == N_out
    assert out.shape[-1] == final_dim
    assert patches.shape == (B, N, model.patch_embed.pixels_per_patch)


@pytest.mark.parametrize("cfg", [ZYXC_CFG, TZYXC_CFG], ids=["ZYXC", "TZYXC"])
def test_hiera_return_intermediates(cfg):
    model = Hiera(**cfg)
    B = 1
    x = torch.randn((B,) + cfg["input_shape"])

    out, intermediates, patches = model(x, return_intermediates=True)
    assert isinstance(intermediates, list)
    assert len(intermediates) == len(model.stage_ends)


@pytest.mark.parametrize("cfg", [ZYXC_CFG, TZYXC_CFG], ids=["ZYXC", "TZYXC"])
def test_hiera_masked_forward(cfg):
    model = Hiera(**cfg)
    B = 2
    N = _num_tokens(cfg)
    x = torch.randn((B,) + cfg["input_shape"])

    num_mus = N // model.flat_mu_size
    keep = max(1, num_mus // 2)

    mask = torch.ones(B, N, dtype=torch.bool)
    ctx_idx = torch.stack([torch.randperm(num_mus)[:keep] for _ in range(B)])

    out, patches = model(x, mask=mask, ctx_idx=ctx_idx)
    expected_tokens = keep * model.flat_mu_size
    final_dim = model.blocks[-1].dim_out
    assert out.shape[0] == B
    assert out.shape[-1] == final_dim
