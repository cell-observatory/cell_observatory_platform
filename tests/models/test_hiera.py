import math

import pytest

import torch

from cell_observatory_platform.models.backbones.hiera import Hiera
from cell_observatory_platform.models.layers.utils import Reroll, Unroll, do_pool_stride


ZYXC_CFG = dict(
    input_fmt="ZYXC",
    input_shape=(128, 128, 128, 1),
    patch_shape=(4, 8, 8, None),
    embed_dim=32,
    num_heads=2,
    q_pool=1,
    q_stride=(2, 2, 2),
    stages=(1, 1, 1, 1),
    mask_unit_size=(2, 2, 2),
)

TZYXC_CFG = dict(
    input_fmt="TZYXC",
    input_shape=(16, 128, 128, 128, 1),
    patch_shape=(2, 4, 8, 8, None),
    embed_dim=32,
    num_heads=2,
    q_pool=1,
    q_stride=(1, 2, 2, 2),
    stages=(1, 1, 1, 1),
    mask_unit_size=(1, 2, 2, 2),
)

# small config for the constructor guards: ZYXC (32, 64, 64, 1), patch (4, 8, 8) -> tokens (8, 8, 8)
_HIERA_CFG = dict(
    input_fmt="ZYXC",
    input_shape=(32, 64, 64, 1),
    patch_shape=(4, 8, 8, None),
    embed_dim=16,
    num_heads=1,
    drop_path_rate=0.0,
    q_pool=2,
    q_stride=(2, 2, 2),
    stages=(1, 1, 1, 1),
    norm_layer="LayerNorm",
    mask_unit_size=(2, 2, 2),
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
    """One intermediate per stage end; unmasked intermediates are rerolled to a
    batch-first, channels-last layout with that stage's output width."""
    model = Hiera(**cfg)
    B = 1
    x = torch.randn((B,) + cfg["input_shape"])

    out, intermediates, patches = model(x, return_intermediates=True)

    assert isinstance(intermediates, list)
    assert len(intermediates) == len(model.stage_ends)
    for inter, end in zip(intermediates, model.stage_ends):
        assert inter.shape[0] == B
        assert inter.shape[-1] == model.blocks[end].dim_out
        assert torch.isfinite(inter).all()
    assert out.shape[-1] == model.blocks[-1].dim_out


@pytest.mark.parametrize("cfg", [ZYXC_CFG, TZYXC_CFG], ids=["ZYXC", "TZYXC"])
def test_hiera_masked_forward(cfg):
    """With ctx_idx the encoder keeps only the listed mask units (flat_mu_size tokens
    each) and q-pools them by the same factor as the unmasked path; a mask without
    ctx_idx is refused."""
    torch.manual_seed(0)
    model = Hiera(**cfg)
    B = 2
    N = _num_tokens(cfg)
    x = torch.randn((B,) + cfg["input_shape"])

    num_mus = N // model.flat_mu_size
    keep = max(1, num_mus // 2)
    mask = torch.ones(B, N, dtype=torch.bool)  # non-None flag; ctx_idx drives the gather
    ctx_idx = torch.stack([torch.randperm(num_mus)[:keep] for _ in range(B)])

    out, patches = model(x, mask=mask, ctx_idx=ctx_idx)

    # kept mask units x tokens per unit, then q-pooled by the same factor as the unmasked path
    expected_tokens = keep * model.flat_mu_size * _num_output_tokens(cfg) // N
    assert out.shape == (B, expected_tokens, model.blocks[-1].dim_out)
    assert patches.shape == (B, N, model.patch_embed.pixels_per_patch)
    assert torch.isfinite(out).all()

    with pytest.raises(ValueError, match="ctx_idx"):
        model(x, mask=mask, ctx_idx=None)


def test_hiera_rejects_q_stride_of_wrong_rank():
    """q_stride must have the token-grid rank (3 for ZYXC); a rank-2 tuple is refused."""
    with pytest.raises(ValueError, match="q_stride .* rank"):
        Hiera(**{**_HIERA_CFG, "q_stride": (2, 2)})


def test_hiera_rejects_mask_unit_size_of_wrong_rank():
    """mask_unit_size must have the token-grid rank; a rank-2 tuple is refused."""
    with pytest.raises(ValueError, match="mask_unit_size .* rank"):
        Hiera(**{**_HIERA_CFG, "mask_unit_size": (2, 2)})


def test_hiera_int_q_stride_expands_to_grid_rank():
    """An int q_stride is broadcast to one entry per spatial axis of the token grid."""
    enc = Hiera(**{**_HIERA_CFG, "q_stride": 2})
    assert enc.q_stride == (2, 2, 2)


# Unroll layout: the mask-unit index is the FASTEST token axis, and the consumers
# (masking view, do_pool_stride, Reroll) agree with it.
GRID = (8, 8, 8)
SCHEDULE = [(2, 2, 2), (2, 2, 2)]
MU = tuple(a * b for a, b in zip(*SCHEDULE))  # (4, 4, 4)
FLAT_MU = int(math.prod(MU))                  # 64
NUM_MUS = int(math.prod(GRID)) // FLAT_MU     # 8


def _unrolled_coords():
    N = int(math.prod(GRID))
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(GRID[0]), torch.arange(GRID[1]), torch.arange(GRID[2]),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(1, N, 3).float()
    unroll = Unroll(input_size=GRID, patch_stride=(1, 1, 1), unroll_schedule=SCHEDULE)
    return coords, unroll(coords)


def _is_one_mu_block(group_coords: torch.Tensor) -> bool:
    mins = group_coords.min(dim=0).values
    maxs = group_coords.max(dim=0).values
    if not torch.equal(maxs - mins + 1, torch.tensor(MU, dtype=group_coords.dtype)):
        return False
    return (
        group_coords.unique(dim=0).shape[0] == FLAT_MU
        and group_coords.shape[0] == FLAT_MU
    )


def test_unroll_puts_mask_unit_index_on_fastest_token_axis():
    """After Unroll, view(B, flat_mu, num_mus, C) groups tokens by mask unit:
    every group is one contiguous MU block. The transposed view is not."""
    _, u = _unrolled_coords()
    g = u.view(1, FLAT_MU, NUM_MUS, 3)
    for m in range(NUM_MUS):
        assert _is_one_mu_block(g[0, :, m].long()), f"group {m} is not one MU block"
    # the opposite convention must NOT satisfy the property, else this test
    # could not tell the two layouts apart
    g_slow = u.view(1, NUM_MUS, FLAT_MU, 3)
    assert not all(_is_one_mu_block(g_slow[0, m].long()) for m in range(NUM_MUS))


def test_reroll_inverts_unroll():
    """Reroll at stage 0 (no pooling) restores the original raster token order."""
    coords, u = _unrolled_coords()
    reroll = Reroll(
        input_size=GRID, patch_stride=(1, 1, 1), unroll_schedule=SCHEDULE,
        stage_ends=[0, 1, 2], q_pool=2,
    )
    r = reroll(u, 0).reshape(1, -1, 3)
    assert torch.equal(r, coords)


def test_do_pool_stride_pools_within_mask_units():
    """After unroll, pooling with the flattened q_stride combines tokens of the SAME
    mask unit (leading stride axis), never across units."""
    _, u = _unrolled_coords()
    stride = int(math.prod(SCHEDULE[0]))  # 8
    pooled = do_pool_stride(u, stride)  # [1, N//8, 3] (max over coords)
    # For coordinate payloads, max over one mask-unit sub-block stays inside
    # that unit's bounding box. Check that each pooled token's coords lie inside
    # the mask unit of its (still MU-fastest) group.
    g = pooled.view(1, FLAT_MU // stride, NUM_MUS, 3)
    for m in range(NUM_MUS):
        grp = g[0, :, m].long()
        span = grp.max(dim=0).values - grp.min(dim=0).values
        assert torch.all(span < torch.tensor(MU)), (
            "do_pool_stride mixed tokens across mask units"
        )
