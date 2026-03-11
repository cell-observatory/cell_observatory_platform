import pytest
import torch

from cell_observatory_platform.models.layers.patch_embeddings import (
    PatchEmbedding,
    ChannelAdaptivePatchEmbedding,
    calc_num_patches,
)


CASES = [
    # 5D: TZYXC
    dict(
        name="TZYXC",
        input_fmt="TZYXC",
        input_shape=(2, 8, 16, 32, 32, 3),  # (B, T, Z, Y, X, C)
        lateral_patch_size=8,
        patch_shape=(2, 4, 8, 8, None),
        axial_patch_size=4,
        temporal_patch_size=2,
        channels=3,
        embed_dim=64,
    ),
    # 4D: ZYXC
    dict(
        name="ZYXC",
        input_fmt="ZYXC",
        input_shape=(2, 12, 32, 32, 1),  # (B, Z, Y, X, C)
        lateral_patch_size=8,
        axial_patch_size=3,
        temporal_patch_size=None,
        patch_shape=(3, 8, 8, None),
        channels=1,
        embed_dim=96,
    ),
    # 4D: TYXC
    dict(
        name="TYXC",
        input_fmt="TYXC",
        input_shape=(2, 10, 24, 24, 2),  # (B, T, Y, X, C)
        lateral_patch_size=6,
        axial_patch_size=None,
        temporal_patch_size=5,
        patch_shape=(5, 6, 6, None),
        channels=2,
        embed_dim=48,
    ),
    # 3D: YXC
    dict(
        name="YXC",
        input_fmt="YXC",
        input_shape=(2, 28, 20, 1),  # (B, Y, X, C)
        lateral_patch_size=4,
        axial_patch_size=None,
        temporal_patch_size=None,
        patch_shape=(4, 4, None),
        channels=1,
        embed_dim=32,
    ),
    # 2D: XC
    dict(
        name="XC",
        input_fmt="XC",
        input_shape=(2, 30, 4),  # (B, X, C)
        lateral_patch_size=5,
        axial_patch_size=None,
        temporal_patch_size=None,
        patch_shape=(5, None),
        channels=4,
        embed_dim=16,
    ),
]


def _expected_pixels_per_patch(case):
    t = case.get("temporal_patch_size") or 1
    z = case.get("axial_patch_size") or 1
    l = case["lateral_patch_size"]
    return case["channels"] * t * z * (l**2) if case["input_fmt"] != "XC" else case["channels"] * t * z * l


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_patchify_shapes_reshape(case):
    pe = PatchEmbedding(
        input_fmt=case["input_fmt"],
        input_shape=case["input_shape"][1:],
        patch_shape=case["patch_shape"],
        embed_dim=case["embed_dim"],
        channels=case["channels"],
    )

    x = torch.randn(case["input_shape"])
    num_patches, token_shape = calc_num_patches(
        input_fmt=case["input_fmt"],
        input_shape=case["input_shape"][1:],
        patch_shape=case["patch_shape"],
    )

    exp_pixels = _expected_pixels_per_patch(case)
    assert pe.pixels_per_patch == exp_pixels

    patches = pe._patchify(x, reshape=True)
    assert patches.shape == (case["input_shape"][0], num_patches, exp_pixels)

    out = pe(x)
    assert out.shape == (case["input_shape"][0], num_patches, case["embed_dim"])


# ---------------------------------------------------------------------------
# ChannelAdaptivePatchEmbedding
# ---------------------------------------------------------------------------


CA_CASES = [
    dict(
        name="ZYXC",
        input_fmt="ZYXC",
        input_shape=(2, 12, 16, 16, 3),  # B, Z, Y, X, C
        patch_shape=(4, 8, 8, None),
        embed_dim=32,
        channels=3,
    ),
    dict(
        name="TZYXC",
        input_fmt="TZYXC",
        input_shape=(2, 4, 8, 16, 16, 2),  # B, T, Z, Y, X, C
        patch_shape=(2, 4, 8, 8, None),
        embed_dim=32,
        channels=2,
    )
]


def _ca_num_patches(case):
    num_patches, _ = calc_num_patches(
        input_fmt=case["input_fmt"],
        input_shape=case["input_shape"][1:],
        patch_shape=case["patch_shape"],
    )
    return num_patches


@pytest.mark.parametrize("case", CA_CASES, ids=[c["name"] for c in CA_CASES])
def test_channel_adaptive_concat_shape(case):
    pe = ChannelAdaptivePatchEmbedding(
        input_fmt=case["input_fmt"],
        patch_shape=case["patch_shape"],
        embed_dim=case["embed_dim"],
        max_channels=16,
        channel_fusion="concat",
    )
    x = torch.randn(case["input_shape"])
    out = pe(x)
    B = case["input_shape"][0]
    C = case["channels"]
    N = _ca_num_patches(case)
    assert out.shape == (B, N * C, case["embed_dim"])


@pytest.mark.parametrize("case", CA_CASES, ids=[c["name"] for c in CA_CASES])
def test_channel_adaptive_attn_pool_shape(case):
    pe = ChannelAdaptivePatchEmbedding(
        input_fmt=case["input_fmt"],
        patch_shape=case["patch_shape"],
        embed_dim=case["embed_dim"],
        max_channels=8,
        channel_fusion="attn_pool",
        attn_pool_num_heads=4,
    )
    x = torch.randn(case["input_shape"])
    out = pe(x)
    B = case["input_shape"][0]
    N = _ca_num_patches(case)
    assert out.shape == (B, N, case["embed_dim"])


def test_channel_adaptive_return_patches():
    pe = ChannelAdaptivePatchEmbedding(
        input_fmt="ZYXC",
        patch_shape=(4, 8, 8, None),
        embed_dim=32,
        max_channels=8,
        channel_fusion="concat",
    )
    x = torch.randn(1, 12, 16, 16, 3)
    out, patches, token_shape = pe(x, return_patches=True)
    assert patches.ndim == 4  # [B, N, C, P]
    assert patches.shape[2] == 3  # C
    assert isinstance(token_shape, tuple)


def test_channel_adaptive_with_channel_ids():
    pe = ChannelAdaptivePatchEmbedding(
        input_fmt="ZYXC",
        patch_shape=(4, 8, 8, None),
        embed_dim=32,
        max_channels=16,
        channel_fusion="concat",
    )
    x = torch.randn(2, 12, 16, 16, 3)
    ids = torch.tensor([5, 10, 15])
    out = pe(x, channel_ids=ids)
    B, N_C, D = out.shape
    assert B == 2
    assert D == 32