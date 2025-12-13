import pytest
import torch

from cell_observatory_platform.models.layers.patch_embeddings import PatchEmbedding, calc_num_patches

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
