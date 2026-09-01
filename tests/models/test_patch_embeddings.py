import pytest
import torch

from cell_observatory_platform.models.layers.patch_embeddings import (
    PatchEmbedding,
    ChannelAdaptivePatchEmbedding,
    calc_num_patches,
    compute_num_pixels_per_patch,
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
    """return_patches exposes the [B, N, C, P] re-layout of the input alongside the token grid."""
    pe = ChannelAdaptivePatchEmbedding(
        input_fmt="ZYXC", patch_shape=(4, 8, 8, None), embed_dim=32, max_channels=8, channel_fusion="concat",
    )
    x = torch.randn(1, 12, 16, 16, 3)
    out, patches, token_shape = pe(x, return_patches=True)
    N, P = 3 * 2 * 2, 4 * 8 * 8
    assert patches.shape == (1, N, 3, P)            # [B, N, C, pixels_per_patch]
    assert token_shape == (None, 3, 2, 2, 3)        # (t, z, y, x, c)
    assert out.shape == (1, N * 3, 32)
    # patches are a pure re-layout of x: voxel (z=0,y=0,x=0,c=1) is patch 0, channel 1, pixel 0
    assert patches[0, 0, 1, 0] == x[0, 0, 0, 0, 1]


def test_channel_adaptive_channel_ids_change_output():
    """Explicit default ids reproduce the implicit output; other ids change it; [B, C] ids apply per sample."""
    torch.manual_seed(0)
    pe = ChannelAdaptivePatchEmbedding(
        input_fmt="ZYXC", patch_shape=(4, 8, 8, None), embed_dim=32, max_channels=16, channel_fusion="concat",
    ).eval()
    x = torch.randn(2, 8, 8, 8, 3)                     # N = 2 patches
    out_default = pe(x)                                # channel_ids=None -> ids 0..C-1
    out_same = pe(x, channel_ids=torch.tensor([0, 1, 2]))
    out_other = pe(x, channel_ids=torch.tensor([5, 10, 15]))
    out_per_sample = pe(x, channel_ids=torch.tensor([[0, 1, 2], [5, 10, 15]]))

    assert out_default.shape == (2, 2 * 3, 32)
    assert torch.equal(out_default, out_same)          # explicit default ids == implicit
    assert not torch.allclose(out_default, out_other)  # channel embedding is actually applied
    assert torch.allclose(out_per_sample[0], out_default[0]) and torch.allclose(out_per_sample[1], out_other[1])


@pytest.mark.parametrize(
    "ids,match",
    [
        (torch.tensor([0, 1, 16]), "out of range for max_channels=16"),
        (torch.tensor([0, 1, -1]), "out of range for max_channels=16"),
        (torch.tensor([0, 1]), "channel_ids has 2 but C=3"),
        (torch.tensor([[0, 1, 2]] * 3), r"channel_ids must be \[B,C\]"),
        (torch.zeros(1, 2, 3, dtype=torch.long), r"must be \[C\] or \[B,C\], got ndim=3"),
    ],
    ids=["too_large", "negative", "wrong_C", "wrong_B", "ndim3"],
)
def test_channel_adaptive_channel_ids_rejected(ids, match):
    """Out-of-range, wrong-length, wrong-batch and wrong-rank channel ids are rejected with a clear message."""
    pe = ChannelAdaptivePatchEmbedding(
        input_fmt="ZYXC", patch_shape=(4, 8, 8, None), embed_dim=32, max_channels=16, channel_fusion="concat",
    )
    x = torch.randn(2, 8, 8, 8, 3)
    with pytest.raises(ValueError, match=match):
        pe(x, channel_ids=ids)


# ------------------------------------------------------
# PatchEmbedding.patchify / _unpatchify layout contract
# ------------------------------------------------------

_PATCHIFY_KW = dict(
    temporal_patch_size=1,
    axial_patch_size=4,
    lateral_patch_size=4,
    token_shape=(1, 2, 2, 2, 2),   # t, z, y, x, c
    channels=2,
    num_patches=8,
    pixels_per_patch=4 * 4 * 4 * 2,
)


def test_patchify_unpatchify_round_trip():
    """_unpatchify(_patchify(x)) reproduces the channels-last input exactly."""
    torch.manual_seed(0)
    pe = PatchEmbedding(
        input_fmt="ZYXC", input_shape=(8, 8, 8, 2), patch_shape=(2, 2, 2),
        embed_dim=16, channels=2,
    )
    x = torch.randn(3, 8, 8, 8, 2)
    patches = pe._patchify(x)
    recon = pe._unpatchify(patches, out_channels=None)
    torch.testing.assert_close(recon, x)


def test_patchify_accepts_channels_last_tzyxc():
    """A (B, T, Z, Y, X, C) tensor patchifies to (B, num_patches, pixels_per_patch)."""
    x = torch.randn(1, 1, 8, 8, 8, 2)
    out = PatchEmbedding.patchify(x, input_format="TZYXC", **_PATCHIFY_KW)
    assert out.shape == (1, 8, 128)


def test_patchify_rejects_channels_first_with_matching_numel():
    """A channels-first (B*T, C, Z, Y, X) tensor has the same numel as the TZYXC
    tensor with T=1; patchify must refuse it instead of silently scrambling tokens."""
    x = torch.randn(1, 2, 8, 8, 8)
    with pytest.raises(ValueError, match="channels-last|D input"):
        PatchEmbedding.patchify(x, input_format="TZYXC", **_PATCHIFY_KW)


def test_patchify_rejects_wrong_rank():
    """A 5D tensor against a TZYXC layout (which needs 6D) is rejected."""
    x = torch.randn(1, 8, 8, 8, 2)
    with pytest.raises(ValueError, match="D input"):
        PatchEmbedding.patchify(x, input_format="TZYXC", **_PATCHIFY_KW)


@pytest.mark.parametrize("fmt,shape,kw", [
    ("TZYXC", (2, 1, 8, 8, 8, 2), _PATCHIFY_KW),
    ("ZYXC", (2, 8, 8, 8, 2), dict(
        temporal_patch_size=1, axial_patch_size=4, lateral_patch_size=4,
        token_shape=(1, 2, 2, 2, 2), channels=2, num_patches=8,
        pixels_per_patch=4 * 4 * 4 * 2,
    )),
])
def test_patchify_unfold_path_matches_reshape_path(fmt, shape, kw):
    """The unfold implementation emits pixels in the same order as the reshape one."""
    x = torch.randn(*shape)
    a = PatchEmbedding.patchify(x, input_format=fmt, reshape=True, **kw)
    b = PatchEmbedding.patchify(x, input_format=fmt, reshape=False, **kw)
    torch.testing.assert_close(a, b)


def test_unpatchify_out_channels_overrides_token_shape_channels():
    """Tokens holding C=1 pixels from a C_in=2 embedding reshape only with an
    explicit out_channels; the token-shape fallback (out_channels=None) raises."""
    pe = PatchEmbedding(
        input_fmt="ZYXC",
        input_shape=(8, 16, 16, 2),        # C_in = 2
        patch_shape=(4, 8, 8),
        embed_dim=16,
        channels=2,
    )
    n_tokens = 2 * 2 * 2
    pixels_c1 = 4 * 8 * 8 * 1              # decoder sized for C_out = 1
    x = torch.zeros(1, n_tokens, pixels_c1)
    with pytest.raises(RuntimeError):
        pe._unpatchify(x, out_channels=None)
    out = pe._unpatchify(x, out_channels=1)
    assert out.shape == (1, 8, 16, 16, 1)


def test_compute_num_pixels_per_patch_xc_uses_equality_not_identity():
    """A runtime-built "XC" (not the interned literal) must take the XC branch."""
    fmt = "".join(["X", "C"])
    assert compute_num_pixels_per_patch(1, None, None, 8, fmt) == 8
    assert compute_num_pixels_per_patch(2, None, 4, 8, "ZYXC") == 2 * 4 * 8 * 8
