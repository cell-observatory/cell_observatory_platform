import pytest
import torch


from cell_observatory_platform.models.adapters.vit_adapter import (
    ConvFFN,
    CrossAttention,
    DWConv,
    EncoderAdapter,
    Extractor,
    SpatialPriorModule,
)

try:
    from ops3d import _C
    OPS3D_AVAILABLE = True
except ImportError:
    OPS3D_AVAILABLE = False


# --- helpers ----


def _prod(tup):
    p = 1
    for v in tup:
        p *= int(v)
    return p


# --- CrossAttention --- #


@pytest.mark.parametrize(
    "B,Nq,Nk,C,H",
    [
        (2, 13, 29, 32, 4),
        (1, 7, 11, 64, 8),
    ],
)
def test_cross_attention_shapes(B, Nq, Nk, C, H):
    m = CrossAttention(dim=C, num_heads=H)
    q = torch.randn(B, Nq, C)
    k = v = torch.randn(B, Nk, C)
    out = m(q, k, v)
    assert out.shape == (B, Nq, C)


# --- DWConv — 3D --- #


@pytest.mark.parametrize(
    "B,C,grids",
    [
        (2, 16, [(6, 8, 10), (3, 4, 5), (2, 2, 2)]),
        (1, 8, [(4, 4, 4), (2, 2, 2), (1, 1, 1)]),
    ],
)
def test_dwconv_3d_shape(B, C, grids):
    offsets = [int(z * y * x) for (z, y, x) in grids]
    N = sum(offsets)
    x = torch.randn(B, N, C)
    m = DWConv(dim=3, embed_dim=C, strategy="axial")
    y = m(x, query_level_shapes=grids, query_offsets=offsets)
    assert y.shape == (B, N, C)


# --- ConvFFN --- #


@pytest.mark.parametrize(
    "dim,offset_grids",
    [
        (3, [(4, 4, 4), (2, 2, 2), (1, 1, 1)]),
        (3, [(3, 5, 7), (2, 2, 2), (1, 1, 1)]),
    ],
)
def test_convffn_tokens_shape(dim, offset_grids):
    B, C = 2, 32
    offsets = [_prod(g) for g in offset_grids]
    N = sum(offsets)
    x = torch.randn(B, N, C)

    grids = offset_grids

    m = ConvFFN(
        in_features=C,
        dim=dim,
        hidden_features=C,
        out_features=C,
        drop=0.0,
        strategy="axial",
    )
    y = m(x, query_level_shapes=grids, query_offsets=offsets)
    assert y.shape == (B, N, C)


# --- Extractor — CrossAttention branch (no deformable) --- #


@pytest.mark.parametrize(
    "B, Nk, C, num_heads, grids",
    [
        (2, 256, 48, 6, [(4, 4, 4), (4, 4, 2), (4, 4, 2)]),  # 64+32+32=128
        (1, 64, 32, 4, [(4, 4, 2), (4, 2, 2), (4, 2, 2)]),  # 32+16+16=64
    ],
)
def test_extractor_cross_attention_shapes(B, Nk, C, num_heads, grids):
    m = Extractor(
        embed_dim=C,
        dim=3,
        use_deform_attention=False,
        num_heads=num_heads,
        with_cffn=True,
        cffn_ratio=0.5,
        strategy="axial",
    )

    offsets = [_prod(g) for g in grids]
    Nq = sum(offsets)

    query = torch.randn(B, Nq, C)
    feat = torch.randn(B, Nk, C)

    out = m(
        query=query,
        features=feat,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        query_level_shapes=grids,
        query_offsets=offsets,
    )
    assert out.shape == (B, Nq, C)


# --- SpatialPriorModule — 3D --- #


@pytest.mark.parametrize(
    "B,in_ch,inplanes,embed_dim,ZYX",
    [
        (2, 2, 16, 32, (32, 48, 64)),
        (1, 3, 8, 16, (64, 64, 80)),
    ],
)
def test_spatial_prior_module_3d(B, in_ch, inplanes, embed_dim, ZYX):
    Z, Y, X = ZYX
    x = torch.randn(B, in_ch, Z, Y, X)
    m = SpatialPriorModule(in_ch=in_ch, inplanes=inplanes, embed_dim=embed_dim, dim=3, strategy="axial")
    out = m(x)
    c1, c2, c3, c4 = out

    # out = floor((n + 2*pad - kernel)/stride + 1). For our blocks: k=3, pad=1.
    def step(n, stride):
        return (n - 1) // stride + 1

    def to_zyx(val):
        return (val, val, val) if isinstance(val, int) else val

    order = ["stem1", "stem2", "stem3", "maxpool", "stage2", "stage3", "stage4"]

    z = Z
    y = Y
    x_ = X
    sizes_after = {}
    for name in order:
        sz, sy, sx = to_zyx(m.strides[name])
        z, y, x_ = step(z, sz), step(y, sy), step(x_, sx)
        sizes_after[name] = (z, y, x_)

    z1, y1, x1 = sizes_after["maxpool"]
    z2, y2, x2 = sizes_after["stage2"]
    z3, y3, x3 = sizes_after["stage3"]
    z4, y4, x4 = sizes_after["stage4"]

    assert c1.shape == (B, z1 * y1 * x1, embed_dim)
    assert c2.shape == (B, z2 * y2 * x2, embed_dim)
    assert c3.shape == (B, z3 * y3 * x3, embed_dim)
    assert c4.shape == (B, z4 * y4 * x4, embed_dim)


# --- EncoderAdapter — metadata helpers --- #


@pytest.mark.parametrize(
    "B, dim, input_format, spatial_shape, patch",
    [
        # Z, Y, X, C
        (2, 3, "ZYXC", (32, 32, 32, 2), (None, 8, 8)),
    ],
)
def test_encoder_adapter_metadata_shapes(B, dim, input_format, spatial_shape, patch):
    Z, Y, X, Cin = spatial_shape

    if dim == 3:
        # patch_shape is (Zp, Yp, Xp, None) for 3D case
        patch_shape = (patch[1], patch[1], patch[1], None)
    else:
        raise NotImplementedError("Only 3D is implemented in this test.")

    adapter = EncoderAdapter(
        dim=dim,
        input_shape=(Z, Y, X, Cin),
        patch_shape=patch_shape,
        backbone_embed_dim=48,
        input_format=input_format,
        use_deform_attention=False,
        deform_num_heads=6,
        strategy="axial",
    )

    dummy_x = torch.randn(B, Z, Y, X, Cin)
    ref, spatial_shapes, lsi, vr = adapter._get_deformable_and_ffn_metadata(dummy_x)

    # spatial_shapes should describe a single 3D patch grid
    assert spatial_shapes.shape == (1, 3)
    assert tuple(spatial_shapes[0].tolist()) == (4, 4, 4)  # (32, 32, 32) / patch 8

    # valid_ratios: [B, 1, 3]
    assert vr.shape == (B, 1, 3)

    # reference_points: [B, Len_q, 1, 3] where Len_q is sum over c2–c4 token counts
    q_shapes = adapter.query_level_shapes[1:]  # skip c1
    len_q = sum(_prod(s) for s in q_shapes)
    assert ref.shape == (B, len_q, 1, 3)
    assert ref.shape[-1] == 3


@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="FlashDeformAttn3D (OPS3D_AVAILABLE) is not installed.",
)
def test_encoder_adapter_forward_deform_attn():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to run deformable attention test.")

    device = torch.device("cuda")

    B, Z, Y, X, Cin = 1, 64, 64, 64, 2
    embed_dim = 48
    Zp, YXp = 8, 8  # patch size

    adapter = EncoderAdapter(
        dim=3,
        input_shape=(Z, Y, X, Cin),
        patch_shape=(Zp, YXp, YXp, None),
        backbone_embed_dim=embed_dim,
        input_format="ZYXC",
        use_deform_attention=True,
        deform_num_heads=6,
        strategy="axial",
        dtype="bfloat16",
    ).to(device)

    # ViT patch grid shape
    Zp_, Yp_, Xp_ = (Z // Zp), (Y // YXp), (X // YXp)
    base = (Zp_, Yp_, Xp_)  # e.g. (8, 8, 8)
    N = _prod(base)  # number of tokens per feature map

    # ViT backbone features: four feature maps with N tokens each
    features = [torch.randn(B, N, embed_dim, device=device) for _ in range(4)]

    x = torch.randn(B, Z, Y, X, Cin, device=device)
    y = adapter(x, features)  # dict {"1": f1, "2": f2, "3": f3, "4": f4}

    f1, f2, f3, f4 = y["1"], y["2"], y["3"], y["4"]

    # Expected pyramid shapes given SPM strides (reference ViT-Adapter
    # semantics: f1 = up(c2) + c1 at c1's NATIVE stride-4 — the old
    # ConvTranspose emitted a stride-2 map that contradicted the declared
    # stride and was deleted):
    # c1 ~ Z/4,  c2 ~ Z/8, c3 ~ Z/16, c4 ~ Z/32
    assert f1.shape == (B, embed_dim, Z // 4, Y // 4, X // 4)
    assert f2.shape == (B, embed_dim, Z // 8, Y // 8, X // 8)
    assert f3.shape == (B, embed_dim, Z // 16, Y // 16, X // 16)
    assert f4.shape == (B, embed_dim, Z // 32, Y // 32, X // 32)


# --- EncoderAdapter geometry: reference points, SPM level shapes, layout guard --- #


def _make_adapter(input_zyx=(32, 32, 32)):
    return EncoderAdapter(
        dim=3,
        input_shape=(*input_zyx, 2),
        patch_shape=(8, 8, 8),
        backbone_embed_dim=16,
        input_format="ZYXC",
        num_backbone_features=1,
        conv_inplane=4,
        use_deform_attention=False,
        deform_num_heads=2,
        with_cffn=False,
        use_extra_extractor=False,
        dtype="float32",
        buffer_device="cpu",
    )


def test_encoder_adapter_reference_points_are_xyz_token_centers():
    """Deformable reference points are normalized token centers in (x, y, z)
    order: ((x+0.5)/X, (y+0.5)/Y, (z+0.5)/Z) for the first query level."""
    adapter = _make_adapter()
    x = torch.zeros(2, 8, 8, 8, 16)  # only shape/device are read
    ref, spatial_shapes, level_start_index, valid_ratios = (
        adapter._get_deformable_and_ffn_metadata(x)
    )
    # first query level is c2 (= query_level_shapes[1])
    Z2, Y2, X2 = adapter.query_level_shapes[1]
    n2 = Z2 * Y2 * X2
    lvl = ref[0, :n2, 0, :].reshape(Z2, Y2, X2, 3)
    for (z, y, xx) in [(0, 0, 0), (Z2 - 1, 0, X2 - 1), (1, 2, 3)]:
        expected = torch.tensor([
            (xx + 0.5) / X2, (y + 0.5) / Y2, (z + 0.5) / Z2,
        ])
        torch.testing.assert_close(lvl[z, y, xx], expected, atol=1e-6, rtol=1e-6)


def test_spm_level_shapes_use_ceil_division_and_match_conv_outputs():
    """query_level_shapes halve with ceil (100 -> 50 -> 25 -> 13 -> 7 -> 4) and the
    SpatialPriorModule emits exactly those token counts per level."""
    adapter = _make_adapter(input_zyx=(100, 100, 100))
    assert adapter.query_level_shapes == [
        (25, 25, 25), (13, 13, 13), (7, 7, 7), (4, 4, 4),
    ]
    with torch.no_grad():
        c1, c2, c3, c4 = adapter.spatial_prior_module(torch.zeros(1, 2, 100, 100, 100))
    assert c1.shape[1] == 25 ** 3
    assert c2.shape[1] == 13 ** 3
    assert c3.shape[1] == 7 ** 3
    assert c4.shape[1] == 4 ** 3


def test_encoder_adapter_forward_rejects_channels_first_input():
    """forward takes channels-last input; a conv-layout tensor (trailing dim is
    not the channel count) fails loudly instead of being permuted into scrambled tokens."""
    a = object.__new__(EncoderAdapter)
    a.dim = 3
    a.in_channels = 2
    x = torch.zeros(1, 2, 4, 4, 4)                   # conv layout, trailing dim 4 != 2
    with pytest.raises(ValueError, match="channels-last"):
        EncoderAdapter.forward(a, x, features=None)
