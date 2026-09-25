import pytest

import torch
import torch.nn as nn

from cell_observatory_platform.models.heads.pixel_decoders import (
    Mask2FormerPixelDecoder,
    MaskDINOEncoder,
    MSDeformAttnTransformerEncoder,
    MSDeformAttnTransformerEncoderLayer,
)

try:
    from ops3d import _C
    OPS3D_AVAILABLE = True
except ImportError:
    OPS3D_AVAILABLE = False


# The CrossAttention fallback is pure torch and runs on CPU always; only the
# OPS3D deformable path needs a GPU (and the compiled extension).
DEFORM_PARAMS = [
    pytest.param(False, id="cross_attn"),
    pytest.param(
        True, id="deform_ops3d",
        marks=[pytest.mark.cuda, pytest.mark.skipif(not OPS3D_AVAILABLE, reason="ops3d not installed")],
    ),
]


def _device(use_deform: bool) -> str:
    return "cuda" if use_deform else "cpu"


def _tokens_total(shapes):
    return int(sum(int(D) * int(H) * int(W) for (D, H, W) in shapes))


class _ZeroPos(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        B, _, D, H, W = x.shape
        return torch.zeros(B, self.channels, D, H, W, dtype=x.dtype, device=x.device)


@pytest.mark.parametrize("use_deform", DEFORM_PARAMS)
@pytest.mark.parametrize("B,C", [(2, 64), (1, 96)])
def test_encoder_layer_shapes(B, C, use_deform):
    dev = _device(use_deform)

    # choose heads so per_head_dim % 8 == 0
    n_heads = C // 8
    assert C % n_heads == 0 and (C // n_heads) % 8 == 0

    layer = MSDeformAttnTransformerEncoderLayer(
        embed_dim=C, feedforward_dim=4 * C, dropout=0.0,
        activation="RELU", n_levels=3, n_heads=n_heads, n_points=4,
        use_deform_attention=use_deform,
    ).to(dev)

    spatial_shapes = torch.as_tensor([[3, 4, 5], [2, 2, 3], [1, 1, 2]], dtype=torch.long, device=dev)
    L = spatial_shapes.size(0)
    tokens_per_level = spatial_shapes[:, 0] * spatial_shapes[:, 1] * spatial_shapes[:, 2]
    S = int(tokens_per_level.sum().item())  # 74
    level_start_index = torch.as_tensor(
        [0, int(tokens_per_level[0]), int(tokens_per_level[:2].sum())], dtype=torch.long, device=dev
    )

    x = torch.randn(B, S, C, device=dev)
    pos = torch.randn(B, S, C, device=dev)
    padding_mask = torch.zeros(B, S, dtype=torch.bool, device=dev)
    reference_points = torch.rand(B, S, L, 3, device=dev)

    y = layer(x, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
    assert y.shape == (B, S, C)
    assert torch.isfinite(y).all()


# MSDeformAttnTransformerEncoder
@pytest.mark.parametrize("use_deform", DEFORM_PARAMS)
@pytest.mark.parametrize(
    "B,C,n_heads,shapes",
    [
        # C=64, heads=8 -> per-head=8
        (2, 64, 8, [(6, 5, 4), (3, 3, 3), (2, 2, 2)]),
        # C=96, heads=12 -> per-head=8
        (1, 96, 12, [(4, 6, 5), (2, 3, 3)]),
    ],
)
def test_encoder_forward_shapes(B, C, n_heads, shapes, use_deform):
    dev = _device(use_deform)

    L = len(shapes)
    enc = MSDeformAttnTransformerEncoder(
        embed_dim=C,
        feedforward_dim=4 * C,
        num_heads=n_heads,
        num_encoder_layers=3,
        dropout=0.0,
        activation="relu",
        num_feature_levels=L,
        enc_num_points=4,
        use_deform_attention=use_deform,
    ).to(dev)

    features = [torch.randn(B, C, D, H, W, device=dev) for (D, H, W) in shapes]
    pos_embs = [torch.zeros(B, C, D, H, W, device=dev) for (D, H, W) in shapes]
    masks = [torch.zeros(B, D, H, W, dtype=torch.bool, device=dev) for (D, H, W) in shapes]

    memory, feature_shapes, level_start_index = enc(features, masks, pos_embs)

    tokens_total = _tokens_total(shapes)
    assert memory.shape == (B, tokens_total, C)

    assert feature_shapes.shape == (L, 3)
    for row, shp in zip(feature_shapes.tolist(), shapes):
        assert tuple(row) == tuple(shp)

    assert level_start_index.shape == (L,)
    expected = [0]
    for i in range(L - 1):
        expected.append(expected[-1] + shapes[i][0] * shapes[i][1] * shapes[i][2])
    assert level_start_index.tolist() == expected


# MaskDINOEncoder.forward_features
def _make_input_shape_dict(C0=48, C1=64, C2=96):
    return {
        "res3": {"channels": C0, "stride": 8},
        "res4": {"channels": C1, "stride": 16},
        "res5": {"channels": C2, "stride": 32},
    }


@pytest.mark.parametrize("use_deform", DEFORM_PARAMS)
@pytest.mark.parametrize("add_extra_levels", [False, True])
def test_maskdino_encoder_forward_features_shapes(add_extra_levels, use_deform):
    dev = _device(use_deform)

    B = 2
    conv_dim = 96
    mask_dim = 16

    input_shape = _make_input_shape_dict(48, 64, 96)
    transformer_in_features = ["res3", "res4", "res5"]

    res3 = (16, 16, 16)
    res4 = (8, 8, 8)
    res5 = (4, 4, 4)

    total_num_feature_levels = len(transformer_in_features) + (1 if add_extra_levels else 0)

    enc = MaskDINOEncoder(
        input_shape_metadata=input_shape,
        transformer_in_features=transformer_in_features,
        target_min_stride=8,
        total_num_feature_levels=total_num_feature_levels,
        transformer_encoder_dropout=0.0,
        transformer_encoder_num_heads=12,
        transformer_encoder_dim_feedforward=4 * conv_dim,
        num_transformer_encoder_layers=2,
        conv_dim=conv_dim,
        mask_dim=mask_dim,
        norm=None,
        use_deform_attention=use_deform,
    ).to(dev)

    enc.pos_embedding = _ZeroPos(conv_dim).to(dev)

    features = {
        "res3": torch.randn(B, input_shape["res3"]["channels"], *res3, device=dev),
        "res4": torch.randn(B, input_shape["res4"]["channels"], *res4, device=dev),
        "res5": torch.randn(B, input_shape["res5"]["channels"], *res5, device=dev),
    }
    masks = None

    mask_feats, finest_map, all_maps = enc.forward_features(features, masks)

    assert isinstance(all_maps, list)
    assert len(all_maps) == total_num_feature_levels

    for t in all_maps:
        assert t.dim() == 5
        assert t.shape[0] == B and t.shape[1] == conv_dim
        assert t.device.type == dev

    assert finest_map.shape == (B, conv_dim, *res3)

    if add_extra_levels:
        expected_coarse = (res5[0] // 2, res5[1] // 2, res5[2] // 2)
        assert all_maps[-1].shape == (B, conv_dim, *expected_coarse)
    else:
        assert all_maps[-1].shape == (B, conv_dim, *res5)

    Dc, Hc, Wc = all_maps[0].shape[-3:]
    assert mask_feats.shape == (B, mask_dim, Dc, Hc, Wc)
    assert mask_feats.device.type == dev


# get_padding_mask() fast path — CPU-only (does not invoke CUDA op)
def test_get_padding_mask_fast_path_all_false():
    B, C = 1, 32
    enc = MSDeformAttnTransformerEncoder(
        embed_dim=C,
        feedforward_dim=128,
        num_heads=8,
        num_encoder_layers=1,
        dropout=0.0,
        activation="relu",
        num_feature_levels=2,
        enc_num_points=4,
        use_deform_attention=False,
    )

    f1 = torch.randn(B, C, 32, 32, 20)
    f2 = torch.randn(B, C, 64, 64, 10)
    masks = None
    out = enc.get_padding_mask(masks, [f1, f2])

    assert isinstance(out, list) and len(out) == 2
    assert out[0].shape == (B, 32, 32, 20)
    assert out[1].shape == (B, 64, 64, 10)
    assert (out[0] == 0).all() and (out[1] == 0).all()


# Token splitting consistency
@pytest.mark.parametrize("use_deform", DEFORM_PARAMS)
@pytest.mark.parametrize(
    "B,C,n_heads,shapes",
    [
        (2, 64, 8, [(5, 4, 3), (4, 3, 2), (2, 2, 2)]),
        (1, 96, 12, [(3, 3, 3), (2, 2, 2), (1, 2, 3)]),
    ],
)
def test_token_splitting_consistency(B, C, n_heads, shapes, use_deform):
    dev = _device(use_deform)

    L = len(shapes)
    enc = MSDeformAttnTransformerEncoder(
        embed_dim=C,
        feedforward_dim=3 * C,
        num_heads=n_heads,
        num_encoder_layers=2,
        dropout=0.0,
        activation="relu",
        num_feature_levels=L,
        enc_num_points=4,
        use_deform_attention=use_deform,
    ).to(dev)

    feats = [torch.randn(B, C, *s, device=dev) for s in shapes]
    pos = [torch.zeros(B, C, *s, device=dev) for s in shapes]
    masks = [torch.zeros(B, s[0], s[1], s[2], dtype=torch.bool, device=dev) for s in shapes]

    memory, feature_shapes, level_start_index = enc(feats, masks, pos)

    tokens_per_level = []
    for i in range(L - 1):
        tokens_per_level.append(int(level_start_index[i + 1].item() - level_start_index[i].item()))
    tokens_per_level.append(int(memory.shape[1] - level_start_index[L - 1].item()))
    assert sum(tokens_per_level) == memory.shape[1]

    chunks = torch.split(memory, tokens_per_level, dim=1)
    assert len(chunks) == L
    for chunk, (D, H, W) in zip(chunks, shapes):
        assert chunk.shape[0] == B
        assert chunk.shape[2] == C
        assert chunk.shape[1] == D * H * W


@pytest.mark.parametrize("use_deform", DEFORM_PARAMS)
@pytest.mark.parametrize("add_extra_levels", [False, True])
def test_mask2former_pixel_decoder_forward_features_shapes(add_extra_levels, use_deform):
    dev = _device(use_deform)

    B = 2
    conv_dim = 64
    mask_dim = 16

    input_shape_metadata = _make_input_shape_dict(48, 64, 96)
    transformer_in_features = ["res3", "res4", "res5"]

    res3 = (16, 16, 16)
    res4 = (8, 8, 8)
    res5 = (4, 4, 4)

    total_num_feature_levels = len(transformer_in_features) + (1 if add_extra_levels else 0)

    dec = Mask2FormerPixelDecoder(
        input_shape_metadata=input_shape_metadata,
        transformer_in_features=transformer_in_features,
        total_num_feature_levels=total_num_feature_levels,
        target_min_stride=8,
        transformer_encoder_dropout=0.0,
        transformer_encoder_num_heads=8,
        transformer_encoder_dim_feedforward=4 * conv_dim,
        transformer_encoder_layers=2,
        conv_dim=conv_dim,
        mask_dim=mask_dim,
        norm=None,
        use_deform_attention=use_deform,
    ).to(dev)

    features = {
        "res3": torch.randn(B, input_shape_metadata["res3"]["channels"], *res3, device=dev),
        "res4": torch.randn(B, input_shape_metadata["res4"]["channels"], *res4, device=dev),
        "res5": torch.randn(B, input_shape_metadata["res5"]["channels"], *res5, device=dev),
    }

    mask_feats, coarsest_map, all_maps = dec.forward_features(features)

    assert isinstance(all_maps, list)

    for t in all_maps:
        assert t.dim() == 5
        assert t.shape[0] == B and t.shape[1] == conv_dim
        assert t.device.type == dev

    # Second return is lowest-res feature map (output_map[0])
    assert coarsest_map.shape == (B, conv_dim, *res5)
    assert all_maps[-1].shape == (B, conv_dim, *res3)

    Dc, Hc, Wc = all_maps[-1].shape[-3:]
    assert mask_feats.shape == (B, mask_dim, Dc, Hc, Wc)
    assert mask_feats.device.type == dev


def test_encoder_layer_fallback_attention_rejects_nontrivial_padding_mask():
    """The non-deformable (CrossAttention) fallback cannot honour a padding mask
    with padded keys; it refuses loudly instead of attending to padding."""
    layer = MSDeformAttnTransformerEncoderLayer(
        embed_dim=32, feedforward_dim=64, dropout=0.0,
        n_heads=4, n_levels=1, n_points=4, use_deform_attention=False,
    )
    x = torch.randn(1, 27, 32)
    pm = torch.zeros(1, 27, dtype=torch.bool)
    pm[0, -5:] = True
    with pytest.raises(NotImplementedError, match="padding"):
        layer(
            x, pos=torch.zeros_like(x),
            reference_points=torch.rand(1, 27, 1, 3),
            spatial_shapes=[(3, 3, 3)],
            level_start_index=torch.tensor([0]),
            padding_mask=pm,
        )
