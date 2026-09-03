import pytest
import torch
import torch.nn.functional as F

from cell_observatory_platform.models.heads.mask2former_decoder import (
    CrossAttentionLayer,
    FFNLayer,
    MultiScaleMaskedTransformerDecoder,
    SelfAttentionLayer,
)


# SelfAttentionLayer
@pytest.mark.parametrize("normalize_before", [False, True])
@pytest.mark.parametrize("Q,B,C,h", [(17, 2, 32, 4), (5, 1, 64, 8)])
def test_self_attention_layer_shapes(Q, B, C, h, normalize_before):
    layer = SelfAttentionLayer(d_model=C, nhead=h, dropout=0.0, normalize_before=normalize_before)
    tgt = torch.randn(Q, B, C)
    pos = torch.randn(Q, B, C)  # query_pos
    out = layer(tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=pos)
    assert out.shape == (Q, B, C)


# CrossAttentionLayer
@pytest.mark.parametrize("normalize_before", [False, True])
@pytest.mark.parametrize("Q,L,B,C,h", [(11, 23, 2, 32, 4), (7, 5, 1, 64, 8)])
def test_cross_attention_layer_shapes(Q, L, B, C, h, normalize_before):
    layer = CrossAttentionLayer(d_model=C, nhead=h, dropout=0.0, normalize_before=normalize_before)
    tgt = torch.randn(Q, B, C)  # queries
    mem = torch.randn(L, B, C)  # keys/values
    qpos = torch.randn(Q, B, C)
    kpos = torch.randn(L, B, C)
    out = layer(tgt, mem, memory_mask=None, memory_key_padding_mask=None, pos=kpos, query_pos=qpos)
    assert out.shape == (Q, B, C)


# FFNLayer
@pytest.mark.parametrize("normalize_before", [False, True])
@pytest.mark.parametrize("Q,B,C,ff", [(13, 2, 48, 128), (9, 1, 64, 256)])
def test_ffn_layer_shapes(Q, B, C, ff, normalize_before):
    layer = FFNLayer(d_model=C, dim_feedforward=ff, dropout=0.0, normalize_before=normalize_before)
    x = torch.randn(Q, B, C)
    y = layer(x)
    assert y.shape == (Q, B, C)


# MultiScaleMaskedTransformerDecoder.forward_prediction_heads shapes
@pytest.mark.parametrize(
    "B,Q,C,heads,num_classes,mask_dim,DM,HM,WM",
    [
        (2, 16, 32, 4, 3, 16, 6, 5, 4),
        (1, 8, 64, 8, 7, 32, 3, 4, 5),
    ],
)
def test_forward_prediction_heads_shapes(B, Q, C, heads, num_classes, mask_dim, DM, HM, WM):
    dec = MultiScaleMaskedTransformerDecoder(
        input_dim=3,
        in_channels=C,  # matches hidden_dim for identity 1x1 if enforce_input_project=False
        mask_classification=True,
        num_classes=num_classes,
        hidden_dim=C,
        num_queries=Q,
        decoder_nheads=heads,
        dim_feedforward=2 * C,
        decoder_num_layers=1,
        decoder_pre_norm=False,
        mask_dim=mask_dim,  # must match mask_features channel
        enforce_input_project=False,
        num_feature_levels=3,
    )

    # Fake decoder output (Q, B, C) and mask_features (B, mask_dim, D, H, W)
    output = torch.randn(Q, B, C)
    mask_features = torch.randn(B, mask_dim, DM, HM, WM)

    # Target attention spatial size (D, H, W) — pick something distinct from mask_features for resize
    target_size = (DM + 2, HM + 1, WM + 3)

    cls_logits, masks, attn_mask = dec.forward_prediction_heads(output, mask_features, target_size)

    # Shapes
    assert cls_logits.shape == (B, Q, num_classes + 1)
    assert masks.shape == (B, Q, DM, HM, WM)

    # Attention mask produced as boolean; last dim must be D*H*W of target_size
    L_target = target_size[0] * target_size[1] * target_size[2]
    assert attn_mask.dtype == torch.bool
    assert attn_mask.shape == (B * heads, Q, L_target)

    # attn_mask is exactly "resized mask probability < 0.5", replicated per head
    resized = F.interpolate(masks, size=target_size, mode="trilinear", align_corners=False)
    expected = (resized.sigmoid().flatten(2) < 0.5).repeat_interleave(heads, dim=0)
    assert torch.equal(attn_mask, expected)


# MultiScaleMaskedTransformerDecoder.forward — end-to-end
@pytest.mark.parametrize(
    "B,in_channels,hidden_dim,num_classes,num_queries,heads,mask_dim,feat_sizes,mask_feat_size,num_layers",
    [
        # 3 feature levels; mask grid distinct size
        (2, 32, 32, 4, 12, 4, 16, [(6, 6, 6), (3, 3, 3), (2, 2, 2)], (5, 4, 3), 2),
        (1, 64, 64, 2, 8, 8, 32, [(4, 5, 6), (2, 3, 3), (1, 2, 2)], (3, 3, 3), 3),
    ],
)
def test_decoder_forward_end_to_end_shapes(
    B, in_channels, hidden_dim, num_classes, num_queries, heads, mask_dim, feat_sizes, mask_feat_size, num_layers
):
    # build decoder
    dec = MultiScaleMaskedTransformerDecoder(
        input_dim=3,
        in_channels=in_channels,
        mask_classification=True,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        decoder_nheads=heads,
        dim_feedforward=2 * hidden_dim,
        decoder_num_layers=num_layers,
        decoder_pre_norm=False,
        mask_dim=mask_dim,
        enforce_input_project=False,
        num_feature_levels=len(feat_sizes),
    )

    # x: list of L feature maps [B, in_channels, D, H, W]
    x = [torch.randn(B, in_channels, D, H, W) for (D, H, W) in feat_sizes]

    # mask_features: [B, mask_dim, Dm, Hm, Wm]
    Dm, Hm, Wm = mask_feat_size
    mask_features = torch.randn(B, mask_dim, Dm, Hm, Wm)

    out = dec(x, mask_features)

    # final predictions
    assert "pred_logits" in out and "pred_masks" in out and "auxiliary_outputs" in out
    assert out["pred_logits"].shape == (B, num_queries, num_classes + 1)
    assert out["pred_masks"].shape == (B, num_queries, Dm, Hm, Wm)

    # aux outputs: num_layers entries, each with same shapes
    assert isinstance(out["auxiliary_outputs"], list)
    assert len(out["auxiliary_outputs"]) == num_layers
    for aux in out["auxiliary_outputs"]:
        if dec.mask_classification:
            assert aux["pred_logits"].shape == (B, num_queries, num_classes + 1)
        assert aux["pred_masks"].shape == (B, num_queries, Dm, Hm, Wm)


def test_forward_with_all_true_attention_rows_stays_finite():
    """A query whose predicted mask is empty everywhere produces an all-True (fully
    masked) attention row; the decoder must un-mask such rows instead of letting
    softmax over all -inf produce NaN."""
    torch.manual_seed(0)
    B, C, Q, heads, mask_dim = 1, 32, 4, 2, 16
    dec = MultiScaleMaskedTransformerDecoder(
        input_dim=3,
        in_channels=C,
        mask_classification=True,
        num_classes=3,
        hidden_dim=C,
        num_queries=Q,
        decoder_nheads=heads,
        dim_feedforward=2 * C,
        decoder_num_layers=1,
        decoder_pre_norm=False,
        mask_dim=mask_dim,
        enforce_input_project=False,
        num_feature_levels=1,
    ).eval()
    # force mask_embed to a constant: every mask logit becomes -mask_dim -> sigmoid < 0.5 everywhere
    with torch.no_grad():
        last = dec.mask_embed.layers[-1]
        last.weight.zero_()
        last.bias.fill_(-1.0)
    x = [torch.randn(B, C, 3, 3, 3)]
    mask_features = torch.ones(B, mask_dim, 2, 2, 2)

    # precondition: the prediction head really emits all-True rows for this input
    _, _, attn_mask = dec.forward_prediction_heads(
        dec.query_feat.weight.unsqueeze(1), mask_features, attn_mask_target_size=(3, 3, 3)
    )
    assert attn_mask.shape == (B * heads, Q, 27) and attn_mask.all()

    out = dec(x, mask_features)
    assert torch.isfinite(out["pred_logits"]).all()
    assert torch.isfinite(out["pred_masks"]).all()


@pytest.mark.parametrize("hidden_dim,expected", [(256, 86), (264, 88), (96, 32)])
def test_pe_layer_uses_ceil_of_hidden_dim_over_three(hidden_dim, expected):
    """The sin-cos PE gets ceil(hidden_dim / 3) feats per axis (256 -> 86, not 85),
    so the concatenated 3-axis embedding covers hidden_dim."""
    dec = MultiScaleMaskedTransformerDecoder(
        input_dim=3,
        in_channels=hidden_dim,
        mask_classification=True,
        num_classes=3,
        hidden_dim=hidden_dim,
        num_queries=4,
        decoder_nheads=8,
        dim_feedforward=hidden_dim,
        decoder_num_layers=1,
        decoder_pre_norm=False,
        mask_dim=16,
        enforce_input_project=False,
        num_feature_levels=1,
    )
    assert dec.pe_layer.num_pos_feats == expected


def _polarity_probe(build_attn_mask):
    """MHA-semantics mask (True = CANNOT attend) -> SDPA -> weights ~0 there."""
    L = 2
    mha_mask = torch.tensor([[False, True], [False, False]])  # q0 must not see k1
    attn_mask = build_attn_mask(mha_mask, torch.device("cpu"))
    assert attn_mask is not None

    torch.manual_seed(0)
    q = torch.randn(1, 1, L, 8)
    k = torch.randn(1, 1, L, 8)
    # v one-hot per key so the output reveals the attention weights
    v = torch.eye(L).reshape(1, 1, L, L)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    weights = out[0, 0]  # (Lq, Lk) attention weights
    assert weights[0, 1].abs().item() < 1e-6, "blocked key still attended"
    assert abs(weights[0, 0].item() - 1.0) < 1e-6
    assert abs(weights[1].sum().item() - 1.0) < 1e-5


def test_self_attention_layer_attn_mask_true_blocks_attention():
    """The layer's SDPA mask keeps nn.MultiheadAttention semantics: True = cannot attend."""
    layer = SelfAttentionLayer(d_model=32, nhead=4)
    _polarity_probe(layer._build_attn_mask)
