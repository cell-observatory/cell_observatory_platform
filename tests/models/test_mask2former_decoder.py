import pytest
import torch

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


# sanity on attention-mask post-processing line
def test_attention_mask_all_true_row_is_unmasked():
    """
    Ensure the safety line:
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
    does not crash and results remain boolean with correct shape.
    """
    B, Q, C = 1, 4, 32
    heads = 2
    num_classes = 3
    L = 3  # keys length for target level

    dec = MultiScaleMaskedTransformerDecoder(
        input_dim=3,
        in_channels=C,
        mask_classification=True,
        num_classes=num_classes,
        hidden_dim=C,
        num_queries=Q,
        decoder_nheads=heads,
        dim_feedforward=2 * C,
        decoder_num_layers=1,
        decoder_pre_norm=False,
        mask_dim=16,
        enforce_input_project=False,
        num_feature_levels=1,
    )

    # build a degenerate attn_mask with some all-True rows
    # shape the decoder expects to pass into MHA: [B*heads, Q, L]
    attn_mask = torch.ones(B * heads, Q, L, dtype=torch.bool)

    attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

    assert attn_mask.dtype == torch.bool
    assert attn_mask.shape == (B * heads, Q, L)
    assert not torch.all(attn_mask)  # at least something is False
