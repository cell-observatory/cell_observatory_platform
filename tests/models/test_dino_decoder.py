import pytest
import torch
import torch.nn.functional as F
from torch import nn

from cell_observatory_platform.models.heads.dino_decoder import DeformableTransformerDecoderLayer, TransformerDecoder


def _pyramid():
    level_shapes = torch.tensor([[8, 8, 8], [4, 4, 4], [2, 2, 2]], dtype=torch.long)
    tokens = level_shapes.prod(dim=1)
    start = torch.cumsum(torch.cat([tokens.new_zeros(1), tokens[:-1]]), dim=0)
    return level_shapes, start, int(tokens.sum())


def test_decoder_layer_fallback_cross_attention_shapes():
    """The default (non-deformable) layer runs on CPU through the pure-PyTorch
    CrossAttention, keeps (Q, B, C), changes its output when self-attention is
    removed, and refuses a non-trivial memory padding mask it cannot honour."""
    torch.manual_seed(0)
    Q, B, C, L = 5, 2, 64, 3  # embed_dim // num_heads must be a multiple of 8
    layer = DeformableTransformerDecoderLayer(
        embed_dim=C, feedforward_dim=64, dropout=0.0, num_levels=L, num_heads=8,
    ).eval()
    level_shapes, start, n_tok = _pyramid()
    kw = dict(
        target_query_pos_embeddings=torch.randn(Q, B, C),
        target_reference_points=torch.rand(Q, B, L, 3),
        memory=torch.randn(n_tok, B, C),
        memory_key_padding_mask=torch.zeros(B, n_tok, dtype=torch.bool),
        memory_level_start_index=start,
        memory_shapes=level_shapes,
    )
    target = torch.randn(Q, B, C)

    out = layer(target=target, **kw)
    assert out.shape == (Q, B, C) and torch.isfinite(out).all()

    layer.remove_self_attn_modules()
    out_no_self = layer(target=target, **kw)
    assert out_no_self.shape == (Q, B, C)
    assert not torch.allclose(out, out_no_self)  # the self-attention branch really contributed

    # the fallback path does not consume a padding mask; it must refuse a padded one
    padded = torch.ones(B, n_tok, dtype=torch.bool)
    with pytest.raises(NotImplementedError, match="memory_key_padding_mask"):
        layer(target=target, **{**kw, "memory_key_padding_mask": padded})


def test_transformer_decoder_returns_per_layer_outputs_and_refined_reference_points():
    """Without bbox_embed the decoder returns one normed output per layer and only the
    sigmoided initial reference points; with one delta head per layer each layer
    appends a refined, still-sigmoided reference set."""
    torch.manual_seed(0)
    Q, B, C, L, num_layers, query_dim = 5, 2, 64, 3, 2, 6
    layer = DeformableTransformerDecoderLayer(
        embed_dim=C, feedforward_dim=64, dropout=0.0, num_levels=L, num_heads=8,
    )
    decoder = TransformerDecoder(
        decoder_layer=layer, num_layers=num_layers, norm=nn.LayerNorm(C), embed_dim=C,
        query_dim=query_dim, num_feature_levels=L, deformable_decoder=True,
    ).eval()
    level_shapes, start, n_tok = _pyramid()
    kw = dict(
        memory=torch.randn(n_tok, B, C), level_start_index=start, shapes=level_shapes,
        valid_ratios=torch.ones(B, L, 3),
    )
    target = torch.randn(Q, B, C)
    reference_points = torch.randn(Q, B, query_dim)  # pre-sigmoid, as the forward expects

    decoder.bbox_embed = None
    outputs, refs = decoder(target=target, reference_points=reference_points, **kw)
    assert len(outputs) == num_layers
    assert all(o.shape == (B, Q, C) and torch.isfinite(o).all() for o in outputs)
    assert len(refs) == 1  # no refinement: only the sigmoided initial points, batch-first
    torch.testing.assert_close(refs[0], reference_points.sigmoid().transpose(0, 1))

    # with one delta head per layer, each layer appends a refined reference set
    decoder.bbox_embed = nn.ModuleList([nn.Linear(C, query_dim) for _ in range(num_layers)])
    outputs, refs = decoder(target=target, reference_points=reference_points, **kw)
    assert len(outputs) == num_layers
    assert len(refs) == num_layers + 1
    assert all(r.shape == (B, Q, query_dim) for r in refs)
    assert all(((r >= 0) & (r <= 1)).all() for r in refs)  # refined points stay sigmoided
    assert not torch.allclose(refs[1], refs[0])


def test_transformer_decoder_rejects_query_dim_other_than_6():
    """Only the 6-D (x, y, z, w, h, d) query path is implemented; query_dim=3 is refused."""
    layer = DeformableTransformerDecoderLayer(
        32, 64, 0.0, nn.ReLU(), 1, 4, 4, use_deform_attention=False,
    )
    with pytest.raises(NotImplementedError, match="query_dim"):
        TransformerDecoder(
            layer, 1, nn.LayerNorm(32), return_intermediates=True,
            embed_dim=32, query_dim=3, num_feature_levels=1,
            share_decoder_layers=False,
        )


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


def test_decoder_layer_attn_mask_true_blocks_attention():
    """The layer's SDPA mask keeps nn.MultiheadAttention semantics: True = cannot attend."""
    layer = DeformableTransformerDecoderLayer(
        embed_dim=64, num_heads=8, use_deform_attention=False
    )
    _polarity_probe(layer._build_attn_mask)
