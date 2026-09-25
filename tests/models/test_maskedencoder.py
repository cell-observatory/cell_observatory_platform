"""`MaskedEncoder` patch-embed switch: joint (default, unchanged) vs channel_adaptive
(per-channel tokens + frozen-vocab token embedding), attn_pool and concat fusion."""
from __future__ import annotations

import pytest
import torch

from cell_observatory_platform.models.backbones.maskedencoder import BUILD, MaskedEncoder
from cell_observatory_platform.models.layers.patch_embeddings import (
    ChannelAdaptivePatchEmbedding,
    PatchEmbedding,
)

VOCAB = {
    "localization": {"<unk>": 0, "cytosol": 1, "membrane": 2},
    "fluorophore": {"<unk>": 0, "electra2": 1, "mstaygold": 2, "mtfp1": 3},
}
SHAPE = (1, 16, 16, 16, 3)          # T, Z, Y, X, C
PATCH = (1, 8, 8, 8)
N = 2 * 2 * 2


def _encoder(patch_embed_type="joint", fusion="attn_pool", channel_embed="factorized",
             vocab=VOCAB, sincos=True, rope=False, extra=4):
    args = None
    if patch_embed_type == "channel_adaptive":
        args = {
            "channel_fusion": fusion,
            "attn_pool_num_heads": 2,
            "channel_embed": channel_embed,
            "channel_vocab": vocab,
            "vocab_extra_slots": extra,
        }
    return MaskedEncoder(
        model_template="mae",
        input_fmt="TZYXC",
        input_shape=SHAPE,
        patch_shape=PATCH,
        embed_dim=32,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        abs_sincos_enc=sincos,
        rope_pos_enc=rope,
        patch_embed_type=patch_embed_type,
        patch_embed_args=args,
    ).eval()


def _ids(B=None):
    ids = torch.tensor([[2, 2], [1, 1], [1, 3]])       # membrane/mstaygold, cytosol/electra2, cytosol/mtfp1
    return ids if B is None else ids.unsqueeze(0).expand(B, -1, -1)


def test_joint_is_the_default_and_builds_the_old_patch_embed():
    enc = _encoder()
    assert isinstance(enc.patch_embedding, PatchEmbedding)
    assert enc.tokens_per_patch == 1
    x = torch.randn(2, *SHAPE)
    out = enc.forward_features(x)
    assert out.shape == (2, N, 32)
    # channel_ids is accepted and ignored on the joint path
    torch.testing.assert_close(enc.forward_features(x, channel_ids=_ids()), out)


def test_channel_adaptive_attn_pool_keeps_the_token_count():
    enc = _encoder("channel_adaptive")
    pe = enc.patch_embedding
    assert isinstance(pe, ChannelAdaptivePatchEmbedding)
    assert pe.localization_embed.num_embeddings == 3 + 4       # vocab + extra slots
    assert pe.fluorophore_embed.num_embeddings == 4 + 4
    x = torch.randn(2, *SHAPE)
    out = enc.forward_features(x, channel_ids=_ids())
    assert out.shape == (2, N, 32)
    other = enc.forward_features(x, channel_ids=torch.tensor([[1, 1], [1, 1], [1, 1]]))
    assert not torch.allclose(out, other)
    per_sample = enc.forward_features(x, channel_ids=_ids(B=2))
    torch.testing.assert_close(per_sample, out)


@pytest.mark.parametrize("sincos,rope", [(True, False), (False, True), (True, True)])
def test_concat_fusion_yields_c_tokens_per_patch_with_repeated_positions(sincos, rope):
    enc = _encoder("channel_adaptive", fusion="concat", sincos=sincos, rope=rope)
    assert enc.tokens_per_patch == 3
    if rope:
        assert enc.freqs_cis.shape[0] == N * 3
    x = torch.randn(2, *SHAPE)
    out = enc.forward_features(x, channel_ids=_ids())
    assert out.shape == (2, N * 3, 32)


def test_channel_adaptive_requires_ids_in_token_modes():
    enc = _encoder("channel_adaptive")
    with pytest.raises(ValueError, match="requires channel_ids"):
        enc.forward_features(torch.randn(1, *SHAPE))


def test_channel_embed_none_needs_no_vocab_and_no_ids():
    enc = _encoder("channel_adaptive", channel_embed="none", vocab=None)
    assert enc.patch_embedding.channel_embed is None
    assert enc.forward_features(torch.randn(1, *SHAPE)).shape == (1, N, 32)


def test_missing_vocab_is_a_construction_error():
    with pytest.raises(ValueError, match="needs patch_embed_args.channel_vocab"):
        _encoder("channel_adaptive", vocab=None)
    with pytest.raises(ValueError, match="unknown patch_embed_type"):
        _encoder("hybrid")


def test_frozen_table_size_wins_over_extra_slots():
    """A vocab carrying table_size (the checkpointed weight shape) sizes the
    tables exactly; vocab_extra_slots only applies to an unfrozen vocab."""
    vocab = dict(VOCAB, table_size={"localization": 40, "fluorophore": 50})
    enc = _encoder("channel_adaptive", vocab=vocab, extra=4)
    assert enc.patch_embedding.localization_embed.num_embeddings == 40
    assert enc.patch_embedding.fluorophore_embed.num_embeddings == 50


def test_build_from_config_forwards_the_switch():
    cfg = {
        "name": "masked_vit", "model_template": "mae", "input_fmt": "TZYXC",
        "input_shape": list(SHAPE), "patch_shape": list(PATCH), "embed_dim": 32,
        "depth": 1, "num_heads": 2, "mlp_ratio": 2.0, "abs_sincos_enc": True, "rope_pos_enc": False,
        "patch_embed_type": "channel_adaptive",
        "patch_embed_args": {"channel_fusion": "attn_pool", "attn_pool_num_heads": 2,
                             "channel_embed": "factorized", "channel_vocab": VOCAB, "vocab_extra_slots": 2},
    }
    enc = BUILD(cfg)
    assert isinstance(enc.patch_embedding, ChannelAdaptivePatchEmbedding)
    assert enc.patch_embedding.localization_embed.num_embeddings == 5
