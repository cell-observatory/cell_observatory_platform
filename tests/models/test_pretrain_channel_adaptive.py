"""MAE / JEPA with the channel-adaptive patch embedding (attn_pool + factorized
token embedding): `patch_embed_type` / `patch_embed_args` reach both encoders,
`metainfo["channel_ids"]` feeds the embedding, and a forward+backward runs on CPU."""
from __future__ import annotations

import pytest
import torch

from cell_observatory_platform.models.layers.patch_embeddings import ChannelAdaptivePatchEmbedding, PatchEmbedding
from cell_observatory_platform.models.meta_arch.jepa import JEPA
from cell_observatory_platform.models.meta_arch.maskedautoencoder import MaskedAutoEncoder
from cell_observatory_platform.training.helpers import get_masked_input_data

VOCAB = {
    "localization": {"<unk>": 0, "cytosol": 1, "membrane": 2},
    "fluorophore": {"<unk>": 0, "electra2": 1, "mstaygold": 2, "mtfp1": 3},
}
SHAPE = (16, 32, 32, 3)          # Z, Y, X, C
PATCH = (8, 8, 8, None)
D = 16
ARGS = {
    "channel_fusion": "attn_pool",
    "attn_pool_num_heads": 2,
    "channel_embed": "factorized",
    "channel_vocab": VOCAB,
    "vocab_extra_slots": 4,
}
COMMON = dict(
    input_fmt="ZYXC", input_shape=SHAPE, patch_shape=PATCH, embed_dim=D, depth=1, num_heads=2,
    drop_path_rate=0.0, abs_sincos_enc=True, rope_pos_enc=False, dtype=torch.float32,
    buffer_device="cpu",
)


def _ids(B):
    ids = torch.tensor([[2, 2], [1, 1], [1, 3]])
    return ids.unsqueeze(0).expand(B, -1, -1).clone()


def _mae(**kw):
    torch.manual_seed(0)
    return MaskedAutoEncoder(model_template="mae", decoder_embed_dim=D, decoder_depth=1, decoder_num_heads=2,
                             **COMMON, **kw)


def _jepa(**kw):
    torch.manual_seed(0)
    return JEPA(predictor_embed_dim=D, predictor_depth=1, predictor_num_heads=2, **COMMON, **kw)


def _sample(model, B=1, channel_ids=True):  # get_masked_input_data builds B=1 masks
    (sample,) = get_masked_input_data(model, (B, *SHAPE), device="cpu", mask_ratio=0.75)
    if channel_ids:
        sample["metainfo"]["channel_ids"] = _ids(B)
    return sample


def test_default_is_joint_for_both():
    assert isinstance(_mae().masked_encoder.patch_embedding, PatchEmbedding)
    j = _jepa()
    assert isinstance(j.input_encoder.patch_embedding, PatchEmbedding)
    assert isinstance(j.target_encoder.patch_embedding, PatchEmbedding)


@pytest.mark.parametrize("build", [_mae, _jepa], ids=["mae", "jepa"])
def test_channel_adaptive_forward_backward(build):
    model = build(patch_embed_type="channel_adaptive", patch_embed_args=ARGS).train()
    encs = [model.masked_encoder] if isinstance(model, MaskedAutoEncoder) else [model.input_encoder, model.target_encoder]
    for enc in encs:
        pe = enc.patch_embedding
        assert isinstance(pe, ChannelAdaptivePatchEmbedding) and pe.channel_fusion == "attn_pool"
        assert pe.localization_embed.num_embeddings == 3 + 4
        assert pe.fluorophore_embed.num_embeddings == 4 + 4

    sample = _sample(model)
    loss_dict, predictions = model(sample)
    loss = loss_dict["step_loss"]
    assert loss.ndim == 0 and torch.isfinite(loss) and torch.isfinite(predictions).all()
    loss.backward()
    trainable = model.masked_encoder if isinstance(model, MaskedAutoEncoder) else model.input_encoder
    pe = trainable.patch_embedding
    for name in ("proj", "localization_embed", "fluorophore_embed", "pool_attn"):
        grads = [p.grad for p in getattr(pe, name).parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name

    # the ids change the output (they are really used)
    model.eval()
    with torch.no_grad():
        a = model(sample)[1]
        other = dict(sample, metainfo={**sample["metainfo"], "channel_ids": torch.ones_like(_ids(1))})
        b = model(other)[1]
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("build", [_mae, _jepa], ids=["mae", "jepa"])
def test_channel_adaptive_requires_channel_ids_in_meta(build):
    model = build(patch_embed_type="channel_adaptive", patch_embed_args=ARGS)
    with pytest.raises(ValueError, match="channel_ids"):
        model(_sample(model, channel_ids=False))


def test_joint_ignores_channel_ids_in_meta():
    model = _mae().eval()
    sample = _sample(model, channel_ids=True)
    ref = model(dict(sample, metainfo={k: v for k, v in sample["metainfo"].items() if k != "channel_ids"}))[1]
    torch.testing.assert_close(model(sample)[1], ref)
