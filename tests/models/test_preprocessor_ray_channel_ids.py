"""RayPreprocessor (pretraining path) emits ``metainfo["channel_ids"]`` from the
loader's ``channel_tokens`` when a ``channel_vocab`` is set -- same contract as
SAM2VideoPreprocessor (shared ``_ChannelIdsMixin``): ``[B, C, 2]`` long ids,
attached BEFORE the transforms, unknown_policy error/unk, unk_token_p dropout."""
from __future__ import annotations

import json

import pytest
import torch

from cell_observatory_platform.models.layers.preprocessor import RayPreprocessor

VOCAB = {
    "localization": {"<unk>": 0, "cytosol": 1, "membrane": 2},
    "fluorophore": {"<unk>": 0, "electra2": 1, "mstaygold": 2},
}
TOKENS = [["membrane", "mstaygold"], ["cytosol", "electra2"]]
SHAPE = (4, 8, 8, 2)            # Z, Y, X, C  (2 signal channels, no mask channel)


def _pp(channel_vocab=VOCAB, **kw):
    return RayPreprocessor(
        dtype=torch.float32, with_masking=False, input_format="ZYXC",
        input_shape=SHAPE, patch_shape=(2, 2, 2, None), mask_generator=None,
        transforms_list=None, channel_vocab=channel_vocab, **kw,
    )


def _sample(B=2, tokens=TOKENS, as_json=False):
    rows = [json.dumps(tokens) if as_json else tokens] * B
    return {"data_tensor": torch.randn(B, *SHAPE), "metainfo": {"channel_tokens": rows}}


def test_forward_emits_channel_ids_in_mae_jepa_shape_and_dtype():
    out = _pp().forward(_sample(), 0.0, 0)
    ids = out["metainfo"]["channel_ids"]
    assert ids.shape == (2, 2, 2) and ids.dtype == torch.long
    assert ids[0].tolist() == [[2, 2], [1, 1]] and torch.equal(ids[0], ids[1])
    # the collator's per-sample JSON strings parse the same way
    assert torch.equal(_pp().forward(_sample(as_json=True), 0.0, 0)["metainfo"]["channel_ids"], ids)


def test_forward_without_vocab_emits_no_ids_and_missing_tokens_raise():
    out = _pp(channel_vocab=None).forward({"data_tensor": torch.randn(1, *SHAPE), "metainfo": {}}, 0.0, 0)
    assert "channel_ids" not in out["metainfo"]
    with pytest.raises(ValueError, match="no 'channel_tokens'"):
        _pp().forward({"data_tensor": torch.randn(1, *SHAPE), "metainfo": {}}, 0.0, 0)


def test_channel_ids_attached_before_transforms():
    seen = {}

    def spy(sample):
        seen["ids"] = sample["metainfo"].get("channel_ids")
        return sample

    pp = _pp()
    pp.transforms = [spy]
    out = pp.forward(_sample(B=1), 0.0, 0)
    assert seen["ids"] is not None and seen["ids"].shape == (1, 2, 2)
    assert torch.equal(out["metainfo"]["channel_ids"], seen["ids"])


def test_unknown_token_policy():
    toks = [["golgi", "mstaygold"], ["cytosol", None]]
    with pytest.raises(ValueError, match="golgi.*not in channel_vocab"):
        _pp().forward(_sample(B=1, tokens=toks), 0.0, 0)
    ids = _pp(unknown_policy="unk").forward(_sample(B=1, tokens=toks), 0.0, 0)["metainfo"]["channel_ids"]
    assert ids[0].tolist() == [[0, 2], [1, 0]]                    # <unk> localization, <unk> fluorophore
    # inference never raises: unknown -> <unk>
    ids_eval = _pp().eval().forward(_sample(B=1, tokens=toks), 0.0, 0)["metainfo"]["channel_ids"]
    assert torch.equal(ids, ids_eval)
    with pytest.raises(ValueError, match="unknown_policy"):
        _pp(unknown_policy="drop")


def test_unk_token_dropout_only_in_training():
    pp = _pp(unk_token_p=1.0)
    assert pp.forward(_sample(B=1), 0.0, 0)["metainfo"]["channel_ids"].tolist() == [[[0, 0], [0, 0]]]
    assert pp.eval().forward(_sample(B=1), 0.0, 0)["metainfo"]["channel_ids"][0].tolist() == [[2, 2], [1, 1]]
    pp = _pp(unk_token_p=0.5).train()
    torch.manual_seed(0)
    many = torch.stack([pp.forward(_sample(B=1), 0.0, 0)["metainfo"]["channel_ids"] for _ in range(200)])
    frac = (many == 0).float().mean().item()
    assert 0.35 < frac < 0.65                                     # per-entry coin, not all-or-nothing
    with pytest.raises(ValueError, match="unk_token_p"):
        _pp(unk_token_p=1.5)
