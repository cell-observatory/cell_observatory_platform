"""`ChannelDropout`: zero a random subset of signal channels per sample, never the
kept ones, never all of them; optional shuffle that also permutes channel_ids."""
import json

import pytest
import torch

from cell_observatory_platform.data.transforms.channel_dropout import ChannelDropout

TOKENS = [["membrane", "mstaygold"], ["cytosol", "electra2"], ["cytosol", "mtfp1"], ["cytosol", "mkok"]]


def _sample(B=2, C=4, with_tokens=True, ids=False, as_json=False):
    x = torch.rand(B, 1, 2, 3, 3, C) + 1.0            # strictly positive: a zeroed channel is detectable
    meta = {}
    if with_tokens:
        meta["channel_tokens"] = [json.dumps(TOKENS[:C]) if as_json else TOKENS[:C] for _ in range(B)]
    if ids:
        meta["channel_ids"] = torch.arange(C * 2).view(1, C, 2).repeat(B, 1, 1)
    return {"data_tensor": x, "metainfo": meta}


def _zeroed(x, b):
    return {c for c in range(x.shape[-1]) if bool((x[b, ..., c] == 0).all())}


def test_kept_token_channel_is_never_zeroed_and_at_most_all_but_one_droppable():
    t = ChannelDropout(p=1.0, always_keep=["membrane"], seed=0)
    out = t(_sample())
    for b in range(2):
        z = _zeroed(out["data_tensor"], b)
        assert 0 not in z
        assert z == {1, 2} or z == {1, 3} or z == {2, 3}      # exactly C-2 of the 3 droppable


def test_kept_position_form_and_json_tokens():
    t = ChannelDropout(p=1.0, always_keep=[2], seed=0)
    out = t(_sample(as_json=True))
    assert all(2 not in _zeroed(out["data_tensor"], b) for b in range(2))
    t2 = ChannelDropout(p=1.0, always_keep=["mstaygold"], seed=0)     # fluorophore token matches too
    out2 = t2(_sample(as_json=True))
    assert all(0 not in _zeroed(out2["data_tensor"], b) for b in range(2))


def test_p_zero_is_identity():
    s = _sample()
    ref = s["data_tensor"].clone()
    out = ChannelDropout(p=0.0, always_keep=["membrane"], seed=1)(s)
    torch.testing.assert_close(out["data_tensor"], ref)


def test_seeded_reproducibility():
    a = ChannelDropout(p=0.5, always_keep=["membrane"], seed=7)(_sample())["data_tensor"]
    torch.manual_seed(0)
    b = ChannelDropout(p=0.5, always_keep=["membrane"], seed=7)(_sample())["data_tensor"]
    assert [_zeroed(a, i) for i in range(2)] == [_zeroed(b, i) for i in range(2)]


def test_shuffle_permutes_data_and_channel_ids_identically():
    s = _sample(ids=True)
    ref = s["data_tensor"].clone()
    ref_ids = s["metainfo"]["channel_ids"].clone()
    out = ChannelDropout(p=0.0, always_keep=["membrane"], shuffle=True, seed=3)(s)
    x, ids = out["data_tensor"], out["metainfo"]["channel_ids"]
    for b in range(2):
        torch.testing.assert_close(x[b, ..., 0], ref[b, ..., 0])            # kept channel untouched
        assert torch.equal(ids[b, 0], ref_ids[b, 0])
        # every droppable channel moved WITH its ids: find where original channel c landed
        for c in (1, 2, 3):
            dst = [d for d in (1, 2, 3) if torch.equal(x[b, ..., d], ref[b, ..., c])]
            assert len(dst) == 1 and torch.equal(ids[b, dst[0]], ref_ids[b, c])
    # a permutation, not a copy: the sorted channel set is unchanged
    torch.testing.assert_close(x.sort(dim=-1).values, ref.sort(dim=-1).values)


def test_token_keep_without_channel_tokens_raises():
    with pytest.raises(ValueError, match="no 'channel_tokens'"):
        ChannelDropout(p=0.5, always_keep=["membrane"])(_sample(with_tokens=False))


def test_token_keep_that_matches_nothing_raises():
    with pytest.raises(ValueError, match="match no signal channel"):
        ChannelDropout(p=0.5, always_keep=["golgi"])(_sample())


def test_mask_positions_in_tokens_are_ignored():
    s = _sample(C=4)
    s["metainfo"]["channel_tokens"] = [[["membrane", "mstaygold"], None, ["cytosol", "mtfp1"], None]] * 2
    out = ChannelDropout(p=1.0, always_keep=["membrane"], seed=0)(s)
    assert all(0 not in _zeroed(out["data_tensor"], b) for b in range(2))


def test_invalid_p_rejected():
    with pytest.raises(ValueError):
        ChannelDropout(p=1.5)
