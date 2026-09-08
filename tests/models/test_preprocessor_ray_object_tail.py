"""RayPreprocessor strips an object-role channel tail (DB cubes carry the
instance-mask channel after the signal channels) BEFORE the dtype cast, the
channel ids and the transforms; a channel-count mismatch that is not explained
by object roles raises a clear ValueError."""
from __future__ import annotations

import json

import pytest
import torch

from cell_observatory_platform.models.layers.preprocessor import RayPreprocessor

VOCAB = {
    "localization": {"<unk>": 0, "cytosol": 1, "membrane": 2, "nucleus": 3},
    "fluorophore": {"<unk>": 0, "electra2": 1, "mstaygold": 2, "halo": 3},
}
N_SIG = 5
TOKENS = [["membrane", "mstaygold"], ["cytosol", "electra2"], ["nucleus", "halo"],
          ["membrane", "electra2"], ["cytosol", "halo"], None]      # None: mask slot
SHAPE_ZYXC = (4, 8, 8, N_SIG)
SHAPE_TZYXC = (2, 4, 8, 8, N_SIG)


def _pp(input_format="ZYXC", input_shape=SHAPE_ZYXC, channel_vocab=VOCAB, transforms=None):
    patch = (2, 2, 2, None) if input_format == "ZYXC" else (1, 2, 2, 2, None)
    pp = RayPreprocessor(
        dtype=torch.float32, with_masking=False, input_format=input_format,
        input_shape=input_shape, patch_shape=patch, mask_generator=None,
        transforms_list=None, channel_vocab=channel_vocab,
    )
    if transforms:
        pp.transforms = list(transforms)
    return pp


def _sample(shape, B=2, mapping=None, tokens=TOKENS, extra_c=1, as_json=True):
    x = torch.randn(B, *shape[:-1], shape[-1] + extra_c)
    meta = {"channel_tokens": [json.dumps(tokens) if as_json else tokens] * B}
    if mapping is not None:
        meta["channel_mapping"] = [json.dumps(mapping) if as_json else mapping] * B
    return {"data_tensor": x, "metainfo": meta}, x


@pytest.mark.parametrize("fmt,shape", [("ZYXC", SHAPE_ZYXC), ("TZYXC", SHAPE_TZYXC)])
def test_object_tail_is_stripped_and_ids_cover_signal_only(fmt, shape):
    sample, x = _sample(shape, mapping={str(N_SIG): "instance_masks"})
    out = _pp(fmt, shape).forward(sample, 0.0, 0)
    y = out["data_tensor"]
    assert y.shape == (2, *shape)
    assert torch.equal(y, x[..., :N_SIG])
    ids = out["metainfo"]["channel_ids"]
    assert ids.shape == (2, N_SIG, 2) and ids.dtype == torch.long
    assert ids[0].tolist() == [[2, 2], [1, 1], [3, 3], [2, 1], [1, 3]]


def test_strip_happens_before_transforms_and_is_a_view():
    seen = {}

    def spy(sample):
        seen["shape"] = tuple(sample["data_tensor"].shape)
        seen["ids"] = sample["metainfo"]["channel_ids"].shape
        return sample

    sample, x = _sample(SHAPE_ZYXC, mapping={N_SIG: "instance_masks"}, as_json=False)
    pp = _pp(transforms=[spy])
    out = pp.forward(sample, 0.0, 0)
    assert seen["shape"] == (2, *SHAPE_ZYXC) and seen["ids"] == (2, N_SIG, 2)
    # zero-copy: the kept prefix shares storage with the loader buffer
    assert pp._signal_prefix(x, sample["metainfo"]).data_ptr() == x.data_ptr()
    assert out["data_tensor"].shape == (2, *SHAPE_ZYXC)


def test_mismatch_without_object_roles_raises():
    sample, _ = _sample(SHAPE_ZYXC, mapping=None)          # 6 channels, no mapping
    with pytest.raises(ValueError, match="6 channels of which 6 are signal"):
        _pp().forward(sample, 0.0, 0)
    sample, _ = _sample(SHAPE_ZYXC, mapping={"0": "membrane", "5": "cytosol"})
    with pytest.raises(ValueError, match="input_shape"):
        _pp().forward(sample, 0.0, 0)


def test_object_channel_not_in_tail_raises():
    sample, _ = _sample(SHAPE_ZYXC, mapping={"2": "instance_masks"})
    with pytest.raises(ValueError, match="contiguous prefix"):
        _pp().forward(sample, 0.0, 0)


def test_exact_channel_count_is_untouched():
    sample, x = _sample(SHAPE_ZYXC, mapping={}, tokens=TOKENS[:N_SIG], extra_c=0)
    out = _pp().forward(sample, 0.0, 0)
    assert torch.equal(out["data_tensor"], x)
