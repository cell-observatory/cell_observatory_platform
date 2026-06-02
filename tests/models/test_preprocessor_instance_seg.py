"""Tests for InstanceSegmentationPreprocessor on-device mask materialization.

`materialize_binary_masks=True` makes the preprocessor build per-instance
binary masks from each target's integer `label_map` (via `mask_ids_to_masks`)
and attach them as `t["masks"]`. Dense-mask heads (Mask2Former / PlainDETR /
multilabel) opt into this; labelmap-native MaskDINO leaves it off. These tests
exercise `_materialize_target_masks` directly, bypassing the base preprocessor
`__init__` which expects a Ray-style runtime config.
"""
from __future__ import annotations

import torch

from cell_observatory_platform.models.layers.preprocessor import InstanceSegmentationPreprocessor


def _make_preprocessor() -> InstanceSegmentationPreprocessor:
    pp = InstanceSegmentationPreprocessor.__new__(InstanceSegmentationPreprocessor)
    pp.materialize_binary_masks = True
    return pp


def test_materialize_target_masks_from_labelmap():
    device = torch.device("cpu")
    Z, Y, X = 2, 3, 4

    # Two samples; each carries an integer labelmap and the instance ids present.
    lm0 = torch.zeros((Z, Y, X), dtype=torch.int32, device=device)
    lm0[0, 0, 0] = 7
    lm0[1, 2, 3] = 11
    lm1 = torch.zeros((Z, Y, X), dtype=torch.int32, device=device)
    lm1[0, 1, 1] = 5

    targets = [
        {"label_map": lm0, "mask_ids": torch.tensor([7, 11], dtype=torch.long)},
        {"label_map": lm1, "mask_ids": torch.tensor([5], dtype=torch.long)},
    ]

    pp = _make_preprocessor()
    pp._materialize_target_masks(targets, device)

    # Sample 0: 2 instances -> [2, Z, Y, X] bool.
    m0 = targets[0]["masks"]
    assert m0.shape == (2, Z, Y, X) and m0.dtype == torch.bool
    assert m0[0].sum().item() == 1 and m0[0, 0, 0, 0]
    assert m0[1].sum().item() == 1 and m0[1, 1, 2, 3]
    # Distinct instances must not overlap.
    assert not torch.any(m0[0] & m0[1])

    # Sample 1: 1 instance.
    m1 = targets[1]["masks"]
    assert m1.shape == (1, Z, Y, X) and m1.dtype == torch.bool
    assert m1[0].sum().item() == 1 and m1[0, 0, 1, 1]


def test_materialize_casts_uint16_labelmap_to_int32():
    # uint16 ids > 256 must survive (no bf16 aliasing, no missing-kernel crash).
    device = torch.device("cpu")
    Z, Y, X = 1, 2, 2
    lm = torch.zeros((Z, Y, X), dtype=torch.uint16, device=device)
    lm[0, 0, 0] = 4097  # > uint8 / bf16-exact range

    targets = [{"label_map": lm, "mask_ids": torch.tensor([4097], dtype=torch.long)}]

    pp = _make_preprocessor()
    pp._materialize_target_masks(targets, device)

    m = targets[0]["masks"]
    assert m.shape == (1, Z, Y, X) and m.dtype == torch.bool
    assert m[0, 0, 0, 0] and m[0].sum().item() == 1


def test_materialize_skips_when_label_map_absent():
    # Ad-hoc inference views without label_map are left untouched.
    device = torch.device("cpu")
    targets = [{"mask_ids": torch.tensor([1], dtype=torch.long)}]

    pp = _make_preprocessor()
    pp._materialize_target_masks(targets, device)

    assert "masks" not in targets[0]


# --------------------------------------------------------------------------- #
# forward(): single-source labelmap split off the data_tensor channel
# --------------------------------------------------------------------------- #


def _make_forward_pp(materialize: bool = True, mask_channel_idx: int = -1) -> InstanceSegmentationPreprocessor:
    pp = InstanceSegmentationPreprocessor.__new__(InstanceSegmentationPreprocessor)
    pp.dtype = torch.float32
    pp.mask_channel_idx = mask_channel_idx
    pp.transforms = None
    pp.debug_savepath = None
    pp.materialize_binary_masks = materialize
    pp.with_masking = False
    return pp


def test_forward_populates_label_map_and_masks_from_channel():
    # The preprocessor (not the collator) owns the labelmap: it splits the
    # int32 labelmap off the channel, attaches it per-target, and materializes
    # masks from it.
    B, Z, Y, X, C = 1, 2, 3, 4, 2
    img = torch.randn(B, Z, Y, X, C)
    lm = torch.zeros(B, Z, Y, X)
    lm[0, 0, 0, 0] = 7
    lm[0, 1, 2, 3] = 11
    inputs = torch.cat([img, lm.unsqueeze(-1)], dim=-1)  # (B,Z,Y,X,C+1)

    targets = [
        {
            "mask_ids": torch.tensor([7, 11], dtype=torch.long),
            "boxes": torch.zeros((2, 6), dtype=torch.float32),
            "labels": torch.zeros((2,), dtype=torch.long),
        }
    ]

    pp = _make_forward_pp(materialize=True)
    out = pp.forward({"data_tensor": inputs, "metainfo": {"targets": targets}}, 0.0, 0)
    tgt = out["metainfo"]["targets"][0][0]

    # Mask channel stripped from the image tensor.
    assert out["data_tensor"].shape[-1] == C
    # label_map attached from the channel (int32), masks built from it.
    assert "label_map" in tgt and torch.equal(tgt["label_map"], lm[0].to(torch.int32))
    assert tgt["masks"].shape == (2, Z, Y, X) and tgt["masks"].dtype == torch.bool


def test_forward_no_mask_channel_skips_label_map():
    # No mask channel (inference) -> no label_map, no masks.
    B, Z, Y, X, C = 1, 2, 3, 4, 2
    inputs = torch.randn(B, Z, Y, X, C)
    targets = [
        {
            "mask_ids": torch.zeros((0,), dtype=torch.long),
            "boxes": torch.zeros((0, 6), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
        }
    ]

    pp = _make_forward_pp(materialize=True, mask_channel_idx=None)
    out = pp.forward({"data_tensor": inputs, "metainfo": {"targets": targets}}, 0.0, 0)
    tgt = out["metainfo"]["targets"][0][0]

    assert out["data_tensor"].shape[-1] == C
    assert "label_map" not in tgt
    assert "masks" not in tgt
