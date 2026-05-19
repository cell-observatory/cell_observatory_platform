"""Tests for SAM2VideoPreprocessor target-view fields (labelmap-native path).

`_build_data_views_lazy` adds labelmap, instance_ids (with sentinel -1 for pad),
valid, presence_t, boxes, and box_format alongside the legacy `masks`/`img_ids`
contract. These tests exercise it through the class directly, bypassing the
base preprocessor `__init__` which expects a Ray-style runtime config.
"""
from __future__ import annotations

import pytest
import torch

from cell_observatory_platform.models.layers.preprocessor import SAM2VideoPreprocessor


def _make_preprocessor(max_masks: int, bbox_format: str = "zyxzyx") -> SAM2VideoPreprocessor:
    # Bypass __init__: the lazy data-view builder only needs max_masks and bbox_format.
    pp = SAM2VideoPreprocessor.__new__(SAM2VideoPreprocessor)
    pp.max_masks = max_masks
    pp.bbox_format = bbox_format
    return pp


def test_lazy_data_view_fields_shapes_and_pad_sentinel():
    device = torch.device("cpu")
    B, T, Z, Y, X = 2, 3, 2, 4, 5
    K_full = 4

    # Construct a labelmap whose voxels are mostly background (0) plus a few
    # objects with known integer ids placed deterministically.
    labelmap = torch.zeros((B, T, Z, Y, X), dtype=torch.int32, device=device)
    labelmap[0, 0, 0, 0, 0] = 7    # video 0, frame 0: id=7 present
    labelmap[0, 1, 1, 2, 3] = 11   # video 0, frame 1: id=11 present
    labelmap[1, 0, 0, 1, 1] = 13   # video 1, frame 0: id=13 present
    # Video 0 has ids [7, 11]; video 1 has id [13].

    targets = [
        {
            "mask_ids": torch.tensor([7, 11], dtype=torch.long, device=device),
            "boxes": torch.tensor(
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                 [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]],
                dtype=torch.float32, device=device,
            ),
        },
        {
            "mask_ids": torch.tensor([13], dtype=torch.long, device=device),
            "boxes": torch.tensor(
                [[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]],
                dtype=torch.float32, device=device,
            ),
        },
    ]

    pp = _make_preprocessor(max_masks=K_full, bbox_format="zyxzyx")
    view = pp._build_data_views_lazy(
        targets=targets,
        num_frames=T,
        num_videos=B,
        device=device,
        mask_labelmap=labelmap,
    )

    # Top-level keys.
    expected_keys = {
        "num_frames", "num_videos", "masks", "img_ids",
        "labelmaps", "instance_ids", "valid", "presence_t", "boxes", "box_format",
    }
    assert expected_keys.issubset(view.keys()), f"missing keys: {expected_keys - view.keys()}"
    assert view["box_format"] == "zyxzyx"
    assert view["num_frames"] == T and view["num_videos"] == B

    # labelmaps shape and indexing: flat_id = b*T + t -> mask_labelmap[b, t].
    flat = view["labelmaps"]
    assert flat.shape == (B * T, Z, Y, X)
    for b in range(B):
        for t in range(T):
            assert torch.equal(flat[b * T + t], labelmap[b, t].to(torch.int32))

    # Per-frame fields: each list has length T, each tensor has B*K_full rows.
    for t in range(T):
        for key, expected_shape in [
            ("img_ids", (B * K_full,)),
            ("instance_ids", (B * K_full,)),
            ("valid", (B * K_full,)),
            ("presence_t", (B * K_full,)),
            ("boxes", (B * K_full, 6)),
            ("masks", (B * K_full, Z, Y, X)),
        ]:
            assert view[key][t].shape == expected_shape, (
                f"{key}[{t}] shape {tuple(view[key][t].shape)} != {expected_shape}"
            )

    # Padding sentinels: video 0 has 2 real ids (K=2), pads 2; video 1 has 1
    # real id (K=1), pads 3. Pad rows must have instance_ids=-1, valid=False,
    # presence_t=False, boxes=0.
    for t in range(T):
        inst = view["instance_ids"][t]
        valid = view["valid"][t]
        presence = view["presence_t"][t]
        boxes = view["boxes"][t]

        # video 0 slots: rows 0..K_full-1
        assert torch.equal(inst[:2], torch.tensor([7, 11], dtype=torch.int64))
        assert torch.all(valid[:2])
        assert torch.equal(inst[2:K_full], torch.full((K_full - 2,), -1, dtype=torch.int64))
        assert not torch.any(valid[2:K_full])
        assert not torch.any(presence[2:K_full])
        assert torch.all(boxes[2:K_full] == 0)

        # video 1 slots: rows K_full..2*K_full-1
        offset = K_full
        assert torch.equal(inst[offset : offset + 1], torch.tensor([13], dtype=torch.int64))
        assert valid[offset]
        assert torch.equal(inst[offset + 1 : 2 * K_full], torch.full((K_full - 1,), -1, dtype=torch.int64))
        assert not torch.any(valid[offset + 1 : 2 * K_full])
        assert not torch.any(presence[offset + 1 : 2 * K_full])
        assert torch.all(boxes[offset + 1 : 2 * K_full] == 0)


def test_lazy_data_view_presence_tracks_frame_membership():
    # presence_t must reflect whether each object's id appears in the frame's
    # labelmap, NOT just whether the row is a real selected object (valid).
    device = torch.device("cpu")
    B, T, Z, Y, X = 1, 3, 2, 3, 4
    K_full = 2

    labelmap = torch.zeros((B, T, Z, Y, X), dtype=torch.int32, device=device)
    labelmap[0, 0, 0, 0, 0] = 5   # frame 0 has id 5 only
    labelmap[0, 1, 1, 2, 3] = 7   # frame 1 has id 7 only
    # frame 2 has neither

    targets = [
        {
            "mask_ids": torch.tensor([5, 7], dtype=torch.long, device=device),
            "boxes": torch.zeros((2, 6), dtype=torch.float32, device=device),
        }
    ]

    pp = _make_preprocessor(max_masks=K_full)
    view = pp._build_data_views_lazy(
        targets=targets,
        num_frames=T,
        num_videos=B,
        device=device,
        mask_labelmap=labelmap,
    )

    # Frame 0: id 5 present, id 7 absent.
    assert torch.equal(view["presence_t"][0], torch.tensor([True, False]))
    # Frame 1: id 5 absent, id 7 present.
    assert torch.equal(view["presence_t"][1], torch.tensor([False, True]))
    # Frame 2: both absent.
    assert torch.equal(view["presence_t"][2], torch.tensor([False, False]))

    # valid stays True for both rows regardless of frame membership.
    for t in range(T):
        assert torch.equal(view["valid"][t], torch.tensor([True, True]))


def test_lazy_data_view_empty_targets_all_pad():
    # Missing targets entry (e.g. inference) should produce all-pad rows.
    device = torch.device("cpu")
    B, T, Z, Y, X = 1, 1, 2, 2, 2
    K_full = 3

    labelmap = torch.randint(0, 4, (B, T, Z, Y, X), dtype=torch.int32, device=device)
    targets: list[dict] = [{}]  # no mask_ids

    pp = _make_preprocessor(max_masks=K_full)
    view = pp._build_data_views_lazy(
        targets=targets,
        num_frames=T,
        num_videos=B,
        device=device,
        mask_labelmap=labelmap,
    )

    assert torch.equal(view["instance_ids"][0], torch.tensor([-1, -1, -1], dtype=torch.int64))
    assert not torch.any(view["valid"][0])
    assert not torch.any(view["presence_t"][0])
    assert torch.all(view["boxes"][0] == 0)
