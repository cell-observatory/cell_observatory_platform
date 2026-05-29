"""Tests for FinetuneCollatorActor target construction.

The collator no longer owns the labelmap or any transforms: it emits only the
lightweight per-target metadata (boxes / mask_ids / labels) and ships the full
data_tensor (image channels + labelmap channel) to VRAM untouched. The model
preprocessor splits the labelmap off the channel, runs transforms, and builds
binary masks. These tests pin that contract by exercising `_build_targets`
directly, bypassing the Ray/shm/CUDA `__init__`.
"""
from __future__ import annotations

import torch

from cell_observatory_platform.data.datasets.pretrain_dataset_ray import FinetuneCollatorActor


def _make_collator() -> FinetuneCollatorActor:
    c = FinetuneCollatorActor.__new__(FinetuneCollatorActor)
    c.bbox_data_format = "zyxzyx"
    c.bbox_output_format = "zyxzyx"
    c.normalize_bboxes = False
    c.spatial_shape = (2, 3, 4)
    c.input_format = "ZYXC"
    return c


def test_build_targets_emits_no_label_map():
    c = _make_collator()
    anns = [
        [
            {"local_segmentation_id": 7, "bbox_zyxzyx": [0, 0, 0, 1, 1, 1], "cell_type_id": 2},
            {"local_segmentation_id": 11, "bbox_zyxzyx": [1, 1, 1, 2, 2, 2]},
        ]
    ]

    targets = c._build_targets(annotations_metadata_batch=anns)

    assert len(targets) == 1
    t = targets[0]
    # The collator never produces a labelmap or binary masks.
    assert set(t.keys()) == {"boxes", "mask_ids", "labels"}
    assert "label_map" not in t and "masks" not in t

    assert t["mask_ids"].tolist() == [7, 11]
    assert t["labels"].tolist() == [2, 0]  # default cell_type_id -> 0
    assert t["boxes"].shape == (2, 6)


def test_build_targets_empty_annotations():
    c = _make_collator()
    targets = c._build_targets(annotations_metadata_batch=[[]])

    assert len(targets) == 1
    t = targets[0]
    assert t["mask_ids"].numel() == 0
    assert t["labels"].numel() == 0
    assert t["boxes"].shape == (0, 6)
    assert "label_map" not in t


def test_collator_has_no_transform_plumbing():
    # The transform attribute/param were removed; the collator is transform-free.
    assert not hasattr(FinetuneCollatorActor, "_get_masks")
    import inspect

    sig = inspect.signature(FinetuneCollatorActor.__init__)
    for removed in ("transforms_list", "use_masks", "mask_channel_idx"):
        assert removed not in sig.parameters, f"{removed} should be removed from collator __init__"
