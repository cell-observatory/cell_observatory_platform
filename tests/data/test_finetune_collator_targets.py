"""Tests for FinetuneCollatorActor target construction.

The collator owns neither the labelmap nor any transforms: it emits only the
lightweight per-target metadata (boxes / mask_ids / labels) and ships the full
data_tensor (image channels + labelmap channel) to VRAM untouched. The model
preprocessor splits the labelmap off the channel, runs transforms, and builds
binary masks. These tests pin that contract by exercising `_build_targets`
directly, bypassing the Ray/shm/CUDA `__init__`.
"""
from __future__ import annotations

import pytest
import torch

from cell_observatory_platform.data.datasets.pretrain_dataset_ray import FinetuneCollatorActor


def _make_collator() -> FinetuneCollatorActor:
    c = FinetuneCollatorActor.__new__(FinetuneCollatorActor)
    c.bbox_data_format = "zyxzyx"
    c.bbox_output_format = "zyxzyx"
    c.normalize_bboxes = False
    c.spatial_shape = (2, 3, 4)
    c.input_format = "ZYXC"
    c._class_index = None          # class-agnostic unless a test says otherwise
    return c


def _bucket(*leaves):
    """One window-local timepoint bucket, the shape the DB now emits.

    annotations_metadata is time-outer / kind-inner with WINDOW-LOCAL keys
    (str(timepoint - time_start)), so a time_size == 1 sample always reads "0".
    """
    return {"0": {"instance": list(leaves), "semantic": []}}


def test_build_targets_emits_boxes_ids_and_labels_only():
    """Each target carries exactly boxes / mask_ids / labels; boxes are the float32
    zyxzyx rows as annotated and a missing object_type_id defaults to label 0."""
    c = _make_collator()
    anns = [_bucket(
        {"local_segmentation_id": 7, "bbox_zyxzyx": [0, 0, 0, 1, 1, 1], "object_type_id": 2},
        {"local_segmentation_id": 11, "bbox_zyxzyx": [1, 1, 1, 2, 2, 2]},
    )]
    (t,) = c._build_targets(annotations_metadata_batch=anns)
    assert set(t) == {"boxes", "mask_ids", "labels"}
    assert t["mask_ids"].tolist() == [7, 11]
    assert t["labels"].tolist() == [0, 0]   # class-agnostic: no catalog supplied
    assert t["boxes"].dtype == torch.float32
    assert t["boxes"].tolist() == [[0, 0, 0, 1, 1, 1], [1, 1, 1, 2, 2, 2]]


def test_build_targets_empty_annotations():
    c = _make_collator()
    targets = c._build_targets(annotations_metadata_batch=[{}])

    assert len(targets) == 1
    t = targets[0]
    assert t["mask_ids"].numel() == 0
    assert t["labels"].numel() == 0
    assert t["boxes"].shape == (0, 6)
    assert "label_map" not in t


def test_build_targets_converts_to_normalized_cxcyczwhd():
    """bbox_output_format="cxcyczwhd" + normalize_bboxes: zyxzyx corners become an
    xyz centre/size box divided by the (W, H, D) spatial extent."""
    c = _make_collator()                                        # spatial_shape = (Z, Y, X) = (2, 3, 4)
    c.bbox_output_format, c.normalize_bboxes = "cxcyczwhd", True
    (t,) = c._build_targets([_bucket({"local_segmentation_id": 1, "bbox_zyxzyx": [0, 0, 0, 1, 1, 1]})])
    # zyx (0,0,0)-(1,1,1) -> xyz centre (.5,.5,.5), size 1 -> divided by (W=4, H=3, D=2)
    torch.testing.assert_close(t["boxes"], torch.tensor([[1 / 8, 1 / 6, 1 / 4, 1 / 4, 1 / 3, 1 / 2]]))


def test_build_targets_skips_malformed_annotations():
    """Rows without an id, without a box, or with a box that is not 6 coordinates
    are skipped rather than raising; well-formed rows in the same sample survive."""
    c = _make_collator()
    anns = [_bucket(
        {"local_segmentation_id": None, "bbox_zyxzyx": [0, 0, 0, 1, 1, 1]},   # no id
        {"local_segmentation_id": 3, "bbox_zyxzyx": [0, 0, 0, 1, 1]},         # 5 coords
        {"local_segmentation_id": 4},                                           # no box
        {"local_segmentation_id": 5, "bbox_zyxzyx": [0, 0, 0, 2, 2, 2]},
    )]
    (t,) = c._build_targets(anns)
    assert t["mask_ids"].tolist() == [5]
    assert t["boxes"].tolist() == [[0, 0, 0, 2, 2, 2]]


def test_build_targets_parses_json_string_rows():
    """DB rows arrive as JSON text, the literal "null", or bytes; each is parsed
    into a per-sample target (empty payloads give empty targets)."""
    c = _make_collator()
    row = (
        '{"0": {"instance": [{"local_segmentation_id": 9, '
        '"bbox_zyxzyx": [0, 0, 0, 1, 2, 3], "object_type_id": 1}], "semantic": []}}'
    )
    t0, t1, t2 = c._build_targets([row, "null", b"{}"])
    assert t0["mask_ids"].tolist() == [9] and t0["labels"].tolist() == [0]
    assert t0["boxes"].tolist() == [[0, 0, 0, 1, 2, 3]]
    assert t1["boxes"].shape == (0, 6) and t2["mask_ids"].numel() == 0


def test_build_targets_reads_only_the_window_local_bucket():
    """Keys are window-local (str(timepoint - time_start)), so a time_size == 1
    sample reads "0" regardless of where it sits in the tile -- a row at
    time_start=240 still keys its single bucket "0", not "240". A bucket under
    any other key belongs to a different timepoint and must not leak in."""
    c = _make_collator()
    anns = [{
        "0": {"instance": [{"local_segmentation_id": 1, "bbox_zyxzyx": [0, 0, 0, 1, 1, 1]}],
              "semantic": []},
        "1": {"instance": [{"local_segmentation_id": 2, "bbox_zyxzyx": [0, 0, 0, 1, 1, 1]}],
              "semantic": []},
    }]
    (t,) = c._build_targets(anns)
    assert t["mask_ids"].tolist() == [1]


# --------------------------------------------------------------------------- #
# 4D signposting: the limitation is targets-only, and it must be loud at both
# the config boundary (actor construction) and the data boundary (payload).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("input_shape", [(16, 32, 32, 32, 2), (2, 8, 8, 8, 1)])
def test_tzyxc_with_targets_is_refused_at_construction(input_shape):
    """Fail at actor startup, not on batch N of epoch 0."""
    with pytest.raises(NotImplementedError, match="cannot build targets for a"):
        FinetuneCollatorActor._assert_targets_supported(
            "TZYXC", input_shape, require_targets=True
        )


def test_tzyxc_without_targets_is_allowed():
    """Inference is 4D-clean: the image path carries T end to end, and no targets
    are built, so there is nothing to drop."""
    FinetuneCollatorActor._assert_targets_supported(
        "TZYXC", (16, 32, 32, 32, 2), require_targets=False
    )


def test_tzyxc_with_unit_time_is_allowed():
    """A TZYXC layout with T=1 is a 3D sample wearing a 4D shape -- one bucket,
    nothing dropped."""
    FinetuneCollatorActor._assert_targets_supported(
        "TZYXC", (1, 32, 32, 32, 2), require_targets=True
    )


def test_zyxc_is_never_refused():
    FinetuneCollatorActor._assert_targets_supported(
        "ZYXC", (32, 32, 32, 2), require_targets=True
    )


def test_build_targets_rejects_the_legacy_bare_list():
    """The payload was a flat List[dict]; it is a time-keyed object now. A bare
    list must raise rather than silently parse as zero annotations."""
    c = _make_collator()
    with pytest.raises(ValueError, match="time-keyed"):
        c._build_targets([[{"local_segmentation_id": 1, "bbox_zyxzyx": [0, 0, 0, 1, 1, 1]}]])


def test_build_targets_rejects_tile_frame_bbox():
    """cube_training publishes cube-local CLIPPED boxes; tiles_training publishes
    tile-relative UNCLIPPED ones. Same key, same six numbers -- a tile-frame box
    would otherwise pass every shape check and produce wrong targets."""
    c = _make_collator()                                   # spatial_shape = (2, 3, 4)
    with pytest.raises(ValueError, match="exceeds the cube extent"):
        c._build_targets([_bucket(
            {"local_segmentation_id": 1, "bbox_zyxzyx": [0, 0, 0, 900, 900, 900]}
        )])


def test_object_type_id_maps_to_a_contiguous_class_index():
    """object_type_id is a DB PRIMARY KEY (1-based); the model's label space is
    0..num_classes-1 with num_classes itself meaning no-object. Passing the raw id
    through would shift every label by one and, for a single-class run, put every
    object on the no-object slot.
    """
    c = _make_collator()
    c._class_index = {1: 0, 4: 1, 9: 2}          # catalog ids 1, 4, 9
    anns = [_bucket(
        {"local_segmentation_id": 1, "bbox_zyxzyx": [0, 0, 0, 1, 1, 1], "object_type_id": 9},
        {"local_segmentation_id": 2, "bbox_zyxzyx": [0, 0, 0, 1, 1, 1], "object_type_id": 1},
    )]
    (t,) = c._build_targets(anns)
    assert t["labels"].tolist() == [2, 0]


def test_no_catalog_is_class_agnostic():
    """With no object-type catalog, every object comes through as class 0
    whatever its object_type_id."""
    c = _make_collator()
    c._class_index = None
    (t,) = c._build_targets([_bucket(
        {"local_segmentation_id": 1, "bbox_zyxzyx": [0, 0, 0, 1, 1, 1], "object_type_id": 7}
    )])
    assert t["labels"].tolist() == [0]


def test_unknown_object_type_id_raises():
    """A stale catalog must not silently relabel objects."""
    c = _make_collator()
    c._class_index = {1: 0}
    with pytest.raises(KeyError, match="object_type_id=7"):
        c._build_targets([_bucket(
            {"local_segmentation_id": 1, "bbox_zyxzyx": [0, 0, 0, 1, 1, 1], "object_type_id": 7}
        )])
