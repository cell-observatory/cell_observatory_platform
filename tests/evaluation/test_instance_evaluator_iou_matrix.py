"""Differential tests: the evaluator's IoU-matrix construction (`label_map`
bincount path, `masks` cached-chunk path) must equal `gt_masks_for_class` +
`_pairwise_mask_iou_3d_bool` exactly.
"""

import pytest
import torch

import cell_observatory_platform.evaluation.evaluate_postprocess as ep
from cell_observatory_platform.evaluation.instance_segmentation_evaluator import (
    InstanceSegmentationEvaluator,
    _pairwise_mask_iou_3d_bool,
)
from cell_observatory_platform.evaluation.metrics import MaskMAPMetric


# ---------------------------------------------------------------------------
# Reference: materialize per-instance GT masks, pairwise IoU.
# ---------------------------------------------------------------------------


def _reference_ious(pred_masks, target, gt_class_mask, eval_frame_size, source):
    """materialize-per-pair reference (chunking cannot change the math)."""
    gt = ep.gt_masks_for_class(target, gt_class_mask, eval_frame_size, source)
    return _pairwise_mask_iou_3d_bool(pred_masks.bool(), gt.bool())


def _make_evaluator(source: str, match_labels: bool, chunk: int):
    return InstanceSegmentationEvaluator(
        metrics=[{"name": "mask_map", "key": "mask_map"}],
        mask_chunk_size=chunk,
        match_labels=match_labels,
        gt_mask_source=source,
        gt_boxes_normalized=False,
        gt_box_format="xyzxyz",
    )


def _sample(pred_masks, class_ids, eval_frame_size):
    n = pred_masks.shape[0]
    return {
        "topk_query_indices": torch.arange(n, dtype=torch.long),
        "topk_class_scores": torch.rand(n),
        "topk_class_ids": torch.as_tensor(class_ids, dtype=torch.long),
        "boxes": torch.zeros(n, 6, dtype=torch.float32),
        "eval_frame_size": tuple(eval_frame_size),
        "pred_masks": pred_masks.bool(),
        "mask_source": "direct",
    }


def _random_case(seed, shape=(8, 16, 16), ids=(3, 7, 12, 500), n_pred=5):
    g = torch.Generator().manual_seed(seed)
    id_list = list(ids)
    lm = torch.zeros(shape, dtype=torch.long)
    # random blobby regions per id (non-overlapping by construction: assign by
    # random voxel draw, later ids overwrite earlier — still a valid label map)
    for i in id_list:
        m = torch.rand(shape, generator=g) < 0.15
        lm[m] = i
    mask_ids = torch.tensor(id_list, dtype=torch.long)
    labels = torch.arange(len(id_list)) % 2  # two classes
    pred = torch.rand((n_pred, *shape), generator=g) < 0.2
    pred_classes = torch.arange(n_pred) % 2
    target = {
        "label_map": lm,
        "mask_ids": mask_ids,
        "labels": labels,
        "boxes": torch.zeros(len(id_list), 6, dtype=torch.float32),
    }
    return pred, pred_classes, target


def _run_and_capture(evaluator, sample, target):
    """Drive process(); spy on `add_image_class` because `_stream` stores fp16 copies."""
    metric = next(
        m for m in evaluator.metrics.values() if isinstance(m, MaskMAPMetric)
    )
    captured = {}
    orig = metric.add_image_class

    def spy(image_id, class_id, scores, ious, n_gt):
        captured[int(class_id)] = ious.detach().clone()
        return orig(
            image_id=image_id, class_id=class_id, scores=scores, ious=ious, n_gt=n_gt
        )

    metric.add_image_class = spy
    evaluator.process(
        {"metainfo": {"targets": [target]}}, [sample]
    )
    return captured


@pytest.mark.parametrize("match_labels", [True, False])
@pytest.mark.parametrize("chunk", [1, 8])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_label_map_source_matches_materialized_reference(match_labels, chunk, seed):
    """label_map source: the per-class IoU matrix the evaluator pushes equals the
    materialize-every-GT-mask reference exactly, for any chunk size and both
    class-aware and class-agnostic matching."""
    pred, pred_classes, target = _random_case(seed)
    shape = target["label_map"].shape
    sample = _sample(pred, pred_classes if match_labels else torch.full_like(pred_classes, -1), shape)
    ev = _make_evaluator("label_map", match_labels, chunk)
    captured = _run_and_capture(ev, sample, target)

    classes = (
        sorted(set(sample["topk_class_ids"].tolist()) | set(target["labels"].tolist()))
        if match_labels else [-1]
    )
    for c in classes:
        pm = (sample["topk_class_ids"] == c) if match_labels else torch.ones(
            pred.shape[0], dtype=torch.bool
        )
        gm = (target["labels"] == c) if match_labels else torch.ones(
            target["labels"].numel(), dtype=torch.bool
        )
        if not pm.any() or not gm.any():
            continue
        ref = _reference_ious(pred[pm], target, gm, shape, "label_map")
        assert torch.equal(captured[c], ref), f"class {c}: bincount != reference"


def test_label_map_absent_id_gives_zero_column():
    """An id in mask_ids with ZERO voxels in the label map (e.g. erased by
    resize) must produce an all-zero IoU column, exactly like an all-False
    materialized mask."""
    shape = (4, 8, 8)
    lm = torch.zeros(shape, dtype=torch.long)
    lm[0, 0, 0] = 3
    target = {
        "label_map": lm,
        "mask_ids": torch.tensor([3, 99]),  # 99 absent
        "labels": torch.tensor([0, 0]),
        "boxes": torch.zeros(2, 6),
    }
    pred = torch.zeros((1, *shape), dtype=torch.bool)
    pred[0, 0, 0, 0] = True
    sample = _sample(pred, torch.tensor([0]), shape)
    ev = _make_evaluator("label_map", True, 8)
    captured = _run_and_capture(ev, sample, target)
    ref = _reference_ious(pred, target, torch.ones(2, dtype=torch.bool), shape, "label_map")
    assert torch.equal(captured[0], ref)
    assert captured[0][0, 0] == pytest.approx(1.0)
    assert captured[0][0, 1] == 0.0


def test_label_map_resize_matches_materialized_reference():
    """eval_frame_size != label_map size: resize_label_map-then-compare must
    equal resize-each-binary-mask (same nearest source-voxel selection)."""
    shape = (6, 12, 12)
    eval_size = (4, 8, 8)
    g = torch.Generator().manual_seed(7)
    lm = (torch.rand(shape, generator=g) * 4).long()  # ids 0..3
    mask_ids = torch.tensor([1, 2, 3])
    target = {
        "label_map": lm,
        "mask_ids": mask_ids,
        "labels": torch.zeros(3, dtype=torch.long),
        "boxes": torch.zeros(3, 6),
    }
    pred = torch.rand((3, *eval_size), generator=g) < 0.3
    sample = _sample(pred, torch.zeros(3, dtype=torch.long), eval_size)
    ev = _make_evaluator("label_map", True, 2)
    captured = _run_and_capture(ev, sample, target)
    ref = _reference_ious(pred, target, torch.ones(3, dtype=torch.bool), eval_size, "label_map")
    assert torch.equal(captured[0], ref)


@pytest.mark.parametrize("chunk", [1, 8])
def test_masks_source_matches_materialized_reference(chunk):
    """masks source: GT chunks cached once and reused across prediction chunks
    yield the same IoU matrix as the materialize-per-pair reference."""
    shape = (8, 16, 16)
    g = torch.Generator().manual_seed(11)
    n_gt = 5
    gt_masks = torch.rand((n_gt, *shape), generator=g) < 0.15
    target = {
        "masks": gt_masks,
        "mask_ids": torch.arange(1, n_gt + 1),
        "labels": torch.zeros(n_gt, dtype=torch.long),
        "boxes": torch.zeros(n_gt, 6),
    }
    pred = torch.rand((4, *shape), generator=g) < 0.2
    sample = _sample(pred, torch.zeros(4, dtype=torch.long), shape)
    ev = _make_evaluator("masks", True, chunk)
    captured = _run_and_capture(ev, sample, target)
    ref = _reference_ious(pred, target, torch.ones(n_gt, dtype=torch.bool), shape, "masks")
    assert torch.equal(captured[0], ref)
