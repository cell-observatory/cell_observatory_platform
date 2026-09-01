import pytest
import torch

from cell_observatory_platform.evaluation.semantic_segmentation_evaluator import (
    SemanticSegmentationEvaluator,
)


def _two_class_target():
    """Two disjoint classes, as build_semantic_targets would produce them."""
    masks = torch.zeros(2, 2, 4, 4, dtype=torch.bool)
    masks[0, 0, 0, :2] = True       # class 0 -> label 1
    masks[1, 0, 1, :2] = True       # class 1 -> label 2
    return {"masks": masks, "labels": torch.tensor([0, 1])}


def _expected_map(target):
    gt = torch.zeros(target["masks"].shape[1:], dtype=torch.long)
    for i in range(target["masks"].shape[0]):
        gt[target["masks"][i]] = int(target["labels"][i]) + 1
    return gt


def _data_sample(target, semantic_classes=None):
    metainfo = {"targets": [target]}
    if semantic_classes is not None:
        metainfo["semantic_classes"] = semantic_classes
    return {"metainfo": metainfo}


def test_perfect_prediction_scores_one():
    target = _two_class_target()
    ev = SemanticSegmentationEvaluator(num_classes=3, exclude_from_mean=0)
    ev.reset()
    ev.process(_data_sample(target, ["boundary", "foreground"]),
               [{"labelmap": _expected_map(target)}])
    assert ev.evaluate()["mask_miou_semantic"] == pytest.approx(1.0)


def test_class_shifted_prediction_scores_zero():
    # The exact off-by-one the broken taxonomy produced: every class shifted by one.
    target = _two_class_target()
    gt = _expected_map(target)
    shifted = torch.where(gt > 0, gt + 1, gt)
    ev = SemanticSegmentationEvaluator(num_classes=4, exclude_from_mean=0)
    ev.reset()
    ev.process(_data_sample(target, ["boundary", "foreground", "phantom"]),
               [{"labelmap": shifted}])
    assert ev.evaluate()["mask_miou_semantic"] == pytest.approx(0.0)


def test_taxonomy_parity_mismatch_raises():
    target = _two_class_target()
    ev = SemanticSegmentationEvaluator(num_classes=9, exclude_from_mean=0)
    ev.reset()
    with pytest.raises(ValueError, match="num_classes"):
        ev.process(_data_sample(target, ["boundary", "foreground"]),
                   [{"labelmap": _expected_map(target)}])


def test_parity_check_skipped_when_taxonomy_absent():
    # Instance datasets do not set semantic_classes; the check must not fire.
    target = _two_class_target()
    ev = SemanticSegmentationEvaluator(num_classes=3, exclude_from_mean=0)
    ev.reset()
    ev.process(_data_sample(target), [{"labelmap": _expected_map(target)}])
    assert ev.evaluate()["mask_miou_semantic"] == pytest.approx(1.0)


def test_evaluator_plumbs_ignore_index_and_exclude_from_mean():
    """SemanticSegmentationEvaluator forwards ignore_index AND exclude_from_mean
    to its metric and still scores a perfect prediction as 1.0."""
    ev = SemanticSegmentationEvaluator(
        num_classes=3, ignore_index=None, exclude_from_mean=0
    )
    metric = ev.metric
    assert metric.exclude_from_mean == 0
    assert metric.ignore_index is None

    target = _two_class_target()
    ev.process(_data_sample(target), [{"labelmap": _expected_map(target)}])
    assert ev.evaluate()["mask_miou_semantic"] == pytest.approx(1.0)


def test_label_map_source_builds_gt_from_instance_ids():
    """gt_mask_source="label_map": instance ids are mapped through `labels` to
    class + 1, so a matching prediction scores 1.0 and a class-swapped one 0.0."""
    label_map = torch.zeros(2, 4, 4, dtype=torch.long)
    label_map[0, 0, :2] = 5            # instance 5 -> class 0 -> semantic label 1
    label_map[0, 1, :2] = 6            # instance 6 -> class 1 -> semantic label 2
    target = {"label_map": label_map, "mask_ids": torch.tensor([5, 6]), "labels": torch.tensor([0, 1])}
    pred = torch.zeros(2, 4, 4, dtype=torch.long)
    pred[0, 0, :2] = 1
    pred[0, 1, :2] = 2

    ev = SemanticSegmentationEvaluator(num_classes=3, exclude_from_mean=0, gt_mask_source="label_map")
    ev.process(_data_sample(target), [{"labelmap": pred}])
    assert ev.evaluate()["mask_miou_semantic"] == pytest.approx(1.0)

    swapped = torch.where(pred == 1, 2, torch.where(pred == 2, 1, pred))
    ev2 = SemanticSegmentationEvaluator(num_classes=3, exclude_from_mean=0, gt_mask_source="label_map")
    ev2.process(_data_sample(target), [{"labelmap": swapped}])
    assert ev2.evaluate()["mask_miou_semantic"] == pytest.approx(0.0)


def test_batch_size_mismatch_raises():
    """More predicted label maps than targets is a RuntimeError, not a silent zip-truncation."""
    target = _two_class_target()
    ev = SemanticSegmentationEvaluator(num_classes=3, exclude_from_mean=0)
    with pytest.raises(RuntimeError, match="batch size mismatch"):
        ev.process(_data_sample(target), [{"labelmap": _expected_map(target)}] * 2)
