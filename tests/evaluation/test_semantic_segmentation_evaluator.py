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
    metainfo = {"targets": [[target]]}
    if semantic_classes is not None:
        metainfo["semantic_classes"] = semantic_classes
    return {"metainfo": metainfo}


def test_perfect_prediction_scores_one():
    target = _two_class_target()
    ev = SemanticSegmentationEvaluator(num_classes=3, ignore_index=0)
    ev.reset()
    ev.process(_data_sample(target, ["boundary", "foreground"]),
               [{"labelmap": _expected_map(target)}])
    assert ev.evaluate()["mask_miou_semantic"] == pytest.approx(1.0)


def test_class_shifted_prediction_scores_zero():
    # The exact off-by-one the broken taxonomy produced: every class shifted by one.
    target = _two_class_target()
    gt = _expected_map(target)
    shifted = torch.where(gt > 0, gt + 1, gt)
    ev = SemanticSegmentationEvaluator(num_classes=4, ignore_index=0)
    ev.reset()
    ev.process(_data_sample(target, ["boundary", "foreground", "phantom"]),
               [{"labelmap": shifted}])
    assert ev.evaluate()["mask_miou_semantic"] == pytest.approx(0.0)


def test_taxonomy_parity_mismatch_raises():
    target = _two_class_target()
    ev = SemanticSegmentationEvaluator(num_classes=9, ignore_index=0)
    ev.reset()
    with pytest.raises(ValueError, match="num_classes"):
        ev.process(_data_sample(target, ["boundary", "foreground"]),
                   [{"labelmap": _expected_map(target)}])


def test_parity_check_skipped_when_taxonomy_absent():
    # Instance datasets do not set semantic_classes; the check must not fire.
    target = _two_class_target()
    ev = SemanticSegmentationEvaluator(num_classes=3, ignore_index=0)
    ev.reset()
    ev.process(_data_sample(target), [{"labelmap": _expected_map(target)}])
    assert ev.evaluate()["mask_miou_semantic"] == pytest.approx(1.0)
