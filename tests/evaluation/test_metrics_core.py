import math

import numpy as np
import pytest
import torch

from cell_observatory_platform.evaluation.metrics import (
    BoxF1Metric,
    BoxMAPMetric,
    BoxMIoUMetric,
    ClassAPMetric,
    MAEMetric,
    MaskMAPMetric,
    MaskMIoUMetric,
    NRMSEMetric,
    ReduceBuffer,
    TrainLosses,
)


def _ap_101_point(recall: np.ndarray, precision: np.ndarray) -> float:
    precision = precision.copy()
    for i in range(len(precision) - 1, 0, -1):
        if precision[i] > precision[i - 1]:
            precision[i - 1] = precision[i]
    thresholds = np.linspace(0.0, 1.0, 101)
    inds = np.searchsorted(recall, thresholds, side="left")
    interp = np.zeros_like(thresholds)
    valid = inds < len(precision)
    interp[valid] = precision[inds[valid]]
    return float(interp.mean())


def _binary_ap_expected(scores: torch.Tensor, targets_bool: torch.Tensor) -> float:
    order = torch.argsort(scores, descending=True)
    sorted_targets = targets_bool[order].float()
    tp = torch.cumsum(sorted_targets, dim=0)
    fp = torch.cumsum(1.0 - sorted_targets, dim=0)
    recall = (tp / sorted_targets.sum().clamp_min(1.0)).numpy()
    precision = (tp / (tp + fp).clamp_min(1e-12)).numpy()
    return _ap_101_point(recall, precision)


def _det(boxes, labels, scores=None):
    out = {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 6),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if scores is not None:
        out["scores"] = torch.tensor(scores, dtype=torch.float32)
    return out


def _mask_det(masks, labels, scores=None):
    out = {
        "masks": masks.bool(),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if scores is not None:
        out["scores"] = torch.tensor(scores, dtype=torch.float32)
    return out


@pytest.mark.parametrize(
    "method,expected",
    [("mean", 2.0), ("min", 1.0), ("max", 3.0)],
)
def test_reduce_buffer_reductions_and_reset(method, expected):
    buf = ReduceBuffer(reduce_method=method)
    for value in [1.0, torch.tensor(3.0), 2.0]:
        buf.add(value)

    assert buf.aggregate() == pytest.approx(expected)

    buf.reset()
    assert buf.values == []


def test_reduce_buffer_errors():
    with pytest.raises(AssertionError, match="No values"):
        ReduceBuffer().aggregate()

    buf = ReduceBuffer(reduce_method="bad")
    buf.add(1.0)
    with pytest.raises(ValueError, match="Unknown reduce method"):
        buf.aggregate()


@pytest.mark.parametrize(
    "method,expected",
    [("mean", 2.0), ("min", 1.0), ("max", 3.0)],
)
def test_train_losses_reductions_and_reset(method, expected):
    metric = TrainLosses(reduce_method=method)
    metric(None, None, torch.tensor(1.0))
    metric(None, None, torch.tensor(3.0))

    assert metric.aggregate() == pytest.approx(expected)

    metric.reset()
    assert metric.loss_values == []


def test_mae_metric_buffers_per_call_values():
    metric = MAEMetric(reduce_method="mean")
    metric(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))
    metric(torch.tensor([2.0, 2.0]), torch.tensor([1.0, 1.0]))

    assert metric.aggregate() == pytest.approx((1.5 + 1.0) / 2)


def test_nrmse_metric_expected_value_and_constant_target_eps():
    metric = NRMSEMetric(reduce_method="mean")
    metric(torch.tensor([0.0, 2.0]), torch.tensor([0.0, 4.0]))

    assert metric.aggregate() == pytest.approx(math.sqrt(2.0) / 4.0)

    constant_target = NRMSEMetric(reduce_method="mean", eps=1e-4)
    constant_target(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]))
    assert math.isfinite(constant_target.aggregate())
    assert constant_target.aggregate() == pytest.approx(0.0)


def test_box_map_perfect_match_and_no_predictions():
    target = _det([[0, 0, 0, 1, 1, 1]], labels=[1])
    pred = _det([[0, 0, 0, 1, 1, 1]], labels=[1], scores=[0.9])
    metric = BoxMAPMetric(iou_thresholds=[0.5])

    metric([pred], [target])

    assert metric.aggregate() == pytest.approx(1.0)

    no_pred = BoxMAPMetric(iou_thresholds=[0.5])
    no_pred([_det([], labels=[], scores=[])], [target])
    assert no_pred.aggregate() == pytest.approx(0.0)

    absent_class = BoxMAPMetric(iou_thresholds=[0.5], class_ids=[99])
    absent_class([pred], [target])
    assert absent_class.aggregate() == pytest.approx(0.0)


def test_box_miou_greedy_matching_and_label_gate():
    target = _det([[0, 0, 0, 1, 1, 1]], labels=[1])
    pred = _det(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ],
        labels=[1, 1],
        scores=[0.9, 0.8],
    )
    metric = BoxMIoUMetric(iou_threshold=0.5)

    metric([pred], [target])

    assert metric.aggregate() == pytest.approx(1.0)

    mismatch = BoxMIoUMetric(iou_threshold=0.5, match_labels=True)
    mismatch([_det([[0, 0, 0, 1, 1, 1]], labels=[2], scores=[0.9])], [target])
    assert mismatch.aggregate() == pytest.approx(0.0)


def test_box_f1_counts_tp_fp_fn_and_reset():
    pred = _det(
        [
            [0, 0, 0, 1, 1, 1],
            [4, 4, 4, 5, 5, 5],
        ],
        labels=[1, 1],
        scores=[0.9, 0.8],
    )
    target = _det(
        [
            [0, 0, 0, 1, 1, 1],
            [2, 2, 2, 3, 3, 3],
        ],
        labels=[1, 1],
    )
    metric = BoxF1Metric(iou_threshold=0.5, score_threshold=0.0, match_labels=True)

    metric([pred], [target])

    assert metric.aggregate() == pytest.approx(0.5)

    metric.reset()
    assert metric._tp == 0
    assert metric._fp == 0
    assert metric._fn == 0


def test_class_ap_binary_and_reset():
    scores = torch.tensor([0.9, 0.8, 0.1])
    targets = torch.tensor([1, 0, 1])
    metric = ClassAPMetric()

    metric(scores, targets)

    assert metric.aggregate() == pytest.approx(_binary_ap_expected(scores, targets.bool()))

    metric.reset()
    assert metric._scores == []
    assert metric._targets == []
    assert metric.aggregate() == pytest.approx(0.0)


def test_class_ap_multiclass_skips_absent_classes():
    scores = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.1, 0.9, 0.0],
        ]
    )
    targets = torch.tensor([0, 0, 1])
    metric = ClassAPMetric(num_classes=3)

    metric(scores, targets)

    expected = (
        _binary_ap_expected(scores[:, 0], targets == 0)
        + _binary_ap_expected(scores[:, 1], targets == 1)
    ) / 2
    assert metric.aggregate() == pytest.approx(expected)


def test_mask_map_batched_perfect_match():
    mask = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    metric = MaskMAPMetric(iou_thresholds=[0.5])

    metric(
        [_mask_det(mask, labels=[1], scores=[0.9])],
        [_mask_det(mask, labels=[1])],
    )

    assert metric.aggregate() == pytest.approx(1.0)


def test_mask_miou_semantic_and_ignore_index():
    pred = torch.tensor(
        [
            [[0, 1], [1, 1]],
            [[0, 0], [1, 0]],
        ]
    )
    target = torch.tensor(
        [
            [[0, 1], [0, 1]],
            [[0, 0], [1, 1]],
        ]
    )
    metric = MaskMIoUMetric(mode="semantic")

    metric(pred, target)

    # Per-class IoU: for each c, intersection = (pred==c) & (gt==c), union = (pred==c) | (gt==c).
    # Class 0 and class 1 both have IoU 3/5 here; mean = 0.6.
    assert metric.aggregate() == pytest.approx(0.6)

    foreground = MaskMIoUMetric(mode="semantic", ignore_index=0)
    foreground(pred, target)
    assert foreground.aggregate() == pytest.approx(3 / 5)


def test_mask_miou_instance_batched_and_label_gate():
    mask = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    metric = MaskMIoUMetric(mode="instance", iou_threshold=0.5)

    metric(
        [_mask_det(mask, labels=[1], scores=[0.9])],
        [_mask_det(mask, labels=[1])],
    )

    assert metric.aggregate() == pytest.approx(1.0)

    mismatch = MaskMIoUMetric(mode="instance", iou_threshold=0.5, match_labels=True)
    mismatch(
        [_mask_det(mask, labels=[2], scores=[0.9])],
        [_mask_det(mask, labels=[1])],
    )
    assert mismatch.aggregate() == pytest.approx(0.0)


def test_mask_miou_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        MaskMIoUMetric(mode="bad")
