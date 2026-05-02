import math

import pytest
import torch

from cell_observatory_platform.evaluation.metrics import MaskMAPMetric, MaskMIoUMetric


def test_mask_map_stream_single_image_perfect_ap():
    metric = MaskMAPMetric(iou_thresholds=[0.5])
    metric.add_image_class(
        image_id=0,
        class_id=1,
        scores=torch.tensor([1.0]),
        ious=torch.tensor([[1.0]]),
        n_gt=1,
    )

    assert metric.aggregate() == pytest.approx(1.0)


def test_mask_map_stream_image_ids_do_not_share_gt():
    metric = MaskMAPMetric(iou_thresholds=[0.5])
    # Image 0 has a high-scoring false positive.
    metric.add_image_class(
        image_id=0,
        class_id=1,
        scores=torch.tensor([0.9]),
        ious=torch.tensor([[0.0]]),
        n_gt=1,
    )
    # Image 1 has a lower-scoring true positive. Matching must remain scoped
    # to image 1's GT, not image 0's already-seen GT bookkeeping.
    metric.add_image_class(
        image_id=1,
        class_id=1,
        scores=torch.tensor([0.8]),
        ious=torch.tensor([[1.0]]),
        n_gt=1,
    )

    # Score order: FP then TP -> recall [0, 0.5], precision envelope [0.5, 0.5].
    # COCO 101-pt interpolation contributes 0.5 at thresholds 0.00..0.50.
    expected = 0.5 * (51 / 101)
    assert metric.aggregate() == pytest.approx(expected)


def test_mask_map_stream_no_predictions_counts_gt_denominator():
    metric = MaskMAPMetric(iou_thresholds=[0.5])
    metric.add_image_class(
        image_id=0,
        class_id=1,
        scores=torch.empty(0),
        ious=torch.zeros(0, 2),
        n_gt=2,
    )

    value = metric.aggregate()

    assert math.isfinite(value)
    assert value == pytest.approx(0.0)


def test_mask_map_stream_max_detections_caps_to_top_scores():
    metric = MaskMAPMetric(iou_thresholds=[0.5], max_detections=1)
    metric.add_image_class(
        image_id=0,
        class_id=1,
        scores=torch.tensor([0.9, 0.8]),
        ious=torch.tensor([[0.0], [1.0]]),
        n_gt=1,
    )

    # Only the high-scoring false positive survives the max_detections cap.
    assert metric.aggregate() == pytest.approx(0.0)


def test_mask_map_reset_clears_stream_and_batched_state():
    metric = MaskMAPMetric(iou_thresholds=[0.5])
    metric.add_image_class(
        image_id=0,
        class_id=1,
        scores=torch.tensor([1.0]),
        ious=torch.tensor([[1.0]]),
        n_gt=1,
    )
    metric(
        [{"masks": torch.ones(1, 2, 2, 2, dtype=torch.bool), "labels": torch.tensor([1]), "scores": torch.tensor([1.0])}],
        [{"masks": torch.ones(1, 2, 2, 2, dtype=torch.bool), "labels": torch.tensor([1])}],
    )

    assert metric._stream
    assert metric._preds
    assert metric._targets

    metric.reset()

    assert metric._stream == []
    assert metric._preds == []
    assert metric._targets == []


def test_mask_miou_add_matched_ious_threshold_and_semantic_guard():
    metric = MaskMIoUMetric(mode="instance", iou_threshold=0.5)
    metric.add_matched_ious([0.49, 0.5, 0.9])

    assert metric.aggregate() == pytest.approx((0.5 + 0.9) / 2)

    semantic_metric = MaskMIoUMetric(mode="semantic")
    with pytest.raises(RuntimeError, match="mode='instance'"):
        semantic_metric.add_matched_ious([1.0])


def test_mask_map_batched_path_smoke():
    metric = MaskMAPMetric(iou_thresholds=[0.5])
    masks = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    masks[:, 0, 0, 0] = True
    pred = {"masks": masks.clone(), "labels": torch.tensor([1]), "scores": torch.tensor([1.0])}
    target = {"masks": masks.clone(), "labels": torch.tensor([1])}

    metric([pred], [target])

    assert math.isfinite(metric.aggregate())
    assert metric.aggregate() == pytest.approx(1.0)


def test_mask_map_stream_shape_validation():
    metric = MaskMAPMetric(iou_thresholds=[0.5])

    with pytest.raises(ValueError, match="disagrees with n_gt"):
        metric.add_image_class(
            image_id=0,
            class_id=1,
            scores=torch.tensor([1.0]),
            ious=torch.zeros(1, 2),
            n_gt=1,
        )


def test_mask_map_streaming_precedence_over_batched_state():
    metric = MaskMAPMetric(iou_thresholds=[0.5])
    mask = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    mask[:, 0, 0, 0] = True
    # Batched path is a false positive.
    metric(
        [{"masks": mask.clone(), "labels": torch.tensor([1]), "scores": torch.tensor([1.0])}],
        [{"masks": torch.zeros_like(mask), "labels": torch.tensor([1])}],
    )
    # Streaming path is perfect and should take precedence.
    metric.add_image_class(
        image_id=0,
        class_id=1,
        scores=torch.tensor([1.0]),
        ious=torch.tensor([[1.0]]),
        n_gt=1,
    )

    assert metric.aggregate() == pytest.approx(1.0)
