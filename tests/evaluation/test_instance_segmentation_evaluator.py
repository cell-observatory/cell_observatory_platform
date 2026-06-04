import math

import pytest
import torch
from omegaconf import DictConfig

from cell_observatory_platform.evaluation.instance_segmentation_evaluator import (
    InstanceSegmentationEvaluator,
    _pairwise_mask_iou_3d_bool,
)
from cell_observatory_platform.evaluation.metrics import MaskMAPMetric, MaskMIoUMetric


def _make_target():
    label_map = torch.zeros(2, 2, 2, dtype=torch.long)
    label_map[0, 0, 0] = 7
    label_map[1, 1, 1] = 9
    return {
        "label_map": label_map,
        "mask_ids": torch.tensor([7, 9], dtype=torch.long),
        "labels": torch.tensor([1, 2], dtype=torch.long),
        "boxes": torch.zeros(2, 6, dtype=torch.float32),
    }


def _make_fake_predict_for_eval_output():
    pixel_decoder_output = torch.full((2, 2, 2, 2), -5.0)
    pixel_decoder_output[0, 0, 0, 0] = 5.0
    pixel_decoder_output[1, 1, 1, 1] = 5.0
    return {
        "mask_embeddings": torch.eye(2, dtype=torch.float32),
        "pixel_decoder_output": pixel_decoder_output,
        "topk_query_indices": torch.tensor([0, 1], dtype=torch.long),
        "topk_class_scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
        "topk_class_ids": torch.tensor([1, 2], dtype=torch.long),
        "boxes": torch.zeros(2, 6, dtype=torch.float32),
        "orig_image_size": (2, 2, 2),
    }


def _make_data_sample(target=None, nested=True):
    target = _make_target() if target is None else target
    targets = [[target]] if nested else [target]
    return {"metainfo": {"targets": targets}}


def test_metric_factory_string_and_errors():
    evaluator = InstanceSegmentationEvaluator(metrics=["mask_map", "mask_miou"])

    assert isinstance(evaluator.metrics["mask_map"], MaskMAPMetric)
    assert isinstance(evaluator.metrics["mask_miou"], MaskMIoUMetric)

    with pytest.raises(ValueError, match="unknown metric name"):
        InstanceSegmentationEvaluator(metrics=["not_a_metric"])

    with pytest.raises(TypeError, match="metric spec"):
        InstanceSegmentationEvaluator(metrics=[123])

    with pytest.raises(ValueError, match="name"):
        InstanceSegmentationEvaluator(metrics=[DictConfig({"_target_": "builtins.dict"})])


def test_extract_targets_flat_vs_nested():
    evaluator = InstanceSegmentationEvaluator(metrics=["mask_map"])
    target0 = _make_target()
    target1 = _make_target()

    flat = evaluator._extract_targets({"metainfo": {"targets": [target0, target1]}})
    nested = evaluator._extract_targets({"metainfo": {"targets": [[target0, target1]]}})

    assert flat == [target0, target1]
    assert nested == [target0, target1]


def test_gt_masks_for_class_label_map_unpack_is_per_instance():
    evaluator = InstanceSegmentationEvaluator(metrics=["mask_map"], gt_mask_source="label_map")
    target = _make_target()

    masks = evaluator._gt_masks_for_class(
        target=target,
        gt_class_mask=torch.tensor([True, True]),
        target_size=(2, 2, 2),
    )

    assert masks.shape == (2, 2, 2, 2)
    assert masks.dtype == torch.bool
    torch.testing.assert_close(masks[0], target["label_map"] == 7)
    torch.testing.assert_close(masks[1], target["label_map"] == 9)


def test_gt_masks_for_class_masks_branch():
    masks_in = torch.zeros(3, 2, 2, 2, dtype=torch.bool)
    masks_in[0, 0, 0, 0] = True
    masks_in[2, 1, 1, 1] = True
    target = {"masks": masks_in}
    evaluator = InstanceSegmentationEvaluator(metrics=["mask_map"], gt_mask_source="masks")

    masks = evaluator._gt_masks_for_class(
        target=target,
        gt_class_mask=torch.tensor([True, False, True]),
        target_size=(2, 2, 2),
    )

    assert masks.shape == (2, 2, 2, 2)
    torch.testing.assert_close(masks, masks_in[[0, 2]])


def test_maybe_resize_gt_masks_nearest():
    masks = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    masks[0, 0, 0, 0] = True

    resized = InstanceSegmentationEvaluator._maybe_resize_gt_masks(masks, target_size=(4, 4, 4))

    assert resized.shape == (1, 4, 4, 4)
    assert resized.dtype == torch.bool
    assert resized.any()


def test_pairwise_mask_iou_3d_bool_manual_cases():
    a = torch.zeros(3, 2, 2, 2, dtype=torch.bool)
    b = torch.zeros(3, 2, 2, 2, dtype=torch.bool)
    a[0, 0, 0, 0] = True
    b[0, 0, 0, 0] = True
    a[1, 0, 0, 0] = True
    b[1, 1, 1, 1] = True
    a[2, 0, 0, 0] = True
    a[2, 0, 0, 1] = True
    b[2, 0, 0, 1] = True
    b[2, 1, 1, 1] = True

    ious = _pairwise_mask_iou_3d_bool(a, b)

    assert ious[0, 0].item() == pytest.approx(1.0)
    assert ious[1, 1].item() == pytest.approx(0.0)
    assert ious[2, 2].item() == pytest.approx(1.0 / 3.0)


def test_gt_boxes_to_abs_xyzxyz_matches_pred_postprocess():
    evaluator = InstanceSegmentationEvaluator(
        metrics=["box_map"], gt_box_format="cxcyczwhd", gt_boxes_normalized=True
    )
    gt = torch.tensor([[0.5, 0.5, 0.5, 1.0, 1.0, 1.0]], dtype=torch.float32)
    out = evaluator._gt_boxes_to_abs_xyzxyz(gt, orig_image_size=(4, 8, 16))  # (D,H,W)
    torch.testing.assert_close(
        out, torch.tensor([[0.0, 0.0, 0.0, 16.0, 8.0, 4.0]])
    )


def test_gt_boxes_to_abs_xyzxyz_absolute_xyzxyz_is_identity():
    evaluator = InstanceSegmentationEvaluator(
        metrics=["box_map"], gt_box_format="xyzxyz", gt_boxes_normalized=False
    )
    gt = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32)
    out = evaluator._gt_boxes_to_abs_xyzxyz(gt, orig_image_size=(10, 10, 10))
    torch.testing.assert_close(out, gt)


def test_greedy_match_per_class_score_sorted():
    scores = torch.tensor([0.9, 0.8, 0.7])
    ious = torch.tensor(
        [
            [0.6, 0.1],
            [0.7, 0.8],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    matched = InstanceSegmentationEvaluator._greedy_match_per_class(scores, ious)

    assert matched == pytest.approx([0.6, 0.8])


def test_process_rejects_non_list_outputs_and_len_mismatch():
    evaluator = InstanceSegmentationEvaluator(metrics=["mask_map"])
    data_sample = _make_data_sample()

    with pytest.raises(TypeError, match="predict_for_eval-style"):
        evaluator.process(data_sample, outputs={"not": "a list"})

    with pytest.raises(RuntimeError, match="batch size mismatch"):
        evaluator.process(data_sample, outputs=[])


def test_process_fake_outputs_streams_mask_map_and_miou():
    evaluator = InstanceSegmentationEvaluator(
        metrics=["mask_map", "mask_miou"],
        mask_chunk_size=1,
        match_labels=True,
    )

    evaluator.process(_make_data_sample(), outputs=[_make_fake_predict_for_eval_output()])
    results = evaluator.evaluate()

    assert math.isfinite(results["mask_map"])
    assert math.isfinite(results["mask_miou"])
    assert results["mask_map"] == pytest.approx(1.0)
    assert results["mask_miou"] == pytest.approx(1.0)


def test_process_match_labels_false_uses_class_agnostic_bucket_once():
    evaluator = InstanceSegmentationEvaluator(
        metrics=["mask_map"],
        mask_chunk_size=1,
        match_labels=False,
    )

    evaluator.process(_make_data_sample(), outputs=[_make_fake_predict_for_eval_output()])
    stream = evaluator.metrics["mask_map"]._stream

    assert len(stream) == 1
    assert stream[0]["class_id"] == -1
    assert stream[0]["ious"].shape == (2, 2)
    assert stream[0]["n_gt"] == 2


def test_reset_clears_metrics_results_and_image_counter():
    evaluator = InstanceSegmentationEvaluator(
        metrics=["mask_map", "mask_miou"],
        mask_chunk_size=1,
    )
    evaluator.process(_make_data_sample(), outputs=[_make_fake_predict_for_eval_output()])

    assert evaluator._image_id_counter == 1
    assert evaluator.metrics["mask_map"]._stream

    evaluator.reset()

    assert evaluator._image_id_counter == 0
    assert evaluator.metrics["mask_map"]._stream == []
    assert all(value is None for value in evaluator._results.values())
