import logging
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
        "mask_source": "query",
        "mask_embeddings": torch.eye(2, dtype=torch.float32),
        "pixel_decoder_output": pixel_decoder_output,
        "topk_query_indices": torch.tensor([0, 1], dtype=torch.long),
        "topk_class_scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
        "topk_class_ids": torch.tensor([1, 2], dtype=torch.long),
        "boxes": torch.zeros(2, 6, dtype=torch.float32),
        "eval_frame_size": (2, 2, 2),
    }


def _make_data_sample(target=None):
    # Form S (see data/data_types.py): a plain per-sample list — no wrap exists anymore.
    target = _make_target() if target is None else target
    return {"metainfo": {"targets": [target]}}


def test_metric_factory_string_and_errors():
    evaluator = InstanceSegmentationEvaluator(
        metrics=["mask_map", {"name": "mask_miou", "mode": "instance"}]
    )

    assert isinstance(evaluator.metrics["mask_map"], MaskMAPMetric)
    assert isinstance(evaluator.metrics["mask_miou"], MaskMIoUMetric)
    assert evaluator.metrics["mask_miou"].mode == "instance"

    # build_metrics looks the name up in METRICS -> unknown names KeyError.
    with pytest.raises(KeyError):
        InstanceSegmentationEvaluator(metrics=["not_a_metric"])

    # A dict spec without "name" cannot be routed to a metric.
    with pytest.raises(KeyError):
        InstanceSegmentationEvaluator(metrics=[DictConfig({"_target_": "builtins.dict"})])


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


def test_match_per_class_hungarian_optimal():
    # Hungarian on the (thresholded) IoU matrix: optimal assignment is
    # pred0->gt0 (0.6) + pred1->gt1 (0.8) = 1.4 (vs 0.1 + 0.7 = 0.8 swapped);
    # pred2 stays unmatched (only two GTs).
    ious = torch.tensor(
        [
            [0.6, 0.1],
            [0.7, 0.8],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    matched = InstanceSegmentationEvaluator._match_per_class(ious, iou_threshold=0.5)

    assert matched == pytest.approx([0.6, 0.8])


def test_process_rejects_non_list_outputs_and_len_mismatch():
    evaluator = InstanceSegmentationEvaluator(metrics=["mask_map"])
    data_sample = _make_data_sample()

    with pytest.raises(TypeError, match="evaluate_step"):
        evaluator.process(data_sample, outputs={"not": "a list"})

    with pytest.raises(RuntimeError, match="batch size mismatch"):
        evaluator.process(data_sample, outputs=[])


def test_process_fake_outputs_streams_mask_map_and_miou():
    evaluator = InstanceSegmentationEvaluator(
        metrics=["mask_map", {"name": "mask_miou", "mode": "instance"}],
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


def test_process_class_agnostic_sentinel_with_match_labels_true_raises():
    evaluator = InstanceSegmentationEvaluator(metrics=["mask_map"], match_labels=True)
    sample = _make_fake_predict_for_eval_output()
    sample["topk_class_ids"] = torch.full((2,), -1, dtype=torch.long)

    with pytest.raises(ValueError, match="class-agnostic"):
        evaluator.process(_make_data_sample(), outputs=[sample])


def test_process_class_agnostic_sentinel_with_match_labels_false_ok():
    evaluator = InstanceSegmentationEvaluator(
        metrics=["mask_map"], match_labels=False, mask_chunk_size=1
    )
    sample = _make_fake_predict_for_eval_output()
    sample["topk_class_ids"] = torch.full((2,), -1, dtype=torch.long)

    evaluator.process(_make_data_sample(), outputs=[sample])  # must not raise
    assert math.isfinite(evaluator.evaluate()["mask_map"])


def test_reset_clears_metrics_results_and_image_counter():
    evaluator = InstanceSegmentationEvaluator(
        metrics=["mask_map", {"name": "mask_miou", "mode": "instance"}],
        mask_chunk_size=1,
    )
    evaluator.process(_make_data_sample(), outputs=[_make_fake_predict_for_eval_output()])

    assert evaluator._image_id_counter == 1
    assert evaluator.metrics["mask_map"]._stream

    evaluator.reset()

    assert evaluator._image_id_counter == 0
    assert evaluator.metrics["mask_map"]._stream == []
    assert all(value is None for value in evaluator._results.values())


def test_label_map_id_zero_raises():
    """label_map contract: instance ids are >= 1, 0 is reserved for background."""
    evaluator = InstanceSegmentationEvaluator(metrics=["mask_map"], match_labels=True)
    target = _make_target()
    target["mask_ids"] = torch.tensor([0, 9], dtype=torch.long)
    with pytest.raises(ValueError, match="ids must be >= 1"):
        evaluator.process(_make_data_sample(target), outputs=[_make_fake_predict_for_eval_output()])


def test_instance_miou_reports_match_recall_for_missed_gt():
    """One perfect match + one completely missed GT: the matched mean stays
    1.0 and match_recall exposes the miss. Driven through the STREAMING00
    evaluator so the always-push (even for match-less images) path is covered."""
    # Image: two GT instances; the model only predicts one of them.
    label_map = torch.zeros(2, 2, 2, dtype=torch.long)
    label_map[0, 0, 0] = 7
    label_map[1, 1, 1] = 9
    target = {
        "label_map": label_map,
        "mask_ids": torch.tensor([7, 9], dtype=torch.long),
        "labels": torch.tensor([1, 1], dtype=torch.long),
        "boxes": torch.zeros(2, 6, dtype=torch.float32),
    }
    pred_masks = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    pred_masks[0, 0, 0, 0] = True  # exact match for instance 7 only
    sample = {
        "mask_source": "direct",
        "pred_masks": pred_masks,
        "topk_query_indices": torch.tensor([0], dtype=torch.long),
        "topk_class_scores": torch.tensor([0.9], dtype=torch.float32),
        "topk_class_ids": torch.tensor([1], dtype=torch.long),
        "boxes": torch.zeros(1, 6, dtype=torch.float32),
        "eval_frame_size": (2, 2, 2),
    }

    evaluator = InstanceSegmentationEvaluator(
        metrics=[{"name": "mask_miou", "mode": "instance"}],
        mask_chunk_size=1,
        match_labels=True,
    )
    evaluator.process({"metainfo": {"targets": [target]}}, outputs=[sample])
    results = evaluator.evaluate()

    assert results["mask_miou"] == pytest.approx(1.0)       # matched mean kept
    assert results["mask_match_recall"] == pytest.approx(0.5)  # 1 of 2 GTs


# ---------------------------------------------------------------------------
# GT-axis chunking equivalence
# ---------------------------------------------------------------------------


def _three_gt_target():
    label_map = torch.zeros(3, 3, 3, dtype=torch.long)
    label_map[0, 0, :2] = 5
    label_map[1, 1, :3] = 6
    label_map[2, 2, 2] = 7
    return {
        "label_map": label_map,
        "mask_ids": torch.tensor([5, 6, 7], dtype=torch.long),
        "labels": torch.tensor([1, 1, 1], dtype=torch.long),
        "boxes": torch.zeros(3, 6, dtype=torch.float32),
    }


def _three_pred_sample(label_map):
    pred_masks = torch.stack([
        label_map == 5,                                   # exact instance 5
        (label_map == 6) | (label_map == 7),              # merges 6 and 7
        torch.zeros_like(label_map, dtype=torch.bool),    # empty prediction
    ])
    pred_masks[2, 0, 0, 0] = True                         # partial overlap w/ 5
    return {
        "mask_source": "direct",
        "pred_masks": pred_masks,
        "topk_query_indices": torch.arange(3, dtype=torch.long),
        "topk_class_scores": torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
        "topk_class_ids": torch.ones(3, dtype=torch.long),
        "boxes": torch.zeros(3, 6, dtype=torch.float32),
        "eval_frame_size": (3, 3, 3),
    }


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_mask_chunk_size_does_not_change_iou_matrix_or_metrics(chunk_size):
    """The IoU matrix pushed to MaskMAP and the aggregated metrics are
    independent of `mask_chunk_size`."""
    def run(cs):
        evaluator = InstanceSegmentationEvaluator(
            metrics=["mask_map", {"name": "mask_miou", "mode": "instance"}],
            mask_chunk_size=cs,
            match_labels=True,
        )
        target = _three_gt_target()
        evaluator.process(
            {"metainfo": {"targets": [target]}},
            outputs=[_three_pred_sample(target["label_map"])],
        )
        return evaluator

    chunked = run(chunk_size)
    unchunked = run(64)  # one chunk covers everything on both axes

    stream_c = chunked.metrics["mask_map"]._stream
    stream_u = unchunked.metrics["mask_map"]._stream
    assert len(stream_c) == len(stream_u) == 1
    torch.testing.assert_close(stream_c[0]["ious"], stream_u[0]["ious"])
    assert stream_c[0]["n_gt"] == stream_u[0]["n_gt"] == 3

    res_c, res_u = chunked.evaluate(), unchunked.evaluate()
    assert res_c["mask_map"] == pytest.approx(res_u["mask_map"], abs=1e-9)
    for key in ("mask_miou", "mask_match_recall"):
        if math.isnan(res_u[key]):
            assert math.isnan(res_c[key])
        else:
            assert res_c[key] == pytest.approx(res_u[key], abs=1e-9)


# ---------------------------------------------------------------------------
# Build-time validation of metric specs
# ---------------------------------------------------------------------------


class TestConflictingInstanceThresholds:
    """Matches are computed once at a single IoU threshold and shared by every
    instance-mode mIoU metric, so differing thresholds are rejected at build."""

    def _make(self, metrics):
        return InstanceSegmentationEvaluator(metrics=metrics)

    def test_two_differing_thresholds_raise(self):
        with pytest.raises(ValueError, match="differing"):
            self._make([
                {"name": "mask_miou", "key": "a", "mode": "instance", "iou_threshold": 0.5},
                {"name": "mask_miou", "key": "b", "mode": "instance", "iou_threshold": 0.25},
            ])

    def test_equal_thresholds_ok(self):
        ev = self._make([
            {"name": "mask_miou", "key": "a", "mode": "instance", "iou_threshold": 0.5},
            {"name": "mask_miou", "key": "b", "mode": "instance", "iou_threshold": 0.5},
        ])
        assert len(ev.metrics) == 2

    def test_semantic_metric_not_counted(self):
        ev = self._make([
            {"name": "mask_miou", "key": "a", "mode": "instance", "iou_threshold": 0.5},
            {"name": "mask_miou", "key": "s", "mode": "semantic", "num_classes": 2,
             "iou_threshold": 0.1},
        ])
        assert len(ev.metrics) == 2


_EV_LOGGER = "cell_observatory_platform.evaluation.instance_segmentation_evaluator"


def test_box_miou_match_labels_diverging_from_evaluator_warns_at_build(caplog):
    """The evaluator's match_labels is not forwarded to box metrics; a
    diverging box_miou setting is surfaced as a build-time warning."""
    with caplog.at_level(logging.WARNING, logger=_EV_LOGGER):
        InstanceSegmentationEvaluator(
            metrics=[{"name": "box_miou", "match_labels": False}],
            match_labels=True,
        )
    assert any("ACROSS classes" in r.message for r in caplog.records)


def test_box_f1_score_threshold_diverging_from_evaluator_warns_at_build(caplog):
    """The evaluator's score_threshold is not forwarded to box metrics; a
    diverging box_f1 gate is surfaced as a build-time warning."""
    with caplog.at_level(logging.WARNING, logger=_EV_LOGGER):
        InstanceSegmentationEvaluator(
            metrics=[{"name": "box_f1", "score_threshold": 0.05}],
            score_threshold=0.3,
        )
    assert any("NOT forwarded" in r.message for r in caplog.records)


def test_default_box_gates_do_not_warn_and_are_not_overwritten(caplog):
    # shipped configs (score_threshold 0.0) must stay warning-free
    with caplog.at_level(logging.WARNING, logger=_EV_LOGGER):
        ev = InstanceSegmentationEvaluator(
            metrics=[{"name": "box_f1"}, {"name": "box_miou", "match_labels": True}],
            score_threshold=0.0,
            match_labels=True,
        )
    assert not caplog.records
    # and the warning path never mutates the metrics' own gates
    assert float(ev.metrics["box_f1"].score_threshold) == 0.05
