import pytest
import torch

from cell_observatory_platform.evaluation.automated_benchmark_evaluator import (
    AutomatedBenchmarkEvaluator,
)


def test_pred_key_selects_output_and_target_role_defaults_to_it():
    """`pred_key` picks the output tensor and, absent `target_role`, also names the
    Form-D target role; other roles/outputs never reach the metrics."""
    evaluator = AutomatedBenchmarkEvaluator(
        metric_reductions=[{"name": "mae", "reduce_method": "mean"}],
        pred_key="x",
        target_key="targets",
    )
    assert evaluator.target_role == "x"

    outputs = {"x": torch.tensor([[1.0, 3.0], [2.0, 2.0]]), "other": torch.full((2, 2), 99.0)}
    # Form-D targets: role-keyed dict; the "other" role must be ignored on both sides.
    data_sample = {"metainfo": {"targets": {"x": torch.zeros(2, 2), "other": torch.zeros(2, 2)}}}
    evaluator.process(data_sample, outputs)

    assert evaluator.evaluate()["mae"] == pytest.approx((1 + 3 + 2 + 2) / 4)


def test_bare_tensor_target_key_passes_through():
    """A `target_key` naming a bare tensor (not a role dict) is used as-is."""
    evaluator = AutomatedBenchmarkEvaluator(
        metric_reductions=[{"name": "mae", "reduce_method": "mean"}],
    )  # pred_key=None -> outputs is the tensor; target_key="data_tensor" is a bare tensor
    data_sample = {"metainfo": {"data_tensor": torch.ones(2, 2)}}
    evaluator.process(data_sample, torch.full((2, 2), 1.5))
    assert evaluator.evaluate()["mae"] == pytest.approx(0.5)


def test_process_rejects_shape_mismatch():
    """Prediction and target of different shapes raise before any metric sees them."""
    evaluator = AutomatedBenchmarkEvaluator(
        metric_reductions=[{"name": "mae", "reduce_method": "mean"}],
        pred_key=None,
        target_key="targets",
        target_role="y",
    )
    data_sample = {"metainfo": {"targets": {"y": torch.zeros(2, 3)}}}

    with pytest.raises(ValueError, match="shape mismatch"):
        evaluator.process(data_sample, torch.zeros(2, 2))
