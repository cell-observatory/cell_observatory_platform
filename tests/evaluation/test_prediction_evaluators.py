import pytest
import torch

from cell_observatory_platform.evaluation.automated_benchmark_evaluator import (
    AutomatedBenchmarkEvaluator,
)
from cell_observatory_platform.evaluation.base_evaluation import BaseEvaluator


def test_base_evaluator_rejects_loss_dict_none():
    """BaseEvaluator.process raises a clear TypeError when loss_dict is None
    (otherwise the loss_dict[metric] subscript would raise an opaque error)."""
    evaluator = BaseEvaluator(
        training_metrics=[{"name": "train_loss", "key": "step_loss", "reduce_method": "mean"}]
    )
    with pytest.raises(TypeError):
        evaluator.process({}, {}, None)


def test_automated_benchmark_select_pred_dict_by_key():
    evaluator = AutomatedBenchmarkEvaluator(
        metric_reductions=[{"name": "mae", "reduce_method": "mean"}],
        pred_key="x",
        target_key="targets",
    )
    data_sample = {"metainfo": {"targets": [torch.zeros(2, 2)]}}
    # pred_key selects outputs["x"]; matching shape -> no error.
    evaluator.process(data_sample, {"x": torch.zeros(2, 2)})


def test_automated_benchmark_process_shape_mismatch():
    evaluator = AutomatedBenchmarkEvaluator(
        metric_reductions=[{"name": "mae", "reduce_method": "mean"}],
        pred_key=None,
        target_key="targets",
    )
    data_sample = {"metainfo": {"targets": [torch.zeros(2, 3)]}}

    with pytest.raises(ValueError, match="shape mismatch"):
        evaluator.process(data_sample, torch.zeros(2, 2))
