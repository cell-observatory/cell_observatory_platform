import pytest
import torch

from cell_observatory_platform.evaluation.automated_benchmark_evaluator import (
    AutomatedBenchmarkEvaluator,
)
from cell_observatory_platform.evaluation.base_evaluation import BaseEvaluator


def test_base_evaluator_rejects_loss_none():
    evaluator = BaseEvaluator(training_metrics=[{"step_loss": "mean"}])

    with pytest.raises(RuntimeError, match="loss_dict=None"):
        evaluator.process({}, {}, None)


def test_base_evaluator_rejects_data_sample_none():
    evaluator = BaseEvaluator(training_metrics=[{"step_loss": "mean"}])

    with pytest.raises(RuntimeError, match="data_sample=None"):
        evaluator.process(None, {}, {"step_loss": 0.0})


def test_base_evaluator_rejects_outputs_none():
    evaluator = BaseEvaluator(training_metrics=[{"step_loss": "mean"}])

    with pytest.raises(RuntimeError, match="outputs=None"):
        evaluator.process({}, None, {"step_loss": 0.0})


def test_automated_benchmark_select_pred_dict_requires_pred_key():
    evaluator = AutomatedBenchmarkEvaluator(
        metric_reductions=[{"mae": "mean"}],
        pred_key=None,
    )

    with pytest.raises(ValueError, match="pred_key is None"):
        evaluator._select_pred({"x": torch.ones(2)})


def test_automated_benchmark_select_pred_tensor_rejects_pred_key():
    evaluator = AutomatedBenchmarkEvaluator(
        metric_reductions=[{"mae": "mean"}],
        pred_key="x",
    )

    with pytest.raises(TypeError, match="non-dict"):
        evaluator._select_pred(torch.ones(2))


def test_automated_benchmark_process_shape_mismatch():
    evaluator = AutomatedBenchmarkEvaluator(
        metric_reductions=[{"mae": "mean"}],
        pred_key=None,
        target_key="targets",
    )
    data_sample = {"metainfo": {"targets": [torch.zeros(2, 3)]}}

    with pytest.raises(ValueError, match="shape mismatch"):
        evaluator.process(data_sample, torch.zeros(2, 2))


def test_automated_benchmark_ssim_is_explicitly_disabled():
    with pytest.raises(NotImplementedError, match="SSIMMetric"):
        AutomatedBenchmarkEvaluator(metric_reductions=[{"ssim": "mean"}])
