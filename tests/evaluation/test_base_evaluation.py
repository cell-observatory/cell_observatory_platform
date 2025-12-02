import logging
import sys
from pathlib import Path

import pytest
import torch
from hydra.utils import get_class
from omegaconf import DictConfig, open_dict
from ray.train import report

from cell_observatory_platform.tests.conftest import config, distributed_test


def _test_base_evaluation(cfg: DictConfig):
    from cell_observatory_platform.utils.context import process_rank

    rank = process_rank()
    trainer_cls = get_class(cfg.trainer)
    trainer = trainer_cls(cfg)
    evaluator = trainer.evaluator

    # test process method
    num_steps = 4
    for step in range(num_steps):
        loss_dict = {"step_loss": torch.tensor(float(rank + step + 1))}

        before = len(evaluator.metrics["step_loss"].loss_values)
        evaluator.process(None, None, loss_dict)
        assert len(evaluator.metrics["step_loss"].loss_values) == before + 1

    # test evaluate method
    results = evaluator.evaluate()
    values = evaluator.metrics["step_loss"].loss_values
    expected = sum(values) / len(values)
    assert pytest.approx(results["step_loss"]) == expected

    # test reset method
    evaluator.reset()
    assert all(v is None for v in evaluator._results.values())

    report({"success": True})


def test_evaluation(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")

    with open_dict(config):
        config.experiment_name = "test_evaluation"
        config.paths.resume_checkpointdir = None

        config.evaluation.evaluator._target_ = "cell_observatory_platform.evaluation.base_evaluation.BaseEvaluator"
        config.evaluation.evaluator.training_metrics = [{"step_loss": "mean"}]

    metrics = distributed_test(
        cfg=config, test="cell_observatory_platform.tests.evaluation.test_base_evaluation._test_base_evaluation"
    )
    assert metrics.get("success", False), "Distributed base_evaluation test failed"
