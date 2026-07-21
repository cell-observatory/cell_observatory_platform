import logging
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from hydra.utils import get_class
from omegaconf import DictConfig, open_dict
from ray.train import report, Checkpoint

from cell_observatory_platform.tests.conftest import config, distributed_test
from cell_observatory_platform.utils.context import is_main_process


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

    metrics = {"success": True}
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Checkpoint.from_directory(tmpdir)
        if is_main_process():
            return report(metrics=metrics, checkpoint=checkpoint)
        else:
            return report(metrics=metrics, checkpoint=None)


# Needs the local sandbox database: the trainer's dataloader hits
# local_database.execute_arrow, which fails with "Connection refused" when the
# DB server is not up -- surfacing only as an opaque Ray WorkerGroupError.
# Opt in with --run-localdb (see tests/conftest.py).
@pytest.mark.localdb
@pytest.mark.cuda
def test_evaluation(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")

    with open_dict(config):
        config.experiment_name = "test_evaluation"
        config.paths.resume_checkpointdir = None

        # No _target_: loops.py builds via REGISTRY.build("evaluator",
        # cfg.evaluation.evaluator.name, ...) and base_evaluator.yaml already sets
        # `name: base` (registered to BaseEvaluator). Setting _target_ was a no-op.
        #
        # Metric specs are {name, key?, **ctor_kwargs} (metrics._build_one_metric).
        # The old {loss_key: reduce_method} form raises KeyError('name') inside the
        # Ray worker, visible only as an opaque WorkerGroupError.
        config.evaluation.evaluator.training_metrics = [
            {"name": "train_loss", "key": "step_loss", "reduce_method": "mean"},
        ]

    metrics = distributed_test(
        cfg=config, test="cell_observatory_platform.tests.evaluation.test_base_evaluation._test_base_evaluation"
    )
    assert metrics.get("success", False), "Distributed base_evaluation test failed"