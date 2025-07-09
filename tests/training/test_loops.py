import os
import pytest
from pathlib import Path

import torch

from omegaconf import open_dict
from hydra.utils import get_class

from ray.train import report

from tests.conftest import distributed_test, config


def _test_train_loop_dist(config):
    trainer_cls = get_class(config.trainer)
    trainer = trainer_cls(config)
    trainer.run()    
    report({"success": True})

def _test_testing_loop_dist(config):
    trainer_cls = get_class(config.trainer)
    trainer = trainer_cls(config)
    trainer.test()
    report({"success": True})

@pytest.mark.order(1)
def test_train_loop(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")

    with open_dict(config):
        config.experiment_name = "test_train_loop"
        config.paths.resume_checkpointdir = None

        config.schedulers.epochs = 1
        # config.datasets.databases.max_hypercubes = 10000

        config.trainer = "training.loops.EpochBasedTrainer"

    metrics = distributed_test(cfg=config, test="tests.training.test_loops._test_train_loop_dist")
    assert metrics.get("success", False), "Distributed loops test failed"

@pytest.mark.order(2)
@pytest.mark.skip(reason="This test is temporarily disabled.")
def test_testing_loop(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")

    with open_dict(config):
        config.experiment_name = "test_testing_loop"
        config.paths.resume_checkpointdir = None

        # TODO: may need a checkpoint directory for testing or a dummy checkpoint
        config.paths.pretrained_checkpointdir = os.path.join(
            Path(config.paths.outdir).parent,
            "test_train_loop",
            "checkpoints"
        )
        config.checkpoint.checkpoint_manager.checkpoint_tag = "latest_model"

        config.trainer = "training.loops.TestTrainer"
        config.evaluation.val_metric = "test_step_loss"

        # config.datasets.databases.max_hypercubes = 10000

    metrics = distributed_test(cfg=config, test="tests.training.test_loops._test_testing_loop_dist")
    assert metrics.get("success", False), "Distributed loops test failed"