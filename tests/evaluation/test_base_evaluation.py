import os
import sys
import pytest
import logging

import torch

from ray import init, cluster_resources
from ray.train import report
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig, RunConfig, FailureConfig, CheckpointConfig

from hydra.utils import get_class
from omegaconf import open_dict

from dotenv import load_dotenv
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
OmegaConf.register_new_resolver("eval", eval)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# TODO: we should probably have separate configs
#       for testing modules
def _make_config() -> DictConfig:
    with initialize(config_path="../../configs"):
        return compose(config_name="pretrain_mae_local")

def _test_base_evaluate_dist(cfg: DictConfig):
    init(log_to_driver=True,
         runtime_env={k: v for k, v in os.environ.items()},
         num_cpus=cfg.clusters.total_cpus + cfg.clusters.cpus_for_training_coordinator,
         num_gpus=cfg.clusters.total_gpus,
         ignore_reinit_error=True
    )

    for resource, count in cluster_resources().items():
        logger.info(f"{resource}: {count}")

    scaling_config = ScalingConfig(
        num_workers=cfg.clusters.scaling_config.num_workers,
        resources_per_worker=cfg.clusters.scaling_config.resources_per_worker,
        trainer_resources=cfg.clusters.scaling_config.trainer_resources,
        use_gpu=cfg.clusters.scaling_config.use_gpu
    )

    checkpoint_config = CheckpointConfig(**cfg.checkpoint.ray_checkpoint_config)
    run_config = RunConfig(
        log_to_file=cfg.clusters.run_config.log_to_file,
        checkpoint_config=checkpoint_config,
        failure_config=FailureConfig(max_failures=0),
        storage_path=cfg.clusters.run_config.storage_path,
    )
    
    torch_config = TorchConfig(timeout_s=cfg.clusters.torch_config.timeout_s)

    def run_init(cfg: DictConfig):
        from utils.context import process_rank
        rank = process_rank()
        trainer_cls = get_class(cfg.trainer)
        trainer = trainer_cls(cfg)
        evaluator = trainer.evaluator

        # test process method
        num_steps = 4
        for step in range(num_steps):
            loss_dict = {
                "step_loss": torch.tensor(float(rank + step + 1))
            }

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

    trainer = TorchTrainer(
        train_loop_per_worker=run_init,
        train_loop_config=cfg,
        run_config=run_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        datasets=None
    )

    result = trainer.fit()
    return result.metrics

def test_evaluation():
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("At least 2 GPUs are required for this test")

    cfg = _make_config()
    load_dotenv(cfg.paths.dotenv_path, verbose=True)

    with open_dict(cfg):
        cfg.experiment_name = "test_evaluation"
        cfg.paths.resume_checkpointdir = None
        cfg.clusters.worker_nodes = 1
        cfg.clusters.gpus_per_worker = 2
        cfg.clusters.cpus_per_gpu = 4
        cfg.clusters.mem_per_cpu = 31000
        cfg.evaluation.evaluator._target_ = "evaluation.base_evaluation.BaseEvaluator" 
        cfg.evaluation.evaluator.training_metrics = [{"step_loss": "mean"}]

    metrics = _test_base_evaluate_dist(cfg)
    assert metrics.get("success", False), "Distributed event-logging test failed"