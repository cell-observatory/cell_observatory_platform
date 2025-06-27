import os
import sys
import pytest
import logging

import pandas as pd

import torch

from ray.train import report
from ray import init, cluster_resources
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig, RunConfig, FailureConfig, CheckpointConfig

from omegaconf import open_dict
from hydra.utils import get_class

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

def _test_loggers_dist(cfg: DictConfig):
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
        from utils.context import process_rank, get_world_size
        from training.loggers import LocalEventWriter
        rank = process_rank()
        world = get_world_size()

        trainer_cls = get_class(cfg.trainer)
        trainer = trainer_cls(cfg)

        recorder = trainer.event_recorder
        writers_list = trainer.event_writers_list
        writer = writers_list.writers[0]
        assert isinstance(writer, LocalEventWriter), \
            "Expected LocalEventWriter for testing writers"

        step_csv = writer.step_scalars_savepath
        epoch_csv = writer.epoch_scalars_savepath

        # test putting scalars
        n_steps = 3
        for it in range(n_steps):
            trainer._iter, recorder._iter = it, it
            trainer._epoch, recorder._epoch = 0, 0
            recorder.put_scalar("loss", float(rank + it + 1), scope="step")

        # test all gathers scalars from all workers
        step_scalars, _ = writers_list.reduce_scalars()
        # test write scalars on rank 0
        writer._write_scalar_impl(step_scalars, scope="step")

        # test clearing scalars method
        assert len(recorder.get_step_scalars()["loss"]) == n_steps
        recorder.clear_scalars()
        assert all(len(v) == 0 for v in recorder.get_step_scalars().values())

        # test putting epoch scalars
        trainer._epoch, recorder._epoch = 0, 0
        recorder.put_scalar("val_loss", float(rank + 10), scope="epoch")

        # test all gathers epoch scalars from all workers
        _, epoch_scalars = writers_list.reduce_scalars()
        # test write epoch scalars on rank 0
        writer._write_scalar_impl(epoch_scalars, scope="epoch")

        # no-op for LocalEventWriter
        writers_list.close()

        # test that the scalars were written 
        # and reduced correctly
        if rank == 0:
            assert step_csv.exists(), "step CSV missing"
            step_df = pd.read_csv(step_csv)
            expected_means = {
                it: sum(float(k + it + 1) for k in range(world)) / world
                for it in range(n_steps)
            }
            for _, row in step_df.iterrows():
                assert pytest.approx(row["loss"]) == expected_means[row["iter"]]

            assert epoch_csv.exists(), "epoch CSV missing"
            epoch_df = pd.read_csv(epoch_csv)
            mean_val_loss = sum(float(k + 10) for k in range(world)) / world
            assert len(epoch_df) == 1
            assert pytest.approx(epoch_df.loc[0, "val_loss"]) == mean_val_loss

        # TODO: test appending to existing CSVs

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

def test_loggers():
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("At least 2 GPUs are required for this test")


    cfg = _make_config()
    load_dotenv(cfg.paths.dotenv_path, verbose=True)

    with open_dict(cfg):
        cfg.experiment_name = "test_event_logging"
        cfg.paths.resume_checkpointdir = None
        cfg.clusters.worker_nodes = 1
        cfg.clusters.gpus_per_worker = 2 
        cfg.clusters.cpus_per_gpu = 4
        cfg.clusters.mem_per_cpu = 31000
        cfg.loggers.event_writers = [
            w for w in cfg.loggers.event_writers
            if w._target_.endswith(".LocalEventWriter")
        ]

    metrics = _test_loggers_dist(cfg)
    assert metrics.get("success", False), "Distributed event-logging test failed"