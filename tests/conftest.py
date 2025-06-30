import os
import sys
import pytest
import logging
from pathlib import Path
from typing import Optional

import torch

from dotenv import load_dotenv
from omegaconf import DictConfig
from hydra.utils import get_method
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass

from ray import init, cluster_resources
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig, RunConfig, FailureConfig, CheckpointConfig

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# keeping this until we migrate models 
# tests to config setup
@pytest.fixture(scope="session")
def kargs():
    repo = Path.cwd()
    kargs = dict(
        repo=repo,
        prediction_filename_pattern=r"*[!_gt|!_realspace|!_noisefree|!_predictions_psf|!_corrected_psf|!_reconstructed_psf].tif",
        dataset=repo/"dataset/training_dataset/YuMB_lambda510/z200-y97-x97/z64-y64-x64/z15/",
        fishdb_dir=repo/"dataset/fishdb_data/",
        outdir=repo/'pretrained_models',
        input_shape=64,
        modes=15,
        batch_size=512,
        hidden_size=768,
        patches=32,
        heads=16,
        repeats=4,
        opt='lamb',
        lr=5e-4,
        wd=5e-5,
        ld=None,
        ema=(.998, 1.),
        epochs=5,
        warmup=1,
        cooldown=1,
        clip_grad=.5,
        fixedlr=False,
        dropout=0.1,
        fixed_dropout_depth=False,
        amp='fp16',
        finetune=None,
        profile=False,
        workers=1,
        gpu_workers=1,
        cpu_workers=8,
    )
    return kargs


@pytest.fixture(scope="session")
def config() -> DictConfig:
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root, verbose=True)
    with initialize(config_path="../configs"):
        cfg = compose(config_name="tests")
    return cfg


def distributed_test(cfg: DictConfig, test: str):
    # test needs to be a string that can 
    # be resolved to a callable to prevent
    # serialization issues
    test = get_method(test)
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

    trainer = TorchTrainer(
        train_loop_per_worker=test,
        train_loop_config=cfg,
        run_config=run_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        datasets=None
    )

    result = trainer.fit()
    return result.metrics