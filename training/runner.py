import os
import sys
import time
import logging

import warnings
warnings.filterwarnings("ignore")

import torch
from ray import init, cluster_resources
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig, CheckpointConfig, RunConfig, FailureConfig

import hydra
from omegaconf import DictConfig, OmegaConf

from training.loops import train_loop_per_worker

logger = logging.getLogger("ray")
logger.setLevel(logging.DEBUG)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


def run_session(cfg: DictConfig):

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

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
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=cfg,
        run_config=run_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        datasets=None
    )

    try:
        result = trainer.fit()
        logger.info(f"Model saved to {result.path}, {result.checkpoint}")
        logger.info(f"Training completed with metrics: {result.metrics}")
        logger.info(f"Error logs: {result.error}")
        logger.info(f"Best model checkpoint: {result.best_checkpoints}")

    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        sys.exit(1)


@hydra.main(config_path="../configs", config_name="pretrain_mae_local")
def main(cfg: DictConfig):

    timeit = time.time()

    # ray cluster already set up in: ray_local_script.sh OR 
    #                                ray_lsf_cluster.sh  OR
    #                                ray_slurm_cluster.sh
    # depending on the cluster Hydra configuration

    if 'head_node_ip' in os.environ and 'port' in os.environ:
        address = os.environ["head_node_ip"]
        port = os.environ["port"]

        logger.info(f"Connecting to address: {address}")
        init(
            address=f"{address}:{port}",
            log_to_driver=True,
            runtime_env={k: v for k, v in os.environ.items()},
            _system_config={"worker_heartbeat_timeout_ms": cfg.clusters.max_worker_heartbeat_timeout},
        )

    else:
        logger.info(f"Starting a new local ray cluster")
        init(
            log_to_driver=True,
            runtime_env={k: v for k, v in os.environ.items()},
            num_cpus=cfg.clusters.total_cpus + cfg.clusters.cpus_for_training_coordinator,
            num_gpus=cfg.clusters.total_gpus,
            ignore_reinit_error=True
        )

    logger.info('\nResources available to this Ray cluster:')
    for resource, count in cluster_resources().items():
        logger.info(f'{resource}: {count}')

    run_session(cfg)

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")
    sys.exit(0)

if __name__ == "__main__":
    main()