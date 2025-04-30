import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore")

import torch
from ray import init, cluster_resources
from ray.train import ScalingConfig,  CheckpointConfig, RunConfig, FailureConfig
from ray.train.torch import TorchTrainer, TorchConfig

import sys
import logging
import os
import time
from pathlib import Path

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf, open_dict
OmegaConf.register_new_resolver("eval", eval)

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["NCCL_DEBUG"] = "TRACE"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "GRAPH"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["NCCL_CUMEM_ENABLE"] = "0"
os.environ["NCCL_CROSS_NIC"] = "1"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"

logger = logging.getLogger("ray")
logger.setLevel(logging.DEBUG)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


def train_model(cfg: DictConfig):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    with open_dict(cfg):
        cfg.outdir = Path(cfg.outdir)
        logger.info(f"{cfg.outdir=}")
        Path(cfg.outdir).mkdir(exist_ok=True, parents=True)

        cfg.logdir = Path(cfg.logdir)
        logger.info(f"{cfg.logdir=}")
        Path(cfg.logdir).mkdir(exist_ok=True, parents=True)

        cfg.checkpointdir = Path(cfg.checkpointdir)
        logger.info(f"{cfg.checkpointdir=}")
        Path(cfg.checkpointdir).mkdir(exist_ok=True, parents=True)

        cfg.run_config.storage_path = str(cfg.outdir)

        cfg.deepspeed.tensorboard.output_path = str(cfg.logdir)
        cfg.deepspeed.tensorboard.job_name = Path(cfg.outdir).name

        cfg.deepspeed.csv_monitor.output_path = str(cfg.logdir)
        cfg.deepspeed.csv_monitor.job_name = Path(cfg.outdir).name

        cfg.deepspeed.flops_profiler.output_file = f"{cfg.logdir}/flops_profiler.log"

        if cfg.clusters.gpu_workers == -1:
            cfg.clusters.gpu_workers = torch.cuda.device_count()

        cfg.clusters.worker_batch_size = cfg.batch_size // (cfg.clusters.workers * cfg.clusters.gpu_workers)
        cfg.clusters.num_workers = int(cfg.clusters.workers) * int(cfg.clusters.gpu_workers)
        cfg.scaling_config.num_workers = cfg.clusters.num_workers
        cfg.clusters.cpus_per_gpu = int(cfg.clusters.cpu_workers) // int(cfg.clusters.gpu_workers)
        cfg.scaling_config.resources_per_worker["CPU"] = cfg.clusters.cpus_per_gpu

        cfg.datasets.batch_size = cfg.clusters.worker_batch_size
        cfg.deepspeed.train_batch_size = cfg.batch_size

    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    scaling_config = ScalingConfig(
        num_workers=cfg.scaling_config.num_workers,
        resources_per_worker=cfg.scaling_config.resources_per_worker,
        trainer_resources=cfg.scaling_config.trainer_resources,
        use_gpu=cfg.scaling_config.use_gpu
    )

    checkpoint_config = CheckpointConfig(**cfg.run_config.checkpoint_config)

    run_config = RunConfig(
        log_to_file=cfg.run_config.log_to_file,
        checkpoint_config=checkpoint_config,
        failure_config=FailureConfig(max_failures=0),
        storage_path=cfg.run_config.storage_path,
    )

    torch_config = TorchConfig(timeout_s=cfg.torch_config.timeout_s)

    training_paradigm = get_method(cfg.paradigm)

    trainer = TorchTrainer(
        train_loop_per_worker=training_paradigm,
        train_loop_config=cfg, # OmegaConf.to_container(cfg, resolve=True),
        run_config=run_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        datasets=None,
    )

    try:
        result = trainer.fit()
        logger.info(f"Model saved to {result.path}, {result.checkpoint}")
        logger.info(f"Training completed with metrics: {result.metrics}")
        logger.info(f"Error logs: {result.error}")
        logger.info(f"Best model checkpoint: {result.best_checkpoints}")

    except Exception as e:
        logger.info(f"Training failed with exception: {e}")
        sys.exit(1)


@hydra.main(config_path="../configs", config_name="ao_vit")
def main(cfg: DictConfig):

    timeit = time.time()

    try:
        address = os.environ["head_node_ip"]
        port = os.environ["port"]
        # address = '127.0.1.1'
        # port = '32032'

        logger.info(f"Connecting to address: {address}")
        init(
            address=f"{address}:{port}",
            log_to_driver=True,
            runtime_env={"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "GRAPH", "NCCL_P2P_LEVEL": "NVL"},
            _system_config={"worker_heartbeat_timeout_ms": cfg.clusters.max_worker_heartbeat_timeout * 60 * 1000},
        )

    except KeyError:
        logger.info(f"Starting a new local ray cluster")
        init(
            log_to_driver=True,
            runtime_env={"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "GRAPH", "NCCL_P2P_LEVEL": "NVL"},
            num_cpus=cfg.clusters.cpu_workers + 1, # 1 cpu core for the training coordinator
            num_gpus=cfg.clusters.gpu_workers,
            ignore_reinit_error=True
        )

    logger.info('\nResources available to this Ray cluster:')
    for resource, count in cluster_resources().items():
        logger.info(f'{resource}: {count}')

    train_model(cfg)

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")
    sys.exit(0)


if __name__ == "__main__":
    main()
