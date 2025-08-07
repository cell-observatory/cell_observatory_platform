import os
import sys
import time
import logging
import uuid
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from ray.tune import Tuner
from ray import init, cluster_resources
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train import ScalingConfig, CheckpointConfig, RunConfig, FailureConfig

import hydra
from hydra.utils import get_method, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)
if not OmegaConf.has_resolver("now"):
    OmegaConf.register_new_resolver("now", lambda fmt: time.strftime(fmt))

from torch.utils.data import random_split

from data.datasets.pretrain_dataset_ray import get_dataset_ray

logger = logging.getLogger("ray")
logger.setLevel(logging.DEBUG)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


def initialize_session(cfg: DictConfig):
    nsys_env = cfg.hooks.get("nsys_env", None)
    
    if nsys_env is not None:
        nsys_env = OmegaConf.to_container(nsys_env, resolve=True, enum_to_str=True)
        runtime_env = { **os.environ, **nsys_env }
    else:
        runtime_env = {**os.environ}

    if 'head_node_ip' in os.environ and 'port' in os.environ:
        address = os.environ["head_node_ip"]
        port = os.environ["port"]

        logger.info(f"Connecting to address: {address}")
        init(
            address=f"{address}:{port}",
            log_to_driver=True,
            runtime_env=runtime_env,
            object_store_memory=cfg.clusters.object_store_memory,
        )

    else:
        logger.info(f"Starting a new local ray cluster")

        tmpdir = f"/tmp/symlink_{uuid.uuid1()}"
        raylogsdir = Path(cfg.paths.outdir)
        os.symlink(raylogsdir, tmpdir, target_is_directory=True)
        logger.info(f"Link outdir to tmpdir: {cfg.paths.outdir} -> {tmpdir}")
        init(
            log_to_driver=True,
            runtime_env=runtime_env,
            num_cpus=cfg.clusters.total_cpus + cfg.clusters.cpus_for_training_coordinator,
            num_gpus=cfg.clusters.total_gpus,
            object_store_memory=cfg.clusters.object_store_memory,
            ignore_reinit_error=True,
            _temp_dir=tmpdir,
        )

    logger.info('\nResources available to this Ray cluster:')
    for resource, count in cluster_resources().items():
        logger.info(f'{resource}: {count}')

    return cluster_resources().items()


def run_session(cfg: DictConfig):

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

    if cfg.datasets.dataset._target_.endswith("PretrainDatasourceRay"):
        db = instantiate(cfg.datasets.databases)
        dataset_len = len(db.hypercubes_dataframe)
        if cfg.datasets.split is not None:
            val_size = round(dataset_len * cfg.datasets.split)
            train_subset, val_subset = random_split(
                range(dataset_len),
                lengths=[dataset_len - val_size, val_size]
            )
            train_indices, val_indices = train_subset.indices, val_subset.indices
            train_dataset = get_dataset_ray(cfg, indices=train_indices)
            val_dataset = get_dataset_ray(cfg, indices=val_indices)
            dataset = {"train": train_dataset, "val": val_dataset}
        else:
            train_dataset = get_dataset_ray(cfg, indices=None)
            dataset = {"train": train_dataset}
    else:
        dataset = None

    trainer = TorchTrainer(
        train_loop_per_worker=get_method(cfg.loop_per_worker_script),
        train_loop_config=cfg,
        run_config=run_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        datasets=dataset
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


def run_tune(cfg: DictConfig):
    logger.info(f"Starting hyperparameter tuning with Ray Tune...")
    logger.info(f"Using parameter space: {cfg.tune.param_space}")
    logger.info(f"Using tune config: {cfg.tune.tune_config}")

    # ensures the output directory is unique for each tuning run
    with open_dict(cfg):
        cfg.paths.outdir = os.path.join(cfg.paths.outdir, "${now:%Y-%m-%d_%H-%M-%S}")

    param_space = instantiate(cfg.tune.param_space)
    tune_cfg = instantiate(cfg.tune.tune_config)
    run_cfg = OmegaConf.merge(param_space, cfg)

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
        train_loop_per_worker=get_method(cfg.loop_per_worker_script),
        run_config=run_config,
        scaling_config=scaling_config,
        torch_config=torch_config,
        datasets=None
    )

    # NOTE: we need to pass the run config to the Tuner instead of 
    #       the trainer directly to allow for Tune to inject
    #       the hyperparameter space and sampling logic into
    #       our train config
    tuner = Tuner(
        trainable=trainer.as_trainable(),
        param_space={"train_loop_config": run_cfg},
        tune_config=tune_cfg,
    )

    try:
        results = tuner.fit()
        logger.info(f"Tuning completed with results: {results}")
    except Exception as e:
        logger.error(f"Tuning failed with exception: {e}")
        sys.exit(1)


@hydra.main(config_path="../configs", config_name="test_pretrain_4d_mae_local")
def main(cfg: DictConfig):

    timeit = time.time()

    # ray cluster already set up in: ray_local_script.sh OR 
    #                                ray_lsf_cluster.sh  OR
    #                                ray_slurm_cluster.sh
    # depending on the cluster Hydra configuration

    cluster_resources = initialize_session(cfg)

    if cfg.run_type == "single_run":
        result = run_session(cfg)
    elif cfg.run_type == "tune":
        result = run_tune(cfg)
    else:
        logger.error(f"Unknown run_type: {cfg.run_type}. Expected 'tune' or 'single_run'.")
        sys.exit(1)

    logger.info(f"Total time elapsed: {time.time() - timeit:.2f} sec.")
    sys.exit(0)

if __name__ == "__main__":
    main()