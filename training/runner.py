import logging
import os
import sys
import time
import uuid
from pathlib import Path

# This ensures both relative imports (training.loops) and absolute imports
# (cell_observatory_platform.training.helpers) work correctly
_pkg_dir = str(Path(__file__).resolve().parent.parent)
_workspace_root = str(Path(__file__).resolve().parent.parent.parent)
for _path in [_pkg_dir, _workspace_root]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import warnings

warnings.filterwarnings("ignore")

import hydra
from hydra.utils import get_method, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from ray import cluster_resources, init
from ray.runtime_env import RuntimeEnv
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer
from ray.tune import Tuner

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)
if not OmegaConf.has_resolver("now"):
    OmegaConf.register_new_resolver("now", lambda fmt: time.strftime(fmt))

logger = logging.getLogger("ray")
logger.setLevel(logging.DEBUG)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


def initialize_session(cfg: DictConfig):
    nsys_env = cfg.hooks.get("nsys_env", None)

    env_vars = {}
    env_vars.update(os.environ)

    if nsys_env is not None:
        env_vars.update(*OmegaConf.to_container(nsys_env, resolve=True, enum_to_str=True))

    workspace_root = str(Path(__file__).resolve().parent.parent)

    runtime_env = RuntimeEnv(working_dir=workspace_root, env_vars=env_vars, py_modules=[workspace_root])

    if "head_node_ip" in os.environ and "port" in os.environ:
        address = os.environ["head_node_ip"]
        port = os.environ["port"]

        logger.info(f"Connecting to address: {address}")
        init(
            address=f"{address}:{port}",
            log_to_driver=True,
            runtime_env=runtime_env,
        )

    else:
        logger.info(f"No existing Ray cluster detected in environment variables.")
        logger.info(f"Port detected: {os.environ.get('port', 'None')}")
        logger.info(f"head_node_ip detected: {os.environ.get('head_node_ip', 'None')}")
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

    logger.info("\nResources available to this Ray cluster:")
    for resource, count in cluster_resources().items():
        logger.info(f"{resource}: {count}")

    return cluster_resources().items()


def run_session(cfg: DictConfig):
    # Debug mode: run training loop directly in main process (bypasses Ray Train)
    # This allows debugpy breakpoints to work inside the training loop
    if cfg.get("debug_mode", False):
        logger.info("DEBUG MODE: Running training loop directly in main process (no Ray Train)")

        # Set up single-GPU distributed environment for DeepSpeed
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")

        # Initialize PyTorch distributed before DeepSpeed
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", world_size=1, rank=0)

        train_loop_fn = get_method(cfg.loop_per_worker_script)
        train_loop_fn(cfg)
        return

    scaling_config = ScalingConfig(
        num_workers=cfg.clusters.scaling_config.num_workers,
        resources_per_worker=cfg.clusters.scaling_config.resources_per_worker,
        trainer_resources=cfg.clusters.scaling_config.trainer_resources,
        use_gpu=cfg.clusters.scaling_config.use_gpu,
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
        train_loop_config=cfg,
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
        use_gpu=cfg.clusters.scaling_config.use_gpu,
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
        datasets=None,
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


@hydra.main(version_base=None, config_path="../configs", config_name=None)
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
