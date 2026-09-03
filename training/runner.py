import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

# This ensures both relative imports (training.loops) and absolute imports
# (cell_observatory_platform.training.helpers) work correctly
_pkg_dir = str(Path(__file__).resolve().parent.parent)
_workspace_root = str(Path(__file__).resolve().parent.parent.parent)
for _path in [_pkg_dir, _workspace_root]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

import warnings

# Scoped suppression only: a blanket ignore silenced torch deprecations,
# overflow warnings, and this repo's OWN warnings.warn() calls driver-wide.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ray(\..*)?")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm(\..*)?")

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
logger.setLevel(logging.INFO)   # DEBUG flooded the shared "ray" logger driver-wide
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


# Env vars propagated to Ray workers. The old behavior shipped the ENTIRE
# driver environment to every worker. Only the cluster/comms/logging-relevant 
# families plus PATH-critical vars pass through.
_ENV_VAR_PREFIXES = (
    "RAY_", "CUDA_", "NCCL_", "WANDB_", "HF_",
    # runtime families workers actually read:
    "SUPABASE_",           # local metadata DB host/port (data/databases, utils/context)
    "TORCH_", "PYTORCH_",  # torch comm/alloc knobs (configure_torch_comm_env etc.)
)
_ENV_VAR_KEYS = (
    "PATH", "LD_LIBRARY_PATH", "PYTHONPATH",
    "HOME", "TMPDIR", "OMP_NUM_THREADS",
)


def _curated_env_vars() -> dict:
    return {
        k: v
        for k, v in os.environ.items()
        if k.startswith(_ENV_VAR_PREFIXES) or k in _ENV_VAR_KEYS
    }


def initialize_session(cfg: DictConfig):
    nsys_env = cfg.hooks.get("nsys_env", None)

    env_vars = _curated_env_vars()

    if nsys_env is not None:
        env_vars.update(OmegaConf.to_container(nsys_env, resolve=True, enum_to_str=True))

    workspace_root = str(Path(__file__).resolve().parent.parent)

    # pip_packages = OmegaConf.to_container(cfg.clusters.get("extra_pip_packages", []), resolve=True)
    # if not pip_packages:
    #     pip_packages = None

    # py_modules (not working_dir) ships the package; excludes keep the upload
    # from dragging along git history, caches, checkpoints, and run outputs.
    runtime_env = RuntimeEnv(
        env_vars=env_vars,
        py_modules=[workspace_root],
        excludes=[
            ".git",
            "**/__pycache__",
            "**/*.egg-info",
            "**/.pytest_cache",
            "**/wandb",
            "**/outputs",
            "**/*.pt",
            "**/*.sif",
        ],
        # pip=pip_packages,
        )

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

        # The Ray session dir holds unix sockets and the object store -- it must
        # live on LOCAL disk (a _temp_dir symlinked onto /clusterfs puts sockets
        # on NFS: flaky connects, plasma on network storage). Keep the session
        # local and expose the logs on the NFS outdir for postmortem instead.
        tmpdir = f"/tmp/ray_{uuid.uuid1()}"
        init(
            log_to_driver=True,
            runtime_env=runtime_env,
            num_cpus=cfg.clusters.total_cpus + cfg.clusters.cpus_for_training_coordinator,
            num_gpus=cfg.clusters.total_gpus,
            object_store_memory=cfg.clusters.object_store_memory,
            ignore_reinit_error=True,
            _temp_dir=tmpdir,
        )
        try:
            raylogs_link = Path(cfg.paths.outdir) / "ray_logs"
            raylogs_link.parent.mkdir(parents=True, exist_ok=True)
            if not raylogs_link.exists():
                os.symlink(
                    os.path.join(tmpdir, "session_latest"), raylogs_link,
                    target_is_directory=True,
                )
            logger.info(f"Ray session on local disk: {tmpdir}; logs linked at {raylogs_link}")
        except OSError as e:
            logger.warning(f"Could not link ray logs into outdir: {e}")

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
        use_gpu=cfg.clusters.scaling_config.use_gpu,
    )

    checkpoint_config = CheckpointConfig(**cfg.checkpoint.ray_checkpoint_config)

    run_config = RunConfig(
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
        logger.exception(f"Training failed with exception: {e}")
        sys.exit(1)


def run_tune(cfg: DictConfig):
    logger.info(f"Starting hyperparameter tuning with Ray Tune...")
    logger.info(f"Using parameter space: {cfg.tune.param_space}")
    logger.info(f"Using tune config: {cfg.tune.tune_config}")

    # ensures the output directory is unique for each tuning run.
    # Resolve the timestamp ONCE and store the literal: an unresolved
    # "${now:...}" re-resolves to a DIFFERENT time on every access, so derived
    # paths (logdir, checkpoint dir) silently diverge across processes.
    with open_dict(cfg):
        cfg.paths.outdir = os.path.join(
            str(cfg.paths.outdir), datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

    param_space = instantiate(cfg.tune.param_space)
    tune_cfg = instantiate(cfg.tune.tune_config)

    # Build the Tune search space by placing each sampler at its REAL nested
    # position inside the full config tree. This fixes three former breaks:
    # (1) merge order -- the static cfg used to override the samplers;
    # (2) double nesting -- param_space already carried a train_loop_config
    #     level and was nested under train_loop_config again, so sampled
    #     values landed where the trainer never reads them;
    # (3) dotted keys -- "clusters.batch_size" was stored as a literal key,
    #     which the worker's attribute access never expands.
    # Constants pass through Tune untouched; sampler leaves get sampled, and
    # each trial's worker receives one complete, concrete config dict.
    base_tree = OmegaConf.to_container(cfg, resolve=False)

    def _set_dotted(tree: dict, dotted: str, value) -> None:
        keys = dotted.split(".")
        node = tree
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

    overrides = param_space.get("train_loop_config", param_space) or {}
    for dotted_key, sampler in overrides.items():
        _set_dotted(base_tree, dotted_key, sampler)
    run_cfg = base_tree

    scaling_config = ScalingConfig(
        num_workers=cfg.clusters.scaling_config.num_workers,
        resources_per_worker=cfg.clusters.scaling_config.resources_per_worker,
        use_gpu=cfg.clusters.scaling_config.use_gpu,
    )
    checkpoint_config = CheckpointConfig(**cfg.checkpoint.ray_checkpoint_config)
    run_config = RunConfig(
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
        logger.exception(f"Tuning failed with exception: {e}")
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
