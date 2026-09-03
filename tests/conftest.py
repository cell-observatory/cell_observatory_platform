import logging
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv
from hydra import compose, initialize
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

try:
    OmegaConf.register_new_resolver("eval", eval)
except ValueError:
    pass

import ray
from ray import cluster_resources, init
from ray.runtime_env import RuntimeEnv
from ray.util import list_named_actors
from ray.train import CheckpointConfig, FailureConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer

from cell_observatory_platform.tests.ray_init_helpers import local_cwd_for_ray_start
from cell_observatory_platform.utils.cleanup import unlink_shared_memory
from cell_observatory_platform.utils.container import get_container_info

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parent.parent / ".env", verbose=True)


def pytest_addoption(parser):
    parser.addoption(
        "--run-benchmarks",
        action="store_true",
        default=False,
        help="run tests marked benchmark",
    )
    parser.addoption(
        "--run-localdb",
        action="store_true",
        default=False,
        help="run tests marked localdb",
    )


def _has_cuda_toolkit() -> bool:
    # Tests marked `cuda` need the CUDA *toolkit* (nvcc), not just the runtime
    # libs. DeepSpeed's `installed_cuda_version()` probe (which fires at
    # module-import time inside several training modules) requires BOTH a
    # callable nvcc AND `CUDA_HOME`/`CUDA_PATH` set, so we mirror that here.
    nvcc = shutil.which("nvcc")
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not nvcc and cuda_home:
        candidate = os.path.join(cuda_home, "bin", "nvcc")
        if os.access(candidate, os.X_OK):
            nvcc = candidate
    if not nvcc:
        return False
    if not cuda_home:
        # nvcc found on PATH but no CUDA_HOME set — DeepSpeed will still fail
        # at import. Treat as "no toolkit" so file-gated tests are skipped.
        return False
    return True


# Module-level DeepSpeed imports inside these files trigger nvcc probing at
# collection time, so we have to skip them earlier than markers can fire.
# Currently empty: no test file imports DeepSpeed at module level any more
# (test_jepa_models.py is CPU-only; test_loggers.py / test_test_trainer_predict_dispatch.py
# were split so their Ray-harness import lives inside @pytest.mark.cuda tests).
_CUDA_TOOLKIT_REQUIRED_FILES: tuple[str, ...] = ()


def pytest_ignore_collect(collection_path, config):
    if _has_cuda_toolkit():
        return False
    repo_root = Path(__file__).resolve().parent.parent
    target = Path(collection_path).resolve()
    for rel in _CUDA_TOOLKIT_REQUIRED_FILES:
        if target == (repo_root / rel).resolve():
            return True
    return False


def pytest_collection_modifyitems(config, items):
    has_cuda_toolkit = _has_cuda_toolkit()
    run_localdb = config.getoption("--run-localdb")
    run_benchmarks = config.getoption("--run-benchmarks")

    skip_cuda = pytest.mark.skip(reason="CUDA toolkit (nvcc) not available")
    skip_localdb = pytest.mark.skip(reason="need --run-localdb to execute localdb tests")
    # The `benchmark` marker documents itself as opt-in, but a marker alone only
    # enables `-m benchmark` SELECTION -- it does not deselect. These spin up Ray
    # actors over SHM buffers and measure the machine rather than the code.
    skip_benchmark = pytest.mark.skip(reason="need --run-benchmarks to execute benchmark tests")

    for item in items:
        if not has_cuda_toolkit and "cuda" in item.keywords:
            item.add_marker(skip_cuda)
        if not run_localdb and "localdb" in item.keywords:
            item.add_marker(skip_localdb)
        if not run_benchmarks and "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)


# (removed: every live sdpa_kernel([...]) call in models/layers/attention.py
#  already lists SDPBackend.MATH; the autouse patch_sdpa_for_tests fixture only
#  forced the attention stack to import for every test, data tests included.)


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def gloo_pg(monkeypatch):
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(free_tcp_port()))
    dist.init_process_group("gloo", rank=0, world_size=1)
    yield
    dist.destroy_process_group()


@pytest.fixture(autouse=True)
def _reset_ray_and_cuda_before_test():
    yield
    if ray.is_initialized():
        try:
            ray.shutdown()
        except Exception:
            pass
    # No sleep: ray.shutdown() is synchronous, and distributed_test() already does its
    # own shutdown + sleep(3) + unlink_shared_memory() in its `finally`.
    # tests/inference/conftest.py still overrides this fixture for its session cluster.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# keeping this until we migrate models
# tests to config setup
@pytest.fixture(scope="session")
def models_kargs():
    # Only the keys read by tests/models/test_{convnext,jepa,mae,vit}_models.py
    # (git grep 'models_kargs\['); delete the fixture outright if the models-area
    # rewrite of those four files stops consuming it.
    return dict(
        outdir=Path(__file__).resolve().parent.parent / "pretrained_models",
        modes=15, batch_size=2, hidden_size=768, patches=32, heads=16, repeats=4,
        dropout=0.1, fixed_dropout_depth=False, abs_sincos_enc=True, rope_pos_enc=False,
    )


@pytest.fixture(scope="session")
def config() -> DictConfig:
    with initialize(config_path="../configs"):
        cfg = compose(config_name="tests")

    container_info = get_container_info()
    print(f"Container type: {container_info['container_type']}")

    assert cfg.paths.outdir is not None, f"Missing output directory: {cfg.paths.outdir}"

    assert (
        Path(cfg.paths.data_path) in Path(cfg.paths.outdir).parents
    ), f"Output directory [{cfg.paths.outdir}] not in data path [{cfg.paths.data_path}]"

    assert cfg.clusters.batch_size % cfg.clusters.worker_nodes == 0, (
        f"batch_size {cfg.clusters.batch_size} must divide evenly among " f"{cfg.clusters.worker_nodes} worker nodes"
    )

    if container_info["container_type"] == "native":
        for k in ["runner_script"]:
            cfg.paths[k] = cfg.paths[k].replace(cfg.paths.repo_path, cfg.paths.workdir)

    else:  # running in a docker/apptainer
        [print(f"\t{k}: {v}") for k, v in container_info["container_details"].items()]

        for k in ["outdir", "ray_script", "runner_script", "dotenv_path"]:
            if cfg.paths[k] is None:
                cfg.paths[k] = None
                logger.warning(f"Path {k} is not set in the config, skipping")
                continue
            cfg.paths[k] = cfg.paths[k].replace(cfg.paths.repo_path, cfg.paths.workdir)

    # TODO need to look into why the abc cluster only works with the cursor protocol
    if Path("/clusterfs").exists():
        cfg.datasets.databases.protocol = "cursor"
    else:
        cfg.datasets.databases.protocol = "binary"

    # load extra env variables
    # assert cfg.paths.dotenv_path is not None and Path(cfg.paths.dotenv_path).exists(), \
    #     f"Missing dotenv path: {cfg.paths.dotenv_path}"
    load_dotenv(cfg.paths.dotenv_path, verbose=True)

    return cfg


def _cleanup_ray_test_actors():
    if not ray.is_initialized():
        return
    try:
        actors = list_named_actors(all_namespaces=True)
    except Exception:
        return
    for entry in actors:
        ns = entry.get("namespace") or ""
        name = entry.get("name") or ""
        if not name:
            continue

        if ns == "schedulers" or (isinstance(ns, str) and ns.startswith("buffers_node_")):
            try:
                handle = ray.get_actor(name, namespace=ns)
                ray.kill(handle, no_restart=True)
            except Exception:
                pass


def distributed_test(cfg: DictConfig, test: str):
    # test needs to be a string that can
    # be resolved to a callable to prevent
    # serialization issues
    test = get_method(test)

    unlink_shared_memory()

    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    runtime_env = RuntimeEnv(
        env_vars={k: v for k, v in os.environ.items()}, working_dir=project_root, py_modules=[project_root]
    )

    with local_cwd_for_ray_start():
        init(
            log_to_driver=True,
            runtime_env=runtime_env,
            num_cpus=cfg.clusters.total_cpus + cfg.clusters.cpus_for_training_coordinator,
            num_gpus=cfg.clusters.total_gpus,
            object_store_memory=cfg.clusters.object_store_memory,
            ignore_reinit_error=True,
        )

    try:
        for resource, count in cluster_resources().items():
            logger.info(f"{resource}: {count}")

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
            train_loop_per_worker=test,
            train_loop_config=cfg,
            run_config=run_config,
            scaling_config=scaling_config,
            torch_config=torch_config,
            datasets=None,
        )

        result = trainer.fit()
        _cleanup_ray_test_actors()
        return result.metrics
    finally:
        ray.shutdown()
        time.sleep(3)
        unlink_shared_memory()
