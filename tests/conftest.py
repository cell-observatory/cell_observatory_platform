import os
import sys
import pytest
import logging
from pathlib import Path

from dotenv import load_dotenv
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

from utils.container import get_container_info

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parent.parent / ".env", verbose=True)


# keeping this until we migrate models 
# tests to config setup
@pytest.fixture(scope="session")
def models_kargs():
    repo = Path(__file__).resolve().parent.parent
    models_kargs = dict(
        repo=repo,
        outdir=repo/'pretrained_models',
        modes=15,
        batch_size=2,
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
    return models_kargs


@pytest.fixture(scope="session")
def config() -> DictConfig:
    with initialize(config_path="../configs"):
        cfg = compose(config_name="tests")

    container_info = get_container_info()
    print(f"Container type: {container_info['container_type']}")

    assert cfg.paths.outdir is not None, f"Missing output directory: {cfg.paths.outdir}"

    assert Path(cfg.paths.data_path) in Path(cfg.paths.outdir).parents, \
        f"Output directory [{cfg.paths.outdir}] not in data path [{cfg.paths.data_path}]"

    assert cfg.clusters.batch_size % cfg.clusters.worker_nodes == 0, (
        f"batch_size {cfg.clusters.batch_size} must divide evenly among "
        f"{cfg.clusters.worker_nodes} worker nodes"
    )

    if container_info['container_type'] == 'native':
        for k in ['runner_script']:
            cfg.paths[k] = cfg.paths[k].replace(cfg.paths.repo_path, cfg.paths.workdir)

    else:  # running in a docker/apptainer
        [print(f"\t{k}: {v}") for k, v in container_info['container_details'].items()]

        for k in ['outdir', 'ray_script', 'runner_script', 'dotenv_path']:
            cfg.paths[k] = cfg.paths[k].replace(cfg.paths.repo_path, cfg.paths.workdir)

    # TODO need to look into why the abc cluster only works with the cursor protocol
    if Path('/clusterfs').exists():
        cfg.datasets.databases.protocol = 'cursor'
    else:
        cfg.datasets.databases.protocol = 'binary'

    # load extra env variables
    # assert cfg.paths.dotenv_path is not None and Path(cfg.paths.dotenv_path).exists(), \
    #     f"Missing dotenv path: {cfg.paths.dotenv_path}"
    load_dotenv(cfg.paths.dotenv_path, verbose=True)

    # print full configuration (for debugging)
    print("\n" + OmegaConf.to_yaml(cfg))

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