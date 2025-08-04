import os
import sys
import shlex
import logging
import subprocess
import warnings
from pathlib import Path
from subprocess import call, run

import hydra
from hydra import compose
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

OmegaConf.register_new_resolver("eval", eval)
from utils.container import get_container_info

# Update environment variables
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"

load_dotenv(Path(__file__).parent / ".env", verbose=True)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def q(x: str) -> str:
    """Shortcut for shlex.quote."""
    return shlex.quote(str(x))


def get_defaults_overrides(defaults_overrides: list[dict] | None):
    defaults = {}
    for default in defaults_overrides or []:
        for key, default_path in default.items():
            defaults[key] = compose(config_name=default_path)
    return defaults



def set_env_from_cfg(cfg: DictConfig) -> None:
    def _to_str(v):
        return "1" if isinstance(v, bool) and v \
            else "0" if isinstance(v, bool) else str(v)

    if not hasattr(cfg.optimizations, "env"):
        warnings.warn("No env section found in config.")
        return

    for key, val in cfg.optimizations.env.items():
        if val is None:
            continue
        env_key = key.upper()
        os.environ[env_key] = _to_str(val)
        logger.debug("Set %s=%s", env_key, os.environ[env_key])

# modify Hydra config on cmd line to use different models
@hydra.main(config_path="configs", config_name="test_pretrain_4d_mae_local")
def main(cfg: DictConfig):
    logger.info(f"Launch config: {OmegaConf.to_yaml(cfg)}")

    if cfg.run_type == "multi_run":
        assert len(list(cfg.runs)) > 0, \
            "cfg.runs must be a list of configurations for multiple training jobs."

        logger.info("Launching multiple training jobs...")
        for run in list(cfg.runs):
            logger.info(f"Launching job with base config: {run.cfg}")
            logger.info(f"Launching job with overrides: {run.overrides}")

            run_cfg = compose(config_name=run.cfg)

            # first we merge the defaults_overrides with the run config
            defaults_overrides = get_defaults_overrides(run.get("defaults_overrides", None))
            if defaults_overrides is not None and len(defaults_overrides) > 0:
                logger.info(f"Defaults overrides: {defaults_overrides}")
                with open_dict(run_cfg):
                    for key, value in defaults_overrides.items():
                        logger.info(f"Overriding Defaults: {key}")
                        # TODO: make sure old default values are removed correctly here
                        run_cfg[key] = {}
                        run_cfg = OmegaConf.merge(run_cfg, value)
            
            # next we merge the run overrides with the resulting run config
            override_cfg = OmegaConf.create(OmegaConf.to_container(run.overrides))
            run_cfg = OmegaConf.merge(run_cfg, override_cfg)

            if cfg.get("data_base_dir"):
                logger.info(f"Root directory for runs set to: {cfg.data_base_dir}")
                run_path = run_cfg.paths.outdir / Path(cfg.data_base_dir) / Path(run.name).with_suffix("")
                run_path.mkdir(parents=True, exist_ok=True)
            else:
                logger.info(f"Root directory for runs set to: {run_cfg.paths.outdir}")
                run_path = run_cfg.paths.outdir / Path(run.name).with_suffix("")
                run_path.mkdir(parents=True, exist_ok=True)

            with open_dict(run_cfg.paths):
                run_cfg.paths.outdir = str(run_path)
                logger.info(f"Output directory for this run: {run_cfg.paths.outdir}")

            if cfg.get("wandb_tags"):
                logger.info(f"Adding W&B tags: {cfg.wandb_tags}")
                # TODO: we should consider making event_writers a dict
                #       instead of a list to prevent these kinds of loops
                with open_dict(run_cfg):
                    for event_writer in run_cfg.loggers.event_writers:
                        if event_writer._target_.endswith("WandBEventWriter"):
                            event_writer.tags = event_writer.tags + list(cfg.wandb_tags)

            with open_dict(run_cfg):
                run_cfg.experiment_name = run.name.replace(".yaml", "")

            # save the run config to a file for reproducibility
            # and so we can pass to the runner and inject
            # package global variable since we are saving
            # config in `experiments` folder
            run_cfg_path = run_path / run.name
            run_cfg_yml = OmegaConf.to_yaml(run_cfg)
            run_cfg_yml = "#@package _global_\n" + run_cfg_yml
            run_cfg_path.write_text(run_cfg_yml)

            logger.info(f"Run config saved to: {run_cfg_path}")

            # launch the job
            logger.info(f"Run config after overrides: {run_cfg_yml}")
            launch_job(run_cfg, run_config_name=run_cfg_path)

    elif cfg.run_type == "single_run" or cfg.run_type == "tune":
        logger.info("Launching a single training job...")
        launch_job(cfg)
    else:
        raise ValueError(f"Unknown run type: {cfg.run_type}. "
                         f"Please set cfg.run_type to either 'single_run', 'multi_run', or 'tune'.")


def launch_job(cfg: DictConfig, run_config_name: str = None):
    # TODO: make sure this recapitulates the old ENV variable
    #       setting logic
    # set environment variables from the config
    set_env_from_cfg(cfg)

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

    # load extra env variables
    # assert cfg.paths.dotenv_path is not None and Path(cfg.paths.dotenv_path).exists(), \
    #     f"Missing dotenv path: {cfg.paths.dotenv_path}"
    load_dotenv(cfg.paths.dotenv_path, verbose=True)

    # ensure correct config is being passed to the runner
    config_name = run_config_name if run_config_name is not None else HydraConfig.get().job.config_name
    print(f"Running with config: {config_name}")

    # print full configuration (for debugging)
    print(f"\nFull run configuration:")
    print("\n" + OmegaConf.to_yaml(cfg))

    print(f"Current working directory: {Path.cwd()}")
    print(f"Creating output directory: {cfg.paths.outdir}...")
    outdir = Path(cfg.paths.outdir).resolve()
    outdir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory for training job: {outdir}")

    # bind path is --bind <host_path> : <container_path>
    if hasattr(cfg.paths, "data_path") and cfg.paths.data_path is not None:
        bind = f'{cfg.paths.data_path}:{cfg.paths.data_path}'
        workspace = f'{cfg.paths.repo_path}:{cfg.paths.workdir}'

    assert (cfg.paths.apptainer_image is None) != (cfg.paths.docker_image is None), \
        "Either apptainer_image or docker_image must be specified, but not both"

    if cfg.paths.apptainer_image is not None:
        # use apptainer for running the job
        image = cfg.paths.apptainer_image
    elif cfg.paths.docker_image is not None:
        # else use docker for running the job
        image = cfg.paths.docker_image
    else:
        raise ValueError("Either apptainer_image or docker_image must be specified in the configuration.")

    if cfg.clusters.launcher_type == "slurm":
        cfg.paths.ray_script = cfg.paths.ray_script.replace("ray_local_cluster.sh", "ray_slurm_cluster.sh")
    elif cfg.clusters.launcher_type == "lsf":
        cfg.paths.ray_script = cfg.paths.ray_script.replace("ray_local_cluster.sh", "ray_lsf_cluster.sh")

    if run_config_name is not None:
        task = f"{cfg.clusters.python_env} {cfg.paths.runner_script} --config-name {Path(config_name).name} --config-dir={Path(config_name).parent}"
    else:
        task = f"{cfg.clusters.python_env} {cfg.paths.runner_script} --config-name {config_name}"

    if cfg.clusters.job_name is None:
        cfg.clusters.job_name = config_name

    ray_wrap = (
        f" bash {q(cfg.paths.ray_script)} "
        f"-b {q(str(bind))} "
        f"-c {q(cfg.clusters.cpus_per_worker)} "
        f"-e {q(image)} "
        f"-g {q(cfg.clusters.gpus_per_worker)} "
        f"-m {q(cfg.clusters.mem_per_worker)} "
        f"-n {q(cfg.clusters.worker_nodes)} "
        f"-o {q(str(outdir))} "
        f"-p {q(cfg.clusters.partition)} "
        f"-s {q(str(workspace))} "
        f"-t {q(task)} "
        f"-j {q(cfg.clusters.job_name)} "
        f"-x {q(cfg.clusters.exclusive)} "
        f"-y {q(cfg.clusters.head_node_gpus)} "
        f"-z {q(cfg.clusters.head_node_cpus)} "
    )

    if cfg.clusters.launcher_type == "local":  # for running jobs on your local workstation without a job scheduler
        if container_info['ide_type'] is None:
            print("Running local training job with configuration:")
            print(ray_wrap)
            call([ray_wrap], shell=True)
        else:
            print(f"Running in {container_info['ide_type']} IDE in {container_info['container_type']} environment")

            # needs to be here to launch jobs in the IDE
            from training import runner
            runner.main(cfg)

    elif cfg.clusters.launcher_type == "slurm":

        # set resources to allocate head node, then the head node will allocate the rest of the worker nodes
        sjob_worker_nodes = [f"/usr/bin/sbatch "]
        sjob_worker_nodes.append(f"--qos={cfg.clusters.qos}")
        sjob_worker_nodes.append(f"--partition={cfg.clusters.partition}")

        if cfg.clusters.exclusive is not None:
            sjob_worker_nodes.append(f"--exclusive")
        else:
            sjob_worker_nodes.append(f"--ntasks 1")
            sjob_worker_nodes.append(f"--nodes 1")
            sjob_worker_nodes.append(f"--cpus-per-task={cfg.clusters.head_node_cpus}")
            sjob_worker_nodes.append(f"--gres=gpu:{cfg.clusters.head_node_gpus}")
            sjob_worker_nodes.append(f"--mem={cfg.clusters.head_node_mem}")

        if cfg.clusters.constraint is not None:
            sjob_worker_nodes.append(f"-C '{cfg.clusters.constraint}'")

        if cfg.clusters.nodelist is not None:
            sjob_worker_nodes.append(f"--nodelist='{cfg.clusters.nodelist}'")

        if cfg.clusters.dependency is not None:
            sjob_worker_nodes.append(f"--dependency={cfg.clusters.job_name}")

        if cfg.clusters.timelimit is not None:
            sjob_worker_nodes.append(f"--time={cfg.clusters.timelimit}")

        sjob_worker_nodes.append(f"--job-name={cfg.clusters.job_name}")
        sjob_worker_nodes.append(f"--output={outdir / cfg.clusters.job_name}.log")
        sjob_worker_nodes.append(f"--export=ALL")
        sjob_worker_nodes.append(f"--wrap={q(ray_wrap)}")

        print("Submitting slurm job with configuration:")
        cmd = " ".join(sjob_worker_nodes)
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)

    elif cfg.clusters.launcher_type == "lsf":

        # set resources to allocate head node, then the head node will allocate the rest of the worker nodes
        sjob_worker_nodes = [f"bsub"]
        sjob_worker_nodes.append(f"-q {cfg.clusters.partition}")

        if cfg.clusters.exclusive is not None:
            sjob_worker_nodes.append(f"-x")
        else:
            sjob_worker_nodes.append(f"-n {cfg.clusters.head_node_cpus}")
            sjob_worker_nodes.append(f'-gpu "num={cfg.clusters.head_node_gpus}:mode=shared"')

        if cfg.clusters.dependency is not None:
            sjob_worker_nodes.append(f'-w "done({cfg.clusters.job_name})"')

        if cfg.clusters.timelimit is not None:
            sjob_worker_nodes.append(f"--We {cfg.clusters.timelimit} ")

        sjob_worker_nodes.append(f"-J {cfg.clusters.job_name}")
        sjob_worker_nodes.append(f"-o {outdir / cfg.clusters.job_name}.log")
        sjob_worker_nodes.append(f"-env 'all'")
        sjob_worker_nodes.append(f"{q(ray_wrap)}")

        print(f"Checking available Janelia cluster resources")
        print(f"Looking for {cfg.clusters.worker_nodes} node(s) on {cfg.clusters.partition} queue")
        try:
            run(
                f'bash {cfg.paths.repo_path}/cluster/check_available_janelia_nodes.sh \
                    -p {cfg.clusters.partition} -n {cfg.clusters.worker_nodes}',
                check=True,
                shell=True
            )

            print("Requested resources are available now!")
        except Exception as e:
            print(f"Error running resources check: {e}")

        print("Submitting lsf job with configuration:")
        cmd = " ".join(sjob_worker_nodes)
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)

    else:
        raise ValueError(
            f"Unknown launcher type: {cfg.clusters.launcher_type}. "
            f"Please set cfg.clusters.launcher_type to either 'local', 'slurm', or 'lsf'."
        )

if __name__ == "__main__":
    main()