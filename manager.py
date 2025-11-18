import base64
import itertools
import logging
import os
import shlex
import subprocess
import sys
import uuid
import warnings
import zlib
from pathlib import Path
from subprocess import call, run

_workspace_root = str(Path(__file__).resolve().parent.parent)
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)

import hydra
from dotenv import load_dotenv
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

OmegaConf.register_new_resolver("eval", eval)

from cell_observatory_platform.utils.container import get_container_info
from cell_observatory_platform.utils.profiling import enable_profiling

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


def _get_sweep_list(d):
    def _walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield from _walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            yield (prefix, obj)

    leafs = list(_walk("", d))
    leaf_lists = [(k, v) for (k, v) in leafs if isinstance(v, list)]
    assert len(leaf_lists) == 1, \
        f"sweep element must contain exactly one list-valued leaf; got: {leaf_lists}"
    return leaf_lists[0]


def get_sweep_axes(sweep_cfg):
    if sweep_cfg is None:
        return []

    d = OmegaConf.to_container(sweep_cfg, resolve=True)

    if isinstance(d, list):
        axes = []
        for item in d:
            assert isinstance(item, dict), f"Each sweep entry must be a dict; got {type(item)}"
            key, values = _get_sweep_list(item)
            axes.append((key, values))
        return axes

    if isinstance(d, dict):
        key, values = _get_sweep_list(d)
        return [(key, values)]

    raise TypeError(f"Unsupported sweep type: {type(d)}")


def sanitize_name(val):
    if isinstance(val, float):
        s = f"{val}"
        return s.replace(".", "p")
    return str(val).lower()


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


def posixify(s: str) -> str:
    if s is None:
        return s
    s = str(s).replace("\\", "/")
    if s.startswith("\\"):
        s = "/" + s.lstrip("\\/")
    return s


# modify Hydra config on cmd line to use different models
@hydra.main(config_path="configs", config_name=None)
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

            load_dotenv(run_cfg.paths.dotenv_path, verbose=True)

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

            sweep_axes = get_sweep_axes(cfg.get("sweep", None))
            if sweep_axes:
                sweep_combinations = itertools.product(*[[(k, v) for v in vals] for (k, vals) in sweep_axes])
            else:
                sweep_combinations = [()]

            for sweep_combination in sweep_combinations:
                run_cfg_sweep = run_cfg
                run_name = run.name

                if sweep_combination:
                    dotlist = [f"{k}={v}" for (k, v) in sweep_combination]
                    run_cfg_sweep = OmegaConf.merge(run_cfg, OmegaConf.from_dotlist(dotlist))

                    sweep_cfg_name_suffix = []
                    for k, v in sweep_combination:
                        safe_key = k.replace('.', '_')
                        sweep_cfg_name_suffix.append(f"{safe_key}_{sanitize_name(v)}")
                    run_name = f"{Path(run_name).with_suffix('')}_sweep_" + "_".join(sweep_cfg_name_suffix) + ".yaml"

                if cfg.get("data_base_dir"):
                    logger.info(f"Root directory for runs set to: {cfg.data_base_dir}")
                    run_path = run_cfg_sweep.paths.outdir / Path(cfg.data_base_dir) / Path(run_name).with_suffix("")
                    run_path.mkdir(parents=True, exist_ok=True)
                else:
                    logger.info(f"Root directory for runs set to: {run_cfg_sweep.paths.outdir}")
                    run_path = run_cfg_sweep.paths.outdir / Path(run_name).with_suffix("")
                    run_path.mkdir(parents=True, exist_ok=True)

                with open_dict(run_cfg_sweep.paths):
                    run_cfg_sweep.paths.outdir = str(run_path)
                    logger.info(f"Output directory for this run: {run_cfg_sweep.paths.outdir}")

                if cfg.get("wandb_tags"):
                    logger.info(f"Adding W&B tags: {cfg.wandb_tags}")
                    # TODO: we should consider making event_writers a dict
                    #       instead of a list to prevent these kinds of loops
                    with open_dict(run_cfg_sweep):
                        for event_writer in run_cfg_sweep.loggers.event_writers:
                            if event_writer._target_.endswith("WandBEventWriter"):
                                event_writer.tags = event_writer.tags + list(cfg.wandb_tags)

                with open_dict(run_cfg_sweep):
                    run_cfg_sweep.experiment_name = run_name.replace(".yaml", "")

                # save the run config to a file for reproducibility
                # and so we can pass to the runner and inject
                # package global variable since we are saving
                # config in `experiments` folder
                run_cfg_path = run_path / run_name
                run_cfg_yml = OmegaConf.to_yaml(run_cfg_sweep)
                run_cfg_yml = "#@package _global_\n" + run_cfg_yml
                run_cfg_path.write_text(run_cfg_yml)

                logger.info(f"Run config saved to: {run_cfg_path}")

                # launch the job
                logger.info(f"Run config after overrides: {run_cfg_yml}")
                launch_job(run_cfg_sweep, run_config_name=run_cfg_path)

    elif cfg.run_type == "single_run" or cfg.run_type == "tune":
        load_dotenv(cfg.paths.dotenv_path, verbose=True)
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
    enable_profiling(cfg)

    container_info = get_container_info()
    print(f"Container type: {container_info['container_type']}")

    assert cfg.paths.outdir is not None, f"Missing output directory: {cfg.paths.outdir}"

    if hasattr(cfg.paths, "data_path") and cfg.paths.data_path is not None:
        assert Path(cfg.paths.data_path) in Path(cfg.paths.outdir).parents, \
            f"Output directory [{cfg.paths.outdir}] not in data path [{cfg.paths.data_path}]"

    assert cfg.clusters.batch_size % cfg.clusters.worker_nodes == 0, (
        f"batch_size {cfg.clusters.batch_size} must divide evenly among "
        f"{cfg.clusters.worker_nodes} worker nodes"
    )

    if container_info['container_type'] == 'native':
        if cfg.clusters.launcher_type != "runai":
            for k in ['runner_script']:
                cfg.paths[k] = cfg.paths[k].replace(cfg.paths.repo_path, cfg.paths.workdir)

    else:  # running in a docker/apptainer
        [print(f"\t{k}: {v}") for k, v in container_info['container_details'].items()]

        for k in ['outdir', 'ray_script', 'runner_script', 'dotenv_path']:
            cfg.paths[k] = cfg.paths[k].replace(cfg.paths.repo_path, cfg.paths.workdir)

    # ensure correct config is being passed to the runner
    config_name = run_config_name if run_config_name is not None else HydraConfig.get().job.config_name
    print(f"Running with config: {config_name}")

    # print full configuration (for debugging)
    print(f"\nFull run configuration:")
    print("\n" + OmegaConf.to_yaml(cfg))

    print(f"Current working directory: {Path.cwd()}")
    is_remote = cfg.clusters.launcher_type in {"runai", "slurm", "lsf"}
    print(f"Creating output directory: {cfg.paths.outdir}...")
    if is_remote:
        outdir = posixify(cfg.paths.outdir)
    else:
        outdir = Path(cfg.paths.outdir).resolve()
        outdir.mkdir(exist_ok=True, parents=True)
        outdir = str(outdir)

    print(f"Output directory for training job: {outdir}")

    # bind path is --bind <host_path> : <container_path>
    if hasattr(cfg.paths, "data_path") and cfg.paths.data_path is not None:
        bind = f'{cfg.paths.data_path}:{cfg.paths.data_path}'
        workspace = f'{cfg.paths.repo_path}:{cfg.paths.workdir}'
        storage_server = f'{cfg.paths.server_folder_path}:{cfg.paths.server_folder_path}'

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
    elif cfg.clusters.launcher_type == "runai":
        cfg.paths.ray_script = cfg.paths.ray_script.replace("ray_local_cluster.sh", "ray_runai_cluster.sh")

    if run_config_name is not None:
        config_dir = Path(config_name).parent
        config_dir = posixify(config_dir if is_remote else config_dir.resolve())
        task = f"{cfg.clusters.python_env} {cfg.paths.runner_script} --config-name {Path(config_name).name} --config-dir={config_dir}"
    else:
        task = f"{cfg.clusters.python_env} {cfg.paths.runner_script} --config-name {config_name}"

    if cfg.clusters.job_name is None:
        cfg.clusters.job_name = config_name

    if cfg.clusters.multijob_submission:
        cfg.paths.ray_script = str(cfg.paths.ray_script).replace('.sh', '_multijob.sh')
        
    ray_wrap = (
        f" bash {q(cfg.paths.ray_script)} "
        f"-b {q(str(bind))} "
        f"-d {q(str(storage_server))} "
        f"-c {q(cfg.clusters.cpus_per_worker)} "
        f"-e {image} "
        f"-g {q(cfg.clusters.gpus_per_worker)} "
        f"-m {q(cfg.clusters.mem_per_worker)} "
        f"-n {q(cfg.clusters.worker_nodes)} "
        f"-o {q(str(outdir))} "
        f"-p {q(cfg.clusters.partition)} "
        f"-q {q(cfg.clusters.object_store_memory)} "
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

        if cfg.clusters.multijob_submission: 
            '''
                Multijob submission (each worker node will be submitted as a separate job)
                Set resources to allocate head node, then the head node will allocate the rest of the worker nodes
                We assume idential configrations for worker nodes, but the head node could be different 
            '''
            sjob_worker_nodes = ["/usr/bin/sbatch "]
            sjob_worker_nodes.append(f"--qos={cfg.clusters.qos}")
            sjob_worker_nodes.append(f"--partition={cfg.clusters.partition}")
            sjob_worker_nodes.append(f"--ntasks 1")
            sjob_worker_nodes.append(f"--nodes 1")
            sjob_worker_nodes.append(f"--cpus-per-task={cfg.clusters.head_node_cpus}")
            sjob_worker_nodes.append(f"--gres=gpu:{cfg.clusters.head_node_gpus}")
            sjob_worker_nodes.append(f"--mem={cfg.clusters.head_node_mem}")
        else:
            '''
                All resources will be requested and allocated at once.
                We assume idential configrations for all worker nodes
            '''
            sjob_worker_nodes = ["/usr/bin/sbatch "]
            sjob_worker_nodes.append(f"--qos={cfg.clusters.qos}")
            sjob_worker_nodes.append(f"--partition={cfg.clusters.partition}")
            sjob_worker_nodes.append(f"--nodes {cfg.clusters.worker_nodes}")
            sjob_worker_nodes.append(f"--ntasks-per-node 1")
            sjob_worker_nodes.append(f"--cpus-per-task={cfg.clusters.cpus_per_worker}")
            sjob_worker_nodes.append(f"--gres=gpu:{cfg.clusters.gpus_per_worker}")
            sjob_worker_nodes.append(f"--mem={cfg.clusters.mem_per_worker}")
            
        if cfg.clusters.constraint is not None:
            sjob_worker_nodes.append(f"-C '{cfg.clusters.constraint}'")

        if cfg.clusters.nodelist is not None:
            sjob_worker_nodes.append(f"--nodelist='{cfg.clusters.nodelist}'")

        if cfg.clusters.dependency is not None:
            sjob_worker_nodes.append(f"--dependency={cfg.clusters.job_name}")

        if cfg.clusters.timelimit is not None:
            sjob_worker_nodes.append(f"--time={cfg.clusters.timelimit}")

        sjob_worker_nodes.append(f"--job-name={cfg.clusters.job_name}")
        sjob_worker_nodes.append(f"--output={outdir}/{cfg.clusters.job_name}.log")
        sjob_worker_nodes.append(f"--export=ALL")
        sjob_worker_nodes.append(f"--wrap={q(ray_wrap)}")

        print("Submitting slurm job with configuration:")
        cmd = " ".join(sjob_worker_nodes)
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)

    elif cfg.clusters.launcher_type == "lsf":

        if cfg.clusters.multijob_submission: 
            '''
                Multijob submission (each worker node will be submitted as a separate job)
                Set resources to allocate head node, then the head node will allocate the rest of the worker nodes
                We assume idential configrations for worker nodes, but the head node could be different 
            '''
            print("Checking available Janelia cluster resources")
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
            
            sjob_worker_nodes = ["bsub"]
            sjob_worker_nodes.append(f"-q {cfg.clusters.partition}")
            sjob_worker_nodes.append(f"-n {cfg.clusters.head_node_cpus}")
            sjob_worker_nodes.append(f'-gpu "num={cfg.clusters.head_node_gpus}:mode=shared"')
        
        else:
            '''
                All resources will be requested and allocated at once.
                We assume idential configrations for all worker nodes
            '''
            sjob_worker_nodes = ["bsub"]
            sjob_worker_nodes.append(f"-q {cfg.clusters.partition}")
            sjob_worker_nodes.append(f"-n {cfg.clusters.cpus_per_worker * cfg.clusters.worker_nodes}")
            sjob_worker_nodes.append(f'-R "span[ptile={cfg.clusters.cpus_per_worker}]"')
            sjob_worker_nodes.append(f'-app parallel-96')
            sjob_worker_nodes.append(f'-gpu "num={cfg.clusters.gpus_per_worker}:mode=exclusive_process"')
        

        if cfg.clusters.dependency is not None:
            sjob_worker_nodes.append(f'-w "done({cfg.clusters.job_name})"')

        if cfg.clusters.timelimit is not None:
            sjob_worker_nodes.append(f"--We {cfg.clusters.timelimit} ")

        sjob_worker_nodes.append(f"-J {cfg.clusters.job_name}")
        sjob_worker_nodes.append(f"-o {outdir}/{cfg.clusters.job_name}.log")
        sjob_worker_nodes.append(f"-env 'all'")
        sjob_worker_nodes.append(f"{q(ray_wrap)}")

        print("Submitting lsf job with configuration:")
        cmd = " ".join(sjob_worker_nodes)
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)

    elif cfg.clusters.launcher_type == "runai":
        '''
            Currently, we only support requesting and allocating all resources
            at once with RUNAI scheduler. We assume identical configurations
            for all worker nodes.
        '''
        def quote_posix_list(argv):
            return " ".join(shlex.quote(str(x)) for x in argv)

        if cfg.clusters.interactive_session_type:
            ray_args = [
                "bash", cfg.paths.ray_script,
                "-c", str(cfg.clusters.cpus_per_worker),
                "-g", str(cfg.clusters.gpus_per_worker),
                "-m", str(cfg.clusters.mem_per_worker),
                "-n", str(cfg.clusters.worker_nodes),
                "-o", str(outdir),
                "-q", str(cfg.clusters.object_store_memory),
                "-t", f"{task}",
            ]

            if cfg.clusters.head_node_gpus not in (None, "", "None"):
                ray_args += ["-y", str(cfg.clusters.head_node_gpus)]
            if cfg.clusters.head_node_cpus not in (None, "", "None"):
                ray_args += ["-z", str(cfg.clusters.head_node_cpus)]

            ray_wrap_posix = " ".join(shlex.quote(str(x)) for x in ray_args)
            cmd = ["bash", "-lc", ray_wrap_posix]

            os.environ["JOB_NAME"] = cfg.clusters.job_name
            os.environ["PROJECT_NAME"] = cfg.clusters.runai_project
            os.environ["CFG_SAVEDIR"] = posixify(OmegaConf.select(cfg, "paths.outdir"))
            os.environ["TMPDIR"] = posixify(OmegaConf.select(cfg, "paths.tmpdir"))
            os.environ["EXP_NAME"] = f"{OmegaConf.select(cfg,'experiment_name')}.yaml"
            os.environ["PYTHONPATH"] = f"{cfg.paths.python_path}"
            os.environ["NUM_NODES"] = str(cfg.clusters.worker_nodes)

            print("Launching interactive job with command:")
            print("bash -lc", shlex.quote(ray_wrap_posix))
            subprocess.run(cmd, check=True)

        else:
            ray_args = [
                "bash", cfg.paths.ray_script,
                "-c", str(cfg.clusters.cpus_per_worker),
                "-g", str(cfg.clusters.gpus_per_worker),
                "-m", str(cfg.clusters.mem_per_worker),
                "-n", str(cfg.clusters.worker_nodes),
                "-o", str(outdir),
                "-q", str(cfg.clusters.object_store_memory),
                "-t", f"{task}",
            ]

            if cfg.clusters.head_node_gpus not in (None, "", "None"):
                ray_args += ["-y", str(cfg.clusters.head_node_gpus)]
            if cfg.clusters.head_node_cpus not in (None, "", "None"):
                ray_args += ["-z", str(cfg.clusters.head_node_cpus)]

            ray_wrap_posix = " ".join(shlex.quote(str(x)) for x in ray_args)
            main_value = f'bash -lc "{ray_wrap_posix}"'

            runai_jobname = f"{str(cfg.job_type).lower().replace('_','-')}-{uuid.uuid4().hex[:8]}"

            args = [
                "ai","job","submit",
                "--project", cfg.clusters.runai_project,
                "--name",    runai_jobname,
                "--gpus",    str(cfg.clusters.gpus_per_worker),
                "--pods",    str(cfg.clusters.worker_nodes),
                "--data",    f"{cfg.paths.pvc_data_name}={cfg.paths.pvc_data_mount_path}",
                "--base",    image,
                "--repo",    cfg.clusters.repo_url,
                "--evar",    f"NUM_NODES={cfg.clusters.worker_nodes}",
                "--evar",    f"JOB_NAME={runai_jobname}",
                "--evar",    f"CFG_SAVEDIR={posixify(OmegaConf.select(cfg,'paths.outdir'))}",
                "--evar",    f"TMPDIR={posixify(OmegaConf.select(cfg,'paths.tmpdir'))}",
                "--evar",    f"EXP_NAME={OmegaConf.select(cfg,'experiment_name')}.yaml",
                "--evar",    f"PYTHONPATH={cfg.paths.python_path}",
                "--init",    cfg.clusters.init_script,
                "--node-pools", cfg.clusters.node_pool,
                "--main",    main_value,
            ]
            print("Submitting Run:AI job with configuration:")
            print(quote_posix_list(args))
            subprocess.run(args, check=True)

    else:
        raise ValueError(
            f"Unknown launcher type: {cfg.clusters.launcher_type}. "
            f"Please set cfg.clusters.launcher_type to either 'local', 'slurm', 'runai', or 'lsf'."
        )

if __name__ == "__main__":
    main()