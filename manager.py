import os
import shlex
import subprocess
from pathlib import Path
from subprocess import call, run

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("eval", eval)

from training import runner
from utils.container import get_container_info

# Update environment variables
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["NCCL_DEBUG"] = "TRACE"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "GRAPH"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["NCCL_CUMEM_ENABLE"] = "0"
os.environ["NCCL_CROSS_NIC"] = "1"
os.environ["NCCL_P2P_LEVEL"] = "NVL"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3600"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
load_dotenv(Path(__file__).parent / ".env", verbose=True)

def q(x: str) -> str:
    """Shortcut for shlex.quote."""
    return shlex.quote(str(x))

# modify Hydra config on cmd line to use different models
@hydra.main(config_path="configs", config_name="test_pretrain_4d_mae_local")
def main(cfg: DictConfig):

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

    else: # running in a docker/apptainer
        [print(f"\t{k}: {v}") for k, v in container_info['container_details'].items()]

        for k in ['outdir', 'ray_script', 'runner_script', 'dotenv_path']:
            cfg.paths[k] = cfg.paths[k].replace(cfg.paths.repo_path, cfg.paths.workdir)

    # load extra env variables
    # assert cfg.paths.dotenv_path is not None and Path(cfg.paths.dotenv_path).exists(), \
    #     f"Missing dotenv path: {cfg.paths.dotenv_path}"
    load_dotenv(cfg.paths.dotenv_path, verbose=True)

    # print full configuration (for debugging)
    print("\n" + OmegaConf.to_yaml(cfg))

    hydra_cfg = HydraConfig.get()

    config_name = hydra_cfg.job.config_name
    print(f"Running with config: {config_name}")

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
        sjob_worker_nodes = [f"/usr/bin/bsub "]
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
        sjob_worker_nodes.append(f"{q(ray_wrap)}")

        print("Checking available Janelia cluster resources...")
        try:
            aval = run(
                ['bash', 'check_available_janelia_nodes.sh'],
                check=True,
                text=True,
                shell=True
            )
            print("Requested resource are available now!")
        except Exception as e:
            print(f"Error running resource check: {e}")

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
