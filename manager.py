import os
import shlex
from pathlib import Path
from subprocess import call, run

import hydra
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("eval", eval)

from training import runner

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
load_dotenv(verbose=True)

def q(x: str) -> str:
    """Shortcut for shlex.quote."""
    return shlex.quote(str(x))

# modify Hydra config on cmd line to use different models
@hydra.main(config_path="configs", config_name="pretrain_mae_local")
def main(cfg: DictConfig):
    # load extra env variables
    load_dotenv(cfg.paths.dotenv_path, verbose=True)

    # print full configuration (for debugging)
    print("\n" + OmegaConf.to_yaml(cfg))

    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name
    print(f"Running with config: {config_name}")

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

    task = f"{cfg.clusters.python_env} {cfg.paths.runner_script} --config-name {config_name}"

    ray_wrap = (
        f" bash {q(cfg.paths.ray_script)} "
        f"-b {q(str(bind))} "
        f"-c {q(cfg.clusters.cpus_per_worker)} "
        f"-e {q(image)} "
        f"-g {q(cfg.clusters.gpus_per_worker)} "
        f"-m {q(cfg.clusters.mem_per_worker)} "
        f"-o {q(str(outdir))} "
        f"-p {q(cfg.clusters.partition)} "
        f"-s {q(str(workspace))} "
        f"-t {q(task)} "
        f"-x {q(cfg.clusters.exclusive)} "
        f"-z {q(cfg.clusters.head_node_cpus)} "
    )

    if cfg.clusters.launcher_type == "local":  # for running jobs on your local workstation without a job scheduler
        print("Running local training job with configuration:")
        runner.main(cfg)
        # print(ray_wrap)
        # call([ray_wrap], shell=True)

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
            sjob_worker_nodes.append(f"--gres=gpu:={cfg.clusters.gpus_per_worker}")
            sjob_worker_nodes.append(f"--mem={cfg.clusters.mem_per_worker}")


        if cfg.clusters.constraint is not None:
            sjob_worker_nodes.append(f"-C '{cfg.clusters.constraint}'")

        if cfg.clusters.nodelist is not None:
            sjob_worker_nodes.append(f"--nodelist='{cfg.clusters.nodelist}'")

        if cfg.clusters.dependency is not None:
            sjob_worker_nodes.append(f"--dependency={cfg.clusters.job_name}")

        if cfg.clusters.timelimit is not None:
            sjob_worker_nodes.append(f" --time={cfg.clusters.timelimit}")

        if cfg.clusters.job_name is not None:
            sjob_worker_nodes.append(f" --job-name={cfg.clusters.job_name}")
            sjob_worker_nodes.append(f"--output={outdir / cfg.clusters.job_name}.log")
        else:
            sjob_worker_nodes.append(f"--job-name=training_job")
            sjob_worker_nodes.append(f"--output={outdir}/training_job.log")

        sjob_worker_nodes.append(f"--export=ALL")
        sjob_worker_nodes.append(f"--wrap={q(ray_wrap)}")

        print("Submitting slurm job with configuration:")
        print(sjob_worker_nodes)
        call(sjob_worker_nodes, shell=True)

    elif cfg.clusters.launcher_type == "lsf":

        # set resources to allocate head node, then the head node will allocate the rest of the worker nodes
        sjob_worker_nodes = [f"/usr/bin/bsub "]
        sjob_worker_nodes.append(f"-q {cfg.clusters.partition}")

        if cfg.clusters.exclusive is not None:
            sjob_worker_nodes.append(f"-x")
        else:
            sjob_worker_nodes.append(f"-n {cfg.clusters.head_node_cpus}")
            sjob_worker_nodes.append(f'-gpu "num={cfg.clusters.gpus_per_worker}:mode=shared"')

        if cfg.clusters.dependency is not None:
            sjob_worker_nodes.append(f' -w "done({cfg.clusters.job_name})"')

        if cfg.clusters.timelimit is not None:
            sjob_worker_nodes.append(f" --We {cfg.clusters.timelimit} ")

        if cfg.clusters.job_name is not None:
            sjob_worker_nodes.append(f"-J {cfg.clusters.job_name}")
            sjob_worker_nodes.append(f"-o {outdir / cfg.clusters.job_name}.log")
        else:
            sjob_worker_nodes.append(f"-J training_job")
            sjob_worker_nodes.append(f"-o {outdir}/training_job.log")

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
        print(sjob_worker_nodes)
        call(sjob_worker_nodes, shell=True)

    else:
        raise ValueError(
            f"Unknown launcher type: {cfg.clusters.launcher_type}. "
            f"Please set cfg.clusters.launcher_type to either 'local', 'slurm', or 'lsf'."
        )

if __name__ == "__main__":
    main()
