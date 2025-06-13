import shlex
from pathlib import Path
from subprocess import call, run

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

def q(x: str) -> str:
    """Shortcut for shlex.quote."""
    return shlex.quote(str(x))

# modify Hydra config on cmd line to use different models
@hydra.main(config_path="../configs")
def main(cfg: DictConfig):
    # print full configuration (for debugging)
    print("\n" + OmegaConf.to_yaml(cfg))

    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name
    print(f"Running with config: {config_name}")

    outdir = Path(f"{cfg.outdir}/{config_name}").resolve()
    outdir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory for training job: {outdir}")

    # bind path is --bind <host_path> : <container_path>
    if hasattr(cfg.clusters, "mount_path") and cfg.clusters.mount_path is not None:
        bind = f'{cfg.clusters.mount_path}:{cfg.clusters.mount_path}'
        workspace = f'{cfg.clusters.repo_path}:/workspace/{cfg.clusters.repo_name}'

    assert (cfg.clusters.apptainer_image is None) != (cfg.clusters.docker_image is None), \
        "Either apptainer_image or docker_image must be specified, but not both"

    task = f"{cfg.clusters.python_env} {cfg.clusters.script} --config-name {config_name}"

    ray_wrap = (
        f" bash {q(cfg.clusters.ray_script)} "
        f"-b {q(str(bind))} "
        f"-c {q(cfg.clusters.cpus_per_worker)} "
        f"-e {q(cfg.clusters.python_env)}"
        f"-g {q(cfg.clusters.gpus_per_worker)} "
        f"-m {q(cfg.clusters.mem_per_worker)} "
        f"-o {q(str(outdir))} "
        f"-p {q(cfg.clusters.partition)} "
        f"-s {q(str(workspace))} "
        f"-t {q(task)} "
        f"-x {q(cfg.clusters.exclusive)} "
        f"-z {q(cfg.clusters.head_node_cpus)} "
    )

    if cfg.clusters.launcher_type == "local":  # for running jobs on your local workstation with a job scheduler
        print("Submitting local training job with configuration:")
        print(ray_wrap)
        call([ray_wrap], shell=True)

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
