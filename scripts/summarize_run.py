import os
import re
from pathlib import Path
import functools, logging

import pandas as pd


# ---------------------------------- Metric/File Names -------------------------------------------


TIMING_FILES = {
    "elapsed_time_ms_forward": "Train_Samples_elapsed_time_ms_forward.csv",
    "elapsed_time_ms_backward": "Train_Samples_elapsed_time_ms_backward.csv",
    "elapsed_time_ms_backward_allreduce": "Train_Samples_elapsed_time_ms_backward_allreduce.csv",
    "elapsed_time_ms_backward_inner": "Train_Samples_elapsed_time_ms_backward_inner.csv",
    "elapsed_time_ms_step": "Train_Samples_elapsed_time_ms_step.csv",
    "train_loss": "Train_Samples_train_loss.csv",
}

FLOPS_METRICS = [
        'world_size', 
        'data_parallel_size', 
        'model_parallel_size',
        'batch_size_per_gpu', 
        'params_per_gpu',
        'params_of_model_=_params_per_gpu_*_mp_size', 'fwd_macs_per_gpu',
        'fwd_flops_per_gpu', 
        'fwd_flops_of_model_=_fwd_flops_per_gpu_*_mp_size',
        'fwd_latency', 
        'fwd_flops_per_gpu_=_fwd_flops_per_gpu_/_fwd_latency',
        'bwd_latency',
        'bwd_flops_per_gpu_=_2_*_fwd_flops_per_gpu_/_bwd_latency',
        'fwd+bwd_flops_per_gpu_=_3_*_fwd_flops_per_gpu_/_(fwd+bwd_latency)',
        'step_latency', 
        'iter_latency',
        'flops_per_gpu_=_3_*_fwd_flops_per_gpu_/_iter_latency',
        'samples/second'
    ]


# --------------------------------- Extract Statistics from Logs --------------------------------------------


def parse_flops_profiler(path: Path) -> pd.DataFrame:
    metrics = {}
    num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    with path.open() as f:
        for line in f:
            m = re.match(r"^(?P<key>[^:]+):\s+(?P<val>.+?)\s*$", line.rstrip())
            if not m:
                continue

            key = m.group("key").strip().lower().replace(" ", "_")
            val = m.group("val").replace(",", "").strip()

            if key not in FLOPS_METRICS:
                continue

            n = num_re.search(val)
            metrics[key] = float(n.group()) if n else val

    return pd.DataFrame([metrics])


def summarise_timings(logdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    dfs = []
    for key, fname in TIMING_FILES.items():
        fpath = logdir / fname
        if not fpath.exists():
            logging.warning(f"Timing file missing: {fpath} - skipping")
            continue

        df = pd.read_csv(fpath, usecols=[0, 1], header=0, names=["global_step", key])
        df[key] = pd.to_numeric(df[key], errors="coerce")
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No timing CSVs in {logdir}")

    merged = functools.reduce(
        lambda left, right: pd.merge(left, right, on="global_step", how="outer"),
        dfs
    ).sort_values("global_step").reset_index(drop=True)

    overall = merged.drop(columns="global_step") \
                    .mean(numeric_only=True) \
                    .to_frame().T
    overall.insert(0, "global_step", "overall")

    return overall


def extract_memory_from_logbook(
    logbook_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    df = pd.read_csv(logbook_path)
    mem_cols = [c for c in df.columns if c.endswith("_mem")]
    epoch_df = df[["epoch", *mem_cols]].copy()
    return epoch_df


def extract_stats(outdir: Path, experiment_name: str):
    # 1) FLOPs profiler
    flops_stats_path = outdir / "logs" / f"{experiment_name}" / "flops_profiler.log"
    flops = parse_flops_profiler(flops_stats_path)

    # 2) timing summaries
    deepspeed_logs_path = outdir / "logs" / f"{experiment_name}"
    timings_merged = summarise_timings(deepspeed_logs_path)

    # 3) memory summaries
    epoch_logbook_path = outdir / "logs" / "scalars" / "epoch_logbook.csv"
    memory = extract_memory_from_logbook(epoch_logbook_path)

    return flops, timings_merged, memory


def summarize_run(results_path, cfg, metadata: dict = None):
    csv_savepath = Path(results_path)
    csv_savepath.parent.mkdir(parents=True, exist_ok=True)

    flops_df, timings_df, memory_df = extract_stats(
        outdir=Path(cfg.paths.outdir),
        experiment_name=cfg.experiment_name,
    )

    timings_df = timings_df.drop(columns=["global_step"], errors="ignore")
    combined = pd.concat([flops_df, timings_df, memory_df], axis=1)
    # TODO: add `total_gpus` to metadata as well for more generality?
    combined["total_gpus"] = cfg.clusters.total_gpus
    if metadata:
        for k, v in metadata.items():
            combined[k] = v

    if csv_savepath.exists():
        combined.to_csv(csv_savepath, mode="a", header=False, index=False)
    else:
        combined.to_csv(csv_savepath, index=False)