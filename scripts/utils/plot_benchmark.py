import os
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import hydra
from omegaconf import DictConfig, OmegaConf
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", verbose=True)
if os.environ["PYTHONPATH"] not in sys.path:
    sys.path.insert(0, os.environ["PYTHONPATH"])

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from cell_observatory_platform.scripts.utils.summarize_run import summarize_run


# -------------------------------------- For Plotting Dataloader Benchmark Statistics ---------------------------------------


def _plot_dataloader_benchmarks(
    df: pd.DataFrame,
    outdir: Path,
    time_cols: list[str] = None,
    fps_cols:  list[str] = None,
    group_by: str = "batch_size",
    plot_best_config: dict | None = None
):
    if time_cols is None:
        time_cols = ["time_disk2ram_per_batch", "time_ram2vram_per_batch", "total_time_per_batch"]
    if fps_cols is None:
        fps_cols  = ["disk2ram_fps", "ram2vram_fps", "total_fps"]

    #­­ optional filtering ­­­­for a specific configuration
    # otherwise take mean of all configurations to plot
    # against the group_by column
    if plot_best_config:
        if not isinstance(plot_best_config, dict):
            raise ValueError("plot_best_config must be a dictionary")

        mask = (
            df[list(plot_best_config)]
              .eq(pd.Series(plot_best_config))
              .all(axis=1)
        )
        df = df[mask]
        if df.empty:
            raise ValueError(f"No rows match {plot_best_config}")

    # ­­­aggregate ­­­­
    grp = (
        df.groupby(group_by, as_index=False)[time_cols + fps_cols]
          .mean()
          .sort_values(group_by)
    )

    # ---------- single figure, two axes ---------------------------------
    
    fig, (ax_t, ax_f) = plt.subplots(
        1, 2, figsize=(14, 5), sharex=True, gridspec_kw=dict(wspace=0.25)
    )
    x  = np.arange(len(grp))

    # ----- LEFT : latency ------------------------------------------------
    
    bw = 0.8 / len(time_cols)
    for i, col in enumerate(time_cols):
        ax_t.bar(
            x + i * bw,
            grp[col],
            width=bw,
            label=col.replace("_", " "),
            edgecolor="black",
        )
    ax_t.set_xticks(x + bw * (len(time_cols) - 1) / 2)
    ax_t.set_xticklabels(grp[group_by].astype(int))
    ax_t.set_xlabel(f"# {group_by}")
    ax_t.set_ylabel("time (s) / batch")
    ax_t.set_title("Data-loading latency")
    ax_t.legend()

    # ----- RIGHT : throughput -------------------------------------------

    bw = 0.8 / len(fps_cols)
    for i, col in enumerate(fps_cols):
        ax_f.bar(
            x + i * bw,
            grp[col],
            width=bw,
            label=col.replace("_", " "),
            edgecolor="black",
        )
    ax_f.set_xticks(x + bw * (len(fps_cols) - 1) / 2)
    ax_f.set_xticklabels(grp[group_by].astype(int))
    ax_f.set_xlabel(f"# {group_by}")
    ax_f.set_ylabel("batch / second")
    ax_f.set_title("Data-loading throughput")
    ax_f.legend()

    # ----- finalize figure ----------------------------------------------

    fig.suptitle("Dataloader benchmark", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"dataloader_benchmark_by_{group_by}.png", dpi=300)
    plt.close(fig)


def plot_dataloader_benchmarks(
        csv_path: str | Path,
        group_by: str = "batch_size",
        plot_best_config: Optional[dict] = None
):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    outdir = csv_path.parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    _plot_dataloader_benchmarks(df, outdir, group_by=group_by, plot_best_config=plot_best_config)


# -------------------------------------- For Plotting Training Run Benchmark Statistics ---------------------------------------


def _plot_training_results(csv_path: Path,
                           xy_pairs: list[tuple[list[str] | str, str]],
                           save_dir: Path,
                           plot_name: str,
                           filters: dict[str, object] | None = None,
                           figsize_per_plot=(6, 4)):
    save_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # optional row filtering
    if filters:
        for col, val in filters.items():
            df = df[df[col].isin(val)] if hasattr(val, "__iter__") and not isinstance(val, str) \
                                         else df[df[col] == val]

    n = len(xy_pairs)
    fig, axes = plt.subplots(
        nrows=n, ncols=1, figsize=(figsize_per_plot[0], figsize_per_plot[1] * n)
    )

    # Matplotlib returns a single Axes when nrows==1
    if n == 1:
        axes = [axes]

    for ax, (x_cols, y) in zip(axes, xy_pairs):
        # handle 1‑col vs N‑col X‑axis
        if isinstance(x_cols, (list, tuple)):
            g = df.groupby(list(x_cols))[y].mean().reset_index()
            x_vals = g[list(x_cols)].astype(str).agg(" | ".join, axis=1)
            ax.plot(x_vals, g[y], marker="o")
            ax.set_xlabel(" & ".join(x_cols))
        else:
            ax.plot(df[x_cols], df[y], marker="o")
            ax.set_xlabel(x_cols)

        ax.set_ylabel(y)
        ax.set_title(f"{y} vs {x_cols if isinstance(x_cols, str) else ', '.join(x_cols)}")
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_dir / plot_name, dpi=300)
    plt.close(fig)


def plot_training_results(cfg: DictConfig, results_path: str | Path, plot_name: str):
    if cfg.get("force_overwrite", False) and (Path(results_path)).exists():
        logger.warning("Force overwriting existing CSV files. "
                       "This will overwrite any previous benchmark results.")
        os.remove(Path(results_path))
    elif (Path(results_path)).exists():
            logger.info("Results CSV already exists. Skipping summarization.")
    else:
        for run_name, run_cfg in cfg.runs.items():
            logger.info(f"Processing run: {run_name}")
            summarize_run(
                results_path=results_path,
                cfg=OmegaConf.load(run_cfg.cfg),
                metadata=run_cfg.get("metadata"),
            )

    if Path(results_path).exists():
        _plot_training_results(
            csv_path=Path(results_path),
            xy_pairs=[tuple(p) for p in cfg.plotting.xy_pairs],
            save_dir=Path(results_path).parent / "plots",
            filters=cfg.plotting.get("filters", None),
            plot_name=plot_name,
        )
    else:
        raise FileNotFoundError(
            f"Results directory {results_path} does not exist. "
        )


# -------------------------------------- --------------------------------------- ---------------------------------------


@hydra.main(config_path="../configs", config_name="plots/benchmark_training_4d.yaml")
def main(cfg: DictConfig):
    if cfg.plot_type == "training_infrastructure_benchmark":
        plot_training_results(cfg=cfg, results_path=cfg.csv_outdir, plot_name=cfg.plotting.name)
    elif cfg.plot_type == "dataloader_benchmark":
        plot_dataloader_benchmarks(
            cfg.csv_outdir,
            group_by=cfg.plotting.group_by,
            plot_best_config=dict(cfg.plotting.plot_best_config)
        )
    else:
        raise NotImplementedError(f"Unknown plot type: {cfg.plot_type}. \
                                    Ensure that the plotting function is implemented.")

if __name__ == "__main__":
    main()