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


# -------------------------------------- For Plotting Dataloader Benchmark Statistics ---------------------------------------


def visualize_dataloader_stats(
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
    ax_t.set_ylabel("time (s) / volume")
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
    ax_f.set_ylabel("volume / second")
    ax_f.set_title("Data-loading throughput")
    ax_f.legend()

    # ----- finalize figure ----------------------------------------------

    fig.suptitle("Dataloader benchmark", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"dataloader_benchmark_by_{group_by}.png", dpi=300)
    plt.close(fig)


def plot_dataloader_benchmarks(csv_path: str | Path, 
                               group_by: str = "batch_size", 
                               plot_best_config: Optional[dict] = None
                               ):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    outdir = csv_path.parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    visualize_dataloader_stats(df, outdir, group_by=group_by, plot_best_config=plot_best_config)


# -------------------------------------- For Plotting Training Run Benchmark Statistics ---------------------------------------

# TODO: Finish Implementation

# -------------------------------------- --------------------------------------- ---------------------------------------


@hydra.main(config_path="../configs", config_name="benchmarks/benchmark_dataloaders_4d")
def main(cfg: DictConfig):
    plot_dataloader_benchmarks(cfg.csv_out, 
                               group_by=cfg.plotting.group_by, 
                               plot_best_config=dict(cfg.plotting.plot_best_config))

if __name__ == "__main__":
    main()