import os
import sys
import time
import logging
from typing import List
from tqdm import tqdm
from pathlib import Path

import torch

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.dataloaders import get_dataloader
from utils.context import process_rank, barrier

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_dataloader_scaling(
    csv_path,
    out_base=None,
    legend_label=None,
):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required = ["cpus_per_worker", "gb_per_s_ram", "items_per_s_ram"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    if legend_label is None:
        shape = df["input_shape"].iloc[0] if "input_shape" in df.columns else None
        if shape is not None:
            legend_label = f"Total {shape}"
        else:
            legend_label = "Total"

    agg = (
        df.groupby("cpus_per_worker", as_index=False)
          .agg({"gb_per_s_ram": "mean", "items_per_s_ram": "mean"})
          .sort_values("cpus_per_worker")
    )

    x = agg["cpus_per_worker"].to_numpy()
    y_gbps = agg["gb_per_s_ram"].to_numpy()
    y_sps  = agg["items_per_s_ram"].to_numpy()

    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: GB/s
    ax1.plot(x, y_gbps, "o-", linewidth=2.5, markersize=8, label=legend_label)
    ax1.set_xlabel("CPUs")
    ax1.set_ylabel("Throughput (GB/s)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", frameon=False)
    ax1.set_xticks(x)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)

    # Right: samples/sec
    ax2.plot(x, y_sps, "s-", linewidth=2.5, markersize=8, label="Samples/sec")
    ax2.set_xlabel("CPUs")
    ax2.set_ylabel("Samples/sec")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", frameon=False)
    ax2.set_xticks(x)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()

    if out_base is None:
        out_base = csv_path.with_suffix("")
    out_base = Path(out_base)

    for ext in ("png", "pdf"):
        dpi = 300 if ext == "png" else None
        plt.savefig(out_base.with_suffix(f".{ext}"),
                    bbox_inches="tight", pad_inches=0.25, dpi=dpi)

    plt.close()


def _tensor_nbytes(x: torch.Tensor) -> int:
    return x.element_size() * x.nelement()


def _get_hypercube_shape(cfg) -> List[int]:
    if cfg.dataset_layout_order == "TZYXC":
        return cfg.datasets.input_shape[0], cfg.datasets.input_shape[1], cfg.datasets.input_shape[4]
    else:
        raise ValueError(f"Unsupported dataset layout order: {cfg.dataset_layout_order}")


def _compute_bytes_per_cube(time_size: int,
                            channel_size: int,
                            cube_size: int, 
                            dtype) -> int:
    voxels = (
            time_size * channel_size * cube_size ** 3
    )
    if dtype.lower() == "float16" or dtype.lower() == "fp16" or dtype.lower() == "bfloat16" or dtype.lower() == "bf16":
        return voxels * 2
    elif dtype.lower() == "float32" or dtype.lower() == "fp32":
        return voxels * 4
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _measure_loader(loader, num_batches: int, warmup: int, batch_size: int, bytes_per_sample: int):
    it = iter(loader)

    # Warmup to fill prefetch queues
    warmed = 0
    while warmed < warmup:
        try:
            next(it)
            warmed += 1
        except StopIteration:
            break

    wait_time = 0.0
    total_items = 0
    total_bytes = 0
    measured = 0

    # read_times, collate_times = [], []

    for _ in tqdm(range(num_batches)):
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter()

        # collate_time = batch['metainfo'].get('collate_time', -1)
        # read_time = batch['metainfo'].get('read_time', -1)

        # collate_times.append(collate_time)
        # read_times.append(read_time)

        wait_time += (t1 - t0)
        total_items += batch_size
        total_bytes += bytes_per_sample * batch_size
        measured += 1

    barrier(device_ids=int(os.environ.get("LOCAL_RANK")))

    # for manual inspection
    # logger.info(f"Collate times: {collate_times}")
    # logger.info(f"Read times: {read_times}")

    stats = {
        "num_batches_requested": num_batches,
        "num_batches_measured": measured,
        "total_items": total_items,
        "total_bytes": total_bytes,
        "wait_time_s": wait_time,                       # consumer-side blocking time
        "items_per_s_ram": total_items / wait_time,     # effective delivery rate
        "gb_per_s_ram": (total_bytes / (1024**3)) / wait_time,
        "time_per_item_ms": (wait_time / total_items) * 1e3,
    }
    return stats


def benchmark_dataloader(cfg):
    loader, _, _, _ = get_dataloader(cfg)

    time_size, cube_size, channel_size = _get_hypercube_shape(cfg)

    bytes_per_sample = _compute_bytes_per_cube(
            time_size=time_size,
            channel_size=channel_size,
            cube_size=cube_size,
            dtype=cfg.dataset_dtype
        )

    stats = _measure_loader(
        loader=loader,
        num_batches=cfg.num_batches,
        warmup=cfg.warmup,
        batch_size=cfg.clusters.batch_size_per_gpu,
        bytes_per_sample=bytes_per_sample
    )

    logger.info(f"Benchmark results: {stats}")

    stats.update(
        bytes_per_sample = bytes_per_sample,
        batch_size=cfg.clusters.batch_size_per_gpu,
        gpus_per_worker=cfg.clusters.gpus_per_worker,
        cpus_per_worker=cfg.clusters.cpus_per_gpu,
        dataloader_num_workers=cfg.datasets.num_workers,
        dataloader_prefetch_factor=cfg.datasets.prefetch_factor,
        pin_memory=getattr(cfg.datasets, "pin_memory", None),
        input_shape=getattr(cfg.datasets, "input_shape", None),
        rank=process_rank(),
    )

    csv_pth = Path(cfg.csv_out)
    csv_pth.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([stats])
    df.to_csv(csv_pth, mode="a" if csv_pth.exists() else "w",
              header=not csv_pth.exists(), index=False)
    

    legend = f"Total {tuple(cfg.datasets.input_shape)}, {cfg.dataset_dtype}"
    plot_dataloader_scaling(csv_path=cfg.csv_out,
                        out_base=Path(cfg.csv_out).with_suffix(""),
                        legend_label=legend)

    return stats