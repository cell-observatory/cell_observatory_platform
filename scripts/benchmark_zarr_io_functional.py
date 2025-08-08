from __future__ import annotations

from pathlib import Path
import csv, os, time, pathlib
from typing import Dict, Tuple, List, Any

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate, get_method

from omegaconf import DictConfig, OmegaConf, open_dict
if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)
if not OmegaConf.has_resolver("now"):
    OmegaConf.register_new_resolver("now", lambda fmt: time.strftime(fmt))

from data.io import read_zarr
from data.data_types import TENSORSTORE_DTYPES

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch.utils.data import DataLoader

from utils.common import multiprocess


def slice_hypercube(data_tensor, meta: Dict[str, Any]):
    t = slice(meta["time_start"], meta["time_start"] + meta["time_size"])
    c = slice(0, meta["channel_size"])
    z = slice(meta["z_start"], meta["z_start"] + meta["cube_size"])
    y = slice(meta["y_start"], meta["y_start"] + meta["cube_size"])
    x = slice(meta["x_start"], meta["x_start"] + meta["cube_size"])
    return data_tensor[t, z, y, x, c].read().result()


def _read_cube(args: Tuple[Any, int]) -> float:
    row, cube = args
    meta = dict(
        time_start=row['time_start'],
        time_size=row['time_size'],
        z_start=row['z_start'],
        y_start=row['y_start'],
        x_start=row['x_start'],
        cube_size=cube,
        channel_size=row['channel_size'],
    )

    handle = read_zarr(
        os.path.join(row['server_folder'], row['output_folder'], row['tile_name']),
        dtype=TENSORSTORE_DTYPES["fp16"].value
    )

    t0 = time.perf_counter()
    volume = slice_hypercube(handle, meta)
    read_time = time.perf_counter() - t0

    # del volume, handle
    # gc.collect()
    return read_time


def benchmark_ts_load(cube_size: int, df, cores: int = -1):
    bytes_per_sample = df.iloc[0]["time_size"] * df.iloc[0]["channel_size"] * cube_size**3 * 2  # fp16

    jobs = [(rec, cube_size) for rec in df.to_dict(orient="records")]

    wall_start = time.perf_counter()
    per_read_times = multiprocess(
        jobs=jobs,
        func=_read_cube,
        desc=f"Reading {cube_size}^3 cubes",
        cores=cores,
        unit="cube"
    )
    wall_elapsed = time.perf_counter() - wall_start

    n = len(jobs)

    sps_wall = n / wall_elapsed
    gibps_wall = (sps_wall * bytes_per_sample) / (1024**3)

    # may be interesting to see read time per cube
    samples_per_sec = 1 / per_read_times.mean()
    gibps_read = (samples_per_sec * bytes_per_sample) / (1024**3)
    print(f"Samples per second (1 core): {samples_per_sec} / second")
    print(f"GiB per second (1 core): {gibps_read}")

    print(f"Wall time: {wall_elapsed} seconds")
    print(f"Samples per second (wall): {sps_wall}")
    print(f"GiB per second (wall): {gibps_wall}")

    return (sps_wall, gibps_wall)


def benchmark_ts_load_with_torch(loader) -> Tuple[float, float]:
    dataframe = loader.dataset.hypercubes_dataframe
    time_size = dataframe.iloc[0]["time_size"]
    channel_size = dataframe.iloc[0]["channel_size"]
    cube_size = dataframe.iloc[0]["cube_size"]
    bytes_per_sample = (
        time_size * channel_size * cube_size ** 3 * 2
    )

    with torch.no_grad():
        for _ in range(3):
            next(iter(loader))
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    n_samples = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            bs = batch["data_tensor"].shape[0] if isinstance(batch["data_tensor"], torch.Tensor) \
                else len(batch["data_tensor"])
            n_samples += bs

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - t0

    samples_per_sec = n_samples / elapsed
    gib_per_sec = (samples_per_sec * bytes_per_sample) / (1024 ** 3)

    return samples_per_sec, gib_per_sec


def append_csv(path: pathlib.Path, row: Dict[str, float | int | str]):
    header = row.keys()
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_loader(cfg: DictConfig) -> DataLoader:
    db = instantiate(cfg.datasets.databases)
    
    transforms = [instantiate(t) for t in cfg.datasets.transforms.transforms_list] \
        if cfg.datasets.transforms.transforms_list else None

    dataset = instantiate(
        cfg.datasets.dataset,
        hypercubes_dataframe_path=Path(cfg.datasets.databases.hypercubes_dataframe_path),
        transforms=transforms,
        server_folder_path=cfg.paths.server_folder_path,
        max_rois=cfg.datasets.max_rois,
        max_tiles=cfg.datasets.max_tiles,
        max_hypercubes=cfg.datasets.max_hypercubes,
        hpf_list=cfg.datasets.hpf_list,
        roi_list=cfg.datasets.roi_list,
        tile_list=cfg.datasets.tile_list,
        occupancy_threshold=cfg.datasets.occupancy_threshold
    )

    if isinstance(cfg.datasets.collate_fn, DictConfig):
        collate_fn = instantiate(cfg.datasets.collate_fn)
    else:
        collate_fn = get_method(cfg.datasets.collate_fn)

    db_worker_init_fn = dataset.worker_init_fn

    loader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=cfg.clusters.batch_size_per_gpu,
        shuffle=False,
        pin_memory=True, # False
        num_workers=cfg.datasets.num_workers,
        prefetch_factor=cfg.datasets.prefetch_factor,
        persistent_workers=False,
        sampler=None,
        worker_init_fn=db_worker_init_fn,
        drop_last=True,
    )
    return loader


def plot_benchmark(csv_path: str | pathlib.Path) -> None:
    df = pd.read_csv(csv_path)

    chunk_sizes = sorted(df["zarr_chunk_size"].astype(str).unique())
    cmap = plt.get_cmap("tab10")
    color_map = {cs: cmap(i % cmap.N) for i, cs in enumerate(chunk_sizes)}
    marker_map = {True: "o", False: "s"}

    fig, ax = plt.subplots()

    for (cs, torch_flag), sub in df.groupby(["zarr_chunk_size", "with_torch"]):
        sub = sub.sort_values("n_cpus")
        label = f"chunk {cs} | {'torch' if torch_flag else 'raw'}"
        ax.plot(
            sub["n_cpus"],
            sub["gib_per_sec"],
            marker=marker_map[torch_flag],
            color=color_map[str(cs)],
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel("CPU cores")
    ax.set_ylabel("GiB / s")
    ax.set_title("I/O throughput")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(title="Legend")
    plt.tight_layout()
    plt.savefig(Path(csv_path).with_suffix(".png"))


@hydra.main(config_path="../configs", config_name="benchmarks/benchmark_zarr_io")
def main(cfg: DictConfig):
    dataloader = build_loader(cfg)

    dataset = dataloader.dataset
    df = dataset.hypercubes_dataframe

    if not cfg.with_torch:
        print("Benchmarking Zarr I/O without Torch DataLoader...")
        sps, gbps = benchmark_ts_load(cube_size=cfg.cube_size, df=df, cores=cfg.cores)
        print("Zarr IO:")
        print(f"  Samples per second: {sps}")
        print(f"  GiB per second: {gbps}")

    else:
        print("Benchmarking Zarr I/O with Torch DataLoader...")
        sps, gbps = benchmark_ts_load_with_torch(dataloader)
        print("Torch DataLoader:")
        print(f"  Samples per second: {sps}")
        print(f"  GiB per second: {gbps}")

    row = dict(
        n_cpus=cfg.cores,
        samples_per_sec=sps,
        gib_per_sec=gbps,
        zarr_chunk_size=str(cfg.chunk_size),
        cube_size=str(cfg.cube_size),
        with_torch=cfg.with_torch,
    )

    append_csv(Path(cfg.csv_out), row)
    if cfg.plot:
        plot_benchmark(cfg.csv_out)

if __name__ == "__main__":
    main()