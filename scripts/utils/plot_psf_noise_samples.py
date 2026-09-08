#!/usr/bin/env python
"""Sanity-check the PSF + sensor-noise augmentation on real synthetic training cubes.

Picks N synthetic cubes (one per tile) from ``api.cube_training``, reads them
with tensorstore, runs ``ConvolveWithPSF`` then ``MixedPoissonGaussianNoise``
exactly as the training preprocessor does, and writes orthoslice PNGs plus a
``stats.txt`` with raw intensity statistics per channel (the noise model treats
its input as photon counts, so the stored intensity scale matters).

CPU only; run inside the project container on a compute node:

    CUDA_VISIBLE_DEVICES= apptainer exec --bind /groups --bind /tmp $SIF \
        python scripts/utils/plot_psf_noise_samples.py --n-cubes 2 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorstore as ts  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

from cell_observatory_platform.data.transforms.noise import MixedPoissonGaussianNoise  # noqa: E402
from cell_observatory_platform.data.transforms.psf import ConvolveWithPSF  # noqa: E402

DATA_ROOT = Path(
    "/groups/betzig/betziglab/CellObservatoryData/forsynthetic/benchmark_tests/data/"
    "iteration_2/synthetic_data_iteration_2"
)
CUBE_SHAPE = (128, 384, 512)  # Z, Y, X
STAGES = ("raw", "psf", "psf+noise")


def query_cubes(n: int) -> list[dict]:
    """One cube per tile (the one with most instances), first n tiles by path."""
    sql = f"""
        select row_to_json(t) from (
          select distinct on (tile_relative_path)
            tile_relative_path, time_start, z_start, y_start, x_start,
            z_size, y_size, x_size, channel_idx, channel_type, localization,
            fluorophore, annotation_type, array_shape, dtype, annotation_count
          from api.cube_training
          where is_synthetic and has_annotations
            and z_size={CUBE_SHAPE[0]} and y_size={CUBE_SHAPE[1]} and x_size={CUBE_SHAPE[2]}
          order by tile_relative_path,
                   (annotation_count->'0'->>'instance')::int desc
        ) t limit {n}"""
    cmd = ["psql", "-A", "-t", "-h", "localhost", "-p", "54322", "-U", "postgres",
           "-d", "postgres", "--command", sql]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def read_cube(row: dict) -> np.ndarray:
    """(Z, Y, X, C) uint16 region of the tile's (T, Z, Y, X, C) zarr3 array."""
    store = ts.open(
        {"driver": "zarr3",
         "kvstore": {"driver": "file", "path": str(DATA_ROOT / row["tile_relative_path"])}},
        read=True,
    ).result()
    assert list(store.shape) == row["array_shape"], (store.shape, row["array_shape"])
    z, y, x = row["z_start"], row["y_start"], row["x_start"]
    region = store[row["time_start"], z:z + row["z_size"], y:y + row["y_size"],
                   x:x + row["x_size"], :]
    return region.read().result()


def channel_stats(a: np.ndarray) -> dict:
    p1, p50, p99 = np.percentile(a, (1, 50, 99))
    return {"dtype": str(a.dtype), "min": float(a.min()), "p1": float(p1),
            "p50": float(p50), "p99": float(p99), "max": float(a.max()),
            "mean": float(a.mean()), "frac_zero": float((a == 0).mean())}


def orthoslices(vol: np.ndarray) -> list[tuple[str, np.ndarray]]:
    zc, yc, xc = [s // 2 for s in vol.shape]
    return [(f"XY z={zc}", vol[zc]), (f"XZ y={yc}", vol[:, yc]), (f"YZ x={xc}", vol[:, :, xc])]


def show_gray(ax, img: np.ndarray, title: str) -> None:
    lo, hi = np.percentile(img, (1, 99.5))
    ax.imshow(img, cmap="gray", vmin=lo, vmax=max(hi, lo + 1e-6), interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def label_cmap(mask: np.ndarray, seed: int) -> tuple[np.ndarray, ListedColormap]:
    """Relabel to dense ids and build a random colormap with 0 = black."""
    ids, dense = np.unique(mask, return_inverse=True)
    colors = np.random.RandomState(seed).rand(len(ids), 3)
    colors[0] = 0  # background (id 0) black; ids are sorted so index 0 is 0
    return dense.reshape(mask.shape), ListedColormap(colors)


def show_labels(ax, mask: np.ndarray, title: str, seed: int) -> None:
    dense, cmap = label_cmap(mask, seed)
    ax.imshow(dense, cmap=cmap, vmin=0, vmax=cmap.N - 1, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def overlay_contours(ax, mask: np.ndarray) -> None:
    edge = find_boundaries(mask, mode="inner")
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[edge] = (1.0, 0.2, 0.2, 0.9)
    ax.imshow(rgba, interpolation="nearest")


def plot_channel(stages: dict[str, np.ndarray], mask: np.ndarray, ch_name: str,
                 out_png: Path, seed: int) -> None:
    """4 rows (raw / psf / psf+noise / mask) x 3 orthoslices for one channel."""
    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    for r, stage in enumerate(STAGES):
        for c, (name, img) in enumerate(orthoslices(stages[stage])):
            show_gray(axes[r, c], img, f"{stage} | {ch_name} | {name}")
    for c, (name, img) in enumerate(orthoslices(mask)):
        show_labels(axes[3, c], img, f"instance mask | {name}", seed)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def plot_overview(stages: dict[str, np.ndarray], mask: np.ndarray, ch_names: list[str],
                  title: str, out_png: Path, seed: int) -> None:
    """Rows: raw / psf / psf+noise / raw+mask contour XY slices; last column: mask labels."""
    C = len(ch_names)
    zc = mask.shape[0] // 2
    fig, axes = plt.subplots(4, C + 1, figsize=(3.2 * (C + 1), 10.5), squeeze=False)
    for r, stage in enumerate(STAGES):
        for c in range(C):
            show_gray(axes[r, c], stages[stage][zc, :, :, c], f"{stage} | {ch_names[c]}")
    for c in range(C):
        show_gray(axes[3, c], stages["raw"][zc, :, :, c], f"raw + mask contour | {ch_names[c]}")
        overlay_contours(axes[3, c], mask[zc])
    show_labels(axes[0, C], mask[zc], f"instance mask XY z={zc}", seed)
    for r in range(1, 4):
        axes[r, C].axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-cubes", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--psf", type=Path,
                    default="/groups/betzig/betziglab/hph/data/psf_lls_gauss_fwhm_z0p80_xy0p32um_at_0p1um.tif")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = query_cubes(args.n_cubes)
    print(f"selected {len(rows)} cubes:")
    for r in rows:
        print(f"  {r['tile_relative_path']} t={r['time_start']} zyx=({r['z_start']},{r['y_start']},{r['x_start']}) "
              f"instances={r['annotation_count']}")

    n_signal = sum(t == "data" for t in rows[0]["channel_type"])
    convolve = ConvolveWithPSF(
        psf=args.psf, pad_type="reflect", input_shape=(*CUBE_SHAPE, n_signal),
        input_pixel_size_um=(0.1, 0.1, 0.1), psf_format="ZYX",
        psf_pixel_size_um=(0.1, 0.1, 0.1), psf_centered=True,
        visualization_dir=str(args.out_dir / "psf"),
    )
    noise = MixedPoissonGaussianNoise(
        quantum_efficiency=0.82, electrons_per_count=0.22,
        sigma_background_noise=40, mean_background_offset=100, seed=args.seed,
    )
    # The PSF class re-plots "before/after" on every call; only wanted once for the PSF itself.
    convolve.visualization_dir = None

    stats_lines = ["cube\tstage\tchannel\tdtype\tmin\tp1\tp50\tp99\tmax\tmean\tfrac_zero"]
    for ci, row in enumerate(rows):
        tag = f"cube{ci}_" + row["tile_relative_path"].replace("/", "_").replace(".zarr", "") \
              + f"_t{row['time_start']}_z{row['z_start']}_y{row['y_start']}_x{row['x_start']}"
        cube = read_cube(row)  # (Z, Y, X, C)
        ctype = row["channel_type"]
        sig_idx = [row["channel_idx"][k] for k, t in enumerate(ctype) if t == "data"]
        mask_idx = [row["channel_idx"][k] for k, t in enumerate(ctype) if t == "mask"]
        assert len(mask_idx) == 1, ctype
        ch_names = [f"c{row['channel_idx'][k]} {row['localization'][k]}/{row['fluorophore'][k]}"
                    for k, t in enumerate(ctype) if t == "data"]
        raw = cube[..., sig_idx]
        mask = cube[..., mask_idx[0]]
        print(f"[{tag}] cube {cube.shape} {cube.dtype}; signal ch {sig_idx}; mask ch {mask_idx[0]} "
              f"({row['annotation_type'][row['channel_idx'].index(mask_idx[0])]}), "
              f"{len(np.unique(mask)) - 1} labels")

        sample = {
            "data_tensor": torch.from_numpy(raw.astype(np.float32))[None],  # (B, Z, Y, X, C)
            "metainfo": {"data_types": {"data_tensor": {"has_time": False}}, "targets": {}},
        }
        sample = convolve(sample)
        after_psf = sample["data_tensor"][0].numpy().copy()
        sample = noise(sample)
        after_noise = sample["data_tensor"][0].numpy()
        stages = {"raw": raw, "psf": after_psf, "psf+noise": after_noise}

        for stage, vol in stages.items():
            for c, name in enumerate(ch_names):
                s = channel_stats(vol[..., c])
                stats_lines.append("\t".join([tag, stage, name, s["dtype"]] + [
                    f"{s[k]:.4g}" for k in ("min", "p1", "p50", "p99", "max", "mean", "frac_zero")]))
        for c, name in enumerate(ch_names):
            plot_channel({k: v[..., c] for k, v in stages.items()}, mask, name,
                         args.out_dir / f"{tag}_ch{sig_idx[c]}.png", args.seed)
        plot_overview(stages, mask, ch_names, f"{tag}  raw dtype={cube.dtype}",
                      args.out_dir / f"{tag}_overview.png", args.seed)
        print(f"[{tag}] wrote PNGs")

    stats = "\n".join(stats_lines)
    (args.out_dir / "stats.txt").write_text(stats + "\n")
    print(stats)


if __name__ == "__main__":
    main()
