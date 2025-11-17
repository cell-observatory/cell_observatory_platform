import hashlib
from pathlib import Path
from typing import Union, Literal, Optional, Tuple

import torch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from data.io import save_file

ArrayLike = Union[np.ndarray, torch.Tensor]


def stable_i64(s: str) -> int:
    d = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "little", signed=True)


def stable_key_owner(roi: int, tile_name: str, world_size: int) -> int:
    h = stable_i64(f"{roi}|{tile_name}")
    return h % world_size


def tile_owner(roi_id: int, tile_name: str, world_size: int) -> int:
    return stable_key_owner(roi_id, tile_name, world_size)


def tile_hash(tile_name: str) -> int:
    return stable_i64(tile_name)


def _normalize_slice(img2d, pmin: float = 1.0, pmax: float = 99.0):
    img2d = np.asarray(img2d)
    lo, hi = np.percentile(img2d, [pmin, pmax])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = img2d.min(), img2d.max()
        if hi <= lo:
            return np.zeros_like(img2d, dtype=np.float32)
    out = (img2d - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def _ensure_numpy_tzyxc(preds: ArrayLike) -> np.ndarray:
    """
    Ensures we have a numpy array in TZYXC:
      - if input is ZYXC -> add T=1
      - if input is TZYXC -> no-op
    """
    if isinstance(preds, torch.Tensor):
        arr = preds.detach().cpu().numpy()
    else:
        arr = np.asarray(preds)

    if arr.ndim == 4:  # Z,Y,X,C
        arr = arr[None, ...]

    if arr.ndim != 5:
        raise ValueError(f"Expected 4D ZYXC or 5D TZYXC, got shape {arr.shape}")

    return arr


def _tzyxc_to_tczyx_or_czyx(
    vol_tzyxc: np.ndarray,
) -> tuple[np.ndarray, str]:
    """
    Convert TZYXC -> TCZYX or CZYX, and return (array, axes).
    - If T > 1: TCZYX
    - If T == 1: CZYX
    """
    T, Z, Y, X, C = vol_tzyxc.shape
    if T > 1:
        # (T,Z,Y,X,C) -> (T,C,Z,Y,X)
        arr = np.transpose(vol_tzyxc, (0, 4, 1, 2, 3))
        axes = "TCZYX"
    else:
        # (1,Z,Y,X,C) -> (C,Z,Y,X)
        arr = np.transpose(vol_tzyxc[0], (3, 0, 1, 2))
        axes = "CZYX"

    return arr, axes


def preds_to_pdf(
    preds_tc_or_czyx: np.ndarray,
    axes: Literal["TCZYX", "CZYX"],
    out_path: Path | str,
    z_step: int = 1,
    pmin: float = 1.0,
    pmax: float = 99.0,
    mip_depth: int = 20,
):
    """
    PDF helper that works on TCZYX or CZYX.
    - If TCZYX: iterate over all T, Z, C
    - If CZYX: treat as T=1
    - Each page: all channels for one (T,Z) as separate panels.
    """
    arr = np.asarray(preds_tc_or_czyx)

    if axes == "TCZYX":
        # (T,C,Z,Y,X) -> (T,Z,Y,X,C)
        T, C, Z, Y, X = arr.shape
        vol_tzyxc = np.transpose(arr, (0, 2, 3, 4, 1))
    elif axes == "CZYX":
        # (C,Z,Y,X) -> (1,Z,Y,X,C)
        C, Z, Y, X = arr.shape
        vol_tzyxc = np.transpose(arr, (1, 2, 3, 0))[None, ...]
        T = 1
    else:
        raise ValueError(f"Unsupported axes for preds_to_pdf: {axes!r}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        for t in range(T):
            for z0 in range(0, Z, z_step):
                z1 = min(z0 + mip_depth, Z)
                # block: (z_block, Y, X, C)
                block = vol_tzyxc[t, z0:z1]  # (z_block, Y, X, C)
                if block.shape[0] == 0:
                    continue

                # Max-intensity projection over z-axis
                mip = block.max(axis=0)  # (Y, X, C)
                _, _, C = mip.shape

                # Channels stacked vertically: C rows x 1 column
                fig, axes_col = plt.subplots(
                    C, 1, figsize=(18, 7 * C), squeeze=False
                )
                axes_col = axes_col[:, 0]

                for c in range(C):
                    img = mip[..., c]
                    img_norm = _normalize_slice(img, pmin=pmin, pmax=pmax)

                    ax = axes_col[c]
                    ax.imshow(img_norm, cmap="gray", interpolation="nearest")
                    ax.set_title(f"T={t}  Z∈[{z0},{z1})  C={c}")
                    ax.axis("off")

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


def _save_tiff_volume(
    predictions: np.ndarray,
    axes: Literal["TCZYX", "CZYX"],
    save_dir: Path,
    name: str,
):
    """
    Save as TIFF, always TCZYX or CZYX.
    """
    if axes not in ("TCZYX", "CZYX"):
        raise ValueError(f"Unsupported axes for TIFF: {axes!r}")

    save_path = save_dir / f"pred_{name}.tiff"
    # OME TIFF typically wants float32
    arr = predictions.astype(np.float32)
    save_file(save_path, arr, axes=axes)


def _save_zarr_volume(
    predictions: np.ndarray,
    axes: Literal["TCZYX", "CZYX"],
    save_dir: Path,
    name: str,
    zarr_chunk_shape: Optional[Tuple[int, ...]] = None,
    zarr_shard_shape: Optional[Tuple[int, ...]] = None,
):
    """
    Save as Zarr, also standardized to TCZYX or CZYX.
    """
    if axes not in ("TCZYX", "CZYX"):
        raise ValueError(f"Unsupported axes for Zarr: {axes!r}")

    save_path = save_dir / f"pred_{name}.zarr"
    arr = predictions.astype(np.float16)

    save_file(
        save_path,
        arr,
        chunk_shape=zarr_chunk_shape,
        shard_cube_shape=zarr_shard_shape,
        input_format=axes,   # axes string encodes layout
        dtype="float16",
    )


def save_predictions(
    name: str,
    predictions: ArrayLike,
    save_dir: Path | str,
    save_as_volume: bool,
    save_as_pdf: bool,
    z_step_pdf: int,
    filetype: Literal["tiff", "zarr"],
    zarr_chunk_shape: Optional[Tuple[int, ...]] = None,
    zarr_shard_shape: Optional[Tuple[int, ...]] = None,
):
    """
    Central helper for saving predictions.
    Inputs:
      - predictions: TZYXC or ZYXC, torch or numpy.
    Standardization:
      - We convert to TCZYX (if T>1) or CZYX (if T==1).
      - That same arr+axes is used for TIFF, Zarr, and PDF.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    predictions = _ensure_numpy_tzyxc(predictions)
    predictions, axes = _tzyxc_to_tczyx_or_czyx(predictions)

    # PDF pages
    if save_as_pdf:
        pdf_path = save_dir / f"pred_{name}_MIP.pdf"
        preds_to_pdf(predictions, axes=axes, out_path=pdf_path, z_step=z_step_pdf)

    # Volume
    if save_as_volume:
        if filetype == "tiff":
            _save_tiff_volume(predictions, axes=axes, save_dir=save_dir, name=name)
        elif filetype == "zarr":
            _save_zarr_volume(
                predictions,
                axes=axes,
                save_dir=save_dir,
                name=name,
                zarr_chunk_shape=zarr_chunk_shape,
                zarr_shard_shape=zarr_shard_shape,
            )
        else:
            raise ValueError(f"Unsupported save format: {filetype!r}")