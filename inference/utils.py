import hashlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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


def _normalize_slice(img2d, pmin=1.0, pmax=99.0):
    img2d = np.asarray(img2d)
    lo, hi = np.percentile(img2d, [pmin, pmax])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = img2d.min(), img2d.max()
        if hi <= lo:
            return np.zeros_like(img2d, dtype=np.float32)
    out = (img2d - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def _extract_volume(np_arr: np.ndarray, axes: str) -> np.ndarray:
    arr = np.asarray(np_arr)

    if axes == "ZYXC":
        return arr

    if axes == "CZYX":
        # (C,Z,Y,X) -> (Z,Y,X,C)
        return np.moveaxis(arr, 0, -1)

    if axes == "TCZYX":
        if arr.shape[0] != 1:
            raise ValueError(f"axes=TCZYX but T={arr.shape[0]}; "
                             "call preds_to_pdf AFTER selecting timepoints.")
        arr = arr[0]       # -> (C,Z,Y,X)
        return np.moveaxis(arr, 0, -1)

    raise ValueError(f"Unsupported axes string {axes}")


def preds_to_pdf(
    preds: np.ndarray,
    axes: str,
    out_path,
    z_step: int = 1,
    pmin: float = 1.0,
    pmax: float = 99.0,
):
    vol = _extract_volume(preds, axes)

    Z, Y, X, C = vol.shape

    with PdfPages(out_path) as pdf:
        for z in range(0, Z, z_step):
            slice_z = vol[z]

            fig, axes_row = plt.subplots(
                1, C, figsize=(4*C, 4),
                squeeze=False
            )
            axes_row = axes_row[0]

            for c in range(C):
                img = slice_z[..., c]
                img_norm = _normalize_slice(img, pmin=pmin, pmax=pmax)

                ax = axes_row[c]
                ax.imshow(img_norm, cmap="gray", interpolation="nearest")
                ax.set_title(f"Z={z}  C={c}")
                ax.axis("off")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)