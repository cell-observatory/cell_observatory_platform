"""Inference visualization helpers.

Architecture (kept deliberately layered so new plot types are cheap to add):

  1. Shared low-level utils  -- ``_ensure_numpy*``, :func:`normalize_slice`,
     :func:`mip_project`, :func:`instance_label_cmap`. These are format/layout
     primitives with no plotting policy and are reused by every higher-level helper.

  2. Per-plot-type helpers   -- :func:`save_prediction_plots` (dense MIP PDF),
     :func:`save_instance_predictions` (box/mask overlays), :func:`save_semantic_predictions`
     (per-class semantic panels), :func:`save_feature_visualizations` (PCA /
     patch-cosine feature maps), and :func:`save_bbox_overlay` (predicted boxes).
     Each owns exactly one artifact type and shares the layer-1 primitives.

  3. Controller / dispatch   -- lives in ``inference/visualizer.py``: each handler
     name (the config key under ``viz_worker.handler_configs``) is a registered
     ``viz_handler`` wrapping one layer-2 helper, and ``VizWorker`` is the runtime
     dispatcher (buffer materialization + batch unpack + per-sample naming).
     Adding a plot type = add a layer-2 helper here + register a ``viz_handler``
     wrapper there; the config surface stays declarative.

All plotting uses the object-oriented matplotlib API (``matplotlib.figure.Figure``
directly, never ``pyplot``): pyplot's global figure registry is not thread-safe and
these helpers run inside the viz worker's thread pool.
"""

from tqdm import tqdm
import hashlib
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union, List
from torch import Tensor
import torch
import numpy as np
import torch.nn.functional as F

from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm


from cell_observatory_platform.data.io import save_file
from cell_observatory_platform.data.data_types import OutputKind
from cell_observatory_platform.data.datasets.buffers import BufferManager
from cell_observatory_platform.data.structures import convert_bbox_format

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


# ============================================================================
# Layer 1: shared low-level utils (normalize / MIP / colormap / volume save)
# ============================================================================


def normalize_slice(img2d, pmin: float = 1.0, pmax: float = 99.0):
    """Percentile-normalize a 2D image to ``[0, 1]`` for display.

    Falls back to min/max if the percentile window is degenerate and to an all-
    zero image if the slice is constant. Shared by every PDF generator.
    """
    img2d = np.asarray(img2d)
    lo, hi = np.percentile(img2d, [pmin, pmax])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = img2d.min(), img2d.max()
        if hi <= lo:
            return np.zeros_like(img2d, dtype=np.float32)
    out = (img2d - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def mip_project(block: np.ndarray, axis: int = 0) -> np.ndarray:
    """Max-intensity projection over ``axis`` (default: leading z-block axis)."""
    return np.asarray(block).max(axis=axis)


def _ensure_numpy(x: ArrayLike):
    """Convert torch/numpy/list-like to numpy on CPU."""
    if isinstance(x, np.ndarray):
        return x

    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return np.asarray(x)

        if len(x) == 1:
            return _ensure_numpy(x[0])

        xs = [_ensure_numpy(v) for v in x]
        try:
            return np.stack(xs, axis=0)
        except Exception:
            return np.asarray(xs, dtype=object)

    return np.asarray(x)

def _ensure_numpy_tzyxc(preds: ArrayLike) -> np.ndarray:
    """
    Ensures we have a numpy array in TZYXC:
      - if input is ZYXC -> add T=1
      - if input is TZYXC -> no-op
    """
    arr = _ensure_numpy(preds)

    if arr.ndim == 3:  # ZYX -> add T=1, C=1
        arr = arr[None, ..., None]
    elif arr.ndim == 4:  # ZYXC -> add T=1
        arr = arr[None, ...]
    elif arr.ndim == 5:  # TZYXC -> no-op
        pass
    else:
        raise ValueError(f"Expected 3D ZYX, 4D ZYXC, or 5D TZYXC, got shape {arr.shape}")

    return arr

def _ensure_numpy_tnc(preds: ArrayLike) -> np.ndarray:
    """
    Ensures we have a numpy array in TNC:
      - if input is NC -> add T=1
      - if input is TNC -> no-op
    """
    arr = _ensure_numpy(preds)
    if arr.ndim == 2:  # NC -> add T=1
        arr = arr[None, ...]
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


def _to_tzyxc(arr: np.ndarray, axes: str) -> np.ndarray:
    """Convert TCZYX or CZYX -> TZYXC."""
    if axes == "TCZYX":
        T, C, Z, Y, X = arr.shape
        return np.transpose(arr, (0, 2, 3, 4, 1))
    elif axes == "CZYX":
        C, Z, Y, X = arr.shape
        return np.transpose(arr, (1, 2, 3, 0))[None, ...]
    else:
        raise ValueError(f"Unsupported axes for _to_tzyxc: {axes}")


def preds_dict_to_pdf(
    preds_tc_or_czyx: dict[str, np.ndarray],
    axes_map: dict[str, Literal["TCZYX", "CZYX"]],
    out_path: Path | str,
    z_step: int = 15,
    pmin: float = 1.0,
    pmax: float = 99.0,
    mip_depth: int = 20,
):
    """
    Multi-output PDF:
      - preds_tc_or_czyx: {data_type_name: array in TCZYX or CZYX}
      - Same T,Z,Y,X for all entries.
      - Layout: rows = channels, cols = data types.
    """
    vols_tzyxc = {}
    T_ref = Z_ref = Y_ref = X_ref = None
    max_C = 0

    for name, arr in preds_tc_or_czyx.items():
        arr = np.asarray(arr)
        axes = axes_map[name]
        vol = _to_tzyxc(arr, axes)
        T, Z, Y, X, C = vol.shape

        if T_ref is None:
            T_ref, Z_ref, Y_ref, X_ref = T, Z, Y, X
        else:
            if (T, Z, Y, X) != (T_ref, Z_ref, Y_ref, X_ref):
                raise ValueError(
                    f"All predictions must share T,Z,Y,X. "
                    f"Got {(T,Z,Y,X)} vs {(T_ref,Z_ref,Y_ref,X_ref)} for {name}."
                )

        vols_tzyxc[name] = vol
        max_C = max(max_C, C)

    if T_ref is None:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(preds_tc_or_czyx.keys())
    num_types = len(names)

    # One figure reused across all (t, z) pages (max_C x num_types is fixed).
    fig = Figure(figsize=(5 * num_types, 4 * max_C))
    axes_grid = fig.subplots(max_C, num_types, squeeze=False)

    with PdfPages(out_path) as pdf:
        for t in range(T_ref):
            for z0 in range(0, Z_ref, z_step):
                z1 = min(z0 + mip_depth, Z_ref)
                if z1 <= z0:
                    continue

                for row in range(max_C):
                    for col in range(num_types):
                        axes_grid[row, col].cla()

                for col, name in enumerate(names):
                    vol = vols_tzyxc[name]
                    block = vol[t, z0:z1]
                    if block.shape[0] == 0:
                        for row in range(max_C):
                            axes_grid[row, col].axis("off")
                        continue

                    mip = mip_project(block)  # (Y,X,C)
                    _, _, C = mip.shape

                    for c in range(max_C):
                        ax = axes_grid[c, col]
                        if c < C:
                            img = mip[..., c]
                            img_norm = normalize_slice(img, pmin=pmin, pmax=pmax)
                            ax.imshow(img_norm, cmap="gray", interpolation="nearest")
                            ax.set_title(f"{name} | T={t}  Z∈[{z0},{z1})  C={c}")
                            ax.axis("off")
                        else:
                            ax.axis("off")

                fig.tight_layout()
                pdf.savefig(fig)


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

    # One figure reused across all (t, z) pages (C is fixed): clearing axes each
    # page avoids reallocating a figure per slice.
    C = vol_tzyxc.shape[-1]
    fig = Figure(figsize=(18, 7 * C))
    axes_col = fig.subplots(C, 1, squeeze=False)
    axes_col = axes_col[:, 0]

    with PdfPages(out_path) as pdf:
        for t in range(T):
            for z0 in range(0, Z, z_step):
                z1 = min(z0 + mip_depth, Z)
                # block: (z_block, Y, X, C)
                block = vol_tzyxc[t, z0:z1]  # (z_block, Y, X, C)
                if block.shape[0] == 0:
                    continue

                # Max-intensity projection over z-axis
                mip = mip_project(block)  # (Y, X, C)

                for c in range(C):
                    img = mip[..., c]
                    img_norm = normalize_slice(img, pmin=pmin, pmax=pmax)

                    ax = axes_col[c]
                    ax.cla()
                    ax.imshow(img_norm, cmap="gray", interpolation="nearest")
                    ax.set_title(f"T={t}  Z∈[{z0},{z1})  C={c}")
                    ax.axis("off")

                fig.tight_layout()
                pdf.savefig(fig)


def pca_reduce(
    vol_tzyxc: np.ndarray,   # (T,Z,Y,X,C)
    k: int = 3,
    sample_voxels: Optional[int] = 50000,
    seed: int = 0,
    fit: Literal["per_t", "per_tz", "global"] = "per_tz",
    chunk: int = 1_000_000,
    ignore_all_zero: bool = False,
) -> np.ndarray:
    """
    Reduce C -> k via PCA, returning float32 (T,Z,Y,X,k).
    Minimal and fast-ish: torch.pca_lowrank on a sample, chunked projection.
    """
    vol_tzyxc = np.asarray(vol_tzyxc)
    if vol_tzyxc.ndim != 5:
        raise ValueError(f"Expected (T,Z,Y,X,C), got {vol_tzyxc.shape}")

    T, Z, Y, X, C = vol_tzyxc.shape

    def _fit_basis(flat_np: np.ndarray, seed_offset: int):
        # NOTE: if we remove background hypercubes, we may remove these
        #       tokens from the PCA fit
        if ignore_all_zero:
            nz = np.any(flat_np != 0, axis=1)
            flat_np = flat_np[nz] if int(nz.sum()) > k else flat_np
        N = flat_np.shape[0]

        # sample or all (avoid allocating a giant permutation)
        if sample_voxels is None or sample_voxels >= N:
            Xs = torch.from_numpy(flat_np)
        else:
            idx = np.random.default_rng(seed + seed_offset).choice(N, size=sample_voxels, replace=False)
            Xs = torch.from_numpy(flat_np[idx])

        if Xs.shape[0] <= k:
            raise ValueError(f"PCA needs >k samples; got {Xs.shape[0]} for k={k}")

        mu = Xs.mean(dim=0, keepdim=True)         # (1,C)
        Xc = Xs - mu                              # (n,C)
        _, _, V = torch.pca_lowrank(Xc, q=k, center=False)  # V: (C,q)
        W = V[:, :k].contiguous()                 # (C,k)
        return mu.squeeze(0).float(), W.float()   # (C,), (C,k)

    out = np.empty((T, Z, Y, X, k), dtype=np.float32)

    if fit == "global":
        flat_all = vol_tzyxc.reshape(-1, C).astype(np.float32, copy=False)
        mu_g, W_g = _fit_basis(flat_all, seed_offset=999)
    else:
        mu_g, W_g = None, None

    for t in range(T):
        if fit == "per_tz":
            for z in range(Z):
                flat = vol_tzyxc[t, z].reshape(-1, C).astype(np.float32, copy=False)  # (Y*X,C)
                mu, W = _fit_basis(flat, seed_offset=(t * 1000003 + z))

                flat_t = torch.from_numpy(flat)
                N = flat.shape[0]
                out_flat = out[t, z].reshape(-1, k)
                for s in range(0, N, chunk):
                    e = min(s + chunk, N)
                    proj = (flat_t[s:e] - mu) @ W
                    out_flat[s:e] = proj.numpy()
        else:
            flat = vol_tzyxc[t].reshape(-1, C).astype(np.float32, copy=False)  # (Z*Y*X,C)

            if fit == "per_t":
                mu, W = _fit_basis(flat, seed_offset=t)
            else:
                mu, W = mu_g, W_g

            flat_t = torch.from_numpy(flat)  # CPU tensor
            N = flat.shape[0]
            out_flat = out[t].reshape(-1, k)

            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                proj = (flat_t[s:e] - mu) @ W
                out_flat[s:e] = proj.numpy()

    return out


def patch_cosine_sim_maps(
    feat_tzyxc: np.ndarray,              # (Tf,Zf,Yf,Xf,Cf) feature volume (patch grid)
    gt_tzyxc: np.ndarray,                # (Tg,Zg,Yg,Xg,Cg) used only to pick reference pixel per GT channel
    *,
    stride_zyx: tuple[int, int, int] = (1, 1, 1),
    sample_p: float = 99.0,
    eps: float = 1e-8,
    ignore_all_zero: bool = True,
) -> np.ndarray:
    """
    Returns sim maps on the *GT grid*: (T,Zg,Yg,Xg,Cg).

    Steps (per t,z,gt_channel):
      1) pick a reference GT pixel (y,x) from gt intensity at that z
      2) map that GT (z,y,x) to feature-grid (zf,yf,xf) via stride_zyx
      3) cosine(sim) on feature slice (t,zf,:,:)
      4) upsample sim slice back to GT (Yg,Xg) by nearest (and repeat in z blocks by stride)
    """
    feat_tzyxc = np.asarray(feat_tzyxc)
    gt_tzyxc = np.asarray(gt_tzyxc)

    if feat_tzyxc.ndim != 5 or gt_tzyxc.ndim != 5:
        raise ValueError(f"Expected feat and gt in TZYXC, got {feat_tzyxc.shape} and {gt_tzyxc.shape}")

    Tf, Zf, Yf, Xf, Cf = feat_tzyxc.shape
    Tg, Zg, Yg, Xg, Cg = gt_tzyxc.shape

    if Tf != Tg:
        raise ValueError(f"feat and gt must share T (or you must align time first). Got Tf={Tf} Tg={Tg}")

    sz, sy, sx = stride_zyx
    if sz <= 0 or sy <= 0 or sx <= 0:
        raise ValueError(f"stride_zyx must be positive ints, got {stride_zyx}")

    # normalize features for cosine: f / ||f||
    f = torch.from_numpy(feat_tzyxc.astype(np.float32, copy=False))  # (T,Zf,Yf,Xf,Cf)
    if ignore_all_zero:
        bg = torch.all(f == 0, dim=-1, keepdim=True)                 # (T,Zf,Yf,Xf,1)
    else:
        bg = None

    f_norm = torch.linalg.norm(f, dim=-1, keepdim=True).clamp_min(eps)
    f_unit = f / f_norm
    if bg is not None:
        f_unit = torch.where(bg, torch.zeros_like(f_unit), f_unit)

    # output on GT grid
    sims_gt = torch.empty((Tg, Zg, Yg, Xg, Cg), dtype=torch.float32)

    gt = gt_tzyxc.astype(np.float32, copy=False)

    for t in range(Tg):
        for z in range(Zg):
            # map GT z -> feature zf
            zf_idx = min(z // sz, Zf - 1)

            # feature slice for this (t,zf): (Yf*Xf, Cf)
            fz = f_unit[t, zf_idx].reshape(-1, Cf)

            for cg in range(Cg):
                img = gt[t, z, :, :, cg]
                flat_img = img.reshape(-1)

                thr = np.percentile(flat_img, sample_p)
                cand = np.where(flat_img >= thr)[0]
                if cand.size == 0:
                    idx_gt = int(flat_img.argmax())
                else:
                    idx_gt = int(cand[flat_img[cand].argmax()])

                yg = idx_gt // Xg
                xg = idx_gt % Xg

                # map GT (y,x) -> feature (yf,xf)
                yf_idx = min(yg // sy, Yf - 1)
                xf_idx = min(xg // sx, Xf - 1)
                idx_feat = yf_idx * Xf + xf_idx

                ref = fz[idx_feat]  # (Cf,) already unit
                sim_feat = (fz @ ref).reshape(Yf, Xf)  # (Yf,Xf)

                # upsample feature sim map -> GT (Yg,Xg)
                sim_gt_2d = F.interpolate(
                    sim_feat[None, None, :, :], size=(Yg, Xg), mode="nearest"
                )[0, 0]
                sims_gt[t, z, :, :, cg] = sim_gt_2d

    return sims_gt.numpy()


# ============================================================================
# Layer 2: Per-plot-type helpers
# ============================================================================


def save_feature_visualizations(
    name: str,
    predictions: Dict[str, ArrayLike],
    save_dir: Path | str,
    *,
    gt_key: str = "data_tensor",
    feat_key: Optional[str] = None,
    z_step_pdf: int = 8,
    pmin: float = 1.0,
    pmax: float = 99.0,
    # PCA knobs
    k: int = 3,
    seed: int = 0,
    fit: Literal["per_t", "global", "per_tz"] = "per_tz",
    sample_voxels: Optional[int] = 50000,
    chunk: int = 1_000_000,
    upsample_to_gt: bool = True,
    # NEW: feature viz mode + patch-sim knobs
    viz: Literal["pca", "patch_cosine"] = "pca",
    stride_zyx: tuple[int, int, int] = (1, 1, 1),
    patch_sample_p: float = 99.0,
    patch_ignore_all_zero: bool = True,
):
    """
    PDF generator:
      viz="pca": top = PCA RGB of feature volume at (t,z), below = all GT channels at (t,z)
      viz="patch_cosine": top = cosine-sim maps (one per GT channel), below = GT channels
    Writes: pred_{name}_FEATURES.pdf
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if gt_key not in predictions:
        raise KeyError(f"Missing gt_key={gt_key!r} in predictions keys={list(predictions.keys())}")
    if feat_key is None:
        raise ValueError("feat_key must be specified for feature visualization")

    gt = _ensure_numpy_tzyxc(predictions[gt_key])       # (Tg,Zg,Yg,Xg,Cg)
    feat = _ensure_numpy_tzyxc(predictions[feat_key])   # (Tf,Zf,Yf,Xf,Cf)

    Tg, Zg, Yg, Xg, Cg = gt.shape

    if viz == "patch_cosine":
        sim_maps = patch_cosine_sim_maps(
            feat,
            gt,
            stride_zyx=stride_zyx,
            sample_p=patch_sample_p,
            ignore_all_zero=patch_ignore_all_zero,
        )  # (Tg,Zg,Yg,Xg,Cg)

        Tf, Zf = Tg, Zg  # for loop bounds below
        feat_rgb = None

    else:
        bg = np.all(feat == 0, axis=-1)  # (Tf,Zf,Yf,Xf) True=background

        # PCA -> (Tf,Zf,Yf,Xf,3)
        feat_rgb = pca_reduce(
            feat,
            k=k,
            sample_voxels=sample_voxels,
            seed=seed,
            fit=fit,
            chunk=chunk,
        )

        # optional sigmoid scaling for nicer visualization (as in DINO)
        feat_rgb = 1.0 / (1.0 + np.exp(-2.0 * feat_rgb))  # sigmoid(2*x)
        feat_rgb[bg] = 0.0  # keep background black

        Tf, Zf, Yf, Xf, _ = feat_rgb.shape

        # upsample PCA RGB to GT spatial size (Z,Y,X)
        if upsample_to_gt and (Zf, Yf, Xf) != (Zg, Yg, Xg):
            ft = torch.from_numpy(feat_rgb).permute(0, 4, 1, 2, 3)  # (T,3,Z,Y,X)
            ft = F.interpolate(ft, size=(Zg, Yg, Xg), mode="nearest")
            feat_rgb = ft.permute(0, 2, 3, 4, 1).cpu().numpy()
            Tf, Zf, _, _, _ = feat_rgb.shape

    pdf_path = save_dir / f"pred_{name}_FEATURES.pdf"
    with PdfPages(pdf_path) as pdf:
        for t in range(min(Tg, Tf)):
            for z in range(0, min(Zg, Zf), z_step_pdf):

                if viz == "patch_cosine":
                    fig = Figure(figsize=(10, 3 * (2 * Cg)))
                    axes = fig.subplots(2 * Cg, 1, squeeze=False)
                    axes = axes[:, 0]

                    # --- cosine sim maps (top) ---
                    for c in range(Cg):
                        ax = axes[c]
                        sim = sim_maps[t, z, :, :, c]  # (Yg,Xg), roughly in [-1,1]
                        sim_vis = np.clip((sim + 1.0) * 0.5, 0.0, 1.0)  # -> [0,1]
                        ax.imshow(sim_vis, cmap="viridis", interpolation="nearest")
                        ax.set_title(f"{name} | {feat_key} cosine-sim | t={t} z={z} c={c} (stride={stride_zyx})")
                        ax.axis("off")

                    # --- GT channels (bottom) ---
                    for c in range(Cg):
                        ax = axes[Cg + c]
                        img = gt[t, z, :, :, c]
                        ax.imshow(normalize_slice(img, pmin=pmin, pmax=pmax), cmap="gray", interpolation="nearest")
                        ax.set_title(f"{gt_key} | t={t} z={z} c={c}")
                        ax.axis("off")

                else:
                    # original PCA layout: 1 (rgb) + Cg rows
                    fig = Figure(figsize=(10, 3 * (1 + Cg)))
                    axes = fig.subplots(1 + Cg, 1, squeeze=False)
                    axes = axes[:, 0]

                    ax0 = axes[0]
                    rgb = feat_rgb[t, z]  # (Yg,Xg,3)
                    ax0.imshow(rgb, interpolation="nearest")
                    ax0.set_title(f"{name} | {feat_key} PCA | t={t} z={z}")
                    ax0.axis("off")

                    for c in range(Cg):
                        ax = axes[1 + c]
                        img = gt[t, z, :, :, c]
                        ax.imshow(normalize_slice(img, pmin=pmin, pmax=pmax), cmap="gray", interpolation="nearest")
                        ax.set_title(f"{gt_key} | t={t} z={z} c={c}")
                        ax.axis("off")

                fig.tight_layout()
                pdf.savefig(fig)

    print(f"[save_feature_visualizations] wrote {pdf_path}")


def save_prediction_plots(
    name: str,
    predictions: ArrayLike | dict[str, ArrayLike],
    save_tensors: List[str],
    save_dir: Path | str,
    z_step_pdf: int,
    pmin: float = 1.0,
    pmax: float = 99.0,
):
    """
    Plotting helper: render predictions as a MIP PDF.

    Volume writing (tiff/zarr) is NOT done here -- it lives in the save path
    (``inference/saver.py`` -> ``data/io.py``). This helper is plot-only.

    Inputs:
      - predictions:
          * single TZYXC or ZYXC array (torch or numpy), OR
          * dict[name -> TZYXC/ZYXC array].
    Standardization:
      - Each array is converted to TCZYX (if T>1) or CZYX (if T==1).
      - That same arr+axes is used for the PDF pages.
      - If predictions is a dict, one PDF with all entries combined
        (columns = outputs).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[save_prediction_plots] Plotting predictions for tile {name}...")

    # Dict case: multi-output
    if isinstance(predictions, dict):
        arr_map: dict[str, np.ndarray] = {}
        axes_map: dict[str, str] = {}

        for key in save_tensors:
            arr = predictions[key]
            arr_tzyxc = _ensure_numpy_tzyxc(arr)
            arr_tc_or_czyx, axes = _tzyxc_to_tczyx_or_czyx(arr_tzyxc)
            arr_map[key] = arr_tc_or_czyx
            axes_map[key] = axes

        # Single PDF with all outputs on the same pages
        pdf_path = save_dir / f"pred_{name}_MIP.pdf"
        print(f"[save_prediction_plots] Writing PDF to: {pdf_path}")
        preds_dict_to_pdf(
            arr_map,
            axes_map,
            out_path=pdf_path,
            z_step=z_step_pdf,
            pmin=pmin,
            pmax=pmax,
        )
        return

    # Single-array case
    arr_tzyxc = _ensure_numpy_tzyxc(predictions)
    arr_tc_or_czyx, axes = _tzyxc_to_tczyx_or_czyx(arr_tzyxc)

    pdf_path = save_dir / f"pred_{name}_MIP.pdf"
    preds_to_pdf(arr_tc_or_czyx, axes=axes, out_path=pdf_path, z_step=z_step_pdf)


def _boxes_in_z_slice(boxes_xyzxyz: np.ndarray, z: int) -> np.ndarray:
    """
    Keep boxes whose z-extent intersects slice plane at (z + 0.5).
    boxes are xyzxyz in local voxel coords.
    """
    b = np.asarray(boxes_xyzxyz, dtype=np.float32)
    if b.size == 0:
        return b.reshape(0, 6)
    plane = float(z) + 0.5
    keep = (plane >= b[:, 2]) & (plane < b[:, 5])
    return b[keep]


def _boxes_in_y_slice(boxes_xyzxyz: np.ndarray, y: int) -> np.ndarray:
    """
    Keep boxes whose y-extent intersects slice plane at (y + 0.5).
    boxes are xyzxyz in local voxel coords.
    """
    b = np.asarray(boxes_xyzxyz, dtype=np.float32)
    if b.size == 0:
        return b.reshape(0, 6)
    plane = float(y) + 0.5
    keep = (plane >= b[:, 1]) & (plane < b[:, 4])
    return b[keep]


def _boxes_in_x_slice(boxes_xyzxyz: np.ndarray, x: int) -> np.ndarray:
    """
    Keep boxes whose x-extent intersects slice plane at (x + 0.5).
    boxes are xyzxyz in local voxel coords.
    """
    b = np.asarray(boxes_xyzxyz, dtype=np.float32)
    if b.size == 0:
        return b.reshape(0, 6)
    plane = float(x) + 0.5
    keep = (plane >= b[:, 0]) & (plane < b[:, 3])
    return b[keep]


def _region_str(region: Optional[Dict[str, Any]]) -> str:
    """
    Human-readable crop descriptor for figure titles.
    Expects region like:
      {"roi":..., "tile_name":..., "coords": (t0,t1,z0,z1,y0,y1,x0,x1), "coord_frame":"voxel", ...}
    """
    if not region:
        return ""
    coords = region.get("coords", None)
    if coords is None:
        return ""
    t0, t1, z0, z1, y0, y1, x0, x1 = coords
    roi = region.get("roi", None)
    tile = region.get("tile_name", None)
    frame = region.get("coord_frame", None)

    parts = []
    if roi is not None:
        parts.append(f"roi={roi}")
    if tile is not None:
        parts.append(f"tile={tile}")
    parts.append(f"crop t[{t0},{t1}) z[{z0},{z1}) y[{y0},{y1}) x[{x0},{x1})")
    if frame is not None:
        parts.append(f"frame={frame}")
    return " | " + " ".join(parts)


def _instance_cmap(n: int):
    """Transparent background + one unique color per instance index 1..n."""
    if n <= 20:
        cols = colormaps["tab20"](np.linspace(0, 1, max(n, 1)))
    else:
        cols = colormaps["hsv"](np.linspace(0, 1, n, endpoint=False))
    cols = np.vstack([[0, 0, 0, 0], cols])  # label 0 transparent
    return ListedColormap(cols), BoundaryNorm(np.arange(n + 2) - 0.5, n + 1)


def _empty_label_render():
    return np.zeros((1, 1), np.int32), ListedColormap([[0, 0, 0, 0]]), BoundaryNorm([-0.5, 0.5], 1)


def _label_and_cmap_from_instance_masks(masks_nyx: ArrayLike, thr: float = 0.5):
    """
    masks_nyx: (N,Y,X) binary or logits/probs.
    Returns:
      label_yx: (Y,X) with values {0..N} where 0=background, i+1 = instance i
      cmap, norm suitable for imshow(label_yx, cmap=cmap, norm=norm)
    """
    m = _ensure_numpy(masks_nyx)
    if m.size == 0:
        return _empty_label_render()

    # binarize if needed
    if m.dtype != np.bool_:
        m = m > thr

    N, Y, X = m.shape
    label = np.zeros((Y, X), dtype=np.int32)

    # simple overwrite in order; good enough for viz
    for i in range(N):
        label[m[i]] = i + 1

    cmap, norm = _instance_cmap(N)
    return label, cmap, norm


def _label_and_cmap_from_label_map(slice_yx: ArrayLike, ids: np.ndarray):
    """Render one 2D slice of an integer instance label map directly.

    ``ids`` are the volume's sorted non-zero ids (``np.unique``, computed ONCE per
    volume by the caller) so an instance keeps the same dense index -- hence the
    same color -- on every slice and ortho plane. ``searchsorted`` instead of a
    LUT gather: ids can be sparse 64-bit tile-global values, so a max_id-sized
    LUT is not safe to allocate. Memory O(Y*X + N), never O(N * Y*X).

    Returns the same (label_yx {0..N}, cmap, norm) triple as
    :func:`_label_and_cmap_from_instance_masks`; because that path also ordered
    instances by ``np.unique``, the output is bit-identical to the old
    explode-then-collapse render.
    """
    m = _ensure_numpy(slice_yx)
    if ids.size == 0:
        return _empty_label_render()
    pos = np.minimum(np.searchsorted(ids, m), ids.size - 1)
    label = np.where(ids[pos] == m, pos + 1, 0).astype(np.int32)  # miss (incl. 0) -> bg
    cmap, norm = _instance_cmap(int(ids.size))
    return label, cmap, norm


def save_bbox_overlay(
    pred_boxes_xyzxyz: ArrayLike,
    image: ArrayLike,
    save_dir: Path | str,
    identifier: str,
    *,
    z_step: int = 10,
    pmin: float = 1.0,
    pmax: float = 99.0,
    background_channel: int = 0,
) -> None:
    """
    Draw predicted boxes (xyzxyz format) on background image slices and save as PDF.
    Used by bbox_overlay viz handler.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    img_tzyxc = _ensure_numpy_tzyxc(image)
    T, Z, Y, X, C = img_tzyxc.shape
    if not (0 <= background_channel < C):
        raise ValueError(f"background_channel={background_channel} out of range for C={C}")
    boxes = _ensure_numpy(pred_boxes_xyzxyz).reshape(-1, 6).astype(np.float32)

    out_pdf = save_dir / f"{identifier}_bboxes.pdf"
    with PdfPages(out_pdf) as pdf:
        for t in range(T):
            for z in range(0, Z, max(1, z_step)):
                bg = normalize_slice(
                    img_tzyxc[t, z, :, :, background_channel],
                    pmin=pmin,
                    pmax=pmax,
                )
                fig = Figure(figsize=(12, 8))
                ax = fig.subplots(1, 1)
                ax.imshow(bg, cmap="gray", interpolation="nearest")
                ax.set_title(f"Pred boxes | T={t} Z={z}")
                ax.axis("off")
                for (x0, y0, z0, x1, y1, z1) in _boxes_in_z_slice(boxes, z=z):
                    w = max(0.0, float(x1 - x0))
                    h = max(0.0, float(y1 - y0))
                    if w > 0 and h > 0:
                        ax.add_patch(
                            Rectangle(
                                (float(x0), float(y0)),
                                w,
                                h,
                                fill=False,
                                linewidth=1.5,
                                edgecolor="cyan",
                            )
                        )
                fig.tight_layout()
                pdf.savefig(fig)


def save_instance_predictions(
    save_dir: Path | str,
    identifier: str,
    image: ArrayLike,
    preds: Dict[str, Any],
    targets: Optional[Dict[str, Any]] = None,
    region: Optional[Dict[str, Any]] = None,
    pred_boxes_key: str = "boxes",
    pred_masks_key: str = "masks",
    gt_boxes_key: str = "boxes",
    gt_masks_key: str = "masks",
    kinds: Optional[Dict[str, Optional[str]]] = None,
    pred_boxes_format: Literal["xyzxyz", "cxcyczwhd"] = "xyzxyz",
    gt_boxes_format: Literal["xyzxyz", "cxcyczwhd"] = "cxcyczwhd",
    z_step: int = 10,
    pmin: float = 1.0,
    pmax: float = 99.0,
    background_channel: int = 0,
    scale_gt_boxes: bool = True,
    input_format: Literal["ZYXC", "TZYXC"] = "ZYXC",
    ortho: bool = False,
    # NOTE: unused for now
    ortho_mode: Literal["center"] = "center",
):
    """
    For each record, for each (t,z):
      Row 1: Background channel (duplicated in both columns for alignment)
      Row 2 (optional): Boxes  [GT | Pred] drawn on background
      Row 3 (optional): Masks  [GT | Pred] mask overlay on background

    Drops row 2 if BOTH GT+Pred boxes missing/empty.
    Drops row 3 if BOTH GT+Pred masks missing/empty.

    ``kinds`` (record.kinds, ``{name -> declared OutputKind value}``) selects the
    pred-mask render path: ``instance_label_map`` preds are the raw integer volume
    and render natively per slice; anything else is a per-object stack.

    Writes: <ident>_instances.pdf
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Single record: render over a length-1 batch (keeps the per-(t,z) renderer below).
    identifiers = [identifier]
    images = [image]
    preds = [preds]
    targets = [targets if targets is not None else {}]
    regions = None if region is None else [region]
    n = 1

    def _has_nonempty(x: Any) -> bool:
        if x is None:
            return False
        arr = _ensure_numpy(x)
        return arr.size > 0

    for i in tqdm(range(n), desc="records", unit="rec"):
        ident = str(identifiers[i])
        img_tzyxc = _ensure_numpy_tzyxc(images[i])  # (T,Z,Y,X,C)
        T, Z, Y, X, C = img_tzyxc.shape

        region = regions[i] if regions is not None else None
        region_s = _region_str(region)

        if not (0 <= background_channel < C):
            raise ValueError(f"background_channel={background_channel} out of range for C={C}")

        # ---- Boxes ----
        gt_boxes = targets[i].get(gt_boxes_key, None)
        pr_boxes = preds[i].get(pred_boxes_key, None)

        if gt_boxes is None:
            gt_xyzxyz = np.zeros((0, 6), dtype=np.float32)
        else:
            # convert to "xyzxyz" and apply scale factors if given
            gt_boxes = convert_bbox_format(
                gt_boxes,
                bbox_input_format=gt_boxes_format,
                bbox_output_format="xyzxyz",
                scale_factors=torch.tensor([X, Y, Z, X, Y, Z], dtype=torch.float32) if scale_gt_boxes else None,
            )
            gt_xyzxyz = _ensure_numpy(gt_boxes).reshape(-1, 6).astype(np.float32)

        if pr_boxes is None:
            pr_xyzxyz = np.zeros((0, 6), dtype=np.float32)
        else:
            # we assume preds are already in "xyzxyz" format and scaled appropriately
            pr_xyzxyz = _ensure_numpy(pr_boxes).reshape(-1, 6).astype(np.float32)

        # ---- Masks ----
        gt_masks = targets[i].get(gt_masks_key, None)
        pr_masks = preds[i].get(pred_masks_key, None)
        pred_masks_kind = (kinds or {}).get(pred_masks_key)

        # TODO: fix this kind of shape juggling

        # GT masks (when the preprocessor materialized them: dense-mask heads,
        # SAM2) arrive as per-instance (N,Z,Y,X) stacks; unsqueeze the singleton
        # time axis the per-slice renderer indexes. Labelmap-native heads
        # (MaskDINO) carry "label_map" instead -> gt_masks stays None, no GT row.
        # Pred masks are either an already-normalized (N,1,Z,Y,X) stack, or --
        # for INSTANCE_LABEL_MAP -- the raw integer volume, rendered natively
        # slice by slice (O(volume) memory, independent of instance count).
        if gt_masks is not None:
            gt_masks = _ensure_numpy(gt_masks)
            if gt_masks.ndim == 4:
                gt_masks = gt_masks[:, None, ...]  # (N,Z,Y,X) -> (N,1,Z,Y,X)

        pr_labelmap = None                          # (T,Z,Y,X), label-map-native path
        pr_ids = np.zeros(0, dtype=np.int64)
        if pr_masks is not None and pred_masks_kind == OutputKind.INSTANCE_LABEL_MAP.value:
            vol = _ensure_numpy(pr_masks)
            if vol.ndim in (4, 5) and vol.shape[-1] == 1:
                vol = vol[..., 0]                   # drop trailing channel
            if vol.ndim == 3:
                vol = vol[None]                     # (Z,Y,X) -> (T=1,Z,Y,X)
            if vol.ndim != 4:
                raise ValueError(
                    f"instance_label_map pred must be (Z,Y,X[,1]) or (T,Z,Y,X[,1]), "
                    f"got shape {vol.shape}"
                )
            pr_labelmap = vol
            pr_ids = np.unique(vol)                 # ONE pass; reused for every slice
            pr_ids = pr_ids[pr_ids != 0]
            pr_masks = None                         # never enters the stack path
        elif pr_masks is not None:
            # already canonicalized to (N,1,Z,Y,X) by the record builder.
            pr_masks = _ensure_numpy(pr_masks)

        # Per-object stacks are (N,1,Z,Y,X) with a SINGLETON axis-1 -- the
        # renderer indexes that axis with t, which is only correct for T=1.
        # For multi-timepoint data fail loudly instead of mis-slicing Z as
        # time (the label-map render path handles real T; use that).
        for _masks, _side in ((gt_masks, "GT"), (pr_masks, "pred")):
            if _masks is not None and T > 1 and _masks.shape[1] != T:
                raise ValueError(
                    f"{_side} instance stack has axis-1 size {_masks.shape[1]} but the "
                    f"image has T={T}: (N,1,Z,Y,X) stacks cannot be indexed by time. "
                    "Use the label-map render path for T>1, or supply (N,T,Z,Y,X) stacks."
                )

        # --- plot ---
        has_boxes_row = _has_nonempty(gt_xyzxyz) or _has_nonempty(pr_xyzxyz)
        has_masks_row = _has_nonempty(gt_masks) or _has_nonempty(pr_masks) or pr_ids.size > 0

        row_kinds: list[str] = ["bg"]
        if has_boxes_row:
            row_kinds.append("boxes")
        if has_masks_row:
            row_kinds.append("masks")

        print(f"[save_instance_predictions] {ident}: T={T} Z={Z} | rows: {row_kinds}")

        out_pdf = save_dir / f"{ident}_instances.pdf"
        with PdfPages(out_pdf) as pdf:
            for t in tqdm(range(T), desc=f"{ident} T", unit="t"):
                # Ortho crosshair selection (currently center)
                if ortho:
                    y_line = Y // 2
                    x_line = X // 2

                    # Precompute ortho background planes for this t (background channel)
                    bg_xz = normalize_slice(
                        img_tzyxc[t, :, y_line, :, background_channel],  # (Z,X)
                        pmin=pmin, pmax=pmax
                    )
                    bg_yz = normalize_slice(
                        img_tzyxc[t, :, :, x_line, background_channel].transpose(1, 0),  # (Y,Z)
                        pmin=pmin, pmax=pmax
                    )

                    def _draw_crosshair_xy(a):
                        a.axhline(y_line, linewidth=1.0, alpha=0.85, color="yellow")
                        a.axvline(x_line, linewidth=1.0, alpha=0.85, color="yellow")

                    def _draw_crosshair_xz(a, z_cur: int):
                        a.axhline(z_cur, linewidth=1.0, alpha=0.85, color="yellow")  # z is vertical
                        a.axvline(x_line, linewidth=1.0, alpha=0.85, color="yellow")  # x is horizontal

                    def _draw_crosshair_yz(a, z_cur: int):
                        a.axhline(y_line, linewidth=1.0, alpha=0.85, color="yellow")  # y is vertical
                        a.axvline(z_cur, linewidth=1.0, alpha=0.85, color="yellow")   # z is horizontal

                for z in tqdm(range(0, Z, max(1, int(z_step))), desc=f"{ident} Z", unit="z", leave=False):
                    # Build figure grid
                    if ortho:
                        nrows = 2 * len(row_kinds)
                        ncols = 4
                        fig_w = 24
                        fig_h = 3.6 * nrows
                    else:
                        nrows = len(row_kinds)
                        ncols = 2
                        fig_w = 12
                        fig_h = 4.8 * nrows

                    fig = Figure(figsize=(fig_w, fig_h))
                    ax = fig.subplots(nrows, ncols, squeeze=False)

                    # Page header: ident + crop info
                    fig.suptitle(f"T={t} Z={z}{region_s}", fontsize=12)

                    # background slice (XY at this z)
                    bg = normalize_slice(
                        img_tzyxc[t, z, :, :, background_channel],
                        pmin=pmin,
                        pmax=pmax,
                    )

                    if ortho:
                        def _axs(kind_idx: int, side: int):
                            """
                            Returns (xy, yz, xz, blank) axes for:
                              side=0 (GT unit)  in cols [0,1]
                              side=1 (Pred unit) in cols [2,3]
                            Unit layout:
                              [XY] [YZ]
                              [XZ] [  ]
                            """
                            rr = 2 * kind_idx
                            cc = 2 * side
                            a_xy = ax[rr, cc]
                            a_yz = ax[rr, cc + 1]
                            a_xz = ax[rr + 1, cc]
                            a_bl = ax[rr + 1, cc + 1]
                            return a_xy, a_yz, a_xz, a_bl
                    else:
                        def _axs(kind_idx: int, side: int):
                            return ax[kind_idx, side], None, None, None

                    # --- Draw each row kind (bg / boxes / masks) ---
                    for kind_idx, kind in enumerate(row_kinds):
                        for side in (0, 1):
                            a_xy, a_yz, a_xz, a_bl = _axs(kind_idx, side)

                            # Base planes
                            a_xy.imshow(bg, cmap="gray", interpolation="nearest")
                            if ortho:
                                a_yz.imshow(bg_yz, cmap="gray", interpolation="nearest", aspect="auto")
                                a_xz.imshow(bg_xz, cmap="gray", interpolation="nearest", aspect="auto")
                                _draw_crosshair_xy(a_xy)
                                _draw_crosshair_yz(a_yz, z_cur=z)
                                _draw_crosshair_xz(a_xz, z_cur=z)
                                a_bl.axis("off")

                            # Choose which annotations to draw on this side
                            if kind == "bg":
                                if ortho:
                                    a_xy.set_title(f"{'GT' if side==0 else 'Pred'} | BG XY (t={t}, z={z})")
                                    a_yz.set_title(f"{'GT' if side==0 else 'Pred'} | BG YZ (x={x_line})")
                                    a_xz.set_title(f"{'GT' if side==0 else 'Pred'} | BG XZ (y={y_line})")
                                else:
                                    a_xy.set_title(f"{'GT' if side==0 else 'Pred'} | BG (t={t}, z={z})")

                            elif kind == "boxes":
                                boxes = gt_xyzxyz if side == 0 else pr_xyzxyz
                                col = "lime" if side == 0 else "cyan"

                                # XY @ z
                                for (x0, y0, z0, x1, y1, z1) in _boxes_in_z_slice(boxes, z=z):
                                    w = max(0.0, float(x1 - x0))
                                    h = max(0.0, float(y1 - y0))
                                    if w > 0 and h > 0:
                                        a_xy.add_patch(
                                            Rectangle(
                                                (float(x0), float(y0)),
                                                w,
                                                h,
                                                fill=False,
                                                linewidth=1.5,
                                                edgecolor=col,
                                            )
                                        )

                                if ortho:
                                    # XZ @ y_line  (axes: x horiz, z vert)
                                    for (x0, y0, z0, x1, y1, z1) in _boxes_in_y_slice(boxes, y=y_line):
                                        w = max(0.0, float(x1 - x0))
                                        h = max(0.0, float(z1 - z0))
                                        if w > 0 and h > 0:
                                            a_xz.add_patch(
                                                Rectangle(
                                                    (float(x0), float(z0)),
                                                    w,
                                                    h,
                                                    fill=False,
                                                    linewidth=1.5,
                                                    edgecolor=col,
                                                )
                                            )

                                    # YZ @ x_line  (axes: z horiz, y vert)
                                    for (x0, y0, z0, x1, y1, z1) in _boxes_in_x_slice(boxes, x=x_line):
                                        w = max(0.0, float(z1 - z0))
                                        h = max(0.0, float(y1 - y0))
                                        if w > 0 and h > 0:
                                            a_yz.add_patch(
                                                Rectangle(
                                                    (float(z0), float(y0)),
                                                    w,
                                                    h,
                                                    fill=False,
                                                    linewidth=1.5,
                                                    edgecolor=col,
                                                )
                                            )

                                    a_xy.set_title(f"{'GT' if side==0 else 'Pred'} | Boxes XY (z={z})")
                                    a_yz.set_title(f"{'GT' if side==0 else 'Pred'} | Boxes YZ (x={x_line})")
                                    a_xz.set_title(f"{'GT' if side==0 else 'Pred'} | Boxes XZ (y={y_line})")
                                else:
                                    a_xy.set_title(f"{'GT' if side==0 else 'Pred'} | Boxes (t={t}, z={z})")

                            elif kind == "masks":
                                use_labelmap = side == 1 and pr_labelmap is not None
                                masks = gt_masks if side == 0 else pr_masks
                                if use_labelmap or (masks is not None and masks.shape[0] > 0):
                                    # XY @ z
                                    if use_labelmap:
                                        lab_xy, cmap_xy, norm_xy = _label_and_cmap_from_label_map(
                                            pr_labelmap[t, z], pr_ids
                                        )
                                    else:
                                        lab_xy, cmap_xy, norm_xy = _label_and_cmap_from_instance_masks(
                                            masks[:, t, z]
                                        )
                                    a_xy.imshow(
                                        lab_xy,
                                        cmap=cmap_xy,
                                        norm=norm_xy,
                                        interpolation="nearest",
                                        alpha=0.45,
                                    )

                                    if ortho:
                                        # XZ @ y_line: (Z,X)
                                        if use_labelmap:
                                            lab_xz, cmap_xz, norm_xz = _label_and_cmap_from_label_map(
                                                pr_labelmap[t, :, y_line, :], pr_ids
                                            )
                                        else:
                                            lab_xz, cmap_xz, norm_xz = _label_and_cmap_from_instance_masks(
                                                masks[:, t, :, y_line, :]
                                            )
                                        a_xz.imshow(
                                            lab_xz,
                                            cmap=cmap_xz,
                                            norm=norm_xz,
                                            interpolation="nearest",
                                            alpha=0.45,
                                            aspect="auto",
                                        )

                                        # YZ @ x_line: (Z,Y) -> (Y,Z)
                                        if use_labelmap:
                                            lab_yz, cmap_yz, norm_yz = _label_and_cmap_from_label_map(
                                                pr_labelmap[t, :, :, x_line].transpose(1, 0), pr_ids
                                            )
                                        else:
                                            m_yz = masks[:, t, :, :, x_line].transpose(0, 2, 1)
                                            lab_yz, cmap_yz, norm_yz = _label_and_cmap_from_instance_masks(m_yz)
                                        a_yz.imshow(
                                            lab_yz,
                                            cmap=cmap_yz,
                                            norm=norm_yz,
                                            interpolation="nearest",
                                            alpha=0.45,
                                            aspect="auto",
                                        )

                                if ortho:
                                    a_xy.set_title(f"{'GT' if side==0 else 'Pred'} | Masks XY (z={z})")
                                    a_yz.set_title(f"{'GT' if side==0 else 'Pred'} | Masks YZ (x={x_line})")
                                    a_xz.set_title(f"{'GT' if side==0 else 'Pred'} | Masks XZ (y={y_line})")
                                else:
                                    a_xy.set_title(f"{'GT' if side==0 else 'Pred'} | Masks (t={t}, z={z})")

                            # Cosmetics
                            a_xy.axis("off")
                            if a_yz is not None:
                                a_yz.axis("off")
                            if a_xz is not None:
                                a_xz.axis("off")
                            if a_bl is not None:
                                a_bl.axis("off")

                    fig.tight_layout(rect=[0, 0, 1, 0.96])
                    pdf.savefig(fig)

        print(f"[save_instance_predictions] wrote {out_pdf}")


def visualize_semantic_labels(
    image: ArrayLike,
    pred: Dict[str, ArrayLike],
    target: Dict[str, ArrayLike] | None,
    out_path: Path | str,
    *,
    class_names: Sequence[str] | None = None,
    z_step: int = 15,
    pmin: float = 1.0,
    pmax: float = 99.0,
    mip_depth: int = 20,
) -> None:
    """
    Visualize semantic segmentation labels with per-class columns, plus a column with merged integer labels.

    Prediction: if pred_is_probabilities is False (default), values are treated as logits
    and sigmoid is applied; if True, values are already in [0, 1] and only clipped.
    GT is clipped to [0, 1]. Only the image column uses percentile normalization (raw image data).

    Layout: Image | Pred_class_0 | ... | Pred_class_K | GT_class_0 | ... | GT_class_K | Merged

    Args:
        image: TZYXC or ZYXC array - real input image
        pred_semantic: TZYXC or ZYXC with C in {1, num_classes} - per-class predictions (logits or probs)
        targets: TZYXC/ZYXC with C in {1, num_classes}, or ZYX (class index), or None
        out_path: Output PDF path
        class_names: Optional list of class names (length num_classes)
        z_step: Step size for z-slices
        pmin: Percentile for image normalization (lower bound)
        pmax: Percentile for image normalization (upper bound)
        mip_depth: Depth for max-intensity projection
    """
    # Normalize inputs to TZYXC
    image_tzyxc = _ensure_numpy_tzyxc(image)
    pred_masks = _ensure_numpy_tzyxc(pred["pred_masks"])
    pred_labels = _ensure_numpy_tnc(pred["pred_classes"])
    masks_labelmap = _ensure_numpy_tzyxc(pred["masks_labelmap"])
    num_classes = pred_labels.shape[-1]
    
    T, Z, Y, X, C_img = image_tzyxc.shape
    T_pred, Z_pred, Y_pred, X_pred, C_pred = pred_masks.shape

    # Validate spatial dimensions match
    if (T, Z, Y, X) != (T_pred, Z_pred, Y_pred, X_pred):
        raise ValueError(
            f"Image and prediction must have same T,Z,Y,X. "
            f"Got image {(T,Z,Y,X)} vs pred {(T_pred,Z_pred,Y_pred,X_pred)}"
        )

    # Handle gt_semantic. When there is no target we leave ``gt_masks`` as None
    # (instead of allocating a full (T,Z,Y,X,C) zero volume just to render blank
    # panels): the GT columns below simply render empty axes.
    has_gt = target is not None
    if has_gt:
        gt_masks = _ensure_numpy_tzyxc(target["masks"])
    else:
        gt_masks = None

    # Prepare class names
    if class_names is None:
        class_names = [f"class_{c}" for c in range(num_classes)]
    elif len(class_names) != num_classes:
        raise ValueError(
            f"class_names length {len(class_names)} != num_classes {num_classes}"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Layout: Image | Pred_c0 | ... | Pred_cK | GT_c0 | ... | GT_cK | Merged
    num_cols = 1 + num_classes + num_classes + 1  # add 1 for merged mask labels column

    # One figure reused across all (t, z) pages (num_cols is fixed).
    fig = Figure(figsize=(5 * num_cols, 5))
    axes_row = fig.subplots(1, num_cols, squeeze=False)
    axes_row = axes_row[0, :]

    with PdfPages(out_path) as pdf:
        for t in range(T):
            for z0 in range(0, Z, z_step):
                z1 = min(z0 + mip_depth, Z)
                if z1 <= z0:
                    continue

                for ax in axes_row:
                    ax.cla()

                # Column 0: Image
                img_block = image_tzyxc[t, z0:z1]  # [z_block, Y, X, C_img]
                img_mip = img_block.max(axis=0)  # [Y, X, C_img]
                # Use first channel or mean across channels
                if C_img == 1:
                    img_2d = img_mip[..., 0]
                else:
                    img_2d = img_mip.mean(axis=-1)
                img_norm = normalize_slice(img_2d, pmin=pmin, pmax=pmax)
                axes_row[0].imshow(img_norm, cmap="gray", interpolation="nearest")
                axes_row[0].set_title(f"Image | T={t}  Z∈[{z0},{z1})")
                axes_row[0].axis("off")

                # Columns 1 to num_classes: Pred per class (sigmoid already applied; clip to [0,1], no min-max)
                for c in range(num_classes):
                    col_idx = 1 + c
                    pred_block = pred_masks[t, z0:z1, ..., c]  # [z_block, Y, X]
                    pred_mip = pred_block.max(axis=0)  # [Y, X]
                    pred_display = np.clip(pred_mip, 0, 1).astype(np.float32)
                    axes_row[col_idx].imshow(pred_display, cmap="viridis", interpolation="nearest")
                    axes_row[col_idx].set_title(f"Pred {class_names[c]}")
                    axes_row[col_idx].axis("off")

                # Columns num_classes+1 to 2*num_classes: GT per class (clip to [0,1], no min-max).
                # With no target we render empty axes rather than zeros.
                for c in range(num_classes):
                    col_idx = 1 + num_classes + c
                    if has_gt:
                        gt_block = gt_masks[t, z0:z1, ..., c]  # [z_block, Y, X]
                        gt_mip = gt_block.max(axis=0)  # [Y, X]
                        gt_display = np.clip(gt_mip, 0, 1).astype(np.float32)
                        axes_row[col_idx].imshow(gt_display, cmap="viridis", interpolation="nearest")
                        axes_row[col_idx].set_title(f"GT {class_names[c]}")
                    else:
                        axes_row[col_idx].set_title(f"GT {class_names[c]} (n/a)")
                    axes_row[col_idx].axis("off")

                # New final column: merged integer labels using mask2former.py logic
                # For pred_masks: 
                # get mask_labels by taking argmax over channels (last axis) plus 1, times a foreground mask
                # Ensure all ops are NumPy, not potentially torch
                pred_block = np.asarray(masks_labelmap[t, z0:z1, ...])  # [z_block, Y, X]
                # Center slice, NOT a max-projection
                mask_labels = pred_block[pred_block.shape[0] // 2]  # [Y, X]

                axes_row[-1].imshow(mask_labels, cmap="tab20", interpolation="nearest")  # Discrete color map
                axes_row[-1].set_title(f"Masks labelmap: class labels {pred_labels[t]}")
                axes_row[-1].axis("off")
       

                fig.tight_layout()
                pdf.savefig(fig)


def save_semantic_predictions(
    name: str,
    preds: Dict[str, ArrayLike],
    image: ArrayLike,
    save_dir: Path | str,
    z_step_pdf: int,
    targets: Dict[str, ArrayLike] | None = None,
    pmin: float = 1.0,
    pmax: float = 99.0,
    mip_depth: int = 20,
    class_names: Sequence[str] | None = None,
) -> None:
    """Plot one sample's semantic-segmentation prediction as a MIP PDF.

    Volume writing lives in the save path (``inference/saver.py`` -> ``data/io.py``);
    this helper is plot-only.

    Args:
        name: identifier for this prediction (drives output filenames).
        preds: prediction dict for this sample (e.g. ``{"pred_masks": [Z,Y,X,C]/[T,Z,Y,X,C]}``).
        image: input image ``[Z,Y,X,C]`` or ``[T,Z,Y,X,C]``.
        targets: this sample's target dict, or None.
        (remaining args: z_step_pdf, pmin/pmax, mip_depth, class_names.)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    image = _ensure_numpy(image)
    preds = {k: _ensure_numpy(v) for k, v in preds.items()}
    if targets is not None:
        targets = {k: _ensure_numpy(v) for k, v in targets.items()}

    # pred is already probabilities from the reducer
    visualize_semantic_labels(
        image=image,
        pred=preds,
        target=targets,
        out_path=save_dir / f"pred_{name}_semantic_MIP.pdf",
        class_names=class_names,
        z_step=z_step_pdf,
        pmin=pmin,
        pmax=pmax,
        mip_depth=mip_depth,
    )