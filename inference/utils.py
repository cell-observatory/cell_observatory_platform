import hashlib
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, Dict

import torch
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from cell_observatory_platform.data.io import save_file

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

    with PdfPages(out_path) as pdf:
        for t in range(T_ref):
            for z0 in range(0, Z_ref, z_step):
                z1 = min(z0 + mip_depth, Z_ref)
                if z1 <= z0:
                    continue

                fig, axes_grid = plt.subplots(
                    max_C,
                    num_types,
                    figsize=(5 * num_types, 4 * max_C),
                    squeeze=False,
                )

                for col, name in enumerate(names):
                    vol = vols_tzyxc[name]
                    block = vol[t, z0:z1]
                    if block.shape[0] == 0:
                        for row in range(max_C):
                            axes_grid[row, col].axis("off")
                        continue

                    mip = block.max(axis=0)  # (Y,X,C)
                    _, _, C = mip.shape

                    for c in range(max_C):
                        ax = axes_grid[c, col]
                        if c < C:
                            img = mip[..., c]
                            img_norm = _normalize_slice(img, pmin=pmin, pmax=pmax)
                            ax.imshow(img_norm, cmap="gray", interpolation="nearest")
                            ax.set_title(f"{name} | T={t}  Z∈[{z0},{z1})  C={c}")
                            ax.axis("off")
                        else:
                            ax.axis("off")

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)


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
                fig, axes_col = plt.subplots(C, 1, figsize=(18, 7 * C), squeeze=False)
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
        input_format=axes,  # axes string encodes layout
        dtype="float16",
    )


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
                    fig, axes = plt.subplots(
                        2 * Cg, 1,
                        figsize=(10, 3 * (2 * Cg)),
                        squeeze=False,
                    )
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
                        ax.imshow(_normalize_slice(img, pmin=pmin, pmax=pmax), cmap="gray", interpolation="nearest")
                        ax.set_title(f"{gt_key} | t={t} z={z} c={c}")
                        ax.axis("off")

                else:
                    # original PCA layout: 1 (rgb) + Cg rows
                    fig, axes = plt.subplots(
                        1 + Cg, 1,
                        figsize=(10, 3 * (1 + Cg)),
                        squeeze=False,
                    )
                    axes = axes[:, 0]

                    ax0 = axes[0]
                    rgb = feat_rgb[t, z]  # (Yg,Xg,3)
                    ax0.imshow(rgb, interpolation="nearest")
                    ax0.set_title(f"{name} | {feat_key} PCA | t={t} z={z}")
                    ax0.axis("off")

                    for c in range(Cg):
                        ax = axes[1 + c]
                        img = gt[t, z, :, :, c]
                        ax.imshow(_normalize_slice(img, pmin=pmin, pmax=pmax), cmap="gray", interpolation="nearest")
                        ax.set_title(f"{gt_key} | t={t} z={z} c={c}")
                        ax.axis("off")

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f"[save_feature_visualizations] wrote {pdf_path}")


def save_predictions(
    name: str,
    predictions: ArrayLike | dict[str, ArrayLike],
    save_dir: Path | str,
    save_as_volume: bool,
    save_as_pdf: bool,
    z_step_pdf: int,
    filetype: Literal["tiff", "zarr"],
    pmin: float = 1.0,
    pmax: float = 99.0,
    zarr_chunk_shape: Optional[Tuple[int, ...]] = None,
    zarr_shard_shape: Optional[Tuple[int, ...]] = None,
):
    """
    Central helper for saving predictions.

    Inputs:
      - predictions:
          * single TZYXC or ZYXC array (torch or numpy), OR
          * dict[name -> TZYXC/ZYXC array].
    Standardization:
      - Each array is converted to TCZYX (if T>1) or CZYX (if T==1).
      - That same arr+axes is used for TIFF, Zarr, and PDF.
      - If predictions is a dict:
          * One PDF with all entries combined (columns = outputs).
          * Separate volume files per entry, with suffix "_{key}".
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[save_predictions] Saving predictions for tile {name}...")

    # Dict case: multi-output
    if isinstance(predictions, dict):
        arr_map: dict[str, np.ndarray] = {}
        axes_map: dict[str, str] = {}

        for key, arr in predictions.items():
            arr_tzyxc = _ensure_numpy_tzyxc(arr)
            arr_tc_or_czyx, axes = _tzyxc_to_tczyx_or_czyx(arr_tzyxc)
            arr_map[key] = arr_tc_or_czyx
            axes_map[key] = axes

        # Single PDF with all outputs on the same pages
        if save_as_pdf:
            pdf_path = save_dir / f"pred_{name}_MIP.pdf"
            print(f"[save_predictions] Writing PDF to: {pdf_path}")
            preds_dict_to_pdf(
                arr_map,
                axes_map,
                out_path=pdf_path,
                z_step=z_step_pdf,
                pmin=pmin,
                pmax=pmax,
            )

        # Separate volumes per output type
        if save_as_volume:
            for key, arr_tc_or_czyx in arr_map.items():
                subname = f"{name}_{key}"
                axes = axes_map[key]
                if filetype == "tiff":
                    _save_tiff_volume(arr_tc_or_czyx, axes=axes, save_dir=save_dir, name=subname)
                elif filetype == "zarr":
                    _save_zarr_volume(
                        arr_tc_or_czyx,
                        axes=axes,
                        save_dir=save_dir,
                        name=subname,
                        zarr_chunk_shape=zarr_chunk_shape,
                        zarr_shard_shape=zarr_shard_shape,
                    )
                else:
                    raise ValueError(f"Unsupported save format: {filetype!r}")

        return

    # Single-array case
    arr_tzyxc = _ensure_numpy_tzyxc(predictions)
    arr_tc_or_czyx, axes = _tzyxc_to_tczyx_or_czyx(arr_tzyxc)

    # PDF pages
    if save_as_pdf:
        pdf_path = save_dir / f"pred_{name}_MIP.pdf"
        preds_to_pdf(arr_tc_or_czyx, axes=axes, out_path=pdf_path, z_step=z_step_pdf)

    # Volume
    if save_as_volume:
        if filetype == "tiff":
            _save_tiff_volume(arr_tc_or_czyx, axes=axes, save_dir=save_dir, name=name)
        elif filetype == "zarr":
            _save_zarr_volume(
                arr_tc_or_czyx,
                axes=axes,
                save_dir=save_dir,
                name=name,
                zarr_chunk_shape=zarr_chunk_shape,
                zarr_shard_shape=zarr_shard_shape,
            )
        else:
            raise ValueError(f"Unsupported save format: {filetype!r}")
