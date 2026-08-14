"""Inference postprocess: processed-resolution model outputs -> original tile frame.

Tile-mode inference resizes inputs down to the model's ``train_shape``. Postprocess
undoes that per sample: DENSE outputs are resized back up to their original tile size
(``metainfo['orig_image_sizes']``) and placed top-left into the full-tile buffer (the
saver later crops the zero-pad); SPARSE boxes are rescaled to the original frame.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cell_observatory_platform.data.data_types import OutputKind


def postprocess(preds: Dict[str, Any], data_sample: dict, outputs_metadata: dict) -> Dict[str, Any]:
    """Restore dense outputs to original tile resolution; rescale sparse boxes.

    Idempotent: when a sample was not resized (orig == processed) each dense resize
    is an identity op and box rescale skips, so this is a no-op.
    """
    metainfo = data_sample["metainfo"]
    orig_sizes = _per_sample_spatial(metainfo["orig_image_sizes"])
    proc_sizes = _per_sample_spatial(metainfo["image_sizes"])

    for name, meta in outputs_metadata["save_tensors"].items():
        annotation_type = meta["annotation_type"]
        data_format = str(meta["data_format"]).upper()
        if name not in preds:
            raise ValueError(
                f"save_tensors entry {name!r} is not among the model outputs "
                f"{sorted(preds)} -- check the inference config's save_tensors "
                "against the model's predict() contract."
            )
        if annotation_type == "dense":
            preds[name] = _restore_dense_tensor(
                preds[name], data_format, orig_sizes, name, proc_sizes, outputs_metadata,
            )
        elif annotation_type == "sparse" and data_format in ("N6", "TN6"):
            preds[name] = _rescale_boxes_to_orig(preds[name], proc_sizes, orig_sizes)
    return preds


def _per_sample_spatial(value: torch.Tensor) -> List[Tuple[int, int, int]]:
    """Per-sample ``(Z, Y, X)`` from an ``image_sizes``-like field.

    The field is always a batched tensor -- ``(B, 3)`` for ZYXC or ``(B, 4)`` for
    TZYXC -- so the trailing 3 entries are the spatial extents. Matches the
    tensor-only contract asserted in ``data/transforms/resize.py``.
    """
    if not torch.is_tensor(value):
        raise TypeError(f"image_sizes-like field must be a tensor, got {type(value)}")
    return [tuple(int(x) for x in row) for row in value[:, -3:].tolist()]


def _buffer_spatial(outputs_metadata: dict, name: str, data_format: str) -> Tuple[int, int, int]:
    """Full-tile restore target ``(Z, Y, X)`` from the (overridden) tensor_info shape.

    A dense output must declare a shape carrying the ``data_format`` spatial axes;
    a missing entry/shape or absent axis raises.
    """
    shape = list(outputs_metadata["tensor_info"][name]["shape"])
    return tuple(int(shape[data_format.index(ax)]) for ax in ("Z", "Y", "X"))


def _restore_dense_tensor(
    tensor: torch.Tensor,
    data_format: str,
    orig_sizes: List[Tuple[int, int, int]],
    name: str,
    proc_sizes: List[Tuple[int, int, int]],
    outputs_metadata: dict,
) -> torch.Tensor:
    """Resize each sample to its original size and place top-left in the full-tile buffer.

    Each sample is first cropped to its valid region (``proc_sizes``) so trailing
    padding is dropped before the up-resize (a no-op when the output is already
    fully valid, e.g. after Resize with ``crop_to_valid``).
    """
    target_spatial = _buffer_spatial(outputs_metadata, name, data_format)
    if tensor.dim() != len(data_format) + 1:  # survives python -O
        raise ValueError(
            f"{name}: dense output rank {tensor.dim()} does not match layout "
            f"{data_format!r} + batch ({len(data_format) + 1})"
        )
    has_time = data_format.startswith("T")
    B, C = tensor.shape[0], tensor.shape[-1]
    fz, fy, fx = target_spatial
    if has_time:
        out = tensor.new_zeros((B, tensor.shape[1], fz, fy, fx, C))
    else:
        out = tensor.new_zeros((B, fz, fy, fx, C))
    # Integer (label/instance) maps -> nearest; continuous outputs -> trilinear.
    mode = "trilinear" if torch.is_floating_point(tensor) else "nearest"
    for b in range(B):
        sample = tensor[b]
        cur_spatial = sample.shape[1:4] if has_time else sample.shape[0:3]
        crop = tuple(min(int(p), int(s)) for p, s in zip(proc_sizes[b], cur_spatial))
        if crop != tuple(int(s) for s in cur_spatial):
            sample = crop_sample_spatial(sample, crop, has_time)
        if any(int(o) > int(f) for o, f in zip(orig_sizes[b], target_spatial)):
            # Clamping here would flow downstream into the saver's crop, which
            # numpy-slices past the end silently -- the PERSISTED volume would
            # be truncated to the buffer size with no error.
            raise ValueError(
                f"{name}: sample {b} original size {tuple(orig_sizes[b])} exceeds "
                f"the declared full-tile buffer spatial {tuple(target_spatial)} -- "
                "tensor_info shape (DB maxima) is too small for this tile."
            )
        oz, oy, ox = (int(o) for o in orig_sizes[b])
        resized = _resize_sample_spatial(sample, (oz, oy, ox), has_time, mode)
        if has_time:
            out[b, :, :oz, :oy, :ox, :] = resized.to(out.dtype)
        else:
            out[b, :oz, :oy, :ox, :] = resized.to(out.dtype)
    return out


def crop_sample_spatial(
    sample: torch.Tensor, spatial: Tuple[int, int, int], has_time: bool
) -> torch.Tensor:
    """Crop a single sample ((Z,Y,X,C) or (T,Z,Y,X,C)) to ``spatial`` (Z,Y,X).

    Pure axis-slicing, so it applies equally to torch tensors (postprocess) and
    numpy arrays (the saver, dropping full-tile buffer zero-pad before writing).
    """
    cz, cy, cx = spatial
    if has_time:
        return sample[:, :cz, :cy, :cx, :]
    return sample[:cz, :cy, :cx, :]


def _resize_sample_spatial(
    sample: torch.Tensor,
    out_spatial: Tuple[int, int, int],
    has_time: bool,
    mode: str,
) -> torch.Tensor:
    """Resize spatial dims of a single sample ((Z,Y,X,C) or (T,Z,Y,X,C))."""
    kwargs: Dict[str, Any] = {"size": out_spatial, "mode": mode}
    if mode == "trilinear":
        kwargs["align_corners"] = False
    if has_time:
        # (T, Z, Y, X, C) -> (T, C, Z, Y, X) -> resize -> (T, oz, oy, ox, C)
        x = sample.permute(0, 4, 1, 2, 3).float()
        x = F.interpolate(x, **kwargs)
        return x.permute(0, 2, 3, 4, 1)
    # (Z, Y, X, C) -> (1, C, Z, Y, X) -> resize -> (oz, oy, ox, C)
    x = sample.permute(3, 0, 1, 2).unsqueeze(0).float()
    x = F.interpolate(x, **kwargs)
    return x.squeeze(0).permute(1, 2, 3, 0)


def _rescale_boxes_to_orig(
    boxes: torch.Tensor,
    proc_sizes: List[Tuple[int, int, int]],
    orig_sizes: List[Tuple[int, int, int]],
) -> torch.Tensor:
    """Scale absolute ``xyzxyz`` boxes ``(B, N, 6)`` from processed to original scale."""
    if boxes.dim() != 3 or boxes.shape[-1] != 6:  # survives python -O
        raise ValueError(
            f"sparse box output must be (B, N, 6), got {tuple(boxes.shape)}"
        )
    out = boxes.clone()
    for b in range(boxes.shape[0]):
        oz, oy, ox = orig_sizes[b]
        pz, py, px = proc_sizes[b]
        if (oz, oy, ox) == (pz, py, px):
            continue
        sx, sy, sz = ox / max(px, 1), oy / max(py, 1), oz / max(pz, 1)
        scale = boxes.new_tensor([sx, sy, sz, sx, sy, sz])
        out[b] = boxes[b] * scale
    return out


# ============================================================================
# Per-sample records: one uniform unpack point for the save + viz workers.
# ----------------------------------------------------------------------------
# Both workers materialize the batched SHM view of the model outputs, then call
# build_records to slice them into per-sample InferenceRecords.
# ============================================================================

# Metainfo columns describing the on-disk / crop location of each tile. Present on
# the tile-inference path; a record's region is None when they are absent.
REGION_COLUMNS = (
    "prepared_id", "tile_name",
    "time_start", "time_size", "z_start", "z_size",
    "y_start", "y_size", "x_start", "x_size",
)



@dataclass
class InferenceRecord:
    index: int                                  # batch position b
    metadata: Dict[str, Any]                    # per-sample sliced columns (path/tile/...)
    region: Optional[Dict[str, Any]]            # coords dict, or None if region cols absent
    image: Optional[np.ndarray]                 # data_tensor[b], or None if not requested
    preds: Dict[str, np.ndarray]                # {name: output_arrays[name][b]}
    targets: Optional[Dict[str, Any]]           # per-sample target dict, or None
    kinds: Dict[str, Optional[str]]             # {name: declared kind} (from tensor_metadata)


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def kinds_from_metainfo(metainfo: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """``{tensor_name -> declared kind}`` from the model-sourced tensor_metadata transport."""
    tm = metainfo.get("tensor_metadata") or {}
    return {name: entry.get("kind") for name, entry in tm.items()}


def build_records(
    output_arrays: Dict[str, np.ndarray],
    metainfo: Dict[str, Any],
    columns: tuple[str, ...],
    image_key: Optional[str] = "data_tensor",
    targets: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[InferenceRecord]:
    """Unpack batched outputs into per-sample records.

    Args:
        output_arrays: ``{name: batched array (B, ...)}`` -- materialized SHM views.
        metainfo: sample metainfo; must carry ``batch_size_actual`` and every ``column``.
        columns: per-sample metadata columns to slice into ``record.metadata``.
        image_key: key in ``output_arrays`` to lift into ``record.image`` (excluded from
            ``preds``); None to keep everything in ``preds``.
        targets: per-sample target list (len B), or None. Supplied explicitly because its
            transport differs by worker (viz attaches it as a top-level key).

    Note: viz renders the PROCESSED frame (tensors at the model's working resolution,
    uniform across the batch) so there is no per-sample crop here; the saver restores +
    crops to the original tile inside ``save_predictions`` itself. Instance-mask
    normalization is per-handler (``instance_overlay`` normalizes per-object stacks via
    :func:`to_instance_stack`; label maps pass through and render natively), not here.
    """
    B = metainfo["batch_size_actual"]                 # KeyError if missing -- required
    kinds = kinds_from_metainfo(metainfo)
    image = output_arrays.get(image_key) if image_key else None

    records: List[InferenceRecord] = []
    for b in range(B):
        preds_b = {
            name: arr[b] for name, arr in output_arrays.items() if name != image_key
        }
        image_b = image[b] if image is not None else None
        records.append(InferenceRecord(
            index=b,
            metadata={col: metainfo[col][b] for col in columns},
            region=_region_at(metainfo, b),
            image=image_b,
            preds=preds_b,
            targets=(targets[b] if targets is not None else None),
            kinds=kinds,
        ))
    return records


def _region_at(metainfo: Dict[str, Any], b: int) -> Optional[Dict[str, Any]]:
    """Per-sample voxel-frame region from the tile columns, or None if they're absent."""
    if any(c not in metainfo for c in REGION_COLUMNS):
        return None
    g = lambda c: int(metainfo[c][b])
    t0, T = g("time_start"), g("time_size")
    z0, sz = g("z_start"), g("z_size")
    y0, sy = g("y_start"), g("y_size")
    x0, sx = g("x_start"), g("x_size")
    return {
        "roi": g("prepared_id"),
        "tile_name": str(metainfo["tile_name"][b]),
        "coords": (t0, t0 + T, z0, z0 + sz, y0, y0 + sy, x0, x0 + sx),
        "coord_frame": "voxel",
    }


def viz_identifier(record: InferenceRecord, rank: int) -> str:
    """One descriptive, filesystem-safe identifier for a record's viz artifacts.

    Uses the tile region when available (unique per crop); otherwise falls back to the
    sample's output folder + tile name. Fail-hard on missing metadata -- no "unknown".
    """
    if record.region is not None:
        r = record.region
        t0, t1, z0, z1, y0, y1, x0, x1 = r["coords"]
        return (
            f"rank{rank:03d}_roi{r['roi']}_{r['tile_name']}"
            f"_t{t0}-{t1}_z{z0}-{z1}_y{y0}-{y1}_x{x0}-{x1}"
        )
    base = str(record.metadata["output_folder"]).replace("/", "_")
    ident = f"{base}_{record.metadata['tile_name']}"
    return ident.replace(".zarr", "").replace(".tiff", "")


def to_instance_stack(masks: Any, kind: Optional[str] = None) -> np.ndarray:
    """Normalize a per-object mask stack to the ``(N, 1, Z, Y, X)`` plotting expects.

    ``INSTANCE_LABEL_MAP`` is NOT exploded here anymore: the overlay renders label
    maps natively, slice by slice (``utils._label_and_cmap_from_label_map``) --
    O(volume) memory instead of O(N * volume), so instance count no longer bounds
    what can be visualized.

      * ``INSTANCE_STACK`` / ``None`` (legacy): a per-object stack; channels-last
        ``(N,Z,Y,X,1)`` (see meta_arch/utils.py instance_stack) drops the trailing
        channel, a bare ``(N,Z,Y,X)`` is unsqueezed, ``(N,1,Z,Y,X)`` passes through.
    """
    masks = _as_numpy(masks)
    if kind == OutputKind.INSTANCE_LABEL_MAP.value:
        raise ValueError(
            "instance_label_map is rendered natively by the overlay (no per-object "
            "explosion); to_instance_stack only normalizes per-object stacks."
        )
    if masks.ndim == 5 and masks.shape[-1] == 1:
        masks = masks[..., 0]        # (N,Z,Y,X,1) -> (N,Z,Y,X)
    if masks.ndim == 4:
        masks = masks[:, None, ...]  # (N,Z,Y,X) -> (N,1,Z,Y,X)
    return masks
