import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F



def stack_metainfo(meta_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not meta_list:
        return {}

    all_keys: set = set()
    for m in meta_list:
        if m is not None:
            all_keys.update(m.keys())

    merged: Dict[str, Any] = {}
    for key in all_keys:
        values = [
            (m.get(key, None) if m is not None else None)
            for m in meta_list
        ]
        merged[key] = _merge_values(values)
    return merged


def _merge_values(values: List[Any]) -> Any:
    if all(v is None for v in values):
        return None

    first = next(v for v in values if v is not None)

    # list: concatenate
    if isinstance(first, list):
        out: list = []
        for v in values:
            if v is not None:
                out.extend(v)
        return out

    # dict: recurse
    if isinstance(first, dict):
        return stack_metainfo([v if v is not None else {} for v in values])

    # tensor: cat along dim 0
    if isinstance(first, torch.Tensor):
        parts = [v for v in values if v is not None]
        return torch.cat(parts, dim=0)

    # scalar / bool / str / other: keep ALL per-crop values as a list
    return list(values)


def parse_target_shape_range(
    target_spatial_shape: Union[Sequence[int], Tuple[Sequence[int], Sequence[int]]],
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], bool]:
    """
    Parse target_spatial_shape which can be either fixed or a range.

    Args:
        target_spatial_shape: Either:
            - Fixed shape: (Z, Y, X) or [Z, Y, X]
            - Range: ((Z_min, Y_min, X_min), (Z_max, Y_max, X_max))

    Returns:
        Tuple of (target_min, target_max, is_random)
        - target_min: (Z_min, Y_min, X_min)
        - target_max: (Z_max, Y_max, X_max)
        - is_random: True if range was specified, False if fixed
    """
    # Check if it's a range (two sequences) or a fixed shape (one sequence)
    if (
        len(target_spatial_shape) == 2
        and hasattr(target_spatial_shape[0], "__len__")
        and hasattr(target_spatial_shape[1], "__len__")
        and len(target_spatial_shape[0]) == 3
        and len(target_spatial_shape[1]) == 3
    ):
        # Range mode: ((Z_min, Y_min, X_min), (Z_max, Y_max, X_max))
        target_min = tuple(int(d) for d in target_spatial_shape[0])
        target_max = tuple(int(d) for d in target_spatial_shape[1])
        return target_min, target_max, True
    else:
        # Fixed mode: (Z, Y, X)
        if len(target_spatial_shape) != 3:
            raise ValueError(
                f"target_spatial_shape must be (Z, Y, X) or ((min), (max)), "
                f"got {target_spatial_shape}"
            )
        target_fixed = tuple(int(d) for d in target_spatial_shape)
        return target_fixed, target_fixed, False


def sample_target_shape(
    target_min: Tuple[int, int, int],
    target_max: Tuple[int, int, int],
    rng: Optional["random.Random"] = None,
) -> Tuple[int, int, int]:
    """
    Sample target shape uniformly from range [target_min, target_max].

    If target_min == target_max, returns target_min (fixed size).

    Args:
        target_min: Minimum shape (Z_min, Y_min, X_min)
        target_max: Maximum shape (Z_max, Y_max, X_max)

    Returns:
        Sampled shape (Z, Y, X)
    """
    r = rng if rng is not None else random
    return (
        r.randint(target_min[0], target_max[0]),
        r.randint(target_min[1], target_max[1]),
        r.randint(target_min[2], target_max[2]),
    )


# TODO: generalize to N-D
def resize_tensor_3d(
    tensor: torch.Tensor,
    target_shape: Tuple[int, int, int],
    input_format: str = "ZYXC",
    mode: str = "trilinear",
    align_corners: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, Tuple[float, float, float]]:
    """
    Resize a 3D tensor to target spatial shape.

    Args:
        tensor: Input tensor of shape (B, Z, Y, X, C) for ZYXC format
        target_shape: Target spatial shape (Z, Y, X)
        input_format: Data format, currently only "ZYXC" supported
        mode: F.interpolate mode -- "trilinear" (default), "area" or "nearest-exact".
            Plain "nearest" is rejected (edge-aligned; see comment below).
        align_corners: Whether to align corners in interpolation
        dtype: Optional dtype to cast to before resize

    Returns:
        Tuple of (resized_tensor, scale_factors)
        scale_factors is (sz, sy, sx) where s = new_size / old_size
    """
    if input_format.upper() != "ZYXC":
        raise ValueError(f"resize_tensor_3d only supports ZYXC, got {input_format}")

    if tensor.ndim != 5:
        raise ValueError(f"Expected 5D tensor (B, Z, Y, X, C), got shape {tensor.shape}")

    B, Z, Y, X, C = tensor.shape
    tZ, tY, tX = target_shape

    if dtype is not None:
        tensor = tensor.to(dtype)

    # (B, Z, Y, X, C) -> (B, C, Z, Y, X)
    x_cf = tensor.permute(0, 4, 1, 2, 3).contiguous()

    # Image resize convention: voxel-CENTER aligned. `trilinear`/`area` with
    # align_corners=False sample output voxel i at input coord (i + 0.5) * s - 0.5,
    # and GT (label maps / masks) go through `_resize_nearest_exact_3d`, i.e.
    # src = floor((i + 0.5) * s). Legacy `"nearest"` is src = floor(i * s): the
    # LEFT EDGE of the voxel, i.e. shifted s/2 input voxels (= half an output
    # voxel) toward the origin relative to GT. Using it for the image would
    # misregister image and GT by up to one voxel at 2x downsample. We refuse it
    # rather than rewriting it so the config says what actually runs.
    if mode == "nearest":
        raise ValueError(
            "resize_tensor_3d: mode='nearest' is edge-aligned (floor(i*s)) and is "
            "half a voxel off the center-aligned GT resize; use mode='nearest-exact' "
            "(floor((i+0.5)*s)), which matches trilinear/align_corners=False and "
            "the label-map/mask resize."
        )

    if mode in ("nearest-exact", "area"):
        x_cf = F.interpolate(x_cf, size=target_shape, mode=mode)
    else:  # linear family takes align_corners
        x_cf = F.interpolate(x_cf, size=target_shape, mode=mode, align_corners=align_corners)

    resized = x_cf.permute(0, 2, 3, 4, 1)

    scale_factors = (tZ / float(Z), tY / float(Y), tX / float(X))

    return resized, scale_factors


def _nearest_exact_indices(in_size: int, out_size: int, device: torch.device) -> torch.Tensor:
    """Source indices for a 1D nearest-exact resize (``F.interpolate``'s
    ``mode="nearest-exact"``): ``floor((i + 0.5) * in/out)`` clamped -- the
    voxel-center convention. Plain ``"nearest"`` uses ``floor(i * in/out)``,
    which shifts GT by half a voxel relative to the (center-aligned) trilinear
    image resize."""
    scale = in_size / out_size
    idx = ((torch.arange(out_size, device=device, dtype=torch.float64) + 0.5) * scale).floor()
    return idx.long().clamp_(0, in_size - 1)


def _resize_nearest_exact_3d(
    vol: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Nearest-exact resize of the trailing (Z, Y, X) axes via index gather.

    Dtype-preserving for ANY dtype (int32/int64/bool/uint16 ids included):
    ``F.interpolate`` has no integer kernels, and the old float round-trip both
    cost a copy and risked precision on ids beyond float32's 24-bit mantissa.
    Index selection is bit-exactly equivalent to ``mode="nearest-exact"``.
    """
    Z, Y, X = (int(s) for s in vol.shape[-3:])
    tZ, tY, tX = (int(s) for s in target_shape)
    iz = _nearest_exact_indices(Z, tZ, vol.device)
    iy = _nearest_exact_indices(Y, tY, vol.device)
    ix = _nearest_exact_indices(X, tX, vol.device)
    return vol.index_select(-3, iz).index_select(-2, iy).index_select(-1, ix)


def resize_masks(
    masks: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Resize binary/stacked label masks with nearest-exact interpolation
    (dtype-preserving; no float cast).

    Args:
        masks: Tensor of shape (..., Z, Y, X) with at least one leading axis --
            (N, Z, Y, X) binary masks / labelmap slices, or a (B, T, Z, Y, X)
            padding mask under a TZYXC layout. Leading axes are untouched.
        target_shape: Target spatial shape (Z, Y, X)

    Returns:
        Resized masks tensor of shape (..., tZ, tY, tX)
    """
    if masks.ndim >= 4:
        return _resize_nearest_exact_3d(masks, target_shape)
    else:
        raise ValueError(f"Unsupported masks ndim={masks.ndim}; expected >= 4 dims (..., Z, Y, X).")


def resize_label_map(
    label_map: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Resize label map with nearest-exact interpolation, preserving integer labels
    exactly (no float round-trip).

    Args:
        label_map: Tensor of shape (Z, Y, X) - single instance label map
        target_shape: Target spatial shape (Z, Y, X)

    Returns:
        Resized label map tensor of shape (tZ, tY, tX)
    """
    if label_map.ndim == 3:
        return _resize_nearest_exact_3d(label_map, target_shape)
    else:
        raise ValueError(f"Unsupported label_map ndim={label_map.ndim}; expected 3 dims.")


def resize_boxes(
    boxes: torch.Tensor,
    scale_factors: Tuple[float, float, float],
    bbox_format: str,
) -> torch.Tensor:
    """
    Scale boxes by given factors.

    Args:
        boxes: Tensor of shape (N, 6) containing bounding boxes
        scale_factors: Tuple (sz, sy, sx) scale factors for Z, Y, X
        bbox_format: Format of boxes - "zyxzyx" or "cxcyczwhd"

    Returns:
        Scaled boxes tensor
    """
    if boxes.numel() == 0:
        return boxes

    sz, sy, sx = scale_factors
    out = boxes.clone()

    if out.shape[-1] != 6:
        raise ValueError(f"Expected boxes with 6 coords, got {out.shape[-1]}")

    fmt = bbox_format.lower()

    if fmt == "zyxzyx":
        # [z1, y1, x1, z2, y2, x2]
        out[..., 0] = out[..., 0] * sz  # z1
        out[..., 1] = out[..., 1] * sy  # y1
        out[..., 2] = out[..., 2] * sx  # x1
        out[..., 3] = out[..., 3] * sz  # z2
        out[..., 4] = out[..., 4] * sy  # y2
        out[..., 5] = out[..., 5] * sx  # x2

    elif fmt == "cxcyczwhd":
        # [cx, cy, cz, w, h, d]
        out[..., 0] = out[..., 0] * sx  # cx
        out[..., 1] = out[..., 1] * sy  # cy
        out[..., 2] = out[..., 2] * sz  # cz
        out[..., 3] = out[..., 3] * sx  # w
        out[..., 4] = out[..., 4] * sy  # h
        out[..., 5] = out[..., 5] * sz  # d

    else:
        raise ValueError(f"Unsupported bbox_format={bbox_format!r}")

    return out


def resize_padding_mask(
    padding_mask: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Resize padding mask with nearest neighbor interpolation.

    Args:
        padding_mask: Tensor of shape (B, Z, Y, X)
        target_shape: Target spatial shape (Z, Y, X)

    Returns:
        Resized padding mask tensor
    """
    orig_dtype = padding_mask.dtype
    if padding_mask.ndim == 4:
        # (B, Z, Y, X) -> (B, 1, Z, Y, X)
        m = padding_mask.to(torch.float32).unsqueeze(1)
        m = F.interpolate(m, size=target_shape, mode="nearest")
        m = m.squeeze(1)
        return m.to(orig_dtype)
    else:
        raise ValueError(f"Unsupported padding_mask ndim={padding_mask.ndim}; expected 4 dims.")
