import random
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


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
    return (
        random.randint(target_min[0], target_max[0]),
        random.randint(target_min[1], target_max[1]),
        random.randint(target_min[2], target_max[2]),
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
        mode: Interpolation mode for F.interpolate
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

    if mode in ("nearest", "area"):
        x_cf = F.interpolate(x_cf, size=target_shape, mode=mode)
    else:
        x_cf = F.interpolate(x_cf, size=target_shape, mode=mode, align_corners=align_corners)

    # (B, C, Z, Y, X) -> (B, Z, Y, X, C)
    resized = x_cf.permute(0, 2, 3, 4, 1).contiguous()

    scale_factors = (tZ / float(Z), tY / float(Y), tX / float(X))

    return resized, scale_factors


def resize_masks(
    masks: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Resize binary masks with nearest neighbor interpolation.

    Args:
        masks: Tensor of shape (N, Z, Y, X) - N binary masks
        target_shape: Target spatial shape (Z, Y, X)

    Returns:
        Resized masks tensor of shape (N, tZ, tY, tX)
    """
    orig_dtype = masks.dtype
    if masks.ndim == 4:
        # (N, Z, Y, X) -> (N, 1, Z, Y, X)
        m = masks.unsqueeze(1).float()
        m = F.interpolate(m, size=target_shape, mode="nearest")
        m = m.squeeze(1)
        return m.to(orig_dtype)
    else:
        raise ValueError(f"Unsupported masks ndim={masks.ndim}; expected 4 dims.")


def resize_label_map(
    label_map: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Resize label map with nearest neighbor interpolation to preserve integer labels.

    Args:
        label_map: Tensor of shape (Z, Y, X) - single instance label map
        target_shape: Target spatial shape (Z, Y, X)

    Returns:
        Resized label map tensor of shape (tZ, tY, tX)
    """
    orig_dtype = label_map.dtype
    if label_map.ndim == 3:
        # (Z, Y, X) -> (1, 1, Z, Y, X)
        m = label_map.unsqueeze(0).unsqueeze(0).float()
        m = F.interpolate(m, size=target_shape, mode="nearest")
        m = m.squeeze(0).squeeze(0)
        return m.to(orig_dtype)
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
