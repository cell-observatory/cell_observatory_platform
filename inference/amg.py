"""
Adapted from: 
https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py
"""

import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Generator, ItemsView, List, Tuple

import numpy as np
import torch

from cell_observatory_platform.data.structures import (
    box_iou_3d,
    box_volume,
    is_box_near_crop_edge_3d,
    masks_to_boxes_v2,
    nms_3d,
    uncrop_boxes_3d,
    uncrop_masks_3d,
    uncrop_points_3d,
)


class MaskData:
    """
    Stores masks and related data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, **kwargs) -> None:
        for v in kwargs.values():
            assert isinstance(
                v, (list, np.ndarray, torch.Tensor)
            ), "MaskData only supports list, numpy arrays, and torch tensors."
        self._stats = dict(**kwargs)

    def to_cpu(self) -> None:
        """Move all tensors to CPU to save GPU memory."""
        for k in self._stats:
            v = self._stats[k]
            if isinstance(v, torch.Tensor) and v.is_cuda:
                self._stats[k] = v.detach().cpu()

    def __setitem__(self, key: str, item: Any) -> None:
        assert isinstance(
            item, (list, np.ndarray, torch.Tensor)
        ), "MaskData only supports list, numpy arrays, and torch tensors."
        self._stats[key] = item

    def __delitem__(self, key: str) -> None:
        del self._stats[key]

    def __getitem__(self, key: str) -> Any:
        return self._stats[key]

    def __len__(self) -> int:
        for v in self._stats.values():
            if hasattr(v, "__len__"):
                return len(v)
        return 0

    def items(self) -> ItemsView[str, Any]:
        return self._stats.items()

    def filter(self, keep: torch.Tensor) -> None:
        for k, v in self._stats.items():
            if v is None:
                self._stats[k] = None
            elif isinstance(v, torch.Tensor):
                self._stats[k] = v[torch.as_tensor(keep, device=v.device)]
            elif isinstance(v, np.ndarray):
                self._stats[k] = v[keep.detach().cpu().numpy()]
            elif isinstance(v, list) and keep.dtype == torch.bool:
                self._stats[k] = [a for i, a in enumerate(v) if keep[i]]
            elif isinstance(v, list):
                self._stats[k] = [v[i] for i in keep]
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def cat(self, new_stats: "MaskData") -> None:
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = deepcopy(v)
            elif isinstance(v, torch.Tensor):
                self._stats[k] = torch.cat([self._stats[k], v], dim=0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + deepcopy(v)
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")

    def to_numpy(self) -> None:
        for k, v in self._stats.items():
            if isinstance(v, torch.Tensor):
                self._stats[k] = v.float().detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 3D point grid generation
# ---------------------------------------------------------------------------

def build_point_grid_3d(n_per_side: int) -> np.ndarray:
    """
    Generates a 3D grid of points evenly spaced in [0,1]^3.
    Returns array of shape (n_per_side**3, 3) with columns (x, y, z).
    """
    offset = 1 / (2 * n_per_side)
    coords_1d = np.linspace(offset, 1 - offset, n_per_side)
    # meshgrid in z, y, x order then stack as (x, y, z)
    gz, gy, gx = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)  # (N, 3)
    return points


def build_all_layer_point_grids_3d(
    n_per_side: int,
    n_layers: int,
    scale_per_layer: int,
) -> List[np.ndarray]:
    """Generates 3D point grids for all crop layers."""
    grids = []
    for i in range(n_layers + 1):
        n_pts = max(int(n_per_side / (scale_per_layer ** i)), 1)
        grids.append(build_point_grid_3d(n_pts))
    return grids


# ---------------------------------------------------------------------------
# 3D crop box generation
# ---------------------------------------------------------------------------

def generate_crop_boxes_3d(
    vol_size: Tuple[int, int, int],
    n_layers: int,
    overlap_ratio: float,
) -> Tuple[List[List[int]], List[int]]:
    """
    Generates 3D crop boxes at multiple scales.
    vol_size: (Z, Y, X)
    Returns crop_boxes as list of [x0, y0, z0, x1, y1, z1] and layer indices.
    """
    crop_boxes, layer_idxs = [], []
    vol_z, vol_y, vol_x = vol_size
    short_side = min(vol_z, vol_y, vol_x)

    crop_boxes.append([0, 0, 0, vol_x, vol_y, vol_z])
    layer_idxs.append(0)

    def _crop_len(orig_len, n_crops, overlap):
        return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))

    for i_layer in range(n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_x = _crop_len(vol_x, n_crops_per_side, overlap)
        crop_y = _crop_len(vol_y, n_crops_per_side, overlap)
        crop_z = _crop_len(vol_z, n_crops_per_side, overlap)

        starts_x = [int((crop_x - overlap) * i) for i in range(n_crops_per_side)]
        starts_y = [int((crop_y - overlap) * i) for i in range(n_crops_per_side)]
        starts_z = [int((crop_z - overlap) * i) for i in range(n_crops_per_side)]

        for x0, y0, z0 in product(starts_x, starts_y, starts_z):
            box = [
                x0, y0, z0,
                min(x0 + crop_x, vol_x),
                min(y0 + crop_y, vol_y),
                min(z0 + crop_z, vol_z),
            ]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


# ---------------------------------------------------------------------------
# Stability score
# ---------------------------------------------------------------------------

def calculate_stability_score_3d(
    masks: torch.Tensor,
    mask_threshold: float,
    threshold_offset: float,
) -> torch.Tensor:
    """
    Stability score = IoU between masks thresholded at (t+offset) and (t-offset).
    masks: (N, Z, Y, X) logits (pre-threshold).
    """
    high = (masks > (mask_threshold + threshold_offset)).sum(dim=(-3, -2, -1)).float()
    low = (masks > (mask_threshold - threshold_offset)).sum(dim=(-3, -2, -1)).float()
    return high / low.clamp(min=1)


# ---------------------------------------------------------------------------
# Batch iterator
# ---------------------------------------------------------------------------

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n = len(args[0])
    n_batches = (n + batch_size - 1) // batch_size
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


# ---------------------------------------------------------------------------
# Remove small regions
# ---------------------------------------------------------------------------

def remove_small_regions_3d(
    mask: np.ndarray,
    volume_thresh: float,
    mode: str,
) -> Tuple[np.ndarray, bool]:
    """
    Remove small disconnected regions or fill small holes in a 3D binary mask.
    Returns the cleaned mask and whether it was modified.

    Args:
        mask: (Z, Y, X) bool/uint8 array
        volume_thresh: minimum region volume in voxels
        mode: "holes" to fill small holes, "islands" to remove small foreground components
    """
    import cc3d

    assert mode in ("holes", "islands"), f"Mode {mode} not supported"
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)

    labels, n_labels = cc3d.connected_components(working_mask, connectivity=26, return_N=True)
    if n_labels == 0:
        return mask, False

    # label 0 is background; component labels are 1..n_labels
    sizes = np.bincount(labels.ravel())[1:]  # drop background count
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < volume_thresh]
    if len(small_regions) == 0:
        return mask, False

    fill_labels = set([0] + small_regions)
    if not correct_holes:
        fill_labels = set(range(n_labels + 1)) - fill_labels
        if len(fill_labels) == 0:
            fill_labels = {int(np.argmax(sizes)) + 1}

    mask = np.isin(labels, list(fill_labels))
    return mask, True


def postprocess_sam_preds(
    preds: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize SAM2 AMG ``predict()`` outputs for the inference save/viz path.

    Returns a ``dict[str, Tensor]`` (NOT a list) so it matches the inferencer's
    ``model.predict() -> dict`` contract. ``masks`` are reshaped to a leading
    (object, channel, ...) layout and a fused ``instance_masks`` label volume is
    added. 
    """
    # Coerce array-like outputs to torch tensors so they satisfy the inferencer
    # save/viz path (dtype cast via .to(), host pinned-copy). SAM2.predict runs
    # MaskData.to_numpy(), so values arrive as ndarrays here.
    p: Dict[str, Any] = {}
    for key, value in preds.items():
        if isinstance(value, np.ndarray):
            p[key] = torch.from_numpy(value)
        else:
            p[key] = value

    m = p.get("masks")
    if m is not None:
        if not isinstance(m, torch.Tensor):
            m = torch.as_tensor(np.asarray(m))
        m = m.float()
        # (N, Z, Y, X, 1) ZYXC -- the instance_stack contract declared in SAM2's
        # output_metadata. 
        if m.ndim == 3:
            m = m[None, ..., None]
        elif m.ndim == 4:
            m = m[..., None]
        p["masks"] = m

        # Fused instance label map for the dense save path. The per-object
        # ``masks`` (N, 1, Z, Y, X) do not fit the batch-oriented save/buffer
        # contract (N is a dynamic object count, not a batch axis), so collapse
        # them into a single fixed-shape (1, Z, Y, X, 1) integer label volume
        # (leading B=1, trailing C=1, ZYXC) where each voxel holds an object id
        # in [1, N] (0 = background). Objects are painted in ascending score
        # order so the highest-scoring object wins any overlap.
        p["instance_masks"] = _fuse_masks_to_label_map(m, p)

    return p


def _fuse_masks_to_label_map(
    masks: torch.Tensor,
    preds: Dict[str, Any],
) -> torch.Tensor:
    """Collapse ``(N, Z, Y, X, 1)`` per-object masks into ``(1, Z, Y, X, 1)`` labels.

    Object ids are assigned in ``[1, N]`` (0 = background). Objects are painted
    in ascending score order (``stability_score`` if present, else ``iou_preds``,
    else input order), so the highest-scoring object wins voxels claimed by more
    than one mask. Returns a ``uint16`` tensor (object counts can exceed 255).
    """
    n = masks.shape[0]
    assert n <= 65535, (
        f"_fuse_masks_to_label_map: object count {n} exceeds uint16 max (65535). "
        "The fused instance_masks label map uses uint16 dtype; more than 65535 "
        "objects would cause id wrapping."
    )
    z, y, x = masks.shape[1:4]
    label_map = torch.zeros((z, y, x), dtype=torch.int32)

    if n > 0:
        binary = masks[..., 0] > 0.5  # (N, Z, Y, X, 1) -> (N, Z, Y, X) bool

        scores = preds.get("stability_score")
        if scores is None:
            scores = preds.get("iou_preds")
        if scores is not None and not isinstance(scores, torch.Tensor):
            scores = torch.as_tensor(np.asarray(scores))
        if scores is not None and scores.numel() == n:
            order = torch.argsort(scores.flatten().float())  # ascending
        else:
            order = torch.arange(n)

        for obj_id, idx in enumerate(order.tolist(), start=1):
            label_map[binary[idx]] = obj_id

    # (Z, Y, X) -> (1, Z, Y, X, 1): leading B=1, trailing C=1 (ZYXC).
    return label_map.to(torch.uint16)[None, ..., None]