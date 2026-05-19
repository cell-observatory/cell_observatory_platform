"""Shared point-sampling helpers used by mask-loss decoders (MaskDINO, SAM2).

Lives outside of `models/layers/utils.py` and `models/ops/losses.py` because it
composes primitives from both modules into a single PointRend-style helper, so
mask decoders do not each re-implement the same boilerplate.

Coordinate convention follows `torch.nn.functional.grid_sample` for 5D inputs
(N, C, D, H, W): normalized `point_coords[..., 0]` indexes the W axis (x),
`[..., 1]` indexes the H axis (y), `[..., 2]` indexes the D axis (z). See
[`models/layers/utils.py`](../layers/utils.py) `point_sample`,
`point_sample_labelmap_batched`.

Box-prompt sampler emits pixel `(x, y, z)` corners to match SAM2's prompt
encoder convention; format conversion happens at the boundary so callers do
not silently infer.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch

from cell_observatory_platform.models.layers.utils import (
    get_uncertain_point_coords_with_randomness,
    point_sample_labelmap_batched,
    sample_one_point_from_error_center,
    sample_random_points_from_errors,
)
from cell_observatory_platform.models.ops.losses import calculate_uncertainty

BoxFormat = Literal["xyzxyz", "zyxzyx", "cxcyczwhd"]
PromptMethod = Literal["uniform", "center"]


@torch.no_grad()
def sample_uncertain_points_and_labelmap_labels(
    src_logits: torch.Tensor,        # [N, 1, Z, Y, X] coarse predicted logits per row
    labelmap: torch.Tensor,          # [B, Z, Y, X] integer instance labelmap
    batch_indices: torch.Tensor,     # [N] which labelmap batch each row belongs to
    instance_ids: torch.Tensor,      # [N] labelmap id per row
    num_points: int,
    oversample_ratio: float = 3.0,
    importance_sample_ratio: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample uncertain points from `src_logits` and label them via labelmap lookup.

    Combines `get_uncertain_point_coords_with_randomness` with
    `point_sample_labelmap_batched`. Both pieces share the grid_sample
    `(x, y, z)` coordinate convention; this helper makes the composition explicit
    so each decoder does not redo the same import/glue.

    Args:
        src_logits: `[N, 1, Z, Y, X]` coarse logits, one channel per row.
            Multimask decoders should reshape `[N, M, Z, Y, X]` to
            `[N*M, 1, Z, Y, X]` and replicate `batch_indices`/`instance_ids`
            `M` times before calling.
        labelmap: `[B, Z, Y, X]` integer labelmap. Pad rows in
            `instance_ids` (e.g. sentinel `-1`) are safe; no labelmap voxel
            matches a sentinel, so their sampled labels are deterministically 0.

    Returns:
        point_coords: `[N, num_points, 3]` normalized `(x, y, z)` coords.
        point_labels: `[N, num_points]` float, 1 where the sampled labelmap voxel
            equals the row's `instance_ids` value, 0 elsewhere.
    """
    point_coords = get_uncertain_point_coords_with_randomness(
        src_logits,
        lambda logits: calculate_uncertainty(logits),
        num_points,
        oversample_ratio,
        importance_sample_ratio,
    )
    point_labels = point_sample_labelmap_batched(
        labelmap=labelmap,
        point_coords=point_coords,
        batch_indices=batch_indices,
        instance_ids=instance_ids,
    )
    return point_coords, point_labels


def _to_xyzxyz(boxes: torch.Tensor, box_format: BoxFormat) -> torch.Tensor:
    """Convert [N, 6] boxes from `box_format` to (x1, y1, z1, x2, y2, z2)."""
    if box_format == "xyzxyz":
        return boxes
    if box_format == "zyxzyx":
        # (z1, y1, x1, z2, y2, x2) -> (x1, y1, z1, x2, y2, z2)
        return boxes[..., [2, 1, 0, 5, 4, 3]]
    if box_format == "cxcyczwhd":
        cx, cy, cz, w, h, d = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        z1 = cz - d / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        z2 = cz + d / 2
        return torch.stack([x1, y1, z1, x2, y2, z2], dim=-1)
    raise ValueError(f"unknown box_format {box_format!r}")


def sample_box_points_from_boxes(
    boxes: torch.Tensor,                  # [N, 6] target boxes
    box_format: BoxFormat,
    image_shape: Tuple[int, int, int],    # (Z, Y, X) pixel extents
    valid: Optional[torch.Tensor] = None,  # [N] bool, pad rows zeroed
    noise: float = 0.1,
    noise_bound: float = 20.0,
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample noised SAM box-corner prompts from pre-existing target boxes.

    SAM2 historically derived boxes from dense binary masks via
    masks_to_boxes_v2 inside `sample_box_points`. With labelmap-native
    targets, the bbox is already known per object and we can avoid the
    `[N, 1, Z, Y, X]` mask materialization entirely.

    Output mirrors `models/layers/utils.py:sample_box_points`:
        points: `[N, 2, 3]` pixel `(x, y, z)`, top-left then bottom-right.
        labels: `[N, 2]` int32 with `[top_left_label, bottom_right_label]`.

    Padded rows (`valid[n] == False`) get all-zero points and label 0 so
    the SAM2 prompt encoder skips them.
    """
    if boxes.dim() != 2 or boxes.shape[-1] != 6:
        raise ValueError(f"boxes must be [N, 6], got {tuple(boxes.shape)}")
    N = boxes.shape[0]
    device = boxes.device
    Z, Y, X = image_shape

    box_coords = _to_xyzxyz(boxes, box_format).to(torch.float32).clone()

    if noise > 0.0 and N > 0:
        bbox_w = box_coords[..., 3] - box_coords[..., 0]
        bbox_h = box_coords[..., 4] - box_coords[..., 1]
        bbox_d = box_coords[..., 5] - box_coords[..., 2]
        nb = torch.tensor(float(noise_bound), device=device, dtype=box_coords.dtype)
        max_dx = torch.minimum(bbox_w * noise, nb)
        max_dy = torch.minimum(bbox_h * noise, nb)
        max_dz = torch.minimum(bbox_d * noise, nb)
        # [N, 6] noise in [-1, 1] scaled by per-axis caps
        box_noise = 2.0 * torch.rand(N, 6, device=device, dtype=box_coords.dtype) - 1.0
        box_noise = box_noise * torch.stack(
            [max_dx, max_dy, max_dz, max_dx, max_dy, max_dz], dim=-1
        )
        box_coords = box_coords + box_noise

    img_bounds = torch.tensor(
        [X - 1, Y - 1, Z - 1, X - 1, Y - 1, Z - 1],
        device=device,
        dtype=box_coords.dtype,
    )
    box_coords = box_coords.clamp(torch.zeros_like(img_bounds), img_bounds)

    points = box_coords.reshape(N, 2, 3)  # top-left, bottom-right pixel (x, y, z)
    labels = torch.tensor(
        [top_left_label, bottom_right_label], dtype=torch.int32, device=device
    ).view(1, 2).expand(N, 2).contiguous()

    if valid is not None:
        if valid.shape != (N,):
            raise ValueError(f"valid must be [N], got {tuple(valid.shape)}")
        invalid = ~valid
        points = points.masked_fill(invalid[:, None, None], 0.0)
        # SAM2 uses 0 as a neutral/ignored label; non-{2, 3} won't trip the
        # corner-specific positional encoding for padded slots.
        labels = labels.masked_fill(invalid[:, None], 0)

    return points, labels


def gt_masks_from_labelmap(
    labelmap: torch.Tensor,        # [B, Z, Y, X] integer instance labelmap
    img_ids: torch.Tensor,         # [N] index into labelmap's batch axis
    instance_ids: torch.Tensor,    # [N] labelmap id per row (-1 = pad)
) -> torch.Tensor:                 # [N, 1, Z, Y, X] bool
    """Materialize per-row binary GT masks from the labelmap target view.

    The SAM2 loss path samples labels directly from the labelmap without
    materializing dense GT. The SAM2 prompt path (initial click + error-region
    correction click) still expects `[N, 1, Z, Y, X]` dense masks because
    `sample_random_points_from_errors` / `sample_one_point_from_error_center`
    reason about FP/FN regions per voxel. This helper bridges the two so the
    preprocessor can stop emitting `data_views["masks"]` once the prompt
    helpers consume labelmap-driven inputs end to end.

    Pad rows (`instance_ids == -1`) yield all-False masks because the sentinel
    cannot match any non-negative labelmap voxel.
    """
    if labelmap.dim() != 4:
        raise ValueError(f"labelmap must be [B, Z, Y, X], got {tuple(labelmap.shape)}")
    if img_ids.shape != instance_ids.shape:
        raise ValueError(
            f"img_ids {tuple(img_ids.shape)} and instance_ids "
            f"{tuple(instance_ids.shape)} must match"
        )
    per_row = labelmap[img_ids.to(torch.long)]
    return (per_row == instance_ids.to(per_row.dtype)[:, None, None, None]).unsqueeze(1)


def sample_prompt_point_from_labelmap(
    labelmap: torch.Tensor,         # [B, Z, Y, X]
    img_ids: torch.Tensor,          # [N]
    instance_ids: torch.Tensor,     # [N], -1 for pad
    pred_masks: Optional[torch.Tensor] = None,  # [N, 1, Z, Y, X] bool, or None
    *,
    input_fmt: str = "TZYXC",
    time_separable: bool = True,
    method: PromptMethod = "uniform",
    num_pt: int = 1,
    exact_edt_for_eval: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample SAM2 prompt clicks driven by the labelmap target view.

    Replaces the dense-GT call to `get_next_point` at SAM2 prompt sites
    with a labelmap-first interface. GT masks are materialized internally
    via `gt_masks_from_labelmap` (cheap: this is a small slice the SAM
    decoder consumes anyway), then the existing point-sampling primitives
    run on top.

    Args:
        method: "uniform" samples a uniformly random voxel inside the error
            region (`gt XOR pred`); "center" picks the voxel with the largest
            distance from the error-region boundary.
        exact_edt_for_eval: only consulted when `method == "center"`. When
            False (training default), the center request is downgraded to
            "uniform" to keep the prompt path on GPU. When True, scipy's
            `distance_transform_edt` runs on CPU per row to produce the exact
            center -- preferred at eval where determinism dominates throughput.

    Returns:
        points: [N, num_pt, 3] pixel (x, y, z) float coords.
        labels: [N, num_pt] int32 click labels (1 positive / 0 negative).
    """
    gt_masks = gt_masks_from_labelmap(labelmap, img_ids, instance_ids)
    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)

    if method == "uniform":
        return sample_random_points_from_errors(
            input_fmt=input_fmt,
            gt_masks=gt_masks,
            pred_masks=pred_masks,
            time_separable=time_separable,
            num_pt=num_pt,
        )
    if method == "center":
        if not exact_edt_for_eval:
            # Training path: pass `num_pt=1` and route through the uniform
            # sampler. scipy distance_transform_edt is a CPU host->device round
            # trip per row, which dominates step time inside the tracking loop.
            return sample_random_points_from_errors(
                input_fmt=input_fmt,
                gt_masks=gt_masks,
                pred_masks=pred_masks,
                time_separable=time_separable,
                num_pt=max(1, num_pt),
            )
        # Eval path: exact center via scipy EDT; only num_pt=1 supported.
        if num_pt != 1:
            raise ValueError(
                f"method='center' with exact_edt_for_eval=True only supports "
                f"num_pt=1, got num_pt={num_pt}"
            )
        return sample_one_point_from_error_center(
            input_fmt=input_fmt,
            gt_masks=gt_masks,
            pred_masks=pred_masks,
            time_separable=time_separable,
        )
    raise ValueError(f"unknown method {method!r}; expected 'uniform' or 'center'")
