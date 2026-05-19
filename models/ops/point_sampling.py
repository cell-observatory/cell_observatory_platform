"""Shared point-sampling helpers used by mask-loss decoders (MaskDINO, SAM2).

Lives outside of `models/layers/utils.py` and `models/ops/losses.py` because it
composes primitives from both modules into a single PointRend-style helper, so
mask decoders do not each re-implement the same boilerplate.

Coordinate convention follows `torch.nn.functional.grid_sample` for 5D inputs
(N, C, D, H, W): normalized `point_coords[..., 0]` indexes the W axis (x),
`[..., 1]` indexes the H axis (y), `[..., 2]` indexes the D axis (z). See
[`models/layers/utils.py`](../layers/utils.py) `point_sample`,
`point_sample_labelmap_batched`.
"""
from __future__ import annotations

from typing import Tuple

import torch

from cell_observatory_platform.models.layers.utils import (
    get_uncertain_point_coords_with_randomness,
    point_sample_labelmap_batched,
)
from cell_observatory_platform.models.ops.losses import calculate_uncertainty


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
