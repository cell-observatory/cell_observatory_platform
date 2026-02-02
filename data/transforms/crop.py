import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_platform.data.transforms.utils import (
    parse_target_shape_range,
    resize_boxes,
    resize_label_map,
    resize_masks,
    resize_tensor_3d,
    sample_target_shape,
)


# TODO: generalize to N-D
class Crop3D:
    """
    3D cropping transform with probabilistic mode selection.

    Modes:
      - "crop": Crop to target size (no resize). If input is smaller than target,
                automatically falls back to crop_resize behavior.
      - "crop_resize": Crop to intermediate size, then resize to target.

    Selection between modes is probabilistic via mode_probs dict.

    Supports:
      - input_format="ZYXC": tensor shape (B, Z, Y, X, C)

    Can be called on a data_sample dict with keys:
      - "data_tensor": image tensor
      - "metainfo": dict containing "targets" (list of dicts with masks/boxes/label_map)
    """

    def __init__(
        self,
        input_format: str,
        target_spatial_shape: Union[Sequence[int], Tuple[Sequence[int], Sequence[int]]],
        crop_dims: str = "YX",
        crop_type: str = "random",
        mode_probs: Optional[Dict[str, float]] = None,
        bbox_format: Optional[str] = None,
        dtype: str = "bfloat16",
        patch_size: Optional[Tuple[int, int, int]] = None,
        resize_mode: str = "trilinear",
        align_corners: bool = False,
    ) -> None:
        """
        Args:
            input_format: Data layout format, currently only "ZYXC" supported.
            target_spatial_shape: Final output spatial shape. Either:
                - Fixed: (Z, Y, X) or [Z, Y, X]
                - Range: ((Z_min, Y_min, X_min), (Z_max, Y_max, X_max))
                  Samples uniformly from range on each call.
            crop_dims: Which dims to crop, e.g., "YX", "ZYX", "X".
            crop_type: How to select crop region - "random", "center", or "start".
            mode_probs: Dict mapping mode names to probabilities.
                        Modes: "crop" (crop only), "crop_resize" (crop then resize).
                        Default is {"crop": 1.0}.
                        If input is too small for "crop" mode, automatically falls back
                        to crop_resize behavior.
            bbox_format: Format of bounding boxes - "zyxzyx".
            dtype: Data type for output tensor.
            patch_size: If set, pad output to multiple of this size.
            resize_mode: Interpolation mode for resize operations.
            align_corners: Whether to align corners in resize interpolation.
        """
        self.input_format = input_format.upper()
        if self.input_format != "ZYXC":
            raise ValueError(f"Crop3D only supports input_format='ZYXC', got {input_format}")

        # Parse target shape (fixed or range)
        self.target_min, self.target_max, self.random_target = parse_target_shape_range(
            target_spatial_shape
        )

        self.crop_dims = crop_dims.upper()
        self.crop_type = crop_type.lower()
        if self.crop_type not in ("random", "center", "start"):
            raise ValueError(f"crop_type must be 'random', 'center', or 'start', got {crop_type}")

        self.bbox_format = bbox_format
        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        self.patch_size = tuple(patch_size) if patch_size is not None else None
        
        self.resize_mode = resize_mode
        self.align_corners = align_corners

        # Default: crop-only if no probs specified
        self.mode_probs = mode_probs or {"crop": 1.0}
        self._validate_mode_probs()

    def _validate_mode_probs(self) -> None:
        """Validate that mode_probs contains valid modes and sums to 1."""
        valid_modes = {"crop", "crop_resize"}
        for m in self.mode_probs:
            if m not in valid_modes:
                raise ValueError(f"Unknown mode '{m}', expected one of {valid_modes}")
        total = sum(self.mode_probs.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"mode_probs must sum to 1.0, got {total}")

    def _sample_target_shape(self) -> Tuple[int, int, int]:
        """Sample target shape (uniform from range, or fixed if no range)."""
        return sample_target_shape(self.target_min, self.target_max)

    def _sample_mode(self) -> str:
        """Sample mode according to mode_probs distribution."""
        r = random.random()
        cumsum = 0.0
        for mode, prob in self.mode_probs.items():
            cumsum += prob
            if r < cumsum:
                return mode
        return list(self.mode_probs.keys())[-1]

    def _compute_crop_region(
        self,
        current_shape: Tuple[int, int, int],
        target_shape: Tuple[int, int, int],
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Compute crop start offsets and crop size based on crop_type and crop_dims.

        Args:
            current_shape: Current spatial shape (Z, Y, X)
            target_shape: Target spatial shape for this call (Z, Y, X)

        Returns:
            Tuple of (offsets (oz, oy, ox), crop_size (cz, cy, cx))
        """
        Z, Y, X = current_shape
        tZ, tY, tX = target_shape

        # Determine which dims to crop
        crop_z = "Z" in self.crop_dims
        crop_y = "Y" in self.crop_dims
        crop_x = "X" in self.crop_dims

        if self.crop_type == "random":
            # Crop to target size (clamped to input size), with random offset
            crop_Z = min(tZ, Z) if crop_z else Z
            crop_Y = min(tY, Y) if crop_y else Y
            crop_X = min(tX, X) if crop_x else X

            # Random offsets within valid range
            off_z = random.randint(0, max(0, Z - crop_Z)) if crop_z else 0
            off_y = random.randint(0, max(0, Y - crop_Y)) if crop_y else 0
            off_x = random.randint(0, max(0, X - crop_X)) if crop_x else 0

        elif self.crop_type == "center":
            # Center crop to target size (clamped to input size)
            crop_Z = min(tZ, Z) if crop_z else Z
            crop_Y = min(tY, Y) if crop_y else Y
            crop_X = min(tX, X) if crop_x else X

            off_z = (Z - crop_Z) // 2 if crop_z else 0
            off_y = (Y - crop_Y) // 2 if crop_y else 0
            off_x = (X - crop_X) // 2 if crop_x else 0

        elif self.crop_type == "start":
            # Crop from start (top-left-front corner)
            crop_Z = min(tZ, Z) if crop_z else Z
            crop_Y = min(tY, Y) if crop_y else Y
            crop_X = min(tX, X) if crop_x else X
            off_z, off_y, off_x = 0, 0, 0

        else:
            raise ValueError(f"Unknown crop_type: {self.crop_type}")

        return (off_z, off_y, off_x), (crop_Z, crop_Y, crop_X)

    def _crop_tensor(
        self,
        tensor: torch.Tensor,
        offsets: Tuple[int, int, int],
        crop_size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Crop (B, Z, Y, X, C) tensor.

        Args:
            tensor: Input tensor of shape (B, Z, Y, X, C)
            offsets: Start offsets (oz, oy, ox)
            crop_size: Crop dimensions (cz, cy, cx)

        Returns:
            Cropped tensor
        """
        oz, oy, ox = offsets
        cz, cy, cx = crop_size
        return tensor[:, oz:oz + cz, oy:oy + cy, ox:ox + cx, :]

    def _adjust_targets_for_crop(
        self,
        targets: List[Dict[str, Any]],
        offsets: Tuple[int, int, int],
        crop_size: Tuple[int, int, int],
    ) -> List[Dict[str, Any]]:
        """
        Adjust boxes, masks, and label_map for crop operation.

        When boxes are filtered (some fall outside crop region), all per-instance
        fields (mask_ids, labels, masks) are filtered with the same valid mask
        to maintain consistency.

        Args:
            targets: List of target dicts
            offsets: Crop offsets (oz, oy, ox)
            crop_size: Crop size (cz, cy, cx)

        Returns:
            List of adjusted target dicts
        """
        oz, oy, ox = offsets
        cz, cy, cx = crop_size

        adjusted = []
        for tgt in targets:
            t = dict(tgt)

            # Compute valid mask from boxes FIRST (which instances survive crop)
            valid_mask = None
            if "boxes" in t and t["boxes"] is not None:
                t["boxes"], valid_mask = self._adjust_boxes_for_crop(
                    t["boxes"], offsets, crop_size
                )

            # Apply valid_mask to all per-instance fields to maintain consistency
            if valid_mask is not None:
                # Filter mask_ids
                if "mask_ids" in t and t["mask_ids"] is not None:
                    t["mask_ids"] = t["mask_ids"][valid_mask]

                # Filter labels
                if "labels" in t and t["labels"] is not None:
                    t["labels"] = t["labels"][valid_mask]

                # Filter and crop masks: first filter N dimension, then crop spatial
                if "masks" in t and t["masks"] is not None:
                    # (N, Z, Y, X) -> (M, Z, Y, X) where M = valid instances
                    t["masks"] = t["masks"][valid_mask]
                    # Then crop spatially: (M, Z, Y, X) -> (M, cz, cy, cx)
                    t["masks"] = t["masks"][:, oz:oz + cz, oy:oy + cy, ox:ox + cx]
            else:
                # No box filtering, just crop masks spatially
                if "masks" in t and t["masks"] is not None:
                    # (N, Z, Y, X) -> (N, cz, cy, cx)
                    t["masks"] = t["masks"][:, oz:oz + cz, oy:oy + cy, ox:ox + cx]

            # label_map is spatial-only (not per-instance), just crop it
            if "label_map" in t and t["label_map"] is not None:
                # (Z, Y, X) -> (cz, cy, cx)
                t["label_map"] = t["label_map"][oz:oz + cz, oy:oy + cy, ox:ox + cx]

            adjusted.append(t)

        return adjusted

    def _adjust_boxes_for_crop(
        self,
        boxes: torch.Tensor,
        offsets: Tuple[int, int, int],
        crop_size: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Shift boxes by offset, clip to crop region, filter invalid boxes.

        Args:
            boxes: Tensor of shape (N, 6) with bounding boxes
            offsets: Crop offsets (oz, oy, ox)
            crop_size: Crop size (cz, cy, cx)

        Returns:
            Tuple of:
              - Adjusted boxes tensor (may have fewer boxes if some were filtered)
              - Valid mask (N,) boolean tensor indicating which boxes survived,
                or None if no filtering was done
        """
        oz, oy, ox = offsets
        cz, cy, cx = crop_size
        out = boxes.clone().float()

        if self.bbox_format is None:
            raise ValueError("bbox_format must be set to adjust boxes")

        fmt = self.bbox_format.lower()

        if fmt == "zyxzyx":
            # [z1, y1, x1, z2, y2, x2]
            # Subtract offsets
            out[:, 0] -= oz  # z1
            out[:, 1] -= oy  # y1
            out[:, 2] -= ox  # x1
            out[:, 3] -= oz  # z2
            out[:, 4] -= oy  # y2
            out[:, 5] -= ox  # x2

            # Clip to crop bounds
            out[:, 0].clamp_(0, cz)
            out[:, 1].clamp_(0, cy)
            out[:, 2].clamp_(0, cx)
            out[:, 3].clamp_(0, cz)
            out[:, 4].clamp_(0, cy)
            out[:, 5].clamp_(0, cx)

            # Filter boxes with zero or negative volume
            valid = (out[:, 3] > out[:, 0]) & (out[:, 4] > out[:, 1]) & (out[:, 5] > out[:, 2])
            out = out[valid]

        else:
            raise ValueError(f"Unsupported bbox_format={self.bbox_format!r}")

        return out.to(boxes.dtype), valid

    def _resize_targets(
        self,
        targets: List[Dict[str, Any]],
        scale_factors: Tuple[float, float, float],
        target_shape: Tuple[int, int, int],
    ) -> List[Dict[str, Any]]:
        """
        Resize targets (masks, label_map, boxes) after crop.

        Args:
            targets: List of target dicts
            scale_factors: Scale factors (sz, sy, sx)
            target_shape: Target spatial shape for masks/label_map

        Returns:
            List of resized target dicts
        """
        resized = []
        for tgt in targets:
            t = dict(tgt)

            if "boxes" in t and t["boxes"] is not None and self.bbox_format is not None:
                t["boxes"] = resize_boxes(t["boxes"], scale_factors, self.bbox_format)

            if "masks" in t and t["masks"] is not None:
                t["masks"] = resize_masks(t["masks"], target_shape)

            if "label_map" in t and t["label_map"] is not None:
                t["label_map"] = resize_label_map(t["label_map"], target_shape)

            resized.append(t)

        return resized

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply crop transform to data sample.

        Args:
            data: Dict with "data_tensor" and optionally "metainfo"

        Returns:
            Dict with cropped/resized "data_tensor" and updated "metainfo"
        """
        if "data_tensor" not in data:
            raise KeyError("Crop3D expects 'data_tensor' in input dict")

        inputs = data["data_tensor"].to(self.dtype)
        metainfo = data.get("metainfo", {})
        targets = metainfo.get("targets", [])

        if inputs.ndim != 5:
            raise ValueError(f"Expected 5D tensor (B, Z, Y, X, C), got shape {inputs.shape}")

        B, Z, Y, X, C = inputs.shape
        current_shape = (Z, Y, X)

        # Sample target shape for this call (uniform from range, or fixed)
        target_shape = self._sample_target_shape()

        # Sample mode
        mode = self._sample_mode()

        # Compute crop region
        offsets, crop_size = self._compute_crop_region(current_shape, target_shape)

        # Crop the tensor and targets
        inputs = self._crop_tensor(inputs, offsets, crop_size)
        targets = self._adjust_targets_for_crop(targets, offsets, crop_size)

        # Get actual shape after crop
        actual_shape = tuple(inputs.shape[1:4])  # (Z, Y, X)

        # Check if we need to resize to reach target shape
        need_resize = actual_shape != target_shape

        if need_resize:
            # Even if mode was "crop", we need to resize because:
            # - Input was too small for pure crop, OR
            # - Mode is "crop_resize"
            # This handles the edge case where input < target automatically
            inputs, scale_factors = resize_tensor_3d(
                inputs,
                target_shape,
                input_format=self.input_format,
                mode=self.resize_mode,
                align_corners=self.align_corners,
                dtype=self.dtype,
            )
            targets = self._resize_targets(targets, scale_factors, target_shape)

        # Update metainfo
        metainfo = dict(metainfo)
        metainfo["targets"] = targets

        final_shape = tuple(inputs.shape[1:4])
        metainfo["image_sizes"] = torch.tensor(
            [list(final_shape)] * B,
            device=inputs.device,
            dtype=torch.long,
        )

        return {"data_tensor": inputs, "metainfo": metainfo}