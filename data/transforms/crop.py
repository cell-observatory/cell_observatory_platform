import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from cell_observatory_platform.data.data_types import TORCH_DTYPES, DataKind, kind_family
from cell_observatory_platform.data.transforms.utils import (
    parse_target_shape_range,
    resize_boxes,
    resize_label_map,
    resize_masks,
    resize_tensor_3d,
    sample_target_shape,
)


def _require_dict_targets(targets) -> None:
    """Crop/Resize warp targets field-by-field over Form S (per-sample List[Dict],
    see data/data_types.py) -- the only form they support.

    Form-D targets (role-keyed dict of batched tensors: DeepCopyInputsAsTargets
    clones, semantic maps) cannot be warped here coherently -- place geometric
    transforms BEFORE the Form-D producer so input and target stay aligned.
    Fail loudly instead of crashing on ``dict(tensor)`` below.
    """
    if isinstance(targets, dict):
        raise TypeError(
            "Crop/Resize expect Form-S targets (per-sample List[Dict] of "
            "boxes/masks/labels; see data/data_types.py). Got a Form-D role dict "
            f"(roles {list(targets)}) — place geometric transforms BEFORE the "
            "Form-D producer (DeepCopyInputsAsTargets / the semantic channel "
            "split) so input and target stay aligned."
        )
    for tgt in targets:
        if not isinstance(tgt, dict):
            raise TypeError(
                "Crop/Resize expect Form-S targets (per-sample List[Dict] of "
                f"boxes/masks/labels; see data/data_types.py). Got {type(tgt)!r}."
            )


class Crop:
    """
    3D/4D cropping transform with probabilistic mode selection.

    Modes:
      - "crop": Crop to target size (no resize). If input is smaller than target,
                automatically falls back to crop_resize behavior.
      - "crop_resize": Crop to intermediate size, then resize to target.

    Selection between modes is probabilistic via mode_probs dict.

    Supports:
      - input_format="ZYXC": tensor shape (B, Z, Y, X, C) — full annotations
      - input_format="TZYXC": tensor shape (B, T, Z, Y, X, C) — same target handling,
        T is a leading axis every handler passes through.

    What gets cropped/resized is metadata-driven (mirroring ``Resize``): the owning
    preprocessor declares ``metainfo["data_types"]`` (``{name -> {"kind", ...}}``)
    and Crop dispatches each declared target field on its ``kind`` via
    ``DataKind``/``kind_family`` -- instance/semantic labelmaps are sliced
    spatially, boxes are shifted/clipped/filtered. All spatial GT rides
    ``metainfo["targets"]``.

    Can be called on a data_sample dict with keys:
      - "data_tensor": image tensor
      - "metainfo": dict containing "targets" (list of dicts with masks/boxes/label_map)
        and "data_types" (the preprocessor-declared field spec).
    """

    def __init__(
        self,
        target_spatial_shape: Union[Sequence[int], Tuple[Sequence[int], Sequence[int]]],
        crop_dims: str = "YX",
        crop_type: str = "random",
        mode_probs: Optional[Dict[str, float]] = None,
        bbox_format: Optional[str] = None,
        dtype: str = "bfloat16",
        patch_size: Optional[Tuple[int, int, int]] = None,
        resize_mode: str = "trilinear",
        align_corners: bool = False,
        boxes_normalized: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
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
            bbox_format: Format of bounding boxes - "zyxzyx" or "cxcyczwhd".
            dtype: Retained for config compatibility; the dense flow now
                preserves the incoming dtype (the preprocessor's float32 count
                intermediate) and _finalize owns the narrowing.
            patch_size: If set, pad output to multiple of this size.
            resize_mode: Interpolation mode for resize operations (trilinear,
                area, nearest-exact; plain "nearest" is rejected).
            align_corners: Whether to align corners in resize interpolation.
            boxes_normalized: Set True when ``targets[*]["boxes"]`` are normalized
                to ``[0, 1]`` against the pre-crop spatial shape (as the collator
                does with ``normalize_bboxes=True``). Coordinates are denormalized
                against the pre-crop shape, shifted/clipped in voxel space, and
                renormalized against the crop size; the subsequent resize leaves
                normalized coords invariant. When False (default), boxes are
                treated as absolute voxel coords.
        """
        # Own RNG (mode, target shape, window offsets): seeded per transform so
        # crops are reproducible across ranks and resumes.
        self._rng = random.Random(seed)

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
        self.boxes_normalized = bool(boxes_normalized)

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
        return sample_target_shape(self.target_min, self.target_max, rng=self._rng)

    def _sample_mode(self) -> str:
        """Sample mode according to mode_probs distribution."""
        r = self._rng.random()
        cumsum = 0.0
        for mode, prob in self.mode_probs.items():
            cumsum += prob
            if r < cumsum:
                return mode
        return list(self.mode_probs.keys())[-1]

    def _target_field_specs(self, metainfo: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Resolve which target fields to warp and how, as ``(name, kind)``.

        Driven by ``metainfo["data_types"]`` (the preprocessor-declared dict
        ``{name -> {"kind", "layout", "role", ...}}``, ported from ``Resize``);
        the ``data_tensor`` entry is excluded (it is cropped on its own dedicated
        path). The concrete kind string is carried through and resolved to a
        handler via ``kind_family``.
        """
        spec = metainfo.get("data_types") or {}
        return [
            (name, entry["kind"])
            for name, entry in spec.items()
            if name != "data_tensor"
        ]

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
            off_z = self._rng.randint(0, max(0, Z - crop_Z)) if crop_z else 0
            off_y = self._rng.randint(0, max(0, Y - crop_Y)) if crop_y else 0
            off_x = self._rng.randint(0, max(0, X - crop_X)) if crop_x else 0

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
        field_specs: List[Tuple[str, str]],
        offsets: Tuple[int, int, int],
        crop_size: Tuple[int, int, int],
        full_spatial: Tuple[int, int, int],
    ) -> List[Dict[str, Any]]:
        """
        Adjust the declared target fields for the crop operation.

        Dispatch is data_types-driven (``field_specs``): instance/semantic
        labelmaps are sliced along their trailing 3 spatial axes; boxes are
        shifted/clipped/filtered. When boxes are filtered (some fall outside the
        crop region), the aligned per-instance fields (mask_ids, labels, masks)
        are filtered with the same valid mask to maintain consistency.

        Args:
            targets: List of target dicts
            field_specs: ``(name, kind)`` pairs from ``data_types``
            offsets: Crop offsets (oz, oy, ox)
            crop_size: Crop size (cz, cy, cx)
            full_spatial: pre-crop spatial (Z, Y, X); denormalization base for
                ``boxes_normalized``

        Returns:
            List of adjusted target dicts
        """
        oz, oy, ox = offsets
        cz, cy, cx = crop_size

        adjusted = []
        for tgt in targets:
            t = dict(tgt)

            valid_mask = None
            for name, kind in field_specs:
                value = t.get(name)
                if value is None:
                    continue
                fam = kind_family(kind)

                if fam in (DataKind.INSTANCE_MASKS, DataKind.SEMANTIC_MASKS):
                    # (Z, Y, X), (N, Z, Y, X) or (T, Z, Y, X): slice the trailing
                    # 3 spatial axes; leading axes pass through untouched.
                    t[name] = value[..., oz:oz + cz, oy:oy + cy, ox:ox + cx]

                elif fam is DataKind.BOXES:
                    t[name], valid_mask = self._adjust_boxes_for_crop(
                        value, offsets, crop_size, full_spatial
                    )

                else:
                    raise ValueError(
                        f"Crop has no handler for kind {kind!r} (field {name!r})"
                    )

            # Apply valid_mask to aligned per-instance fields for consistency.
            # Binary "masks" are never present at transform time (every task
            # preprocessor materializes them AFTER transforms from label_map).
            if valid_mask is not None:
                if "mask_ids" in t and t["mask_ids"] is not None:
                    t["mask_ids"] = t["mask_ids"][valid_mask]
                if "labels" in t and t["labels"] is not None:
                    t["labels"] = t["labels"][valid_mask]

            adjusted.append(t)

        return adjusted

    def _adjust_boxes_for_crop(
        self,
        boxes: torch.Tensor,
        offsets: Tuple[int, int, int],
        crop_size: Tuple[int, int, int],
        full_spatial: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Shift boxes by offset, clip to crop region, filter invalid boxes.

        Supports "zyxzyx" corners and "cxcyczwhd" center/size (converted to
        corners for the shift/clip, converted back after). When
        ``boxes_normalized`` is set, coords are denormalized against
        ``full_spatial`` first and renormalized against ``crop_size`` after.

        Args:
            boxes: Tensor of shape (N, 6) with bounding boxes
            offsets: Crop offsets (oz, oy, ox)
            crop_size: Crop size (cz, cy, cx)
            full_spatial: pre-crop spatial (Z, Y, X); normalization base

        Returns:
            Tuple of:
              - Adjusted boxes tensor (may have fewer boxes if some were filtered)
              - Valid mask (N,) boolean tensor indicating which boxes survived,
                or None if there were no boxes to filter
        """
        oz, oy, ox = offsets
        cz, cy, cx = crop_size
        fz, fy, fx = full_spatial

        if self.bbox_format is None:
            raise ValueError("bbox_format must be set to adjust boxes")
        fmt = self.bbox_format.lower()
        if fmt not in ("zyxzyx", "cxcyczwhd"):
            raise ValueError(f"Unsupported bbox_format={self.bbox_format!r}")

        out = boxes.clone().float()
        if out.numel() == 0:
            return out.to(boxes.dtype), None

        if fmt == "cxcyczwhd":
            # (cx, cy, cz, w, h, d) -> corner form (z1, y1, x1, z2, y2, x2)
            cx_, cy_, cz_, w_, h_, d_ = out.unbind(-1)
            out = torch.stack(
                [
                    cz_ - d_ / 2, cy_ - h_ / 2, cx_ - w_ / 2,
                    cz_ + d_ / 2, cy_ + h_ / 2, cx_ + w_ / 2,
                ],
                dim=-1,
            )

        if self.boxes_normalized:
            # Denormalize against the PRE-crop base so the voxel-space shift/clip
            # below is exact.
            scale = out.new_tensor([fz, fy, fx, fz, fy, fx])
            out = out * scale

        # [z1, y1, x1, z2, y2, x2]: subtract offsets
        shift = out.new_tensor([oz, oy, ox, oz, oy, ox])
        out = out - shift

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

        if self.boxes_normalized:
            # Renormalize against the crop, the new coordinate base.
            scale = out.new_tensor(
                [max(cz, 1), max(cy, 1), max(cx, 1)] * 2
            )
            out = out / scale

        if fmt == "cxcyczwhd":
            z1, y1, x1, z2, y2, x2 = out.unbind(-1)
            out = torch.stack(
                [
                    (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2,
                    x2 - x1, y2 - y1, z2 - z1,
                ],
                dim=-1,
            )

        return out.to(boxes.dtype), valid

    def _resize_targets(
        self,
        targets: List[Dict[str, Any]],
        field_specs: List[Tuple[str, str]],
        scale_factors: Tuple[float, float, float],
        target_shape: Tuple[int, int, int],
    ) -> List[Dict[str, Any]]:
        """
        Resize the declared target fields after crop (data_types-driven, same
        handlers as ``Resize``).

        Args:
            targets: List of target dicts
            field_specs: ``(name, kind)`` pairs from ``data_types``
            scale_factors: Scale factors (sz, sy, sx)
            target_shape: Target spatial shape for masks/label_map

        Returns:
            List of resized target dicts
        """
        resized = []
        for tgt in targets:
            t = dict(tgt)

            for name, kind in field_specs:
                value = t.get(name)
                if value is None:
                    continue
                fam = kind_family(kind)

                if fam is DataKind.INSTANCE_MASKS:
                    t[name] = resize_label_map(value, target_shape)
                elif fam is DataKind.SEMANTIC_MASKS:
                    # single integer class labelmap (semantic is one squashed
                    # channel now), same handling as instance -- see resize.py.
                    t[name] = resize_label_map(value, target_shape)
                elif fam is DataKind.BOXES:
                    if self.bbox_format is None:
                        raise ValueError("bbox_format must be set to resize boxes")
                    if self.boxes_normalized:
                        # Pure resize leaves normalized coords invariant.
                        continue
                    t[name] = resize_boxes(value, scale_factors, self.bbox_format)
                else:
                    raise ValueError(
                        f"Crop has no resize handler for kind {kind!r} (field {name!r})"
                    )

            resized.append(t)

        return resized

    def _update_image_sizes(
        self,
        metainfo: Dict[str, Any],
        batch_size: int,
        offsets: Tuple[int, int, int],
        crop_size: Tuple[int, int, int],
        final_shape: Tuple[int, int, int],
        device: torch.device,
    ) -> None:
        """Update ``metainfo["image_sizes"]`` in place for the crop (+resize).

        The valid (content) region is origin-anchored, so after taking the window
        ``[off : off + crop]`` the remaining valid extent per axis is
        ``min(old_valid - off, crop)`` (clamped at 0) -- NOT the full crop size:
        a crop window can retain trailing buffer padding, and overwriting with
        the window size would relabel padding as content. When a resize follows,
        the valid extent scales with it (``final / crop`` per axis).
        """
        img = metainfo.get("image_sizes")
        if img is not None and torch.is_tensor(img):
            updated = img.clone()
            spatial = updated[:, -3:].to(torch.float64)
            offs = torch.tensor(offsets, dtype=torch.float64, device=img.device)
            crop_t = torch.tensor(crop_size, dtype=torch.float64, device=img.device)
            final_t = torch.tensor(final_shape, dtype=torch.float64, device=img.device)
            new_valid = torch.minimum(spatial - offs, crop_t).clamp(min=0)
            # Scale by the resize factor (identity when final == crop).
            new_valid = (new_valid * (final_t / crop_t)).round().clamp(min=0)
            new_valid = torch.minimum(new_valid, final_t)
            updated[:, -3:] = new_valid.to(updated.dtype)
            metainfo["image_sizes"] = updated
        else:
            # No prior sizes: assume all-valid content at the final shape.
            metainfo["image_sizes"] = torch.tensor(
                [list(final_shape)] * batch_size,
                device=device,
                dtype=torch.long,
            )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """dict in -> inspect data_types layout -> dispatch 3D/4D -> dict out."""
        if not isinstance(data, dict):
            raise TypeError(f"Crop expects a dict sample, got {type(data)}")
        if "data_tensor" not in data:
            raise KeyError("Crop expects 'data_tensor' in input dict")

        has_time = data["metainfo"]["data_types"]["data_tensor"]["has_time"]
        return self._run_4d(data) if has_time else self._run_3d(data)

    def _run_3d(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """ZYXC crop path — full dense + annotation handling."""
        if "data_tensor" not in data:
            raise KeyError("Crop expects 'data_tensor' in input dict")

        # Preserve the incoming dtype: the preprocessor keeps an exact float32
        # count intermediate through the transform chain and narrows to the
        # config dtype only in _finalize. Casting here would quantize counts
        # before the noise model sees them.
        inputs = data["data_tensor"]
        metainfo = data.get("metainfo", {})
        targets = metainfo.get("targets", [])
        _require_dict_targets(targets)
        field_specs = self._target_field_specs(metainfo)

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
        targets = self._adjust_targets_for_crop(
            targets, field_specs, offsets, crop_size, current_shape
        )

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
                input_format="ZYXC",
                mode=self.resize_mode,
                align_corners=self.align_corners,
                dtype=None,  # keep input dtype; _finalize owns the narrowing
            )
            targets = self._resize_targets(targets, field_specs, scale_factors, target_shape)

        # Update metainfo
        metainfo = dict(metainfo)
        metainfo["targets"] = targets

        final_shape = tuple(inputs.shape[1:4])
        self._update_image_sizes(
            metainfo, B, offsets, crop_size, final_shape, inputs.device
        )

        return {"data_tensor": inputs, "metainfo": metainfo}

    def _run_4d(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        TZYXC crop path: T is preserved on the image, label maps and the
        padding mask; boxes are shifted/clipped/filtered exactly as in 3D
        (per-instance fields carry no time axis).
        """
        if "data_tensor" not in data:
            raise KeyError("Crop expects 'data_tensor' in input dict")

        # Preserve incoming dtype -- see _run_3d; _finalize owns the narrowing.
        inputs = data["data_tensor"]
        metainfo = data.get("metainfo", {})
        targets = metainfo.get("targets", [])
        _require_dict_targets(targets)
        field_specs = self._target_field_specs(metainfo)

        if inputs.ndim != 6:
            raise ValueError(
                f"_run_4d expects 6D tensor (B, T, Z, Y, X, C), got shape {inputs.shape}"
            )

        B, T, Z, Y, X, C = inputs.shape
        current_shape = (Z, Y, X)

        # Sample target shape and crop region
        target_shape = self._sample_target_shape()
        offsets, crop_size = self._compute_crop_region(current_shape, target_shape)
        oz, oy, ox = offsets
        cz, cy, cx = crop_size

        # Crop the spatial axes; T is preserved: (B, T, Z, Y, X, C) -> (B, T, cz, cy, cx, C)
        inputs = inputs[:, :, oz:oz + cz, oy:oy + cy, ox:ox + cx, :]

        # Label maps ((T, Z, Y, X) per target) are sliced on their trailing 3
        # axes; boxes are shifted/clipped/filtered and mask_ids/labels follow.
        new_targets = self._adjust_targets_for_crop(
            targets, field_specs, offsets, crop_size, current_shape
        )

        padding_mask = metainfo.get("padding_mask")
        has_pm = torch.is_tensor(padding_mask) and padding_mask.ndim == 5
        if has_pm:
            padding_mask = padding_mask[:, :, oz:oz + cz, oy:oy + cy, ox:ox + cx]

        # Handle resize if needed (actual crop smaller than target)
        actual_shape = (int(inputs.shape[2]), int(inputs.shape[3]), int(inputs.shape[4]))
        if actual_shape != target_shape:
            # Fold T into batch for resize_tensor_3d, then unfold
            folded = inputs.reshape(B * T, cz, cy, cx, C)
            resized_folded, scale_factors = resize_tensor_3d(
                folded,
                target_shape,
                input_format="ZYXC",
                mode=self.resize_mode,
                align_corners=self.align_corners,
                dtype=None,  # keep input dtype; _finalize owns the narrowing
            )
            inputs = resized_folded.reshape(B, T, *target_shape, C)
            new_targets2 = []
            for tgt in new_targets:
                t = dict(tgt)
                for name, kind in field_specs:
                    value = t.get(name)
                    if value is None:
                        continue
                    fam = kind_family(kind)
                    if fam in (DataKind.INSTANCE_MASKS, DataKind.SEMANTIC_MASKS):
                        # rank picks the kernel: a bare (Z, Y, X) map goes
                        # through resize_label_map, anything with a leading axis
                        # ((T, Z, Y, X)) through resize_masks (nearest-exact per
                        # leading slice; same kernel underneath).
                        t[name] = (
                            resize_label_map(value, target_shape)
                            if value.ndim == 3
                            else resize_masks(value, target_shape)
                        )
                    elif fam is DataKind.BOXES:
                        if self.bbox_format is None:
                            raise ValueError("bbox_format must be set to resize boxes")
                        if not self.boxes_normalized:
                            # normalized coords are invariant under a pure resize
                            t[name] = resize_boxes(value, scale_factors, self.bbox_format)
                    else:
                        raise ValueError(
                            f"Crop has no resize handler for kind {kind!r} (field {name!r})"
                        )
                new_targets2.append(t)
            new_targets = new_targets2
            if has_pm:
                padding_mask = resize_masks(padding_mask, target_shape)

        metainfo = dict(metainfo)
        metainfo["targets"] = new_targets
        if has_pm:
            metainfo["padding_mask"] = padding_mask

        final_shape = (int(inputs.shape[2]), int(inputs.shape[3]), int(inputs.shape[4]))
        self._update_image_sizes(
            metainfo, B, offsets, crop_size, final_shape, inputs.device
        )

        return {"data_tensor": inputs, "metainfo": metainfo}
        