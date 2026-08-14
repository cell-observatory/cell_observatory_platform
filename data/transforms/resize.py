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


class Resize:
    """
    Spatial resize transform.

    Layout (3D vs 4D) is read per-call from the sample's declared
    ``metainfo["data_types"]["data_tensor"]["layout"]``:
      - ``ZYXC``  (B, Z, Y, X, C): resized directly.
      - ``TZYXC`` (B, T, Z, Y, X, C): the time axis is folded into the batch so
        each frame is resized independently.
    Target size is fixed or sampled uniformly from a range.

    What gets resized is metadata-driven: the owning preprocessor declares
    ``metainfo["data_types"]`` (``{name -> {"kind", "layout", "role", ...}}``) and
    Resize dispatches each declared target field on its ``kind`` via
    ``DataKind``/``kind_family`` -- instance/semantic labelmaps use nearest, boxes
    are coordinate scaled. All spatial GT rides ``metainfo["targets"]``.

    ``orig_image_sizes`` is left untouched and remains the authoritative restore
    target that downstream inference post-processing maps predictions back to.
    """

    def __init__(
        self,
        target_spatial_shape: Union[Sequence[int], Tuple[Sequence[int], Sequence[int]]],
        mode: str = "trilinear",
        align_corners: bool = False,
        dtype: str = "bfloat16",
        bbox_format: Optional[str] = None,
        crop_to_valid: bool = True,
        boxes_normalized: bool = False,
    ) -> None:
        """
        Args:
            target_spatial_shape: Target output spatial shape. Either:
                - Fixed: (Z, Y, X) or [Z, Y, X]
                - Range: ((Z_min, Y_min, X_min), (Z_max, Y_max, X_max))
                  Samples uniformly from range on each call.
            mode: Interpolation mode for F.interpolate (trilinear, nearest, area).
            align_corners: Whether to align corners in interpolation.
            dtype: Retained for config compatibility; the dense flow now
                preserves the incoming dtype (the preprocessor's float32 count
                intermediate) and _finalize owns the narrowing.
            bbox_format: Format of bounding boxes - "zyxzyx" or "cxcyczwhd".
            crop_to_valid: When True (default), each sample is first cropped to
                its valid region (``metainfo["image_sizes"]``) BEFORE resizing,
                so trailing buffer padding is never squeezed into the content.
                Content is origin-anchored (valid = ``[:z, :y, :x]``); padding
                trails each axis (see ``get_image_sizes``). A no-op when a sample
                already fills the buffer. Requires ``metainfo["image_sizes"]``;
                falls back to no crop when absent.
            boxes_normalized: Set True when ``targets[*]["boxes"]`` are normalized
                to ``[0, 1]`` against the (padded) buffer spatial shape -- as the
                collator does with ``normalize_bboxes=True``. Pure resize leaves
                normalized coords invariant, but ``crop_to_valid`` changes the
                normalization base from the buffer to the valid region, so boxes
                are renormalized by ``buffer / valid`` per axis. When False
                (default), boxes are treated as absolute voxel coords and scaled
                by ``target / valid``.
        """
        # Parse target shape (fixed or range)
        self.target_min, self.target_max, self.random_target = parse_target_shape_range(
            target_spatial_shape
        )

        self.mode = mode
        self.align_corners = align_corners
        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        self.bbox_format = bbox_format
        self.crop_to_valid = bool(crop_to_valid)
        self.boxes_normalized = bool(boxes_normalized)

    def _sample_target_shape(self) -> Tuple[int, int, int]:
        """Sample target shape (uniform from range, or fixed if no range)."""
        return sample_target_shape(self.target_min, self.target_max)

    def _resize_data_tensor(
        self, tensor: torch.Tensor, target_shape: Tuple[int, int, int], has_time: bool
    ) -> Tuple[torch.Tensor, Tuple[float, float, float]]:
        """Resize the image tensor, folding a leading time axis when present.

        ZYXC: (B, Z, Y, X, C) -> resized directly.
        TZYXC: (B, T, Z, Y, X, C) -> fold to (B*T, Z, Y, X, C), resize, unfold.
        """
        if not has_time:
            return resize_tensor_3d(
                tensor,
                target_shape,
                input_format="ZYXC",
                mode=self.mode,
                align_corners=self.align_corners,
                # Keep the input dtype: the preprocessor holds an exact float32
                # count intermediate through the transform chain and narrows to
                # the config dtype only in _finalize. Casting here would
                # quantize counts mid-pipeline (e.g. before the noise model).
                dtype=None,
            )

        if tensor.ndim != 6:
            raise ValueError(
                f"Resize(input_format='TZYXC') expects a 6D (B, T, Z, Y, X, C) "
                f"tensor, got shape {tuple(tensor.shape)}"
            )
        B, T = tensor.shape[0], tensor.shape[1]
        C = tensor.shape[-1]
        folded = tensor.reshape(B * T, *tensor.shape[2:])
        resized, scale_factors = resize_tensor_3d(
            folded,
            target_shape,
            input_format="ZYXC",
            mode=self.mode,
            align_corners=self.align_corners,
            dtype=None,  # keep input dtype; _finalize owns the narrowing
        )
        resized = resized.reshape(B, T, *target_shape, C)
        return resized, scale_factors

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """dict in -> inspect data_types layout -> dispatch 3D/4D -> dict out."""
        if not isinstance(data, dict):
            raise TypeError(f"Resize expects a dict sample, got {type(data)}")
        if "data_tensor" not in data:
            raise KeyError("Expected key 'data_tensor' in data_sample dict.")

        # 3D vs 4D is precomputed by the preprocessor in data_types; read it and
        # thread it down as a plain argument.
        has_time = data["metainfo"]["data_types"]["data_tensor"]["has_time"]

        target_shape = self._sample_target_shape()
        inputs = data["data_tensor"]
        metainfo = data.get("metainfo", {})

        B = inputs.shape[0]
        # Full buffer spatial (Z, Y, X) -- the last 3 dims before C, regardless of
        # ZYXC (B,Z,Y,X,C) or TZYXC (B,T,Z,Y,X,C).
        full_spatial = tuple(int(s) for s in inputs.shape[-4:-1])

        # Per-sample valid spatial sizes (content region) from metainfo; padding
        # is the trailing remainder of each axis. None => crop disabled / unknown.
        valid_sizes = self._valid_spatial_sizes(metainfo, B)
        needs_crop = (
            self.crop_to_valid
            and valid_sizes is not None
            and any(tuple(v) != full_spatial for v in valid_sizes)
        )

        if needs_crop:
            # Per-sample: crop each sample to its valid region, then resize. This
            # removes padding before interpolation (no distortion from padded
            # voxels). Scale factors are per-sample (target / valid).
            resized_inputs, per_sample_scales, source_sizes = (
                self._crop_resize_per_sample(inputs, valid_sizes, target_shape, has_time)
            )
        else:
            # Fast path: no padding to remove -> single batched resize.
            resized_inputs, scale_factors = self._resize_data_tensor(inputs, target_shape, has_time)
            # source_sizes must follow the SAME flag that gates the image crop:
            # with crop_to_valid disabled the image was resized from the FULL
            # buffer, so targets must be treated the same way -- deriving them
            # from valid_sizes here would crop GT (and rescale boxes) against a
            # region the image path never used.
            source_sizes = (
                [tuple(v) for v in valid_sizes]
                if (self.crop_to_valid and valid_sizes is not None)
                else [full_spatial] * B
            )
            per_sample_scales = [scale_factors] * B

        resized_metainfo = self._resize_metainfo(
            metainfo, per_sample_scales, target_shape, source_sizes, inputs, has_time,
            cropped_to_valid=needs_crop,
        )

        # All spatial GT rides in metainfo["targets"] (warped in _resize_metainfo);
        # preserve any other top-level keys untouched.
        out: Dict[str, Any] = {
            k: v for k, v in data.items() if k not in ("data_tensor", "metainfo")
        }
        out["data_tensor"] = resized_inputs
        out["metainfo"] = resized_metainfo
        return out

    def _valid_spatial_sizes(
        self, metainfo: Dict[str, Any], batch_size: int
    ) -> Optional[List[Tuple[int, int, int]]]:
        """Per-sample valid spatial sizes ``(Z, Y, X)`` from ``image_sizes``.

        ``image_sizes`` is ``(B, 3)`` (ZYXC) or ``(B, 4)`` (TZYXC: T, Z, Y, X);
        the spatial region is always the trailing 3 entries. Returns None when
        unavailable (crop is then skipped).
        """
        img = metainfo.get("image_sizes") if isinstance(metainfo, dict) else None
        if img is None or not torch.is_tensor(img):
            return None
        spatial = img[:, -3:]
        return [
            tuple(int(v) for v in spatial[b].tolist())
            for b in range(min(batch_size, spatial.shape[0]))
        ]

    def _crop_resize_per_sample(
        self,
        inputs: torch.Tensor,
        valid_sizes: List[Tuple[int, int, int]],
        target_shape: Tuple[int, int, int],
        has_time: bool,
    ) -> Tuple[torch.Tensor, List[Tuple[float, float, float]], List[Tuple[int, int, int]]]:
        """Crop each sample to its valid region, resize to ``target_shape``, stack.

        The time axis (TZYXC) is preserved as-is (only spatial padding is removed);
        single-frame inputs are the supported temporal case.
        """
        outs: List[torch.Tensor] = []
        scales: List[Tuple[float, float, float]] = []
        sources: List[Tuple[int, int, int]] = []
        for b, valid in enumerate(valid_sizes):
            vz, vy, vx = (int(s) for s in valid)
            if has_time:
                crop = inputs[b : b + 1, :, :vz, :vy, :vx, :]  # (1, T, vz, vy, vx, C)
            else:
                crop = inputs[b : b + 1, :vz, :vy, :vx, :]      # (1, vz, vy, vx, C)
            resized, scale = self._resize_data_tensor(crop, target_shape, has_time)
            outs.append(resized)
            scales.append(scale)
            sources.append((vz, vy, vx))
        return torch.cat(outs, dim=0), scales, sources

    def _resize_metainfo(
        self,
        metainfo: Dict[str, Any],
        per_sample_scales: List[Tuple[float, float, float]],
        target_shape: Tuple[int, int, int],
        source_sizes: List[Tuple[int, int, int]],
        inputs: torch.Tensor,
        has_time: bool,
        cropped_to_valid: bool = False,
    ) -> Dict[str, Any]:
        """
        Resize metainfo fields using per-sample scale factors.

        Note: ``orig_image_sizes`` is intentionally NOT modified here -- it is the
        authoritative original tile size (from the DB) that inference restores
        predictions back to. ``image_sizes`` is updated to the new (resized) size,
        and ``padding_mask`` is reset to all-valid ONLY when crop-to-valid
        actually removed the padding before resizing (``cropped_to_valid``);
        otherwise the mask is resized alongside the data so padding provenance
        survives a full-buffer resize.

        Args:
            metainfo: Dict containing targets, padding_mask, image_sizes, etc.
            per_sample_scales: list of (sz, sy, sx) scale factors, one per sample
            target_shape: Target spatial shape for this call
            source_sizes: per-sample pre-resize valid spatial shapes (Z, Y, X)
            inputs: original (pre-resize) data tensor, for batch/time geometry

        Returns:
            Updated metainfo dict
        """
        out = dict(metainfo)

        if "targets" in out:
            # Pre-crop buffer spatial (Z, Y, X) -- the normalization base for
            # buffer-normalized boxes when crop_to_valid renormalizes them.
            full_spatial = tuple(int(s) for s in inputs.shape[-4:-1])
            field_specs = self._target_field_specs(metainfo)
            out["targets"] = self._resize_targets(
                out["targets"], field_specs, per_sample_scales, target_shape,
                source_sizes, full_spatial,
            )

        if "image_sizes" in out:
            if not torch.is_tensor(out["image_sizes"]):  # survives python -O
                raise TypeError(
                    f"Expected image_sizes to be a tensor, got {type(out['image_sizes'])}"
                )
            img_sizes = out["image_sizes"]
            B, n_axes = img_sizes.shape[0], img_sizes.shape[1]
            # image_sizes is (B, 3) for ZYXC or (B, 4) for TZYXC (T, Z, Y, X).
            # Replace only the trailing 3 spatial entries; keep any leading time.
            new_spatial = torch.tensor(
                target_shape,
                device=img_sizes.device,
                dtype=img_sizes.dtype,
            )
            if n_axes == 3:
                out["image_sizes"] = new_spatial.unsqueeze(0).expand(B, -1).clone()
            else:
                updated = img_sizes.clone()
                updated[:, -3:] = new_spatial.unsqueeze(0)
                out["image_sizes"] = updated

        if "padding_mask" in out:
            pm = out["padding_mask"]
            if not (torch.is_tensor(pm) and pm.ndim in (4, 5)):  # survives python -O
                raise TypeError(
                    f"Expected padding_mask of shape (B, Z, Y, X) or (B, T, Z, Y, X), got {pm.shape if torch.is_tensor(pm) else type(pm)}"
                )
            if cropped_to_valid:
                # Crop-to-valid removed the padding BEFORE the resize, so the
                # content genuinely fills target_shape: all-valid (False).
                if has_time:
                    num_frames = int(inputs.shape[1])
                    new_shape = (pm.shape[0], num_frames, *target_shape)
                else:
                    new_shape = (pm.shape[0], *target_shape)
                out["padding_mask"] = torch.zeros(new_shape, dtype=pm.dtype, device=pm.device)
            else:
                # No crop ran (crop_to_valid=False or nothing to crop): the FULL
                # padded buffer was resized, squeezing any padding INTO the
                # output. Blanking the mask here would silently launder padding
                # into "valid content" -- resize the mask instead (nearest-exact,
                # dtype-preserving) so provenance survives.
                out["padding_mask"] = resize_masks(pm, target_shape)

        return out

    def _target_field_specs(self, metainfo: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Resolve which target fields to resize and how, as ``(name, kind)``.

        Driven by ``metainfo["data_types"]`` (the preprocessor-declared dict
        ``{name -> {"kind", "layout", "role", ...}}``); the ``data_tensor`` entry
        is excluded (it is resized on its own dedicated path). The concrete kind
        string is carried through and resolved to a handler in
        ``_resize_single_target`` via ``kind_family``.
        """
        spec = metainfo["data_types"]
        return [
            (name, entry["kind"])
            for name, entry in spec.items()
            if name != "data_tensor"
        ]

    def _resize_targets(
        self,
        targets: List[Dict[str, Any]],
        field_specs: List[Tuple[str, str]],
        per_sample_scales: List[Tuple[float, float, float]],
        target_shape: Tuple[int, int, int],
        source_sizes: List[Tuple[int, int, int]],
        full_spatial: Tuple[int, int, int],
    ) -> List[Dict[str, Any]]:
        """
        Crop each target to its sample's valid region, then resize.

        Targets are aligned 1:1 with the batch (``targets[b]`` <-> sample ``b``),
        so each uses its own scale factors and valid crop size.
        """
        assert len(targets) == len(per_sample_scales) == len(source_sizes), (
            f"targets/batch misalignment: {len(targets)} targets, "
            f"{len(per_sample_scales)} scales, {len(source_sizes)} source sizes"
        )
        return [
            self._resize_single_target(
                tgt, field_specs, per_sample_scales[b], target_shape,
                source_sizes[b], full_spatial,
            )
            for b, tgt in enumerate(targets)
        ]

    def _resize_single_target(
        self,
        target: Dict[str, Any],
        field_specs: List[Tuple[str, str]],
        scale_factors: Tuple[float, float, float],
        target_shape: Tuple[int, int, int],
        source_size: Tuple[int, int, int],
        full_spatial: Tuple[int, int, int],
    ) -> Dict[str, Any]:
        """
        Crop a single target to ``source_size`` (valid region) then resize.

        Each ``(name, policy)`` in ``field_specs`` is applied to the matching
        target field when present. ``policy`` comes from the declared field
        ``kind`` (``dense``/``masks``/``label_map``/``boxes``) -- no type sniffing.

        Args:
            target: Target dict (e.g. masks, boxes, label_map, dense fields).
            field_specs: ``(name, policy)`` pairs for the fields to resize.
            scale_factors: (sz, sy, sx) scale factors (target / valid)
            target_shape: Target spatial shape for this call
            source_size: valid spatial region (Z, Y, X) to crop to before resize
            full_spatial: pre-crop buffer spatial (Z, Y, X); box renorm base

        Returns:
            Resized target dict
        """
        if not isinstance(target, dict):
            raise TypeError(
                "Crop/Resize expect Form-S targets (per-sample List[Dict] of "
                f"boxes/masks/labels; see data/data_types.py). Got {type(target)!r} — "
                "a Form-D role dict (DeepCopyInputsAsTargets clones, semantic maps) "
                "cannot be warped here; place geometric transforms BEFORE the "
                "Form-D producer so input and target stay aligned."
            )

        t = dict(target)
        vz, vy, vx = (int(s) for s in source_size)

        for name, kind in field_specs:
            value = t.get(name)
            # TODO: potential source of errors, but there may be situations like inferece
            # where we don't have all targets for a sample so not clear we can just assert here?
            if value is None:
                continue

            fam = kind_family(kind)

            if fam is DataKind.INSTANCE_MASKS:
                # single integer labelmap (Z, Y, X): crop spatial padding, then
                # nearest-resize (id-preserving).
                t[name] = resize_label_map(value[..., :vz, :vy, :vx], target_shape)

            elif fam is DataKind.SEMANTIC_MASKS:
                # stacked labelmaps (N, Z, Y, X): crop spatial padding, then
                # nearest-resize each slice (id-preserving).
                t[name] = resize_masks(value[..., :vz, :vy, :vx], target_shape)

            elif fam is DataKind.BOXES:
                # Crop is origin-anchored, so box coords never shift. The per-axis
                # scale depends on the coordinate convention:
                #   - normalized (to the padded buffer): renormalize buffer -> valid,
                #     i.e. multiply by full / valid; resize is invariant.
                #   - absolute voxel coords: multiply by target / valid (the resize).
                assert self.bbox_format is not None, \
                    "bbox_format must be set to resize boxes"
                if self.boxes_normalized:
                    fz, fy, fx = (int(s) for s in full_spatial)
                    box_scale = (fz / max(vz, 1), fy / max(vy, 1), fx / max(vx, 1))
                else:
                    box_scale = scale_factors
                t[name] = resize_boxes(value, box_scale, self.bbox_format)

            else:
                raise ValueError(
                    f"Resize has no handler for kind {kind!r} (field {name!r})"
                )

        return t
