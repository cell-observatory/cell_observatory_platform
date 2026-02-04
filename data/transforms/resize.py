from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_platform.data.transforms.utils import (
    parse_target_shape_range,
    resize_boxes,
    resize_label_map,
    resize_masks,
    resize_padding_mask,
    resize_tensor_3d,
    sample_target_shape,
)


# TODO: generalize to N-D
class Resize:
    """
    3D resize transform.

    Supports:
      - input_format="ZYXC": tensor shape (B, Z, Y, X, C)
      - Fixed or random target size (uniform sampling from range)

    Can be called on:
      - a plain tensor, or
      - a data_sample dict with keys:
          - "data_tensor": image tensor
          - "metainfo": dict containing "targets", "padding_mask", etc.
    """

    def __init__(
        self,
        input_format: str,
        target_spatial_shape: Union[Sequence[int], Tuple[Sequence[int], Sequence[int]]],
        mode: str = "trilinear",
        align_corners: bool = False,
        dtype: str = "bfloat16",
        bbox_format: Optional[str] = None,
    ) -> None:
        """
        Args:
            input_format: Data layout format, currently only "ZYXC" supported.
            target_spatial_shape: Target output spatial shape. Either:
                - Fixed: (Z, Y, X) or [Z, Y, X]
                - Range: ((Z_min, Y_min, X_min), (Z_max, Y_max, X_max))
                  Samples uniformly from range on each call.
            mode: Interpolation mode for F.interpolate (trilinear, nearest, area).
            align_corners: Whether to align corners in interpolation.
            dtype: Data type for output tensor.
            bbox_format: Format of bounding boxes - "zyxzyx" or "cxcyczwhd".
        """
        input_format = input_format.upper()
        if input_format != "ZYXC":
            raise ValueError(
                f"Resize only supports input_format='ZYXC', got {input_format!r}"
            )

        self.input_format = input_format

        # Parse target shape (fixed or range)
        self.target_min, self.target_max, self.random_target = parse_target_shape_range(
            target_spatial_shape
        )

        self.mode = mode
        self.align_corners = align_corners
        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        self.bbox_format = bbox_format

    def _sample_target_shape(self) -> Tuple[int, int, int]:
        """Sample target shape (uniform from range, or fixed if no range)."""
        return sample_target_shape(self.target_min, self.target_max)

    def __call__(
        self, data: Union[torch.Tensor, Dict[str, Any]]
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Apply resize transform.

        Args:
            data: Either a tensor (B, Z, Y, X, C) or a dict with "data_tensor" and "metainfo"

        Returns:
            Resized tensor or dict with resized tensor and updated metainfo
        """
        # Sample target shape for this call
        target_shape = self._sample_target_shape()

        if isinstance(data, torch.Tensor):
            resized, _ = resize_tensor_3d(
                data,
                target_shape,
                input_format=self.input_format,
                mode=self.mode,
                align_corners=self.align_corners,
                dtype=self.dtype,
            )
            return resized

        if not isinstance(data, dict):
            raise TypeError(
                f"Resize expects a torch.Tensor or dict with 'data_tensor'/'metainfo', "
                f"got {type(data)}"
            )

        if "data_tensor" not in data:
            raise KeyError("Expected key 'data_tensor' in data_sample dict.")

        inputs = data["data_tensor"]
        metainfo = data.get("metainfo", {})

        resized_inputs, scale_factors = resize_tensor_3d(
            inputs,
            target_shape,
            input_format=self.input_format,
            mode=self.mode,
            align_corners=self.align_corners,
            dtype=self.dtype,
        )
        resized_metainfo = self._resize_metainfo(metainfo, scale_factors, target_shape)

        return {
            "data_tensor": resized_inputs,
            "metainfo": resized_metainfo,
        }

    def _resize_metainfo(
        self,
        metainfo: Dict[str, Any],
        scale_factors: Tuple[float, float, float],
        target_shape: Tuple[int, int, int],
    ) -> Dict[str, Any]:
        """
        Resize metainfo fields using uniform scale factors.

        Args:
            metainfo: Dict containing targets, padding_mask, image_sizes, etc.
            scale_factors: (sz, sy, sx) scale factors
            target_shape: Target spatial shape for this call

        Returns:
            Updated metainfo dict
        """
        out = dict(metainfo)

        if "targets" in out:
            out["targets"] = self._resize_targets(
                out["targets"], scale_factors, target_shape
            )

        if "image_sizes" in out:
            assert torch.is_tensor(out["image_sizes"]), \
                f"Expected image_sizes to be a tensor, got {type(out['image_sizes'])}"
            img_sizes = out["image_sizes"]
            B = img_sizes.shape[0]
            new_sz = torch.tensor(
                target_shape,
                device=img_sizes.device,
                dtype=img_sizes.dtype,
            )
            out["image_sizes"] = new_sz.unsqueeze(0).expand(B, -1)

        if "padding_mask" in out:
            assert torch.is_tensor(out["padding_mask"]) and out["padding_mask"].ndim == 4, \
                f"Expected padding_mask to be a tensor of shape (B, Z, Y, X), got {out['padding_mask'].shape}"
            pm = out["padding_mask"]
            out["padding_mask"] = resize_padding_mask(pm, target_shape)

        return out

    def _resize_targets(
        self,
        targets: List[Dict[str, Any]],
        scale_factors: Tuple[float, float, float],
        target_shape: Tuple[int, int, int],
    ) -> List[Dict[str, Any]]:
        """
        Resize all targets with uniform scale factors.

        Args:
            targets: List of target dicts
            scale_factors: (sz, sy, sx) scale factors
            target_shape: Target spatial shape for this call

        Returns:
            List of resized target dicts
        """
        if not isinstance(targets, list):
            raise TypeError(f"Expected targets to be a list, got {type(targets)}")

        return [
            self._resize_single_target(tgt, scale_factors, target_shape)
            for tgt in targets
        ]

    def _resize_single_target(
        self,
        target: Dict[str, Any],
        scale_factors: Tuple[float, float, float],
        target_shape: Tuple[int, int, int],
    ) -> Dict[str, Any]:
        """
        Resize a single target dict.

        Args:
            target: Target dict with masks, boxes, label_map, etc.
            scale_factors: (sz, sy, sx) scale factors
            target_shape: Target spatial shape for this call

        Returns:
            Resized target dict
        """
        if not isinstance(target, dict):
            raise TypeError(f"Expected target to be dict, got {type(target)}")

        t = dict(target)

        if "masks" in t and t["masks"] is not None:
            t["masks"] = resize_masks(t["masks"], target_shape)

        if "label_map" in t and t["label_map"] is not None:
            t["label_map"] = resize_label_map(t["label_map"], target_shape)

        if "boxes" in t and t["boxes"] is not None:
            assert self.bbox_format is not None, \
                "bbox_format must be set to resize boxes"
            t["boxes"] = resize_boxes(t["boxes"], scale_factors, self.bbox_format)

        return t
