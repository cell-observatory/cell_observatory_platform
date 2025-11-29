from typing import Sequence, Tuple, Dict, Any, List, Union, Optional

import torch
import torch.nn.functional as F

try:
    from scipy.ndimage import zoom as scipy_zoom
except ImportError:
    scipy_zoom = None

from cell_observatory_platform.data.data_types import TORCH_DTYPES


class Resize:
    """
    Channels-last 3D resize.

    Supports:
      - input_format="ZYXC":  tensor shape (B, Z, Y, X, C)

    Can be called on:
      - a plain tensor, or
      - a data_sample dict with keys:
          - "data_tensor": image tensor
          - "metainfo": {
                "targets": List[Dict] or [List[Dict]],
                each target dict may have "masks" and "boxes",
                and optionally "padding_mask"
            }
    """

    def __init__(
        self,
        input_format: str,
        target_spatial_shape: Sequence[int],
        mode: str = "trilinear",
        align_corners: bool = False,
        dtype: str = "bfloat16",
        bbox_format: str | None = None,
        crop_to_meta_spatial: bool = False,
    ) -> None:
        input_format = input_format.upper()
        if input_format not in ("ZYXC"):
            raise ValueError(
                f"Resize only supports input_format='ZYXC', "
                f"got {input_format!r}"
            )

        self.input_format = input_format
        self.target_spatial_shape: Tuple[int, int, int] = tuple(
            int(d) for d in target_spatial_shape
        )
        self.dim = len(self.target_spatial_shape)
        if self.dim != 3:
            raise ValueError(
                f"target_spatial_shape currently must be (Z, Y, X), "
                f"got {self.target_spatial_shape}"
            )

        self.mode = mode
        self.align_corners = align_corners
        self.use_scipy_zoom = (self.mode == "zoom")
        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        self.bbox_format = bbox_format

        self.crop_to_meta_spatial = crop_to_meta_spatial

    def __call__(
        self,
        data: Union[torch.Tensor, Dict[str, Any]],
        resize_buffer: Optional[torch.Tensor] = None,
    ):
        if isinstance(data, torch.Tensor):
            resized_tensor, _ = self._resize_tensor(data, out=resize_buffer)
            return resized_tensor

        if not isinstance(data, dict):
            raise TypeError(
                f"Resize expects a torch.Tensor or dict with 'data_tensor'/'metainfo', got {type(data)}"
            )

        if "data_tensor" not in data:
            raise KeyError("Expected key 'data_tensor' in data_sample dict.")

        inputs = data["data_tensor"]
        metainfo = data.get("metainfo")
        if resize_buffer is None:
            resize_buffer = metainfo.get("resize_buffer", None) \
                if metainfo is not None else None

        z_sizes = metainfo.get("z_size", None)
        y_sizes = metainfo.get("y_size", None)
        x_sizes = metainfo.get("x_size", None)

        if z_sizes is not None and y_sizes is not None \
            and x_sizes is not None and self.crop_to_meta_spatial:
            resized_inputs, scale_factors_per_sample = self._resize_tensor_with_meta(
                inputs, z_sizes, y_sizes, x_sizes, out=resize_buffer
            )
            resized_metainfo = self._resize_metainfo_with_meta(
                metainfo,
                scale_factors_per_sample,
                z_sizes,
                y_sizes,
                x_sizes,
                inputs.shape[0]
            )
        else:
            resized_inputs, scale_factors = self._resize_tensor(inputs, out=resize_buffer)
            resized_metainfo = self._resize_metainfo(metainfo, scale_factors)

        return {
            "data_tensor": resized_inputs,
            "metainfo": resized_metainfo,
        }
    
    def _resize_tensor_with_meta(
        self,
        data_tensor: torch.Tensor,
        z_sizes: Any,
        y_sizes: Any,
        x_sizes: Any,
        out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[float, float, float]]]:
        x = data_tensor

        if self.input_format == "ZYXC":
            if x.ndim != 5:
                raise ValueError(
                    f"Resize (meta): expected 5D tensor (B, Z, Y, X, C) for 'ZYXC', "
                    f"got shape {tuple(x.shape)}"
                )
            B, Z_pad, Y_pad, X_pad, C = x.shape
        else:
            raise RuntimeError("Unsupported input_format in Resize._resize_tensor_with_meta")

        if out is None:
            resized_samples = []

        scale_factors_list: List[Tuple[float, float, float]] = []
        for b in range(B):
            zb = min(z_sizes[b], Z_pad)
            yb = min(y_sizes[b], Y_pad)
            xb = min(x_sizes[b], X_pad)

            if self.input_format == "ZYXC":
                # (1, Zb, Yb, Xb, C)
                sample = x[b : b + 1, :zb, :yb, :xb, :]
            else:
                raise RuntimeError("Unsupported input_format in Resize._resize_tensor_with_meta")

            sample_out: Optional[torch.Tensor]
            if out is not None:
                sample_out = out[b : b + 1, ...]
            else:
                sample_out = None

            sample_resized, scale_factors = self._resize_tensor(sample, out=sample_out)
            
            if out is None:
                resized_samples.append(sample_resized)

            scale_factors_list.append(scale_factors)

        if out is not None:
            resized_tensor = out
        else:
            resized_tensor = torch.cat(resized_samples, dim=0)

        return resized_tensor, scale_factors_list

    def _resize_metainfo_with_meta(
        self,
        metainfo: Dict[str, Any],
        scale_factors_per_sample: List[Tuple[float, float, float]],
        z_sizes: Any,
        y_sizes: Any,
        x_sizes: Any,
        batch_size: int,
    ) -> Dict[str, Any]:
        if (
            "targets" not in metainfo
            and "padding_mask" not in metainfo
            and "image_sizes" not in metainfo
            and "orig_image_sizes" not in metainfo
        ):
            raise ValueError(
                "Resize (meta): metainfo does not have expected resizable fields."
            )

        out = dict(metainfo)

        if "targets" in out:
            targets = out["targets"]
            resized_targets: List[Dict[str, Any]] = []
            for b, tgt in enumerate(targets):
                if not isinstance(tgt, dict):
                    raise TypeError(
                        f"Resize (meta): expected target to be dict, got {type(tgt)}"
                    )

                t = dict(tgt)
                if "masks" in t and t["masks"] is not None:
                    t["masks"] = self._resize_masks(t["masks"])

                if "boxes" in t and t["boxes"] is not None:
                    sf = scale_factors_per_sample[b]
                    t["boxes"] = self._resize_boxes(t["boxes"], sf)

                resized_targets.append(t)

            out["targets"] = resized_targets

        if "image_sizes" in out:
            img_sizes = out["image_sizes"]
            new_sz = torch.tensor(
                self.target_spatial_shape,
                device=img_sizes.device,
                dtype=img_sizes.dtype,
            )
            out["image_sizes"] = new_sz.unsqueeze(0).repeat(batch_size, 1)

        if "orig_image_sizes" in out:
            orig = out["orig_image_sizes"]
            device = orig.device
            dtype = orig.dtype

            new_orig = torch.empty((batch_size, 3), device=device, dtype=dtype)
            for b in range(batch_size):
                new_orig[b, 0] = z_sizes[b]
                new_orig[b, 1] = y_sizes[b]
                new_orig[b, 2] = x_sizes[b]
            out["orig_image_sizes"] = new_orig

        if "padding_mask" in out:
            pm = out["padding_mask"]
            if torch.is_tensor(pm) and pm.numel() > 0 and pm.ndim in (4, 5):
                out["padding_mask"] = self._resize_padding_mask(pm)

        return out

    def _resize_tensor(
        self,
        data_tensor: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ):
        x = data_tensor

        if self.input_format == "ZYXC":
            # (B, Z, Y, X, C)
            if x.ndim != 5:
                raise ValueError(
                    f"Expected 5D tensor (B, Z, Y, X, C) for input_format='ZYXC', "
                    f"got shape {tuple(x.shape)}"
                )
            B, Z, Y, X, C = x.shape

        else:
            raise RuntimeError("Unsupported input_format in Resize")

        x = x.to(self.dtype)

        if self.use_scipy_zoom:
            if scipy_zoom is None:
                raise RuntimeError("Resize: mode='zoom' requires SciPy to be installed.")

            sz = self.target_spatial_shape[0] / float(Z)
            sy = self.target_spatial_shape[1] / float(Y)
            sx = self.target_spatial_shape[2] / float(X)

            if self.input_format == "ZYXC":
                # (B, Z, Y, X, C)
                zoom_factors = (1.0, sz, sy, sx, 1.0)
            else:
                raise RuntimeError("Unsupported input_format in Resize")

            x_np = x.detach().cpu().numpy()

            if out is not None:
                if out.shape != (B, *self.target_spatial_shape, C):
                    raise ValueError(
                        f"Resize: provided out tensor has shape {tuple(out.shape)}, "
                        f"but expected {(B, *self.target_spatial_shape, C)}"
                    )
                out_np = out.detach().cpu().numpy()
                scipy_zoom(x_np, zoom=zoom_factors, order=1, output=out_np)
                x_resized = out
            else:
                x_resized_np = scipy_zoom(x_np, zoom=zoom_factors, order=1)
                x_resized = torch.from_numpy(x_resized_np).to(self.dtype)

            x = x_resized

        else:
            Z_new, Y_new, X_new = self.target_spatial_shape

            if self.input_format == "ZYXC":
                # (B, Z, Y, X, C) -> (B, C, Z, Y, X)
                x_cf = x.permute(0, 4, 1, 2, 3).contiguous()
                B_eff = B
            else:
                raise RuntimeError("Unsupported input_format in Resize")

            if self.mode in ("nearest", "area"):
                x_cf = F.interpolate(
                    x_cf,
                    size=self.target_spatial_shape,
                    mode=self.mode,
                )
            else:
                x_cf = F.interpolate(
                    x_cf,
                    size=self.target_spatial_shape,
                    mode=self.mode,
                    align_corners=self.align_corners,
                )

            if self.input_format == "ZYXC":
                # (B, C, Z_new, Y_new, X_new) -> (B, Z_new, Y_new, X_new, C)
                x_new = x_cf.permute(0, 2, 3, 4, 1).contiguous()
            else:
                raise RuntimeError("Unsupported input_format in Resize")

            if out is not None:
                if out.shape != x_new.shape:
                    raise ValueError(
                        f"Resize: provided out tensor has shape {tuple(out.shape)}, "
                        f"but resized tensor has shape {tuple(x_new.shape)}"
                    )
                # NOTE: one extra copy compared to zoom mode
                out.copy_(x_new)
                x = out
            else:
                x = x_new

        # scale factors based on original spatial size
        scale_factors = (
            self.target_spatial_shape[0] / float(Z),
            self.target_spatial_shape[1] / float(Y),
            self.target_spatial_shape[2] / float(X),
        )

        return x, scale_factors

    def _resize_metainfo(
        self,
        metainfo: Dict[str, Any],
        scale_factors: Tuple[float, float, float],
    ) -> Dict[str, Any]:
        if "targets" not in metainfo and "padding_mask" not in metainfo \
            and "image_sizes" not in metainfo:
            raise ValueError(
                "Resize: metainfo does not have expected resizable fields."
            )

        if "targets" in metainfo:
            targets = metainfo["targets"]
            if not isinstance(targets, list):
                raise TypeError(
                    f"Resize: expected targets to be a list, got {type(targets)}"
                )
            resized_targets: List[Dict[str, Any]] = []
            for tgt in targets:
                if not isinstance(tgt, dict):
                    raise TypeError(
                        f"Resize: expected target to be dict, got {type(tgt)}"
                    )

                tgt = dict(tgt)

                if "masks" in tgt and tgt["masks"] is not None:
                    tgt["masks"] = self._resize_masks(tgt["masks"])

                if "boxes" in tgt and tgt["boxes"] is not None:
                    tgt["boxes"] = self._resize_boxes(tgt["boxes"], scale_factors)

                resized_targets.append(tgt)

            metainfo["targets"] = resized_targets

        if "image_sizes" in metainfo:
            img_sizes = metainfo["image_sizes"]
            if torch.is_tensor(img_sizes):
                B_img = img_sizes.shape[0]
                new_sz = torch.tensor(
                    self.target_spatial_shape,
                    device=img_sizes.device,
                    dtype=img_sizes.dtype,
                )
                metainfo["image_sizes"] = new_sz.unsqueeze(0).repeat(B_img, 1)

        if "padding_mask" in metainfo:
            pm = metainfo["padding_mask"]
            if torch.is_tensor(pm) and pm.numel() > 0:
                if pm.ndim in (4, 5):
                    metainfo["padding_mask"] = self._resize_padding_mask(pm)

        return metainfo

    def _resize_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Resize masks with nearest neighbor.

        Expected shapes:
          - (N, Z, Y, X)
        """
        if masks.numel() == 0:
            return masks

        orig_dtype = masks.dtype

        if masks.ndim == 4:
            # (N, Z, Y, X) -> (N, 1, Z, Y, X)
            N, Zm, Ym, Xm = masks.shape
            m = masks.unsqueeze(1)
            m = F.interpolate(m, size=self.target_spatial_shape, mode="nearest")
            m = m.squeeze(1)
            return m.to(orig_dtype)

        else:
            raise ValueError(
                f"Unsupported masks ndim={masks.ndim}; expected 4 or 5 dims."
            )

    def _resize_padding_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Resize padding_mask with nearest neighbor.

        Supported shapes:
          - (B, Z, Y, X)

        Returns same dtype as input (typically bool or uint8).
        """
        if padding_mask.numel() == 0:
            return padding_mask

        orig_dtype = padding_mask.dtype

        if padding_mask.ndim == 4:
            # (B, Z, Y, X) -> (B, 1, Z, Y, X)
            B, Zm, Ym, Xm = padding_mask.shape
            m = padding_mask.to(torch.float32).unsqueeze(1)
            m = F.interpolate(m, size=self.target_spatial_shape, mode="nearest")
            m = m.squeeze(1)
            return m.to(orig_dtype)

        else:
            raise ValueError(
                f"Unsupported padding_mask ndim={padding_mask.ndim}; expected 4 dims."
            )

    def _resize_boxes(
        self,
        boxes: torch.Tensor,
        scale_factors: Tuple[float, float, float],
    ) -> torch.Tensor:
        """
        Resize boxes in absolute voxel coordinates.

        Supported formats:
          - "zyxzyx":   [z1, y1, x1, z2, y2, x2]
          - "cxcyczwhd": [cx, cy, cz, w, h, d]
        """
        if boxes.numel() == 0:
            return boxes

        sz, sy, sx = scale_factors
        out = boxes.clone()

        if out.shape[-1] != 6:
            raise ValueError(
                f"Resize expects last dim of boxes to be 6, got {out.shape[-1]}"
            )

        fmt = self.bbox_format.lower()

        if fmt == "zyxzyx":
            # z1, z2
            out[..., 0] = out[..., 0] * sz
            out[..., 3] = out[..., 3] * sz
            # y1, y2
            out[..., 1] = out[..., 1] * sy
            out[..., 4] = out[..., 4] * sy
            # x1, x2
            out[..., 2] = out[..., 2] * sx
            out[..., 5] = out[..., 5] * sx

        elif fmt == "cxcyczwhd":
            # We assume ordering:
            # [cx, cy, cz, w, h, d]
            out[..., 0] = out[..., 0] * sx  # cx
            out[..., 1] = out[..., 1] * sy  # cy
            out[..., 2] = out[..., 2] * sz  # cz

            out[..., 3] = out[..., 3] * sx  # w
            out[..., 4] = out[..., 4] * sy  # h
            out[..., 5] = out[..., 5] * sz  # d

        else:
            raise ValueError(
                f"Unsupported bbox_format={self.bbox_format!r} in Resize._resize_boxes"
            )

        return out