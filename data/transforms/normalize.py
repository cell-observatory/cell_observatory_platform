import sys
import logging

import torch

logger = logging.getLogger(__name__)


class Normalize:
    def __init__(self, input_layout, eps: float = 1e-4, normalize_target_roles=None) -> None:
        """
        Args:
            input_layout: MULTICHANNEL_HYPERCUBE layout object
            eps: minimum std value (to avoid division by zero)
            normalize_target_roles: Names of Form-D target roles (see
                data/data_types.py) to normalize with the SAME per-sample (and
                per-channel) statistics computed from ``data_tensor``. This is
                the denoising contract: the clean clone snapshotted by
                ``DeepCopyInputsAsTargets`` BEFORE this transform is placed in
                the same AFFINE FRAME as the input the model sees -- otherwise
                the loss compares a z-scored input against raw values
                (scale/offset gap dominates). None/[] normalizes no targets;
                roles not named here (e.g. masks) pass through untouched.
        """
        self.input_layout = input_layout
        self.eps = eps
        self.normalize_target_roles = list(normalize_target_roles or [])

    def _reduce_dims(self) -> tuple[int, ...]:
        """Per-sample (and per-channel) reduction dims for the BATCHED tensor."""
        if self.input_layout.is_3d():
            return (2, 3, 4) if self.input_layout.is_channel_first() else (1, 2, 3)
        if self.input_layout.is_4d():
            return (2, 3, 4, 5) if self.input_layout.is_channel_first() else (1, 2, 3, 4)
        raise NotImplementedError(f"Unsupported layout {self.input_layout}")

    def _compute_mean_std(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_work = x

        # unsqueeze batch dim if missing
        if self.input_layout.is_3d() and x_work.ndim == 4:
            # (C, D, H, W) or (D, H, W, C) -> (1, C, D, H, W) / (1, D, H, W, C)
            x_work = x_work.unsqueeze(0)
        elif self.input_layout.is_4d() and x_work.ndim == 5:
            # (T, C, D, H, W) or (T, D, H, W, C) -> (1, T, C, D, H, W) / (1, T, D, H, W, C)
            x_work = x_work.unsqueeze(0)

        reduce_dims = self._reduce_dims()

        if padding_mask is not None and padding_mask.any():
            # Padded voxels (True in the mask) are buffer filler, not content:
            # including them dilutes mean toward 0 and skews std for every
            # variable-size tile squeezed into the dataset-max buffer. Compute
            # masked per-sample moments instead (matching torch.std's default
            # unbiased correction=1 so the no-padding case stays equivalent).
            if padding_mask.shape[0] != x_work.shape[0]:
                raise ValueError(
                    f"padding_mask batch {tuple(padding_mask.shape)} does not "
                    f"match data batch {tuple(x_work.shape)}"
                )
            valid = ~padding_mask
            if self.input_layout.is_channel_first():
                valid = valid.unsqueeze(1)              # (B, 1, [T,] D, H, W)
            else:
                valid = valid.unsqueeze(-1)             # (B, [T,] D, H, W, 1)
            if valid.ndim != x_work.ndim:
                raise ValueError(
                    f"padding_mask rank {padding_mask.ndim} incompatible with "
                    f"data rank {x_work.ndim} for layout {self.input_layout}"
                )
            valid_f = valid.to(x_work.dtype)
            n = valid_f.sum(dim=reduce_dims, keepdim=True)
            mean = (x_work * valid_f).sum(dim=reduce_dims, keepdim=True) / n.clamp(min=1)
            var = (((x_work - mean) ** 2) * valid_f).sum(dim=reduce_dims, keepdim=True) / (
                n - 1
            ).clamp(min=1)
            std = var.sqrt()
        else:
            mean = x_work.mean(dim=reduce_dims, keepdim=True)
            std = x_work.std(dim=reduce_dims, keepdim=True)

        return x_work, mean, std

    def _normalize_tensor(self, data_tensor: torch.Tensor) -> torch.Tensor:
        x_work, mean, std = self._compute_mean_std(data_tensor)
        std = std.clamp_min(self.eps)
        image = (x_work - mean) / std
        return image

    def _normalize_batched(
        self, t: torch.Tensor, x_work: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Apply the data tensor's per-sample stats to one batched Form-D role.

        ``mean``/``std`` are keepdim per-sample (and per-channel) stats, so a
        plain broadcast aligns sample b's target with sample b's statistics.
        Shape equality with the (batch-normalized) input is enforced: anything
        else means this option names a non-clone role, which must fail loudly
        rather than corrupt GT.
        """
        t_work = t
        if self.input_layout.is_3d() and t_work.ndim == 4:
            t_work = t_work.unsqueeze(0)
        elif self.input_layout.is_4d() and t_work.ndim == 5:
            t_work = t_work.unsqueeze(0)
        if t_work.shape != x_work.shape:
            raise ValueError(
                "normalize_target_roles: target shape "
                f"{tuple(t_work.shape)} != data_tensor shape "
                f"{tuple(x_work.shape)} -- per-sample stats cannot be "
                "aligned. Normalizable roles must be DeepCopyInputsAsTargets "
                "clones of the input."
            )
        return (t_work - mean) / std

    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return self._normalize_tensor(data)

        if isinstance(data, dict):
            if "data_tensor" not in data:
                raise KeyError("Normalize expects 'data_tensor' in dict.")
            data_tensor = data["data_tensor"]
            # Exclude buffer zero-padding from the per-sample statistics when
            # the pipeline tracks it (True == padded voxel).
            pm = (data.get("metainfo") or {}).get("padding_mask")
            if not torch.is_tensor(pm):
                pm = None
            x_work, mean, std = self._compute_mean_std(data_tensor, padding_mask=pm)
            std = std.clamp_min(self.eps)
            norm_tensor = (x_work - mean) / std

            out = dict(data)
            out["data_tensor"] = norm_tensor
            if self.normalize_target_roles:
                metainfo = data.get("metainfo") or {}
                targets = metainfo.get("targets")
                if not targets:
                    raise ValueError(
                        "normalize_target_roles set but metainfo has no targets. "
                        "Place DeepCopyInputsAsTargets BEFORE Normalize in the "
                        "transforms list."
                    )
                if not isinstance(targets, dict):
                    raise ValueError(
                        "normalize_target_roles requires Form-D targets (role-keyed "
                        f"dict, see data/data_types.py); got {type(targets).__name__}. "
                        "Form-S (per-sample) GT must never be z-scored -- remove "
                        "normalize_target_roles from this config."
                    )
                out["metainfo"] = dict(metainfo)
                targets = dict(targets)
                for role in self.normalize_target_roles:
                    t = targets.get(role)
                    if t is None:
                        raise KeyError(
                            f"normalize_target_roles names {role!r}; targets has "
                            f"{list(targets)}"
                        )
                    if not torch.is_tensor(t):
                        raise ValueError(
                            f"normalize_target_roles: role {role!r} is not a tensor "
                            f"(got {type(t).__name__}) -- only batched Form-D roles "
                            "can be normalized."
                        )
                    targets[role] = self._normalize_batched(t, x_work, mean, std)
                out["metainfo"]["targets"] = targets
            return out

        raise TypeError(f"Normalize expects torch.Tensor or dict, got {type(data)}")