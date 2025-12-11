import sys
import logging

import torch

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Normalize:
    def __init__(self, input_layout, eps: float = 1e-4) -> None:
        """
        Args:
            input_layout: MULTICHANNEL_HYPERCUBE layout object
            eps: minimum std value (to avoid division by zero)
        """
        self.input_layout = input_layout
        self.eps = eps

    def _compute_mean_std(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_work = x

        # unsqueeze batch dim if missing
        if self.input_layout.is_3d() and x_work.ndim == 4:
            # (C, D, H, W) or (D, H, W, C) -> (1, C, D, H, W) / (1, D, H, W, C)
            x_work = x_work.unsqueeze(0)
        elif self.input_layout.is_4d() and x_work.ndim == 5:
            # (T, C, D, H, W) or (T, D, H, W, C) -> (1, T, C, D, H, W) / (1, T, D, H, W, C)
            x_work = x_work.unsqueeze(0)

        if self.input_layout.is_3d():
            # 3D volumes: (B, C, D, H, W) or (B, D, H, W, C)
            if self.input_layout.is_channel_first():  # (B, C, D, H, W)
                mean = x_work.mean(dim=(2, 3, 4), keepdim=True)
                std = x_work.std(dim=(2, 3, 4), keepdim=True)
            else:  # (B, D, H, W, C)
                mean = x_work.mean(dim=(1, 2, 3), keepdim=True)
                std = x_work.std(dim=(1, 2, 3), keepdim=True)

        elif self.input_layout.is_4d():
            # 4D data: (B, C, T, D, H, W) or (B, T, D, H, W, C)
            if self.input_layout.is_channel_first():  # (B, C, T, D, H, W)
                mean = x_work.mean(dim=(2, 3, 4, 5), keepdim=True)
                std = x_work.std(dim=(2, 3, 4, 5), keepdim=True)
            else:  # (B, T, D, H, W, C)
                mean = x_work.mean(dim=(1, 2, 3, 4), keepdim=True)
                std = x_work.std(dim=(1, 2, 3, 4), keepdim=True)
        else:
            raise NotImplementedError(f"Unsupported layout {self.input_layout}")

        return x_work, mean, std

    def _normalize_tensor(self, data_tensor: torch.Tensor) -> torch.Tensor:
        x_work, mean, std = self._compute_mean_std(data_tensor)
        std = std.clamp_min(self.eps)
        image = (x_work - mean) / std
        return image

    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return self._normalize_tensor(data)

        if isinstance(data, dict):
            if "data_tensor" not in data:
                raise KeyError("Normalize expects 'data_tensor' in dict.")
            data_tensor = data["data_tensor"]
            norm_tensor = self._normalize_tensor(data_tensor)

            out = dict(data)
            out["data_tensor"] = norm_tensor
            return out

        raise TypeError(f"Normalize expects torch.Tensor or dict, got {type(data)}")