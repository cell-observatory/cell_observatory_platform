import sys
import logging

import torch

from nvidia.dali import fn

from data.data_types import DALI_DTYPES
from data.structures.image_list import ImageList

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Normalize:
    def __init__(self, mean=None, std=None, eps=1e-4):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, data_sample):
        image = data_sample.data_tensor.tensor

        if self.mean is None and self.std is None:
            mean, std = data_sample.data_tensor.get_image_stats()
            std = std.clamp_min(self.eps)
            image = (image - mean) / std
        else:
            mean = torch.tensor(self.mean, dtype=image.dtype, device=image.device)
            std = torch.tensor(self.std, dtype=image.dtype, device=image.device)
            image = (image - mean) / std

        data_sample.data_tensor.tensor = image
        return data_sample


def NormalizeDaliWrapper(data, dtype):
    vol_f32 = fn.cast(data, dtype=DALI_DTYPES.float32.value)
    vol_norm = fn.normalize(
        vol_f32,
        axes=[1, 2, 3, 4],
        batch=True
    )
    vol_out = fn.cast(vol_norm, dtype=dtype)
    return vol_out


class NormalizeRayWrapper:
    def __init__(self, input_layout, eps=1e-4) -> None:
        self.input_layout = input_layout
        self.eps = eps

    def _normalize_tensor(self, data_tensor: torch.Tensor) -> torch.Tensor:
        image_list = ImageList(
            data_tensor, layout=self.input_layout, image_sizes=[data_tensor.shape]
        )
        mean, std = image_list.get_image_stats()
        std = std.clamp_min(self.eps)
        image = (image_list.tensor - mean) / std
        return image

    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return self._normalize_tensor(data)
        if isinstance(data, dict):
            if "data_tensor" not in data:
                raise KeyError("NormalizeRayWrapper expects 'data_tensor' in dict.")
            data_tensor = data["data_tensor"]
            norm_tensor = self._normalize_tensor(data_tensor)

            out = dict(data)
            out["data_tensor"] = norm_tensor
            return out
        raise TypeError(
            f"NormalizeRayWrapper expects torch.Tensor or dict, got {type(data)}"
        )