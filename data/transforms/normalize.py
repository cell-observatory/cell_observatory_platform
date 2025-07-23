import sys
import logging
import torch

from nvidia.dali import fn
from data.data_types import DALI_DTYPES

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, data_sample):
        image = data_sample.data_tensor.tensor

        if self.mean is None and self.std is None:
            mean, std = data_sample.data_tensor.get_image_stats()
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