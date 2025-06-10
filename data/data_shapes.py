import sys
import logging
import torch
from enum import Enum
from typing import Tuple, Dict

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MULTICHANNEL_3D_HYPERCUBE(Enum):
    """Spatiotemporal 3D multichannel hypercube layouts.

    * ``C[Z/T]YX`` - channel-first
    ``(..., C, Z/T, Y, X)``  e.g. ``(N, C, D/T, H, W)``

    * ``[Z/T]YXC`` - channel-last
      ``(..., Z/T, Y, X, C)``  e.g. ``(N, D/T, H, W, C)``
    """

    # spatial
    CZYX = "CZYX"
    ZYXC = "ZYXC"

    # temporal
    CTYX = "CTYX"
    TYXC = "TYXC"

    @property
    def axes(self) -> Tuple[str, ...]:
        return tuple(self.value)  # e.g. ("C","Z","Y","X")

    def is_channel_first(self) -> bool:
        return self in (MULTICHANNEL_3D_HYPERCUBE.CZYX, MULTICHANNEL_3D_HYPERCUBE.CTYX)

    def is_channel_last(self) -> bool:
        return self in (MULTICHANNEL_3D_HYPERCUBE.ZYXC, MULTICHANNEL_3D_HYPERCUBE.TYXC)

    def has_temporal_dim(self) -> bool:
        return self in (MULTICHANNEL_3D_HYPERCUBE.CTYX, MULTICHANNEL_3D_HYPERCUBE.TYXC)

    def get_image_shape_tuple(self, tensor: torch.Tensor) -> Tuple:
        has_batch = tensor.ndim == 5  # (N, Z/T, Y, X, C)
        shape = tensor.shape #eg (512, 128, 128, 128, 2)

        if self is MULTICHANNEL_3D_HYPERCUBE.ZYXC or self is MULTICHANNEL_3D_HYPERCUBE.TYXC:
            return shape[1:-1] if has_batch else shape[:-1] #eg (128, 128, 128)

        elif self is MULTICHANNEL_3D_HYPERCUBE.CZYX or self is MULTICHANNEL_3D_HYPERCUBE.CTYX:
            return shape[-3:]

        else:
            raise NotImplementedError(f"Unsupported layout {shape}")

    def get_image_shape_dict(self, tensor: torch.Tensor) -> Dict:
        has_batch = tensor.ndim == 5  # (N, Z/T, Y, X, C)
        shape = tensor.shape #eg (512, 128, 128, 128, 2)

        if self is MULTICHANNEL_3D_HYPERCUBE.ZYXC:
            return dict(z=shape[1], y=shape[2], x=shape[3], c=shape[4]) \
                if has_batch else dict(z=shape[0], y=shape[1], x=shape[2], c=shape[3])

        elif self is MULTICHANNEL_3D_HYPERCUBE.TYXC:
            return dict(t=shape[1], y=shape[2], x=shape[3], c=shape[4]) \
                if has_batch else dict(t=shape[0], y=shape[1], x=shape[2], c=shape[3])

        elif self is MULTICHANNEL_3D_HYPERCUBE.CZYX:
            return dict(c=shape[-4], z=shape[-3], y=shape[-2], x=shape[-1])

        elif self is MULTICHANNEL_3D_HYPERCUBE.CTYX:
            return dict(c=shape[-4], t=shape[-3], y=shape[-2], x=shape[-1])
        else:
            raise NotImplementedError(f"Unsupported layout {shape}")

    def get_spatial_shape(self, tensor: torch.Tensor) -> Tuple[int, int, int] | Tuple[int, int]:
        d = self.get_image_shape_dict(tensor)

        if self is MULTICHANNEL_3D_HYPERCUBE.ZYXC:
            return (d['z'], d['y'], d['x'])
        elif self is MULTICHANNEL_3D_HYPERCUBE.TYXC:
            return (d['y'], d['x'])
        else:
            raise ValueError(f'Tensor has an unsupported layout {self}')

    def get_temporal_shape(self, tensor: torch.Tensor) -> Tuple:
        d = self.get_image_shape_dict(tensor)

        if self is MULTICHANNEL_3D_HYPERCUBE.TYXC:
            return d['t']
        else:
            raise ValueError(f'Tensor does not have a temporal dim {self}')

    def to_channel_first(self, tensor: torch.Tensor) -> torch.Tensor:

        if self is MULTICHANNEL_3D_HYPERCUBE.CZYX or self is MULTICHANNEL_3D_HYPERCUBE.CTYX:
            return tensor  # already correct

        has_batch = tensor.ndim == 5  # (N, Z/T, Y, X, C)
        perm = (0, 4, 1, 2, 3) if has_batch else (3, 0, 1, 2)
        return tensor.permute(*perm)

    def to_channel_last(self, tensor: torch.Tensor) -> torch.Tensor:
        if self is MULTICHANNEL_3D_HYPERCUBE.ZYXC or self is MULTICHANNEL_3D_HYPERCUBE.TYXC :
            return tensor

        has_batch = tensor.ndim == 5  # (N, C, Z/T, Y, X)
        perm = (0, 2, 3, 4, 1) if has_batch else (1, 2, 3, 0)
        return tensor.permute(*perm)

    def num_channels(self, tensor: torch.Tensor) -> int:
        d = self.get_image_shape_dict(tensor)
        return d['c']

    def num_timepoints(self, tensor: torch.Tensor) -> int | None:
        d = self.get_image_shape_dict(tensor)
        return d.get('t', None)


class MULTICHANNEL_4D_HYPERCUBE(Enum):
    """Spatiotemporal 4D multichannel hypercube layouts.

    * ``CTZYX`` - channel-first
    ``(..., C, T, Z, Y, X)``  e.g. ``(N, C, T, D, H, W)``

    * ``TZYXC`` - channel-last
      ``(..., T, Z, Y, X, C)``  e.g. ``(N, T, D, H, W, C)``
    """

    CTZYX = "CTZYX"
    TZYXC = "TZYXC"

    @property
    def axes(self) -> Tuple[str, ...]:
        return tuple(self.value)  # e.g. ("C","T","Z","Y","X")

    def is_channel_first(self) -> bool:
        return self is MULTICHANNEL_4D_HYPERCUBE.CTZYX

    def is_channel_last(self) -> bool:
        return self is MULTICHANNEL_4D_HYPERCUBE.TZYXC

    def has_temporal_dim(self) -> bool:
        return self is MULTICHANNEL_4D_HYPERCUBE.CTZYX

    def get_image_shape_tuple(self, tensor: torch.Tensor) -> Tuple:

        has_batch = tensor.ndim == 6  # (N, T, Z, Y, X, C)
        shape = tensor.shape #eg (512, 16, 128, 128, 128, 2)

        if self is MULTICHANNEL_4D_HYPERCUBE.TZYXC:
            return shape[1:-1] if has_batch else shape[:-1] #eg (16, 128, 128, 128)

        elif self is MULTICHANNEL_4D_HYPERCUBE.CTZYX:
            return shape[-4:]

        else:
            raise NotImplementedError(f"Unsupported layout {shape}")

    def get_image_shape_dict(self, tensor: torch.Tensor) -> Dict:
        has_batch = tensor.ndim == 6  # (N, T, Z, Y, X, C)
        shape = tensor.shape  # eg (512, 16, 128, 128, 128, 2)

        if self is MULTICHANNEL_4D_HYPERCUBE.TZYXC:
            return dict(t=shape[1], z=shape[2], y=shape[3], x=shape[4], c=shape[5]) \
                if has_batch else dict(t=shape[0], z=shape[1], y=shape[2], x=shape[3], c=shape[4])

        elif self is MULTICHANNEL_4D_HYPERCUBE.CTZYX:
            return dict(c=shape[-5], t=shape[-4], z=shape[-3], y=shape[-2], x=shape[-1])

        else:
            raise NotImplementedError(f"Unsupported layout {shape}")

    def get_spatial_shape(self, tensor: torch.Tensor) -> Tuple[int, int, int]:
        d = self.get_image_shape_dict(tensor)
        return (d['z'], d['y'], d['x'])

    def get_temporal_shape(self, tensor: torch.Tensor) -> Tuple:
        d = self.get_image_shape_dict(tensor)
        return d['t']

    def to_channel_first(self, tensor: torch.Tensor) -> torch.Tensor:
        if self is MULTICHANNEL_4D_HYPERCUBE.CTZYX:
            return tensor  # already correct

        has_batch = tensor.ndim == 6  # (N, T, Z, Y, X, C)
        perm = (0, 5, 1, 2, 3, 4) if has_batch else (4, 0, 1, 2, 3)
        return tensor.permute(*perm)

    def to_channel_last(self, tensor: torch.Tensor) -> torch.Tensor:
        if self is MULTICHANNEL_4D_HYPERCUBE.TZYXC:
            return tensor

        has_batch = tensor.ndim == 6  # (N, C, T, Z, Y, X)
        perm = (0, 2, 3, 4, 5, 1) if has_batch else (1, 2, 3, 4, 0)
        return tensor.permute(*perm)

    def num_channels(self, tensor: torch.Tensor) -> int:
        d = self.get_image_shape_dict(tensor)
        return d['c']

    def num_timepoints(self, tensor: torch.Tensor) -> int | None:
        d = self.get_image_shape_dict(tensor)
        return d.get('t', None)