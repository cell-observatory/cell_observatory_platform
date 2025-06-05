import sys
import logging
import torch
from enum import Enum
from typing import Tuple


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
