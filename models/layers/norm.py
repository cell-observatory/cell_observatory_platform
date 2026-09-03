"""
Adapted from:
https://github.com/facebookresearch/dinov3/main/dinov3/layers/rms_norm.py
"""

import sys
import logging
from typing import Literal, Union

import torch.nn as nn
from timm.layers import RmsNorm

logger = logging.getLogger(__name__)


def get_norm(norm: Union[nn.Module, Literal["RmsNorm", "LayerNorm", "SyncBatchNorm", "GroupNorm"]] = "LayerNorm"):
    if isinstance(norm, str):
        if norm == "RmsNorm":
            return RmsNorm
        elif norm == "LayerNorm":
            return nn.LayerNorm
        elif norm == "SyncBatchNorm":
            return nn.SyncBatchNorm
        elif norm == "GroupNorm":
            return nn.GroupNorm
        raise ValueError(f"Unknown normalization layer: {norm}")
    # Already-resolved class: pass through unchanged. The old
    # `type(norm) == type(RmsNorm)` check was `type == type` — true for EVERY
    # class — so any class input collapsed to the first branch (RmsNorm).
    if isinstance(norm, type):
        return norm
    raise TypeError(f"get_norm expects a name or a class, got {type(norm)!r}")


class LayerNorm3D(nn.Module):
    def __init__(self, normalized_shape, norm_layer=nn.LayerNorm):
        super().__init__()
        self.ln = norm_layer(normalized_shape) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        """
        x: N C D H W
        """
        x = x.permute(0, 2, 3, 4, 1)
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class LayerNorm4D(nn.Module):
    def __init__(self, normalized_shape, norm_layer=nn.LayerNorm):
        super().__init__()
        self.ln = norm_layer(normalized_shape) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        """
        x: N C T D H W
        """
        x = x.permute(0, 2, 3, 4, 5, 1)
        x = self.ln(x)
        x = x.permute(0, 5, 1, 2, 3, 4)
        return x


# TODO: consider following DINOv3 implementation of RMSNorm
# class RMSNorm(nn.Module):
#     def __init__(self, dim: int, eps: float = 1e-5):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(dim))
#         self.eps = eps

#     def reset_parameters(self) -> None:
#         nn.init.constant_(self.weight, 1)

#     def _norm(self, x: Tensor) -> Tensor:
#         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

#     def forward(self, x: Tensor) -> Tensor:
#         output = self._norm(x.float()).type_as(x)
#         return output * self.weight