import logging
import sys
from typing import Literal, Union

import torch.nn as nn
from timm.layers import RmsNorm

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_norm(norm: Union[nn.Module, Literal["RmsNorm", "LayerNorm", "SyncBatchNorm", "GroupNorm"]] = "LayerNorm"):
    if norm == "RmsNorm" or type(norm) == type(RmsNorm):
        return RmsNorm

    elif norm == "LayerNorm" or type(norm) == type(nn.LayerNorm):
        return nn.LayerNorm

    elif norm == "SyncBatchNorm" or type(norm) == type(nn.SyncBatchNorm):
        return nn.SyncBatchNorm

    elif norm == "GroupNorm" or type(norm) == type(nn.GroupNorm):
        return nn.GroupNorm

    else:
        raise ValueError(f"Unknown normalization layer: {norm}")


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
