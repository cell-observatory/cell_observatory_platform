import logging
import sys
from typing import Literal, Union
import torch.nn as nn
from timm.layers import RmsNorm


logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_norm(norm: Union[nn.Module, Literal['RmsNorm', 'LayerNorm', 'SyncBatchNorm', 'GroupNorm']] = 'LayerNorm'):
    if norm == "RmsNorm" or isinstance(norm, RmsNorm):
        return RmsNorm

    elif norm == "LayerNorm" or isinstance(norm, nn.LayerNorm):
        return nn.LayerNorm

    elif norm == "SyncBatchNorm" or isinstance(norm, nn.SyncBatchNorm):
        return nn.SyncBatchNorm

    elif norm == "GroupNorm" or isinstance(norm, nn.GroupNorm):
        return nn.GroupNorm

    else:
        raise ValueError(f"Unknown normalization layer: {norm}")
