import logging
import sys
from typing import Literal, Union

import torch.nn as nn

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_activation(act: Union[nn.Module, Literal['GELU', 'SiLU', 'LeakyReLU', 'GLU', 'Sigmoid', 'Tanh']] = 'GELU'):
    if act == "GELU" or isinstance(act, nn.GELU):
        return nn.GELU

    elif act == "SiLU" or isinstance(act, nn.SiLU):
        return nn.SiLU

    elif act == "LeakyReLU" or isinstance(act, nn.LeakyReLU):
        return nn.LeakyReLU

    elif act == "GLU" or isinstance(act, nn.GLU):
        return nn.GLU

    elif act == "Sigmoid" or isinstance(act, nn.Sigmoid):
        return nn.Sigmoid

    elif act == "Tanh" or isinstance(act, nn.Tanh):
        return nn.Tanh

    else:
        raise ValueError(f"Unknown activation layer: {act}")
