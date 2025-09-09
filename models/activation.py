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
    if act == "GELU" or type(act) == type(nn.GELU):
        return nn.GELU

    elif act == "SiLU" or type(act) == type(nn.SiLU):
        return nn.SiLU

    elif act == "LeakyReLU" or type(act) == type(nn.LeakyReLU):
        return nn.LeakyReLU

    elif act == "GLU" or type(act) == type(nn.GLU):
        return nn.GLU

    elif act == "Sigmoid" or type(act) == type(nn.Sigmoid):
        return nn.Sigmoid

    elif act == "Tanh" or type(act) == type(nn.Tanh):
        return nn.Tanh

    else:
        raise ValueError(f"Unknown activation layer: {act}")