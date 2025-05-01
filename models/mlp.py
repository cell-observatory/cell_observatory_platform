import logging
import sys
from typing import Literal, Union

from torch.nn import Module
from timm.layers import Mlp, SwiGLU


logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_mlp(ff: Union[Module, Literal['Mlp', 'SwiGLU']] = 'Mlp'):
    if ff == "Mlp" or isinstance(ff, Mlp):
        return Mlp

    elif ff == "SwiGLU" or isinstance(ff, SwiGLU):
        return SwiGLU

    else:
        raise ValueError(f"Unknown MLP layer: {ff}")
