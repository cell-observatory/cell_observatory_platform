import logging
import sys
from typing import Literal, Union

import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp, SwiGLU
from torch.nn import Module

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_mlp(ff: Union[Module, Literal["Mlp", "SwiGLU"]] = "Mlp"):
    if ff == "Mlp" or type(ff) == type(Mlp):
        return Mlp

    elif ff == "SwiGLU" or type(ff) == type(SwiGLU):
        return SwiGLU

    else:
        raise ValueError(f"Unknown MLP layer: {ff}")


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
