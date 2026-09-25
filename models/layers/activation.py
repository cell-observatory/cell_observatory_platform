import logging
import sys
from typing import Literal, Union

import torch.nn as nn

logger = logging.getLogger(__name__)


def get_activation(act='GELU'):
    # already an instance, return its class
    if isinstance(act, nn.Module):
        return act.__class__

    # a class and a subclass of nn.Module, return as-is
    if isinstance(act, type) and issubclass(act, nn.Module):
        return act

    # a string, map case-insensitively
    if isinstance(act, str):
        key = act.strip().lower()
        table = {
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'swish': nn.SiLU,
            'leakyrelu': nn.LeakyReLU,
            'lrelu': nn.LeakyReLU,
            'glu': nn.GLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
        }
        if key in table:
            return table[key]

    raise ValueError(f"Unknown activation layer: {act!r}")