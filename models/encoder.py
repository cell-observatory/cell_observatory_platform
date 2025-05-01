import logging
import sys
from typing import Literal, Union

import numpy as np
import torch
import torch.nn as nn

from models.norm import get_norm
from models.activation import get_activation
from models.mlp import get_mlp
from models.transformer import Transformer

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        proj_drop_rate=0.0,
        att_drop_rate=0.0,
        drop_path_rate=0.1,
        init_std=0.02,
        fixed_dropout_depth=False,
        norm_layer: Union[nn.Module, Literal['RmsNorm', 'LayerNorm', 'SyncBatchNorm', 'GroupNorm']] = 'RmsNorm',
        act_layer: Union[nn.Module, Literal['GELU', 'SiLU', 'LeakyReLU', 'GLU', 'Sigmoid', 'Tanh']] = 'SiLU',
        mlp_layer: Union[nn.Module, Literal['Mlp', 'SwiGLU']] = 'SwiGLU',
        activation_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.proj_drop_rate = proj_drop_rate
        self.att_drop_rate = att_drop_rate
        self.drop_path_rate = drop_path_rate
        self.activation_checkpointing = activation_checkpointing

        # stochastic depth decay rule
        if not fixed_dropout_depth and self.drop_path_rate > 0.0:
            dpr = np.linspace(0, self.drop_path_rate, self.depth)

        self.norm_layer = get_norm(norm_layer)
        self.act_layer = get_activation(act_layer)
        self.mlp_layer = get_mlp(mlp_layer)

        self.transformer_blocks = nn.ModuleList([
            Transformer(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=mlp_ratio,
                proj_drop=self.proj_drop_rate,
                att_drop=self.att_drop_rate,
                drop_path=self.drop_path_rate if fixed_dropout_depth and self.drop_path_rate > 0.0 else dpr[i],
                norm_layer=self.norm_layer,
                act_layer=self.act_layer,
                mlp_layer=self.mlp_layer,
                activation_checkpointing=activation_checkpointing
            )
            for i in range(self.depth)
        ])
        self.feature_info = [dict(module=f'transformer_blocks.{i}', num_chs=self.embed_dim) for i in range(self.depth)]
        self.init_std = init_std

    @torch.jit.ignore
    def get_num_layers(self):
        return len(self.transformer_blocks)

    def forward(self, x):
        for i, t in enumerate(self.transformer_blocks):
            x = t(x, return_attention=False)
        return x
