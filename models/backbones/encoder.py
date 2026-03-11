import logging
import sys
from typing import Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from cell_observatory_platform.models.layers.activation import get_activation
from cell_observatory_platform.models.layers.mlp import get_mlp
from cell_observatory_platform.models.layers.norm import get_norm
from cell_observatory_platform.models.layers.transformer import Transformer

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
        norm_layer: Union[nn.Module, Literal["RmsNorm", "LayerNorm", "SyncBatchNorm", "GroupNorm"]] = "RmsNorm",
        act_layer: Union[nn.Module, Literal["GELU", "SiLU", "LeakyReLU", "GLU", "Sigmoid", "Tanh"]] = "SiLU",
        mlp_layer: Union[nn.Module, Literal["Mlp", "SwiGLU"]] = "SwiGLU",
        rope_pos_enc: bool = True,
        rope_random_rotation_per_head: bool = True,
        rope_type: Literal["mixed", "axial", "custom"] = "axial",
        rope_theta: float = 10.0,
        input_fmt: str = "TZYXC",
        input_shape: tuple = (16, 128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        wide_silu: bool = False,
        out_layers: list = None,
        dtype: torch.dtype = torch.bfloat16,
        use_deformable_attn: bool = False,
        da_n_points: int = 4,
        da_n_levels: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.patch_shape = patch_shape
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.proj_drop_rate = proj_drop_rate
        self.att_drop_rate = att_drop_rate
        self.drop_path_rate = drop_path_rate

        if not fixed_dropout_depth:
            dpr = np.linspace(0, self.drop_path_rate, self.depth)

        self.norm_layer = get_norm(norm_layer)
        self.act_layer = get_activation(act_layer)
        self.mlp_layer = get_mlp(mlp_layer)

        self.transformer_blocks = nn.ModuleList(
            [
                Transformer(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    proj_drop=self.proj_drop_rate,
                    att_drop=self.att_drop_rate,
                    drop_path=self.drop_path_rate if fixed_dropout_depth else dpr[i],
                    norm_layer=self.norm_layer,
                    act_layer=self.act_layer,
                    mlp_layer=self.mlp_layer,
                    rope_pos_enc=rope_pos_enc,
                    rope_random_rotation_per_head=rope_random_rotation_per_head,
                    rope_type=rope_type,
                    rope_theta=rope_theta,
                    input_fmt=input_fmt,
                    input_shape=input_shape,
                    patch_shape=self.patch_shape,
                    wide_silu=wide_silu,
                    dtype=dtype,
                    use_deformable_attn=use_deformable_attn,
                    da_n_points=da_n_points,
                    da_n_levels=da_n_levels,
                )
                for i in range(self.depth)
            ]
        )
        self.feature_info = [
            dict(module=f"transformer_blocks.{i}", num_chs=self.embed_dim)
            for i in range(self.depth)
        ]
        self.init_std = init_std
        self.out_layers = out_layers

    @torch.jit.ignore
    def get_num_layers(self):
        return len(self.transformer_blocks)

    @torch.jit.ignore
    def get_num_heads(self):
        return self.num_heads

    @torch.jit.ignore
    def get_head_dims(self):
        return self.embed_dim // self.num_heads

    def forward(self, x, masks=None, pos_enc=None, spatial_kwargs=None):
        outs = []
        for i, t in enumerate(self.transformer_blocks):
            x = t(x, masks=masks, pos_enc=pos_enc, spatial_kwargs=spatial_kwargs)
            if self.out_layers is not None and i in self.out_layers:
                outs.append(x)

        if self.out_layers is not None:
            return outs

        return x
