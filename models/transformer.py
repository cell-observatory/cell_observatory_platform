import sys
import logging
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from timm.layers import SwiGLU, DropPath

from cell_observatory_platform.models.attention import Attention, RopeAttention

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.,
        att_drop: float = 0.,
        drop_path: float = 0.,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-5),
        act_layer: nn.Module = nn.SiLU,
        mlp_layer: nn.Module = SwiGLU,
        rope_pos_enc: bool = True,
        rope_random_rotation_per_head: bool = True,
        rope_mixed: bool = True,
        rope_theta: float = 10.0,
        input_fmt: str = "TZYXC",
        input_shape: tuple = (16, 128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        wide_silu: bool = False,
        dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        if rope_pos_enc:
            self.att = RopeAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                att_drop=att_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                random_rotation_per_head=rope_random_rotation_per_head,
                rope_mixed=rope_mixed,
                rope_theta=rope_theta,
                input_fmt=input_fmt,
                input_shape=input_shape,
                patch_shape=patch_shape,
                dtype=dtype
            )
        
        else:
            self.att = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                att_drop=att_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer
            )
            
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        # from: 
        # https://github.com/facebookresearch/vjepa2/blob/main/src/models/utils/modules.py
        if mlp_layer == SwiGLU:
            hidden_features = int(dim * mlp_ratio)
            if wide_silu:
                swiglu_hidden_features = int(2 * hidden_features / 3)
                align_as = 8
                swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
                hidden_features = swiglu_hidden_features

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio) if not wide_silu \
                else hidden_features,
            drop=proj_drop,
            act_layer=act_layer,
        )
        
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, masks=None, return_attention=False):
        ln1 = self.norm1(x)

        if return_attention:
            return self.att(ln1, masks=masks, return_attention=True)
        else:
            att = self.att(ln1, masks=masks, return_attention=False)
            p1 = x + self.drop_path1(att)

            ffn = self.norm2(p1)
            ffn = self.mlp(ffn)

            p2 = p1 + self.drop_path2(ffn)
            return p2