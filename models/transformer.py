import logging
import sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import SwiGLU, DropPath
from torch.utils.checkpoint import checkpoint

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        att_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-5),
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.att_drop = nn.Dropout(att_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.att_drop.p if self.training else 0.,
                )

        except NotImplementedError:
            att = q @ k.transpose(-2, -1)
            att = att.softmax(dim=-1)
            att = self.att_drop(att)
            x = att @ v

        if return_attention:
            return att
        else:
            x = x.transpose(1, 2).reshape(B, L, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x


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
        activation_checkpointing: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
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
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=proj_drop,
            act_layer=act_layer,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.activation_checkpointing = activation_checkpointing

    def forward(self, x, return_attention=False):

        ln1 = self.norm1(x)

        if return_attention:
            return self.att(ln1, return_attention=True)
        else:
            att = self.att(ln1, return_attention=False)
            p1 = x + self.drop_path1(att)

            ffn = self.norm2(p1)
            if self.activation_checkpointing:
                ffn = checkpoint(self.mlp, ffn, use_reentrant=False)
            else:
                ffn = self.mlp(ffn)

            p2 = p1 + self.drop_path2(ffn)
            return p2
