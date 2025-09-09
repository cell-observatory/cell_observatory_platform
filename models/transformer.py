import sys
import math
import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import SwiGLU, DropPath
from torch.nn.attention import SDPBackend, sdpa_kernel

from data.masking.mask_generator import apply_masks_rope
from models.positional_encoding import (
    generate_frequency_spectrum,
    generate_grid_indices,
    compute_axial_cis,
    compute_mixed_cis,
    apply_rotary_emb,
)

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

        if qk_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm is True'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.att_drop = nn.Dropout(att_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, masks=None, return_attention=False):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        try:
            # priority: flash > efficient > math
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.att_drop.p if self.training else 0.,
                )

        except NotImplementedError:
            q = q * self.scale
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


class RopeAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_bias: bool = True,
        att_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-5),
        random_rotation_per_head: bool = True,
        rope_mixed: bool = True,
        rope_theta: float = 10.0,
        input_fmt: str = "TZXYC",
        input_shape: tuple = (16,128,128,128,2),
        patch_size: tuple = (4,16,16,16)
    ) -> None:
        super().__init__()

        self.dim = dim

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        if qk_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm is True'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.att_drop = nn.Dropout(att_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # RoPE parameters
        self.input_fmt = input_fmt
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        self.random_rotation_per_head = random_rotation_per_head

        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, 
                                       num_heads=self.num_heads, 
                                       input_fmt=input_fmt)
            
            # learnable frequency spectrum but initialized with 
            # standard fixed RoPE frequencies
            freqs = generate_frequency_spectrum(
                dim=self.dim // self.num_heads, 
                num_heads=self.num_heads, 
                theta=rope_theta,
                random_rotation_per_head=self.random_rotation_per_head,
                input_fmt=input_fmt
            )
                
            self.freqs = nn.Parameter(freqs, requires_grad=True)

            # NOTE: works as long as we follow channel-last convention
            #       otherwise we need a more robust solution
            assert input_fmt[-1] == 'C', "input_fmt must follow channel-last convention"
            end_x = input_shape[1:][input_fmt.index('X')] // patch_size[input_fmt.index('X')]
            end_y = input_shape[1:][input_fmt.index('Y')] // patch_size[input_fmt.index('Y')]

            if self.input_fmt == "YXC":
                _, _, t_y, t_x = generate_grid_indices(end_x=end_x, 
                                                       end_y=end_y, 
                                                       input_fmt=input_fmt)
                self.register_buffer('freqs_t_x', t_x)
                self.register_buffer('freqs_t_y', t_y)

                self.grid_indices = (None, None, t_y, t_x)

            elif self.input_fmt == "ZYXC":
                end_z = input_shape[1:][input_fmt.index('Z')] // patch_size[input_fmt.index('Z')]
                _, t_z, t_y, t_x = generate_grid_indices(end_x=end_x, 
                                                         end_y=end_y, 
                                                         end_z=end_z, 
                                                         input_fmt=input_fmt)
                self.register_buffer('freqs_t_x', t_x)
                self.register_buffer('freqs_t_y', t_y)
                self.register_buffer('freqs_t_z', t_z)

                self.grid_indices = (None, t_z, t_y, None)

            elif self.input_fmt == "TYXC":
                end_t = input_shape[1:][input_fmt.index('T')] // patch_size[input_fmt.index('T')]
                t_t, _, t_y, t_x = generate_grid_indices(end_x=end_x, 
                                                         end_y=end_y, 
                                                         end_t=end_t, 
                                                         input_fmt=input_fmt)
                self.register_buffer('freqs_t_x', t_x)
                self.register_buffer('freqs_t_y', t_y)
                self.register_buffer('freqs_t_t', t_t)

                self.grid_indices = (t_t, None, t_y, t_x)

            elif self.input_fmt == "TZYXC":
                end_z = input_shape[1:][input_fmt.index('Z')] // patch_size[input_fmt.index('Z')]
                end_t = input_shape[1:][input_fmt.index('T')] // patch_size[input_fmt.index('T')]
                t_t, t_z, t_y, t_x = generate_grid_indices(end_x=end_x, 
                                                           end_y=end_y, 
                                                           end_z=end_z, 
                                                           end_t=end_t, 
                                                           input_fmt=input_fmt)
                self.register_buffer('freqs_t_x', t_x)
                self.register_buffer('freqs_t_y', t_y)
                self.register_buffer('freqs_t_z', t_z)
                self.register_buffer('freqs_t_t', t_t)

                self.grid_indices = (t_t, t_z, t_x, t_y)

            else:
                raise NotImplementedError(f"Unknown input_fmt={input_fmt}")

        else:
            end_x = input_shape[1:][input_fmt.index('X')] // patch_size[input_fmt.index('X')]
            end_y = input_shape[1:][input_fmt.index('Y')] // patch_size[input_fmt.index('Y')]

            if 'Z' in input_fmt:
                end_z = input_shape[1:][input_fmt.index('Z')] // patch_size[input_fmt.index('Z')]
            else:
                end_z = None

            if 'T' in input_fmt:
                end_t = input_shape[1:][input_fmt.index('T')] // patch_size[input_fmt.index('T')]
            else:
                end_t = None

            self.freqs_cis = compute_axial_cis(theta=rope_theta, 
                                               dim=self.dim // self.num_heads,
                                               end_x=end_x,
                                               end_y=end_y, 
                                               end_z=end_z, 
                                               end_t=end_t, 
                                               input_fmt=input_fmt)

    def forward(self, x, masks=None, return_attention=False):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # apply rotary position embedding
        if self.rope_mixed:
            if masks is not None:
                t_t, t_z, t_y, t_x = apply_masks_rope(self.grid_indices, masks, type="mixed")
            else:
                t_t, t_z, t_y, t_x = self.grid_indices

            # compute learnable frequencies
            # works no matter what input_fmt is since unused t_* are None
            freqs_cis = self.compute_cis(freqs=self.freqs, 
                                         num_heads=self.num_heads, 
                                         t_t=t_t, 
                                         t_z=t_z, 
                                         t_y=t_y, 
                                         t_x=t_x,
                                         input_fmt=self.input_fmt)
        
        else:
            # axial RoPE does not use learnable frequencies
            if masks is not None:
                freqs_cis = apply_masks_rope(self.freqs_cis, masks, type="axial")
            else:
                freqs_cis = self.freqs_cis
        
        q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        try:
            # priority: flash > efficient > math
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                x = F.scaled_dot_product_attention(
                    q_rope, k_rope, v,
                    dropout_p=self.att_drop.p if self.training else 0.,
                )

        except NotImplementedError:
            q_rope = q_rope * self.scale
            att = q_rope @ k_rope.transpose(-2, -1)
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
        rope_pos_enc: bool = True,
        rope_random_rotation_per_head: bool = True,
        rope_mixed: bool = True,
        rope_theta: float = 10.0,
        input_fmt: str = "TZYXC",
        input_shape: tuple = (16,128,128,128,2),
        patch_size: tuple = (4,16,16,16),
        wide_silu: bool = False,
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
                patch_size=patch_size
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