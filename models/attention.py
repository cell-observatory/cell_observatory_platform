import sys
import logging
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cell_observatory_platform.data.masking.mask_generator import apply_masks_rope
from cell_observatory_platform.models.rope import (
    generate_frequency_spectrum,
    generate_grid_indices,
    compute_axial_cis,
    compute_mixed_cis,
    apply_rotary_emb,
)
from cell_observatory_platform.training.helpers import get_patch_sizes

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
        patch_shape: tuple = (4,16,16,16),
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16
    ) -> None:
        super().__init__()

        self.dim = dim
        self.device = device

        temporal_patch_size, axial_patch_size, lateral_patch_size = get_patch_sizes(
            input_format=input_fmt,
            patch_shape=patch_shape
        )

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
                input_fmt=input_fmt,
                dtype=dtype,
            )
                
            self.freqs = nn.Parameter(freqs, requires_grad=True)

            end_x = input_shape[input_fmt.index('X')] // lateral_patch_size
            end_y = input_shape[input_fmt.index('Y')] // lateral_patch_size

            if self.input_fmt == "YXC":
                _, _, t_y, t_x = generate_grid_indices(end_x=end_x, 
                                                       end_y=end_y, 
                                                       input_fmt=input_fmt,
                                                       device=self.device,
                                                       dtype=dtype)
                self.register_buffer('freqs_t_x', t_x)
                self.register_buffer('freqs_t_y', t_y)

                self.grid_indices = (None, None, t_y, t_x)

            elif self.input_fmt == "ZYXC":
                end_z = input_shape[input_fmt.index('Z')] // axial_patch_size
                _, t_z, t_y, t_x = generate_grid_indices(end_x=end_x, 
                                                         end_y=end_y, 
                                                         end_z=end_z, 
                                                         input_fmt=input_fmt,
                                                         device=self.device,
                                                         dtype=dtype)
                self.register_buffer('freqs_t_x', t_x)
                self.register_buffer('freqs_t_y', t_y)
                self.register_buffer('freqs_t_z', t_z)

                self.grid_indices = (None, t_z, t_y, t_x)

            elif self.input_fmt == "TYXC":
                end_t = input_shape[input_fmt.index('T')] // temporal_patch_size
                t_t, _, t_y, t_x = generate_grid_indices(end_x=end_x, 
                                                         end_y=end_y, 
                                                         end_t=end_t, 
                                                         input_fmt=input_fmt,
                                                         device=self.device,
                                                         dtype=dtype)
                self.register_buffer('freqs_t_x', t_x)
                self.register_buffer('freqs_t_y', t_y)
                self.register_buffer('freqs_t_t', t_t)

                self.grid_indices = (t_t, None, t_y, t_x)

            elif self.input_fmt == "TZYXC":
                end_z = input_shape[input_fmt.index('Z')] // axial_patch_size
                end_t = input_shape[input_fmt.index('T')] // temporal_patch_size
                t_t, t_z, t_y, t_x = generate_grid_indices(end_x=end_x, 
                                                           end_y=end_y, 
                                                           end_z=end_z, 
                                                           end_t=end_t, 
                                                           input_fmt=input_fmt,
                                                           device=self.device,
                                                           dtype=dtype)
                self.register_buffer('freqs_t_x', t_x)
                self.register_buffer('freqs_t_y', t_y)
                self.register_buffer('freqs_t_z', t_z)
                self.register_buffer('freqs_t_t', t_t)

                self.grid_indices = (t_t, t_z, t_y, t_x)

            else:
                raise NotImplementedError(f"Unknown input_fmt={input_fmt}")

        else:
            end_x = input_shape[input_fmt.index('X')] // lateral_patch_size
            end_y = input_shape[input_fmt.index('Y')] // lateral_patch_size

            if 'Z' in input_fmt:
                end_z = input_shape[input_fmt.index('Z')] // axial_patch_size
            else:
                end_z = None

            if 'T' in input_fmt:
                end_t = input_shape[input_fmt.index('T')] // temporal_patch_size
            else:
                end_t = None

            self.freqs_cis = compute_axial_cis(theta=rope_theta, 
                                               dim=self.dim // self.num_heads,
                                               end_x=end_x,
                                               end_y=end_y, 
                                               end_z=end_z, 
                                               end_t=end_t, 
                                               input_fmt=input_fmt,
                                               device=self.device)

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
            freqs_cis = self.compute_cis(freqs=self.freqs.to(x.device), 
                                         num_heads=self.num_heads, 
                                         t_t=t_t, 
                                         t_z=t_z, 
                                         t_y=t_y, 
                                         t_x=t_x,
                                         input_fmt=self.input_fmt)
        
        else:
            # axial RoPE does not use learnable frequencies
            if masks is not None:
                freqs_cis = apply_masks_rope(self.freqs_cis.to(x.dtype), masks, type="axial")
            else:
                freqs_cis = self.freqs_cis.to(x.dtype)
        
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