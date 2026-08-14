"""
Adapted from:
https://github.com/facebookresearch/hiera/blob/main/hiera/hiera.py
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp

from timm.models.layers import DropPath
from cell_observatory_platform.models.layers.norm import get_norm
from cell_observatory_platform.models.layers.attention import MaskUnitAttention
from cell_observatory_platform.models.layers.patch_embeddings import PatchEmbedding, calc_num_patches
from cell_observatory_platform.models.layers.utils import Unroll, Reroll, do_pool_stride


class HieraBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.norm1 = norm_layer(dim)
        self.attn = MaskUnitAttention(
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn
        )

        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(dim_out * mlp_ratio), act_layer=act_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        if self.dim != self.dim_out:
            x = do_pool_stride(self.proj(x_norm), stride=self.attn.q_stride)
        x = x + self.drop_path(self.attn(x_norm))

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    """
    Hiera MAE-style backbone with Unroll/Reroll and mask support.
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        input_fmt: str,
        input_shape: tuple,
        patch_shape: tuple,
        embed_dim: int = 96,
        num_heads: int = 1,
        drop_path_rate: float = 0.0,
        q_pool: int = 3, # number of stages to pool q
        # int default: expanded to the token-grid rank below. A fixed-rank tuple
        # default (the old (2, 2)) silently zip-truncated against 3-D/4-D grids.
        q_stride: Union[Tuple[int, ...], int] = 2,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        mask_unit_size: Optional[Tuple[int, ...]] = None,
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        use_mask_unit_attn: Optional[List[bool]] = None,
        norm_layer: str = "LayerNorm",
        act_layer: nn.Module = nn.GELU
    ):
        super().__init__()

        self.input_fmt = input_fmt
        self.input_shape = input_shape
        self.patch_shape = patch_shape

        axis_to_value = dict(zip(input_fmt, input_shape))
        channels = axis_to_value.get("C", 1)

        if isinstance(q_stride, int):
            q_stride_tuple = (q_stride,) * self._num_spatial_dims()
        else:
            q_stride_tuple = tuple(q_stride)
        self.q_stride = q_stride_tuple
        q_stride_flat = int(math.prod(q_stride_tuple))

        num_patches, token_shape = calc_num_patches(
            input_fmt=input_fmt,
            input_shape=input_shape,
            patch_shape=patch_shape,
        )
        assert self.input_fmt in ["TZYXC", "ZYXC"], "Hiera only supports TZYXC, and ZYXC input formats"
        self.tokens_spatial_shape = [
            x for x in token_shape[:-1] # ignore channel dimension
            if x is not None and isinstance(x, int)
        ]
        D = len(self.tokens_spatial_shape)

        # Rank guards: every later zip(...) against tokens_spatial_shape
        # (mask-unit divisibility, Unroll views, Reroll schedules) silently
        # TRUNCATES to the shorter rank -- a rank-2 stride against a rank-3
        # grid mis-factors the mask-unit layout instead of failing.
        if len(self.q_stride) != D:
            raise ValueError(
                f"q_stride {self.q_stride} has rank {len(self.q_stride)} but the "
                f"token grid {self.tokens_spatial_shape} has rank {D}; pass an "
                f"int or a rank-{D} tuple."
            )
        if mask_unit_size is not None and len(tuple(mask_unit_size)) != D:
            raise ValueError(
                f"mask_unit_size {tuple(mask_unit_size)} has rank "
                f"{len(tuple(mask_unit_size))} but the token grid "
                f"{self.tokens_spatial_shape} has rank {D}."
            )

        depth = sum(stages)
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool = q_pool

        unroll_schedule_len = len(self.stage_ends[:-1])
        self.unroll_schedule = [q_stride_tuple] * unroll_schedule_len

        if mask_unit_size is None:
            self.mask_unit_size = tuple(
                s ** unroll_schedule_len for s in q_stride_tuple
            )
        else:
            self.mask_unit_size = tuple(mask_unit_size)
        for ts, mu in zip(self.tokens_spatial_shape, self.mask_unit_size):
            assert ts % mu == 0, (
                f"mask_unit_size {self.mask_unit_size} must divide "
                f"tokens_spatial_shape {self.tokens_spatial_shape}; {ts} % {mu} != 0"
            )
        self.flat_mu_size = int(math.prod(self.mask_unit_size))

        patch_stride_for_unroll = (1,) * D
        self.unroll = Unroll(
            input_size=tuple(self.tokens_spatial_shape),
            patch_stride=patch_stride_for_unroll,
            unroll_schedule=self.unroll_schedule,
        )
        if use_mask_unit_attn is not None:
            assert len(use_mask_unit_attn) == depth, (
                f"use_mask_unit_attn must have length {depth} (num blocks), got {len(use_mask_unit_attn)}"
            )
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]

        self.reroll = Reroll(
            input_size=tuple(self.tokens_spatial_shape),
            patch_stride=patch_stride_for_unroll,
            unroll_schedule=self.unroll_schedule,
            stage_ends=self.stage_ends,
            q_pool=q_pool,
        )

        norm_layer = get_norm(norm_layer)

        self.patch_embed = PatchEmbedding(
            input_fmt=input_fmt,
            input_shape=input_shape,
            patch_shape=patch_shape,
            embed_dim=embed_dim,
            channels=channels,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        cur_flat_mu = self.flat_mu_size
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                if i in self.q_pool_blocks:
                    cur_flat_mu //= q_stride_flat

            use_mask_unit_attn_block = (
                use_mask_unit_attn[i] if use_mask_unit_attn is not None else cur_flat_mu > 1
            )
            window_size = cur_flat_mu if use_mask_unit_attn_block else 0
            q_stride_block = q_stride_flat if i in self.q_pool_blocks else 1

            block = HieraBlock(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                mlp_ratio=4.0,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                q_stride=q_stride_block,
                window_size=window_size,
                use_mask_unit_attn=use_mask_unit_attn_block,
            )
            embed_dim = dim_out
            self.blocks.append(block)

    def _num_spatial_dims(self) -> int:
        if self.input_fmt == "TZYXC":
            return 4
        elif self.input_fmt == "ZYXC":
            return 3
        elif self.input_fmt == "TYXC":
            return 3
        elif self.input_fmt == "YXC":
            return 2
        else:
            return 2

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        ctx_idx: torch.Tensor = None,
        return_intermediates: bool = False,
        return_windowed_intermediates: Optional[bool] = None,
    ):
        x, patches = self.patch_embed(x, return_patches=True)
        B, N, C = x.shape

        # TODO: add more positional encoding options for hiera
        x = x + self.pos_embed

        x = self.unroll(x)

        if mask is not None:
            if ctx_idx is None:
                raise ValueError("ctx_idx [B, K] is required when mask is provided (HIERA_MU). Precompute in mask generator.")
            num_mus = N // self.flat_mu_size
            # Unroll layout: token n = intra_mu_offset * num_mus + mu_index, i.e.
            # the mask-unit index is the FASTEST axis (verified by the layout
            # test in tests/models/test_fix_regressions_models.py). View
            # accordingly and gather kept mask units on the MU dim; the result
            # keeps the same convention with num_mus -> K (kept units).
            x = x.view(B, self.flat_mu_size, num_mus, C)
            gather_idx = ctx_idx[:, None, :, None].expand(-1, self.flat_mu_size, -1, C)
            x = x.gather(dim=2, index=gather_idx).reshape(B, self.flat_mu_size * ctx_idx.shape[1], C)

        intermediates = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if return_intermediates and i in self.stage_ends:
                # Default: masked -> windowed, unmasked -> spatial. Override via return_windowed_intermediates.
                if return_windowed_intermediates is None:
                    want_windowed = mask is not None
                else:
                    want_windowed = bool(return_windowed_intermediates)
                rerolled = self.reroll(x, i, mask=mask, return_windowed=want_windowed)
                intermediates.append(rerolled)

        if return_intermediates:
            return x, intermediates, patches
        return x, patches
