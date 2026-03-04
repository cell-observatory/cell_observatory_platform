import sys
import logging
from functools import partial
from typing import Optional, List, Literal

import torch
import torch.nn as nn
from torch import Tensor

from timm.layers import DropPath, SwiGLU

from cell_observatory_platform.models.layers.layer_scale import LayerScale
from cell_observatory_platform.models.layers.mlp import SwiGLUFFN_ListFwdMixin
from cell_observatory_platform.models.layers.attention import (
    Attention, RopeAttention, DeformableAttention, DeformableRopeAttention,
)
from cell_observatory_platform.models.layers.utils import cat_keep_shapes, uncat_with_shapes
from cell_observatory_platform.models.layers.positional_encoding import _maybe_index_rope

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        att_drop: float = 0.0,
        drop_path: float = 0.0,
        sample_drop_ratio: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-5),
        act_layer: nn.Module = nn.SiLU,
        mlp_layer: nn.Module = SwiGLU,
        rope_pos_enc: bool = True,
        rope_random_rotation_per_head: bool = True,
        rope_type: Literal["mixed", "axial", "custom"] = "axial",
        rope_theta: float = 10.0,
        input_fmt: str = "TZYXC",
        input_shape: tuple = (16, 128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        wide_silu: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        layer_scale_init_values: float | None = None,
        ffn_mask_k_bias: bool = False,
        # DA params
        use_deformable_attn: bool = False,
        da_n_points: int = 4,
        da_n_levels: int = 1,
    ) -> None:
        super().__init__()

        if use_deformable_attn:
            if rope_pos_enc:
                self.att = DeformableRopeAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                    att_drop=att_drop, proj_drop=proj_drop, proj_bias=proj_bias,
                    ffn_mask_k_bias=ffn_mask_k_bias, norm_layer=norm_layer,
                    random_rotation_per_head=rope_random_rotation_per_head,
                    rope_type=rope_type, rope_theta=rope_theta,
                    input_fmt=input_fmt, input_shape=input_shape,
                    patch_shape=patch_shape, dtype=dtype,
                    n_points=da_n_points, n_levels=da_n_levels,
                )
            else:
                self.att = DeformableAttention(
                    dim, num_heads=num_heads,
                    n_points=da_n_points, n_levels=da_n_levels,
                )
        else:
            if rope_pos_enc:
                self.att = RopeAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                    att_drop=att_drop, proj_drop=proj_drop, proj_bias=proj_bias,
                    ffn_mask_k_bias=ffn_mask_k_bias, norm_layer=norm_layer,
                    random_rotation_per_head=rope_random_rotation_per_head,
                    rope_type=rope_type, rope_theta=rope_theta,
                    input_fmt=input_fmt, input_shape=input_shape,
                    patch_shape=patch_shape, dtype=dtype,
                )
            else:
                self.att = Attention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
                    att_drop=att_drop, proj_drop=proj_drop, norm_layer=norm_layer,
                )

        self.with_deformable_attn = use_deformable_attn

        if layer_scale_init_values is not None:
            self.ls1 = LayerScale(dim, init_values=layer_scale_init_values)
        else:
            self.ls1 = nn.Identity()

        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        # from:
        # https://github.com/facebookresearch/vjepa2/blob/main/src/models/utils/modules.py
        if mlp_layer == SwiGLU or mlp_layer == SwiGLUFFN_ListFwdMixin:
            hidden_features = int(dim * mlp_ratio)
            if wide_silu:
                swiglu_hidden_features = int(2 * hidden_features / 3)
                align_as = 8
                swiglu_hidden_features = (swiglu_hidden_features + align_as - 1) // align_as * align_as
                hidden_features = swiglu_hidden_features

        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio) if not wide_silu else hidden_features,
            drop=proj_drop,
            act_layer=act_layer,
            bias=ffn_bias
        )

        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if layer_scale_init_values is not None:
            self.ls2 = LayerScale(dim, init_values=layer_scale_init_values)
        else:
            self.ls2 = nn.Identity()

        self.sample_drop_ratio = sample_drop_ratio

    def init_model_weights(self, buffer_device: str | None = None):
        # TODO: move model inits back into each model class
        # FIXME: add proper weight init logic for MaskDINO
        # init_weights(self, weight_init_type=self.weight_init_type)
        for mod in self.modules():
            if isinstance(mod, RopeAttention):
                mod.init_rope_parameters(device=buffer_device)

    def _forward(self, x, masks=None, pos_enc=None, spatial_kwargs=None):
        ln1 = self.norm1(x)
        att = self.att(ln1, masks=masks, pos_enc=pos_enc, spatial_kwargs=spatial_kwargs)
        p1 = x + self.ls1(self.drop_path1(att))

        ffn = self.norm2(p1)
        ffn = self.mlp(ffn)

        p2 = p1 + self.ls2(self.drop_path2(ffn))
        return p2

    def _forward_list(self, x_list: List[Tensor], masks=None, pos_enc=None, spatial_kwargs=None) -> List[Tensor]:
        if masks is None:
            masks = [None] * len(x_list)
        if pos_enc is None:
            pos_enc = [None] * len(x_list)
        assert len(x_list) == len(masks) == len(pos_enc), "x_list, masks, and pos_enc must have the same length"

        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if pos_enc is not None:
                pos_enc_subset_list = [
                    _maybe_index_rope(pos_enc_i, indices_1) for pos_enc_i, indices_1 in zip(pos_enc, indices_1_list)
                ]
            else:
                pos_enc_subset_list = pos_enc

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.att.forward_list(
                norm1, masks=masks, pos_enc=pos_enc_subset_list, spatial_kwargs=spatial_kwargs,
            )

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]

        else:
            if self.with_deformable_attn:
                # DA path: single forward_list call processes all levels together
                norm1_list = [self.norm1(x) for x in x_list]
                att_list = self.att.forward_list(
                    norm1_list, masks=masks, pos_enc=pos_enc, spatial_kwargs=spatial_kwargs,
                )
                x_ffn = []
                for x, att_out in zip(x_list, att_list):
                    x_attn = x + self.ls1(self.drop_path1(att_out))
                    x_ffn_i = x_attn + self.ls2(self.drop_path2(self.mlp(self.norm2(x_attn))))
                    x_ffn.append(x_ffn_i)
            else:
                # Standard self-attention: per-element loop
                x_ffn = []
                for x, pe_i, mask_i in zip(x_list, pos_enc, masks):
                    x_attn = x + self.ls1(self.att(self.norm1(x), masks=mask_i, pos_enc=pe_i, spatial_kwargs=spatial_kwargs))
                    x_ffn_i = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                    x_ffn.append(x_ffn_i)

        return x_ffn

    def forward(self, x, masks=None, pos_enc=None, spatial_kwargs=None):
        if isinstance(x, Tensor):
            return self._forward(x, masks=masks, pos_enc=pos_enc, spatial_kwargs=spatial_kwargs)
        elif isinstance(x, list):
            return self._forward_list(x, masks=masks, pos_enc=pos_enc, spatial_kwargs=spatial_kwargs)
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")