import sys
import math
import logging
import warnings

from functools import partial
from typing import Optional, Literal, List, Tuple, Union, Type

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.init import constant_, xavier_uniform_

from cell_observatory_platform.data.masking.mask_generator import apply_masks_rope
from cell_observatory_platform.models.layers.positional_encoding import (
    apply_rope,
    compute_axial_cis,
    compute_mixed_cis,
    generate_frequency_spectrum,
    generate_grid_indices,
    make_axial_rope_freqs,
    make_cross_rope_pos_enc_qk,
)
from cell_observatory_platform.models.layers.mlp import MLP
from cell_observatory_platform.training.helpers import get_patch_sizes
from cell_observatory_platform.models.layers.activation import get_activation
from cell_observatory_platform.models.layers.utils import cat_keep_shapes, uncat_with_shapes
from cell_observatory_platform.models.ops.flash_deform_attn import FlashDeformAttnFunction, _is_power_of_2

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        att_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-5),
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        if qk_norm:
            assert norm_layer is not None, "norm_layer must be provided if qk_norm is True"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.att_drop = nn.Dropout(att_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, masks=None, pos_enc=None):
        B, L, C = x.shape
        # NOTE: Use -1 instead of `n_heads` to infer the actual
        #       local heads from sizes of xq, xk, and xv as TP
        #       may have sharded them after the above linear ops.
        q = self.wq(x).view(B, L, -1, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, -1, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, -1, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Removed: SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.att_drop.p if self.training else 0.0,
            )

        # REMOVED: standard attention fallback
        # except NotImplementedError:
        #     q = q * self.scale
        #     att = q @ k.transpose(-2, -1)
        #     att = att.softmax(dim=-1)
        #     att = self.att_drop(att)
        #     x = att @ v

        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# reference: https://github.com/facebookresearch/dinov3/dinov3/layers/attention.py
class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class LinearMaskedBias(nn.Linear):
    """
    Like nn.Linear, but multiplies bias by a fixed mask before adding.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            self.register_buffer(
                "bias_mask",
                torch.full_like(self.bias, fill_value=math.nan),
            )

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is None:
            return F.linear(input, self.weight, None)
        return F.linear(input, self.weight, self.bias * self.bias_mask.to(self.bias.dtype))


class RopeAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_bias: bool = True,
        att_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-5),
        random_rotation_per_head: bool = True,
        rope_type: Literal["mixed", "axial", "custom"] = "axial",
        rope_theta: float = 10.0,
        input_fmt: str = "TZYXC",
        input_shape: tuple = (16, 128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        ffn_mask_k_bias: bool = False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.dtype = dtype
        self.device = device

        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        if qk_norm:
            assert norm_layer is not None, "norm_layer must be provided if qk_norm is True"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.input_fmt = input_fmt
        self.input_shape = input_shape
        self.patch_shape = patch_shape

        if ffn_mask_k_bias:
            self.wq = nn.Linear(dim, dim, bias=qkv_bias)
            self.wk = LinearMaskedBias(dim, dim, bias=qkv_bias)
            self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.wq = nn.Linear(dim, dim, bias=qkv_bias)
            self.wk = nn.Linear(dim, dim, bias=qkv_bias)
            self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.att_drop = nn.Dropout(att_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope_type = rope_type
        assert self.rope_type in ["mixed", "axial", "custom"], "Invalid rope_type"
        self.rope_theta = rope_theta
        self.random_rotation_per_head = random_rotation_per_head

        self._rope_inited = False
        # placeholders; real init will happen later
        if self.rope_type == "mixed":
            freq_shape = self._get_freqs_shape()
            freqs_meta = torch.empty(freq_shape, dtype=dtype)
            self.freqs = nn.Parameter(freqs_meta, requires_grad=True)
            self.grid_indices = (None, None, None, None)

    def _get_freqs_shape(self) -> torch.Size:
        dim_per_head = self.dim // self.num_heads
        freqs = generate_frequency_spectrum(
            dim=dim_per_head,
            num_heads=self.num_heads,
            theta=self.rope_theta,
            random_rotation_per_head=self.random_rotation_per_head,
            input_fmt=self.input_fmt,
            dtype=self.dtype,
            device="cpu",
        )
        return freqs.shape

    def init_rope_parameters(self, device: torch.device, dtype: Optional[torch.dtype] = None):
        if self._rope_inited:
            return

        if dtype is None:
            dtype = self.dtype

        temporal_patch_size, axial_patch_size, lateral_patch_size = get_patch_sizes(
            input_format=self.input_fmt, patch_shape=self.patch_shape
        )

        if self.rope_type == "mixed":
            self.compute_cis = partial(compute_mixed_cis, input_fmt=self.input_fmt)

            # learnable frequency spectrum but initialized with
            # standard fixed RoPE frequencies
            freqs = generate_frequency_spectrum(
                dim=self.dim // self.num_heads,
                num_heads=self.num_heads,
                theta=self.rope_theta,
                random_rotation_per_head=self.random_rotation_per_head,
                input_fmt=self.input_fmt,
                dtype=dtype,
                device=device,
            )

            assert freqs.shape == self.freqs.shape, (
                f"freqs shape mismatch: param={self.freqs.shape}, " f"generated={freqs.shape}"
            )

            with torch.no_grad():
                self.freqs.data.copy_(freqs)

            end_x = self.input_shape[self.input_fmt.index("X")] // lateral_patch_size
            end_y = self.input_shape[self.input_fmt.index("Y")] // lateral_patch_size

            if self.input_fmt == "YXC":
                _, _, t_y, t_x = generate_grid_indices(
                    end_x=end_x, end_y=end_y, input_fmt=self.input_fmt, device=device, dtype=dtype
                )
                self.register_buffer("freqs_t_x", t_x)
                self.register_buffer("freqs_t_y", t_y)

                self.grid_indices = (None, None, t_y, t_x)

            elif self.input_fmt == "ZYXC":
                end_z = self.input_shape[self.input_fmt.index("Z")] // axial_patch_size
                _, t_z, t_y, t_x = generate_grid_indices(
                    end_x=end_x, end_y=end_y, end_z=end_z, input_fmt=self.input_fmt, device=device, dtype=dtype
                )
                self.register_buffer("freqs_t_x", t_x)
                self.register_buffer("freqs_t_y", t_y)
                self.register_buffer("freqs_t_z", t_z)

                self.grid_indices = (None, t_z, t_y, t_x)

            elif self.input_fmt == "TYXC":
                end_t = self.input_shape[self.input_fmt.index("T")] // temporal_patch_size
                t_t, _, t_y, t_x = generate_grid_indices(
                    end_x=end_x, end_y=end_y, end_t=end_t, input_fmt=self.input_fmt, device=device, dtype=dtype
                )
                self.register_buffer("freqs_t_x", t_x)
                self.register_buffer("freqs_t_y", t_y)
                self.register_buffer("freqs_t_t", t_t)

                self.grid_indices = (t_t, None, t_y, t_x)

            elif self.input_fmt == "TZYXC":
                end_z = self.input_shape[self.input_fmt.index("Z")] // axial_patch_size
                end_t = self.input_shape[self.input_fmt.index("T")] // temporal_patch_size
                t_t, t_z, t_y, t_x = generate_grid_indices(
                    end_x=end_x,
                    end_y=end_y,
                    end_z=end_z,
                    end_t=end_t,
                    input_fmt=self.input_fmt,
                    device=device,
                    dtype=dtype,
                )
                self.register_buffer("freqs_t_x", t_x)
                self.register_buffer("freqs_t_y", t_y)
                self.register_buffer("freqs_t_z", t_z)
                self.register_buffer("freqs_t_t", t_t)

                self.grid_indices = (t_t, t_z, t_y, t_x)

            else:
                raise NotImplementedError(f"Unknown input_fmt={self.input_fmt}")

        self._rope_inited = True

    # FIXME: we do not adequately deal with slicing off register/class tokens in non-custom branches
    def compute_attention(self, q: Tensor, k: Tensor, v: Tensor, masks=None, pos_enc=None) -> Tensor:
        if not self._rope_inited and self.rope_type == "mixed":
            raise RuntimeError("RopeAttention.init_rope_parameters() must be called before forward.")

        # apply rotary position embedding
        if self.rope_type == "mixed":
            if masks is not None:
                t_t, t_z, t_y, t_x = apply_masks_rope(self.grid_indices, masks, type="mixed")
            else:
                t_t, t_z, t_y, t_x = self.grid_indices

            # compute learnable frequencies
            # works no matter what input_fmt is since unused t_* are None
            freqs_cis = self.compute_cis(
                freqs=self.freqs.to(q.device),
                # num_heads=self.num_heads,
                t_t=t_t,
                t_z=t_z,
                t_y=t_y,
                t_x=t_x,
                input_fmt=self.input_fmt,
            )

            q_rope, k_rope = apply_rope(q, k, pos_enc=freqs_cis, rope_type="mixed")

        elif self.rope_type == "axial":
            # axial RoPE: freqs_cis is passed in via pos_enc from the top-level module
            assert pos_enc is not None, (
                "rope_type='axial' requires pos_enc (freqs_cis) to be passed from the top-level module"
            )
            if masks is not None:
                freqs_cis = apply_masks_rope(pos_enc, masks, type="axial")
            else:
                freqs_cis = pos_enc

            q_rope, k_rope = apply_rope(q, k, pos_enc=freqs_cis, rope_type="axial")

        else:
            q_rope, k_rope = apply_rope(q, k, pos_enc=pos_enc, rope_type="custom")

        # priority: flash > efficient > math
        # TODO: consider adding back SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            x = F.scaled_dot_product_attention(
                q_rope,
                k_rope,
                v,
                dropout_p=self.att_drop.p if self.training else 0.0,
            )

        # REMOVED: standard attention fallback
        # except NotImplementedError:
        #     q_rope = q_rope * self.scale
        #     att = q_rope @ k_rope.transpose(-2, -1)
        #     att = att.softmax(dim=-1)
        #     att = self.att_drop(att)
        #     x = att @ v

        return x

    def forward_list(self, x_list, masks=None, pos_enc=None) -> List[Tensor]:
        if masks is None:
            masks = [None] * len(x_list)
        if pos_enc is None:
            pos_enc = [None] * len(x_list)
        assert len(x_list) == len(masks) == len(pos_enc), "x_list, masks, and pos_enc must have the same length"

        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)

        q_flat = self.wq(x_flat)  # (T_total, C)
        k_flat = self.wk(x_flat)
        v_flat = self.wv(x_flat)

        q_list = uncat_with_shapes(q_flat, shapes, num_tokens)
        k_list = uncat_with_shapes(k_flat, shapes, num_tokens)
        v_list = uncat_with_shapes(v_flat, shapes, num_tokens)

        att_out = []
        for q, k, v, pos_enc_i, mask_i in zip(q_list, k_list, v_list, pos_enc, masks):
            B, L, C = q.shape
            q = q.view(B, L, -1, self.head_dim).transpose(1, 2)
            k = k.view(B, L, -1, self.head_dim).transpose(1, 2)
            v = v.view(B, L, -1, self.head_dim).transpose(1, 2)
            q = self.q_norm(q)
            k = self.k_norm(k)
            out = self.compute_attention(q, k, v, masks=mask_i, pos_enc=pos_enc_i)  # (B, H, L, D)
            out = out.transpose(1, 2).reshape(B, L, -1)             # (B, L, C)
            att_out.append(out)
        
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        x_flat = self.proj_drop(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def forward(self, x, masks=None, pos_enc=None):
        B, L, C = x.shape

        # TODO: evaluate impact of using fused qkv linear ops vs not
        # NOTE: Use -1 instead of `n_heads` to infer the actual
        #       local heads from sizes of xq, xk, and xv as TP
        #       may have sharded them after the above linear ops.
        q = self.wq(x).view(B, L, -1, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, -1, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, -1, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        x = self.compute_attention(q, k, v, masks=masks, pos_enc=pos_enc)
        x = x.transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    # def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
    #     assert attn_bias is None
    #     B, N, _ = qkv.shape
    #     C = self.qkv.in_features

    #     qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
    #     q, k, v = torch.unbind(qkv, 2)
    #     q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
    #     if rope is not None:
    #         q, k = self.apply_rope(q, k, rope)
    #     x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    #     x = x.transpose(1, 2)
    #     return x.reshape([B, N, C])

    # def forward(self, x, masks=None, return_attention=False):
    #     if not self._rope_inited:
    #         raise RuntimeError("RopeAttention.init_rope_parameters() must be called before forward.")

    #     B, L, C = x.shape
    #     # NOTE: Use -1 instead of `n_heads` to infer the actual
    #     #       local heads from sizes of xq, xk, and xv as TP
    #     #       may have sharded them after the above linear ops.
    #     q = self.wq(x).view(B, L, -1, self.head_dim).transpose(1, 2)
    #     k = self.wk(x).view(B, L, -1, self.head_dim).transpose(1, 2)
    #     v = self.wv(x).view(B, L, -1, self.head_dim).transpose(1, 2)

    #     q = self.q_norm(q)
    #     k = self.k_norm(k)

    #     # apply rotary position embedding
    #     if self.rope_mixed:
    #         if masks is not None:
    #             t_t, t_z, t_y, t_x = apply_masks_rope(self.grid_indices, masks, type="mixed")
    #         else:
    #             t_t, t_z, t_y, t_x = self.grid_indices

    #         # compute learnable frequencies
    #         # works no matter what input_fmt is since unused t_* are None
    #         freqs_cis = self.compute_cis(
    #             freqs=self.freqs.to(x.device),
    #             # num_heads=self.num_heads,
    #             t_t=t_t,
    #             t_z=t_z,
    #             t_y=t_y,
    #             t_x=t_x,
    #             input_fmt=self.input_fmt,
    #         )

    #     else:
    #         # axial RoPE does not use learnable frequencies
    #         if masks is not None:
    #             freqs_cis = apply_masks_rope(self.freqs_cis, masks, type="axial")
    #         else:
    #             freqs_cis = self.freqs_cis

    #     q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

    #     try:
    #         # priority: flash > efficient > math
    #         with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
    #             x = F.scaled_dot_product_attention(
    #                 q_rope,
    #                 k_rope,
    #                 v,
    #                 dropout_p=self.att_drop.p if self.training else 0.0,
    #             )

    #     except NotImplementedError:
    #         q_rope = q_rope * self.scale
    #         att = q_rope @ k_rope.transpose(-2, -1)
    #         att = att.softmax(dim=-1)
    #         att = self.att_drop(att)
    #         x = att @ v

    #     if return_attention:
    #         return att

    #     else:
    #         x = x.transpose(1, 2).reshape(B, L, C)
    #         x = self.proj(x)
    #         x = self.proj_drop(x)
    #         return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        att_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-5),
        # NOTE: non-standard but for SAM2 they require these fields
        kv_in_dim: Optional[int] = None,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()

        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else dim
        q_dim = dim // downsample_rate

        assert q_dim % num_heads == 0, "dim // downsample_rate should be divisible by num_heads"
        if qk_norm:
            assert norm_layer is not None, "norm_layer must be provided if qk_norm is True"

        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.scale = self.head_dim**-0.5
        self._query_dim = dim

        self.q_proj = nn.Linear(dim, q_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(self.kv_in_dim, q_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.kv_in_dim, q_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(q_dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.att_drop = nn.Dropout(att_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, keys, values):
        B, Nq, _ = query.shape
        Nk = keys.shape[1]
        H = self.num_heads
        Hd = self.head_dim
        q = self.q_proj(query).view(B, Nq, H, Hd).transpose(1, 2)
        k = self.k_proj(keys).view(B, Nk, H, Hd).transpose(1, 2)
        v = self.v_proj(values).view(B, Nk, H, Hd).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # REMOVED: SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.att_drop.p if self.training else 0.0,
            )

        out = out.transpose(1, 2).contiguous().view(B, Nq, -1)
        out = self.o_proj(out)
        out = self.proj_drop(out)
        return out


def _apply_cross_rope(
    q: Tensor,
    k: Tensor,
    pos_enc: Union[Tensor, Tuple[Tensor, Tensor]],
    rope_type: Literal["mixed", "axial", "custom"],
    prefix_q: int = 0,
    suffix_q: int = 0,
    prefix_k: int = 0,
    suffix_k: int = 0,
) -> Tuple[Tensor, Tensor]:
    """
    Slice off prefix/suffix (e.g. register/class) tokens from q and k, apply RoPE
    to the core sequences, then reassemble. Supports separate pos_enc for Q and K
    via pos_enc=(pos_enc_q, pos_enc_k); caller must pass encodings with seq dims
    matching q_core and k_core lengths.
    """
    Nq, Nk = q.shape[2], k.shape[2]

    q_prefix = q[:, :, :prefix_q, :] if prefix_q else None
    q_core = q[:, :, prefix_q : Nq - suffix_q, :]
    q_suffix = q[:, :, -suffix_q:, :] if suffix_q else None

    k_prefix = k[:, :, :prefix_k, :] if prefix_k else None
    k_core = k[:, :, prefix_k : Nk - suffix_k, :]
    k_suffix = k[:, :, -suffix_k:, :] if suffix_k else None

    if isinstance(pos_enc, (tuple, list)) and len(pos_enc) == 2:
        pos_enc_for_apply: Union[Tensor, Tuple[Tensor, Tensor]] = (pos_enc[0], pos_enc[1])
    else:
        pos_enc_for_apply = pos_enc

    q_rope_core, k_rope_core = apply_rope(q_core, k_core, pos_enc_for_apply, rope_type)

    parts_q = [p for p in (q_prefix, q_rope_core, q_suffix) if p is not None]
    parts_k = [p for p in (k_prefix, k_rope_core, k_suffix) if p is not None]
    q_rope = torch.cat(parts_q, dim=2)
    k_rope = torch.cat(parts_k, dim=2)
    return q_rope, k_rope


class RopeCrossAttention(RopeAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_bias: bool = True,
        att_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-5),
        random_rotation_per_head: bool = True,
        rope_type: Literal["mixed", "axial", "custom"] = "axial",
        rope_theta: float = 10.0,
        input_fmt: str = "TZXYC",
        input_shape: tuple = (16, 128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        ffn_mask_k_bias: bool = False,
    ) -> None:
        super().__init__(
            dim, 
            num_heads, 
            qkv_bias, 
            qk_norm, 
            proj_bias, 
            att_drop, 
            proj_drop, 
            norm_layer, 
            random_rotation_per_head, 
            rope_type, 
            rope_theta, 
            input_fmt, 
            input_shape, 
            patch_shape, 
            device, 
            dtype, 
            ffn_mask_k_bias
        )

    def compute_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        masks=None,
        pos_enc=None,
        prefix_q: int = 0,
        suffix_q: int = 0,
        prefix_k: int = 0,
        suffix_k: int = 0,
    ) -> Tensor:
        if not self._rope_inited and self.rope_type == "mixed":
            raise RuntimeError("RopeCrossAttention.init_rope_parameters() must be called before forward.")

        # apply rotary position embedding (slicing and optional separate Q/K pos_enc in helper)
        if self.rope_type == "mixed":
            raise ValueError("RopeCrossAttention does not support rope_type='mixed' for cross-attention.")

        elif self.rope_type == "axial":
            assert pos_enc is not None, (
                "rope_type='axial' requires pos_enc (freqs_cis) to be passed from the top-level module"
            )
            if isinstance(pos_enc, (tuple, list)) and len(pos_enc) == 2:
                pos_enc_q, pos_enc_k = pos_enc
                if masks is not None:
                    freqs_cis = (
                        apply_masks_rope(pos_enc_q, masks, type="axial"),
                        apply_masks_rope(pos_enc_k, masks, type="axial") if pos_enc_k is not None else None,
                    )
                else:
                    freqs_cis = (pos_enc_q, pos_enc_k)
            else:
                if masks is not None:
                    freqs_cis = apply_masks_rope(pos_enc, masks, type="axial")
                else:
                    freqs_cis = pos_enc

            q_rope, k_rope = _apply_cross_rope(
                q, k, freqs_cis, "axial",
                prefix_q=prefix_q, suffix_q=suffix_q, prefix_k=prefix_k, suffix_k=suffix_k,
            )

        else:
            raise ValueError("RopeCrossAttention does not support rope_type='custom' for cross-attention.")

        # REMOVED: SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            x = F.scaled_dot_product_attention(
                q_rope,
                k_rope,
                v,
                dropout_p=self.att_drop.p if self.training else 0.0,
            )

        return x

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        B, N, C = x.shape
        x = x.reshape(B, N, num_heads, C // num_heads)
        # x: [B, N, H, D_head] -> [B, H, N, D_head]
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        B, H, N, D_head = x.shape
        x = x.transpose(1, 2)
        # x: [B, H, N, D_head] -> [B, N, H * D_head]
        return x.reshape(B, N, H * D_head)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        num_k_exclude_rope: int = 0,
        masks=None,
        pos_enc=None,
        prefix_q: int = 0,
        suffix_q: int = 0,
        prefix_k: int = 0,
        suffix_k: int = 0,
    ) -> Tensor:
        q = self._separate_heads(self.wq(q), self.num_heads)
        k = self._separate_heads(self.wk(k), self.num_heads)
        v = self._separate_heads(self.wv(v), self.num_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        x = self.compute_attention(
            q, k, v,
            masks=masks,
            pos_enc=pos_enc,
            prefix_q=prefix_q,
            suffix_q=suffix_q,
            prefix_k=prefix_k,
            suffix_k=suffix_k,
        )
        x = self._recombine_heads(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashDeformAttn3D(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, use_reg=True):
        """
        Multi-Scale Deformable Attention Module

        Args:
            d_model: hidden dimension
            n_levels: number of feature levels
            n_heads: number of attention heads
            n_points: number of sampling points per attention head per feature level
        """
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads, but got {} and {}".format(d_model, n_heads))
        _d_per_head = d_model // n_heads

        if _d_per_head % 8 != 0:
            raise ValueError(
                f"Per-head dim must be multiple of 8 for this kernel, "
                f"got d_model={d_model}, n_heads={n_heads}, per_head={_d_per_head}."
            )

        # set _d_per_head to a power of 2
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "Set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 3)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self.use_reg = use_reg

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)

        # --- --- start of sampling offsets initialization --- ---

        # azimuth
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        phis = torch.tensor([math.pi / 4, -math.pi / 4], dtype=torch.float32)
        # alternate heads: up, down, up, down, ...
        phis = phis.repeat((self.n_heads + 1) // 2)[: self.n_heads]

        # unit vectors on the sphere
        dirs_x = torch.cos(thetas) * torch.cos(phis)  # cosϕ cosθ
        dirs_y = torch.sin(thetas) * torch.cos(phis)  # cosϕ sinθ
        dirs_z = torch.sin(phis)  # sinϕ

        dirs = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)

        # shape: (H, 1, 1, 3), then broadcast to (H, L, P, 3)
        grid_init = dirs[:, None, None, :].repeat(1, self.n_levels, self.n_points, 1)

        # scale radius by (i+1)
        for i in range(self.n_points):
            scale = (i + 1) / (self.n_points + 1)
            grid_init[:, :, i, :].mul_(scale)

        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))

        # --- --- end of sampling offsets initialization --- ---

        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)

        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)

        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        """
        Args:

            query: (N, Length_{query}, C)
            reference_points: (N, Length_{query}, n_levels, 3), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                or (N, Length_{query}, n_levels, 6), add additional (d, w, h) to form reference boxes
            input_flatten: (N, \sum_{l=0}^{L-1} D_l \cdot H_l \cdot W_l, C)
            input_spatial_shapes: (n_levels, 3), [(D_0, H_0, W_0), (D_{1}, H_1, W_1), ..., (D_{L-1}, H_{L-1}, W_{L-1})]
            input_level_start_index: (n_levels, ), [0, D_0*H_0*W_0, D_0*H_0*W_0+D_1*H_1*W_1, ..., D_0*H_0*W_0+D_1*H_1*W_1+...+D_{L-1}*H_{L-1}*W_{L-1}]
            input_padding_mask: (N, \sum_{l=0}^{L-1} D_l \cdot H_l \cdot W_l), True for padding elements, False for non-padding elements

        returns: (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1] * input_spatial_shapes[:, 2]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # (N, Len_in, C=d_model) -> (N, Len_in, n_heads, d_model // n_heads)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads).to(query.dtype)

        # offsets: (N, Len_q, C=d_model) -> (N, Len_q, n_heads * n_levels * n_points * 3)
        #                                -> (N, Len_q, n_heads, n_levels, n_points, 3)
        # weights: (N, Len_q, C=d_model) -> (N, Len_q, n_heads * n_levels * n_points)
        #                                -> (N, Len_q, n_heads, n_levels * n_points)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 3)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        # conventions: ref_points X,Y,Z => a single point in the feature map, already normalised to [0, 1]
        #              ref_points X,Y,Z,D,W,H => centre + size of a 3-D bounding box, all in normalised units
        #              [..., :3] is box centre, [..., 3:] size (d, h, w) learned offsets here are applied to
        #              box dimensions, not the box centre so we do: loc = box_centre + (δ / n_points) * 0.5 * box_offset
        #              we dampen magnitude by nr. points s.t. changing n_points does not explode gradients
        #              (δ / n_points) * 0.5 * box_offset thus makes attn sampling offset displacements inside box
        if reference_points.shape[-1] == 3:
            # offset_normalizer = input_spatial_shapes with (D,H,W) reversed to (W,H,D)
            offset_normalizer = input_spatial_shapes[..., [2, 1, 0]]
            # (N, Len_q, 1, n_levels, 1, 3) + (N, Len_q, n_heads, n_levels, n_points, 3)
            #                               / (1, 1, 1, n_levels, 1, 3)
            #               -> (N, Len_q, n_heads, n_levels, n_points, 3)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 6:
            # (N, Len_q, 1, n_levels, 1, 3) + (N, Len_q, n_heads, n_levels, n_points, 3) / (N, Len_q, 1, n_levels, 1, 3)
            # -> (N, Len_q, n_heads, n_levels, n_points, 3)
            sampling_locations = (
                reference_points[:, :, None, :, None, :3]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 3:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 3 or 6, but get {} instead.".format(reference_points.shape[-1])
            )

        # cat sampling_offsets and attention_weights, generate sampling_loc_attn
        # (N, Len_q, n_heads, n_levels, n_levels, n_points, 3) -> (N, Len_q, n_heads, n_levels * n_points * 3)
        sampling_locations = sampling_locations.flatten(-3).to(query.dtype)
        # sampling_loc_attn: (N, Len_q, n_heads, n_levels * n_points * 4)
        # 3 for sampling locations, 1 for attention weights
        sampling_loc_attn = torch.cat([sampling_locations, attention_weights], dim=-1)

        output = FlashDeformAttnFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_loc_attn,
            self.im2col_step,
            self.n_points,
            self.use_reg,
        )
        output = self.output_proj(output)
        return output


# NOTE: used in SAM2
class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()

        self.self_attn = CrossAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = CrossAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            input_dim=embedding_dim, 
            hidden_dim=mlp_dim, 
            output_dim=embedding_dim, 
            num_layers=2, 
            activation=activation,
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = CrossAttention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(queries, queries, queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q, q, queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q, k, keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(k, q, queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class MemoryAttentionLayer(nn.Module):
    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: nn.Module,
    ):
        super().__init__()

        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation(activation)()

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(
        self,
        tgt,
        memory,
        query_pos,
        pos,
        pos_enc=None,
        prefix_k: int = 0,
    ):
        kwds = {}
        if isinstance(self.cross_attn_image, RopeCrossAttention):
            kwds["prefix_k"] = prefix_k
            if pos_enc is not None:
                kwds["pos_enc"] = pos_enc

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        pos_enc=None,
        prefix_k: int = 0,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(
            tgt, memory, query_pos, pos,
            pos_enc=pos_enc,
            prefix_k=prefix_k,
        )
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str = "GELU",
        pos_enc_at_input: bool = False,
        pos_enc_at_attn: bool = True,
        pos_enc_at_cross_attn_queries: bool = True,
        pos_enc_at_cross_attn_keys: bool = True,
        batch_first: bool = True,
        # Axial RoPE for cross-attention (same as make_axial_rope_freqs)
        input_fmt: str = "TZYXC",
        input_shape: tuple = (128, 128, 128, 2),
        patch_shape: tuple = (16, 16, 16),
        rope_theta: float = 10.0,
        # Optional: pass-through for Attention (self-attn) and RopeCrossAttention
        self_attn_downsample_rate: int = 1,
        cross_attn_qkv_bias: bool = True,
        cross_attn_qk_norm: bool = False,
        cross_attn_att_drop: float = 0.0,
        cross_attn_proj_drop: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

        # NOTE: only axial RoPE is supported for now
        head_dim = d_model // num_heads
        freqs_cis_q = make_axial_rope_freqs(
            input_fmt=input_fmt,
            input_shape=input_shape,
            patch_shape=patch_shape,
            dim=head_dim,
            theta=rope_theta,
            device=None,
        )
        self.register_buffer("freqs_cis_q", freqs_cis_q)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self_attn = CrossAttention(
                d_model,
                num_heads,
                downsample_rate=self_attn_downsample_rate,
            )
            cross_attn = RopeCrossAttention(
                dim=d_model,
                num_heads=num_heads,
                rope_type="axial",
                rope_theta=rope_theta,
                input_fmt=input_fmt,
                input_shape=input_shape,
                patch_shape=patch_shape,
                qkv_bias=cross_attn_qkv_bias,
                qk_norm=cross_attn_qk_norm,
                att_drop=cross_attn_att_drop,
                proj_drop=cross_attn_proj_drop,
            )
            layer = MemoryAttentionLayer(
                activation=activation,
                cross_attention=cross_attn,
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pos_enc_at_attn=pos_enc_at_attn,
                pos_enc_at_cross_attn_keys=pos_enc_at_cross_attn_keys,
                pos_enc_at_cross_attn_queries=pos_enc_at_cross_attn_queries,
                self_attention=self_attn,
            )
            self.layers.append(layer)

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list), "curr_pos must be a list"
            assert len(curr) == len(curr_pos) == 1, "curr and curr_pos must have the same length"
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        # Seq dim is 1 in batch-first (B, L, C)
        num_k_content = memory.shape[1] - num_obj_ptr_tokens
        pos_enc = None
        if isinstance(self.layers[0].cross_attn_image, RopeCrossAttention) and num_k_content > 0:
            freqs = self.freqs_cis_q.to(curr.device)
            if num_k_content == 1:
                # No-memory dummy token: Q gets RoPE, K gets no pos encoding
                pos_enc = (freqs, None)
            else:
                pos_enc = make_cross_rope_pos_enc_qk(
                    freqs, num_k_content, cross_rope_type="k_repeat_q"
                )

        for layer in self.layers:
            kwds = {
                "prefix_k": num_obj_ptr_tokens,
            }
            if pos_enc is not None:
                kwds["pos_enc"] = pos_enc

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output


class MaskUnitAttention(nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` in cell_observatory_platform.models.layers.utils for more details.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        """
        Args:
        - dim, dim_out: The input and output feature dimensions.
        - heads: The number of attention heads.
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).
        - use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride

        self.head_dim = dim_out // heads
        self.scale = (self.head_dim) ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Input should be of shape [batch, tokens, channels]. """
        B, N, _ = x.shape
        num_windows = (
            (N // (self.q_stride * self.window_size)) if self.use_mask_unit_attn else 1
        )

        qkv = (
            self.qkv(x)
            .reshape(B, num_windows, -1, 3, self.heads, self.head_dim) # [B, num_windows, *spatial, 3, heads, head_dim]
            .permute(3, 0, 4, 1, 2, 5) # [3, B, heads, num_windows, *spatial, head_dim]
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_stride > 1:
            # Refer to Unroll to see how this performs a maxpool-Nd
            q = (
                q.view(B, self.heads, num_windows, self.q_stride, -1, self.head_dim)
                .max(dim=3)
                .values
            )

        # q: [B, H, W, Lq, Dh]
        # k,v: [B, H, W, Lk, Dh]
        B, H, W, Lq, Dh = q.shape
        Lk = k.shape[3]

        # Move W next to B and merge -> 4D (for Flash SDPA)
        q_ = q.permute(0, 2, 1, 3, 4).reshape(B * W, H, Lq, Dh) # [B * W, H, Lq, Dh]
        k_ = k.permute(0, 2, 1, 3, 4).reshape(B * W, H, Lk, Dh) # [B * W, H, Lk, Dh]
        v_ = v.permute(0, 2, 1, 3, 4).reshape(B * W, H, Lk, Dh) # [B * W, H, Lk, Dh]

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            out_ = F.scaled_dot_product_attention(q_, k_, v_)

        # Un-merge back to [B, H, W, Lq, Dh]
        out = out_.reshape(B, W, H, Lq, Dh).permute(0, 2, 1, 3, 4)
        x = out

        # [B, H, W, Lq, Dh] -> [B, W, Lq, H, Dh] for (W, Lq) token order
        x = x.permute(0, 2, 3, 1, 4).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x