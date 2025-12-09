import logging
import math
import sys
from functools import partial
from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.init import constant_, xavier_uniform_

from models.ops.flash_deform_attn import FlashDeformAttnFunction, _is_power_of_2
from cell_observatory_platform.data.masking.mask_generator import apply_masks_rope
from cell_observatory_platform.models.ops.rope import (
    apply_rotary_emb,
    compute_axial_cis,
    compute_mixed_cis,
    generate_frequency_spectrum,
    generate_grid_indices,
)
from cell_observatory_platform.training.helpers import get_patch_sizes

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
                    q,
                    k,
                    v,
                    dropout_p=self.att_drop.p if self.training else 0.0,
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
        att_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-5),
        random_rotation_per_head: bool = True,
        rope_mixed: bool = True,
        rope_theta: float = 10.0,
        input_fmt: str = "TZXYC",
        input_shape: tuple = (16, 128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.device = device

        temporal_patch_size, axial_patch_size, lateral_patch_size = get_patch_sizes(
            input_format=input_fmt, patch_shape=patch_shape
        )

        temporal_patch_size, axial_patch_size, lateral_patch_size = get_patch_sizes(
            input_format=input_fmt, patch_shape=patch_shape
        )

        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        if qk_norm:
            assert norm_layer is not None, "norm_layer must be provided if qk_norm is True"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

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
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads, input_fmt=input_fmt)

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

            end_x = input_shape[input_fmt.index("X")] // lateral_patch_size
            end_y = input_shape[input_fmt.index("Y")] // lateral_patch_size
            end_x = input_shape[input_fmt.index("X")] // lateral_patch_size
            end_y = input_shape[input_fmt.index("Y")] // lateral_patch_size

            if self.input_fmt == "YXC":
                _, _, t_y, t_x = generate_grid_indices(
                    end_x=end_x, end_y=end_y, input_fmt=input_fmt, device=self.device, dtype=dtype
                )
                self.register_buffer("freqs_t_x", t_x)
                self.register_buffer("freqs_t_y", t_y)

                self.grid_indices = (None, None, t_y, t_x)

            elif self.input_fmt == "ZYXC":
                end_z = input_shape[input_fmt.index("Z")] // axial_patch_size
                end_z = input_shape[input_fmt.index("Z")] // axial_patch_size
                _, t_z, t_y, t_x = generate_grid_indices(
                    end_x=end_x, end_y=end_y, end_z=end_z, input_fmt=input_fmt, device=self.device, dtype=dtype
                )
                self.register_buffer("freqs_t_x", t_x)
                self.register_buffer("freqs_t_y", t_y)
                self.register_buffer("freqs_t_z", t_z)

                self.grid_indices = (None, t_z, t_y, t_x)

            elif self.input_fmt == "TYXC":
                end_t = input_shape[input_fmt.index("T")] // temporal_patch_size
                end_t = input_shape[input_fmt.index("T")] // temporal_patch_size
                t_t, _, t_y, t_x = generate_grid_indices(
                    end_x=end_x, end_y=end_y, end_t=end_t, input_fmt=input_fmt, device=self.device, dtype=dtype
                )
                self.register_buffer("freqs_t_x", t_x)
                self.register_buffer("freqs_t_y", t_y)
                self.register_buffer("freqs_t_t", t_t)

                self.grid_indices = (t_t, None, t_y, t_x)

            elif self.input_fmt == "TZYXC":
                end_z = input_shape[input_fmt.index("Z")] // axial_patch_size
                end_t = input_shape[input_fmt.index("T")] // temporal_patch_size
                end_z = input_shape[input_fmt.index("Z")] // axial_patch_size
                end_t = input_shape[input_fmt.index("T")] // temporal_patch_size
                t_t, t_z, t_y, t_x = generate_grid_indices(
                    end_x=end_x,
                    end_y=end_y,
                    end_z=end_z,
                    end_t=end_t,
                    input_fmt=input_fmt,
                    device=self.device,
                    dtype=dtype,
                )
                self.register_buffer("freqs_t_x", t_x)
                self.register_buffer("freqs_t_y", t_y)
                self.register_buffer("freqs_t_z", t_z)
                self.register_buffer("freqs_t_t", t_t)

                self.grid_indices = (t_t, t_z, t_y, t_x)

            else:
                raise NotImplementedError(f"Unknown input_fmt={input_fmt}")

        else:
            end_x = input_shape[input_fmt.index("X")] // lateral_patch_size
            end_y = input_shape[input_fmt.index("Y")] // lateral_patch_size
            end_x = input_shape[input_fmt.index("X")] // lateral_patch_size
            end_y = input_shape[input_fmt.index("Y")] // lateral_patch_size

            if "Z" in input_fmt:
                end_z = input_shape[input_fmt.index("Z")] // axial_patch_size
                end_z = input_shape[input_fmt.index("Z")] // axial_patch_size
            else:
                end_z = None

            if "T" in input_fmt:
                end_t = input_shape[input_fmt.index("T")] // temporal_patch_size
                end_t = input_shape[input_fmt.index("T")] // temporal_patch_size
            else:
                end_t = None

            self.freqs_cis = compute_axial_cis(
                theta=rope_theta,
                dim=self.dim // self.num_heads,
                end_x=end_x,
                end_y=end_y,
                end_z=end_z,
                end_t=end_t,
                input_fmt=input_fmt,
                device=self.device,
            )

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
            freqs_cis = self.compute_cis(
                freqs=self.freqs.to(x.device),
                num_heads=self.num_heads,
                t_t=t_t,
                t_z=t_z,
                t_y=t_y,
                t_x=t_x,
                input_fmt=self.input_fmt,
            )

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
                    q_rope,
                    k_rope,
                    v,
                    dropout_p=self.att_drop.p if self.training else 0.0,
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
