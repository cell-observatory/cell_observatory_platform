from __future__ import absolute_import, division, print_function

import math
import warnings

import torch
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from cell_observatory_platform.models.ops.flash_deform_attn_func import FlashDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


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
