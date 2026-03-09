from __future__ import absolute_import, division, print_function

import pytest
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from ops3d import _C
except ImportError:
    print("FlashDeformAttnFunction failed to load. Please compile ops3d if needed.")


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


def factors(N):
    res = []
    # find all integer divisors of N
    for i in range(1, N + 1):
        if N % i == 0:
            res.append(i)
    return res


def findspec(B, Q, G, C):
    # we fix d_stride=8
    d_stride = 8
    ms = factors(B * Q)
    multiplier = 1

    # total number of threads per block
    # is equal to multiplier * G * C / d_stride
    # maximum = 512 for performance reasons
    # TODO: check if can go to 1024
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m

    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


def findspec_bwd(B, Q, G, C, max_tpb=256, max_mult=64):
    # 1) pick d_stride so channels_per_thread = C//d_stride fits
    for d in (1, 2, 4, 8, 16, 32):
        if C % d == 0 and G * (C // d) <= max_tpb:
            d_stride = d
            break
    else:
        raise RuntimeError(f"Cannot fit G={G}, C={C} into {max_tpb} threads")

    # 2) pick multiplier so we can split B*Q / multiplier blocks
    #    without exceeding max_tpb
    best_mult = 1
    for m in factors(B * Q):
        thr = m * G * (C // d_stride)
        if m <= max_mult and thr <= max_tpb:
            best_mult = max(best_mult, m)

    blockthread = best_mult * G * (C // d_stride)
    return d_stride, blockthread


class FlashDeformAttnFunction(Function):
    @staticmethod
    # @torch.autocast("cuda", enabled=True, dtype=torch.float16)
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_loc_attn,
        im2col_step,  # partitions batch into bs/im2col calls
        K=8,  # num points sampled
        use_reg=True,  # use warp based implementation
    ):
        ctx.K = K
        ctx.im2col_step = im2col_step

        num_levels = value_spatial_shapes.shape[0]
        expected_last_dim = num_levels * K * 4
        if sampling_loc_attn.shape[-1] != expected_last_dim:
            raise ValueError(
                f"sampling_loc_attn last dim {sampling_loc_attn.shape[-1]} does not match "
                f"spatial_shapes levels {num_levels} and K {K}: expected {expected_last_dim}"
            )

        # findspec(Batch Size, Queries, Num_heads = G, Channels per Group = C)
        # determine number of channels per thread inside group = d_stride and total number of threads in block = multiplier * G * C / d_stride
        # where we partition threads into blocks of dimension Z=multiplier, Y=G, X=C/d_stride
        d_stride, blockthread = findspec(value.shape[0], sampling_loc_attn.shape[1], value.shape[2], value.shape[3])
        d_stride_backward, blockthread_backward = findspec_bwd(
            value.shape[0], sampling_loc_attn.shape[1], value.shape[2], value.shape[3]
        )

        ctx.d_stride_backward = d_stride_backward
        ctx.blockthread_backward = blockthread_backward

        output = _C.flash_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_loc_attn,
            ctx.im2col_step,
            K,
            d_stride,
            blockthread,
            use_reg,
        )

        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_loc_attn)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_loc_attn = ctx.saved_tensors

        grad_value, grad_sampling_loc_attn = _C.flash_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_loc_attn,
            grad_output.contiguous(),
            ctx.im2col_step,
            ctx.K,
            ctx.d_stride_backward,
            ctx.blockthread_backward,
        )

        return grad_value, None, None, grad_sampling_loc_attn, None, None, None


def ms_deform_attn_core_pytorch_3d(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug / testing only, use cuda version otherwise
    # N_ = batch_size, S_ = total_spatial_tokens, M_ = num_heads, E_ = embed_dim
    N_, S_, M_, E_ = value.shape
    # Lq = num_query_points, L = num_levels, P = num_points (sampling)
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([D_ * H_ * W_ for (D_, H_, W_) in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (D_, H_, W_) in enumerate(value_spatial_shapes):
        # N_, D_*H_*W_, M_, E_ -> N_, D_*H_*W_, M_*E_ -> N_, M_*E_, D_*H_*W_ -> N_*M_, E_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, E_, D_, H_, W_)
        # N_, Lq_, M_, L, P_, 3 -> N_, Lq_, M_, P_, 3 -> N_, M_, Lq_, P_, 3 -> N_*M_, Lq_, P_, 3
        # NOTE: grid_sample expects sampling grid to have 3 spatial dims to iterate across
        #       hence we unsqueeze and add a dummy dim
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1).unsqueeze(1)
        # N_*M_, E_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_l_ = sampling_value_l_.squeeze(2)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    #  List[N_*M_, E_, Lq_, P_] -> (N*M, E, Lq, L, P) -> (broadcast mul & sum last dim) -> (N*M, E, Lq) -> (N, M*E, Lq)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * E_, Lq_)
    return output.transpose(1, 2).contiguous()  #  (N, Lq, M*E)
