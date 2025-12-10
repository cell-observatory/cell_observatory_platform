from __future__ import absolute_import, division, print_function

import pytest
import torch

from cell_observatory_platform.models.ops.flash_deform_attn import (
    FlashDeformAttnFunction,
    ms_deform_attn_core_pytorch_3d,
)

try:
    import ops3d._C as _C
except ImportError:
    print("FlashDeformAttnFunction op failed to load. Please compile ops3d if needed.")
    pytestmark = pytest.mark.skip(reason="This module is temporarily disabled till we add ops3d to the docker image")


# ------------------------ FIXED TESTING PARAMS -------------------------------------


torch.manual_seed(42)
device = torch.device("cuda")

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

N = 1  # batch size
M = 8  # number of attention heads
D = 288  # feature dimension (per head)
Lq = 32 * 32 * 32  # query length
L = 4  # number of feature levels
K = 8  # sampling points per query / head / level
im2col_step = 128  # # partitions batch into bs/im2col calls

# spatial shapes for each feature level (D, H, W)
shapes = torch.tensor(
    [
        [64, 64, 64],
        [32, 32, 32],
        [16, 16, 16],
        [8, 8, 8],
    ],
    dtype=torch.long,
    device="cuda",
)


level_start_index = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
S = int((shapes.prod(1)).sum())


# ------------------------  -------------------------------------  -------------------------------------


@torch.no_grad()
def test_forward_equal_with_pytorch_half():
    value = torch.rand(N, S, M, D).cuda() * 0.01

    sampling_locations = torch.rand(N, Lq, M, L, K, 3).cuda()
    attention_weights = torch.rand(N, Lq, M, L, K).cuda() + 1e-5
    sampling_loc_attn = torch.cat(
        [sampling_locations.reshape(N, Lq, M, L * K * 3), attention_weights.reshape(N, Lq, M, L * K)], dim=-1
    )
    attention_weights = torch.nn.functional.softmax(attention_weights.flatten(-2, -1), dim=-1).unflatten(-1, (L, K))

    output_cuda = (
        FlashDeformAttnFunction.apply(
            value.half(),
            shapes,
            level_start_index,
            sampling_loc_attn.half(),
            im2col_step,
            K,
            True,  # register vs shared memory kernel
        )
        .detach()
        .cpu()
        .double()
    )

    output_pytorch = (
        ms_deform_attn_core_pytorch_3d(
            value,
            shapes,
            sampling_locations,
            attention_weights,
        )
        .detach()
        .double()
        .cpu()
    )

    # check if any outputs are NaN or Inf
    if torch.isnan(output_pytorch).any() or torch.isinf(output_pytorch).any():
        raise ValueError("Output from ms_deform_attn_core_pytorch_3d contains NaN or Inf values.")
    if torch.isnan(output_cuda).any() or torch.isinf(output_cuda).any():
        raise ValueError("Output from FlashDeformAttnFunction contains NaN or Inf values.")

    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()

    assert torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3), (
        f"forward mismatch; " f"max abs : {max_abs_err}, max rel : {max_rel_err}"
    )
