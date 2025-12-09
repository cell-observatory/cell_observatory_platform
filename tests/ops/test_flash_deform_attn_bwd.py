from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from cell_observatory_platform.models.ops.flash_deform_attn_func import (
    FlashDeformAttnFunction,
    ms_deform_attn_core_pytorch_3d,
)

try:
    import ops3d._C as _C
except ImportError:
    print("FlashDeformAttnFunction op failed to load. Please compile ops3d if needed.")
    pytestmark = pytest.mark.skip(reason="This module is temporarily disabled till we add ops3d to the docker image")


# ---------------------------- HELPERS -------------------------------------


def to_kernel_coords(locs):  # (..., d, h, w) -> (..., w, h, d)
    return locs[..., [2, 1, 0]].contiguous()


def from_kernel_grads(g_locs):  # (..., w, h, d) -> (..., d, h, w)
    return g_locs[..., [2, 1, 0]]


# ----------------------------- FIXED TESTING PARAMS -------------------------------------


torch.manual_seed(42)
device = torch.device("cuda")

# problem sizes
N, M, D = 1, 8, 48
Lq, L, K = (64 * 64 * 64 + 32 * 32 * 32), 2, 4
im2col_step = 64

shapes = torch.tensor(
    [
        [64, 64, 64],
        [32, 32, 32],
        #  [16, 16, 16],
        #  [ 8,  8,  8]
    ],
    dtype=torch.long,
    device=device,
)
level_start_index = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
S = int(shapes.prod(1).sum())


# -------------------------------------------------------------------------


def test_flash_backward_matches_reference():
    value = (torch.rand(N, S, M, D, device=device) * 0.01).half().requires_grad_(True)

    sampling_locs = torch.rand(N, Lq, M, L, K, 3, device=device).half()
    raw_attn = (torch.rand(N, Lq, M, L, K, device=device) + 1e-5).half()
    packed = torch.cat(
        [sampling_locs.reshape(N, Lq, M, L * K * 3), raw_attn.reshape(N, Lq, M, L * K)], dim=-1
    ).requires_grad_(True)

    flash_out = FlashDeformAttnFunction.apply(value, shapes, level_start_index, packed, im2col_step, K, True)
    (flash_out.sum() / 10).backward()

    g_val_flash = value.grad.float()
    g_loc_flash = packed.grad[..., : L * K * 3].reshape_as(sampling_locs).float()
    g_att_flash = packed.grad[..., L * K * 3 :].reshape_as(raw_attn).float()

    value_ref = value.detach().float().clone().requires_grad_(True)
    loc_ref = sampling_locs.detach().float().clone().requires_grad_(True)
    raw_attn_ref = raw_attn.detach().float().clone().requires_grad_(True)

    attn_soft = F.softmax(raw_attn_ref.flatten(-2, -1), dim=-1).unflatten(-1, (L, K))

    ref_out = ms_deform_attn_core_pytorch_3d(value_ref, shapes, loc_ref, attn_soft)
    (ref_out.sum() / 10).backward()

    g_val_ref = value_ref.grad
    g_loc_ref = loc_ref.grad
    g_att_ref = raw_attn_ref.grad

    assert torch.allclose(flash_out.float(), ref_out, rtol=1e-2, atol=1e-3)
    assert torch.allclose(g_val_flash, g_val_ref, rtol=1e-2, atol=1e-3)
    assert torch.allclose(g_loc_flash, g_loc_ref, rtol=1e-2, atol=1e-3)
    assert torch.allclose(g_att_flash, g_att_ref, rtol=1e-2, atol=1e-3)
