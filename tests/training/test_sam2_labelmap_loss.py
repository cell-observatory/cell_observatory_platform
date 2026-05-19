"""Tests for the SAM2 labelmap-native loss path.

`MultiStepMultiMasksAndIousLoss._forward_labelmap_path` computes focal /
dice / sampled-soft-IoU / obj-score losses by sampling uncertain points
directly from the integer labelmap target view, with valid+presence gating.
This file exercises the path end-to-end on small CUDA inputs and checks
the gating contract by comparing pad-augmented vs no-pad runs.
"""
from __future__ import annotations

import pytest
import torch

from cell_observatory_platform.training.losses import MultiStepMultiMasksAndIousLoss

CUDA_AVAILABLE = torch.cuda.is_available()


def _make_criterion(
    num_points: int = 64,
    importance_sample_ratio: float = 0.75,
) -> MultiStepMultiMasksAndIousLoss:
    weight_dict = {
        "loss_mask": 20.0,
        "loss_dice": 1.0,
        "loss_iou": 1.0,
        "loss_class": 1.0,
    }
    return MultiStepMultiMasksAndIousLoss(
        input_fmt="BTZYXC",
        weight_dict=weight_dict,
        pred_obj_scores=True,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        num_points=num_points,
        oversample_ratio=3.0,
        importance_sample_ratio=importance_sample_ratio,
    )


def _make_outs(N: int, M: int, Z: int, Y: int, X: int, device, dtype=torch.float32):
    src = torch.randn(N, M, Z, Y, X, device=device, dtype=dtype, requires_grad=True)
    pred_ious = torch.rand(N, M, device=device, dtype=dtype)
    obj_logits = torch.randn(N, 1, device=device, dtype=dtype)
    return {
        "multistep_pred_multimasks_high_res": [src],
        "multistep_pred_ious": [pred_ious],
        "multistep_object_score_logits": [obj_logits],
    }


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_labelmap_path_runs_and_keys():
    torch.manual_seed(0)
    device = torch.device("cuda")
    T, B, Z, Y, X = 1, 1, 4, 6, 8
    K = 3
    N = B * K

    labelmap = torch.zeros((B * T, Z, Y, X), dtype=torch.int32, device=device)
    labelmap[0, 1, 2, 3] = 7
    labelmap[0, 2, 3, 4] = 11

    target_view = {
        "num_frames": T,
        "num_videos": B,
        "labelmaps": labelmap,
        "img_ids": [torch.zeros(N, dtype=torch.int32, device=device)],
        "instance_ids": [torch.tensor([7, 11, -1], dtype=torch.int64, device=device)],
        "valid": [torch.tensor([True, True, False], device=device)],
        "presence_t": [torch.tensor([True, True, False], device=device)],
        "boxes": [torch.zeros(N, 6, dtype=torch.float32, device=device)],
        "box_format": "zyxzyx",
        "masks": [torch.zeros(N, Z, Y, X, dtype=torch.bool, device=device)],
    }

    criterion = _make_criterion(num_points=128).to(device)
    outs_batch = [_make_outs(N, M=3, Z=Z, Y=Y, X=X, device=device)]
    losses = criterion(outs_batch, target_view)

    for key in ("loss_mask", "loss_dice", "loss_iou", "loss_class", criterion.core_loss_key):
        assert key in losses, f"missing {key}"
        v = losses[key]
        assert torch.is_tensor(v) and v.ndim == 0, f"{key} expected scalar, got {v}"
        assert torch.isfinite(v).item(), f"{key} not finite"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_labelmap_path_pad_rows_are_invariant():
    # Adding pad rows (valid=False, presence_t=False, instance_id=-1) must not
    # change any of the divided losses. Predictions for pad rows are random
    # noise -- if the gating were broken, focal/dice/iou would shift.
    #
    # Pinned to T=1 so the test sees one RNG draw inside
    # get_uncertain_point_coords_with_randomness. With T>1, pad rows in
    # frame t advance the global CUDA RNG and the t+1 oversample draws from
    # a different state, producing different (but equally valid in expectation)
    # coords for valid rows -- a sampling-noise artifact unrelated to gating.
    torch.manual_seed(123)
    device = torch.device("cuda")
    T, B, Z, Y, X = 1, 1, 3, 5, 7
    M = 3
    K_real = 2
    K_pad = 5
    K_full = K_real + K_pad

    labelmap = torch.zeros((B * T, Z, Y, X), dtype=torch.int32, device=device)
    labelmap[0, 0, 1, 2] = 4
    labelmap[0, 2, 3, 4] = 9

    instance_real = torch.tensor([4, 9], dtype=torch.int64, device=device)
    valid_real = torch.tensor([True, True], device=device)
    presence_real_t = [torch.tensor([True, True], device=device)]

    # Build two outs_batch: one with K_real rows, one with K_full rows where
    # the first K_real rows ARE THE SAME predictions and the rest are noise.
    torch.manual_seed(7)
    src_real = [torch.randn(K_real, M, Z, Y, X, device=device) for _ in range(T)]
    ious_real = [torch.rand(K_real, M, device=device) for _ in range(T)]
    obj_real = [torch.randn(K_real, 1, device=device) for _ in range(T)]

    torch.manual_seed(7)
    src_full = []
    ious_full = []
    obj_full = []
    for t in range(T):
        s_r = torch.randn(K_real, M, Z, Y, X, device=device)
        i_r = torch.rand(K_real, M, device=device)
        o_r = torch.randn(K_real, 1, device=device)
        s_p = torch.randn(K_pad, M, Z, Y, X, device=device)
        i_p = torch.rand(K_pad, M, device=device)
        o_p = torch.randn(K_pad, 1, device=device)
        src_full.append(torch.cat([s_r, s_p], dim=0))
        ious_full.append(torch.cat([i_r, i_p], dim=0))
        obj_full.append(torch.cat([o_r, o_p], dim=0))

    outs_real = [
        {
            "multistep_pred_multimasks_high_res": [src_real[t]],
            "multistep_pred_ious": [ious_real[t]],
            "multistep_object_score_logits": [obj_real[t]],
        }
        for t in range(T)
    ]
    outs_full = [
        {
            "multistep_pred_multimasks_high_res": [src_full[t]],
            "multistep_pred_ious": [ious_full[t]],
            "multistep_object_score_logits": [obj_full[t]],
        }
        for t in range(T)
    ]

    view_real = {
        "num_frames": T,
        "num_videos": B,
        "labelmaps": labelmap,
        "img_ids": [torch.zeros(K_real, dtype=torch.int32, device=device) + t for t in range(T)],
        "instance_ids": [instance_real.clone() for _ in range(T)],
        "valid": [valid_real.clone() for _ in range(T)],
        "presence_t": [p.clone() for p in presence_real_t],
        "boxes": [torch.zeros(K_real, 6, dtype=torch.float32, device=device) for _ in range(T)],
        "box_format": "zyxzyx",
        "masks": [torch.zeros(K_real, Z, Y, X, dtype=torch.bool, device=device) for _ in range(T)],
    }

    instance_full = torch.cat(
        [instance_real, torch.full((K_pad,), -1, dtype=torch.int64, device=device)]
    )
    valid_full = torch.cat(
        [valid_real, torch.zeros(K_pad, dtype=torch.bool, device=device)]
    )
    presence_full_t = [
        torch.cat([p, torch.zeros(K_pad, dtype=torch.bool, device=device)])
        for p in presence_real_t
    ]
    view_full = {
        "num_frames": T,
        "num_videos": B,
        "labelmaps": labelmap,
        "img_ids": [torch.zeros(K_full, dtype=torch.int32, device=device) + t for t in range(T)],
        "instance_ids": [instance_full.clone() for _ in range(T)],
        "valid": [valid_full.clone() for _ in range(T)],
        "presence_t": [p.clone() for p in presence_full_t],
        "boxes": [torch.zeros(K_full, 6, dtype=torch.float32, device=device) for _ in range(T)],
        "box_format": "zyxzyx",
        "masks": [torch.zeros(K_full, Z, Y, X, dtype=torch.bool, device=device) for _ in range(T)],
    }

    # importance_sample_ratio=1.0 skips the random-points concat so the only
    # RNG draw inside get_uncertain_point_coords_with_randomness is the
    # initial oversample. CUDA's RNG is row-deterministic across shapes for
    # a single draw, so valid rows get identical coords whether or not pad
    # rows are appended. With ratio<1.0, pad rows would advance the RNG
    # between draws and change the random-points-only portion's coords for
    # valid rows -- the expected loss is the same but MC noise differs.
    criterion = _make_criterion(num_points=128, importance_sample_ratio=1.0).to(device)

    torch.manual_seed(99)
    losses_real = criterion(outs_real, view_real)
    torch.manual_seed(99)
    losses_full = criterion(outs_full, view_full)

    for key in ("loss_mask", "loss_dice", "loss_iou", "loss_class"):
        v_r = losses_real[key].detach()
        v_f = losses_full[key].detach()
        assert torch.allclose(v_r, v_f, atol=1e-5, rtol=1e-5), (
            f"{key}: real={v_r.item():.6f} vs pad={v_f.item():.6f}"
        )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_labelmap_path_presence_zero_zeros_mask_losses():
    # If presence_t is all-False across the batch, mask/dice/iou must be 0
    # (the gate zeros every row); loss_class still flows from valid.
    torch.manual_seed(0)
    device = torch.device("cuda")
    T, B, Z, Y, X = 1, 1, 2, 3, 4
    K = 2
    N = B * K

    labelmap = torch.zeros((B * T, Z, Y, X), dtype=torch.int32, device=device)
    # Note: ids 1 and 2 do NOT appear in labelmap -> presence=False naturally,
    # but we set explicitly here to test the gate, not torch.isin.
    target_view = {
        "num_frames": T,
        "num_videos": B,
        "labelmaps": labelmap,
        "img_ids": [torch.zeros(N, dtype=torch.int32, device=device)],
        "instance_ids": [torch.tensor([1, 2], dtype=torch.int64, device=device)],
        "valid": [torch.tensor([True, True], device=device)],
        "presence_t": [torch.tensor([False, False], device=device)],
        "boxes": [torch.zeros(N, 6, dtype=torch.float32, device=device)],
        "box_format": "zyxzyx",
        "masks": [torch.zeros(N, Z, Y, X, dtype=torch.bool, device=device)],
    }

    criterion = _make_criterion(num_points=64).to(device)
    outs_batch = [_make_outs(N, M=3, Z=Z, Y=Y, X=X, device=device)]
    losses = criterion(outs_batch, target_view)

    for key in ("loss_mask", "loss_dice", "loss_iou"):
        assert losses[key].item() == 0.0, f"{key} expected 0, got {losses[key].item()}"
    # obj-score loss still flows
    assert losses["loss_class"].item() > 0.0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_labelmap_path_backprops_to_predictions():
    torch.manual_seed(0)
    device = torch.device("cuda")
    T, B, Z, Y, X = 1, 1, 3, 5, 7
    K = 2

    labelmap = torch.zeros((B * T, Z, Y, X), dtype=torch.int32, device=device)
    labelmap[0, 1, 2, 3] = 5
    labelmap[0, 2, 3, 4] = 9

    target_view = {
        "num_frames": T,
        "num_videos": B,
        "labelmaps": labelmap,
        "img_ids": [torch.zeros(K, dtype=torch.int32, device=device)],
        "instance_ids": [torch.tensor([5, 9], dtype=torch.int64, device=device)],
        "valid": [torch.tensor([True, True], device=device)],
        "presence_t": [torch.tensor([True, True], device=device)],
        "boxes": [torch.zeros(K, 6, dtype=torch.float32, device=device)],
        "box_format": "zyxzyx",
        "masks": [torch.zeros(K, Z, Y, X, dtype=torch.bool, device=device)],
    }

    criterion = _make_criterion(num_points=128).to(device)
    outs = _make_outs(K, M=3, Z=Z, Y=Y, X=X, device=device)
    src = outs["multistep_pred_multimasks_high_res"][0]
    losses = criterion([outs], target_view)
    losses[criterion.core_loss_key].backward()
    assert src.grad is not None
    assert torch.any(src.grad != 0.0), "grads should flow into predictions"
