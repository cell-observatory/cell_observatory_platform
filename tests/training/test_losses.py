import pytest
import torch

from training.losses import FourierLoss

@pytest.mark.parametrize(
    "input_fmt,input_shape,patch_shape,in_chans",
    [
        # 4D case: TZYXC 
        ("TZYXC", (16, 128, 256, 256, 2), (16, 16, 16, 16), 2),
        # 3D case: ZYXC 
        ("ZYXC",  (128, 256, 256, 2),     (16, 16, 16),   2),
    ],
)
def test_fourier_loss_forward_backward(input_fmt, input_shape, patch_shape, in_chans):
    torch.manual_seed(0)
    device = "cuda"
    B = 2

    loss_mod = FourierLoss(
        alpha=0.01,
        fft_loss="l1_masked",
        spatial_loss="l2_masked",
        input_fmt=input_fmt,
        input_shape=input_shape,
        patch_shape=patch_shape,
        embed_dim=256,
        # in_chans=in_chans, # FIXME
    ).to(device)

    full_shape = (B, *input_shape)
    full_targets_img = torch.randn(full_shape, device=device)

    full_targets_patches = loss_mod.patch_embedding.patchify(full_targets_img)
    B_, P, D = full_targets_patches.shape
    assert B_ == B and P > 4, "Test expects more than 4 patches to mask a few."

    full_predictions_patches = (full_targets_patches + 0.01 * torch.randn_like(full_targets_patches)).detach().requires_grad_(True)

    M = 4
    masked_idx = torch.arange(M, device=device).unsqueeze(0).repeat(B, 1)
    masks_grid = torch.zeros((B, P), dtype=torch.float32, device=device)
    masks_grid.scatter_(1, masked_idx, 1.0)

    batch_index = torch.arange(B, device=device).unsqueeze(-1)
    targets_masked = full_targets_patches[batch_index, masked_idx, :]
    predictions_masked = full_predictions_patches[batch_index, masked_idx, :]

    aux = {
        "targets": full_targets_patches,
        "predictions": full_predictions_patches,
        "target_masks": masked_idx,
    }

    print(f"targets_masked.shape: {targets_masked.shape}")
    print(f"predictions_masked.shape: {predictions_masked.shape}")
    print(f"masks_grid.shape: {masks_grid.shape}")
    print(f"full_targets_patches.shape: {full_targets_patches.shape}")
    print(f"full_predictions_patches.shape: {full_predictions_patches.shape}")
    print(f"target_masks.shape: {aux['target_masks'].shape}")

    out = loss_mod(
        targets=targets_masked,
        predictions=predictions_masked,
        masks=masks_grid,
        aux_loss_meta=aux,
    )

    assert isinstance(out, torch.Tensor)
    assert out.ndim == 0
    assert torch.isfinite(out), "Loss produced inf/NaN"

    out.backward()
    assert full_predictions_patches.grad is not None
    assert torch.isfinite(full_predictions_patches.grad).all()
    assert full_predictions_patches.grad.abs().sum() > 0