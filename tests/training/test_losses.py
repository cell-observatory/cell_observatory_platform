import pytest
import torch

from cell_observatory_platform.training.losses import FourierLoss
from cell_observatory_platform.models.layers.patch_embeddings import PatchEmbedding


@pytest.mark.parametrize(
    "input_fmt,input_shape,patch_shape,in_chans",
    [
        # 4D case: TZYXC
        ("TZYXC", (2, 128, 128, 128, 2), (2, 16, 16, 16), 2),
        # 3D case: ZYXC
        ("ZYXC", (128, 128, 128, 2), (16, 16, 16), 2),
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

    pe = PatchEmbedding(
        input_fmt=input_fmt,
        input_shape=input_shape,
        patch_shape=patch_shape,
        embed_dim=256,
        channels=in_chans,
    )
    full_targets_patches = pe._patchify(full_targets_img)
    
    B_, P, D = full_targets_patches.shape
    assert B_ == B and P > 4, "Test expects more than 4 patches to mask a few."

    full_predictions_patches = (
        (full_targets_patches + 0.01 * torch.randn_like(full_targets_patches)).detach().requires_grad_(True)
    )

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

    loss, out = loss_mod(
        targets=targets_masked,
        predictions=predictions_masked,
        num_patches=masks_grid.sum(),
        aux_loss_meta=aux,
    )
    print(f"Loss output: {loss.item()}")

    assert loss.ndim == 0
    assert torch.isfinite(loss), "Loss produced inf/NaN"

    loss.backward()
    assert full_predictions_patches.grad is not None
    assert torch.isfinite(full_predictions_patches.grad).all()
    assert full_predictions_patches.grad.abs().sum() > 0
