import torch
import torch.nn.functional as F

from models.patch_embeddings import PatchEmbedding
from data.masking.mask_generator import apply_masks


def get_loss_fn(loss_type: str ):
    if loss_type == "l2_masked":
        return L2_masked_loss
    
    elif loss_type == "l1_masked":
        return L1_masked_loss

    elif loss_type == "smooth_l1_masked":
        return smooth_L1_masked_loss

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    

def L2_masked_loss(targets, predictions, masks, aux_loss_meta=None):
    loss = (targets - predictions) ** 2
    loss = loss.mean(dim=-1)  # mean loss per patch
    loss = loss.sum() / masks.sum()
    return loss


def L1_masked_loss(targets, predictions, masks, aux_loss_meta=None):
    # compute loss over masked patches
    loss = torch.abs(targets - predictions)
    loss = loss.mean(dim=-1)  # mean loss per patch
    loss = loss.sum() / masks.sum()
    return loss


# see: https://github.com/facebookresearch/ijepa/main/src/train.py
def smooth_L1_masked_loss(targets, predictions, masks, aux_loss_meta=None):
    return F.smooth_l1_loss(targets, predictions)


class FourierLoss(torch.nn.Module):
    def __init__(self, 
                 alpha,
                 fft_loss,
                 spatial_loss,
                 input_fmt, 
                 input_shape, 
                 patch_shape, 
                 embed_dim, 
                 in_chans=1
    ):
        super(FourierLoss, self).__init__()
        self.input_fmt = input_fmt
        self.input_shape = input_shape
        self.patch_shape = patch_shape
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.patch_embedding = PatchEmbedding(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            patch_shape=self.patch_shape,
            embed_dim=self.embed_dim,
            channels=self.in_chans,
        )
        
        self.alpha = alpha
        if fft_loss == "L1":
            self.fft_loss = get_loss_fn("l1_masked")
        elif fft_loss == "L2":
            self.fft_loss = get_loss_fn("l2_masked")
        else:
            raise ValueError(f"Unknown Fourier loss type: {fft_loss}")
        if spatial_loss == "L1":
            self.spatial_loss = get_loss_fn("l1_masked")
        elif spatial_loss == "L2":
            self.spatial_loss = get_loss_fn("l2_masked")
        else:
            raise ValueError(f"Unknown spatial loss type: {spatial_loss}")

    def forward(self, targets, predictions, masks, aux_loss_meta):
        patches_used = aux_loss_meta.get('patches_used', None)
        target_masks = aux_loss_meta['target_masks']

        full_targets, full_predictions = aux_loss_meta['targets'], aux_loss_meta['predictions']
        
        full_targets = self.patch_embedding.unpatchify(full_targets, out_channels=None)
        full_predictions = self.patch_embedding.unpatchify(full_predictions, out_channels=None)
        
        # NOTE: works for ZYXC and TZYXC formats
        full_targets_fft = torch.fft.fftn(full_targets, dim=(-4, -3, -2))
        full_predictions_fft = torch.fft.fftn(full_predictions, dim=(-4, -3, -2))

        full_targets_fft = torch.abs(full_targets_fft)
        full_predictions_fft = torch.abs(full_predictions_fft)

        full_targets_fft_patches = self.patch_embedding.patchify(full_targets_fft)
        full_predictions_fft_patches = self.patch_embedding.patchify(full_predictions_fft)

        # compute loss over masked patches (re-index if blocked masking removed some patches)
        if patches_used is not None:
            target_idx_in_patches_used = torch.searchsorted(patches_used, target_masks)
        else:
            target_idx_in_patches_used = target_masks
        targets_fft = apply_masks(full_targets_fft_patches, masks=target_masks)
        predictions_fft = apply_masks(full_predictions_fft_patches, masks=target_idx_in_patches_used)

        fft_loss = self.fft_loss(targets_fft, predictions_fft, masks, aux_loss_meta=None)
        spatial_loss = self.spatial_loss(targets, predictions, masks, aux_loss_meta=None)

        loss = self.alpha * fft_loss + (1 - self.alpha) * spatial_loss
        return loss