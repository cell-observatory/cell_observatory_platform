import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig, OmegaConf

from cell_observatory_platform.models.patch_embeddings import PatchEmbedding
from cell_observatory_platform.data.masking.mask_generator import apply_masks


def get_loss_fn(loss):
    if isinstance(loss, str):
        mapping = {
            "l2_masked": L2_masked_loss,
            "l1_masked": L1_masked_loss,
            "smooth_l1_masked": smooth_L1_masked_loss,
        }
        if loss in mapping:
            return mapping[loss]
        raise ValueError(f"Unknown loss type: {loss}")

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)

    if isinstance(loss, dict) and loss.get("loss_type") == "fourier_loss":
        return FourierLoss(
            alpha=loss.get("alpha", 0.001),
            fft_loss=loss.get("fft_loss", "l1_masked"),
            spatial_loss=loss.get("spatial_loss", "l2_masked"),
            input_fmt=loss["input_fmt"],
            input_shape=loss["input_shape"],
            patch_shape=loss["patch_shape"],
            embed_dim=loss["embed_dim"],
        )

    raise ValueError(f"Unknown loss type.")
    

def L2_masked_loss(targets, predictions, masks, aux_loss_meta=None):
    loss = (targets - predictions) ** 2
    loss = loss.mean(dim=-1)  # mean loss per patch
    loss = loss.sum() / masks.sum()
    return loss, None


def L1_masked_loss(targets, predictions, masks, aux_loss_meta=None):
    # compute loss over masked patches
    loss = torch.abs(targets - predictions)
    loss = loss.mean(dim=-1)  # mean loss per patch
    loss = loss.sum() / masks.sum()
    return loss, None


# see: https://github.com/facebookresearch/ijepa/main/src/train.py
def smooth_L1_masked_loss(targets, predictions, masks, aux_loss_meta=None):
    return F.smooth_l1_loss(targets, predictions), None


class FourierLoss(torch.nn.Module):
    def __init__(self, 
                 alpha,
                 fft_loss,
                 spatial_loss,
                 input_fmt, 
                 input_shape, 
                 patch_shape,
                 embed_dim
    ):
        super(FourierLoss, self).__init__()
        self.loss_type = "fourier_loss"

        self.input_fmt = input_fmt
        self.input_shape = input_shape
        self.patch_shape = patch_shape
        
        self.embed_dim = embed_dim
        
        axis_to_value = dict(zip(input_fmt, input_shape))
        self.in_chans = axis_to_value['C']
        self.num_frames = axis_to_value.get("T", None)

        self.patch_embedding = PatchEmbedding(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            patch_shape=self.patch_shape,
            embed_dim=self.embed_dim,
            channels=self.in_chans,
        )
        for p in self.patch_embedding.parameters():
            p.requires_grad_(False)
        self.patch_embedding.eval()
        
        self.alpha = alpha
        self.fft_loss = fft_loss
        if spatial_loss == "l1_masked":
            self.spatial_loss = get_loss_fn("l1_masked")
        elif spatial_loss == "l2_masked":
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
        full_targets_fft = torch.fft.fftn(full_targets.to(torch.float32), dim=(-4, -3, -2))
        full_predictions_fft = torch.fft.fftn(full_predictions.to(torch.float32), dim=(-4, -3, -2))

        full_targets_fft = torch.abs(full_targets_fft)
        full_predictions_fft = torch.abs(full_predictions_fft)

        if self.fft_loss == "l1_masked":
            fft_loss = (full_targets_fft - full_predictions_fft).abs().mean()
        elif self.fft_loss == "l2_masked":
            fft_loss = ((full_targets_fft - full_predictions_fft) ** 2).mean()
        else:
            raise ValueError(f"Unknown fft loss type: {self.fft_loss}, {type(self.fft_loss)}")
        
        spatial_loss, _ = self.spatial_loss(targets, predictions, masks, aux_loss_meta=None)

        fft_loss = self.alpha * fft_loss
        spatial_loss = (1 - self.alpha) * spatial_loss

        aux_losses = {
            "fft_loss": fft_loss, 
            "spatial_loss": spatial_loss
        }

        loss = fft_loss + spatial_loss
        return loss, aux_losses