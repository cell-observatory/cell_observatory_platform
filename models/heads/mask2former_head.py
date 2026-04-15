"""
Adapted from:
https://github.com/facebookresearch/dinov3/blob/main/dinov3/eval/segmentation/models/heads/mask2former_head.py
"""

from typing import Optional

from torch import nn
from torch.nn import functional as F
import torch
from cell_observatory_platform.models.heads.mask2former_decoder import MultiScaleMaskedTransformerDecoder
from cell_observatory_platform.models.heads.pixel_decoders import Mask2FormerPixelDecoder


class Mask2FormerHead(nn.Module):
    def __init__(
        self,
        pixel_decoder: Mask2FormerPixelDecoder,
        predictor: MultiScaleMaskedTransformerDecoder,
    ):
        super().__init__()

        
        self.pixel_decoder = pixel_decoder
        self.predictor = predictor

    def forward_features(self, features, mask=None):
        return self.layers(features, mask)

    def forward(self, features, mask=None):
        output = self.forward_features(features, mask)
        return output

    def predict(self, features, rescale_size: Optional[tuple] = None, mask=None):
        output = self.forward_features(features, mask)
        if rescale_size is not None:
            output["pred_masks"] = F.interpolate(
                output["pred_masks"], # B, Q, D, H, W
                size=rescale_size, # D, H, W
                mode="trilinear",
                align_corners=False,
            )
        return output

    def layers(self, features, mask=None):
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features)
        predictions = self.predictor(multi_scale_features, mask_features, mask)
        return predictions
