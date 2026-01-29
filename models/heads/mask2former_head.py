"""
Adapted from:
https://github.com/facebookresearch/dinov3/blob/main/dinov3/eval/segmentation/models/heads/mask2former_head.py
"""

from typing import Dict, Tuple

from torch import nn
from torch.nn import functional as F

from cell_observatory_platform.models.heads.mask2former_decoder import MultiScaleMaskedTransformerDecoder
from cell_observatory_platform.models.heads.pixel_decoders import Mask2FormerPixelDecoder


class Mask2FormerHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pixel_decoder: Mask2FormerPixelDecoder,
        predictor: MultiScaleMaskedTransformerDecoder,
    ):
        super().__init__()

        
        self.pixel_decoder = pixel_decoder
        self.predictor = predictor
        self.num_classes = num_classes

    def forward_features(self, features, mask=None):
        return self.layers(features, mask)

    def forward(self, features, mask=None):
        output = self.forward_features(features, mask)
        return output

    def predict(self, features, rescale_size, mask=None):
        output = self.forward_features(features, mask)
        output["pred_masks"] = F.interpolate(
            output["pred_masks"],
            size=rescale_size,
            mode="trilinear",
            align_corners=False,
        )
        return output

    def layers(self, features, mask=None):
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features)
        predictions = self.predictor(multi_scale_features, mask_features, mask)
        return predictions
