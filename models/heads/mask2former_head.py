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
        input_shape: Dict[str, Tuple[int]],
        input_dim: int = 3,
        hidden_dim: int = 2048,
        num_classes: int = 150,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        transformer_in_feature: str = "multi_scale_pixel_decoder",
        # decoder params
        decoder_num_feature_levels: int = 3,
        decoder_num_queries: int = 100,
        decoder_nheads: int = 16,
        decoder_dim_feedforward: int = 4096,
        decoder_layers: int = 9,
        decoder_pre_norm: bool = False,
        decoder_enforce_input_project: bool = False,
        # pixel decoder params
        pixel_decoder_transformer_dropout: float = 0.0,
        pixel_decoder_transformer_nheads: int = 8,
        pixel_decoder_transformer_dim_feedforward: int = 4096,
        pixel_decoder_transformer_enc_layers: int = 6,
        pixel_decoder_norm: str = "GroupNorm",
        pixel_decoder_transformer_in_features: Tuple[str] = ("1", "2", "3", "4"),
        pixel_decoder_common_stride: int = 4,
        use_deform_attention: bool = True,
    ):
        super().__init__()

        self.use_deform_attention = use_deform_attention

        orig_input_shape = input_shape
        input_shape = sorted(input_shape.items(), key=lambda x: x[1]["stride"])
        self.in_features = [k for k, _ in input_shape]

        self.ignore_value = ignore_value
        self.target_min_stride = pixel_decoder_common_stride
        self.loss_weight = loss_weight

        self.pixel_decoder = Mask2FormerPixelDecoder(
            input_shape=orig_input_shape,
            total_num_feature_levels=decoder_num_feature_levels,
            transformer_encoder_dropout=pixel_decoder_transformer_dropout,
            transformer_encoder_num_heads=pixel_decoder_transformer_nheads,
            transformer_encoder_dim_feedforward=pixel_decoder_transformer_dim_feedforward,
            transformer_encoder_layers=pixel_decoder_transformer_enc_layers,
            conv_dim=hidden_dim,
            mask_dim=hidden_dim,
            norm=pixel_decoder_norm,
            transformer_in_features=pixel_decoder_transformer_in_features,
            target_min_stride=pixel_decoder_common_stride,
            use_deform_attention=use_deform_attention,
        )
        self.predictor = MultiScaleMaskedTransformerDecoder(
            input_dim=input_dim,
            in_channels=hidden_dim,
            mask_classification=True,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=decoder_num_queries,
            decoder_nheads=decoder_nheads,
            dim_feedforward=decoder_dim_feedforward,
            decoder_layers=decoder_layers,
            decoder_pre_norm=decoder_pre_norm,
            mask_dim=hidden_dim,
            enforce_input_project=decoder_enforce_input_project,
            num_feature_levels=decoder_num_feature_levels,
        )

        self.transformer_in_feature = transformer_in_feature
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
