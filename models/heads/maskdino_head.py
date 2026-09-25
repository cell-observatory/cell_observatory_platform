"""
Adapted from:
https://github.com/IDEA-Research/MaskDINO/maskdino/modeling/meta_arch/maskdino_head.py#L4
"""

from torch import nn


class MaskDINOHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pixel_decoders: nn.Module,
        decoders: nn.Module,
    ):
        """
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            ignore_value: category id to be ignored during training.
            decoder: the transformer decoder that makes prediction
        """
        super().__init__()
        self.num_classes = num_classes
        self.decoder = decoders
        self.pixel_decoder = pixel_decoders

    def forward(self, features, mask=None, targets=None, predict_mask: bool = True):
        """Run the pixel decoder + transformer decoder.

        Args:
            features: backbone feature pyramid (dict).
            mask: optional padding mask (currently unsupported in 3D).
            targets: ground-truth dicts (training/denoising).
            predict_mask: forwarded to ``MaskDINODecoder.forward``. Set to
                ``False`` at inference when the caller drives chunked mask
                materialization itself (skips the last-layer einsum).
        """
        if mask is not None:
            # we currently do not properly ensure that masks are sorted in the correct
            # coarse to fine order anywhere so better to raise an error for now since we 
            # genereally do not use masks anyways
            raise NotImplementedError("3D segmentation not supported with input mask yet.")

        mask_features, transformer_encoder_features, \
            multi_scale_features = self.pixel_decoder.forward_features(features, mask)
        predictions, denoise_predictions = self.decoder(
            multi_scale_features,
            mask_features,
            mask,
            targets=targets,
            predict_mask=predict_mask,
        )
        return predictions, denoise_predictions