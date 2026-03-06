import os
from typing import Any, Dict, List, Literal, Mapping, Optional

import torch
from hydra.utils import get_method
from torch import nn
from torch.nn import functional as F
from omegaconf import DictConfig, OmegaConf

from cell_observatory_platform.data.structures import box_cxcyczwhd_to_xyzxyz
from cell_observatory_platform.models.backbones.mask2former_backbone import Mask2FormerBackbone
from cell_observatory_platform.models.heads.mask2former_decoder import MultiScaleMaskedTransformerDecoder
from cell_observatory_platform.models.heads.mask2former_head import Mask2FormerHead
from cell_observatory_platform.models.heads.pixel_decoders import Mask2FormerPixelDecoder
from cell_observatory_platform.models.layers.attention import RopeAttention
from cell_observatory_platform.models.layers.matchers import Mask2FormerHungarianMatcher
from cell_observatory_platform.training.helpers import get_input_data, get_nparams_and_flops
from cell_observatory_platform.training.losses import Mask2FormerSetLoss


class Mask2Former(nn.Module):
    def __init__(self,
        backbone: Mask2FormerBackbone,
        segmentation_head: Mask2FormerHead,
        matcher: Mask2FormerHungarianMatcher,
        criterion: Mask2FormerSetLoss,
    ):
        super().__init__()

        self.backbone = backbone
        self.segmentation_head = segmentation_head
        self.matcher = matcher
        self.criterion = criterion
        
    def init_model_weights(self, buffer_device: str | None = None):
        # TODO: move model inits back into each model class
        # FIXME: add proper weight init logic for Mask2Former
        # init_weights(self, weight_init_type=self.weight_init_type)
        for mod in self.modules():
            if isinstance(mod, RopeAttention):
                mod.init_rope_parameters(device=buffer_device)

    def forward(self, data_sample: dict):
        features_dict = self.backbone(data_sample)
        outputs = self.segmentation_head(features_dict)

        losses = self.criterion(outputs, data_sample["metainfo"]["targets"][0])
        
        # for k in list(losses.keys()):
        #     if k in self.criterion.weight_dict:
        #         losses[k] *= self.criterion.weight_dict[k]
        #     else:
        #         # remove this loss if not specified in `weight_dict`
        #         losses.pop(k)
        
        losses["step_loss"] = sum(
            losses[k] * self.criterion.loss_weight_dict[k] 
            for k in losses.keys() 
            if k in self.criterion.loss_weight_dict
        )

        return losses, outputs

    def predict(self, data_sample: dict, rescale_size: Optional[tuple] = None):
        """
        Forward pass with pred_masks interpolated to target resolution (e.g. original input size).
        Use this for inference/eval when you need masks at 128×256×512 or other full resolution.
        """
        features_dict = self.backbone(data_sample)
        # NOTE: Remove this in the Inference refactor -- this should be handled by the InferenceWorker or the PostProcessor
        # if rescale_size is None:
        #     # Default: original input spatial shape (Z, Y, X)
        #     t = data_sample["data_tensor"]
        #     rescale_size = tuple(t.shape[1:4])  # channels-last: (Z, Y, X)
        outputs = self.segmentation_head.predict(features_dict, rescale_size=rescale_size)
        return outputs
    
    
    @staticmethod
    def adjust_loss_weight_dict(
        loss_weight_dict: dict,
        decoder_num_layers: int,
    ) -> dict:
        """
        Expand a base loss_weight_dict (e.g. {"loss_ce": 4., "loss_mask": 5., ...})
        to include:
          - aux decoder layer losses:      k + f"_{i}"                for i in [0, dec_layers)
        """
        weight_dict = dict(loss_weight_dict)

        # mapping from group name -> scalar loss keys that group produces
        group_to_keys = {
            "labels": ["loss_ce"],
            "masks": ["loss_mask", "loss_dice"],
        }

        #    Deep supervision over decoder layers: k -> k + f"_{i}"
        if decoder_num_layers > 0:
            current_items = list(weight_dict.items())
            aux_weight_dict = {}
            for i in range(decoder_num_layers):
                for k, v in current_items:
                    aux_weight_dict[f"{k}_{i}"] = v
            weight_dict.update(aux_weight_dict)

        return weight_dict

def _extract_kwargs(cfg: Mapping[str, Any], extra_ignores: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Drop Hydra/meta keys like _target_, BUILD, and any explicitly ignored keys.
    """
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    ignore = {"_target_", "BUILD"}
    if extra_ignores:
        ignore.update(extra_ignores)
    return {k: v for k, v in cfg.items() if k not in ignore}


def BUILD(cfg: Mapping[str, Any]) -> Mask2Former:
    """
    Factory for Mask2Former using nested cfg dicts.

    Expected keys in `cfg`:
      - backbone_args:       Hydra config to build the backbone
      - segmentation_head_args:  kwargs for Mask2FormerHead
      - matcher_args:        kwargs for HungarianMatcher
      - criterion_args:      kwargs for Mask2FormerSetLoss + base loss_weight_dict, denoise, etc.
      - topk_per_image:      int
    """

    model_cfg = cfg.models.meta_arch.mask2former

    # ----------------------------------------------------
    # 1) Backbone
    # ----------------------------------------------------

    bw_cfg = model_cfg["backbone_wrapper_args"]
    build_backbone_wrapper = get_method(bw_cfg.BUILD)

    adapter_cfg = model_cfg.get("adapter_args", None)
    if adapter_cfg is not None:
        backbone = build_backbone_wrapper(bw_cfg, adapter_cfg)
    else:
        backbone = build_backbone_wrapper(bw_cfg, None)

    # ----------------------------------------------------
    # 3) Pixel decoder (Mask2FormerPixelDecoder)
    # ----------------------------------------------------

    pixel_decoder_cfg = model_cfg["pixel_decoder_args"]
    # TODO: move to BUILD function for Mask2FormerPixelDecoder
    pixel_decoder = Mask2FormerPixelDecoder(**_extract_kwargs(pixel_decoder_cfg))

    # ----------------------------------------------------
    # 4) Transformer decoder (MultiScaleMaskedTransformerDecoder)
    # ----------------------------------------------------

    decoder_cfg = model_cfg["decoder_args"]
    # TODO: move to BUILD function for MultiScaleMaskedTransformerDecoder
    # TODO: Rename the predictor to be consistent with the other models
    predictor = MultiScaleMaskedTransformerDecoder(**_extract_kwargs(decoder_cfg))

    num_classes = decoder_cfg["num_classes"]
    decoder_num_layers = decoder_cfg["decoder_num_layers"]

    # ----------------------------------------------------
    # 5) Segmentation head
    # ----------------------------------------------------

    segmentation_head = Mask2FormerHead(
        num_classes=num_classes,
        pixel_decoder=pixel_decoder,
        predictor=predictor,
    )

    # ----------------------------------------------------
    # 6) Matcher
    # ----------------------------------------------------

    matcher_cfg = model_cfg["matcher_args"]
    # TODO: move to BUILD function for Mask2FormerHungarianMatcher
    matcher = Mask2FormerHungarianMatcher(**_extract_kwargs(matcher_cfg))

    # ----------------------------------------------------
    # 7) Criterion: adjust loss weights, then build Mask2FormerSetLoss
    # ----------------------------------------------------

    criterion_cfg = model_cfg["criterion_args"]

    base_loss_weight_dict = criterion_cfg["loss_weight_dict"]

    adjusted_loss_weight_dict = Mask2Former.adjust_loss_weight_dict(
        loss_weight_dict=base_loss_weight_dict,
        decoder_num_layers=decoder_num_layers,
    )

    # TODO: move into BUILD function for Mask2FormerSetLoss
    criterion = Mask2FormerSetLoss(
        num_classes=criterion_cfg["num_classes"],
        matcher=matcher,
        loss_weight_dict=adjusted_loss_weight_dict,
        no_object_loss_weight=criterion_cfg["no_object_loss_weight"],
        losses=criterion_cfg["losses"],
        num_points=criterion_cfg["num_points"],
        oversample_ratio=criterion_cfg["oversample_ratio"],
        importance_sample_ratio=criterion_cfg["importance_sample_ratio"],
    )

    # ----------------------------------------------------
    # 8) Final Mask2Former module
    # ----------------------------------------------------

    return Mask2Former(
        backbone=backbone,
        segmentation_head=segmentation_head,
        matcher=matcher,
        criterion=criterion,
    )