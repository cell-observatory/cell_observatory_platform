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
from cell_observatory_platform.utils.shape_format import get_spatial_shape
from cell_observatory_platform.models.meta_arch import utils as mo
from cell_observatory_platform.inference.utils import reduce_queries_to_semantic_map


class Mask2Former(nn.Module):
    def __init__(self,
        backbone: Mask2FormerBackbone,
        segmentation_head: Mask2FormerHead,
        matcher: Mask2FormerHungarianMatcher,
        criterion: Mask2FormerSetLoss,
        input_shape: tuple,
        input_fmt: str,
        num_classes: int,
        topk_queries: int,
        output_metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.segmentation_head = segmentation_head
        self.matcher = matcher
        self.criterion = criterion

        self.num_classes = num_classes
        self.topk_queries = topk_queries
        self.spatial_shape = get_spatial_shape(input_shape, input_fmt)
        # Inference contract (see meta_arch/utils.py). masks_labelmap is the saved
        # semantic artifact (argmax+1 / bg-0). pred_masks is the soft probability
        # map (kept float16 so the inferencer cast doesn't truncate to 0); kind=dense
        # so it routes to the save path only if a config lists it.
        self.output_metadata = mo.output_metadata(
            masks_labelmap=mo.semantic_map(self.spatial_shape),          # (Z,Y,X,1) uint16
            pred_masks={
                "shape": (*self.spatial_shape, self.num_classes),
                "dtype": "float16",
                "kind": "dense",
            },
            pred_classes={
                "shape": (self.num_classes,),
                "dtype": "float32",
                "kind": "scores",
            },
        )
        if output_metadata is not None:
            self.output_metadata.merge_with(output_metadata)

    @torch.jit.ignore
    def get_output_metadata(self):
        return self.output_metadata

    def _init_model_weights(self, buffer_device: str | None = None):
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

    def inference_step(self, data_sample: dict, rescale_size: Optional[tuple] = None):
        """
        Forward pass with pred_masks interpolated to target resolution (e.g. original input size).
        Use this for inference/eval when you need masks at 128×256×512 or other full resolution.
        """
        features_dict = self.backbone(data_sample)
        # NOTE(perf/OOM): segmentation_head.predict() interpolates ALL Q queries at full
        # (B,Q,D,H,W) resolution before topk and will OOM on large volumes; see MaskMaterializer
        # in models/meta_arch/maskdino.py for the chunked-materialization solution.
        # NOTE: Remove this in the Inference refactor -- this should be handled by the InferenceWorker or the PostProcessor
        if rescale_size is None:
            rescale_size = self.spatial_shape
        output = self.segmentation_head.predict(features_dict, rescale_size=rescale_size)

        # Convert masks to semantic map where each binary mask is replaced with the class index 
        # and all masks are flattened into a single channel-last tensor
        # classes are reduced to the average probability of the top k queries for each class
        masks_reduced, classes_reduced = reduce_queries_to_semantic_map(
            pred_masks=output["pred_masks"],
            pred_logits=output["pred_logits"],
            num_classes=self.num_classes,
            topk_per_image=self.topk_queries,
        )
        # Collapse the per-class channel to a semantic label map (class+1 / bg-0).
        # masks_reduced is channels-last and already in PROBABILITY space (the
        # sigmoid lives in reduce_queries_to_semantic_map), hence dim=-1 / 0.5.
        mask_labels = mo.collapse_to_semantic_map(  # [B,D,H,W,num_classes] -> [B,D,H,W,1]
            masks_reduced, threshold=0.5, dim=-1,
        )

        # NOTE: class_labels was a duplicate of pred_classes (both classes_reduced)
        # and was declared uint16, which truncated the float class probabilities to
        # 0 on the dtype cast. Dropped; consumers read pred_classes.
        final_output = {
            "pred_masks": masks_reduced,
            "pred_classes": classes_reduced,
            "masks_labelmap": mask_labels,
        }
        return final_output

    @torch.no_grad()
    def evaluate_step(
        self, data_sample: dict, rescale_size: Optional[tuple] = None
    ) -> List[Dict[str, torch.Tensor]]:
        """Semantic-segmentation eval entrypoint: per-image ``(D, H, W)`` label maps.

        Reuses :meth:`inference_step` (which already reduces queries to an argmax
        semantic map using the ``class + 1`` / background ``0`` convention),
        drops the trailing singleton channel, and returns a per-sample list of
        dicts (``{"labelmap": (D, H, W) long}``) so
        :class:`SemanticSegmentationEvaluator` can pair each predicted label map
        with its ground-truth label map. The trainer dispatches here because the
        evaluator sets ``predict_method = "evaluate_step"``.
        """
        out = self.inference_step(data_sample, rescale_size=rescale_size)
        lm = out["masks_labelmap"]  # (B, D, H, W, 1) uint16 -- shape guaranteed by the helper
        return [{"labelmap": lm[b, ..., 0].long()} for b in range(lm.shape[0])]

    
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
    ignore = {"_target_", "BUILD", "name"}
    if extra_ignores:
        ignore.update(extra_ignores)
    return {k: v for k, v in cfg.items() if k not in ignore}


from cell_observatory_platform.utils.registry import REGISTRY


@REGISTRY.register("model", "mask2former")
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
    adapter_cfg = model_cfg.get("adapter_args", None)
    backbone = REGISTRY.build("backbone", bw_cfg.name, bw_cfg, adapter_args=adapter_cfg)

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

    decoder_num_layers = decoder_cfg["decoder_num_layers"]

    # ----------------------------------------------------
    # 5) Segmentation head
    # ----------------------------------------------------

    segmentation_head = Mask2FormerHead(
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
        input_shape=model_cfg.input_shape,
        input_fmt=model_cfg.input_fmt,
        topk_queries=model_cfg.topk_queries,
        num_classes=model_cfg.num_classes,
    )