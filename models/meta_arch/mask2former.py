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
        semantic_reduction: str = "canonical",
    ):
        super().__init__()

        self.backbone = backbone
        self.segmentation_head = segmentation_head
        self.matcher = matcher
        self.criterion = criterion

        self.num_classes = num_classes
        self.topk_queries = topk_queries
        self.semantic_reduction = semantic_reduction
        self.spatial_shape = get_spatial_shape(input_shape, input_fmt)
        # The canonical reduction adds a leading no-object/background channel; the
        # legacy top-k/max reduction does not.
        _pred_mask_channels = (
            self.num_classes + 1 if semantic_reduction == "canonical" else self.num_classes
        )
        # Inference contract (see meta_arch/utils.py). masks_labelmap is the saved
        # semantic artifact (argmax+1 / bg-0). pred_masks is the soft probability
        # map (kept float16 so the inferencer cast doesn't truncate to 0); kind=dense
        # so it routes to the save path only if a config lists it.
        self.output_metadata = mo.output_metadata(
            masks_labelmap=mo.semantic_map(self.spatial_shape),          # (Z,Y,X,1) uint16
            pred_masks={
                "shape": (*self.spatial_shape, _pred_mask_channels),
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

        self._init_model_weights()

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

        losses = self.criterion(outputs, data_sample["metainfo"]["targets"])
        
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
        # NOTE: Remove this in the Inference refactor -- this should be handled by the InferenceWorker or the PostProcessor
        if rescale_size is None:
            rescale_size = self.spatial_shape
        # Low-res query masks from the head; the streaming reducer below
        # upsamples per query chunk so the full (B, Q, D, H, W) volume — ~6.7 GB
        # fp32 at Q=100, 256³ — is never materialized.
        output = self.segmentation_head.predict(features_dict)

        # Convert masks to semantic map where each binary mask is replaced with the class index
        # and all masks are flattened into a single channel-last tensor
        # classes are reduced to the average probability of the top k queries for each class
        # Reduce queries to a dense per-class map, then collapse to a label map. The
        # two reductions derive background differently, so each pairs with its own
        # collapse: canonical carries an explicit no-object channel at index 0 (plain
        # argmax); topk_max returns probabilities with no background channel and needs
        # a threshold. See reduce_queries_to_semantic_map.
        masks_reduced, classes_reduced = self._reduce_queries_streaming(
            pred_masks=output["pred_masks"],
            pred_logits=output["pred_logits"],
            rescale_size=tuple(rescale_size),
        )
        if self.semantic_reduction == "canonical":
            mask_labels = masks_reduced.argmax(dim=-1).to(torch.uint16).unsqueeze(-1)
        else:
            mask_labels = mo.collapse_to_semantic_map(masks_reduced, threshold=0.5, dim=-1)

        # NOTE: class_labels was a duplicate of pred_classes (both classes_reduced)
        # and was declared uint16, which truncated the float class probabilities to
        # 0 on the dtype cast. Dropped; consumers read pred_classes.
        final_output = {
            "pred_masks": masks_reduced,
            "pred_classes": classes_reduced,
            "masks_labelmap": mask_labels,
        }
        return final_output

    def _reduce_queries_streaming(
        self,
        pred_masks: torch.Tensor,
        pred_logits: torch.Tensor,
        rescale_size: tuple,
        q_chunk: int = 8,
    ):
        """Streaming equivalent of ``F.interpolate(all Q) -> reduce_queries_to_semantic_map``.

        Upsampling, sigmoid, gather and per-query weighting are all per-query
        ops, and both reductions aggregate with sum (canonical / topk nc=1) or
        max (topk nc>1) — so the reduction streams over query chunks with peak
        memory ``q_chunk × volume`` instead of ``Q × volume``. Query selection
        and weights (topk path) depend only on ``pred_logits``, so they are
        computed once up front; unselected queries get weight 0, which cannot
        change a sum or a max over the (non-negative) weighted masks. Output
        contract identical to :func:`reduce_queries_to_semantic_map` applied to
        fully-materialized upsampled masks (up to float summation order).
        """
        B, Q = pred_masks.shape[:2]

        def _up(chunk: torch.Tensor) -> torch.Tensor:
            return F.interpolate(
                chunk, size=rescale_size, mode="trilinear", align_corners=False
            )

        if self.semantic_reduction == "canonical":
            probs = pred_logits.softmax(-1)                       # [B, Q, C+1]
            semseg = None
            for q0 in range(0, Q, q_chunk):
                m = _up(pred_masks[:, q0 : q0 + q_chunk]).sigmoid()
                part = torch.einsum("bqk,bqdhw->bdhwk", probs[:, q0 : q0 + q_chunk], m)
                semseg = part if semseg is None else semseg + part
            # Roll no-object to channel 0 (class+1 / bg-0 argmax convention).
            semseg = torch.cat([semseg[..., -1:], semseg[..., :-1]], dim=-1)
            average_probability = probs[..., :-1].mean(dim=1)     # [B, C]
            return semseg, average_probability

        # topk_max (legacy): per-class top-k selection from logits, weight 0
        # for unselected queries, then stream sum (nc==1) / max (nc>1).
        num_classes, topk = self.num_classes, self.topk_queries
        if num_classes == 1:
            probs = pred_logits.softmax(-1)[..., 0]               # [B, Q]
            topk_idx = probs.topk(k=min(topk, Q), dim=1).indices  # [B, K]
            topk_probs = probs.gather(1, topk_idx)                # [B, K]
            weights = torch.zeros_like(probs).scatter(1, topk_idx, topk_probs)
            semantic = None
            for q0 in range(0, Q, q_chunk):
                m = _up(pred_masks[:, q0 : q0 + q_chunk]).sigmoid()
                part = (weights[:, q0 : q0 + q_chunk].view(B, -1, 1, 1, 1) * m).sum(
                    1, keepdim=True
                )
                semantic = part if semantic is None else semantic + part
            semantic = semantic.permute(0, 2, 3, 4, 1)            # [B, D, H, W, 1]
            return semantic, topk_probs.mean(dim=1)               # [B, 1]

        class_probs = pred_logits.softmax(-1)[..., :-1]           # [B, Q, C]
        weights = torch.zeros_like(class_probs)
        avg_per_class = []
        for c in range(num_classes):
            topk_idx = class_probs[..., c].topk(k=min(topk, Q), dim=1).indices
            topk_probs = class_probs[..., c].gather(1, topk_idx)
            weights[..., c] = weights[..., c].scatter(1, topk_idx, topk_probs)
            avg_per_class.append(topk_probs.mean(dim=1))
        semantic = None
        for q0 in range(0, Q, q_chunk):
            m = _up(pred_masks[:, q0 : q0 + q_chunk]).sigmoid()   # [B, q, D, H, W]
            # [B, q, 1, 1, 1, C] * [B, q, D, H, W, 1] -> max over q
            part = (
                weights[:, q0 : q0 + q_chunk].view(B, m.shape[1], 1, 1, 1, -1)
                * m.unsqueeze(-1)
            ).amax(dim=1)                                          # [B, D, H, W, C]
            semantic = part if semantic is None else torch.maximum(semantic, part)
        return semantic, torch.stack(avg_per_class, dim=-1)       # [B, C]

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