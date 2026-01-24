from typing import Any, Dict, List, Literal, Mapping, Optional

import torch
from hydra.utils import get_method
from torch import nn
from torch.nn import functional as F

from cell_observatory_platform.data.structures import box_cxcyczwhd_to_xyzxyz
from cell_observatory_platform.models.heads.maskdino_decoder import MaskDINODecoder
from cell_observatory_platform.models.heads.maskdino_head import MaskDINOHead
from cell_observatory_platform.models.heads.pixel_decoders import MaskDINOEncoder
from cell_observatory_platform.models.layers.attention import RopeAttention
from cell_observatory_platform.models.layers.matchers import HungarianMatcher
from cell_observatory_platform.training.helpers import get_input_data, get_nparams_and_flops, init_weights
from cell_observatory_platform.training.losses import DETR_Set_Loss


class MaskDINO(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        segmentation_head: MaskDINOHead,
        matcher: HungarianMatcher,
        criterion: DETR_Set_Loss,
        num_queries: int,
        instance_segmentation_flag: bool,
        topk_per_image: int,
        focus_on_boxes: bool = False,
    ):
        super().__init__()

        self.backbone = backbone
        self.segmentation_head = segmentation_head
        
        self.matcher = matcher
        self.criterion = criterion

        self.num_queries = num_queries
        self.topk_per_image = topk_per_image
        self.instance_segmentation_flag = instance_segmentation_flag
        self.focus_on_boxes = focus_on_boxes

    def init_model_weights(self, buffer_device: str | None = None):
        # TODO: move model inits back into each model class
        # FIXME: add proper weight init logic for MaskDINO
        # init_weights(self, weight_init_type=self.weight_init_type)
        for mod in self.modules():
            if isinstance(mod, RopeAttention):
                mod.init_rope_parameters(device=buffer_device)

    @torch.jit.ignore
    def _get_nparams_and_flops(
        self, batch_size: int, device: Literal["cuda", "meta"] = "cuda", masking_ratio: float = 0.0
    ):
        if device == "cuda":
            # TODO: test this path more thoroughly
            with torch.cuda.device(device):
                input_shape = (batch_size, *self.input_shape)
                data_sample = get_input_data(
                    inputs=input_shape,
                    device="cuda",
                )
                seq_len = int(self.get_num_patches()) * (1 - masking_ratio)
                model_summary = get_nparams_and_flops(self, data_sample, seq_len)
                model_param_count, num_flops_per_token = (
                    model_summary["total_params"],
                    model_summary["training_flops"],
                )
        elif device == "meta":
            print(f"Warning: using 'meta' device for flops/nparams calculation is not yet supported.")
            return -1, -1
        else:
            # TODO: add support for meta device calculation for other backends
            raise ValueError(f"Unsupported device for flops/nparams calculation: {device}")

        return model_param_count, num_flops_per_token

    @staticmethod
    def adjust_loss_weight_dict(
        loss_weight_dict: dict,
        two_stage_flag: bool,
        denoise: bool,
        denoise_losses: List[str],
        decoder_num_layers: int,
    ) -> dict:
        """
        Expand a base loss_weight_dict (e.g. {"loss_ce": 4., "loss_mask": 5., ...})
        to include:
          - intermediate head losses:      k + "_intermediate"        (if two_stage_flag)
          - denoising losses:              k + "_denoise"             (driven by denoise_losses groups)
          - aux decoder layer losses:      k + f"_{i}"                for i in [0, dec_layers)
          - aux denoising decoder losses:  k + f"_denoise_{i}"
        """
        weight_dict = dict(loss_weight_dict)

        # mapping from group name -> scalar loss keys that group produces
        group_to_keys = {
            "labels": ["loss_ce"],
            "boxes": ["loss_bbox", "loss_giou"],
            "masks": ["loss_mask", "loss_dice"],
        }

        # 1. Denoising: base k -> k + "_denoise" for selected groups
        if denoise:
            for group in denoise_losses:
                for k in group_to_keys.get(group, []):
                    if k in loss_weight_dict:
                        weight_dict[f"{k}_denoise"] = loss_weight_dict[k]

        # 2. Two-stage: intermediate head k -> k + "_intermediate"
        if two_stage_flag:
            for base_key, v in loss_weight_dict.items():
                if base_key.endswith("_denoise"):
                    continue
                weight_dict[f"{base_key}_intermediate"] = v

        # 3. Deep supervision over decoder layers: k -> k + f"_{i}"
        #    This covers both main and *_denoise keys
        if decoder_num_layers > 0:
            current_items = list(weight_dict.items())
            aux_weight_dict = {}
            for i in range(decoder_num_layers):
                for k, v in current_items:
                    aux_weight_dict[f"{k}_{i}"] = v
            weight_dict.update(aux_weight_dict)

        return weight_dict

    def forward(self, data_sample: dict):
        features_dict = self.backbone(data_sample)

        outputs, denoise_predictions = self.segmentation_head(
            features_dict, targets=data_sample["metainfo"]["targets"][0]
        )

        # bipartite matching-based loss
        losses = self.criterion(outputs, data_sample["metainfo"]["targets"][0], denoise_predictions)

        # for loss in list(losses.keys()):
        #     if loss in self.criterion.loss_weight_dict:
        #         losses[loss] *= self.criterion.loss_weight_dict[loss]
        #     else:
        #         # remove this loss if not specified in loss_weight_dict
        #         losses.pop(loss)

        losses["step_loss"] = sum(
            losses[k] * self.criterion.loss_weight_dict[k] for k in losses.keys() if k in self.criterion.loss_weight_dict
        )

        return losses, outputs

    def predict(self, data_sample: dict):
        features_dict = self.backbone(data_sample)

        outputs, _ = self.segmentation_head(features_dict, targets=None)
        predicted_labels, predicted_boxes, predicted_masks = [
            outputs[key] for key in ("pred_logits", "pred_boxes", "pred_masks")
        ]

        image_sizes = [tuple(int(x) for x in img_size.tolist()) for img_size in data_sample["metainfo"]["image_sizes"]]
        orig_image_sizes = [tuple(int(x) for x in orig_img_size.tolist()) for orig_img_size in data_sample["metainfo"]["orig_image_sizes"]]

        # upsample masks to original image size
        predicted_masks = F.interpolate(
            predicted_masks,
            # see data/datasets/pretrain_dataset_ray.py for details on image_sizes
            size=image_sizes[0],
            mode="trilinear",
            align_corners=False,
        )

        del outputs

        predictions = []
        for predicted_label, predicted_mask, predicted_box, image_size_pad, orig_image_size in zip(
            predicted_labels,
            predicted_masks,
            predicted_boxes,
            image_sizes,
            orig_image_sizes,
        ):
            # padded size (divisible by 32)
            depth, height, width = [
                new_dim / image_dim_pad * orig_dim
                for new_dim, image_dim_pad, orig_dim in zip(
                    predicted_mask.shape[-3:],  # (new_d, new_h, new_w)
                    image_size_pad,  # (orig_d, orig_h, orig_w)
                    orig_image_size,  # (orig_d, orig_h, orig_w)
                )
            ]
            # scale postprocess boxes to original image size
            predicted_box = self.box_postprocess(predicted_box, depth, height, width)

            instance_predictions = self._predict(predicted_label, predicted_mask, predicted_box)
            predictions.append(instance_predictions)

        return predictions

    def _predict(self, predicted_labels, predicted_masks, predicted_boxes):
        # (num_queries, num_classes) -> (num_queries, num_classes)
        predicted_labels = predicted_labels.sigmoid()

        # (num_queries, num_classes) -> (num_queries * num_classes,) -> Tuple(topk predicted labels, indices)
        predicted_labels_topk, topk_indices = predicted_labels.flatten(0, 1).topk(self.topk_per_image, sorted=False)

        # recover which query (0...Q-1) each top-K came from
        # flattened index is q*C + c => integer-dividing by C retrieves q
        topk_query_indices = topk_indices // self.segmentation_head.num_classes

        predicted_masks = predicted_masks[topk_query_indices]

        instance_predictions = {}
        # predicted masks pre-sigmoid (0.5 threshold)
        instance_predictions["masks"] = (predicted_masks > 0).float()
        instance_predictions["boxes"] = predicted_boxes[topk_query_indices]

        # average mask confidence inside each mask
        predicted_masks_flattened = instance_predictions["masks"].flatten(1)
        predicted_masks_sigmoid_flattened = predicted_masks.sigmoid().flatten(1)
        # keep probs only inside the predicted mask (all pixels > 0 times their prob)
        mask_confidence_score = (predicted_masks_sigmoid_flattened * predicted_masks_flattened).sum(1) / (
            predicted_masks_flattened.sum(1) + 1e-6
        )

        if self.focus_on_boxes:
            instance_predictions["predicted_labels"] = predicted_labels_topk
        else:
            instance_predictions["predicted_labels"] = predicted_labels_topk * mask_confidence_score

        return instance_predictions

    def box_postprocess(self, bboxes, depth, height, width):
        # postprocess box height and width
        scale_factor = torch.tensor(
            [
                width,
                height,
                depth,
                width,
                height,
                depth,
            ]
        )
        scale_factor = scale_factor.to(bboxes)
        bboxes = box_cxcyczwhd_to_xyzxyz(bboxes)
        bboxes = bboxes * scale_factor
        return bboxes


def _extract_kwargs(cfg: Mapping[str, Any], extra_ignores: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Drop Hydra/meta keys like _target_, BUILD, and any explicitly ignored keys.
    """
    ignore = {"_target_", "BUILD"}
    if extra_ignores:
        ignore.update(extra_ignores)
    return {k: v for k, v in cfg.items() if k not in ignore}


def BUILD(cfg: Mapping[str, Any]) -> MaskDINO:
    """
    Factory for MaskDINO using nested cfg dicts.

    Expected keys in `cfg`:
      - backbone_args:       Hydra config to build the backbone
      - adapter_args:        (optional) Hydra config / kwargs for EncoderAdapter
      - pixel_decoder_args:  kwargs for MaskDINOEncoder
      - decoder_args:        kwargs for MaskDINODecoder
      - matcher_args:        kwargs for HungarianMatcher
      - criterion_args:      kwargs for DETR_Set_Loss + base loss_weight_dict, denoise, etc.
      - topk_per_image:      int
      - instance_segmentation_flag: bool
      - focus_on_boxes:      bool (optional)
    """

    model_cfg = cfg.models.meta_arch.maskdino

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
    # 3) Pixel decoder (MaskDINOEncoder)
    # ----------------------------------------------------

    pixel_decoder_cfg = model_cfg["pixel_decoder_args"]
    # TODO: move to BUILD function for MaskDINOEncoder
    pixel_decoder = MaskDINOEncoder(**_extract_kwargs(pixel_decoder_cfg))

    # ----------------------------------------------------
    # 4) Transformer decoder (MaskDINODecoder)
    # ----------------------------------------------------

    decoder_cfg = model_cfg["decoder_args"]
    # TODO: move to BUILD function for MaskDINODecoder
    decoder = MaskDINODecoder(**_extract_kwargs(decoder_cfg))

    num_classes = decoder_cfg["num_classes"]
    num_queries = decoder_cfg["num_queries"]
    decoder_num_layers = decoder_cfg["decoder_num_layers"]
    two_stage_flag = decoder_cfg["two_stage_flag"]

    # ----------------------------------------------------
    # 5) Segmentation head
    # ----------------------------------------------------

    segmentation_head = MaskDINOHead(
        num_classes=num_classes,
        pixel_decoders=pixel_decoder,
        decoders=decoder,
    )

    # ----------------------------------------------------
    # 6) Matcher
    # ----------------------------------------------------

    matcher_cfg = model_cfg["matcher_args"]
    # TODO: move to BUILD function for HungarianMatcher
    matcher = HungarianMatcher(**_extract_kwargs(matcher_cfg))

    # ----------------------------------------------------
    # 7) Criterion: adjust loss weights, then build DETR_Set_Loss
    # ----------------------------------------------------

    criterion_cfg = model_cfg["criterion_args"]

    base_loss_weight_dict = criterion_cfg["loss_weight_dict"]
    denoise = criterion_cfg["denoise"]
    denoise_losses = criterion_cfg["denoise_losses"]

    adjusted_loss_weight_dict = MaskDINO.adjust_loss_weight_dict(
        loss_weight_dict=base_loss_weight_dict,
        two_stage_flag=two_stage_flag,
        denoise=denoise,
        denoise_losses=denoise_losses,
        decoder_num_layers=decoder_num_layers,
    )

    # TODO: move into BUILD function for DETR_Set_Loss
    criterion = DETR_Set_Loss(
        num_classes=criterion_cfg["num_classes"],
        matcher=matcher,
        loss_weight_dict=adjusted_loss_weight_dict,
        no_object_loss_weight=criterion_cfg["no_object_loss_weight"],
        losses=criterion_cfg["losses"],
        num_points=criterion_cfg["num_points"],
        oversample_ratio=criterion_cfg["oversample_ratio"],
        importance_sample_ratio=criterion_cfg["importance_sample_ratio"],
        denoise=criterion_cfg["denoise"],
        with_segmentation=criterion_cfg["with_segmentation"],
        denoise_losses=criterion_cfg["denoise_losses"],
        semantic_ce_loss=criterion_cfg["semantic_ce_loss"],
        focal_alpha=criterion_cfg["focal_alpha"],
    )

    # ----------------------------------------------------
    # 8) Final MaskDINO module
    # ----------------------------------------------------

    instance_segmentation_flag = model_cfg.get("instance_segmentation_flag", True)
    topk_per_image = model_cfg["topk_per_image"]
    focus_on_boxes = model_cfg.get("focus_on_boxes", False)

    return MaskDINO(
        backbone=backbone,
        segmentation_head=segmentation_head,
        matcher=matcher,
        criterion=criterion,
        num_queries=num_queries,
        instance_segmentation_flag=instance_segmentation_flag,
        topk_per_image=topk_per_image,
        focus_on_boxes=focus_on_boxes,
    )