from typing import Any, Dict, List, Literal, Mapping, Optional, Iterator, Sequence, Tuple

import torch
from hydra.utils import get_method
from collections import OrderedDict
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from cell_observatory_platform.data.structures import box_cxcyczwhd_to_xyzxyz
from cell_observatory_platform.models.heads.maskdino_decoder import MaskDINODecoder
from cell_observatory_platform.models.heads.maskdino_head import MaskDINOHead
from cell_observatory_platform.models.heads.pixel_decoders import MaskDINOEncoder
from cell_observatory_platform.models.layers.attention import RopeAttention
from cell_observatory_platform.models.layers.maskmaterializer import MaskMaterializer
from cell_observatory_platform.models.layers.matchers import HungarianMatcher
from cell_observatory_platform.models.meta_arch import utils as mo
from cell_observatory_platform.training.helpers import get_input_data, get_nparams_and_flops
from cell_observatory_platform.training.losses import DETR_Set_Loss
from cell_observatory_platform.utils.shape_format import get_spatial_shape


class MaskDINO(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        segmentation_head: MaskDINOHead,
        matcher: HungarianMatcher,
        criterion: DETR_Set_Loss,
        input_shape: tuple,
        input_fmt: str,
        num_queries: int,
        instance_segmentation_flag: bool,
        topk_per_image: int,
        focus_on_boxes: bool = False,
        buffer_device: str = "cuda",
        output_metadata: Optional[Dict[str, Any]] = None,
        mask_chunk_size: int = 8,
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
        # Number of queries to materialize at full resolution simultaneously in
        # `predict` / `predict_for_eval`. Tune down on tighter vRAM budgets.
        self.mask_chunk_size = int(mask_chunk_size)
        self.input_fmt = input_fmt
        spatial_shape = get_spatial_shape(input_shape, input_fmt)
        self.spatial_shape = spatial_shape
        # Inference contract (see meta_arch/utils.py). inference_step collapses the
        # topk queries into ONE instance label map (channels-last ZYXC) plus
        # per-instance boxes/scores at the processed scale. NOTE: the "labels" key
        # actually carries the per-instance CONFIDENCE score (kept for the save
        # contract), hence the scores kind.
        self.output_metadata = mo.output_metadata(
            masks  = mo.instance_label_map(spatial_shape),   # (Z,Y,X,1) uint16
            boxes  = mo.boxes(topk_per_image),               # (topk,6) xyzxyz, processed scale
            labels = mo.scores(topk_per_image),              # (topk,) per-instance confidence
        )
        if output_metadata is not None:
            self.output_metadata.merge_with(output_metadata)

        self._init_model_weights(buffer_device=buffer_device)

    def _init_model_weights(self, buffer_device: str):
        # Weight init for MaskDINO submodules happens inside their respective
        # __init__ methods (if at all), following the reference implementation:
        # - MaskDINOEncoder
        # - MaskDINOHead
        # - MaskDINODecoder
        # - MaskDinoBackbone (implicitly via backbone build)
        # - TransformerDecoder
        for mod in self.modules():
            if isinstance(mod, RopeAttention):
                mod.init_rope_parameters(device=buffer_device)

    def get_param_groups(self, weight_decay: float, **kwargs) -> list[dict]:
        """
        Get param groups with decay/no-decay split.
        TODO: consider more options such as layer-wise decay, etc.
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or "bias" in name or "level_embed" in name:
                no_decay.append(p)
            else:
                decay.append(p)

        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # TODO: implement for each meta_arch
    # @torch.jit.ignore
    # def _get_nparams_and_flops(
    #     self, batch_size: int, device: Literal["cuda", "meta"] = "cuda", masking_ratio: float = 0.0
    # ):
    #     if device == "cuda":
    #         # TODO: test this path more thoroughly
    #         with torch.cuda.device(device):
    #             input_shape = (batch_size, *self.input_shape)
    #             data_sample = get_input_data(
    #                 inputs=input_shape,
    #                 device="cuda",
    #             )
    #             seq_len = int(self.get_num_patches()) * (1 - masking_ratio)
    #             model_summary = get_nparams_and_flops(self, data_sample, seq_len)
    #             model_param_count, num_flops_per_token = (
    #                 model_summary["total_params"],
    #                 model_summary["training_flops"],
    #             )
    #     elif device == "meta":
    #         print(f"Warning: using 'meta' device for flops/nparams calculation is not yet supported.")
    #         return -1, -1
    #     else:
    #         # TODO: add support for meta device calculation for other backends
    #         raise ValueError(f"Unsupported device for flops/nparams calculation: {device}")

    #     return model_param_count, num_flops_per_token

    @torch.jit.ignore
    def get_output_metadata(self):
        return self.output_metadata

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
            features_dict, targets=data_sample["metainfo"]["targets"]
        )

        # bipartite matching-based loss
        losses = self.criterion(outputs, data_sample["metainfo"]["targets"], denoise_predictions)

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

    def inference_step(self, data_sample: dict):
        """Run inference and return per-sample collapsed instance predictions.

        Mask materialization is performed in chunks of ``self.mask_chunk_size``
        queries to keep peak vRAM bounded; see :class:`MaskMaterializer` and
        :meth:`evaluate_step` for the underlying primitives. The returned
        dict has the following keys / shapes:

            * ``boxes``: ``(B, topk, 6)`` in xyzxyz at the PROCESSED scale (the
              resized volume the model saw). Tile-mode inference rescales these
              to the original tile frame in the inferencer post-processor.
            * ``labels``: ``(B, topk)`` per-instance scores (optionally
              re-weighted by mean mask probability when ``focus_on_boxes`` is
              False).
            * ``masks``: ``(B, *processed_size)`` uint16 instance label maps at
              the resized resolution; the inferencer restores them to the
              original tile size for saving.
        """
        intermediates = self.evaluate_step(data_sample)

        per_sample_preds = []
        for sample in intermediates:
            per_sample_preds.append(self._collapse_sample_chunked(sample))

        batched = {
            key: torch.stack([s[key] for s in per_sample_preds], dim=0)
            for key in per_sample_preds[0]
        }
        # masks is a (B, Z, Y, X) instance label map. Add a trailing C=1 so it
        # conforms to the channels-last ZYXC contract used by the save path
        # (io.py ndim==len(data_format) gate); kind=instance_label_map.
        if "masks" in batched:
            batched["masks"] = batched["masks"].unsqueeze(-1)
        return batched

    def evaluate_step(self, data_sample: dict) -> List[Dict[str, Any]]:
        """Return per-sample inference intermediates without collapsing masks.

        Skips the global low-res einsum (``predict_mask=False``) so the caller
        owns mask materialization (typically via :class:`MaskMaterializer`
        chunked over the topk query indices).

        Each returned per-sample dict contains:
            * ``mask_embeddings``: ``(Q, mask_dim)``.
            * ``pixel_decoder_output``: ``(mask_dim, d, h, w)``.
            * ``topk_query_indices``: ``(topk,)`` long, into ``Q``.
            * ``topk_class_scores``: ``(topk,)`` float — max-class sigmoid
              score per topk pick (the COCO-style class-aware ranking score).
            * ``topk_class_ids``: ``(topk,)`` long — the predicted class id
              corresponding to each topk pick.
            * ``boxes``: ``(topk, 6)`` xyzxyz at the PROCESSED scale.
            * ``eval_frame_size``: ``(D, H, W)`` ints -- the PROCESSED size the
              model saw (kept under this key for the downstream mask
              materializer / evaluator contract), NOT the original tile size.
        """
        features_dict = self.backbone(data_sample)
        outputs, _ = self.segmentation_head(
            features_dict, targets=None, predict_mask=False
        )
        pred_logits = outputs["pred_logits"]            # (B, Q, num_classes)
        pred_boxes = outputs["pred_boxes"]              # (B, Q, 6) cxcyczwhd
        mask_embeddings_b = outputs["mask_embeddings"]      # (B, Q, mask_dim)
        pixel_decoder_output_b = outputs["pixel_decoder_output"]  # (B, mask_dim, d, h, w)

        # Evaluate at the PROCESSED resolution -- the spatial size of the tensor
        # the model actually saw (after any Resize/crop in the preprocessor) --
        # NOT metainfo["orig_image_sizes"] (the original tile size). GT rides the
        # data tensor and was resized in lockstep, so both live here; restoring
        # to the original tile size is an inference-only concern. Per-sample sizes
        # are identical across the batch (the data tensor is a single dense
        # buffer), so derive one processed (D, H, W) and reuse it.
        processed_size = get_spatial_shape(
            tuple(int(s) for s in data_sample["data_tensor"].shape[1:]),
            self.input_fmt,
        )
        batch_size = pred_logits.shape[0]
        eval_image_sizes = [processed_size] * batch_size

        num_classes = self.segmentation_head.num_classes
        per_sample: List[Dict[str, Any]] = []
        for b, eval_frame_size in enumerate(eval_image_sizes):
            scores = pred_logits[b].sigmoid()                                 # (Q, C)
            topk_scores, topk_idx = scores.flatten(0, 1).topk(
                self.topk_per_image, sorted=False
            )
            # query/class recovery from the (Q*C,) flat index:
            #   q = idx // C, c = idx % C
            topk_query_idx = topk_idx // num_classes                          # (topk,)
            topk_class_id = topk_idx % num_classes                            # (topk,)

            depth, height, width = eval_frame_size
            boxes_topk = self.box_postprocess(
                pred_boxes[b][topk_query_idx], depth, height, width
            )

            per_sample.append({
                "mask_source": "query",
                "mask_embeddings": mask_embeddings_b[b],
                "pixel_decoder_output": pixel_decoder_output_b[b],
                "topk_query_indices": topk_query_idx,
                "topk_class_scores": topk_scores,
                "topk_class_ids": topk_class_id,
                "boxes": boxes_topk,
                "eval_frame_size": eval_frame_size,
            })
        return per_sample

    def _collapse_sample_chunked(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Build a single-image instance prediction dict via chunked materialization.

        Mirrors the legacy :meth:`_predict` semantics:

            * Per-instance confidence score = ``predicted_labels_topk * mask_confidence``
              (unless ``focus_on_boxes`` is True), where ``mask_confidence`` is
              the mean sigmoid probability inside the binarized mask.
            * Mask outputs are collapsed into a single uint16 label map with
              instances 1 .. K assigned in ascending score order, so the
              highest-confidence instance "wins" overlapping voxels.

        ONE pass through the chunked materializer: each chunk's mask confidence
        (and hence its final score) depends only on its own masks, so the
        higher-score-wins collapse can run in the same pass that computes
        confidence (peak memory stays ``mask_chunk_size * D*H*W``). Provisional
        per-voxel slot indices are remapped to ascending-score IDs afterwards,
        and ``boxes``/``labels`` are permuted by the SAME order, so returned row
        ``j-1`` always corresponds to label-map ID ``j``.
        """
        materializer = MaskMaterializer(
            mask_embeddings=sample["mask_embeddings"],
            pixel_decoder_output=sample["pixel_decoder_output"],
            target_size=sample["eval_frame_size"],
            chunk_size=self.mask_chunk_size,
        )
        topk_query_idx = sample["topk_query_indices"]
        topk_scores = sample["topk_class_scores"]

        device = topk_query_idx.device
        K = int(topk_query_idx.numel())
        mask_confidence = torch.zeros(K, device=device, dtype=torch.float32)

        spatial = tuple(sample["eval_frame_size"])
        # Winner-take-all state: best score per voxel, and the 1-based topk SLOT
        # (original topk order) that claimed it. Higher-score-wins in one pass is
        # equivalent to the old "stamp in ascending-score order, later overwrite
        # wins" (ties: the first-seen instance keeps the voxel).
        best = torch.zeros(spatial, dtype=torch.float32, device=device)
        slot_map = torch.zeros(spatial, dtype=torch.int32, device=device)

        # ONE materialize pass: confidence + collapse per chunk.
        slot = 0
        for chunk_idx, mask_logits in materializer.chunks(topk_query_idx):
            k = int(chunk_idx.numel())
            sigm = mask_logits.sigmoid()
            binary = (mask_logits > 0)
            sigm_inside = (sigm * binary.to(sigm.dtype)).flatten(1).sum(dim=1)
            count_inside = binary.flatten(1).sum(dim=1).to(torch.float32)
            conf = sigm_inside.to(torch.float32) / torch.clamp(count_inside, min=1.0)
            mask_confidence[slot:slot + k] = conf

            chunk_scores = topk_scores[slot:slot + k].to(torch.float32)
            if not self.focus_on_boxes:
                chunk_scores = chunk_scores * conf

            for i in range(k):
                win = binary[i] & (chunk_scores[i] > best)
                best = torch.where(win, chunk_scores[i], best)
                slot_map[win] = slot + i + 1
            slot += k

        if self.focus_on_boxes:
            scores_per_instance = topk_scores.to(torch.float32)
        else:
            scores_per_instance = topk_scores.to(torch.float32) * mask_confidence

        # Label IDs 1..K in ascending-score order (old semantics: the last —
        # highest-score — stamp got the highest ID). Remap the provisional slot
        # indices to those IDs, and permute boxes/labels by the SAME order so
        # returned row j-1 corresponds to label-map ID j.
        order = scores_per_instance.argsort()  # ascending
        id_of_slot = torch.empty(K, dtype=torch.int32, device=device)
        id_of_slot[order] = torch.arange(1, K + 1, dtype=torch.int32, device=device)
        label_map = torch.where(
            slot_map > 0,
            id_of_slot[(slot_map.long() - 1).clamp(min=0)],
            slot_map,
        )

        return {
            "boxes": sample["boxes"][order],
            "labels": scores_per_instance[order],
            "masks": label_map.to(torch.uint16),
        }

    @staticmethod
    def collapse_instance_masks(
        binary_masks: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """Collapse per-instance binary masks into a single instance label map.

        Kept for backward compatibility with callers that already hold the
        materialized binary masks. New code paths should prefer
        :meth:`predict` / :meth:`_collapse_sample_chunked`, which avoid ever
        materializing all masks at once.

        Args:
            binary_masks: ``(K, *spatial)`` binary masks (one per instance).
            scores: ``(K,)`` confidence score for each instance.

        Returns:
            ``(*spatial)`` uint16 label map.  Background = 0, instances are
            labelled 1 .. N in ascending confidence order so that
            higher-scoring instances overwrite lower-scoring ones in
            overlapping regions.
        """
        label_map = torch.zeros(
            binary_masks.shape[1:], dtype=torch.int32, device=binary_masks.device
        )
        order = scores.argsort()
        for new_id, idx in enumerate(order, start=1):
            label_map[binary_masks[idx] > 0] = new_id
        return label_map.to(torch.uint16)

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
    ignore = {"_target_", "BUILD", "name"}
    if extra_ignores:
        ignore.update(extra_ignores)
    return {k: v for k, v in cfg.items() if k not in ignore}


from cell_observatory_platform.utils.registry import REGISTRY


@REGISTRY.register("model", "maskdino")
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
    adapter_cfg = model_cfg.get("adapter_args", None)
    backbone = REGISTRY.build("backbone", bw_cfg.name, bw_cfg, adapter_args=adapter_cfg)

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
    mask_chunk_size = model_cfg.get("mask_chunk_size", 8)

    return MaskDINO(
        backbone=backbone,
        segmentation_head=segmentation_head,
        matcher=matcher,
        criterion=criterion,
        num_queries=num_queries,
        instance_segmentation_flag=instance_segmentation_flag,
        topk_per_image=topk_per_image,
        focus_on_boxes=focus_on_boxes,
        mask_chunk_size=mask_chunk_size,
        input_shape=cfg.datasets.train_shape,
        input_fmt=cfg.dataset_layout_order,
    )