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
from cell_observatory_platform.models.layers.matchers import HungarianMatcher
from cell_observatory_platform.training.helpers import get_input_data, get_nparams_and_flops
from cell_observatory_platform.training.losses import DETR_Set_Loss
from cell_observatory_platform.utils.shape_format import get_spatial_shape

class MaskMaterializer:
    """Chunked, per-image mask materializer for MaskDINO.

    Args:
        mask_embeddings: ``(Q, mask_dim)`` projected query embeddings for one
            image (last decoder layer). May be in any floating dtype.
        pixel_decoder_output: ``(mask_dim, d, h, w)`` finest pixel-decoder
            feature map for the same image.
        target_size: ``(D, H, W)`` spatial size to upsample masks to (typically
            the original input image size).
        chunk_size: maximum number of queries to materialize simultaneously.
        upsample_dtype: dtype used inside the trilinear interpolate. Defaults
            to ``torch.float32`` because trilinear in fp16 has noticeable
            artefacts on small structures.

    Chunked mask materialization for MaskDINO-style detectors.

    MaskDINO produces ``num_queries`` object queries that, at the final decoder
    layer, project to a ``mask_dim``-vector each. To turn each query into a 3D
    binary mask the model evaluates::

        low_res_mask[b, q] = einsum("bqc, bcdhw->bqdhw", mask_embeddings, pixel_decoder_output)
        full_res_mask[b, q] = F.interpolate(low_res_mask[b, q], size=(D, H, W), mode="trilinear")

    The interpolation step is the memory bottleneck for large 3D volumes: at
    ``D = H = W = 256`` and ``topk_per_image = 100``, a single batch element costs
    ``100 * 256**3 * 4 B = 6.7 GB`` in fp32 (3.3 GB in fp16) just for the mask
    logits — before binarization, IoU computation, etc.

    ``MaskMaterializer`` offers a single entry point — ``chunks(query_indices)`` —
    that yields the upsampled mask logits for ``chunk_size`` queries at a time.
    Callers (``MaskDINO.predict``, ``InstanceSegmentationEvaluator``) consume one
    chunk, free it, and ask for the next. Peak memory drops from
    ``topk * D*H*W`` to ``chunk_size * D*H*W``.

    Notes:
        * The chunked einsum is identical to the global one because the einsum is
        parallelizable along the query axis.
        * Per-chunk logits are produced in the model's compute dtype (typically
        bf16/fp16 when AMP is active); upsample is forced to fp32 internally to
        avoid trilinear precision issues, then cast back. Callers that need a
        different dtype should cast on consumption.
        * Bbox-cropped materialization is not implemented yet but is a natural
        extension: take the union of ``pred_box`` and ``gt_box``, crop the
        pixel-decoder feature map to the corresponding low-res region, and only
        upsample within that crop. We expose enough state (the raw embeddings)
        to add this without changing callers.
    """

    def __init__(
        self,
        mask_embeddings: torch.Tensor,
        pixel_decoder_output: torch.Tensor,
        target_size: Sequence[int],
        chunk_size: int = 8,
        upsample_dtype: torch.dtype = torch.float32,
    ) -> None:
        if mask_embeddings.dim() != 2:
            raise ValueError(
                f"mask_embeddings must be (Q, mask_dim); got {tuple(mask_embeddings.shape)}"
            )
        if pixel_decoder_output.dim() != 4:
            raise ValueError(
                f"pixel_decoder_output must be (mask_dim, d, h, w); got "
                f"{tuple(pixel_decoder_output.shape)}"
            )
        if mask_embeddings.shape[1] != pixel_decoder_output.shape[0]:
            raise ValueError(
                "mask_dim mismatch: mask_embeddings has "
                f"{mask_embeddings.shape[1]} channels but pixel_decoder_output has "
                f"{pixel_decoder_output.shape[0]} channels"
            )
        if len(target_size) != 3:
            raise ValueError(f"target_size must be 3D; got {tuple(target_size)}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive; got {chunk_size}")

        self.mask_embeddings = mask_embeddings
        self.pixel_decoder_output = pixel_decoder_output
        self.target_size = tuple(int(s) for s in target_size)
        self.chunk_size = int(chunk_size)
        self.upsample_dtype = upsample_dtype

    @property
    def num_queries(self) -> int:
        return int(self.mask_embeddings.shape[0])

    @property
    def mask_dim(self) -> int:
        return int(self.mask_embeddings.shape[1])

    @torch.no_grad()
    def materialize(self, query_indices: torch.Tensor) -> torch.Tensor:
        """Materialize the upsampled mask logits for ``query_indices`` at once.

        Use sparingly — defeats the purpose of chunking when the index set is
        large. Provided for parity with the legacy non-chunked path.
        """
        if query_indices.numel() == 0:
            return self.mask_embeddings.new_zeros((0, *self.target_size))
        embed = self.mask_embeddings.index_select(0, query_indices)
        return self._materialize_from_embeds(embed)

    @torch.no_grad()
    def chunks(
        self,
        query_indices: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(chunk_query_indices, upsampled_mask_logits)`` per chunk.

        ``upsampled_mask_logits`` has shape ``(k, D, H, W)`` where ``k`` is the
        number of queries in this chunk (last chunk may be smaller). Logits are
        in the model's mask-embedding dtype (no sigmoid applied).
        """
        if query_indices.numel() == 0:
            return
        cs = int(chunk_size) if chunk_size is not None else self.chunk_size
        if cs <= 0:
            raise ValueError(f"chunk_size must be positive; got {cs}")
        n = int(query_indices.numel())
        for start in range(0, n, cs):
            end = min(start + cs, n)
            chunk_idx = query_indices[start:end]
            embed = self.mask_embeddings.index_select(0, chunk_idx)
            yield chunk_idx, self._materialize_from_embeds(embed)

    def _materialize_from_embeds(self, mask_embeds_chunk: torch.Tensor) -> torch.Tensor:
        """Run einsum + trilinear upsample for one chunk of queries.

        ``mask_embeds_chunk``: ``(k, mask_dim)``.
        Returns: ``(k, D, H, W)`` logits at ``self.target_size``.
        """
        # Low-res einsum: (k, mask_dim) @ (mask_dim, d*h*w) -> (k, d, h, w).
        # Using matmul on the flattened pixel decoder is slightly cheaper than
        # einsum and easier to reason about for memory.
        c, d, h, w = self.pixel_decoder_output.shape
        low_res = mask_embeds_chunk @ self.pixel_decoder_output.reshape(c, d * h * w)
        low_res = low_res.reshape(-1, d, h, w)  # (k, d, h, w)

        # Trilinear upsample: F.interpolate expects (N, C, D, H, W); treat each
        # query as its own "batch" item with C=1 so memory scales linearly.
        original_dtype = low_res.dtype
        low_res = low_res.to(self.upsample_dtype)
        upsampled = F.interpolate(
            low_res.unsqueeze(1),
            size=self.target_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)  # (k, D, H, W)
        return upsampled.to(original_dtype)

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
        spatial_shape = get_spatial_shape(input_shape, input_fmt)
        default_output_metadata = DictConfig({
            "tensor_info": {
                "masks": {
                    "shape": spatial_shape,
                    "dtype": "uint16",
                },
                "boxes": {
                    "shape": (topk_per_image, 6),
                    "dtype": "float32",
                },
                "labels": {
                    "shape": (topk_per_image,),
                    "dtype": "float32",
                },
            },
        })
        if output_metadata is not None:
            default_output_metadata.merge_with(output_metadata)
        self.output_metadata = default_output_metadata

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

    @torch.jit.ignore
    def validate_outputs(self, preds: Dict[str, torch.Tensor]) -> None:
        """Verify that prediction keys and per-sample shapes match tensor_info.

        Raises ``ValueError`` with a detailed message on any mismatch so
        shape/key bugs surface immediately instead of silently corrupting
        downstream buffers.
        """
        tensor_info = self.output_metadata["tensor_info"]
        expected_keys = set(tensor_info.keys())
        actual_keys = set(preds.keys())

        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys
        parts: List[str] = []
        if missing:
            parts.append(f"missing keys {missing}")
        if extra:
            parts.append(f"unexpected keys {extra}")

        for key in expected_keys & actual_keys:
            expected_shape = tuple(tensor_info[key]["shape"])
            actual_shape = tuple(preds[key].shape[1:])  # strip batch dim
            if "mask" in key.lower():
                if len(actual_shape) != len(expected_shape):
                    parts.append(
                        f"'{key}' rank mismatch: tensor_info declares "
                        f"{len(expected_shape)} spatial dims {expected_shape} but got "
                        f"{len(actual_shape)} {actual_shape} (batch shape {tuple(preds[key].shape)})"
                    )
            elif actual_shape != expected_shape:
                parts.append(
                    f"'{key}' shape mismatch: tensor_info declares "
                    f"{expected_shape} but got {actual_shape} (batch shape {tuple(preds[key].shape)})"
                )

        if parts:
            raise ValueError(
                "Model output / tensor_info mismatch:\n  " + "\n  ".join(parts)
            )

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
        """Run inference and return per-sample collapsed instance predictions.

        Mask materialization is performed in chunks of ``self.mask_chunk_size``
        queries to keep peak vRAM bounded; see :class:`MaskMaterializer` and
        :meth:`predict_for_eval` for the underlying primitives. The returned
        dict has the following keys / shapes:

            * ``boxes``: ``(B, topk, 6)`` in xyzxyz at original-image scale.
            * ``labels``: ``(B, topk)`` per-instance scores (optionally
              re-weighted by mean mask probability when ``focus_on_boxes`` is
              False).
            * ``masks``: ``(B, *orig_image_size)`` uint16 instance label maps.
        """
        intermediates = self.predict_for_eval(data_sample)

        per_sample_preds = []
        for sample in intermediates:
            per_sample_preds.append(self._collapse_sample_chunked(sample))

        batched = {
            key: torch.stack([s[key] for s in per_sample_preds], dim=0)
            for key in per_sample_preds[0]
        }
        return batched

    def predict_for_eval(self, data_sample: dict) -> List[Dict[str, Any]]:
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
            * ``boxes``: ``(topk, 6)`` xyzxyz at original-image scale.
            * ``orig_image_size``: ``(D, H, W)`` ints.
        """
        features_dict = self.backbone(data_sample)
        outputs, _ = self.segmentation_head(
            features_dict, targets=None, predict_mask=False
        )
        pred_logits = outputs["pred_logits"]            # (B, Q, num_classes)
        pred_boxes = outputs["pred_boxes"]              # (B, Q, 6) cxcyczwhd
        mask_embeddings_b = outputs["mask_embeddings"]      # (B, Q, mask_dim)
        pixel_decoder_output_b = outputs["pixel_decoder_output"]  # (B, mask_dim, d, h, w)

        orig_image_sizes = [
            tuple(int(x) for x in orig.tolist())
            for orig in data_sample["metainfo"]["orig_image_sizes"]
        ]

        num_classes = self.segmentation_head.num_classes
        per_sample: List[Dict[str, Any]] = []
        for b, orig_size in enumerate(orig_image_sizes):
            scores = pred_logits[b].sigmoid()                                 # (Q, C)
            topk_scores, topk_idx = scores.flatten(0, 1).topk(
                self.topk_per_image, sorted=False
            )
            # query/class recovery from the (Q*C,) flat index:
            #   q = idx // C, c = idx % C
            topk_query_idx = topk_idx // num_classes                          # (topk,)
            topk_class_id = topk_idx % num_classes                            # (topk,)

            depth, height, width = orig_size
            boxes_topk = self.box_postprocess(
                pred_boxes[b][topk_query_idx], depth, height, width
            )

            per_sample.append({
                "mask_embeddings": mask_embeddings_b[b],
                "pixel_decoder_output": pixel_decoder_output_b[b],
                "topk_query_indices": topk_query_idx,
                "topk_class_scores": topk_scores,
                "topk_class_ids": topk_class_id,
                "boxes": boxes_topk,
                "orig_image_size": orig_size,
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

        Two passes through the chunked materializer: the first computes
        mask-confidence so we know the final per-instance score; the second
        re-materializes in ascending-score order to stamp label IDs. We trade
        ~2x compute for keeping peak memory at ``mask_chunk_size * D*H*W``.
        """
        materializer = MaskMaterializer(
            mask_embeddings=sample["mask_embeddings"],
            pixel_decoder_output=sample["pixel_decoder_output"],
            target_size=sample["orig_image_size"],
            chunk_size=self.mask_chunk_size,
        )
        topk_query_idx = sample["topk_query_indices"]
        topk_scores = sample["topk_class_scores"]

        device = topk_query_idx.device
        K = int(topk_query_idx.numel())
        mask_confidence = torch.zeros(K, device=device, dtype=torch.float32)

        # Pass 1: per-instance mask confidence in original topk order.
        slot = 0
        for chunk_idx, mask_logits in materializer.chunks(topk_query_idx):
            k = int(chunk_idx.numel())
            sigm = mask_logits.sigmoid()
            binary = (mask_logits > 0)
            sigm_inside = (sigm * binary.to(sigm.dtype)).flatten(1).sum(dim=1)
            count_inside = binary.flatten(1).sum(dim=1).to(torch.float32)
            mask_confidence[slot:slot + k] = sigm_inside.to(torch.float32) / torch.clamp(
                count_inside, min=1.0
            )
            slot += k

        if self.focus_on_boxes:
            scores_per_instance = topk_scores.to(torch.float32)
        else:
            scores_per_instance = topk_scores.to(torch.float32) * mask_confidence

        # Pass 2: collapse per-instance binary masks into a label map in
        # ascending-score order so highest-score instances overwrite earlier
        # ones (matches `collapse_instance_masks` semantics).
        order = scores_per_instance.argsort()  # ascending
        label_map = torch.zeros(
            sample["orig_image_size"], dtype=torch.int32, device=device
        )
        ordered_query_idx = topk_query_idx[order]
        # Label IDs 1..K assigned in iteration order = ascending-score order.
        next_id = 1
        for chunk_idx, mask_logits in materializer.chunks(ordered_query_idx):
            k = int(chunk_idx.numel())
            for i in range(k):
                label_map[mask_logits[i] > 0] = next_id
                next_id += 1

        return {
            "boxes": sample["boxes"],
            "labels": scores_per_instance,
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