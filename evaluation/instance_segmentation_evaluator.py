"""
Streaming, memory-bounded instance-segmentation evaluator.
"""

from typing import Any, Dict, List, Optional

import torch

import cell_observatory_platform.evaluation.evaluate_postprocess as ep
from cell_observatory_platform.evaluation.evaluator import DatasetEvaluator
from cell_observatory_platform.evaluation.metrics import (
    BoxF1Metric,
    BoxMAPMetric,
    BoxMIoUMetric,
    MaskMAPMetric,
    MaskMIoUMetric,
    PredictedIoUEvalMetric,
    build_metrics,
)
from cell_observatory_platform.evaluation.mask_source import build_mask_source

from cell_observatory_platform.utils.registry import REGISTRY


class InstanceSegmentationEvaluator(DatasetEvaluator):
    """Per-image streaming evaluator for MaskDINO instance segmentation.

    Args:
        metrics: list of metric names (or DictConfigs to instantiate). When a
            string is provided, a sensible default metric of that name is
            constructed (see ``_DEFAULT_METRIC_FACTORIES``). When a DictConfig
            is provided it is passed through Hydra ``instantiate``.
        mask_chunk_size: queries to materialize at full resolution at once.
            Trades compute for peak GPU memory (``~chunk_size * D*H*W * 4 B``
            for fp32 logits during materialization).
        score_threshold: per-prediction score threshold applied before
            metrics computation. Predictions below threshold are dropped from
            both ranking and matching.
        match_labels: when True (recommended for COCO-style mAP), only count
            a prediction as TP when its predicted class matches the GT class.
        gt_mask_source: either ``"label_map"`` (build per-instance binary
            masks from ``target["label_map"]`` + ``target["mask_ids"]``) or
            ``"masks"`` (use the pre-materialized ``target["masks"]`` tensor).
        gt_box_format: coordinate format of ``target["boxes"]``.
            The model's predicted boxes are absolute ``xyzxyz`` and ``box_iou_3d`` assumes ``xyzxyz`` corners,
            so GT must be converted to that same space before the box metrics. 
        gt_boxes_normalized: a toggle to normalize ``target["boxes"]``.
    """

    def __init__(
        self,
        metrics: List[Any],
        mask_chunk_size: int = 8,
        score_threshold: float = 0.0,
        match_labels: bool = True,
        gt_mask_source: str = "label_map",
        gt_box_format: str = "cxcyczwhd",
        gt_boxes_normalized: bool = True,
    ):
        if gt_mask_source not in ("label_map", "masks"):
            raise ValueError(
                f"gt_mask_source must be 'label_map' or 'masks'; got {gt_mask_source!r}"
            )
        gt_box_format = str(gt_box_format).lower()
        if gt_box_format not in ep.GT_BOX_FORMATS:
            raise ValueError(
                f"gt_box_format must be one of {ep.GT_BOX_FORMATS}; got {gt_box_format!r}"
            )
        self.mask_chunk_size = int(mask_chunk_size)
        self.score_threshold = float(score_threshold)
        self.match_labels = bool(match_labels)
        self.gt_mask_source = gt_mask_source
        self.gt_box_format = gt_box_format
        self.gt_boxes_normalized = bool(gt_boxes_normalized)

        self.metrics: Dict[str, Any] = build_metrics(metrics)

        # Stable per-image id counter; we need it as the bucketing key for
        # per-class GT matching across images in MaskMAPMetric.
        self._image_id_counter = 0
        self._results: Dict[str, Optional[float]] = {name: None for name in self.metrics}

    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()
        self._image_id_counter = 0
        self._results = {name: None for name in self.metrics}

    @torch.no_grad()
    def process(self, data_sample: dict, outputs: Any, loss_dict=None) -> None:
        """Drive chunked IoU computation for each image in the batch.

        ``outputs`` must be the per-batch list of intermediates returned by the
        model's ``evaluate_step`` — see that docstring for the dict contract.
        """
        if not isinstance(outputs, (list, tuple)):
            raise TypeError(
                "InstanceSegmentationEvaluator expects a per-sample list of "
                f"evaluate_step intermediates; got {type(outputs).__name__}."
            )

        targets_list = ep.extract_targets(data_sample, squeeze_label_map=True)
        if len(targets_list) != len(outputs):
            raise RuntimeError(
                f"batch size mismatch: outputs has {len(outputs)} samples but "
                f"metainfo['targets'] has {len(targets_list)}"
            )

        for sample_intermediates, target in zip(outputs, targets_list):
            self._process_one(sample_intermediates, target)
            self._image_id_counter += 1

    # evaluate() is inherited from DatasetEvaluator: it gathers + aggregates
    # every metric and flattens Mapping returns (PredictedIoUEvalMetric) under
    # ``f"{name}/{subkey}"`` into a flat dict[str, float].

    # ------------------------------------------------------------------
    # Per-image driver
    # ------------------------------------------------------------------

    def _process_one(self, sample: Dict[str, Any], target: Dict[str, Any]) -> None:
        image_id = self._image_id_counter
        device = sample["topk_query_indices"].device

        # ---- 0. Rank gate: instance IoU is strictly 3D. ----
        # ``eval_frame_size`` is the (D, H, W) target resolution; a 4D (T,Z,Y,X)
        # input would surface a length-4 spatial size, which the pairwise 3D IoU
        # path cannot handle. Reject up front with a clear message rather than
        # crashing deep inside mask materialization.
        eval_frame_size = sample["eval_frame_size"]
        if len(tuple(eval_frame_size)) != 3:
            raise ValueError(
                "InstanceSegmentationEvaluator is 3D-only: expected a (D, H, W) "
                f"eval_frame_size, got {tuple(eval_frame_size)} (ndim>4 / temporal "
                "inputs are not supported for instance IoU)."
            )

        # ---- 1. Build per-class index slices (predictions and GT). ----
        # AP (box + mask) must see EVERY detection: COCO AP and the dense
        # _per_class_ap / BoxMAPMetric.__call__ counterparts ingest the full,
        # unfiltered prediction set and rank globally by score. The
        # score_threshold belongs ONLY to the mIoU/F1 matching subsets, and
        # those metrics apply it themselves internally (BoxMIoUMetric /
        # BoxF1Metric filter inside __call__; the streaming instance-mIoU is
        # filtered explicitly below before greedy matching). So we deliberately
        # do NOT pre-filter here.
        # ``device`` is read from topk_query_indices above; coerce every tensor
        # sliced by the per-class boolean mask (derived from topk_class_ids) onto
        # that same device before advanced-indexing. A producer (e.g. SAM2)
        # returning some tensors on CPU and others on CUDA would otherwise crash
        # at the masked-index / downstream IoU step (WF1 hardening). Slicing
        # tensors must therefore co-locate with the mask we index them with.
        topk_scores = sample["topk_class_scores"].to(device)
        topk_class_ids = sample["topk_class_ids"].to(device)
        topk_query_idx = sample["topk_query_indices"]  # defines ``device``
        topk_boxes = sample["boxes"].to(device)

        if (
            self.match_labels
            and topk_class_ids.numel()
            and bool((topk_class_ids < 0).any())
        ):
            raise ValueError(
                "InstanceSegmentationEvaluator got class-agnostic predictions while match_labels=True."
                "Set match_labels=False in the evaluator config for class-agnostic"
            )

        # Self-assessed per-prediction mask quality from the model's PREDICTED-IoU
        # head (SAM2 contract, exposed under "iou_preds"); MaskDINO has no IoU head
        # and provides none, in which case ``topk_pred_ious`` stays None and we skip
        # the PredictedIoUEvalMetric push entirely (the metric is only meaningful for
        # models with an IoU head). It is aligned 1:1 with the UNFILTERED topk
        # predictions (same ordering as topk_class_scores), so it slices with the
        # very same per-class index masks used for scores/boxes/ious below.
        topk_pred_ious = sample.get("iou_preds")   # canonical key; None when no IoU head
        if topk_pred_ious is not None:
            # Co-locate with the per-class index mask (WF1 hardening): SAM2 emits
            # iou_preds on CPU while indices may be CUDA, or vice versa.
            topk_pred_ious = topk_pred_ious.to(device)

        gt_labels = target["labels"]
        # ---- 2. Box-only metrics first (cheap, no masks involved). ----
        # The box metrics (BoxMAPMetric / BoxF1Metric) match predictions to GT
        # within the SAME class label. Under class-agnostic eval (match_labels
        # =False, e.g. SAM2 AMG whose preds carry the sentinel class -1 while GT
        # keeps real labels), passing raw labels would score ~0 because no pred
        # label matches any GT label. Collapse both sides to a single sentinel
        # class so box metrics are genuinely class-agnostic — mirroring the mask
        # path's single [-1] bucket below.
        if self.match_labels:
            box_pred_labels, box_gt_labels = topk_class_ids, gt_labels
        else:
            box_pred_labels = torch.zeros_like(topk_class_ids)
            box_gt_labels = torch.zeros_like(gt_labels)
        gt_boxes_xyzxyz = ep.gt_boxes_abs_xyzxyz(
            target, sample["eval_frame_size"], self.gt_box_format, self.gt_boxes_normalized
        )
        self._update_box_metrics(
            pred_boxes=topk_boxes,
            pred_labels=box_pred_labels,
            pred_scores=topk_scores,
            gt_boxes=gt_boxes_xyzxyz,
            gt_labels=box_gt_labels,
        )

        # ---- 3. Mask metrics: drive chunked IoU per class. ----
        if not self._has_mask_metrics():
            return

        # The mask source abstracts where per-instance binary masks come from
        # (query-embedding materialization for MaskDINO vs direct bool masks for
        # SAM2 AMG). The model DECLARES which it produces via sample["mask_source"];
        # build_mask_source dispatches on that and then asserts the rank/dtype
        # contract, so soft float ``pred_masks`` (e.g. Mask2Former) are rejected
        # instead of being silently ``.bool()``-cast into garbage hard masks.
        mask_source = build_mask_source(
            sample, target_size=sample["eval_frame_size"]
        )

        # Per-(image, class) loop when labels matter; a single sentinel bucket
        # when evaluating class-agnostically. Without the sentinel, the
        # class-agnostic path would duplicate the same all-vs-all IoU matrix
        # once per observed class.
        unique_classes = (
            sorted(
                set(int(c) for c in topk_class_ids.tolist()) 
                | set(int(c) for c in gt_labels.tolist())
            )
            if self.match_labels
            else [-1]
        )

        # For instance-mode mIoU we accumulate matched IoUs (a single number per
        # match) across classes per image and push them in one shot.
        per_image_matched_ious: List[float] = []

        # Gate greedy matches on the consuming instance-mode mIoU metric's
        # IoU threshold so a trivial-IoU prediction can't steal a GT from a
        # better-overlapping, lower-scored prediction.
        match_iou_threshold = next(
            (
                m.iou_threshold
                for m in self.metrics.values()
                if isinstance(m, MaskMIoUMetric) and m.mode == "instance"
            ),
            0.0,
        )

        for class_id in unique_classes:
            pred_mask_c = (topk_class_ids == class_id) if self.match_labels else \
                torch.ones_like(topk_class_ids, dtype=torch.bool)
            gt_mask_c = (gt_labels == class_id) if self.match_labels else \
                torch.ones_like(gt_labels, dtype=torch.bool)

            pred_query_idx_c = topk_query_idx[pred_mask_c]
            pred_scores_c = topk_scores[pred_mask_c]
            # Per-class predicted IoUs: sliced with the SAME unfiltered class
            # mask as scores so they stay row-aligned with ious_c. None when the
            # model has no IoU head -> no PredictedIoUEvalMetric push.
            pred_ious_c = (
                topk_pred_ious[pred_mask_c] if topk_pred_ious is not None else None
            )
            n_gt_c = int(gt_mask_c.sum().item())

            # Materialize GT masks for this class on the fly (one class at a
            # time bounds memory: ``n_gt_c * D*H*W`` bool).
            if n_gt_c > 0:
                gt_masks_c = ep.gt_masks_for_class(
                    target, gt_mask_c, sample["eval_frame_size"], self.gt_mask_source
                ).to(device)
            else:
                gt_masks_c = None

            # Stream chunks of predicted binary masks (already on ``device``)
            # from the mask source and accumulate IoU rows. Peak memory is bound
            # to ``chunk_size * D*H*W`` regardless of the underlying model.
            ious_rows: List[torch.Tensor] = []
            if pred_query_idx_c.numel() > 0 and gt_masks_c is not None:
                for pred_bin in mask_source.binary_mask_chunks(
                    pred_query_idx_c, self.mask_chunk_size, device
                ):
                    iou_chunk = _pairwise_mask_iou_3d_bool(pred_bin, gt_masks_c)
                    ious_rows.append(iou_chunk.cpu())
                    del pred_bin
            if ious_rows:
                ious_c = torch.cat(ious_rows, dim=0)
            else:
                # No predictions -> empty IoU matrix but still need to record n_gt so this class contributes to the recall denominator.
                # 1) no predictions of this class -> (0, n_gt)
                # 2) predictions but no GT (gt_masks_c is None) -> (k, 0)
                ious_c = torch.zeros((pred_scores_c.numel(), n_gt_c), dtype=torch.float32) # (k, n_gt)

            # Push to MaskMAP via streaming API. AP must see the FULL,
            # unfiltered prediction set (no score threshold) so the pooled
            # global score-sort recapitulates the dense _per_class_ap exactly.
            for metric in self.metrics.values():
                if isinstance(metric, MaskMAPMetric):
                    metric.add_image_class(
                        image_id=image_id,
                        class_id=class_id,
                        scores=pred_scores_c,
                        ious=ious_c,
                        n_gt=n_gt_c,
                    )
                # PredictedIoUEvalMetric rides the SAME streaming push (so its
                # selection / calibration / ranked-AP stats reduce across ranks
                # exactly like MaskMAP), but only when the model actually emitted
                # an IoU head (pred_ious_c is not None). It additionally carries
                # the per-prediction pred_ious aligned with the unfiltered
                # scores/ious. Skipped gracefully for IoU-head-less models.
                elif isinstance(metric, PredictedIoUEvalMetric) and pred_ious_c is not None:
                    metric.add_image_class(
                        image_id=image_id,
                        class_id=class_id,
                        scores=pred_scores_c,
                        ious=ious_c,
                        n_gt=n_gt_c,
                        pred_ious=pred_ious_c,
                    )

            # Greedy match for instance-mode mIoU (per-class to keep label
            # constraint when match_labels=True). Unlike the AP push, mIoU sees
            # only the score-thresholded subset: this mirrors the dense
            # MaskMIoUMetric._update_instance which does
            # ``keep = scores >= self.score_threshold`` before matching. Filter
            # scores AND the corresponding ROWS of the IoU matrix together so
            # they stay aligned.
            if pred_scores_c.numel() and ious_c.numel():
                miou_keep = pred_scores_c >= self.score_threshold
                pred_scores_miou = pred_scores_c[miou_keep]
                ious_miou = ious_c[miou_keep]
                if pred_scores_miou.numel() and ious_miou.numel():
                    per_image_matched_ious.extend(
                        self._greedy_match_per_class(
                            pred_scores_miou, ious_miou, iou_threshold=match_iou_threshold
                        )
                    )

            # Free GPU mask now that we're done with this class.
            del gt_masks_c

        # Push matched IoUs once per image.
        if per_image_matched_ious:
            for metric in self.metrics.values():
                if isinstance(metric, MaskMIoUMetric) and metric.mode == "instance":
                    metric.add_matched_ious(per_image_matched_ious)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_mask_metrics(self) -> bool:
        # PredictedIoUEvalMetric also derives its (true-IoU based) stats from
        # the per-(image, class) IoU matrix, so its presence must drive mask
        # materialization too — otherwise _process_one returns early and never
        # builds ious_c for it.
        return any(
            isinstance(m, (MaskMAPMetric, PredictedIoUEvalMetric))
            or (isinstance(m, MaskMIoUMetric) and m.mode == "instance")
            for m in self.metrics.values()
        )

    def _update_box_metrics(
        self,
        pred_boxes: torch.Tensor,
        pred_labels: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> None:
        # Box metrics use the existing batched API. We push single-image lists.
        pred_dict = {
            "boxes": pred_boxes.detach().cpu(),
            "labels": pred_labels.detach().cpu(),
            "scores": pred_scores.detach().cpu(),
        }
        gt_dict = {
            "boxes": gt_boxes.detach().cpu(),
            "labels": gt_labels.detach().cpu(),
        }
        for metric in self.metrics.values():
            if isinstance(metric, (BoxMAPMetric, BoxMIoUMetric, BoxF1Metric)):
                metric([pred_dict], [gt_dict], None)

    @staticmethod
    def _greedy_match_per_class(
        scores: torch.Tensor, ious: torch.Tensor, iou_threshold: float = 0.0
    ) -> List[float]:
        """Greedy score-sorted matching; returns matched-pair IoUs.

        Only pairs whose IoU is ``>= iou_threshold`` are accepted, so a
        trivial-IoU prediction cannot consume a GT that a better-overlapping,
        lower-scored prediction would have matched.
        """
        # stable=True mirrors the dense MaskMIoUMetric._update_instance so that,
        # under exactly-equal scores competing for the same GT, the streaming
        # path reproduces the dense matched-IoU list byte-for-byte.
        order = torch.argsort(scores, descending=True, stable=True)
        matched_gt = torch.zeros(ious.shape[1], dtype=torch.bool)
        matched_ious: List[float] = []
        for i in order.tolist():
            row = ious[i].clone()
            if matched_gt.any():
                row[matched_gt] = -1.0
            best, best_idx = torch.max(row, dim=0)
            if best.item() >= iou_threshold:
                matched_ious.append(float(best.item()))
                matched_gt[int(best_idx.item())] = True
            else:
                # A sub-threshold best for this prediction must not abort
                # matching: a lower-scored prediction may still validly match a
                # different remaining GT.
                continue
        return matched_ious


def _pairwise_mask_iou_3d_bool(masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for ``(N, D, H, W)`` and ``(M, D, H, W)`` bool tensors.

    Mirrors :func:`evaluation.metrics._pairwise_mask_iou_3d` but kept here so
    we can keep the computation on the prediction's device when it's CUDA
    (the metrics-module helper assumes CPU)."""
    if masks_a.shape[0] == 0 or masks_b.shape[0] == 0:
        return masks_a.new_zeros((masks_a.shape[0], masks_b.shape[0]), dtype=torch.float32)
    a = masks_a.flatten(1).to(torch.float32)
    b = masks_b.flatten(1).to(torch.float32)
    inter = a @ b.t()
    sum_a = a.sum(dim=1, keepdim=True)
    sum_b = b.sum(dim=1, keepdim=True)
    union = sum_a + sum_b.t() - inter
    return inter / torch.clamp(union, min=1e-12)


from cell_observatory_platform.utils.config import register_class as _register_class
_register_class("evaluator", "instance_segmentation", InstanceSegmentationEvaluator)
