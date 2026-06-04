"""Streaming, memory-bounded instance-segmentation evaluator for MaskDINO.

Drives chunked mask materialization (see :class:`MaskMaterializer`) so that
peak GPU memory is bounded regardless of ``topk_per_image`` and full-resolution
volume size. Instead of receiving materialized per-image masks like a vanilla
detection evaluator, this evaluator consumes the *intermediates* returned by
:meth:`MaskDINO.predict_for_eval` and computes per-(image, class) IoU rows
itself, pushing only summary statistics (scores + IoU rows + n_gt) to the
metrics.

Why not the dict-based flow?
    The legacy detection-style evaluator path keeps every predicted instance
    mask in CPU memory across the whole epoch. For 3D MaskDINO with
    ``topk_per_image=100`` and ``D=H=W=256``, that is ~6.4 GB per batch
    element of fp32 logits before binarization — quickly fatal on long eval
    runs. The streaming flow caps memory at one chunk of upsampled masks at
    a time and only retains compact summary stats (per-image, per-class IoU
    rows of shape ``(k, m)`` where ``k <= max_detections`` and ``m`` is the
    GT count for that class) on CPU.
"""

from typing import Any, Dict, List, Mapping, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from torch.nn import functional as F

from cell_observatory_platform.data.structures import (
    box_cxcyczwhd_to_xyzxyz,
    box_iou_3d,
)
from cell_observatory_platform.evaluation.evaluator import DatasetEvaluator
from cell_observatory_platform.evaluation.metrics import (
    BoxF1Metric,
    BoxMAPMetric,
    BoxMIoUMetric,
    MaskMAPMetric,
    MaskMIoUMetric,
)

# PredictedIoUEvalMetric (SAM2-style self-assessed mask-quality head eval) lives
# in metrics.py and is the consumer of the per-detection ``pred_ious`` stat we
# pass through below. It is imported tolerantly so this evaluator keeps working
# even when run against an older metrics.py that predates the metric: we only
# ever ``isinstance``-check against it, so a ``None`` placeholder simply means
# "no PredictedIoUEvalMetric will ever be matched", which is the correct no-op.
try:
    from cell_observatory_platform.evaluation.metrics import (
        PredictedIoUEvalMetric,
    )
except ImportError:  # pragma: no cover - metric not yet present in metrics.py
    PredictedIoUEvalMetric = ()  # isinstance(x, ()) is always False -> graceful skip

from cell_observatory_platform.models.meta_arch.maskdino import MaskMaterializer


# Tells the trainer dispatcher to call ``model.predict_for_eval`` instead of
# the default ``model.predict`` so we get raw intermediates and own mask
# materialization. Keeping it as a class attribute (not a method) means the
# dispatcher can read it without instantiating the evaluator twice.
_PREDICT_METHOD_FOR_INSTANCE_SEG = "predict_for_eval"


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
        target_key: name of the key in ``data_sample["metainfo"]`` that holds
            the per-batch list of target dicts. Defaults to ``"targets"``.
        gt_box_format: coordinate format of ``target["boxes"]``.
            The model's predicted boxes are absolute ``xyzxyz`` and ``box_iou_3d`` assumes ``xyzxyz`` corners,
            so GT must be converted to that same space before the box metrics. 
        gt_boxes_normalized: a toggle to normalize ``target["boxes"]``.
    """

    # Surfaced to the trainer dispatcher (see TestTrainer.run_test_step).
    predict_method = _PREDICT_METHOD_FOR_INSTANCE_SEG

    _SUPPORTED_GT_BOX_FORMATS = ("cxcyczwhd", "xyzxyz")

    def __init__(
        self,
        metrics: List[Any],
        mask_chunk_size: int = 8,
        score_threshold: float = 0.0,
        match_labels: bool = True,
        gt_mask_source: str = "label_map",
        target_key: str = "targets",
        gt_box_format: str = "cxcyczwhd",
        gt_boxes_normalized: bool = True,
    ):
        if gt_mask_source not in ("label_map", "masks"):
            raise ValueError(
                f"gt_mask_source must be 'label_map' or 'masks'; got {gt_mask_source!r}"
            )
        gt_box_format = str(gt_box_format).lower()
        if gt_box_format not in self._SUPPORTED_GT_BOX_FORMATS:
            raise ValueError(
                f"gt_box_format must be one of {self._SUPPORTED_GT_BOX_FORMATS}; "
                f"got {gt_box_format!r}"
            )
        self.mask_chunk_size = int(mask_chunk_size)
        self.score_threshold = float(score_threshold)
        self.match_labels = bool(match_labels)
        self.gt_mask_source = gt_mask_source
        self.target_key = target_key
        self.gt_box_format = gt_box_format
        self.gt_boxes_normalized = bool(gt_boxes_normalized)

        self.metrics: Dict[str, Any] = {}
        if isinstance(metrics, (list, ListConfig)):
            metric_iter = list(metrics)
        else:
            raise TypeError(
                f"metrics must be a list/ListConfig; got {type(metrics).__name__}"
            )
        for spec in metric_iter:
            name, metric = self._instantiate_metric(spec)
            if name in self.metrics:
                raise ValueError(f"duplicate metric name: {name!r}")
            self.metrics[name] = metric

        # Stable per-image id counter; we need it as the bucketing key for
        # per-class GT matching across images in MaskMAPMetric.
        self._image_id_counter = 0
        self._results: Dict[str, Optional[float]] = {name: None for name in self.metrics}

    @staticmethod
    def _instantiate_metric(spec: Any):
        """Build a (name, metric) pair from a flexible config spec."""
        if isinstance(spec, str):
            factory = _DEFAULT_METRIC_FACTORIES.get(spec)
            if factory is None:
                raise ValueError(
                    f"unknown metric name {spec!r}; expected one of "
                    f"{sorted(_DEFAULT_METRIC_FACTORIES)}"
                )
            return spec, factory()
        if isinstance(spec, (dict, DictConfig)):
            d = dict(spec) if isinstance(spec, DictConfig) else dict(spec)
            name = d.pop("name", None)
            if name is None:
                raise ValueError("metric DictConfig must include a 'name' field")
            metric = instantiate(d)
            return str(name), metric
        raise TypeError(
            f"metric spec must be str or DictConfig; got {type(spec).__name__}"
        )

    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()
        self._image_id_counter = 0
        self._results = {name: None for name in self.metrics}

    @torch.no_grad()
    def process(self, data_sample: dict, outputs: Any, loss_dict=None) -> None:
        """Drive chunked IoU computation for each image in the batch.

        ``outputs`` must be the per-batch list returned by
        :meth:`MaskDINO.predict_for_eval` — see that docstring for the dict
        contract.
        """
        if loss_dict is not None:
            # We don't need losses; defensive about flow misuse.
            pass

        if not isinstance(outputs, (list, tuple)):
            raise TypeError(
                "InstanceSegmentationEvaluator expects model.predict_for_eval-style "
                "per-sample list of intermediates; got "
                f"{type(outputs).__name__}. Make sure cfg.evaluation.evaluator points "
                "to this evaluator and the trainer dispatches via predict_method."
            )

        targets_list = self._extract_targets(data_sample)
        if len(targets_list) != len(outputs):
            raise RuntimeError(
                f"batch size mismatch: outputs has {len(outputs)} samples but "
                f"metainfo[{self.target_key!r}] has {len(targets_list)}"
            )

        for sample_intermediates, target in zip(outputs, targets_list):
            self._process_one(sample_intermediates, target)
            self._image_id_counter += 1

    def evaluate(self) -> Dict[str, float]:
        # Cross-rank reduction happens by gathering each metric's COMPACT
        # sufficient statistics (IoU matrices / scalar IoU lists / counts) and
        # aggregating ONCE over the pooled multiset — never by reducing the
        # per-rank scalar. ``gather()`` is idempotent and a no-op when
        # world_size == 1, so this is safe in single-process runs.
        for name, metric in self.metrics.items():
            metric.gather()
            value = metric.aggregate()
            # A metric whose aggregate() returns a Mapping (e.g. the
            # PredictedIoUEvalMetric reporting its selection curve / calibration
            # / ranked-AP keys) is FLATTENED into self._results under
            # ``f"{name}/{subkey}"``; a scalar return keeps the legacy behavior.
            if isinstance(value, Mapping):
                self._results.pop(name, None)
                for subkey, subval in value.items():
                    self._results[f"{name}/{subkey}"] = float(subval)
            else:
                self._results[name] = float(value)
        return self._results

    # ------------------------------------------------------------------
    # Per-image driver
    # ------------------------------------------------------------------

    def _extract_targets(self, data_sample: dict) -> List[Dict[str, Any]]:
        targets_field = data_sample["metainfo"][self.target_key]
        # The platform convention is List[List[dict]] (outer wraps batch); strip if present.
        if isinstance(targets_field, (list, tuple)) and targets_field and \
                isinstance(targets_field[0], (list, tuple)):
            targets_field = targets_field[0]
        return list(targets_field)

    def _process_one(self, sample: Dict[str, Any], target: Dict[str, Any]) -> None:
        image_id = self._image_id_counter
        device = sample["topk_query_indices"].device

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

        # Self-assessed per-prediction mask quality from the model's PREDICTED-IoU
        # head (SAM2 contract). predict_for_eval exposes this as one of the keys
        # tried below (e.g. "pred_ious"); MaskDINO has no IoU head and provides
        # none, in which case ``topk_pred_ious`` stays None and we skip the
        # PredictedIoUEvalMetric push entirely (the metric is only meaningful for
        # models with an IoU head). It is aligned 1:1 with the UNFILTERED topk
        # predictions (same ordering as topk_class_scores), so it slices with the
        # very same per-class index masks used for scores/boxes/ious below.
        topk_pred_ious = self._lookup_pred_ious(sample)
        if topk_pred_ious is not None:
            # Co-locate with the per-class index mask (WF1 hardening): SAM2 emits
            # iou_preds on CPU while indices may be CUDA, or vice versa.
            topk_pred_ious = topk_pred_ious.to(device)

        gt_labels = target["labels"]
        gt_boxes = target["boxes"]
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
        gt_boxes_xyzxyz = self._gt_boxes_to_abs_xyzxyz(
            gt_boxes, sample["orig_image_size"]
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

        # Two mask sources are supported:
        #   * MaskDINO: lazily materialize masks from decoder intermediates
        #     (``mask_embeddings`` + ``pixel_decoder_output``) via
        #     :class:`MaskMaterializer`, which owns trilinear upsample and
        #     logit>0 binarization.
        #   * SAM2 (AMG): masks are produced DIRECTLY by the model — already
        #     binarized bool tensors at orig-volume resolution — and surfaced
        #     under the ``pred_masks`` key (SAM2.predict_for_eval). SAM2 has no
        #     pixel decoder, so there are no embeddings to materialize.
        # The branch is gated purely on presence of a non-None ``pred_masks``
        # key, so the MaskDINO path (which never emits that key) is byte-identical
        # to before.
        _has_direct_masks = (
            "pred_masks" in sample and sample["pred_masks"] is not None
        )

        if _has_direct_masks:
            # SAM2 direct-mask source: (N, Z, Y, X) bool. Keep on its own device;
            # we move per-chunk slices to ``device`` (the indexing-mask device)
            # right before the pairwise IoU so a producer returning masks on a
            # different device can't crash the advanced-index/IoU.
            pred_masks_all = sample["pred_masks"]
            materializer = None
        else:
            materializer = MaskMaterializer(
                mask_embeddings=sample["mask_embeddings"],
                pixel_decoder_output=sample["pixel_decoder_output"],
                target_size=sample["orig_image_size"],
                chunk_size=self.mask_chunk_size,
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
                gt_masks_c = self._gt_masks_for_class(
                    target=target,
                    gt_class_mask=gt_mask_c,
                    target_size=sample["orig_image_size"],
                ).to(device)
            else:
                gt_masks_c = None

            # Stream chunks of predicted masks and accumulate IoU rows.
            ious_rows: List[torch.Tensor] = []
            if pred_query_idx_c.numel() > 0 and gt_masks_c is not None:
                if _has_direct_masks:
                    # SAM2 direct-mask path: ``pred_query_idx_c`` is a direct slot
                    # index (arange(N) sliced by class) into ``pred_masks_all``.
                    # Chunk over proposals to bound peak memory at
                    # ``chunk_size * Z*Y*X`` masks materialized on ``device`` at a
                    # time (same bound as the MaskDINO materializer). Move both the
                    # index and the gathered masks onto a common device before
                    # indexing/IoU so a CPU-vs-CUDA mismatch can't crash.
                    cs = self.mask_chunk_size
                    src_device = pred_masks_all.device
                    idx_on_src = pred_query_idx_c.to(src_device)
                    for start in range(0, idx_on_src.numel(), cs):
                        chunk_idx = idx_on_src[start : start + cs]
                        # Index on the masks' own device, then move the (small)
                        # bool chunk to ``device`` for the pairwise IoU vs GT.
                        pred_bin = pred_masks_all[chunk_idx].to(device).bool()
                        iou_chunk = _pairwise_mask_iou_3d_bool(pred_bin, gt_masks_c)
                        ious_rows.append(iou_chunk.cpu())
                        del pred_bin
                else:
                    for _, mask_logits in materializer.chunks(pred_query_idx_c):
                        # Binarize at logit > 0, i.e. sigmoid(logit) > 0.5. This
                        # matches the dense MaskMIoUMetric/MaskMAPMetric path, which
                        # calls .bool() on already-thresholded masks. The producing
                        # model emits mask LOGITS on this path (predict_for_eval),
                        # so > 0 is the correct probability-0.5 cut.
                        pred_bin = (mask_logits > 0)
                        iou_chunk = _pairwise_mask_iou_3d_bool(pred_bin, gt_masks_c)
                        ious_rows.append(iou_chunk.cpu())
            if ious_rows:
                ious_c = torch.cat(ious_rows, dim=0)
            else:
                # No predictions -> empty IoU matrix. Still record n_gt so this
                # class contributes to the recall denominator.
                ious_c = torch.zeros((0, n_gt_c), dtype=torch.float32)

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

    # Keys under which predict_for_eval may expose the per-prediction PREDICTED
    # IoU vector. Tried in order; the first present, non-None entry wins. Keep
    # this list the single source of truth for the SAM2 IoU-head contract.
    _PRED_IOU_SAMPLE_KEYS = (
        "pred_ious",
        "predicted_ious",
        "iou_preds",
        "mask_iou_preds",
    )

    def _lookup_pred_ious(self, sample: Mapping[str, Any]) -> Optional[torch.Tensor]:
        """Tolerantly fetch the per-prediction predicted-IoU vector ``(k,)``.

        Returns the model's self-assessed mask-quality vector aligned 1:1 with
        the UNFILTERED topk predictions, or ``None`` when the model exposes no
        IoU head (e.g. MaskDINO) so callers skip the PredictedIoUEvalMetric push.
        Skipping the lookup entirely when no consumer metric is registered avoids
        forcing the contract on models that don't have an IoU head.
        """
        if not any(
            isinstance(m, PredictedIoUEvalMetric) for m in self.metrics.values()
        ):
            return None
        for key in self._PRED_IOU_SAMPLE_KEYS:
            value = sample.get(key) if isinstance(sample, Mapping) else None
            if value is not None:
                return value
        return None

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

    def _gt_boxes_to_abs_xyzxyz(self, gt_boxes: torch.Tensor, orig_image_size: Any) -> torch.Tensor:
        if gt_boxes.numel() == 0:
            return gt_boxes
        if self.gt_box_format == "cxcyczwhd":
            boxes = box_cxcyczwhd_to_xyzxyz(gt_boxes)
        else:  # already xyzxyz
            boxes = gt_boxes
        if self.gt_boxes_normalized:
            depth, height, width = (int(s) for s in orig_image_size)
            scale = boxes.new_tensor([width, height, depth, width, height, depth])
            boxes = boxes * scale
        return boxes

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

    def _gt_masks_for_class(
        self,
        target: Dict[str, Any],
        gt_class_mask: torch.Tensor,
        target_size: Any,
    ) -> torch.Tensor:
        """Return ``(n_gt_class, D, H, W)`` bool masks at original image size."""
        if self.gt_mask_source == "masks":
            masks = target["masks"][gt_class_mask].bool()
            return self._maybe_resize_gt_masks(masks, target_size)
        # label_map path: build instance masks via mask_ids equality.
        label_map = target["label_map"]
        mask_ids = target["mask_ids"][gt_class_mask].to(label_map.dtype)
        if mask_ids.numel() == 0:
            return label_map.new_zeros((0, *label_map.shape), dtype=torch.bool)
        view_shape = (mask_ids.numel(),) + (1,) * label_map.dim()
        masks = label_map.unsqueeze(0) == mask_ids.view(view_shape)
        return self._maybe_resize_gt_masks(masks, target_size)

    @staticmethod
    def _maybe_resize_gt_masks(masks: torch.Tensor, target_size: Any) -> torch.Tensor:
        """Trilinear-resize GT masks to ``target_size`` if shapes differ."""
        target_size = tuple(int(s) for s in target_size)
        if masks.shape[-3:] == target_size:
            return masks
        # Resize via nearest-neighbor on float to avoid kernel artifacts on
        # binary masks.
        resized = F.interpolate(
            masks.unsqueeze(0).float(),
            size=target_size,
            mode="nearest",
        ).squeeze(0)
        return resized > 0.5

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


def _default_box_map() -> BoxMAPMetric:
    return BoxMAPMetric()


def _default_box_miou() -> BoxMIoUMetric:
    return BoxMIoUMetric()


def _default_box_f1() -> BoxF1Metric:
    return BoxF1Metric()


def _default_mask_map() -> MaskMAPMetric:
    return MaskMAPMetric()


def _default_mask_miou() -> MaskMIoUMetric:
    return MaskMIoUMetric(mode="instance")


_DEFAULT_METRIC_FACTORIES = {
    "box_map": _default_box_map,
    "box_miou": _default_box_miou,
    "box_f1": _default_box_f1,
    "mask_map": _default_mask_map,
    "mask_miou": _default_mask_miou,
}


# Re-export so callers don't have to import from data.structures separately.
__all__ = ["InstanceSegmentationEvaluator", "box_iou_3d"]
