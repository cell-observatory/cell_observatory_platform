import abc
from typing import Callable, Dict, List, Literal, Optional, Sequence

import numpy as np
import torch

from cell_observatory_platform.data.structures import box_iou_3d


class Metric(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, outputs, targets, loss):
        pass

    @abc.abstractmethod
    def aggregate(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class TrainLosses(Metric):
    def __init__(self, reduce_method: str = "mean"):
        self.reduce_method = reduce_method
        self.loss_values = []

    def __call__(self, outputs, targets, loss):
        self.loss_values.append(loss.item())

    def aggregate(self):
        assert self.loss_values, "No loss values to aggregate."
        if self.reduce_method == "mean":
            return sum(self.loss_values) / len(self.loss_values)
        elif self.reduce_method == "min":
            return min(self.loss_values)
        elif self.reduce_method == "max":
            return max(self.loss_values)
        else:
            raise ValueError(f"Unknown reduce method: {self.reduce_method}")

    def reset(self):
        self.loss_values.clear()


class ReduceBuffer:
    def __init__(self, reduce_method: str = "mean"):
        self.reduce_method = reduce_method
        self.values: List[float] = []

    def add(self, v: torch.Tensor | float):
        v = float(v.item() if torch.is_tensor(v) else v)
        self.values.append(v)

    def aggregate(self) -> float:
        assert self.values, "No values to aggregate."
        if self.reduce_method == "mean":
            return sum(self.values) / len(self.values)
        elif self.reduce_method == "min":
            return min(self.values)
        elif self.reduce_method == "max":
            return max(self.values)
        else:
            raise ValueError(f"Unknown reduce method: {self.reduce_method}")

    def reset(self):
        self.values.clear()


# class SSIMMetric(Metric):
#     def __init__(self,
#                  data_range: Optional[float] = 1.0,
#                  kernel_size: int = 11,
#                  sigma: float = 1.5,
#                  K1: float = 0.01,
#                  K2: float = 0.03,
#                  reduction: Literal["elementwise_mean", "sum"] = "elementwise_mean",
#                  reduce_method: str = "mean",
#     ):
#         super().__init__()
#         self.data_range = data_range
#         self.kernel_size = kernel_size

#         self.sigma = sigma

#         self.K1 = K1
#         self.K2 = K2

#         self.buf = ReduceBuffer(reduce_method)
#         self.reduction = reduction

#     @torch.no_grad()
#     def __call__(self, outputs, targets, loss=None):
#         assert outputs.shape == targets.shape, f"SSIM: mismatched shapes {outputs.shape} vs {targets.shape}"
#         outputs, targets = _ssim_check_inputs(outputs, targets)
#         ssim_val = _ssim_update(
#             outputs,
#             targets,
#             data_range=self.data_range,
#             kernel_size=self.kernel_size,
#             sigma=self.sigma,
#             K1=self.K1,
#             K2=self.K2,
#             nonnegative_ssim=True,
#             full=False,
#         )

#         if self.reduction == "elementwise_mean":
#             ssim_val = ssim_val / outputs.shape[0]

#         self.buf.add(ssim_val)

#     def aggregate(self):
#         return self.buf.aggregate()

#     def reset(self):
#         self.buf.reset()


class NRMSEMetric(Metric):
    def __init__(self, reduce_method: str = "mean", eps: float = 1e-8):
        self.buf = ReduceBuffer(reduce_method)
        self.eps = eps

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        x = outputs.to(dtype=torch.float32)
        y = targets.to(dtype=torch.float32)
        diff = x - y
        mse = (diff * diff).mean()
        rmse = torch.sqrt(mse)
        denom = (torch.amax(y) - torch.amin(y)).clamp_min(self.eps)
        nrmse = rmse / denom
        self.buf.add(nrmse)

    def aggregate(self):
        return self.buf.aggregate()

    def reset(self):
        self.buf.reset()


class MAEMetric(Metric):
    def __init__(self, reduce_method: str = "mean"):
        self.buf = ReduceBuffer(reduce_method)

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        x = outputs.to(dtype=torch.float32)
        y = targets.to(dtype=torch.float32)
        mae = (x - y).abs().mean()
        self.buf.add(mae)

    def aggregate(self):
        return self.buf.aggregate()

    def reset(self):
        self.buf.reset()


# ---------------------------------------------------------------------------
# Detection / segmentation metrics
#
# Input contract (mirrors torchvision/COCO conventions, in 3D):
#   * detection-style metrics (Box*/Mask*MAP/MIoU/F1) consume per-image dicts:
#       - preds  : List[dict] of length B, each with keys
#                  {"boxes": (N,6)  in xyzxyz, "scores": (N,), "labels": (N,)}
#                  or {"masks": (N,D,H,W) bool/float, "scores": (N,), "labels": (N,)}
#       - targets: List[dict] of length B, each with keys
#                  {"boxes": (M,6), "labels": (M,)} or {"masks": (M,D,H,W), "labels": (M,)}
#   * ClassAPMetric consumes raw score / target tensors (binary or multi-class).
#   * MaskMIoUMetric supports two modes:
#       - "semantic": per-image int label maps (D,H,W) for both pred and target.
#       - "instance": per-image dicts with masks/scores/labels (same as MaskMAPMetric).
#
# Per-class AP at a single IoU threshold uses 101-point COCO interpolation.
# All accumulation happens on CPU to bound device memory across long eval runs.
# ---------------------------------------------------------------------------


def _ap_101_point(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style 101-point AP interpolation."""
    if recall.size == 0:
        return 0.0
    p = precision.copy()
    # right-to-left envelope so precision is monotonically non-increasing in score
    for i in range(len(p) - 1, 0, -1):
        if p[i] > p[i - 1]:
            p[i - 1] = p[i]
    rec_thresh = np.linspace(0.0, 1.0, 101)
    p_interp = np.zeros_like(rec_thresh)
    inds = np.searchsorted(recall, rec_thresh, side="left")
    valid = inds < len(p)
    p_interp[valid] = p[inds[valid]]
    return float(p_interp.mean())


def _binary_ap(scores: torch.Tensor, targets_bool: torch.Tensor) -> float:
    """Binary classification Average Precision via PR curve."""
    if targets_bool.numel() == 0 or not targets_bool.any():
        return 0.0
    order = torch.argsort(scores, descending=True)
    targets_sorted = targets_bool[order].to(torch.float32)
    tp = torch.cumsum(targets_sorted, dim=0)
    fp = torch.cumsum(1.0 - targets_sorted, dim=0)
    n_pos = float(targets_sorted.sum().item())
    recall = (tp / max(n_pos, 1.0)).numpy()
    precision = (tp / torch.clamp(tp + fp, min=1e-12)).numpy()
    return _ap_101_point(recall, precision)


def _pairwise_mask_iou_3d(masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of bool masks of shape (N,D,H,W) / (M,D,H,W)."""
    if masks_a.shape[0] == 0 or masks_b.shape[0] == 0:
        return masks_a.new_zeros((masks_a.shape[0], masks_b.shape[0]), dtype=torch.float32)
    a = masks_a.flatten(1).to(torch.float32)
    b = masks_b.flatten(1).to(torch.float32)
    inter = a @ b.t()
    sum_a = a.sum(dim=1, keepdim=True)
    sum_b = b.sum(dim=1, keepdim=True)
    union = sum_a + sum_b.t() - inter
    return inter / torch.clamp(union, min=1e-12)


def _per_class_ap(
    preds: List[dict],
    targets: List[dict],
    class_id: int,
    iou_thr: float,
    max_dets: int,
    geom_key: str,
    iou_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Optional[float]:
    """COCO-style per-class AP at one IoU threshold for boxes or masks.

    Returns ``None`` when the class is absent from the ground-truth (so it can
    be excluded from the per-class average), and ``0.0`` when GT exists but no
    predictions matched.
    """
    target_geom_per_img: Dict[int, torch.Tensor] = {}
    n_gt = 0
    for img_id, t in enumerate(targets):
        mask = (t["labels"] == class_id)
        geom_t = t[geom_key][mask]
        target_geom_per_img[img_id] = geom_t
        n_gt += int(geom_t.shape[0])
    if n_gt == 0:
        return None

    all_scores: List[torch.Tensor] = []
    all_pred_geom: List[torch.Tensor] = []
    pred_img_idx: List[torch.Tensor] = []
    for img_id, p in enumerate(preds):
        mask = (p["labels"] == class_id)
        if not mask.any():
            continue
        sc = p["scores"][mask]
        gm = p[geom_key][mask]
        if sc.numel() > max_dets:
            topk = torch.topk(sc, max_dets)
            sc = topk.values
            gm = gm[topk.indices]
        all_scores.append(sc)
        all_pred_geom.append(gm)
        pred_img_idx.append(torch.full((sc.numel(),), img_id, dtype=torch.long))

    if not all_scores:
        return 0.0

    scores = torch.cat(all_scores)
    geom_p = torch.cat(all_pred_geom, dim=0)
    img_idx = torch.cat(pred_img_idx)

    order = torch.argsort(scores, descending=True)
    geom_p = geom_p[order]
    img_idx = img_idx[order]

    tp = torch.zeros(geom_p.shape[0], dtype=torch.float32)
    fp = torch.zeros(geom_p.shape[0], dtype=torch.float32)
    gt_matched: Dict[int, torch.Tensor] = {
        i: torch.zeros(target_geom_per_img[i].shape[0], dtype=torch.bool)
        for i in target_geom_per_img
    }

    for i in range(geom_p.shape[0]):
        img_id = int(img_idx[i].item())
        gts = target_geom_per_img[img_id]
        if gts.shape[0] == 0:
            fp[i] = 1.0
            continue
        ious = iou_fn(geom_p[i:i + 1], gts).squeeze(0)
        ious = ious.clone()
        ious[gt_matched[img_id]] = -1.0
        best_iou, best_idx = torch.max(ious, dim=0)
        if best_iou.item() >= iou_thr:
            tp[i] = 1.0
            gt_matched[img_id][int(best_idx.item())] = True
        else:
            fp[i] = 1.0

    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)
    recall = (tp_cum / max(n_gt, 1)).numpy()
    precision = (tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-12)).numpy()
    return _ap_101_point(recall, precision)


def _coco_iou_thresholds() -> List[float]:
    return [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50, 0.55, ..., 0.95


def _gather_class_ids(
    preds: List[dict], targets: List[dict], explicit: Optional[Sequence[int]]
) -> List[int]:
    if explicit is not None:
        return list(explicit)
    class_set = set()
    for t in targets:
        class_set.update(int(c) for c in t["labels"].tolist())
    for p in preds:
        class_set.update(int(c) for c in p["labels"].tolist())
    return sorted(class_set)


def _detect_to_cpu(per_image: List[dict], geom_key: str) -> List[dict]:
    """Detach + move per-image preds/targets to CPU (bool for masks)."""
    out = []
    for d in per_image:
        item = {
            geom_key: (d[geom_key].detach().cpu().bool()
                       if geom_key == "masks" else d[geom_key].detach().cpu()),
            "labels": d["labels"].detach().cpu(),
        }
        if "scores" in d:
            item["scores"] = d["scores"].detach().cpu()
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Box metrics
# ---------------------------------------------------------------------------


class BoxMAPMetric(Metric):
    """COCO-style 3D box mean Average Precision averaged over IoU thresholds.

    Boxes are expected in ``xyzxyz`` format ``(N, 6)`` (matches
    :func:`box_iou_3d`). Convert from ``cxcyczwhd`` upstream if needed.
    """

    def __init__(
        self,
        iou_thresholds: Optional[Sequence[float]] = None,
        max_detections: int = 100,
        class_ids: Optional[Sequence[int]] = None,
    ):
        self.iou_thresholds = list(iou_thresholds) if iou_thresholds is not None else _coco_iou_thresholds()
        self.max_detections = int(max_detections)
        self.class_ids = list(class_ids) if class_ids is not None else None
        self._preds: List[dict] = []
        self._targets: List[dict] = []

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        self._preds.extend(_detect_to_cpu(outputs, geom_key="boxes"))
        self._targets.extend(_detect_to_cpu(targets, geom_key="boxes"))

    def aggregate(self) -> float:
        class_ids = _gather_class_ids(self._preds, self._targets, self.class_ids)
        if not class_ids:
            return 0.0
        ap_per_iou = []
        for iou_thr in self.iou_thresholds:
            ap_per_class = []
            for cls in class_ids:
                ap = _per_class_ap(
                    self._preds, self._targets, class_id=cls,
                    iou_thr=iou_thr, max_dets=self.max_detections,
                    geom_key="boxes", iou_fn=box_iou_3d,
                )
                if ap is not None:
                    ap_per_class.append(ap)
            if ap_per_class:
                ap_per_iou.append(sum(ap_per_class) / len(ap_per_class))
        return float(sum(ap_per_iou) / len(ap_per_iou)) if ap_per_iou else 0.0

    def reset(self):
        self._preds.clear()
        self._targets.clear()


class BoxMIoUMetric(Metric):
    """Mean IoU over greedy-matched pred/GT 3D box pairs.

    Within each image, predictions are sorted by descending score and assigned
    to the highest-IoU unmatched GT box of any class (set ``match_labels=True``
    to require label match). Only pairs with IoU >= ``iou_threshold`` count.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.0,
        match_labels: bool = False,
    ):
        self.iou_threshold = float(iou_threshold)
        self.score_threshold = float(score_threshold)
        self.match_labels = bool(match_labels)
        self._matched_ious: List[float] = []

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        for p, t in zip(outputs, targets):
            scores = p["scores"]
            keep = scores >= self.score_threshold
            boxes_p = p["boxes"][keep].detach().cpu()
            scores_p = scores[keep].detach().cpu()
            labels_p = p["labels"][keep].detach().cpu()
            boxes_t = t["boxes"].detach().cpu()
            labels_t = t["labels"].detach().cpu()
            if boxes_p.shape[0] == 0 or boxes_t.shape[0] == 0:
                continue
            order = torch.argsort(scores_p, descending=True)
            boxes_p = boxes_p[order]
            labels_p = labels_p[order]
            ious = box_iou_3d(boxes_p, boxes_t)
            if self.match_labels:
                ious = ious * (labels_p[:, None] == labels_t[None, :]).to(ious.dtype)
            matched = torch.zeros(boxes_t.shape[0], dtype=torch.bool)
            for i in range(boxes_p.shape[0]):
                row = ious[i].clone()
                row[matched] = -1.0
                best, idx = torch.max(row, dim=0)
                if best.item() >= self.iou_threshold:
                    self._matched_ious.append(float(best.item()))
                    matched[int(idx.item())] = True

    def aggregate(self) -> float:
        return float(sum(self._matched_ious) / len(self._matched_ious)) if self._matched_ious else 0.0

    def reset(self):
        self._matched_ious.clear()


class BoxF1Metric(Metric):
    """Micro-averaged F1 for 3D box detection at a fixed IoU + score threshold.

    A prediction is a true positive if it shares the GT's class label (when
    ``match_labels=True``) and matches an unused GT with IoU >= ``iou_threshold``.
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.05,
        match_labels: bool = True,
    ):
        self.iou_threshold = float(iou_threshold)
        self.score_threshold = float(score_threshold)
        self.match_labels = bool(match_labels)
        self._tp = 0
        self._fp = 0
        self._fn = 0

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        for p, t in zip(outputs, targets):
            scores = p["scores"]
            keep = scores >= self.score_threshold
            boxes_p = p["boxes"][keep].detach().cpu()
            labels_p = p["labels"][keep].detach().cpu()
            scores_p = scores[keep].detach().cpu()
            boxes_t = t["boxes"].detach().cpu()
            labels_t = t["labels"].detach().cpu()

            unmatched_gt = torch.ones(boxes_t.shape[0], dtype=torch.bool)
            tp = fp = 0
            if boxes_p.shape[0]:
                order = torch.argsort(scores_p, descending=True)
                boxes_p = boxes_p[order]
                labels_p = labels_p[order]
                if boxes_t.shape[0]:
                    ious = box_iou_3d(boxes_p, boxes_t)
                    if self.match_labels:
                        ious = ious * (labels_p[:, None] == labels_t[None, :]).to(ious.dtype)
                else:
                    ious = None
                for i in range(boxes_p.shape[0]):
                    if ious is None:
                        fp += 1
                        continue
                    row = ious[i].clone()
                    row[~unmatched_gt] = -1.0
                    best, idx = torch.max(row, dim=0)
                    if best.item() >= self.iou_threshold:
                        tp += 1
                        unmatched_gt[int(idx.item())] = False
                    else:
                        fp += 1
            self._tp += tp
            self._fp += fp
            self._fn += int(unmatched_gt.sum().item())

    def aggregate(self) -> float:
        if self._tp == 0:
            return 0.0
        precision = self._tp / max(self._tp + self._fp, 1)
        recall = self._tp / max(self._tp + self._fn, 1)
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def reset(self):
        self._tp = 0
        self._fp = 0
        self._fn = 0


# ---------------------------------------------------------------------------
# Class metrics
# ---------------------------------------------------------------------------


class ClassAPMetric(Metric):
    """Classification Average Precision.

    * Binary inputs: ``outputs`` shape ``(N,)`` of scores, ``targets`` shape
      ``(N,)`` of {0, 1}. Returns the binary AP.
    * Multi-class inputs: ``outputs`` shape ``(N, C)`` of per-class scores,
      ``targets`` shape ``(N,)`` of class ids. Returns macro-averaged AP across
      classes that appear in the targets.
    """

    def __init__(self, num_classes: Optional[int] = None):
        self.num_classes = int(num_classes) if num_classes is not None else None
        self._scores: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        self._scores.append(outputs.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def aggregate(self) -> float:
        if not self._scores:
            return 0.0
        scores = torch.cat(self._scores, dim=0)
        targets = torch.cat(self._targets, dim=0)
        if scores.dim() == 1:
            return _binary_ap(scores, targets.bool())
        C = scores.shape[1] if self.num_classes is None else int(self.num_classes)
        per_class: List[float] = []
        for c in range(C):
            tgt_c = (targets == c)
            if not tgt_c.any():
                continue
            per_class.append(_binary_ap(scores[:, c], tgt_c))
        return float(sum(per_class) / len(per_class)) if per_class else 0.0

    def reset(self):
        self._scores.clear()
        self._targets.clear()


# ---------------------------------------------------------------------------
# Mask metrics
# ---------------------------------------------------------------------------


class MaskMAPMetric(Metric):
    """COCO-style 3D mask mAP averaged over IoU thresholds.

    Two push APIs:

    * Batched (default, used by the validation flow): ``__call__(outputs,
      targets, loss=None)`` with per-image dicts containing materialized
      ``masks`` / ``labels`` / ``scores``. All masks are kept on CPU as bool
      tensors — fine for small 2D batches but blows up memory in 3D.
    * Streaming (used by :class:`InstanceSegmentationEvaluator` for 3D):
      :meth:`add_image_class` accepts already-computed per-(image, class) IoU
      rows + scores + GT count. The evaluator owns mask materialization and
      can chunk it via :class:`MaskMaterializer`. The metric only stores a
      compact tuple per (image, class) — never a 3D mask.

    Aggregation auto-detects which API was used: streaming entries take
    precedence when present.
    """

    def __init__(
        self,
        iou_thresholds: Optional[Sequence[float]] = None,
        max_detections: int = 100,
        class_ids: Optional[Sequence[int]] = None,
    ):
        self.iou_thresholds = list(iou_thresholds) if iou_thresholds is not None else _coco_iou_thresholds()
        self.max_detections = int(max_detections)
        self.class_ids = list(class_ids) if class_ids is not None else None
        # Batched accumulation (legacy path).
        self._preds: List[dict] = []
        self._targets: List[dict] = []
        # Streaming accumulation: list of dicts with keys
        # {image_id: int, class_id: int, scores: (k,), ious: (k, m), n_gt: m}.
        self._stream: List[dict] = []

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        self._preds.extend(_detect_to_cpu(outputs, geom_key="masks"))
        self._targets.extend(_detect_to_cpu(targets, geom_key="masks"))

    @torch.no_grad()
    def add_image_class(
        self,
        image_id: int,
        class_id: int,
        scores: torch.Tensor,
        ious: torch.Tensor,
        n_gt: int,
    ) -> None:
        """Push a per-(image, class) PR-curve fragment.

        Args:
            image_id: stable int id for the image (used to bucket GT matching).
            class_id: int class id.
            scores: ``(k,)`` prediction scores for this image's predictions of
                this class. Empty if the model produced no predictions of this
                class.
            ious: ``(k, n_gt)`` IoU matrix between those predictions and the
                image's GT instances of the same class. ``(k, 0)`` when this
                image has no GT for the class. ``(0, n_gt)`` when there are no
                predictions but GT exists (counts toward recall).
            n_gt: number of GT instances of this class in this image (must
                equal ``ious.shape[1]`` when ``k > 0``).
        """
        scores = scores.detach().cpu()
        ious = ious.detach().cpu().to(torch.float32)
        if scores.numel() and ious.shape[1] != n_gt:
            raise ValueError(
                f"ious shape {tuple(ious.shape)} disagrees with n_gt={n_gt} "
                "(when scores is non-empty, ious.shape[1] must equal n_gt)"
            )
        # Cap to top max_detections per (image, class) for COCO parity.
        if scores.numel() > self.max_detections:
            topk = torch.topk(scores, self.max_detections)
            scores = topk.values
            ious = ious[topk.indices]
        self._stream.append({
            "image_id": int(image_id),
            "class_id": int(class_id),
            "scores": scores,
            "ious": ious,
            "n_gt": int(n_gt),
        })

    def aggregate(self) -> float:
        if self._stream:
            return self._aggregate_stream()
        return self._aggregate_batched()

    def _aggregate_batched(self) -> float:
        class_ids = _gather_class_ids(self._preds, self._targets, self.class_ids)
        if not class_ids:
            return 0.0
        ap_per_iou = []
        for iou_thr in self.iou_thresholds:
            ap_per_class = []
            for cls in class_ids:
                ap = _per_class_ap(
                    self._preds, self._targets, class_id=cls,
                    iou_thr=iou_thr, max_dets=self.max_detections,
                    geom_key="masks", iou_fn=_pairwise_mask_iou_3d,
                )
                if ap is not None:
                    ap_per_class.append(ap)
            if ap_per_class:
                ap_per_iou.append(sum(ap_per_class) / len(ap_per_class))
        return float(sum(ap_per_iou) / len(ap_per_iou)) if ap_per_iou else 0.0

    def _aggregate_stream(self) -> float:
        # Bucket entries per class.
        per_class: Dict[int, List[dict]] = {}
        for entry in self._stream:
            per_class.setdefault(entry["class_id"], []).append(entry)
        if self.class_ids is not None:
            class_iter = [c for c in self.class_ids if c in per_class]
        else:
            class_iter = sorted(per_class.keys())
        if not class_iter:
            return 0.0

        ap_per_iou = []
        for iou_thr in self.iou_thresholds:
            ap_per_class = []
            for class_id in class_iter:
                entries = per_class[class_id]
                # Total GT for this class across all reporting images.
                n_gt_total = sum(int(e["n_gt"]) for e in entries)
                if n_gt_total == 0:
                    continue
                ap = self._stream_class_ap(entries, iou_thr=iou_thr, n_gt_total=n_gt_total)
                ap_per_class.append(ap)
            if ap_per_class:
                ap_per_iou.append(sum(ap_per_class) / len(ap_per_class))
        return float(sum(ap_per_iou) / len(ap_per_iou)) if ap_per_iou else 0.0

    @staticmethod
    def _stream_class_ap(entries: List[dict], iou_thr: float, n_gt_total: int) -> float:
        # Pool predictions across all images for this class. We keep per-row
        # IoU vectors as separate tensors (rather than padding to a rectangular
        # ``(K, max_n_gt)`` tensor) because per-image n_gt is ragged.
        flat_scores: List[torch.Tensor] = []
        flat_iou_rows: List[torch.Tensor] = []  # one (n_gt_img,) tensor per prediction
        flat_image_ids: List[int] = []
        n_gt_per_image: Dict[int, int] = {}
        for entry in entries:
            img_id = entry["image_id"]
            # Allow multiple entries for the same (image, class): keep the max.
            n_gt_per_image[img_id] = max(n_gt_per_image.get(img_id, 0), int(entry["n_gt"]))
            scores = entry["scores"]
            ious = entry["ious"]
            if scores.numel() == 0:
                continue
            flat_scores.append(scores)
            for i in range(int(scores.numel())):
                flat_iou_rows.append(ious[i])
                flat_image_ids.append(img_id)
        if not flat_scores:
            return 0.0
        scores_cat = torch.cat(flat_scores)
        order = torch.argsort(scores_cat, descending=True)

        # Per-image GT-matched bookkeeping.
        gt_matched: Dict[int, torch.Tensor] = {
            img: torch.zeros(n, dtype=torch.bool) for img, n in n_gt_per_image.items()
        }
        tp = torch.zeros(int(order.numel()), dtype=torch.float32)
        fp = torch.zeros(int(order.numel()), dtype=torch.float32)
        for rank, idx in enumerate(order.tolist()):
            img_id = flat_image_ids[idx]
            iou_row = flat_iou_rows[idx]
            n_gt_img = int(gt_matched[img_id].numel())
            if n_gt_img == 0 or iou_row.numel() == 0:
                fp[rank] = 1.0
                continue
            row = iou_row.clone()
            row[gt_matched[img_id]] = -1.0
            best, best_idx = torch.max(row, dim=0)
            if best.item() >= iou_thr:
                tp[rank] = 1.0
                gt_matched[img_id][int(best_idx.item())] = True
            else:
                fp[rank] = 1.0

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recall = (tp_cum / max(n_gt_total, 1)).numpy()
        precision = (tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-12)).numpy()
        return _ap_101_point(recall, precision)

    def reset(self):
        self._preds.clear()
        self._targets.clear()
        self._stream.clear()


class MaskMIoUMetric(Metric):
    """Mean IoU for 3D masks. Two modes:

    * ``"semantic"``: ``outputs`` and ``targets`` are per-image int label maps
      of shape ``(D, H, W)`` (or batched lists thereof). Per-class IoU is
      accumulated as ``sum(intersection) / sum(union)`` across the dataset
      (Jaccard index), then averaged over classes that appear in the data.
    * ``"instance"``: same per-image dict input as :class:`MaskMAPMetric` for
      the batched flow; greedy-match pred to GT instance masks at
      ``iou_threshold`` and average matched-pair IoUs.

    The instance mode also supports a streaming push via
    :meth:`add_matched_ious`, used by :class:`InstanceSegmentationEvaluator`
    to avoid keeping per-instance 3D masks in memory.
    """

    def __init__(
        self,
        mode: Literal["semantic", "instance"] = "semantic",
        num_classes: Optional[int] = None,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.0,
        ignore_index: Optional[int] = None,
        match_labels: bool = False,
    ):
        if mode not in ("semantic", "instance"):
            raise ValueError(f"MaskMIoUMetric mode must be 'semantic' or 'instance', got {mode!r}")
        self.mode = mode
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.iou_threshold = float(iou_threshold)
        self.score_threshold = float(score_threshold)
        self.ignore_index = ignore_index
        self.match_labels = bool(match_labels)
        self._inter: Dict[int, int] = {}
        self._union: Dict[int, int] = {}
        self._matched_ious: List[float] = []

    @torch.no_grad()
    def add_matched_ious(self, ious: Sequence[float]) -> None:
        """Append already-matched IoU values (instance mode, streaming flow)."""
        if self.mode != "instance":
            raise RuntimeError(
                "add_matched_ious is only valid for MaskMIoUMetric(mode='instance')"
            )
        for v in ious:
            v = float(v.item() if torch.is_tensor(v) else v)
            if v >= self.iou_threshold:
                self._matched_ious.append(v)

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        if self.mode == "semantic":
            self._update_semantic(outputs, targets)
        else:
            self._update_instance(outputs, targets)

    def _update_semantic(self, outputs, targets):
        pred_list = [outputs] if torch.is_tensor(outputs) else list(outputs)
        gt_list = [targets] if torch.is_tensor(targets) else list(targets)
        for pred_map, gt_map in zip(pred_list, gt_list):
            pred_map = pred_map.detach()
            gt_map = gt_map.detach()
            if self.num_classes is not None:
                classes: Sequence[int] = range(int(self.num_classes))
            else:
                classes = [int(c) for c in torch.unique(gt_map).tolist()]
            for c in classes:
                if self.ignore_index is not None and c == self.ignore_index:
                    continue
                p = (pred_map == c)
                g = (gt_map == c)
                if not g.any() and not p.any():
                    # class absent in both pred and GT -> contributes nothing
                    continue
                inter = int((p & g).sum().item())
                union = int((p | g).sum().item())
                self._inter[c] = self._inter.get(c, 0) + inter
                self._union[c] = self._union.get(c, 0) + union

    def _update_instance(self, outputs, targets):
        for p, t in zip(outputs, targets):
            scores = p["scores"]
            keep = scores >= self.score_threshold
            masks_p = p["masks"][keep].detach().cpu().bool()
            scores_p = scores[keep].detach().cpu()
            labels_p = p["labels"][keep].detach().cpu()
            masks_t = t["masks"].detach().cpu().bool()
            labels_t = t["labels"].detach().cpu()
            if masks_p.shape[0] == 0 or masks_t.shape[0] == 0:
                continue
            order = torch.argsort(scores_p, descending=True)
            masks_p = masks_p[order]
            labels_p = labels_p[order]
            ious = _pairwise_mask_iou_3d(masks_p, masks_t)
            if self.match_labels:
                ious = ious * (labels_p[:, None] == labels_t[None, :]).to(ious.dtype)
            matched = torch.zeros(masks_t.shape[0], dtype=torch.bool)
            for i in range(masks_p.shape[0]):
                row = ious[i].clone()
                row[matched] = -1.0
                best, idx = torch.max(row, dim=0)
                if best.item() >= self.iou_threshold:
                    self._matched_ious.append(float(best.item()))
                    matched[int(idx.item())] = True

    def aggregate(self) -> float:
        if self.mode == "semantic":
            ious: List[float] = []
            for c, inter in self._inter.items():
                union = self._union.get(c, 0)
                if union > 0:
                    ious.append(inter / union)
            return float(sum(ious) / len(ious)) if ious else 0.0
        return float(sum(self._matched_ious) / len(self._matched_ious)) if self._matched_ious else 0.0

    def reset(self):
        self._inter.clear()
        self._union.clear()
        self._matched_ious.clear()
