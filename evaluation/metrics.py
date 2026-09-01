import abc
import logging
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist

from cell_observatory_platform.data.structures import box_iou_3d
from cell_observatory_platform.utils.context import (
    get_world_size,
    is_torch_dist_initialized,
    reduce_values,
)


logger = logging.getLogger(__name__)

# FIXME: move to correct place
def _dist_inactive() -> bool:
    """True when there is no real multi-rank process group to reduce over."""
    return not is_torch_dist_initialized() or get_world_size() == 1


def _all_reduce_stat(t: torch.Tensor, op=None) -> torch.Tensor:
    """All-reduce a (CPU) sufficient-statistic tensor across ranks.

    NCCL only reduces CUDA tensors, so under an NCCL process group the stat is
    round-tripped through the local device; gloo/mpi reduce the CPU tensor
    directly. Returns the reduced tensor (same device as the input).
    """
    op = op if op is not None else dist.ReduceOp.SUM
    if dist.get_backend() == "nccl" and t.device.type != "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
        on_device = t.to(device)
        dist.all_reduce(on_device, op=op)
        return on_device.to(t.device)
    dist.all_reduce(t, op=op)
    return t


def _hungarian_match(
    ious: torch.Tensor, iou_threshold: float
) -> List[Tuple[int, int, float]]:
    """Max-cardinality 1:1 matching among above-threshold pairs, IoU-sum tie-break.

    Assignment runs on ``(big_m + iou)`` for pairs with ``iou >= threshold``
    (and ``> 0``), 0 otherwise, where ``big_m = min(n, m) + 1`` strictly
    exceeds the maximum achievable IoU-sum spread of any matching (each IoU
    < 1, at most ``min(n, m)`` edges). Every additional eligible edge is then
    worth more than ANY re-shuffle of edge quality, so maximum weight ==
    maximum cardinality first, maximum total IoU second — the true guarantee.

    A flat ``1.0 + iou`` bonus is NOT sufficient: an augmenting swap through k
    old edges changes weight by ``1 + sum(new) - sum(old)``, which goes
    negative once ``(k+1)*threshold < k - 1`` — e.g. a 5x5 near-diagonal chain
    (diag 0.5, off-diag 1.0, threshold 0.5) keeps a 4-match set over the
    5-match optimum. The ``> 0`` guard keeps a disjoint (zero-overlap) pair
    from ever "matching" when ``iou_threshold == 0.0``.

    Returns assigned pairs as ``(pred_idx, gt_idx, iou)`` triples (row order).
    Used for F1 and instance-mIoU; AP metrics match greedily by score instead
    (COCO standard).
    """
    if ious.numel() == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    ious32 = ious.detach().to(torch.float32)
    eligible = (ious32 >= iou_threshold) & (ious32 > 0.0)
    big_m = float(min(ious32.shape)) + 1.0
    weights = torch.where(eligible, big_m + ious32, torch.zeros_like(ious32))
    rows, cols = linear_sum_assignment(weights.cpu().numpy(), maximize=True)
    matches: List[Tuple[int, int, float]] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        if eligible[r, c]:
            matches.append((r, c, float(ious32[r, c].item())))
    return matches


class Metric(metaclass=abc.ABCMeta):
    # When True and aggregate() returns a Mapping, DatasetEvaluator.evaluate
    # writes the mapping's keys into the results dict verbatim (no
    # "{metric_key}/" prefix). Used by metrics whose sub-keys are already
    # fully-qualified log keys (e.g. BoxMIoUMetric's "box_miou" +
    # "box_match_recall") so dashboards keep the historical flat names.
    flat_result_keys: bool = False

    @abc.abstractmethod
    def __call__(self, outputs, targets, loss):
        pass

    @abc.abstractmethod
    def aggregate(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    def gather(self) -> None:
        """All-reduce sufficient statistics across ranks (no-op by default)."""
        return


# ---------------------------------------------------------------------------
# Pretraining/General metrics
# ---------------------------------------------------------------------------


class TrainLosses(Metric):
    """Per-loss-key accumulator over per-step scalar losses.

    Keeps one float per processed step. ``gather()`` pools the per-rank value
    lists via ``all_gather_object`` so ``aggregate()`` applies
    ``reduce_method`` (mean/median/min/max/sum) over the POOLED multiset --
    exact under any per-rank shard imbalance for every supported reduction
    (a per-rank pre-reduction would weight ranks, not steps, and is not even
    well-defined for median). An empty buffer aggregates to NaN, never asserts:
    an empty validation shard on one rank must not crash the run.
    """

    def __init__(self, reduce_method: str = "mean"):
        self.reduce_method = reduce_method
        self.loss_values: List[float] = []
        self._gathered = False

    def __call__(self, outputs, targets, loss):
        self.loss_values.append(
            float(loss.item()) if torch.is_tensor(loss) else float(loss)
        )

    @staticmethod
    def _merge_values(per_rank_values: List[List[float]]) -> List[float]:
        """Concatenate per-rank per-step value lists (pooled multiset)."""
        merged: List[float] = []
        for values in per_rank_values:
            merged.extend(values)
        return merged

    def gather(self) -> None:
        if self._gathered:
            return
        if _dist_inactive():
            self._gathered = True
            return
        world = get_world_size()
        buf: List = [None] * world
        dist.all_gather_object(buf, self.loss_values)
        self.loss_values = self._merge_values(buf)
        self._gathered = True

    def aggregate(self):
        if not self.loss_values:
            return float("nan")
        return reduce_values(self.reduce_method, self.loss_values)

    def reset(self):
        self.loss_values.clear()
        self._gathered = False


class ReduceBuffer:
    def __init__(self, reduce_method: str = "mean"):
        self.reduce_method = reduce_method
        self.values: List[float] = []

    def add(self, v: torch.Tensor | float):
        v = float(v.item() if torch.is_tensor(v) else v)
        self.values.append(v)

    def aggregate(self) -> float:
        # Empty shard -> NaN (skip), never an assert: one rank with no batches
        # must not crash the whole evaluation.
        if not self.values:
            return float("nan")
        return reduce_values(self.reduce_method, self.values)

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
    """Range-normalized RMSE as pooled sufficient statistics.

    Accumulates ``(sum of squared error, element count, global target min/max)``
    and computes ``sqrt(sum_sq / n) / (y_max - y_min)`` ONCE over the pooled
    statistics. Pooling before dividing is what makes the value exact under
    unequal shard sizes, and it keeps a single constant tile (range clamped to
    eps) from contributing a ~1/eps term: its squared error enters the sum, but
    the range it would have been divided by never does.

    ``gather()`` all-reduces the four statistics (SUM/SUM/MIN/MAX), so the
    cross-rank value is exact under any shard imbalance. Degenerate inputs --
    no elements, or a constant target (zero range) -- aggregate to NaN so
    checkpoint selection SKIPS them instead of comparing against a garbage
    sentinel.

    Args:
        reduce_method: accepted for config compatibility; the aggregation is
            always the exact pooled value above (there is no per-batch list
            left to reduce).
        eps: accepted for config compatibility; unused (the degenerate
            zero-range case now yields NaN instead of an eps-clamped blow-up).
    """

    def __init__(self, reduce_method: str = "mean", eps: float = 1e-8):
        self.reduce_method = reduce_method
        self.eps = eps
        self.reset()

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        diff = (outputs.detach() - targets.detach()).to(dtype=torch.float32)
        # Reduce IN float64: summing millions of voxels in float32 loses
        # low-order bits before the (previously post-reduction) widening.
        self.sum_sq += (diff.double() * diff.double()).sum().cpu()
        self.n += diff.numel()
        if targets.numel():
            y = targets.detach()
            self.y_min = torch.minimum(self.y_min, y.min().double().cpu())
            self.y_max = torch.maximum(self.y_max, y.max().double().cpu())

    def gather(self) -> None:
        if self._gathered:
            return
        if _dist_inactive():
            self._gathered = True
            return
        self.sum_sq = _all_reduce_stat(self.sum_sq)
        self.n = _all_reduce_stat(self.n)
        self.y_min = _all_reduce_stat(self.y_min, op=dist.ReduceOp.MIN)
        self.y_max = _all_reduce_stat(self.y_max, op=dist.ReduceOp.MAX)
        self._gathered = True

    def aggregate(self) -> float:
        rng = float(self.y_max - self.y_min)
        if int(self.n) == 0 or not rng > 0:
            # Degenerate (empty shard or constant target): SKIP via NaN --
            # never a clamped ~1/eps sentinel that poisons checkpoint selection.
            return float("nan")
        return float((self.sum_sq / self.n).sqrt() / rng)

    def reset(self):
        self.sum_sq = torch.zeros((), dtype=torch.float64)
        self.n = torch.zeros((), dtype=torch.int64)
        self.y_min = torch.full((), float("inf"), dtype=torch.float64)
        self.y_max = torch.full((), float("-inf"), dtype=torch.float64)
        self._gathered = False


class MAEMetric(Metric):
    """Mean absolute error as pooled sufficient statistics ``(sum |err|, n)``.

    Accumulating the two statistics and dividing once is what makes the value a
    true pooled mean: batches contribute in proportion to their element count,
    so unequal batch sizes cannot skew it. ``gather()`` all-reduces both
    statistics, extending that exactness across ranks; an empty accumulator
    aggregates to NaN.

    Args:
        reduce_method: accepted for config compatibility; the aggregation is
            always the exact pooled mean ``sum_abs / n``.
    """

    def __init__(self, reduce_method: str = "mean"):
        self.reduce_method = reduce_method
        self.reset()

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        diff = (outputs.detach() - targets.detach()).to(dtype=torch.float32)
        # float64 BEFORE the reduction (see NRMSE note).
        self.sum_abs += diff.double().abs().sum().cpu()
        self.n += diff.numel()

    def gather(self) -> None:
        if self._gathered:
            return
        if _dist_inactive():
            self._gathered = True
            return
        self.sum_abs = _all_reduce_stat(self.sum_abs)
        self.n = _all_reduce_stat(self.n)
        self._gathered = True

    def aggregate(self) -> float:
        if int(self.n) == 0:
            return float("nan")
        return float(self.sum_abs / self.n)

    def reset(self):
        self.sum_abs = torch.zeros((), dtype=torch.float64)
        self.n = torch.zeros((), dtype=torch.int64)
        self._gathered = False


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
    """Binary classification Average Precision via PR curve.

    Zero positives (or an empty batch) means there is NO evidence to rank
    against -> NaN, never 0.0 (NaN-as-skip convention; checkpoint selection
    and epoch reductions already NaN-skip). The multi-class caller filters
    zero-positive classes before calling, so the NaN never enters a macro mean.
    """
    if targets_bool.numel() == 0 or not targets_bool.any():
        return float("nan")
    # stable=True: tied scores previously ordered nondeterministically across
    # devices/backends, jittering AP in the last decimals.
    order = torch.argsort(scores, descending=True, stable=True)
    targets_sorted = targets_bool[order].to(torch.float32)
    tp = torch.cumsum(targets_sorted, dim=0)
    fp = torch.cumsum(1.0 - targets_sorted, dim=0)
    n_pos = float(targets_sorted.sum().item())
    recall = (tp / max(n_pos, 1.0)).numpy()
    precision = (tp / torch.clamp(tp + fp, min=1e-12)).numpy()
    return _ap_101_point(recall, precision)


def _pairwise_mask_iou_3d(masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between two sets of bool masks of shape (N,D,H,W) / (M,D,H,W).

    Precision note: intersections are 0/1 matmuls in float32, exact for
    integer counts up to 2**24 (= 16.7M voxels, i.e. up to a 256^3 mask).
    Inputs on this dense path are CPU tensors (``_detect_to_cpu``), so the
    global TF32 matmul setting (which truncates to a 10-bit mantissa and WOULD
    corrupt counts) does not apply; keep it that way if this ever moves to CUDA.
    """
    if masks_a.shape[0] == 0 or masks_b.shape[0] == 0:
        return masks_a.new_zeros((masks_a.shape[0], masks_b.shape[0]), dtype=torch.float32)
    a = masks_a.flatten(1).to(torch.float32)
    b = masks_b.flatten(1).to(torch.float32)
    inter = a @ b.t()
    sum_a = a.sum(dim=1, keepdim=True)
    sum_b = b.sum(dim=1, keepdim=True)
    union = sum_a + sum_b.t() - inter
    return inter / torch.clamp(union, min=1e-12)


def _collect_class_detections(
    preds: List[dict],
    targets: List[dict],
    class_id: int,
    max_dets: int,
    geom_key: str,
    iou_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Optional[dict]:
    """Score-sorted pooled detections + per-image IoU rows for one class.

    The (predictions x GT) IoU matrix of each (image, class) is computed ONCE
    here and reused for every IoU threshold: previously ``_per_class_ap``
    recomputed a one-row IoU against all GTs per detection, per threshold
    (measured ~40x slower on realistic prediction counts).

    Returns ``None`` when the class is absent from the ground truth (so it can
    be excluded from the per-class average); otherwise a dict with:
        * ``n_gt``: total GT count for the class,
        * ``iou_pad``: ``(M, max_gt)`` padded per-detection IoU matrix
          (padding ``-1.0``), globally score-sorted (descending, stable),
        * ``img_ids``: the detection's image id, aligned with ``iou_pad`` rows,
        * ``gt_sizes``: ``{img_id: n_gt_img}`` for the matched bookkeeping.
    Built ONCE per class and reused across every IoU threshold.
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
    block_rows: List[torch.Tensor] = []   # one (k_i, n_gt_img) block per image
    block_imgs: List[int] = []
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
        gts = target_geom_per_img[img_id]
        if gts.shape[0]:
            ious = iou_fn(gm, gts).to(torch.float32)  # hoisted: one matrix per (image, class)
        else:
            ious = torch.zeros((int(sc.numel()), 0), dtype=torch.float32)
        block_rows.append(ious)
        block_imgs.append(img_id)
        all_scores.append(sc)

    gt_sizes = {i: int(g.shape[0]) for i, g in target_geom_per_img.items()}
    if not all_scores:
        return {
            "n_gt": n_gt,
            "iou_pad": torch.zeros((0, 0), dtype=torch.float32),
            "img_ids": [],
            "gt_sizes": gt_sizes,
        }

    scores = torch.cat(all_scores)
    order = torch.argsort(scores, descending=True, stable=True)
    iou_pad, img_t = _pad_detection_rows(block_rows, block_imgs)
    return {
        "n_gt": n_gt,
        "iou_pad": iou_pad[order],
        "img_ids": img_t[order].tolist(),
        "gt_sizes": gt_sizes,
    }


def _pad_detection_rows(
    blocks: List[torch.Tensor], block_img_ids: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack per-(image, class) ``(k_i, n_gt_i)`` IoU blocks into ONE padded
    ``(M, max_gt)`` float32 matrix (padding ``-1.0``, below any real IoU) plus
    the per-detection image-id vector.

    O(#blocks) python ops rather than O(#detections): one padded matrix is
    built per block, so a reorder is a single tensor index instead of ~M
    allocations plus an M-element list comprehension.
    """
    M = sum(int(b.shape[0]) for b in blocks)
    max_gt = max((int(b.shape[1]) for b in blocks), default=0)
    out = torch.full((M, max_gt), -1.0, dtype=torch.float32)
    imgs = torch.empty(M, dtype=torch.int64)
    r = 0
    for b, img in zip(blocks, block_img_ids):
        k = int(b.shape[0])
        if k:
            out[r:r + k, : int(b.shape[1])] = b
            imgs[r:r + k] = img
            r += k
    return out, imgs


def _greedy_tp_fp(
    iou_pad: torch.Tensor,
    img_ids: List[int],
    gt_sizes: Dict[int, int],
    iou_thr: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequential COCO greedy consume over a rank-ordered padded IoU matrix.

    COCO semantics: first-index argmax tie-break, ``-1`` padding never wins,
    ``>= iou_thr`` counts as a match. The loop is inherently sequential (each TP
    consumes a GT), so it runs on numpy scalars rather than per-detection
    ``clone()`` / ``torch.max`` / ``.item()``.
    """
    n_det = int(iou_pad.shape[0])
    tp = np.zeros(n_det, dtype=np.float32)
    iou_np = iou_pad.numpy()
    offsets: Dict[int, int] = {}
    total = 0
    for img, n in gt_sizes.items():
        offsets[img] = total
        total += int(n)
    taken = np.zeros(total, dtype=bool)
    for i in range(n_det):
        img = img_ids[i]
        n = int(gt_sizes.get(img, 0))
        if n == 0:
            continue                                  # FP: no GT in this image
        off = offsets[img]
        masked = np.where(taken[off:off + n], -1.0, iou_np[i, :n])
        j = int(masked.argmax())
        if masked[j] >= iou_thr:
            tp[i] = 1.0
            taken[off + j] = True
    tp_t = torch.from_numpy(tp)
    return tp_t, 1.0 - tp_t


def _ap_from_class_detections(data: dict, iou_thr: float) -> float:
    """Greedy COCO matching + 101-point AP from precomputed class detections."""
    iou_pad = data["iou_pad"]
    if iou_pad.shape[0] == 0:
        return 0.0
    tp, fp = _greedy_tp_fp(iou_pad, data["img_ids"], data["gt_sizes"], iou_thr)
    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)
    recall = (tp_cum / max(data["n_gt"], 1)).numpy()
    precision = (tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-12)).numpy()
    return _ap_101_point(recall, precision)


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
    predictions matched. Thin single-threshold wrapper; the mAP aggregates use
    :func:`_batched_map` which hoists the IoU matrices across thresholds.
    """
    data = _collect_class_detections(preds, targets, class_id, max_dets, geom_key, iou_fn)
    if data is None:
        return None
    return _ap_from_class_detections(data, iou_thr)


def _batched_map(
    preds: List[dict],
    targets: List[dict],
    explicit_class_ids: Optional[Sequence[int]],
    iou_thresholds: Sequence[float],
    max_dets: int,
    geom_key: str,
    iou_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    """mAP over IoU thresholds for the batched (dense) push path.

    Collects each class's detections (and per-image IoU matrices) ONCE, then
    sweeps thresholds over the precomputed data. Numerically identical to the
    former per-threshold ``_per_class_ap`` double loop: the class set only
    depends on GT presence, which is threshold-independent.
    """
    class_ids = _gather_class_ids(preds, targets, explicit_class_ids)
    if not class_ids:
        return 0.0
    per_class_data = []
    for cls in class_ids:
        data = _collect_class_detections(preds, targets, cls, max_dets, geom_key, iou_fn)
        if data is not None:
            per_class_data.append(data)
    if not per_class_data:
        return 0.0
    ap_per_iou = []
    for iou_thr in iou_thresholds:
        aps = [_ap_from_class_detections(d, iou_thr) for d in per_class_data]
        ap_per_iou.append(sum(aps) / len(aps))
    return float(sum(ap_per_iou) / len(ap_per_iou))


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
        if geom_key == "masks" and torch.is_floating_point(d[geom_key]):
            raise ValueError(
                "masks is float — binarize explicitly (threshold) before "
                "pushing: .bool() on soft masks/logits marks every nonzero "
                "voxel foreground (p=0.01 everywhere becomes a full-volume "
                "mask). The streaming mask-source path enforces the same "
                "hard-mask contract."
            )
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

    .. note:: ``max_detections`` caps detections per **(image, class)** (see
        ``_collect_class_detections``), NOT per image across classes as true
        COCO ``maxDets`` does. With many classes per image the effective cap is
        looser than COCO's; numbers are comparable across runs of this metric
        but not strictly to pycocotools. Behavior kept for continuity —
        revisit at the next re-baseline.
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
        self._gathered = False

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        self._preds.extend(_detect_to_cpu(outputs, geom_key="boxes"))
        self._targets.extend(_detect_to_cpu(targets, geom_key="boxes"))

    @staticmethod
    def _merge_detection_lists(per_rank_preds, per_rank_targets):
        """Concatenate per-rank (preds, targets) lists into a single pooled pair.

        ``img_id`` is the list position, so plain concatenation auto-namespaces
        images across ranks (no remap needed).
        """
        merged_preds: List[dict] = []
        merged_targets: List[dict] = []
        for preds in per_rank_preds:
            merged_preds.extend(preds)
        for targets in per_rank_targets:
            merged_targets.extend(targets)
        return merged_preds, merged_targets

    def gather(self) -> None:
        if self._gathered:
            return
        if _dist_inactive():
            self._gathered = True
            return
        world = get_world_size()
        preds_buf: List = [None] * world
        targets_buf: List = [None] * world
        dist.all_gather_object(preds_buf, self._preds)
        dist.all_gather_object(targets_buf, self._targets)
        self._preds, self._targets = self._merge_detection_lists(preds_buf, targets_buf)
        self._gathered = True

    def aggregate(self) -> float:
        return _batched_map(
            self._preds, self._targets, self.class_ids,
            iou_thresholds=self.iou_thresholds, max_dets=self.max_detections,
            geom_key="boxes", iou_fn=box_iou_3d,
        )

    def reset(self):
        self._preds.clear()
        self._targets.clear()
        self._gathered = False


class BoxMIoUMetric(Metric):
    """Mean IoU over optimally matched pred/GT 3D box pairs.

    Within each image, predictions above ``score_threshold`` are matched 1:1
    to GT boxes by Hungarian assignment (``scipy.optimize.
    linear_sum_assignment``) on the IoU matrix thresholded at
    ``iou_threshold`` (set ``match_labels=True`` to require label match; a
    label mismatch zeroes the pair's IoU). Only pairs with IoU >=
    ``iou_threshold`` count.

    .. note::
        Matching switched from greedy score-sorted to Hungarian: greedy let a
        high-scored prediction steal a GT that another prediction overlapped
        better, so both the matched mean and the match count could be
        undercounted. Values will shift versus runs recorded before this
        change.

    ``aggregate`` returns a flat mapping (``flat_result_keys``):
        * ``box_miou`` -- mean IoU over matched pairs; NaN when nothing
          matched (there is no evidence, not perfect-zero quality).
        * ``box_match_recall`` -- matched pairs / total GT boxes: the
          denominator that makes the matched-pairs-only mean honest (a model
          matching one box perfectly and missing 99 scores miou 1.0 but
          recall 0.01).
    """

    flat_result_keys = True

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
        self._n_gt = 0
        self._gathered = False

    @staticmethod
    def _merge_matched_ious(per_rank_ious):
        """Concatenate per-rank matched-IoU lists (mean over the union is exact)."""
        merged: List[float] = []
        for ious in per_rank_ious:
            merged.extend(ious)
        return merged

    def gather(self) -> None:
        if self._gathered:
            return
        if _dist_inactive():
            self._gathered = True
            return
        world = get_world_size()
        buf: List = [None] * world
        dist.all_gather_object(buf, (self._matched_ious, self._n_gt))
        self._matched_ious = self._merge_matched_ious([b[0] for b in buf])
        self._n_gt = sum(int(b[1]) for b in buf)
        self._gathered = True

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        for p, t in zip(outputs, targets):
            scores = p["scores"]
            keep = scores >= self.score_threshold
            boxes_p = p["boxes"][keep].detach().cpu()
            labels_p = p["labels"][keep].detach().cpu()
            boxes_t = t["boxes"].detach().cpu()
            labels_t = t["labels"].detach().cpu()
            # GT counts toward the recall denominator even when there are no
            # predictions at all (the recall-blind path of the old metric).
            self._n_gt += int(boxes_t.shape[0])
            if boxes_p.shape[0] == 0 or boxes_t.shape[0] == 0:
                continue
            ious = box_iou_3d(boxes_p, boxes_t)
            if self.match_labels:
                ious = ious * (labels_p[:, None] == labels_t[None, :]).to(ious.dtype)
            for _, _, iou in _hungarian_match(ious, self.iou_threshold):
                self._matched_ious.append(iou)

    def aggregate(self) -> Dict[str, float]:
        n_m = len(self._matched_ious)
        return {
            "box_miou": float(sum(self._matched_ious) / n_m) if n_m else float("nan"),
            # n_gt == 0: recall of nothing is undefined evidence, not 0.0
            # (NaN-as-skip convention; checkpoint selection NaN-skips).
            "box_match_recall": (
                n_m / self._n_gt if self._n_gt > 0 else float("nan")
            ),
        }

    def reset(self):
        self._matched_ious.clear()
        self._n_gt = 0
        self._gathered = False


class BoxF1Metric(Metric):
    """Micro-averaged F1 for 3D box detection at a fixed IoU + score threshold.

    Predictions above ``score_threshold`` are matched 1:1 to GT boxes per
    image by Hungarian assignment on the IoU matrix thresholded at
    ``iou_threshold``; when ``match_labels=True`` a label mismatch zeroes the
    pair's IoU so it cannot match. Matched pairs are TPs, unmatched
    predictions FPs, unmatched GTs FNs.

    .. note::
        Matching switched from greedy score-sorted to Hungarian: greedy could
        assign a GT to a worse-overlapping, higher-scored prediction and
        undercount TPs. Values will shift versus runs recorded before this
        change. ``tp == fp == fn == 0`` (no GT, no predictions anywhere) now
        aggregates to 1.0 -- predicting nothing where nothing exists is
        perfect, not zero.
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
        self._gathered = False

    @staticmethod
    def _merge_counts(per_rank_counts):
        """Sum per-rank (tp, fp, fn) triples (micro-averaged F1 is exact)."""
        tp = fp = fn = 0
        for c_tp, c_fp, c_fn in per_rank_counts:
            tp += c_tp
            fp += c_fp
            fn += c_fn
        return tp, fp, fn

    def gather(self) -> None:
        if self._gathered:
            return
        if _dist_inactive():
            self._gathered = True
            return
        world = get_world_size()
        buf: List = [None] * world
        dist.all_gather_object(buf, (self._tp, self._fp, self._fn))
        self._tp, self._fp, self._fn = self._merge_counts(buf)
        self._gathered = True

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        for p, t in zip(outputs, targets):
            scores = p["scores"]
            keep = scores >= self.score_threshold
            boxes_p = p["boxes"][keep].detach().cpu()
            labels_p = p["labels"][keep].detach().cpu()
            boxes_t = t["boxes"].detach().cpu()
            labels_t = t["labels"].detach().cpu()

            n_p = int(boxes_p.shape[0])
            n_t = int(boxes_t.shape[0])
            tp = 0
            if n_p and n_t:
                ious = box_iou_3d(boxes_p, boxes_t)
                if self.match_labels:
                    ious = ious * (labels_p[:, None] == labels_t[None, :]).to(ious.dtype)
                tp = len(_hungarian_match(ious, self.iou_threshold))
            self._tp += tp
            self._fp += n_p - tp
            self._fn += n_t - tp

    def aggregate(self) -> float:
        if self._tp == 0:
            # No GT and no predictions anywhere is a PERFECT empty prediction,
            # not a zero-quality one.
            return 1.0 if (self._fp == 0 and self._fn == 0) else 0.0
        precision = self._tp / max(self._tp + self._fp, 1)
        recall = self._tp / max(self._tp + self._fn, 1)
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def reset(self):
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._gathered = False


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
        self._gathered = False

    @staticmethod
    def _merge_score_target_lists(per_rank_scores, per_rank_targets):
        """Concatenate per-rank score/target tensor lists into a single pooled pair."""
        merged_scores: List[torch.Tensor] = []
        merged_targets: List[torch.Tensor] = []
        for scores in per_rank_scores:
            merged_scores.extend(scores)
        for targets in per_rank_targets:
            merged_targets.extend(targets)
        return merged_scores, merged_targets

    def gather(self) -> None:
        if self._gathered:
            return
        if _dist_inactive():
            self._gathered = True
            return
        world = get_world_size()
        # Move to CPU before gathering so the object payload is device-agnostic.
        local_scores = [s.detach().cpu() for s in self._scores]
        local_targets = [t.detach().cpu() for t in self._targets]
        scores_buf: List = [None] * world
        targets_buf: List = [None] * world
        dist.all_gather_object(scores_buf, local_scores)
        dist.all_gather_object(targets_buf, local_targets)
        self._scores, self._targets = self._merge_score_target_lists(scores_buf, targets_buf)
        self._gathered = True

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        self._scores.append(outputs.detach().cpu())
        self._targets.append(targets.detach().cpu())

    def aggregate(self) -> float:
        if not self._scores:
            # NaN-as-skip convention (matches _binary_ap and every other
            # detection metric): 0.0 would read as a real worst score to
            # checkpoint selection and macro means.
            return float("nan")
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
        return float(sum(per_class) / len(per_class)) if per_class else float("nan")

    def reset(self):
        self._scores.clear()
        self._targets.clear()
        self._gathered = False


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

    .. note:: ``max_detections`` caps detections per **(image, class)** — both
        in ``add_image_class`` and in the batched ``_collect_class_detections``
        path — NOT per image across classes as true COCO ``maxDets`` does.
        Comparable across runs of this metric, not strictly to pycocotools.
        Behavior kept for continuity — revisit at the next re-baseline.
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
        # Track pushed (image_id, class_id) keys to reject duplicates that would
        # produce inconsistent n_gt and crash deep inside _aggregate_stream.
        self._seen_keys: set = set()
        self._gathered = False

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
        key = (int(image_id), int(class_id))
        if key in self._seen_keys:
            prev_n_gt = next(
                e["n_gt"] for e in self._stream
                if e["image_id"] == key[0] and e["class_id"] == key[1]
            )
            raise ValueError(
                f"Duplicate add_image_class for image_id={key[0]}, "
                f"class_id={key[1]}: already pushed with n_gt={prev_n_gt}, "
                f"now pushing n_gt={int(n_gt)}. Each (image_id, class_id) may "
                "be pushed at most once."
            )
        scores = scores.detach().cpu()
        ious = ious.detach().cpu().to(torch.float32)
        if ious.shape[0] != scores.numel():
            raise ValueError(
                f"ious rows {ious.shape[0]} must equal number of scores "
                f"{scores.numel()} (each prediction needs exactly one IoU row; "
                "a preds-but-no-GT class must pass (k, 0), not (0, 0))."
            )
        if scores.numel() and ious.shape[1] != n_gt:
            raise ValueError(
                f"ious shape {tuple(ious.shape)} disagrees with n_gt={n_gt} "
                "(when scores is non-empty, ious.shape[1] must equal n_gt)"
            )
        # Cap to top max_detections per (image, class). NOTE: true COCO maxDets
        # caps per image ACROSS classes; this per-(image, class) cap is looser
        # (documented, kept for continuity — see class docstring).
        if scores.numel() > self.max_detections:
            topk = torch.topk(scores, self.max_detections)
            scores = topk.values
            ious = ious[topk.indices]
        # IoU fragments stored fp16: IoU needs far less than fp16 precision
        # (quantization ~1e-3 vs the 101-point AP bin width), and fp32 fragments
        # were ~600 MB per 1000 images per rank in both this metric and
        # PredictedIoUEvalMetric. (Sharing one fragment store between the two
        # metrics is deliberately DEFERRED — evaluator-level wiring.)
        self._stream.append({
            "image_id": int(image_id),
            "class_id": int(class_id),
            "scores": scores,
            "ious": ious.to(torch.float16),
            "n_gt": int(n_gt),
        })
        self._seen_keys.add(key)

    @staticmethod
    def _merge_streams(per_rank_streams: List[List[dict]]) -> List[dict]:
        """Pool per-rank streaming entries, namespacing ``image_id`` by rank.

        Each rank's per-image GT bucket must stay disjoint from every other
        rank's, so we remap rank ``r``'s ``image_id`` to ``r * stride +
        image_id`` with ``stride = 1 + global max image_id``. ``image_id`` is
        only ever used as an opaque dict key in ``_stream_class_ap``, so this
        remap is sound and ``n_gt`` (summed across images) is unaffected.
        """
        max_image_id = -1
        for stream in per_rank_streams:
            for entry in stream:
                if entry["image_id"] > max_image_id:
                    max_image_id = entry["image_id"]
        stride = max_image_id + 1  # >= 1 whenever any entry exists
        merged: List[dict] = []
        for rank, stream in enumerate(per_rank_streams):
            offset = rank * stride
            for entry in stream:
                new_entry = dict(entry)
                new_entry["image_id"] = offset + entry["image_id"]
                merged.append(new_entry)
        return merged

    @staticmethod
    def _merge_batched(per_rank_preds, per_rank_targets):
        """Concatenate per-rank batched preds/targets (list position = img_id)."""
        merged_preds: List[dict] = []
        merged_targets: List[dict] = []
        for preds in per_rank_preds:
            merged_preds.extend(preds)
        for targets in per_rank_targets:
            merged_targets.extend(targets)
        return merged_preds, merged_targets

    def gather(self) -> None:
        if self._gathered:
            return
        if _dist_inactive():
            self._gathered = True
            return
        world = get_world_size()
        # Exactly one push API is populated in practice (streaming for 3D
        # evaluators, batched for the dense flow), so first agree on which
        # collectives are needed via a tiny mode-flag gather; the empty
        # payloads' all_gather_objects are then skipped SYMMETRICALLY (the
        # decision uses the global flags, so every rank runs the same set of
        # collectives even when its local buffers are empty).
        flags_buf: List = [None] * world
        dist.all_gather_object(
            flags_buf, (bool(self._stream), bool(self._preds) or bool(self._targets))
        )
        any_stream = any(f[0] for f in flags_buf)
        any_batched = any(f[1] for f in flags_buf)
        if any_stream:
            stream_buf: List = [None] * world
            dist.all_gather_object(stream_buf, self._stream)
            self._stream = self._merge_streams(stream_buf)
        if any_batched:
            preds_buf: List = [None] * world
            targets_buf: List = [None] * world
            dist.all_gather_object(preds_buf, self._preds)
            dist.all_gather_object(targets_buf, self._targets)
            self._preds, self._targets = self._merge_batched(preds_buf, targets_buf)
        # Rebuild the dup-key guard over the now-namespaced stream.
        self._seen_keys = {
            (e["image_id"], e["class_id"]) for e in self._stream
        }
        self._gathered = True

    def aggregate(self) -> float:
        if self._stream:
            return self._aggregate_stream()
        return self._aggregate_batched()

    def _aggregate_batched(self) -> float:
        return _batched_map(
            self._preds, self._targets, self.class_ids,
            iou_thresholds=self.iou_thresholds, max_dets=self.max_detections,
            geom_key="masks", iou_fn=_pairwise_mask_iou_3d,
        )

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
    def _stream_class_ap(
        entries: List[dict],
        iou_thr: float,
        n_gt_total: int,
        rank_fn: Optional[Callable[[dict], torch.Tensor]] = None,
    ) -> float:
        """Pooled per-class AP at one IoU threshold from streaming fragments.

        ``rank_fn`` maps an ``entry`` dict to a ``(k,)`` per-detection ranking
        vector used ONLY for the global descending stable sort. It defaults to
        the entry's class ``scores`` (standard COCO ranking). The matching,
        TP/FP bookkeeping, recall denominator (``n_gt_total``), and 101-point
        interpolation are unaffected by the choice of ``rank_fn`` — only the
        order in which detections are greedily matched changes. Callers that
        want Mask Scoring R-CNN-style rankings (pred-IoU, score*pred-IoU) pass
        a ``rank_fn`` that returns the corresponding per-detection vector.
        """
        if rank_fn is None:
            rank_fn = lambda e: e["scores"]
        # Pool predictions across all images for this class into ONE padded
        # (M, max_gt) IoU matrix at ENTRY granularity: O(#entries) block copies,
        # rather than ~M tensor allocations and a per-detection python loop of
        # clone()/max()/.item() per IoU threshold per ranking.
        flat_rank: List[torch.Tensor] = []  # per-detection ranking vector segments
        block_rows: List[torch.Tensor] = []
        block_imgs: List[int] = []
        n_gt_per_image: Dict[int, int] = {}
        for entry in entries:
            img_id = entry["image_id"]
            # Allow multiple entries for the same (image, class): keep the max.
            # (A duplicate entry with FEWER GT columns pads with -1.0: those GT
            # slots are simply unreachable from its rows.)
            n_gt_per_image[img_id] = max(n_gt_per_image.get(img_id, 0), int(entry["n_gt"]))
            scores = entry["scores"]
            if scores.numel() == 0:
                continue
            flat_rank.append(rank_fn(entry).reshape(-1))
            block_rows.append(entry["ious"].to(torch.float32))
            block_imgs.append(img_id)
        if not flat_rank:
            return 0.0
        rank_cat = torch.cat(flat_rank).to(torch.float32)
        order = torch.argsort(rank_cat, descending=True, stable=True)

        iou_pad, img_t = _pad_detection_rows(block_rows, block_imgs)
        tp, fp = _greedy_tp_fp(
            iou_pad[order], img_t[order].tolist(), n_gt_per_image, iou_thr
        )

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recall = (tp_cum / max(n_gt_total, 1)).numpy()
        precision = (tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-12)).numpy()
        return _ap_101_point(recall, precision)

    def reset(self):
        self._preds.clear()
        self._targets.clear()
        self._stream.clear()
        self._seen_keys.clear()
        self._gathered = False


class PredictedIoUEvalMetric(Metric):
    """Model-agnostic evaluation of a predicted-IoU (self-assessed mask-quality)
    head, SAM2 / Mask Scoring R-CNN style, distinct from the classification
    score.

    All three measurements below are derived from a single per-detection
    sufficient statistic — ``pred_ious`` — carried in the SAME streaming
    ``_stream`` that :class:`MaskMAPMetric` already gathers across ranks, so
    cross-rank reduction is automatic and exact (it rides the existing
    rank-namespacing merge). NO dense masks are stored or gathered; only
    ``(scores, ious, n_gt, pred_ious)`` per ``(image, class)``.

    A prediction's *true quality* is its BEST IoU to ANY GT (``ious[i].max()``;
    ``0.0`` for an empty IoU row, i.e. an image with no GT of that class).

    :meth:`aggregate` returns a flat ``Dict[str, float]`` with these keys:

    Calibration (does ``pred_iou`` track ``true_iou``?):
        * ``iou_head_mae``    — mean abs error ``|pred_iou - true_iou|``.
        * ``iou_head_rmse``   — root-mean-square error.
        * ``iou_head_spearman`` — Spearman rank correlation.
        * ``iou_head_pearson``  — Pearson correlation.
      (Each ``0.0`` when fewer than 2 pooled detections exist.)

    Selection / quality-vs-coverage (raise a ``pred_iou`` threshold ``t`` and
    keep predictions with ``pred_iou >= t``; report the true quality of the
    retained set) for each ``t`` in ``pred_iou_thresholds``:
        * ``true_miou@{t}``  — mean true_iou of kept set (``0.0`` if none).
        * ``precision@{t}``  — fraction of kept with ``true_iou >= match_iou_threshold``.
        * ``coverage@{t}``   — ``kept / total``.
      Plus two coverage-integrated areas (the key comparison):
        * ``selection_auc_prediou`` — area under (true_miou-of-retained vs
          coverage) when ranking detections by ``pred_iou`` descending.
        * ``selection_auc_score``   — same, ranking by class score descending
          (the baseline selector).

    Ranked AP (Mask Scoring R-CNN lever — which ranking key maximizes mask
    AP? matching/TP-FP use the true ``ious``; only the RANKING changes):
        * ``map_rank_score``            — rank by class score (numerically
          equals a standard :class:`MaskMAPMetric` on the same fragments).
        * ``map_rank_prediou``          — rank by ``pred_iou``.
        * ``map_rank_score_x_prediou``  — rank by ``score * pred_iou``.
    """

    def __init__(
        self,
        iou_thresholds: Optional[Sequence[float]] = None,
        pred_iou_thresholds: Optional[Sequence[float]] = None,
        match_iou_threshold: float = 0.5,
        max_detections: int = 100,
        class_ids: Optional[Sequence[int]] = None,
    ):
        self.iou_thresholds = (
            list(iou_thresholds) if iou_thresholds is not None
            else _coco_iou_thresholds()
        )
        self.pred_iou_thresholds = (
            list(pred_iou_thresholds) if pred_iou_thresholds is not None
            else [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95
        )
        self.match_iou_threshold = float(match_iou_threshold)
        self.max_detections = int(max_detections)
        self.class_ids = list(class_ids) if class_ids is not None else None
        # Streaming accumulation: list of dicts with keys
        # {image_id, class_id, scores: (k,), ious: (k, m), n_gt: m, pred_ious: (k,)}.
        self._stream: List[dict] = []
        self._seen_keys: set = set()
        self._gathered = False

    @torch.no_grad()
    def __call__(self, outputs, targets, loss=None):
        # No batched push path: this metric requires the pred-IoU head output,
        # which the streaming evaluator supplies via add_image_class.
        return

    @torch.no_grad()
    def add_image_class(
        self,
        image_id: int,
        class_id: int,
        scores: torch.Tensor,
        ious: torch.Tensor,
        n_gt: int,
        pred_ious: torch.Tensor,
    ) -> None:
        """Push a per-(image, class) fragment, REQUIRING the pred-IoU head.

        Mirrors :meth:`MaskMAPMetric.add_image_class` but additionally carries
        ``pred_ious`` ``(k,)`` — the model's self-assessed mask quality for each
        of this image's predictions of this class, aligned with ``scores``. The
        ``max_detections`` top-k subselection is applied jointly to ``scores``,
        ``ious`` and ``pred_ious`` so they stay aligned.
        """
        key = (int(image_id), int(class_id))
        if key in self._seen_keys:
            prev_n_gt = next(
                e["n_gt"] for e in self._stream
                if e["image_id"] == key[0] and e["class_id"] == key[1]
            )
            raise ValueError(
                f"Duplicate add_image_class for image_id={key[0]}, "
                f"class_id={key[1]}: already pushed with n_gt={prev_n_gt}, "
                f"now pushing n_gt={int(n_gt)}. Each (image_id, class_id) may "
                "be pushed at most once."
            )
        scores = scores.detach().cpu()
        ious = ious.detach().cpu().to(torch.float32)
        pred_ious = pred_ious.detach().cpu().to(torch.float32)
        # Non-finite predicted IoU (e.g. from a diverged/degraded head) is mapped
        # to lowest quality (0.0) so calibration and selection AGREE on it: it is
        # dropped by the ``pred_iou >= t`` selection gate AND scored as
        # "predicted-bad" in calibration, instead of poisoning mae/rmse/pearson
        # with NaN while spearman/selection silently ignore it.
        pred_ious = torch.nan_to_num(pred_ious, nan=0.0, posinf=1.0, neginf=0.0)
        if pred_ious.shape != scores.shape:
            raise ValueError(
                f"pred_ious shape {tuple(pred_ious.shape)} must equal scores "
                f"shape {tuple(scores.shape)}."
            )
        if ious.shape[0] != scores.numel():
            raise ValueError(
                f"ious rows {ious.shape[0]} must equal number of scores "
                f"{scores.numel()} (each prediction needs exactly one IoU row; "
                "a preds-but-no-GT class must pass (k, 0), not (0, 0))."
            )
        if scores.numel() and ious.shape[1] != n_gt:
            raise ValueError(
                f"ious shape {tuple(ious.shape)} disagrees with n_gt={n_gt} "
                "(when scores is non-empty, ious.shape[1] must equal n_gt)"
            )
        # NOTE: unlike MaskMAPMetric, we deliberately do NOT apply a score-based
        # top-k cap here. Calibration (b) and selection (a) assess EVERY proposal,
        # and the COCO ``max_detections`` cap for ranked AP (c) is applied
        # per-ranking inside ``_ranked_map`` — capping by ``scores`` up front would
        # bias the pred_iou ranking by dropping high-pred-iou/low-score proposals
        # before they can be ranked.
        # fp16 fragments: same rationale (and same deferred sharing note) as
        # MaskMAPMetric.add_image_class.
        self._stream.append({
            "image_id": int(image_id),
            "class_id": int(class_id),
            "scores": scores,
            "ious": ious.to(torch.float16),
            "n_gt": int(n_gt),
            "pred_ious": pred_ious,
        })
        self._seen_keys.add(key)

    def gather(self) -> None:
        if self._gathered:
            return
        if _dist_inactive():
            self._gathered = True
            return
        world = get_world_size()
        stream_buf: List = [None] * world
        dist.all_gather_object(stream_buf, self._stream)
        # Reuse MaskMAPMetric's rank-namespacing merge; pred_ious rides along in
        # each entry dict (dict() copy in _merge_streams preserves all keys).
        self._stream = MaskMAPMetric._merge_streams(stream_buf)
        self._seen_keys = {
            (e["image_id"], e["class_id"]) for e in self._stream
        }
        self._gathered = True

    @staticmethod
    def _true_quality(entry: dict) -> torch.Tensor:
        """Per-detection best IoU to ANY GT (``(k,)``); 0 for empty IoU rows."""
        ious = entry["ious"]
        k = int(entry["scores"].numel())
        if k == 0:
            return torch.zeros(0, dtype=torch.float32)
        if ious.numel() == 0 or ious.shape[1] == 0:
            return torch.zeros(k, dtype=torch.float32)
        return ious.max(dim=1).values.to(torch.float32)

    @staticmethod
    def _selection_auc(selector: torch.Tensor, true_iou: torch.Tensor) -> float:
        """Area under (mean-true-IoU-of-retained vs coverage).

        Sort detections by ``selector`` descending; sweeping the retained
        prefix from coverage 0->1 traces the curve. Integrate the running
        mean true IoU of the retained prefix against coverage via the
        trapezoid rule (coverage spacing is uniform at ``1/N``).
        """
        n = int(selector.numel())
        if n == 0:
            return 0.0
        order = torch.argsort(selector, descending=True, stable=True)
        ti = true_iou[order].to(torch.float64)
        running_mean = torch.cumsum(ti, dim=0) / torch.arange(
            1, n + 1, dtype=torch.float64
        )
        coverage = torch.arange(1, n + 1, dtype=torch.float64) / n
        # Trapezoid integral with an implicit (coverage=0, value=running_mean[0])
        # left anchor so the curve starts at the first retained detection.
        x = torch.cat([torch.zeros(1, dtype=torch.float64), coverage])
        y = torch.cat([running_mean[:1], running_mean])
        return float(torch.trapz(y, x).item())

    def aggregate(self) -> Dict[str, float]:
        out: Dict[str, float] = {}

        # Pool every detection across the gathered stream. Empty-score entries
        # (GT-only, no predictions) contribute nothing to the per-detection
        # pools but their n_gt still matters for ranked-AP recall.
        true_iou_segs: List[torch.Tensor] = []
        pred_iou_segs: List[torch.Tensor] = []
        score_segs: List[torch.Tensor] = []
        for entry in self._stream:
            if int(entry["scores"].numel()) == 0:
                continue
            true_iou_segs.append(self._true_quality(entry))
            pred_iou_segs.append(entry["pred_ious"].reshape(-1).to(torch.float32))
            score_segs.append(entry["scores"].reshape(-1).to(torch.float32))

        if true_iou_segs:
            true_iou = torch.cat(true_iou_segs)
            pred_iou = torch.cat(pred_iou_segs)
            score = torch.cat(score_segs)
        else:
            true_iou = torch.zeros(0, dtype=torch.float32)
            pred_iou = torch.zeros(0, dtype=torch.float32)
            score = torch.zeros(0, dtype=torch.float32)
        n = int(true_iou.numel())

        # ---- Calibration (b) ----
        # ``iou_head_n`` reports the pooled detection count so a NaN below is
        # attributable (0/1 detections, or constant vectors for the
        # correlations) instead of silently looking like a perfect score.
        out["iou_head_n"] = float(n)
        if n >= 2:
            err = pred_iou - true_iou
            out["iou_head_mae"] = float(err.abs().mean().item())
            out["iou_head_rmse"] = float(torch.sqrt((err ** 2).mean()).item())
            # _pearson/_spearman themselves return NaN on degenerate
            # (zero-variance) inputs: an undefined correlation must never be
            # reported as 0.0 -- and certainly not as anything comparable.
            out["iou_head_pearson"] = self._pearson(pred_iou, true_iou)
            out["iou_head_spearman"] = self._spearman(pred_iou, true_iou)
        else:
            # Fewer than 2 pooled detections: calibration is undefined, NaN
            # (skip), never 0.0 (which reads as "perfectly calibrated" for MAE
            # and "uncorrelated" for the correlations).
            out["iou_head_mae"] = float("nan")
            out["iou_head_rmse"] = float("nan")
            out["iou_head_pearson"] = float("nan")
            out["iou_head_spearman"] = float("nan")

        # ---- Selection / quality-vs-coverage (a) ----
        total = max(n, 1)
        for t in self.pred_iou_thresholds:
            keep = pred_iou >= t
            n_keep = int(keep.sum().item())
            kept_true = true_iou[keep]
            out[f"true_miou@{t}"] = (
                float(kept_true.mean().item()) if n_keep > 0 else 0.0
            )
            out[f"precision@{t}"] = (
                float((kept_true >= self.match_iou_threshold).float().mean().item())
                if n_keep > 0 else 0.0
            )
            out[f"coverage@{t}"] = float(n_keep) / float(total)
        out["selection_auc_prediou"] = self._selection_auc(pred_iou, true_iou)
        out["selection_auc_score"] = self._selection_auc(score, true_iou)

        # ---- Ranked AP (c) ----
        out["map_rank_score"] = self._ranked_map(lambda e: e["scores"])
        out["map_rank_prediou"] = self._ranked_map(lambda e: e["pred_ious"])
        out["map_rank_score_x_prediou"] = self._ranked_map(
            lambda e: e["scores"] * e["pred_ious"]
        )
        return out

    def _ranked_map(self, rank_fn: Callable[[dict], torch.Tensor]) -> float:
        """mAP (avg over iou_thresholds, avg over classes) using ``rank_fn`` as
        the per-detection ranking vector in :meth:`MaskMAPMetric._stream_class_ap`.
        """
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
                n_gt_total = sum(int(e["n_gt"]) for e in entries)
                if n_gt_total == 0:
                    continue
                # Apply the per-image max_detections cap BY THE RANKING BEING
                # evaluated (COCO parity), so each ranking variant keeps its own
                # top-max_detections rather than a single score-based subset.
                capped = [self._cap_entry_by_rank(e, rank_fn) for e in entries]
                ap = MaskMAPMetric._stream_class_ap(
                    capped, iou_thr=iou_thr, n_gt_total=n_gt_total,
                    rank_fn=rank_fn,
                )
                ap_per_class.append(ap)
            if ap_per_class:
                ap_per_iou.append(sum(ap_per_class) / len(ap_per_class))
        return float(sum(ap_per_iou) / len(ap_per_iou)) if ap_per_iou else 0.0

    def _cap_entry_by_rank(
        self, entry: dict, rank_fn: Callable[[dict], torch.Tensor]
    ) -> dict:
        """Truncate one (image, class) entry to top-``max_detections`` detections
        by ``rank_fn`` (n_gt is unchanged — it counts GT, not predictions)."""
        if int(entry["scores"].numel()) <= self.max_detections:
            return entry
        rank = rank_fn(entry).reshape(-1)
        keep = torch.topk(rank, self.max_detections).indices
        capped = dict(entry)
        capped["scores"] = entry["scores"][keep]
        capped["pred_ious"] = entry["pred_ious"][keep]
        capped["ious"] = entry["ious"][keep]
        return capped

    @staticmethod
    def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
        a = a.to(torch.float64)
        b = b.to(torch.float64)
        a = a - a.mean()
        b = b - b.mean()
        denom = torch.sqrt((a ** 2).sum()) * torch.sqrt((b ** 2).sum())
        if denom.item() <= 1e-12:
            # Zero variance on either side: the correlation is UNDEFINED
            # (scipy also returns NaN here), not 0.0.
            return float("nan")
        return float((a * b).sum().item() / denom.item())

    @staticmethod
    def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
        # Spearman = Pearson on ranks (average-rank ties via argsort-of-argsort
        # is exact only for distinct values; use scipy-free average ranking).
        ra = PredictedIoUEvalMetric._avg_rank(a)
        rb = PredictedIoUEvalMetric._avg_rank(b)
        return PredictedIoUEvalMetric._pearson(ra, rb)

    @staticmethod
    def _avg_rank(x: torch.Tensor) -> torch.Tensor:
        """Average (fractional) ranks of ``x``, handling ties."""
        x = x.to(torch.float64)
        n = int(x.numel())
        order = torch.argsort(x, stable=True)
        ranks = torch.empty(n, dtype=torch.float64)
        ranks[order] = torch.arange(n, dtype=torch.float64)
        # Average ranks within tied groups.
        sorted_vals = x[order]
        i = 0
        while i < n:
            j = i + 1
            while j < n and sorted_vals[j] == sorted_vals[i]:
                j += 1
            if j - i > 1:
                avg = (i + j - 1) / 2.0
                ranks[order[i:j]] = avg
            i = j
        return ranks

    def reset(self):
        self._stream.clear()
        self._seen_keys.clear()
        self._gathered = False


class MaskMIoUMetric(Metric):
    """Mean IoU for 3D masks. Two modes:

    * ``"semantic"``: ``outputs`` and ``targets`` are per-image int label maps
      of shape ``(D, H, W)`` (or batched lists thereof). Per-class IoU is
      accumulated as ``sum(intersection) / sum(union)`` across the dataset
      (Jaccard index), then averaged over classes that appear in the data.
      ``ignore_index`` voxels (by GT label) are masked out of EVERY class's
      intersection/union; ``exclude_from_mean`` drops one class (typically
      background) from the final class average only. An empty accumulation
      (all-ignore / all-background data) aggregates to NaN with a warning,
      never 0.0.
    * ``"instance"``: same per-image dict input as :class:`MaskMAPMetric` for
      the batched flow; predictions above ``score_threshold`` are Hungarian-
      matched to GT instance masks on the IoU matrix thresholded at
      ``iou_threshold``, and matched-pair IoUs are averaged.

      .. note::
          Instance matching switched from greedy score-sorted to Hungarian
          (optimal 1:1); values will shift versus runs recorded before this
          change. ``aggregate`` returns a flat mapping (``flat_result_keys``):
          ``mask_miou`` (matched-pair mean IoU, NaN when nothing matched) and
          ``mask_match_recall`` (matched pairs / total GT instances -- the
          denominator that makes the matched-pairs-only mean honest).

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
        exclude_from_mean: Optional[int] = None,
        match_labels: bool = False,
    ):
        if mode not in ("semantic", "instance"):
            raise ValueError(f"MaskMIoUMetric mode must be 'semantic' or 'instance', got {mode!r}")
        self.mode = mode
        # Instance-mode aggregate() returns already-qualified flat keys
        # ("mask_miou" / "mask_match_recall"); semantic mode stays scalar.
        self.flat_result_keys = (mode == "instance")
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.iou_threshold = float(iou_threshold)
        self.score_threshold = float(score_threshold)
        # Two DISTINCT concepts (semantic mode):
        #   ignore_index     -> GT voxels with this label are unlabeled/void and
        #                       leave every class's intersection AND union.
        #   exclude_from_mean-> the class is skipped entirely: not scored and
        #                       absent from the mean (background convention).
        #                       Other classes are unaffected -- per-class binary
        #                       IoU is independent, so skip == score-then-drop.
        self.ignore_index = ignore_index
        self.exclude_from_mean = exclude_from_mean
        self.match_labels = bool(match_labels)
        self._inter: Dict[int, int] = {}
        self._union: Dict[int, int] = {}
        self._matched_ious: List[float] = []
        self._n_gt = 0
        self._gathered = False

    @staticmethod
    def _merge_matched_ious(per_rank_ious):
        """Concatenate per-rank matched-IoU lists (instance mode)."""
        merged: List[float] = []
        for ious in per_rank_ious:
            merged.extend(ious)
        return merged

    @staticmethod
    def _merge_inter_union(per_rank_inter, per_rank_union):
        """Sum per-class intersection/union across ranks (semantic Jaccard).

        Returns merged ``(inter, union)`` dicts so aggregate() computes
        ``Sum(inter) / Sum(union)`` per class (NOT a mean of per-rank ratios).
        """
        inter: Dict[int, int] = {}
        union: Dict[int, int] = {}
        for d in per_rank_inter:
            for c, v in d.items():
                inter[c] = inter.get(c, 0) + v
        for d in per_rank_union:
            for c, v in d.items():
                union[c] = union.get(c, 0) + v
        return inter, union

    def gather(self) -> None:
        if self._gathered:
            return
        if _dist_inactive():
            self._gathered = True
            return
        world = get_world_size()
        if self.mode == "instance":
            buf: List = [None] * world
            dist.all_gather_object(buf, (self._matched_ious, self._n_gt))
            self._matched_ious = self._merge_matched_ious([b[0] for b in buf])
            self._n_gt = sum(int(b[1]) for b in buf)
        else:
            inter_buf: List = [None] * world
            union_buf: List = [None] * world
            dist.all_gather_object(inter_buf, self._inter)
            dist.all_gather_object(union_buf, self._union)
            self._inter, self._union = self._merge_inter_union(inter_buf, union_buf)
        self._gathered = True

    @torch.no_grad()
    def add_matched_ious(self, ious: Sequence[float], n_gt: int = 0) -> None:
        """Append already-matched IoU values (instance mode, streaming flow).

        ``n_gt`` is the number of GT instances the pushing image contributed
        (matched or not); it feeds the ``mask_match_recall`` denominator, so
        streaming callers must push it even when ``ious`` is empty.
        """
        if self.mode != "instance":
            raise RuntimeError(
                "add_matched_ious is only valid for MaskMIoUMetric(mode='instance')"
            )
        self._n_gt += int(n_gt)
        for v in ious:
            v = float(v.item() if torch.is_tensor(v) else v)
            # `and v > 0.0`: at iou_threshold=0.0 the >= alone would admit a
            # zero-overlap (disjoint) pair as a "match".
            if v >= self.iou_threshold and v > 0.0:
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
                # Union of GT and PRED classes: a hallucinated class (predicted
                # but absent from GT) must be scored (IoU 0), not silently
                # dropped from the mean. ignore_index is NOT a class: without
                # this exclusion a prediction emitting the ignore label creates
                # a spurious IoU-0 "class" (its GT side is void by definition)
                # that drags the mean down. NOTE: the derived set makes the
                # mean's denominator data-dependent -- an explicit num_classes
                # is the honest fixed-taxonomy denominator.
                classes = sorted(
                    (set(int(c) for c in torch.unique(gt_map).tolist())
                     | set(int(c) for c in torch.unique(pred_map).tolist()))
                    - ({int(self.ignore_index)} if self.ignore_index is not None else set())
                )
            # ignore_index masks VOXELS: an ignore-labelled GT voxel leaves
            # every class's intersection and union (it is unlabeled, so neither
            # a hit nor a miss for any class).
            valid = (gt_map != self.ignore_index) if self.ignore_index is not None else None
            for c in classes:
                if self.exclude_from_mean is not None and c == self.exclude_from_mean:
                    # Fully-labeled class dropped from the MEAN only (the old
                    # `ignore_index` behavior, renamed honestly).
                    continue
                p = (pred_map == c)
                g = (gt_map == c)
                if valid is not None:
                    p = p & valid
                    g = g & valid
                if not g.any() and not p.any():
                    # class absent in both pred and GT -> contributes nothing
                    continue
                inter = int((p & g).sum().item())
                union = int((p | g).sum().item())
                self._inter[c] = self._inter.get(c, 0) + inter
                self._union[c] = self._union.get(c, 0) + union

    def _update_instance(self, outputs, targets):
        for p, t in zip(outputs, targets):
            for side, m in (("pred", p["masks"]), ("target", t["masks"])):
                if torch.is_floating_point(m):
                    raise ValueError(
                        f"{side} masks is float — binarize explicitly before "
                        "pushing (.bool() on soft masks marks every nonzero "
                        "voxel foreground)."
                    )
            scores = p["scores"]
            keep = scores >= self.score_threshold
            masks_p = p["masks"][keep].detach().cpu().bool()
            labels_p = p["labels"][keep].detach().cpu()
            masks_t = t["masks"].detach().cpu().bool()
            labels_t = t["labels"].detach().cpu()
            # GT counts toward the recall denominator even when there are no
            # (kept) predictions for the image.
            self._n_gt += int(masks_t.shape[0])
            if masks_p.shape[0] == 0 or masks_t.shape[0] == 0:
                continue
            ious = _pairwise_mask_iou_3d(masks_p, masks_t)
            if self.match_labels:
                ious = ious * (labels_p[:, None] == labels_t[None, :]).to(ious.dtype)
            for _, _, iou in _hungarian_match(ious, self.iou_threshold):
                self._matched_ious.append(iou)

    def aggregate(self):
        if self.mode == "semantic":
            ious: List[float] = []
            for c, inter in self._inter.items():
                union = self._union.get(c, 0)
                if union > 0:
                    ious.append(inter / union)
            if not ious:
                # No class ever had a nonzero union (e.g. all-background data
                # under exclude_from_mean, an all-ignore shard, or an empty
                # shard). There is NO evidence to score -> NaN, never 0.0 (a
                # perfect all-background prediction is not zero-quality).
                logger.warning(
                    "MaskMIoUMetric(semantic): no class accumulated a nonzero "
                    "union (all-background / all-ignored / empty data); "
                    "reporting NaN instead of a misleading 0.0."
                )
                return float("nan")
            return float(sum(ious) / len(ious))
        n_m = len(self._matched_ious)
        return {
            "mask_miou": float(sum(self._matched_ious) / n_m) if n_m else float("nan"),
            # n_gt == 0: recall of nothing is undefined evidence, not 0.0
            # (NaN-as-skip convention; checkpoint selection NaN-skips).
            "mask_match_recall": (
                n_m / self._n_gt if self._n_gt > 0 else float("nan")
            ),
        }

    def reset(self):
        self._inter.clear()
        self._union.clear()
        self._matched_ious.clear()
        self._n_gt = 0
        self._gathered = False


# ---------------------------------------------------------------------------
# Single metric registry + builder
# ---------------------------------------------------------------------------

METRICS: Dict[str, Callable[..., Metric]] = {
    "train_loss": TrainLosses,
    "nrmse": NRMSEMetric,
    "mae": MAEMetric,
    "box_map": BoxMAPMetric,
    "box_miou": BoxMIoUMetric,
    "box_f1": BoxF1Metric,
    "class_ap": ClassAPMetric,
    "mask_map": MaskMAPMetric,
    "mask_miou": MaskMIoUMetric,
    "predicted_iou": PredictedIoUEvalMetric,
}

def _as_plain(obj):
    """Convert an OmegaConf node into a plain python container (else pass through)."""
    from omegaconf import OmegaConf
    if OmegaConf.is_config(obj):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def _build_one_metric(item):
    """``(key, Metric)`` for one spec entry.

    Each entry is either ``"name"`` (the named metric with its default ctor args) or
    ``{"name": str, "key"?: str, **ctor_kwargs}`` (the named metric built with those
    kwargs, keyed by ``key``, which defaults to ``name``). Every metric must be
    registered in :data:`METRICS`; anything a config wants to configure it declares
    explicitly in the kwargs.
    """
    item = _as_plain(item)
    if isinstance(item, str):
        return item, METRICS[item]()
    d = dict(item)
    name = d.pop("name")
    key = d.pop("key", name)
    return str(key), METRICS[name](**d)


def build_metrics(spec) -> Dict[str, Metric]:
    """Ordered ``{key: Metric}`` from a spec: a single entry or a list of entries.

    See :func:`_build_one_metric` for the accepted entry forms.
    """
    spec = _as_plain(spec)
    if isinstance(spec, (str, dict)):
        spec = [spec]
    return dict(_build_one_metric(item) for item in spec)