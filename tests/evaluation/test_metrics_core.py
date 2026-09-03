import itertools
import math
from typing import Dict

import numpy as np
import pytest
import torch

from cell_observatory_platform.evaluation.metrics import (
    BoxF1Metric,
    BoxMAPMetric,
    BoxMIoUMetric,
    ClassAPMetric,
    MAEMetric,
    MaskMAPMetric,
    MaskMIoUMetric,
    NRMSEMetric,
    ReduceBuffer,
    TrainLosses,
    _ap_from_class_detections,
    _greedy_tp_fp,
    _hungarian_match,
    _pad_detection_rows,
)


def _ap_101_point(recall: np.ndarray, precision: np.ndarray) -> float:
    precision = precision.copy()
    for i in range(len(precision) - 1, 0, -1):
        if precision[i] > precision[i - 1]:
            precision[i - 1] = precision[i]
    thresholds = np.linspace(0.0, 1.0, 101)
    inds = np.searchsorted(recall, thresholds, side="left")
    interp = np.zeros_like(thresholds)
    valid = inds < len(precision)
    interp[valid] = precision[inds[valid]]
    return float(interp.mean())


def _binary_ap_expected(scores: torch.Tensor, targets_bool: torch.Tensor) -> float:
    order = torch.argsort(scores, descending=True)
    sorted_targets = targets_bool[order].float()
    tp = torch.cumsum(sorted_targets, dim=0)
    fp = torch.cumsum(1.0 - sorted_targets, dim=0)
    recall = (tp / sorted_targets.sum().clamp_min(1.0)).numpy()
    precision = (tp / (tp + fp).clamp_min(1e-12)).numpy()
    return _ap_101_point(recall, precision)


def _det(boxes, labels, scores=None):
    out = {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 6),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if scores is not None:
        out["scores"] = torch.tensor(scores, dtype=torch.float32)
    return out


def _mask_det(masks, labels, scores=None):
    out = {
        "masks": masks.bool(),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    if scores is not None:
        out["scores"] = torch.tensor(scores, dtype=torch.float32)
    return out


@pytest.mark.parametrize(
    "method,expected",
    [("mean", 2.0), ("min", 1.0), ("max", 3.0)],
)
def test_reduce_buffer_reductions_and_reset(method, expected):
    buf = ReduceBuffer(reduce_method=method)
    for value in [1.0, torch.tensor(3.0), 2.0]:
        buf.add(value)

    assert buf.aggregate() == pytest.approx(expected)

    buf.reset()
    assert buf.values == []


def test_reduce_buffer_errors():
    # Empty buffer -> NaN (skip), never an assert: one rank with an empty
    # validation shard must not crash the run. (Was: AssertionError.)
    assert math.isnan(ReduceBuffer().aggregate())

    buf = ReduceBuffer(reduce_method="bad")
    buf.add(1.0)
    with pytest.raises(ValueError, match="Unknown reduce method"):
        buf.aggregate()


@pytest.mark.parametrize(
    "method,expected",
    [("mean", 2.0), ("min", 1.0), ("max", 3.0)],
)
def test_train_losses_reductions_and_reset(method, expected):
    metric = TrainLosses(reduce_method=method)
    metric(None, None, torch.tensor(1.0))
    metric(None, None, torch.tensor(3.0))

    assert metric.aggregate() == pytest.approx(expected)

    metric.reset()
    assert metric.loss_values == []


def test_mae_metric_pooled_mean():
    metric = MAEMetric(reduce_method="mean")
    metric(torch.tensor([1.0, 2.0, 3.0]), torch.zeros(3))   # |err| sum 6 over 3
    metric(torch.tensor([10.0]), torch.zeros(1))            # |err| 10 over 1

    # Pooled: 16 / 4. A mean of per-call means would give (2 + 10) / 2 = 6.
    assert metric.aggregate() == pytest.approx(4.0)


def test_nrmse_metric_expected_value_and_constant_target_nan():
    metric = NRMSEMetric(reduce_method="mean")
    metric(torch.tensor([0.0, 2.0]), torch.tensor([0.0, 4.0]))

    assert metric.aggregate() == pytest.approx(math.sqrt(2.0) / 4.0)

    # Constant target -> zero range -> NRMSE is UNDEFINED. NaN (skip), never
    # the old eps-clamped value (which exploded to ~1/eps for any nonzero
    # error and reported a fake 0.0 here). (Was: pytest.approx(0.0).)
    constant_target = NRMSEMetric(reduce_method="mean", eps=1e-4)
    constant_target(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]))
    assert math.isnan(constant_target.aggregate())


def test_box_map_perfect_match_and_no_predictions():
    target = _det([[0, 0, 0, 1, 1, 1]], labels=[1])
    pred = _det([[0, 0, 0, 1, 1, 1]], labels=[1], scores=[0.9])
    metric = BoxMAPMetric(iou_thresholds=[0.5])

    metric([pred], [target])

    assert metric.aggregate() == pytest.approx(1.0)

    no_pred = BoxMAPMetric(iou_thresholds=[0.5])
    no_pred([_det([], labels=[], scores=[])], [target])
    assert no_pred.aggregate() == pytest.approx(0.0)

    absent_class = BoxMAPMetric(iou_thresholds=[0.5], class_ids=[99])
    absent_class([pred], [target])
    assert absent_class.aggregate() == pytest.approx(0.0)


def test_box_miou_matching_and_label_gate():
    target = _det([[0, 0, 0, 1, 1, 1]], labels=[1])
    pred = _det(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ],
        labels=[1, 1],
        scores=[0.9, 0.8],
    )
    metric = BoxMIoUMetric(iou_threshold=0.5)

    metric([pred], [target])

    # aggregate() now returns the flat mapping {box_miou, box_match_recall}.
    result = metric.aggregate()
    assert result["box_miou"] == pytest.approx(1.0)
    assert result["box_match_recall"] == pytest.approx(1.0)

    # Label mismatch -> nothing matched -> NaN (no evidence), not the old 0.0
    # sentinel; recall exposes the missed GT.
    mismatch = BoxMIoUMetric(iou_threshold=0.5, match_labels=True)
    mismatch([_det([[0, 0, 0, 1, 1, 1]], labels=[2], scores=[0.9])], [target])
    mismatch_result = mismatch.aggregate()
    assert math.isnan(mismatch_result["box_miou"])
    assert mismatch_result["box_match_recall"] == pytest.approx(0.0)


def test_box_f1_counts_tp_fp_fn_and_reset():
    pred = _det(
        [
            [0, 0, 0, 1, 1, 1],
            [4, 4, 4, 5, 5, 5],
        ],
        labels=[1, 1],
        scores=[0.9, 0.8],
    )
    target = _det(
        [
            [0, 0, 0, 1, 1, 1],
            [2, 2, 2, 3, 3, 3],
        ],
        labels=[1, 1],
    )
    metric = BoxF1Metric(iou_threshold=0.5, score_threshold=0.0, match_labels=True)

    metric([pred], [target])

    assert metric.aggregate() == pytest.approx(0.5)

    metric.reset()
    assert metric._tp == 0
    assert metric._fp == 0
    assert metric._fn == 0


def test_class_ap_binary_and_reset():
    # Ranked targets [1, 0, 1, 1]: recall [1/3, 1/3, 2/3, 1], precision envelope [1, .75, .75, .75].
    # 101-pt grid: 34 thresholds <= 1/3 read 1.0, the other 67 read 0.75.
    scores = torch.tensor([0.9, 0.8, 0.7, 0.1])
    targets = torch.tensor([1, 0, 1, 1])
    metric = ClassAPMetric()

    metric(scores, targets)

    assert metric.aggregate() == pytest.approx((34 * 1.0 + 67 * 0.75) / 101)
    assert metric.aggregate() == pytest.approx(_binary_ap_expected(scores, targets.bool()))

    metric.reset()
    assert metric._scores == [] and metric._targets == []
    assert math.isnan(metric.aggregate())     # empty accumulator = no evidence, not 0.0


def test_class_ap_multiclass_skips_absent_classes():
    scores = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.1, 0.9, 0.0],
        ]
    )
    targets = torch.tensor([0, 0, 1])
    metric = ClassAPMetric(num_classes=3)

    metric(scores, targets)

    expected = (
        _binary_ap_expected(scores[:, 0], targets == 0)
        + _binary_ap_expected(scores[:, 1], targets == 1)
    ) / 2
    assert metric.aggregate() == pytest.approx(expected)


def test_mask_map_batched_perfect_match():
    mask = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    metric = MaskMAPMetric(iou_thresholds=[0.5])

    metric(
        [_mask_det(mask, labels=[1], scores=[0.9])],
        [_mask_det(mask, labels=[1])],
    )

    assert metric.aggregate() == pytest.approx(1.0)


def test_mask_miou_semantic_and_ignore_index():
    pred = torch.tensor(
        [
            [[0, 1], [1, 1]],
            [[0, 0], [1, 0]],
        ]
    )
    target = torch.tensor(
        [
            [[0, 1], [0, 1]],
            [[0, 0], [1, 1]],
        ]
    )
    metric = MaskMIoUMetric(mode="semantic")

    metric(pred, target)

    # Per-class IoU: for each c, intersection = (pred==c) & (gt==c), union = (pred==c) | (gt==c).
    # Class 0 and class 1 both have IoU 3/5 here; mean = 0.6.
    assert metric.aggregate() == pytest.approx(0.6)

    # exclude_from_mean=0 drops background from the MEAN only (the behavior
    # ignore_index used to provide before it became true voxel masking).
    foreground = MaskMIoUMetric(mode="semantic", exclude_from_mean=0)
    foreground(pred, target)
    assert foreground.aggregate() == pytest.approx(3 / 5)

    # ignore_index=0 masks the GT-background VOXELS out of every class:
    # valid = (gt != 0) leaves 4 voxels. Class 1: inter 3, union 4 -> 0.75.
    # ignore_index is NOT itself a class: with num_classes=None
    # the derived class set excludes it, so a prediction emitting label 0 does
    # not create a spurious IoU-0 "class 0" entry. Mean = 0.75.
    masked = MaskMIoUMetric(mode="semantic", ignore_index=0)
    masked(pred, target)
    assert masked.aggregate() == pytest.approx(3 / 4)


def test_mask_miou_instance_batched_and_label_gate():
    mask = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    metric = MaskMIoUMetric(mode="instance", iou_threshold=0.5)

    metric(
        [_mask_det(mask, labels=[1], scores=[0.9])],
        [_mask_det(mask, labels=[1])],
    )

    # Instance mode aggregates to the flat {mask_miou, mask_match_recall} map.
    result = metric.aggregate()
    assert result["mask_miou"] == pytest.approx(1.0)
    assert result["mask_match_recall"] == pytest.approx(1.0)

    # Label mismatch -> nothing matched -> NaN (no evidence), not the old 0.0
    # sentinel; recall exposes the missed GT.
    mismatch = MaskMIoUMetric(mode="instance", iou_threshold=0.5, match_labels=True)
    mismatch(
        [_mask_det(mask, labels=[2], scores=[0.9])],
        [_mask_det(mask, labels=[1])],
    )
    mismatch_result = mismatch.aggregate()
    assert math.isnan(mismatch_result["mask_miou"])
    assert mismatch_result["mask_match_recall"] == pytest.approx(0.0)


def test_mask_miou_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        MaskMIoUMetric(mode="bad")


# ---------------------------------------------------------------------------
# Pooled sufficient statistics: MAE / NRMSE / TrainLosses across shards
# ---------------------------------------------------------------------------


def _merge_mae(a: MAEMetric, b: MAEMetric) -> MAEMetric:
    """Mimic gather()'s all_reduce(SUM) over two simulated ranks."""
    merged = MAEMetric()
    merged.sum_abs = a.sum_abs + b.sum_abs
    merged.n = a.n + b.n
    return merged


def _merge_nrmse(a: NRMSEMetric, b: NRMSEMetric) -> NRMSEMetric:
    """Mimic gather()'s all_reduce(SUM/SUM/MIN/MAX) over two simulated ranks."""
    merged = NRMSEMetric()
    merged.sum_sq = a.sum_sq + b.sum_sq
    merged.n = a.n + b.n
    merged.y_min = torch.minimum(a.y_min, b.y_min)
    merged.y_max = torch.maximum(a.y_max, b.y_max)
    return merged


def test_mae_pooled_across_shards_weights_elements_not_shards():
    """Shard sizes (9, 1): the pooled MAE weights ELEMENTS, not shards, and
    equals a single metric fed the union."""
    # Shard A: 9 elements, |err| = 1 each. Shard B: 1 element, |err| = 11.
    pred_a, tgt_a = torch.ones(9) * 2.0, torch.ones(9)
    pred_b, tgt_b = torch.tensor([11.0]), torch.tensor([0.0])

    rank_a, rank_b = MAEMetric(), MAEMetric()
    rank_a(pred_a, tgt_a)
    rank_b(pred_b, tgt_b)

    pooled = _merge_mae(rank_a, rank_b).aggregate()

    # Direct single-metric run over the union == the merged two-shard value.
    direct = MAEMetric()
    direct(torch.cat([pred_a, pred_b]), torch.cat([tgt_a, tgt_b]))
    assert pooled == pytest.approx(direct.aggregate(), abs=1e-12)

    # Exact expected value: (9 * 1 + 11) / 10 (a mean of shard means would be 6).
    assert pooled == pytest.approx((9 * 1.0 + 11.0) / 10)


def test_nrmse_pooled_across_shards_uses_global_target_range():
    """NRMSE pools sum-of-squares/count AND normalizes by the GLOBAL target
    range -- not a per-batch range averaged unweighted."""
    # Shard A: 9 elements, err 1 each, target range [0, 1].
    tgt_a = torch.cat([torch.zeros(8), torch.ones(1)])
    pred_a = tgt_a + 1.0
    # Shard B: 1 element, err 3, target value 10 (extends global range to 10).
    pred_b, tgt_b = torch.tensor([13.0]), torch.tensor([10.0])

    rank_a, rank_b = NRMSEMetric(), NRMSEMetric()
    rank_a(pred_a, tgt_a)
    rank_b(pred_b, tgt_b)
    pooled = _merge_nrmse(rank_a, rank_b).aggregate()

    direct = NRMSEMetric()
    direct(torch.cat([pred_a, pred_b]), torch.cat([tgt_a, tgt_b]))
    assert pooled == pytest.approx(direct.aggregate(), abs=1e-12)

    # Hand computation: sum_sq = 9*1 + 9 = 18, n = 10, range = 10 - 0.
    assert pooled == pytest.approx(math.sqrt(18 / 10) / 10.0)


def test_nrmse_constant_batch_does_not_dominate_pooled_range():
    """One constant-target (blank) batch amid normal batches: its error is
    absorbed into the pooled sum_sq while the normalizer stays the finite
    global range (a per-batch range would clamp to eps and explode)."""
    metric = NRMSEMetric()
    metric(torch.tensor([0.0, 2.0]), torch.tensor([0.0, 4.0]))   # range [0, 4]
    metric(torch.tensor([1.5, 1.5]), torch.tensor([1.0, 1.0]))   # blank tile

    value = metric.aggregate()
    assert math.isfinite(value)
    # sum_sq = (0 + 4) + (0.25 + 0.25) = 4.5, n = 4, range 4.
    assert value == pytest.approx(math.sqrt(4.5 / 4) / 4.0)


def test_nrmse_and_mae_empty_aggregate_nan():
    assert math.isnan(NRMSEMetric().aggregate())
    assert math.isnan(MAEMetric().aggregate())


def test_train_losses_merge_pools_steps_and_empty_is_nan():
    """TrainLosses merges per-rank per-STEP values; mean weights steps, not
    ranks. Empty accumulator -> NaN."""
    merged = TrainLosses._merge_values([[1.0] * 9, [11.0]])
    metric = TrainLosses(reduce_method="mean")
    metric.loss_values = merged
    assert metric.aggregate() == pytest.approx(2.0)      # (9 + 11) / 10

    assert math.isnan(TrainLosses().aggregate())


def test_nrmse_accumulates_sum_sq_in_float64():
    """sum_sq is accumulated in float64 (bit-equal to a double reference)."""
    m = NRMSEMetric()
    g = torch.Generator().manual_seed(0)
    out = torch.rand(100_000, generator=g) * 1e3
    tgt = torch.rand(100_000, generator=g) * 1e3
    m(out, tgt)
    d = (out - tgt).to(torch.float32).double()
    assert m.sum_sq.item() == (d * d).sum().item()


def test_mae_accumulates_sum_abs_in_float64():
    """sum_abs is accumulated in float64 (bit-equal to a double reference)."""
    m = MAEMetric()
    g = torch.Generator().manual_seed(1)
    out = torch.rand(50_000, generator=g)
    tgt = torch.rand(50_000, generator=g)
    m(out, tgt)
    d = (out - tgt).to(torch.float32).double()
    assert m.sum_abs.item() == d.abs().sum().item()


# ---------------------------------------------------------------------------
# Hungarian matcher
# ---------------------------------------------------------------------------


def _brute_force_max_cardinality(ious: torch.Tensor, thr: float) -> int:
    """Exhaustive max-cardinality matching over eligible pairs (n, m <= 5)."""
    elig = (ious >= thr) & (ious > 0.0)
    n, m = ious.shape
    for k in range(min(n, m), 0, -1):
        for rows in itertools.combinations(range(n), k):
            for cols in itertools.permutations(range(m), k):
                if all(elig[r, c] for r, c in zip(rows, cols)):
                    return k
    return 0


class TestHungarianMatch:
    """Max-cardinality first, then max IoU-sum, over pairs with iou >= thr and iou > 0."""

    def test_matches_brute_force_cardinality_randomized(self):
        g = torch.Generator().manual_seed(0)
        for trial in range(500):
            n = int(torch.randint(1, 6, (1,), generator=g))
            m = int(torch.randint(1, 6, (1,), generator=g))
            ious = torch.rand(n, m, generator=g)
            thr = [0.0, 0.25, 0.5, 0.75][trial % 4]
            got = _hungarian_match(ious, thr)
            assert len(got) == _brute_force_max_cardinality(ious, thr), f"trial {trial}: thr={thr}\n{ious}"
            assert all(v >= thr and v > 0.0 for _, _, v in got)

    def test_five_chain_prefers_five_weak_matches_over_four_perfect(self):
        M = torch.zeros(5, 5)
        for i in range(5):
            M[i, i] = 0.5
            if i + 1 < 5:
                M[i, i + 1] = 1.0          # IoU-sum alone would take the 4 perfect pairs
        assert len(_hungarian_match(M, 0.5)) == 5

    def test_three_chain_at_thr_zero(self):
        M = torch.tensor([[0.01, 1.0, 0.0], [0.0, 0.01, 1.0], [0.0, 0.0, 0.01]])
        assert len(_hungarian_match(M, 0.0)) == 3

    @pytest.mark.parametrize("ious,thr", [([[1.0, 0.4], [0.35, 0.0]], 0.3), ([[1.0, 0.5], [0.5, 0.0]], 0.5)])
    def test_two_weak_pairs_beat_one_perfect_pair(self, ious, thr):
        assert len(_hungarian_match(torch.tensor(ious), thr)) == 2

    def test_iou_sum_maximal_among_max_cardinality(self):
        got = _hungarian_match(torch.tensor([[0.9, 0.6, 0.0], [0.6, 0.9, 0.0]]), 0.5)
        assert len(got) == 2 and sum(v for _, _, v in got) == pytest.approx(1.8)

    def test_disjoint_pair_never_matches_at_thr_zero(self):
        assert _hungarian_match(torch.zeros(2, 2), 0.0) == []

    def test_only_pairs_clearing_threshold_are_returned(self):
        assert _hungarian_match(torch.tensor([[0.6, 0.0], [0.0, 0.0]]), 0.5) == [(0, 0, pytest.approx(0.6))]

    def test_recovers_both_matches_on_crossing_fixture(self):
        # pred0's best GT (gt1, 0.60) is pred1's only eligible GT; optimum is pred0->gt0, pred1->gt1.
        ious = torch.tensor([[0.55, 0.60], [0.10, 0.65]])
        assert sorted(v for _, _, v in _hungarian_match(ious, 0.5)) == pytest.approx([0.55, 0.65])


# ---------------------------------------------------------------------------
# Greedy (COCO) TP/FP consume and 101-point AP
# ---------------------------------------------------------------------------


def _reference_greedy_tp_fp(iou_rows, img_ids, gt_sizes, iou_thr):
    """per-detection greedy reference: in rank order each detection claims its
    best still-unmatched GT of its own image."""
    n_det = len(iou_rows)
    gt_matched = {img: torch.zeros(n, dtype=torch.bool) for img, n in gt_sizes.items()}
    tp = torch.zeros(n_det, dtype=torch.float32)
    fp = torch.zeros(n_det, dtype=torch.float32)
    for i, (row, img_id) in enumerate(zip(iou_rows, img_ids)):
        if row.numel() == 0 or gt_matched[img_id].numel() == 0:
            fp[i] = 1.0
            continue
        row = row.clone()
        row[gt_matched[img_id]] = -1.0
        best_iou, best_idx = torch.max(row, dim=0)
        if best_iou.item() >= iou_thr:
            tp[i] = 1.0
            gt_matched[img_id][int(best_idx.item())] = True
        else:
            fp[i] = 1.0
    return tp, fp


def _random_detection_set(seed: int):
    """Ragged per-image blocks: some empty-GT images, varied n_gt, shared imgs."""
    g = torch.Generator().manual_seed(seed)
    gt_sizes: Dict[int, int] = {}
    blocks, block_imgs = [], []
    for img in range(int(torch.randint(2, 6, (1,), generator=g))):
        n_gt = int(torch.randint(0, 5, (1,), generator=g))
        gt_sizes[img] = n_gt
        k = int(torch.randint(0, 6, (1,), generator=g))
        blocks.append(torch.rand(k, n_gt, generator=g))
        block_imgs.append(img)
    return blocks, block_imgs, gt_sizes


class TestGreedyTpFp:
    @pytest.mark.parametrize("seed", range(30))
    @pytest.mark.parametrize("thr", [0.0, 0.3, 0.5, 0.75])
    def test_greedy_tp_fp_matches_reference_loop(self, seed, thr):
        """The padded-matrix consume equals the per-detection reference loop
        exactly (same first-index tie-break, padding never wins)."""
        blocks, block_imgs, gt_sizes = _random_detection_set(seed)
        iou_pad, img_t = _pad_detection_rows(blocks, block_imgs)
        M = iou_pad.shape[0]
        # random rank order (stable argsort of a random vector)
        g = torch.Generator().manual_seed(seed + 1000)
        order = torch.argsort(torch.rand(M, generator=g), descending=True, stable=True)
        img_ids = img_t[order].tolist()
        tp_new, fp_new = _greedy_tp_fp(iou_pad[order], img_ids, gt_sizes, thr)

        # the reference consumes ragged per-detection rows in the same order
        ragged = [b[i] for b, img in zip(blocks, block_imgs) for i in range(b.shape[0])]
        ragged_imgs = [img for b, img in zip(blocks, block_imgs) for _ in range(b.shape[0])]
        ragged_ordered = [ragged[i] for i in order.tolist()]
        imgs_ordered = [ragged_imgs[i] for i in order.tolist()]
        tp_old, fp_old = _reference_greedy_tp_fp(ragged_ordered, imgs_ordered, gt_sizes, thr)

        assert tp_new.shape == (M,) and fp_new.shape == (M,)
        assert torch.equal(tp_new, tp_old)
        assert torch.equal(fp_new, fp_old)

    def test_ap_from_class_detections_matches_reference(self):
        """AP from precomputed class detections equals greedy-reference TP/FP
        cumsums pushed through the 101-point interpolation."""
        for seed in range(10):
            blocks, block_imgs, gt_sizes = _random_detection_set(seed)
            n_gt = sum(gt_sizes.values())
            if n_gt == 0:
                continue
            iou_pad, img_t = _pad_detection_rows(blocks, block_imgs)
            g = torch.Generator().manual_seed(seed)
            scores = torch.rand(iou_pad.shape[0], generator=g)
            order = torch.argsort(scores, descending=True, stable=True)
            data = {
                "n_gt": n_gt,
                "iou_pad": iou_pad[order],
                "img_ids": img_t[order].tolist(),
                "gt_sizes": gt_sizes,
            }
            for thr in (0.3, 0.5):
                ap_new = _ap_from_class_detections(data, thr)
                ragged = [b[i] for b, img in zip(blocks, block_imgs) for i in range(b.shape[0])]
                imgs = [img for b, img in zip(blocks, block_imgs) for _ in range(b.shape[0])]
                tp, fp = _reference_greedy_tp_fp(
                    [ragged[i] for i in order.tolist()],
                    [imgs[i] for i in order.tolist()],
                    gt_sizes, thr,
                )
                tp_cum = torch.cumsum(tp, 0)
                fp_cum = torch.cumsum(fp, 0)
                ap_old = _ap_101_point(
                    (tp_cum / max(n_gt, 1)).numpy(),
                    (tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-12)).numpy(),
                )
                assert ap_new == ap_old

    def test_stream_class_ap_matches_reference_loop(self):
        """Pooled per-class AP over streaming fragments (duplicate image ids,
        ragged n_gt widths) equals the flatten-then-greedy reference."""
        g = torch.Generator().manual_seed(7)
        entries = []
        for img, n_gt, k in ((0, 3, 4), (1, 0, 2), (2, 2, 0), (0, 3, 2)):
            entries.append({
                "image_id": img,
                "n_gt": n_gt,
                "scores": torch.rand(k, generator=g),
                "pred_ious": torch.rand(k, generator=g),
                "ious": torch.rand(k, n_gt, generator=g),
            })
        n_gt_total = 3 + 0 + 2
        for thr in (0.25, 0.5):
            ap_new = MaskMAPMetric._stream_class_ap(entries, iou_thr=thr, n_gt_total=n_gt_total)
            # reference: flatten every fragment row, rank globally by score
            flat_rows, flat_imgs, flat_scores = [], [], []
            gt_sizes = {0: 3, 1: 0, 2: 2}
            for e in entries:
                for i in range(int(e["scores"].numel())):
                    flat_rows.append(e["ious"][i])
                    flat_imgs.append(e["image_id"])
                flat_scores.append(e["scores"])
            order = torch.argsort(torch.cat(flat_scores), descending=True, stable=True)
            tp, fp = _reference_greedy_tp_fp(
                [flat_rows[i] for i in order.tolist()],
                [flat_imgs[i] for i in order.tolist()],
                gt_sizes, thr,
            )
            tp_cum, fp_cum = torch.cumsum(tp, 0), torch.cumsum(fp, 0)
            ap_old = _ap_101_point(
                (tp_cum / max(n_gt_total, 1)).numpy(),
                (tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-12)).numpy(),
            )
            assert ap_new == ap_old


# ---------------------------------------------------------------------------
# Instance / semantic mIoU, BoxF1, ClassAP edge semantics
# ---------------------------------------------------------------------------


def test_mask_miou_instance_dense_path_matches_optimally():
    """Mask-level fixture where greedy and Hungarian genuinely differ:
    pred0 (score .9) ties at IoU 0.5 vs both GTs and greedy would hand it gt0,
    stranding pred1 (a PERFECT match for gt0). Hungarian assigns pred1->gt0
    (1.0) and pred0->gt1 (0.5)."""
    # 2x2x2 grid; voxels A=(0,0,0), B=(0,0,1), C=(0,1,0), D=(0,1,1).
    def mask(coords):
        m = torch.zeros(2, 2, 2, dtype=torch.bool)
        for c in coords:
            m[c] = True
        return m

    gt0 = mask([(0, 0, 0), (0, 0, 1)])                       # {A, B}
    gt1 = mask([(0, 1, 0), (0, 1, 1)])                       # {C, D}
    pred0 = mask([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)])  # {A,B,C,D}
    pred1 = mask([(0, 0, 0), (0, 0, 1)])                     # {A, B}

    metric = MaskMIoUMetric(mode="instance", iou_threshold=0.5)
    metric(
        [{"masks": torch.stack([pred0, pred1]),
          "labels": torch.tensor([1, 1]),
          "scores": torch.tensor([0.9, 0.8])}],
        [{"masks": torch.stack([gt0, gt1]), "labels": torch.tensor([1, 1])}],
    )

    result = metric.aggregate()
    # Hungarian: pred1->gt0 IoU 1.0, pred0->gt1 IoU 0.5 -> mean 0.75, recall 1.
    assert result["mask_miou"] == pytest.approx(0.75)
    assert result["mask_match_recall"] == pytest.approx(1.0)


def test_box_f1_no_gt_no_preds_is_perfect_but_missed_gt_is_zero():
    """tp == fp == fn == 0 (no GT, no predictions anywhere) -> 1.0: predicting
    nothing where nothing exists is perfect; a missed GT is still 0.0."""
    metric = BoxF1Metric(iou_threshold=0.5, score_threshold=0.0)
    metric(
        [{"boxes": torch.zeros(0, 6), "labels": torch.zeros(0, dtype=torch.long),
          "scores": torch.zeros(0)}],
        [{"boxes": torch.zeros(0, 6), "labels": torch.zeros(0, dtype=torch.long)}],
    )
    assert metric.aggregate() == pytest.approx(1.0)

    # But a miss (GT without predictions) is still 0.0.
    missed = BoxF1Metric(iou_threshold=0.5, score_threshold=0.0)
    missed(
        [{"boxes": torch.zeros(0, 6), "labels": torch.zeros(0, dtype=torch.long),
          "scores": torch.zeros(0)}],
        [{"boxes": torch.tensor([[0., 0., 0., 1., 1., 1.]]),
          "labels": torch.tensor([1])}],
    )
    assert missed.aggregate() == pytest.approx(0.0)


def test_mask_miou_semantic_num_classes_none_scores_hallucinated_classes():
    """num_classes=None iterates the UNION of GT and pred classes: a
    hallucinated class (predicted, absent from GT) scores IoU 0 instead of
    being silently dropped from the mean."""
    gt = torch.zeros(1, 2, 2, dtype=torch.long)
    gt[0, 0, 0] = 1
    pred = gt.clone()
    pred[0, 1, 1] = 2  # hallucinated class 2 on a background voxel

    metric = MaskMIoUMetric(mode="semantic", num_classes=None)
    metric(pred, gt)
    # class 0: inter 2, union 3 (pred stole one bg voxel) -> 2/3.
    # class 1: perfect -> 1.0. class 2: inter 0, union 1 -> 0.0.
    assert metric.aggregate() == pytest.approx((2 / 3 + 1.0 + 0.0) / 3)


def test_mask_miou_semantic_all_background_is_nan_with_warning(caplog):
    """All-background GT + all-background prediction with background excluded
    from the mean: there is NO evidence to score. NaN + warning, never 0.0
    (which would punish a perfect prediction)."""
    gt = torch.zeros(1, 2, 2, dtype=torch.long)
    metric = MaskMIoUMetric(mode="semantic", num_classes=2, exclude_from_mean=0)
    metric(gt.clone(), gt)
    with caplog.at_level("WARNING"):
        value = metric.aggregate()
    assert math.isnan(value)
    assert any("NaN" in rec.message for rec in caplog.records)


def test_mask_miou_ignore_index_masks_voxels_out_of_every_class():
    """Voxels GT-labeled ignore_index leave EVERY class's intersection and
    union: garbage predictions inside the ignored region cost nothing, and the
    region's own label never scores."""
    gt = torch.zeros(1, 2, 4, dtype=torch.long)
    gt[0, 0, :2] = 1          # class 1 region
    gt[0, 1, :2] = 255        # ignored region
    # Prediction: perfect on labeled voxels, garbage inside the ignored region.
    pred = torch.zeros(1, 2, 4, dtype=torch.long)
    pred[0, 0, :2] = 1
    pred[0, 1, :2] = 1        # garbage: class 1 sprayed into the ignore region

    masked = MaskMIoUMetric(mode="semantic", num_classes=2, ignore_index=255)
    masked(pred, gt)
    # valid = gt != 255. class 0: 4 bg voxels perfect. class 1: 2 voxels
    # perfect. The garbage falls entirely on masked voxels -> mIoU 1.0.
    assert masked.aggregate() == pytest.approx(1.0)
    # The ignored label never accumulates a bucket of its own.
    assert 255 not in masked._inter and 255 not in masked._union

    # Without masking, the same prediction is penalized (class 1 union grows).
    unmasked = MaskMIoUMetric(mode="semantic", num_classes=2)
    unmasked(pred, gt)
    assert unmasked.aggregate() < 1.0


def test_mask_miou_exclude_from_mean_scores_background_but_drops_it_from_mean():
    """exclude_from_mean=0: background is fully scored per-voxel (it still
    absorbs misclassified voxels) but dropped from the class mean."""
    gt = torch.zeros(1, 2, 4, dtype=torch.long)
    gt[0, 0, :2] = 1
    pred = gt.clone()
    pred[0, 1, 0] = 1  # one background voxel misclassified as class 1

    metric = MaskMIoUMetric(mode="semantic", num_classes=2, exclude_from_mean=0)
    metric(pred, gt)
    # Only class 1 in the mean: inter 2, union 3 -> 2/3. (Background's own
    # 5/6 does not enter.)
    assert metric.aggregate() == pytest.approx(2 / 3)


def test_mask_miou_semantic_derived_class_set_excludes_ignore_index():
    """With num_classes=None the derived class set never contains ignore_index,
    even when the prediction emits that label."""
    m = MaskMIoUMetric(mode="semantic", num_classes=None, ignore_index=255)
    gt = torch.tensor([[0, 1], [255, 1]])
    pred = torch.tensor([[0, 1], [255, 255]])   # pred emits the ignore label
    m(pred, gt)
    # scored classes: {0, 1} only -- 255 must not appear as an IoU-0 class
    assert set(m._inter.keys()) <= {0, 1}
    # class 0: gt {0,0-pos}, valid excludes the 255-gt voxel; pred 0 at [0,0]
    assert m._inter[0] == 1 and m._union[0] == 1
    assert m._inter[1] == 1 and m._union[1] == 2


def test_class_ap_no_positive_class_returns_nan():
    """Targets matching none of the classes leave no positives anywhere -> NaN."""
    m = ClassAPMetric(num_classes=3)
    m(torch.rand(4, 3), torch.full((4,), 99))    # targets match no class
    assert math.isnan(m.aggregate())


class TestFloatMaskRejection:
    """Soft (float) masks must be rejected at both batched mask entry points."""

    def _pred(self, dtype):
        return [{
            "masks": torch.rand(2, 2, 4, 4).to(dtype),
            "labels": torch.zeros(2, dtype=torch.long),
            "scores": torch.ones(2),
        }]

    def _tgt(self, dtype):
        return [{
            "masks": (torch.rand(2, 2, 4, 4) > 0.5).to(dtype),
            "labels": torch.zeros(2, dtype=torch.long),
        }]

    def test_mask_map_batched_raises_on_float(self):
        with pytest.raises(ValueError, match="binarize"):
            MaskMAPMetric()(self._pred(torch.float32), self._tgt(torch.bool))

    def test_mask_miou_instance_raises_on_float(self):
        with pytest.raises(ValueError, match="binarize"):
            MaskMIoUMetric(mode="instance")(
                self._pred(torch.float32), self._tgt(torch.bool)
            )

    def test_bool_masks_pass(self):
        map_metric = MaskMAPMetric()
        map_metric(self._pred(torch.bool), self._tgt(torch.bool))
        assert len(map_metric._preds) == 1 and len(map_metric._targets) == 1
        miou_metric = MaskMIoUMetric(mode="instance")
        miou_metric(self._pred(torch.bool), self._tgt(torch.bool))
        assert math.isfinite(miou_metric.aggregate()["mask_match_recall"])


class TestNaNForNoEvidence:
    """Recalls / APs with nothing to score report NaN, never a fake 0.0."""

    def test_box_match_recall_nan_when_no_gt(self):
        metric = BoxMIoUMetric()
        # one image, zero GT boxes, zero preds
        metric(
            [{"boxes": torch.zeros(0, 6), "labels": torch.zeros(0, dtype=torch.long),
              "scores": torch.zeros(0)}],
            [{"boxes": torch.zeros(0, 6), "labels": torch.zeros(0, dtype=torch.long)}],
        )
        result = metric.aggregate()
        assert math.isnan(result["box_match_recall"])
        assert math.isnan(result["box_miou"])

    def test_mask_match_recall_nan_when_no_gt(self):
        metric = MaskMIoUMetric(mode="instance")
        metric.add_matched_ious([], n_gt=0)
        result = metric.aggregate()
        assert math.isnan(result["mask_match_recall"])

    def test_binary_ap_nan_on_zero_positives(self):
        metric = ClassAPMetric()
        metric(torch.rand(8), torch.zeros(8))
        assert math.isnan(metric.aggregate())

    def test_multiclass_macro_mean_not_poisoned(self):
        # class 1 has zero positives -> skipped, macro mean over class 0 only.
        scores = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.2, 0.6]])
        targets = torch.tensor([0, 0, 0])
        metric = ClassAPMetric(num_classes=2)
        metric(scores, targets)
        result = metric.aggregate()
        assert not math.isnan(result)
        assert result == pytest.approx(1.0)
