"""Exact-assertion tests for PredictedIoUEvalMetric.

Model-agnostic evaluation of a predicted-IoU head (SAM2 / Mask Scoring R-CNN
self-assessed mask quality). All three measurements ride the SAME streaming
``_stream`` MaskMAPMetric gathers across ranks, so cross-rank reduction is
exact and is tested through the PURE merge helper (no live process group).

Each measurement is checked against a brute-force / scipy / hand recompute:

  1. CALIBRATION:  iou_head_mae / rmse vs numpy; pearson / spearman vs scipy.
  2. SELECTION CURVE: true_miou@t / precision@t / coverage@t vs brute force;
     selection_auc_prediou > selection_auc_score when pred_iou is a perfect
     quality predictor, and NOT when it is anti-correlated.
  3. RANKED AP CONSISTENCY: map_rank_score == a standard MaskMAPMetric on the
     identical fragments; pred_iou ranking beats score ranking when a high-IoU
     mask has low class score; score_x_prediou >= max of the two there.
  4. CROSS-RANK EXACT: _merge_streams over two shards with COLLIDING image_ids
     == a single metric fed the union with globally-unique ids (calibration,
     selection, and ranked-AP keys); plus the un-namespaced counterexample.

All CPU-only; the pure merge path is exercised directly.
"""

import math

import numpy as np
import pytest
import torch
from scipy import stats

from cell_observatory_platform.evaluation.metrics import (
    MaskMAPMetric,
    PredictedIoUEvalMetric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _true_quality_np(ious_row):
    """Best IoU to any GT for one detection (0.0 for empty row)."""
    ious_row = np.asarray(ious_row, dtype=np.float64)
    return float(ious_row.max()) if ious_row.size else 0.0


def _selection_auc_bruteforce(selector, true_iou):
    """Reference for PredictedIoUEvalMetric._selection_auc.

    Sort by selector desc (stable), running mean of true_iou over the retained
    prefix vs coverage = i/N, trapezoid with a left anchor at coverage 0 equal
    to running_mean[0].
    """
    selector = np.asarray(selector, dtype=np.float64)
    true_iou = np.asarray(true_iou, dtype=np.float64)
    n = selector.size
    if n == 0:
        return 0.0
    # stable descending sort == argsort of -selector with stable kind on the
    # negated values reproduces torch.argsort(descending=True, stable=True).
    order = np.argsort(-selector, kind="stable")
    ti = true_iou[order]
    running_mean = np.cumsum(ti) / np.arange(1, n + 1)
    coverage = np.arange(1, n + 1) / n
    x = np.concatenate([[0.0], coverage])
    y = np.concatenate([running_mean[:1], running_mean])
    return float(np.trapezoid(y, x))


# ===========================================================================
# 1. CALIBRATION
# ===========================================================================


def test_calibration_mae_rmse_pearson_spearman_exact():
    """Push detections with known (pred_iou, true_iou=max-over-row) pairs and
    assert mae/rmse equal numpy and pearson/spearman equal scipy."""
    # Hand-built fragments across two images, one class. true_iou = max over row.
    # Image 0: 3 preds. true_iou rows give max = [0.80, 0.20, 0.55].
    # Image 1: 2 preds. true_iou rows give max = [0.95, 0.10].
    metric = PredictedIoUEvalMetric(
        iou_thresholds=[0.5], pred_iou_thresholds=[0.5]
    )
    metric.add_image_class(
        image_id=0, class_id=1,
        scores=torch.tensor([0.6, 0.4, 0.5]),
        ious=torch.tensor([[0.80, 0.10],
                           [0.20, 0.05],
                           [0.55, 0.30]]),
        n_gt=2,
        pred_ious=torch.tensor([0.70, 0.30, 0.60]),
    )
    metric.add_image_class(
        image_id=1, class_id=1,
        scores=torch.tensor([0.9, 0.2]),
        ious=torch.tensor([[0.95], [0.10]]),
        n_gt=1,
        pred_ious=torch.tensor([0.90, 0.15]),
    )

    out = metric.aggregate()

    # Pooled in stream order (image 0 then image 1).
    pred = np.array([0.70, 0.30, 0.60, 0.90, 0.15])
    true = np.array([0.80, 0.20, 0.55, 0.95, 0.10])
    err = pred - true

    assert out["iou_head_mae"] == pytest.approx(np.abs(err).mean(), abs=1e-6)
    assert out["iou_head_rmse"] == pytest.approx(
        np.sqrt((err ** 2).mean()), abs=1e-6
    )
    assert out["iou_head_pearson"] == pytest.approx(
        stats.pearsonr(pred, true)[0], abs=1e-6
    )
    assert out["iou_head_spearman"] == pytest.approx(
        stats.spearmanr(pred, true)[0], abs=1e-6
    )


def test_calibration_spearman_handles_ties():
    """Spearman with tied pred_iou values must match scipy's fractional-rank
    Spearman (the metric uses average/fractional ranks)."""
    metric = PredictedIoUEvalMetric(
        iou_thresholds=[0.5], pred_iou_thresholds=[0.5]
    )
    # pred_ious has a tie (0.5, 0.5); true_iou distinct.
    metric.add_image_class(
        image_id=0, class_id=1,
        scores=torch.tensor([0.5, 0.5, 0.5, 0.5]),
        ious=torch.tensor([[0.10], [0.90], [0.40], [0.60]]),
        n_gt=1,
        pred_ious=torch.tensor([0.50, 0.50, 0.30, 0.80]),
    )
    out = metric.aggregate()
    pred = np.array([0.50, 0.50, 0.30, 0.80])
    true = np.array([0.10, 0.90, 0.40, 0.60])
    assert out["iou_head_spearman"] == pytest.approx(
        stats.spearmanr(pred, true)[0], abs=1e-6
    )
    assert out["iou_head_pearson"] == pytest.approx(
        stats.pearsonr(pred, true)[0], abs=1e-6
    )


def test_calibration_empty_row_true_quality_is_zero():
    """A prediction on an image with NO GT of the class (empty IoU row) has
    true_iou 0.0, which must flow into the calibration error."""
    metric = PredictedIoUEvalMetric(
        iou_thresholds=[0.5], pred_iou_thresholds=[0.5]
    )
    # Image 0: 2 preds, n_gt=0 -> ious shape (2,0). true_iou = [0,0].
    metric.add_image_class(
        image_id=0, class_id=1,
        scores=torch.tensor([0.7, 0.3]),
        ious=torch.zeros(2, 0),
        n_gt=0,
        pred_ious=torch.tensor([0.40, 0.60]),
    )
    out = metric.aggregate()
    pred = np.array([0.40, 0.60])
    true = np.array([0.0, 0.0])
    assert out["iou_head_mae"] == pytest.approx(np.abs(pred - true).mean(), abs=1e-6)


# ===========================================================================
# 2. SELECTION CURVE
# ===========================================================================


def test_selection_true_miou_precision_coverage_bruteforce():
    """true_miou@t / precision@t / coverage@t match a brute-force recompute at
    several thresholds."""
    thresholds = [0.3, 0.5, 0.7, 0.95]
    metric = PredictedIoUEvalMetric(
        iou_thresholds=[0.5],
        pred_iou_thresholds=thresholds,
        match_iou_threshold=0.5,
    )
    # 5 preds across two images; known pred_iou and true_iou (max over row).
    metric.add_image_class(
        image_id=0, class_id=1,
        scores=torch.tensor([0.5, 0.5, 0.5]),
        ious=torch.tensor([[0.80], [0.45], [0.20]]),
        n_gt=1,
        pred_ious=torch.tensor([0.90, 0.50, 0.10]),
    )
    metric.add_image_class(
        image_id=1, class_id=1,
        scores=torch.tensor([0.5, 0.5]),
        ious=torch.tensor([[0.60], [0.30]]),
        n_gt=1,
        pred_ious=torch.tensor([0.70, 0.35]),
    )
    out = metric.aggregate()

    pred = np.array([0.90, 0.50, 0.10, 0.70, 0.35])
    true = np.array([0.80, 0.45, 0.20, 0.60, 0.30])
    total = pred.size
    for t in thresholds:
        keep = pred >= t
        n_keep = int(keep.sum())
        kept_true = true[keep]
        exp_miou = float(kept_true.mean()) if n_keep else 0.0
        exp_prec = float((kept_true >= 0.5).mean()) if n_keep else 0.0
        exp_cov = n_keep / total
        assert out[f"true_miou@{t}"] == pytest.approx(exp_miou, abs=1e-6)
        assert out[f"precision@{t}"] == pytest.approx(exp_prec, abs=1e-6)
        assert out[f"coverage@{t}"] == pytest.approx(exp_cov, abs=1e-6)


def test_selection_auc_keys_present_and_bruteforce():
    """selection_auc_prediou / selection_auc_score are present and equal the
    brute-force trapezoid integral."""
    metric = PredictedIoUEvalMetric(
        iou_thresholds=[0.5], pred_iou_thresholds=[0.5]
    )
    metric.add_image_class(
        image_id=0, class_id=1,
        scores=torch.tensor([0.30, 0.90, 0.60, 0.10]),
        ious=torch.tensor([[0.80], [0.20], [0.55], [0.95]]),
        n_gt=1,
        pred_ious=torch.tensor([0.85, 0.25, 0.50, 0.99]),
    )
    out = metric.aggregate()
    assert "selection_auc_prediou" in out
    assert "selection_auc_score" in out

    pred = np.array([0.85, 0.25, 0.50, 0.99])
    score = np.array([0.30, 0.90, 0.60, 0.10])
    true = np.array([0.80, 0.20, 0.55, 0.95])
    assert out["selection_auc_prediou"] == pytest.approx(
        _selection_auc_bruteforce(pred, true), abs=1e-6
    )
    assert out["selection_auc_score"] == pytest.approx(
        _selection_auc_bruteforce(score, true), abs=1e-6
    )


def test_selection_auc_prediou_beats_score_when_perfect_predictor():
    """When pred_iou is a PERFECT quality predictor (monotone in true_iou) and
    the class score is anti-correlated with quality, selecting by pred_iou
    yields a strictly higher quality-vs-coverage AUC than selecting by score."""
    metric = PredictedIoUEvalMetric(
        iou_thresholds=[0.5], pred_iou_thresholds=[0.5]
    )
    # pred_iou == true_iou (perfect). score is the REVERSE ordering (worst
    # quality has highest score), so score-selection front-loads bad masks.
    true = [0.95, 0.80, 0.60, 0.40, 0.20, 0.05]
    metric.add_image_class(
        image_id=0, class_id=1,
        scores=torch.tensor([0.10, 0.20, 0.30, 0.40, 0.50, 0.60]),
        ious=torch.tensor([[v] for v in true]),
        n_gt=1,
        pred_ious=torch.tensor(true),
    )
    out = metric.aggregate()
    assert out["selection_auc_prediou"] > out["selection_auc_score"]


def test_selection_auc_prediou_does_not_beat_score_when_anticorrelated():
    """When pred_iou is ANTI-correlated with true quality (it actively
    front-loads the worst masks) it must NOT beat the score selector."""
    metric = PredictedIoUEvalMetric(
        iou_thresholds=[0.5], pred_iou_thresholds=[0.5]
    )
    true = [0.95, 0.80, 0.60, 0.40, 0.20, 0.05]
    # pred_iou is the reverse of quality (anti-correlated): highest pred_iou on
    # the worst mask. score is a PERFECT predictor here.
    metric.add_image_class(
        image_id=0, class_id=1,
        scores=torch.tensor(true),
        ious=torch.tensor([[v] for v in true]),
        n_gt=1,
        pred_ious=torch.tensor(list(reversed(true))),
    )
    out = metric.aggregate()
    assert out["selection_auc_prediou"] < out["selection_auc_score"]


# ===========================================================================
# 3. RANKED AP CONSISTENCY
# ===========================================================================


def _fragments_for_ranked_ap():
    """Two images, one class, with scores/ious/pred_ious where score and
    pred_iou disagree about ordering."""
    # Image 0: 2 preds, 1 gt.
    #   pred A: IoU 0.0 (FP), score 0.95 (high), pred_iou 0.10 (low).
    #   pred B: IoU 1.0 (TP), score 0.70 (low),  pred_iou 0.99 (high).
    # Image 1: 1 pred, 1 gt: IoU 1.0 (TP), score 0.80, pred_iou 0.90.
    frag0 = dict(
        image_id=0, class_id=1,
        scores=torch.tensor([0.95, 0.70]),
        ious=torch.tensor([[0.0], [1.0]]),
        n_gt=1,
        pred_ious=torch.tensor([0.10, 0.99]),
    )
    frag1 = dict(
        image_id=1, class_id=1,
        scores=torch.tensor([0.80]),
        ious=torch.tensor([[1.0]]),
        n_gt=1,
        pred_ious=torch.tensor([0.90]),
    )
    return frag0, frag1


def test_map_rank_score_equals_standard_mask_map():
    """map_rank_score must EXACTLY equal a standard MaskMAPMetric fed the
    identical fragments (same iou_thresholds)."""
    frag0, frag1 = _fragments_for_ranked_ap()
    iou_thrs = [0.5, 0.75]

    metric = PredictedIoUEvalMetric(iou_thresholds=iou_thrs)
    metric.add_image_class(**frag0)
    metric.add_image_class(**frag1)
    out = metric.aggregate()

    ref = MaskMAPMetric(iou_thresholds=iou_thrs)
    # Feed the same fragments WITHOUT pred_ious (MaskMAPMetric has no such arg).
    ref.add_image_class(
        image_id=frag0["image_id"], class_id=frag0["class_id"],
        scores=frag0["scores"], ious=frag0["ious"], n_gt=frag0["n_gt"],
    )
    ref.add_image_class(
        image_id=frag1["image_id"], class_id=frag1["class_id"],
        scores=frag1["scores"], ious=frag1["ious"], n_gt=frag1["n_gt"],
    )
    ref_value = ref.aggregate()

    assert out["map_rank_score"] == pytest.approx(ref_value, abs=1e-9)


def test_map_rank_prediou_beats_score_when_prediou_orders_better():
    """When a high-IoU mask has a low class score but a high pred_iou, ranking
    by pred_iou recovers a better AP than ranking by score; and the
    score*pred_iou ranking is at least as good as the best of the two."""
    frag0, frag1 = _fragments_for_ranked_ap()
    iou_thrs = [0.5]

    metric = PredictedIoUEvalMetric(iou_thresholds=iou_thrs)
    metric.add_image_class(**frag0)
    metric.add_image_class(**frag1)
    out = metric.aggregate()

    # Score ranking puts the FP (score 0.95) ahead of the TP (0.70) in image 0,
    # hurting precision; pred_iou ranking puts the TP (0.99) first -> better AP.
    assert out["map_rank_prediou"] > out["map_rank_score"]
    # score*pred_iou: image-0 keys are 0.95*0.10=0.095 (FP) and 0.70*0.99=0.693
    # (TP), so the TP leads -> as good as the pred_iou ranking here.
    assert out["map_rank_score_x_prediou"] >= max(
        out["map_rank_score"], out["map_rank_prediou"]
    ) - 1e-9


# ===========================================================================
# 4. CROSS-RANK EXACT (pure merge helper, colliding image_ids)
# ===========================================================================


def _shard_a():
    """Rank-0 shard: image_id 0, one class."""
    return [dict(
        image_id=0, class_id=1,
        scores=torch.tensor([0.95, 0.70]),
        ious=torch.tensor([[0.0], [1.0]]),
        n_gt=1,
        pred_ious=torch.tensor([0.10, 0.99]),
    )]


def _shard_b():
    """Rank-1 shard: image_id 0 (COLLIDES with shard A), one class."""
    return [dict(
        image_id=0, class_id=1,
        scores=torch.tensor([0.80, 0.40]),
        ious=torch.tensor([[1.0], [0.0]]),
        n_gt=1,
        pred_ious=torch.tensor([0.90, 0.20]),
    )]


def _clone_stream(stream):
    """Deep-ish copy so merge/aggregate of one path can't mutate the other."""
    out = []
    for e in stream:
        ne = dict(e)
        for k in ("scores", "ious", "pred_ious"):
            ne[k] = e[k].clone()
        out.append(ne)
    return out


def test_cross_rank_merge_equals_union_all_key_families():
    """_merge_streams over two shards with COLLIDING image_ids == a single
    metric fed the union with globally-unique image_ids, for calibration,
    selection, and ranked-AP keys."""
    iou_thrs = [0.5, 0.75]
    pred_thrs = [0.5, 0.9]

    # Merged path: pure helper namespaces shard-1's image_id 0 -> 1.
    merged = PredictedIoUEvalMetric(
        iou_thresholds=iou_thrs, pred_iou_thresholds=pred_thrs
    )
    merged._stream = MaskMAPMetric._merge_streams([_shard_a(), _shard_b()])
    merged_out = merged.aggregate()

    # Reference: one metric fed the union with UNIQUE image_ids (0 and 1).
    ref = PredictedIoUEvalMetric(
        iou_thresholds=iou_thrs, pred_iou_thresholds=pred_thrs
    )
    a = _shard_a()[0]
    b = _shard_b()[0]
    ref.add_image_class(image_id=0, class_id=1, scores=a["scores"],
                        ious=a["ious"], n_gt=a["n_gt"], pred_ious=a["pred_ious"])
    ref.add_image_class(image_id=1, class_id=1, scores=b["scores"],
                        ious=b["ious"], n_gt=b["n_gt"], pred_ious=b["pred_ious"])
    ref_out = ref.aggregate()

    # Calibration scalars (pooled-detection order is identical -> exact).
    for k in ("iou_head_mae", "iou_head_rmse",
              "iou_head_pearson", "iou_head_spearman"):
        assert merged_out[k] == pytest.approx(ref_out[k], abs=1e-6), k
    # A couple of selection keys + the coverage-integrated AUCs.
    for k in ("true_miou@0.5", "coverage@0.5", "precision@0.9",
              "selection_auc_prediou", "selection_auc_score"):
        assert merged_out[k] == pytest.approx(ref_out[k], abs=1e-6), k
    # Ranked-AP keys.
    for k in ("map_rank_score", "map_rank_prediou", "map_rank_score_x_prediou"):
        assert merged_out[k] == pytest.approx(ref_out[k], abs=1e-9), k


def test_cross_rank_unnamespaced_merge_is_wrong_counterexample():
    """Namespacing is load-bearing: a naive un-namespaced merge collapses the
    two image_id=0 GT buckets into one, corrupting the ranked-AP answer.

    Shard A's TP (IoU 1.0) and shard B's TP (IoU 1.0) each match their OWN
    image's single GT. Un-namespaced, both image_id=0 entries share ONE 1-GT
    bucket: the first-ranked TP grabs the GT, the second equally-perfect TP is
    forced to be a false positive AND the recall denominator collapses from 2
    GT to 1 -> map_rank_score drops below the correctly-namespaced value.
    """
    iou_thrs = [0.5]

    correct = PredictedIoUEvalMetric(iou_thresholds=iou_thrs)
    correct._stream = MaskMAPMetric._merge_streams([_shard_a(), _shard_b()])
    correct_out = correct.aggregate()

    wrong = PredictedIoUEvalMetric(iou_thresholds=iou_thrs)
    # Un-namespaced: both shards keep image_id 0 (collision NOT resolved).
    wrong._stream = _clone_stream(_shard_a()) + _clone_stream(_shard_b())
    wrong_out = wrong.aggregate()

    assert wrong_out["map_rank_score"] != pytest.approx(
        correct_out["map_rank_score"], abs=1e-6
    )
    assert wrong_out["map_rank_score"] < correct_out["map_rank_score"]


def test_merge_single_shard_noop_identity():
    """A single-shard merge is a no-op identity (image_id unchanged, stride
    applies offset 0 to rank 0): aggregate equals the un-merged metric."""
    direct = PredictedIoUEvalMetric(iou_thresholds=[0.5], pred_iou_thresholds=[0.5])
    a = _shard_a()[0]
    direct.add_image_class(image_id=a["image_id"], class_id=a["class_id"],
                           scores=a["scores"], ious=a["ious"], n_gt=a["n_gt"],
                           pred_ious=a["pred_ious"])
    direct_out = direct.aggregate()

    merged = PredictedIoUEvalMetric(iou_thresholds=[0.5], pred_iou_thresholds=[0.5])
    merged._stream = MaskMAPMetric._merge_streams([_shard_a()])
    merged_out = merged.aggregate()

    for k in direct_out:
        assert merged_out[k] == pytest.approx(direct_out[k], abs=1e-9), k


# ===========================================================================
# Validation / API guards
# ===========================================================================


def test_add_image_class_rejects_pred_iou_shape_mismatch():
    metric = PredictedIoUEvalMetric()
    with pytest.raises(ValueError, match="pred_ious shape"):
        metric.add_image_class(
            image_id=0, class_id=1,
            scores=torch.tensor([0.9, 0.8]),
            ious=torch.tensor([[1.0], [0.0]]),
            n_gt=1,
            pred_ious=torch.tensor([0.5]),  # wrong length
        )


def test_add_image_class_rejects_iou_ngt_mismatch():
    metric = PredictedIoUEvalMetric()
    with pytest.raises(ValueError, match="disagrees with n_gt"):
        metric.add_image_class(
            image_id=0, class_id=1,
            scores=torch.tensor([0.9]),
            ious=torch.zeros(1, 2),
            n_gt=1,
            pred_ious=torch.tensor([0.5]),
        )


def test_add_image_class_rejects_duplicate_key():
    metric = PredictedIoUEvalMetric()
    metric.add_image_class(
        image_id=0, class_id=1,
        scores=torch.tensor([0.9]), ious=torch.tensor([[1.0]]),
        n_gt=1, pred_ious=torch.tensor([0.8]),
    )
    with pytest.raises(ValueError, match="Duplicate add_image_class"):
        metric.add_image_class(
            image_id=0, class_id=1,
            scores=torch.tensor([0.5]), ious=torch.tensor([[1.0]]),
            n_gt=1, pred_ious=torch.tensor([0.4]),
        )


def test_max_detections_capped_per_ranking_not_by_score():
    """max_detections is applied PER RANKING in ranked-AP (not by score up
    front), and calibration sees EVERY proposal (uncapped).

    Constructed so a single global score-cap would corrupt the pred_iou
    ranking: the only true positive (det C) has a LOW score but the HIGHEST
    predicted IoU. With the old score-based cap (keep top-2 by score) C would
    be dropped before the pred_iou ranking could use it, giving a WRONG
    map_rank_prediou of 0.0. The per-ranking cap keeps C for the pred_iou
    ranking, recovering the correct AP of 1.0.
    """
    metric = PredictedIoUEvalMetric(
        iou_thresholds=[0.5], pred_iou_thresholds=[0.5], max_detections=2
    )
    # A: score 0.9, pred_iou 0.1, true 0.0 (FP, confidently-scored, low quality)
    # B: score 0.8, pred_iou 0.2, true 0.0 (FP)
    # C: score 0.3, pred_iou 0.95, true 1.0 (the ONLY TP; low score, high quality)
    metric.add_image_class(
        image_id=0, class_id=1,
        scores=torch.tensor([0.9, 0.8, 0.3]),
        ious=torch.tensor([[0.0], [0.0], [1.0]]),
        n_gt=1,
        pred_ious=torch.tensor([0.1, 0.2, 0.95]),
    )
    out = metric.aggregate()

    # Calibration is UNCAPPED: all three (pred, true) pairs contribute.
    exp_mae = (abs(0.1 - 0.0) + abs(0.2 - 0.0) + abs(0.95 - 1.0)) / 3
    assert out["iou_head_mae"] == pytest.approx(exp_mae, abs=1e-6)

    # Ranking by score caps to {A, B} (both FP) -> no TP reachable -> AP 0.0.
    assert out["map_rank_score"] == pytest.approx(0.0, abs=1e-9)
    # Ranking by pred_iou caps to its OWN top-2 {C, B}; C (TP) ranks first ->
    # recall hits 1.0 at precision 1.0 -> AP 1.0. Under the old score-cap C was
    # dropped and this would have been 0.0.
    assert out["map_rank_prediou"] == pytest.approx(1.0, abs=1e-9)


def test_non_finite_pred_iou_sanitized_consistently():
    """A NaN/inf predicted IoU is mapped to 0.0 so calibration and selection
    agree, instead of NaN-poisoning mae/rmse/pearson while selection silently
    drops it."""
    metric = PredictedIoUEvalMetric(
        iou_thresholds=[0.5], pred_iou_thresholds=[0.5], max_detections=100
    )
    metric.add_image_class(
        image_id=0, class_id=1,
        scores=torch.tensor([0.9, 0.8]),
        ious=torch.tensor([[1.0], [1.0]]),
        n_gt=1,
        pred_ious=torch.tensor([float("nan"), 0.5]),
    )
    out = metric.aggregate()
    # NaN -> 0.0: pairs become (0.0, true 1.0) and (0.5, true 1.0); all finite.
    assert math.isfinite(out["iou_head_mae"])
    assert math.isfinite(out["iou_head_rmse"])
    assert math.isfinite(out["iou_head_pearson"])
    assert math.isfinite(out["iou_head_spearman"])
    exp_mae = (abs(0.0 - 1.0) + abs(0.5 - 1.0)) / 2
    assert out["iou_head_mae"] == pytest.approx(exp_mae, abs=1e-6)
    # The sanitized det (pred_iou 0.0) is consistently EXCLUDED from selection.
    assert out["coverage@0.5"] == pytest.approx(0.5, abs=1e-9)


def test_call_is_noop_and_reset_clears_state():
    metric = PredictedIoUEvalMetric()
    metric(outputs=None, targets=None)  # no-op, must not raise
    metric.add_image_class(
        image_id=0, class_id=1,
        scores=torch.tensor([0.9]), ious=torch.tensor([[1.0]]),
        n_gt=1, pred_ious=torch.tensor([0.8]),
    )
    assert metric._stream
    metric.reset()
    assert metric._stream == []
    assert metric._seen_keys == set()
    assert metric._gathered is False


def test_empty_stream_aggregate_returns_zero_dict():
    metric = PredictedIoUEvalMetric(
        iou_thresholds=[0.5], pred_iou_thresholds=[0.5]
    )
    out = metric.aggregate()
    assert out["iou_head_mae"] == 0.0
    assert out["iou_head_pearson"] == 0.0
    assert out["selection_auc_prediou"] == 0.0
    assert out["map_rank_score"] == 0.0
    assert all(isinstance(v, float) for v in out.values())
