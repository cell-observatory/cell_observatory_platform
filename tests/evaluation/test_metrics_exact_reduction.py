"""Exact-reduction tests for the streaming push-API metrics.

Covers the three properties of the agreed compact-sufficient-statistics design:

1. STREAMING == DENSE (single process): the streaming push API per-rank
   perfectly recapitulates its dense ``__call__`` counterpart.
2. MERGE(gathered) == UNION: the pure, dist-free static merge helpers pool
   per-rank shards into exactly the state a single dense run over the union
   would produce -- including the image_id namespacing that keeps per-image GT
   buckets disjoint across ranks.
3. AP SCORE-THRESHOLD PARITY: the evaluator feeds the FULL unfiltered detection
   set to MaskMAP (AP) while applying score_threshold only to the matched-IoU
   (instance mIoU) subset.

All tests are CPU-only and exercise the PURE merge helpers directly (no live
process group / all_gather_object).
"""

import math

import pytest
import torch

from cell_observatory_platform.evaluation.metrics import (
    BoxF1Metric,
    BoxMAPMetric,
    BoxMIoUMetric,
    ClassAPMetric,
    MaskMAPMetric,
    MaskMIoUMetric,
)
from cell_observatory_platform.evaluation.instance_segmentation_evaluator import (
    InstanceSegmentationEvaluator,
)


# ---------------------------------------------------------------------------
# Helpers: tiny hand-built bool masks with known IoUs.
# ---------------------------------------------------------------------------


def _voxel_mask(coords, shape=(2, 2, 2)):
    """Build a (D,H,W) bool mask with the given list of True voxel coords."""
    m = torch.zeros(*shape, dtype=torch.bool)
    for c in coords:
        m[c] = True
    return m


# ===========================================================================
# 1. STREAMING == DENSE (single process)
# ===========================================================================


def test_mask_map_streaming_equals_dense_multi_image():
    """MaskMAP: dense __call__ over masks == streaming add_image_class fed the
    same per-image IoU matrices/scores/n_gt."""
    # Two images, one class. Build masks so IoUs are exactly computable.
    # Image 0: 1 pred, 1 gt, IoU = 1.0, score 0.9 (TP).
    # Image 1: 2 preds, 1 gt. pred A (score 0.95) IoU 0.0 (FP),
    #          pred B (score 0.7) IoU 1.0 (TP).
    iou_thr = 0.5

    # ---- Dense inputs (materialized bool masks). ----
    full = [(0, 0, 0), (0, 0, 1), (1, 1, 1)]  # arbitrary distinct voxels

    img0_gt = _voxel_mask([(0, 0, 0)])
    img0_pred = _voxel_mask([(0, 0, 0)])  # IoU 1.0 with gt

    img1_gt = _voxel_mask([(1, 1, 1)])
    img1_pred_fp = _voxel_mask([(0, 0, 1)])  # IoU 0.0 with gt
    img1_pred_tp = _voxel_mask([(1, 1, 1)])  # IoU 1.0 with gt

    preds = [
        {
            "masks": img0_pred.unsqueeze(0),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.9]),
        },
        {
            "masks": torch.stack([img1_pred_fp, img1_pred_tp]),
            "labels": torch.tensor([1, 1]),
            "scores": torch.tensor([0.95, 0.7]),
        },
    ]
    targets = [
        {"masks": img0_gt.unsqueeze(0), "labels": torch.tensor([1])},
        {"masks": img1_gt.unsqueeze(0), "labels": torch.tensor([1])},
    ]

    dense = MaskMAPMetric(iou_thresholds=[iou_thr])
    dense(preds, targets)
    dense_value = dense.aggregate()

    # ---- Streaming inputs (the SAME IoU matrices, computed by hand). ----
    stream = MaskMAPMetric(iou_thresholds=[iou_thr])
    # Image 0: (1 pred x 1 gt) IoU matrix.
    stream.add_image_class(
        image_id=0,
        class_id=1,
        scores=torch.tensor([0.9]),
        ious=torch.tensor([[1.0]]),
        n_gt=1,
    )
    # Image 1: (2 preds x 1 gt). Row order matches score order of the preds.
    stream.add_image_class(
        image_id=1,
        class_id=1,
        scores=torch.tensor([0.95, 0.7]),
        ious=torch.tensor([[0.0], [1.0]]),
        n_gt=1,
    )
    stream_value = stream.aggregate()

    assert stream_value == pytest.approx(dense_value, abs=1e-6)
    # And both are finite / in a sane range.
    assert math.isfinite(dense_value)


def test_mask_miou_instance_streaming_equals_dense():
    """instance-mode MaskMIoU: dense __call__ (greedy match over synthetic
    masks) == add_matched_ious fed the greedy-matched IoUs."""
    iou_thr = 0.5

    # One image, 2 preds, 2 gts.
    # gt0 at (0,0,0); gt1 at (1,1,1).
    # pred0 (score 0.9): voxels {(0,0,0),(0,0,1)} -> IoU with gt0 = 1/2 = 0.5,
    #                    IoU with gt1 = 0.
    # pred1 (score 0.8): voxels {(1,1,1)} -> IoU gt0 = 0, IoU gt1 = 1.0.
    gt0 = _voxel_mask([(0, 0, 0)])
    gt1 = _voxel_mask([(1, 1, 1)])
    pred0 = _voxel_mask([(0, 0, 0), (0, 0, 1)])
    pred1 = _voxel_mask([(1, 1, 1)])

    preds = [
        {
            "masks": torch.stack([pred0, pred1]),
            "labels": torch.tensor([1, 1]),
            "scores": torch.tensor([0.9, 0.8]),
        }
    ]
    targets = [{"masks": torch.stack([gt0, gt1]), "labels": torch.tensor([1, 1])}]

    dense = MaskMIoUMetric(mode="instance", iou_threshold=iou_thr)
    dense(preds, targets)
    dense_value = dense.aggregate()

    # Greedy by score: pred0 -> gt0 (IoU 0.5), pred1 -> gt1 (IoU 1.0).
    # Both >= 0.5, so matched list = [0.5, 1.0].
    stream = MaskMIoUMetric(mode="instance", iou_threshold=iou_thr)
    stream.add_matched_ious([0.5, 1.0])
    stream_value = stream.aggregate()

    assert dense_value == pytest.approx((0.5 + 1.0) / 2)
    assert stream_value == pytest.approx(dense_value, abs=1e-6)


# ===========================================================================
# 2. MERGE(gathered) == UNION  (pure static merge helpers)
# ===========================================================================


def test_mask_map_merge_streams_equals_union_with_namespacing():
    """MaskMAPMetric._merge_streams over two shards (with COLLIDING image_ids)
    equals a single metric fed the union with globally-unique image_ids; and
    namespacing is what makes it correct.

    The collision is constructed so that NOT namespacing genuinely corrupts the
    answer: each shard's image_id 0 has its own single GT, and each shard's
    single prediction is a perfect (IoU 1.0) TP for ITS OWN GT. If the two
    image_id=0 buckets are wrongly merged into one shared 1-GT bucket, the
    higher-scored prediction steals the only GT slot and the lower-scored
    (equally perfect) prediction is forced to be a false positive.
    """
    iou_thr = 0.5

    # Rank 0, image_id 0: 1 pred, IoU 1.0 vs its own 1 GT (TP), score 0.9.
    rank0_stream = [
        {
            "image_id": 0,
            "class_id": 1,
            "scores": torch.tensor([0.9]),
            "ious": torch.tensor([[1.0]]),
            "n_gt": 1,
        }
    ]
    # Rank 1, image_id 0 (COLLIDES): 1 pred, IoU 1.0 vs its own 1 GT (TP),
    # score 0.95 (ranks ABOVE rank0's TP globally).
    rank1_stream = [
        {
            "image_id": 0,
            "class_id": 1,
            "scores": torch.tensor([0.95]),
            "ious": torch.tensor([[1.0]]),
            "n_gt": 1,
        }
    ]

    # ---- Merge via the pure helper, then aggregate. ----
    merged = MaskMAPMetric._merge_streams([rank0_stream, rank1_stream])
    merged_metric = MaskMAPMetric(iou_thresholds=[iou_thr])
    merged_metric._stream = merged
    merged_value = merged_metric.aggregate()

    # ---- Reference: single metric fed the union with UNIQUE image_ids. ----
    ref = MaskMAPMetric(iou_thresholds=[iou_thr])
    ref.add_image_class(image_id=0, class_id=1,
                        scores=torch.tensor([0.9]), ious=torch.tensor([[1.0]]), n_gt=1)
    ref.add_image_class(image_id=1, class_id=1,
                        scores=torch.tensor([0.95]), ious=torch.tensor([[1.0]]), n_gt=1)
    ref_value = ref.aggregate()

    # Two perfect TPs over two disjoint 1-GT buckets -> AP 1.0.
    assert ref_value == pytest.approx(1.0)
    assert merged_value == pytest.approx(ref_value, abs=1e-6)

    # ---- Prove namespacing is load-bearing. ----
    # Un-namespaced: both image_id=0 entries share ONE 1-GT bucket. The
    # higher-scored pred (0.95) matches the lone GT; the equally-perfect 0.9
    # pred then finds the GT already taken -> forced FP. n_gt_total is also
    # collapsed (max per image = 1, not summed to 2). The AP drops below 1.0.
    wrong = MaskMAPMetric(iou_thresholds=[iou_thr])
    wrong._stream = [dict(rank0_stream[0]), dict(rank1_stream[0])]  # both image_id=0
    wrong_value = wrong.aggregate()
    assert wrong_value < 1.0
    assert wrong_value != pytest.approx(ref_value, abs=1e-6)


def test_mask_map_merge_streams_namespacing_disjoint_gt_buckets():
    """Direct proof: two shards each with image_id 0 AND 1, perfect TPs. The
    merged answer must be perfect AP (1.0); a naive (un-namespaced) merge would
    collapse 4 images into 2 GT buckets and double-count GT -> wrong recall."""
    iou_thr = 0.5

    def perfect_shard():
        return [
            {"image_id": 0, "class_id": 1, "scores": torch.tensor([0.9]),
             "ious": torch.tensor([[1.0]]), "n_gt": 1},
            {"image_id": 1, "class_id": 1, "scores": torch.tensor([0.8]),
             "ious": torch.tensor([[1.0]]), "n_gt": 1},
        ]

    merged = MaskMAPMetric._merge_streams([perfect_shard(), perfect_shard()])
    # Namespacing: stride = 1 + max_image_id = 2. Rank 1 images become 2 and 3.
    image_ids = sorted(e["image_id"] for e in merged)
    assert image_ids == [0, 1, 2, 3]

    m = MaskMAPMetric(iou_thresholds=[iou_thr])
    m._stream = merged
    # 4 perfect TPs across 4 disjoint GT buckets -> AP 1.0.
    assert m.aggregate() == pytest.approx(1.0)


def test_mask_miou_instance_merge_matched_ious_equals_union_mean():
    """instance mIoU: concat per-rank matched-IoU lists -> mean over union."""
    rank0 = [0.6, 0.9]
    rank1 = [0.5, 1.0, 0.7]

    merged = MaskMIoUMetric._merge_matched_ious([rank0, rank1])
    m = MaskMIoUMetric(mode="instance", iou_threshold=0.5)
    m._matched_ious = merged
    merged_value = m.aggregate()

    ref = MaskMIoUMetric(mode="instance", iou_threshold=0.5)
    ref.add_matched_ious(rank0 + rank1)
    ref_value = ref.aggregate()

    assert merged == rank0 + rank1
    assert merged_value == pytest.approx(sum(rank0 + rank1) / 5)
    assert merged_value == pytest.approx(ref_value, abs=1e-6)


def test_mask_miou_semantic_merge_inter_union_sums_per_class():
    """semantic mIoU: sum per-class inter/union across ranks; aggregate is
    Sum(inter)/Sum(union) per class, NOT a mean of per-rank ratios."""
    # Rank 0: class 1 inter=2 union=4 (ratio .5); class 2 inter=1 union=10 (.1).
    # Rank 1: class 1 inter=8 union=8 (ratio 1.0); class 2 inter=4 union=10 (.4).
    rank0_inter = {1: 2, 2: 1}
    rank0_union = {1: 4, 2: 10}
    rank1_inter = {1: 8, 2: 4}
    rank1_union = {1: 8, 2: 10}

    inter, union = MaskMIoUMetric._merge_inter_union(
        [rank0_inter, rank1_inter], [rank0_union, rank1_union]
    )
    assert inter == {1: 10, 2: 5}
    assert union == {1: 12, 2: 20}

    m = MaskMIoUMetric(mode="semantic")
    m._inter, m._union = inter, union
    merged_value = m.aggregate()

    # Pooled Jaccard: class1 = 10/12, class2 = 5/20; mean over classes.
    expected = ((10 / 12) + (5 / 20)) / 2
    assert merged_value == pytest.approx(expected)

    # Prove it is NOT the (wrong) mean-of-per-rank-ratios. Per-rank class means:
    # rank0 = (.5 + .1)/2 = .3 ; rank1 = (1.0 + .4)/2 = .7 ; mean = .5.
    wrong_mean_of_ratios = 0.5
    assert merged_value != pytest.approx(wrong_mean_of_ratios)


def test_box_f1_merge_counts_sums_tp_fp_fn():
    """BoxF1: sum (tp,fp,fn) across ranks; micro-F1 over the pooled counts."""
    rank0 = (3, 1, 2)  # tp, fp, fn
    rank1 = (5, 4, 1)

    tp, fp, fn = BoxF1Metric._merge_counts([rank0, rank1])
    assert (tp, fp, fn) == (8, 5, 3)

    m = BoxF1Metric()
    m._tp, m._fp, m._fn = tp, fp, fn
    merged_value = m.aggregate()

    precision = 8 / (8 + 5)
    recall = 8 / (8 + 3)
    expected = 2 * precision * recall / (precision + recall)
    assert merged_value == pytest.approx(expected)


def test_box_map_merge_detection_lists_equals_union():
    """BoxMAP: concatenating per-rank (preds,targets) lists == a single metric
    fed the union (list position auto-namespaces img_id)."""
    iou_thr = 0.5

    def box(c):  # axis-aligned 2x2x2 box at origin offset c
        return torch.tensor([[c, c, c, c + 2.0, c + 2.0, c + 2.0]])

    # Rank 0: 1 image, perfect TP.
    r0_preds = [{"boxes": box(0.0), "labels": torch.tensor([1]),
                 "scores": torch.tensor([0.9])}]
    r0_targets = [{"boxes": box(0.0), "labels": torch.tensor([1])}]
    # Rank 1: 1 image, a FP (disjoint box).
    r1_preds = [{"boxes": box(0.0), "labels": torch.tensor([1]),
                 "scores": torch.tensor([0.8])}]
    r1_targets = [{"boxes": box(10.0), "labels": torch.tensor([1])}]

    merged_preds, merged_targets = BoxMAPMetric._merge_detection_lists(
        [r0_preds, r1_preds], [r0_targets, r1_targets]
    )
    m = BoxMAPMetric(iou_thresholds=[iou_thr])
    m._preds, m._targets = merged_preds, merged_targets
    merged_value = m.aggregate()

    ref = BoxMAPMetric(iou_thresholds=[iou_thr])
    ref([r0_preds[0]], [r0_targets[0]])
    ref([r1_preds[0]], [r1_targets[0]])
    ref_value = ref.aggregate()

    assert merged_value == pytest.approx(ref_value, abs=1e-6)


def test_box_miou_merge_matched_ious_equals_union_mean():
    rank0 = [0.55, 0.8]
    rank1 = [0.9]
    merged = BoxMIoUMetric._merge_matched_ious([rank0, rank1])
    m = BoxMIoUMetric()
    m._matched_ious = merged
    assert merged == rank0 + rank1
    assert m.aggregate() == pytest.approx(sum(rank0 + rank1) / 3)


def test_class_ap_merge_score_target_lists_equals_union():
    """ClassAP: concat per-rank score/target tensor lists == union; binary AP
    over the pooled detections."""
    r0_scores = [torch.tensor([0.9, 0.1])]
    r0_targets = [torch.tensor([1, 0])]
    r1_scores = [torch.tensor([0.8, 0.2])]
    r1_targets = [torch.tensor([0, 1])]

    merged_s, merged_t = ClassAPMetric._merge_score_target_lists(
        [r0_scores, r1_scores], [r0_targets, r1_targets]
    )
    m = ClassAPMetric()
    m._scores, m._targets = merged_s, merged_t
    merged_value = m.aggregate()

    ref = ClassAPMetric()
    ref(torch.tensor([0.9, 0.1]), torch.tensor([1, 0]))
    ref(torch.tensor([0.8, 0.2]), torch.tensor([0, 1]))
    ref_value = ref.aggregate()

    assert merged_value == pytest.approx(ref_value, abs=1e-6)


# ===========================================================================
# 3. AP SCORE-THRESHOLD PARITY (driven through the evaluator)
# ===========================================================================


def _eval_target():
    """One image, two GT instances (classes 1 and 2) at distinct voxels."""
    label_map = torch.zeros(2, 2, 2, dtype=torch.long)
    label_map[0, 0, 0] = 7
    label_map[1, 1, 1] = 9
    return {
        "label_map": label_map,
        "mask_ids": torch.tensor([7, 9], dtype=torch.long),
        "labels": torch.tensor([1, 2], dtype=torch.long),
        "boxes": torch.zeros(2, 6, dtype=torch.float32),
    }


def _eval_sample_with_subthreshold_pred():
    """predict_for_eval output: two predictions, one HIGH score (0.9, class 1)
    and one BELOW the evaluator score_threshold (0.05, class 2). Each query's
    pixel-decoder activation is placed to exactly cover one GT voxel so masks
    are perfect (IoU 1.0)."""
    pixel_decoder_output = torch.full((2, 2, 2, 2), -5.0)
    pixel_decoder_output[0, 0, 0, 0] = 5.0  # query 0 -> voxel (0,0,0) == gt class 1
    pixel_decoder_output[1, 1, 1, 1] = 5.0  # query 1 -> voxel (1,1,1) == gt class 2
    return {
        "mask_source": "query",
        "mask_embeddings": torch.eye(2, dtype=torch.float32),
        "pixel_decoder_output": pixel_decoder_output,
        "topk_query_indices": torch.tensor([0, 1], dtype=torch.long),
        # pred for class 2 is BELOW the score_threshold we set (0.5).
        "topk_class_scores": torch.tensor([0.9, 0.05], dtype=torch.float32),
        "topk_class_ids": torch.tensor([1, 2], dtype=torch.long),
        "boxes": torch.zeros(2, 6, dtype=torch.float32),
        "eval_frame_size": (2, 2, 2),
    }


def test_evaluator_ap_sees_all_dets_but_miou_applies_score_threshold():
    """With score_threshold=0.5, the sub-threshold class-2 detection (score
    0.05) is PUSHED to MaskMAP (AP) but is EXCLUDED from the matched-IoU list
    consumed by instance mIoU."""
    evaluator = InstanceSegmentationEvaluator(
        metrics=["mask_map", {"name": "mask_miou", "mode": "instance"}],
        mask_chunk_size=1,
        match_labels=True,
        score_threshold=0.5,
    )
    evaluator.process(
        _make_sample_wrap(_eval_target()),
        outputs=[_eval_sample_with_subthreshold_pred()],
    )

    mask_map = evaluator.metrics["mask_map"]
    mask_miou = evaluator.metrics["mask_miou"]

    # --- AP side: BOTH classes' fragments are present in the stream, including
    # the sub-threshold class 2 (its score 0.05 < 0.5 was NOT filtered). ---
    stream_classes = sorted(e["class_id"] for e in mask_map._stream)
    assert stream_classes == [1, 2]
    class2_entry = next(e for e in mask_map._stream if e["class_id"] == 2)
    # The full unfiltered score survives into the AP fragment.
    assert class2_entry["scores"].tolist() == pytest.approx([0.05])
    assert class2_entry["n_gt"] == 1

    # --- mIoU side: only the above-threshold match (class 1) is recorded. ---
    # Class 1 (score 0.9 >= 0.5) is a perfect match (IoU 1.0); class 2 (0.05) is
    # dropped before greedy matching, so exactly one matched IoU of 1.0.
    assert mask_miou._matched_ious == pytest.approx([1.0])

    results = evaluator.evaluate()
    # mIoU is the mean over the single kept match -> 1.0.
    assert results["mask_miou"] == pytest.approx(1.0)
    # AP includes both classes' GT (both perfectly matched at their own scores,
    # one TP per class), so AP is 1.0 as well -- the sub-threshold detection is
    # a perfect-IoU TP for its class, NOT discarded.
    assert results["mask_map"] == pytest.approx(1.0)


def _make_sample_wrap(target):
    return {"metainfo": {"targets": [[target]]}}


def test_evaluator_subthreshold_only_pred_still_pushed_to_ap():
    """A class whose ONLY prediction is below score_threshold still contributes
    its detection (and GT denominator) to AP, while contributing NOTHING to the
    matched-IoU list."""
    target = _eval_target()
    sample = _eval_sample_with_subthreshold_pred()
    # Make BOTH predictions sub-threshold.
    sample["topk_class_scores"] = torch.tensor([0.04, 0.05], dtype=torch.float32)

    evaluator = InstanceSegmentationEvaluator(
        metrics=["mask_map", {"name": "mask_miou", "mode": "instance"}],
        mask_chunk_size=1,
        match_labels=True,
        score_threshold=0.5,
    )
    evaluator.process(_make_sample_wrap(target), outputs=[sample])

    mask_map = evaluator.metrics["mask_map"]
    mask_miou = evaluator.metrics["mask_miou"]

    # Both classes' detections were pushed to AP despite being sub-threshold.
    assert sorted(e["class_id"] for e in mask_map._stream) == [1, 2]
    for e in mask_map._stream:
        assert e["scores"].numel() == 1  # the sub-threshold pred is retained
        assert e["n_gt"] == 1

    # No matched IoUs survive the score filter -> mIoU has nothing.
    assert mask_miou._matched_ious == []

    results = evaluator.evaluate()
    assert results["mask_miou"] == pytest.approx(0.0)
    # AP still recovers the (perfect-IoU) TPs since AP ignores score_threshold.
    assert results["mask_map"] == pytest.approx(1.0)
