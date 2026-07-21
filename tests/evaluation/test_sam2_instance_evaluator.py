"""SAM2 (AMG) <-> InstanceSegmentationEvaluator integration + device hardening.

These tests exercise the SAM2 direct-binary-mask source path of
``InstanceSegmentationEvaluator._process_one`` (the ``pred_masks`` branch) WITHOUT
needing a GPU or real SAM2 weights. We build FAKE per-image sample dicts in the
exact shape ``SAM2.evaluate_step`` returns (see its docstring / the frozen
contract) and a matching synthetic GT target, then drive ``process()`` +
``evaluate()`` with a ``PredictedIoUEvalMetric`` (plus ``MaskMAP`` / ``MaskMIoU``)
registered.

Coverage:
  1. END-TO-END (also closes the WF1 coverage gap): pred_ious actually reach the
     metric, calibration MAE matches a hand computation, flattened keys appear.
  2. NO iou-pred key -> PredictedIoUEvalMetric skipped gracefully while MaskMAP /
     box metrics still populate.
  3. Per-class slice alignment with ``match_labels=True`` and >1 class.
  4. DEVICE HARDENING: pred_ious / masks / indices on (possibly mismatched)
     devices don't crash the coercion path.
  5. SAM2.evaluate_step shape/dtype/key contract -- run against the REAL
     method by faking only ``_predict_generate_masks`` (no GPU / real backbone),
     plus a CUDA-gated structural smoke if a model can be built.

All CPU-only except the explicitly CUDA-gated cases.
"""

import math

import numpy as np
import pytest
import torch

from cell_observatory_platform.evaluation.instance_segmentation_evaluator import (
    InstanceSegmentationEvaluator,
)


CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Fake SAM2 evaluate_step sample builders
# ---------------------------------------------------------------------------


def _sam2_sample(
    pred_masks: torch.Tensor,
    iou_preds,
    stability,
    boxes=None,
    class_ids=None,
    eval_frame_size=None,
    include_iou_preds: bool = True,
):
    """Build one per-image dict exactly as SAM2.evaluate_step returns.

    ``pred_masks`` is (N, Z, Y, X) bool; everything else is derived/overridable.
    """
    n = pred_masks.shape[0]
    zyx = tuple(pred_masks.shape[-3:])
    sample = {
        "topk_query_indices": torch.arange(n, dtype=torch.long),
        "topk_class_scores": torch.as_tensor(stability, dtype=torch.float32),
        "topk_class_ids": (
            torch.full((n,), -1, dtype=torch.long)
            if class_ids is None
            else torch.as_tensor(class_ids, dtype=torch.long)
        ),
        "boxes": (
            torch.zeros(n, 6, dtype=torch.float32) if boxes is None
            else torch.as_tensor(boxes, dtype=torch.float32)
        ),
        "eval_frame_size": eval_frame_size if eval_frame_size is not None else zyx,
        "pred_masks": pred_masks.bool(),
        "mask_source": "direct",
    }
    if include_iou_preds:
        sample["iou_preds"] = torch.as_tensor(iou_preds, dtype=torch.float32)
    return sample


def _data_sample(target, nested=True):
    targets = [[target]] if nested else [target]
    return {"metainfo": {"targets": targets}}


# ---------------------------------------------------------------------------
# 1. END-TO-END: SAM2-shaped masks -> evaluator -> PredictedIoUEvalMetric
# ---------------------------------------------------------------------------


def _classagnostic_target_single_gt():
    """One GT instance in a 4x4x4 labelmap; instance id 5, class label arbitrary.

    With match_labels=False the evaluator collapses to class -1, so the GT class
    label value is irrelevant to matching -- only the mask geometry matters.
    """
    label_map = torch.zeros(4, 4, 4, dtype=torch.long)
    label_map[0, :2, :2] = 5  # a 1x2x2 = 4-voxel instance
    return {
        "label_map": label_map,
        "mask_ids": torch.tensor([5], dtype=torch.long),
        "labels": torch.tensor([1], dtype=torch.long),
        "boxes": torch.zeros(1, 6, dtype=torch.float32),
    }


def test_end_to_end_sam2_pred_ious_reach_metric_and_keys_flatten():
    """Drive a SAM2-shaped sample through process()+evaluate() with a
    PredictedIoUEvalMetric + MaskMAP + MaskMIoU; assert flattened keys appear,
    the pred_ious reached the metric (calibration MAE matches a hand compute),
    and true_miou@0.5 reflects the synthetic geometry."""
    target = _classagnostic_target_single_gt()
    gt_mask = (target["label_map"] == 5)  # the true mask

    # 3 SAM2 proposals at 4x4x4 (orig size == labelmap dims -> exact IoU):
    #   prop 0: EXACT GT mask -> true IoU 1.0.  pred_iou 0.92 (slightly off).
    #   prop 1: HALF the GT (2 of 4 voxels) -> true IoU 0.5. pred_iou 0.40.
    #   prop 2: DISJOINT voxel -> true IoU 0.0 (FP). pred_iou 0.10.
    pred_masks = torch.zeros(3, 4, 4, 4, dtype=torch.bool)
    pred_masks[0] = gt_mask
    pred_masks[1, 0, 0, :2] = True            # half of the instance
    pred_masks[2, 3, 3, 3] = True             # disjoint
    iou_preds = [0.92, 0.40, 0.10]
    stability = [0.99, 0.80, 0.50]            # distinct from iou_preds

    sample = _sam2_sample(pred_masks, iou_preds, stability)

    evaluator = InstanceSegmentationEvaluator(
        metrics=[
            {"name": "predicted_iou", "key": "pred_iou",
             "iou_thresholds": [0.5], "pred_iou_thresholds": [0.5]},
            "mask_map",
            {"name": "mask_miou", "mode": "instance"},
        ],
        mask_chunk_size=2,
        match_labels=False,        # SAM2 is class-agnostic
        gt_mask_source="label_map",
    )

    evaluator.process(_data_sample(target), outputs=[sample])
    results = evaluator.evaluate()

    # (a) flattened keys present.
    assert "pred_iou/iou_head_mae" in results
    assert "pred_iou/map_rank_score" in results
    assert "pred_iou/true_miou@0.5" in results
    assert "mask_map" in results and math.isfinite(results["mask_map"])
    assert "mask_miou" in results and math.isfinite(results["mask_miou"])

    # (b) pred_ious actually reached the metric: true_iou = max-over-row =
    # [1.0, 0.5, 0.0]; pred_ious = [0.92, 0.40, 0.10]. MAE is the hand compute.
    true = np.array([1.0, 0.5, 0.0])
    pred = np.array([0.92, 0.40, 0.10])
    assert results["pred_iou/iou_head_mae"] == pytest.approx(
        np.abs(pred - true).mean(), abs=1e-6
    )

    # (c) selection@0.5: keep pred_iou >= 0.5 -> only prop 0 (true 1.0).
    assert results["pred_iou/true_miou@0.5"] == pytest.approx(1.0, abs=1e-6)
    assert results["pred_iou/coverage@0.5"] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_sam2_stream_pushed_to_pred_iou_metric_carries_true_iou_rows():
    """White-box: the PredictedIoUEvalMetric stream must contain the SAM2 entry
    with the (k x n_gt) TRUE IoU rows computed by the evaluator from pred_masks
    vs the labelmap GT, plus the pred_ious vector -- proving the masks really
    went through _pairwise_mask_iou_3d_bool (not a passthrough)."""
    target = _classagnostic_target_single_gt()
    gt_mask = (target["label_map"] == 5)

    pred_masks = torch.zeros(2, 4, 4, 4, dtype=torch.bool)
    pred_masks[0] = gt_mask                    # IoU 1.0
    pred_masks[1, 0, 0, :2] = True             # IoU 0.5
    sample = _sam2_sample(pred_masks, [0.9, 0.3], [0.95, 0.6])

    evaluator = InstanceSegmentationEvaluator(
        metrics=[
            {"name": "predicted_iou", "key": "pred_iou",
             "iou_thresholds": [0.5]},
        ],
        mask_chunk_size=1,
        match_labels=False,
        gt_mask_source="label_map",
    )
    evaluator.process(_data_sample(target), outputs=[sample])

    stream = evaluator.metrics["pred_iou"]._stream
    assert len(stream) == 1
    entry = stream[0]
    assert entry["class_id"] == -1
    assert entry["n_gt"] == 1
    # True IoU rows: (k=2, n_gt=1) == [[1.0], [0.5]].
    torch.testing.assert_close(
        entry["ious"], torch.tensor([[1.0], [0.5]]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        entry["pred_ious"], torch.tensor([0.9, 0.3]), atol=1e-6, rtol=0
    )


# ---------------------------------------------------------------------------
# 2. NO iou-pred key -> PredictedIoUEvalMetric skipped gracefully
# ---------------------------------------------------------------------------


def test_no_iou_pred_key_skips_predicted_iou_but_mask_and_box_populate():
    """A SAM2-shaped sample WITHOUT the iou_preds key must skip
    PredictedIoUEvalMetric gracefully (empty stream) while MaskMAP / BoxMAP
    still populate from the same pred_masks/boxes."""
    target = _classagnostic_target_single_gt()
    gt_mask = (target["label_map"] == 5)
    pred_masks = torch.zeros(2, 4, 4, 4, dtype=torch.bool)
    pred_masks[0] = gt_mask
    pred_masks[1, 0, 0, :2] = True
    # Non-trivial boxes so box metrics have something to chew on.
    boxes = torch.tensor(
        [[0, 0, 0, 1, 2, 2], [0, 0, 0, 1, 1, 2]], dtype=torch.float32
    )
    target["boxes"] = torch.tensor([[0, 0, 0, 1, 2, 2]], dtype=torch.float32)

    sample = _sam2_sample(
        pred_masks, iou_preds=None, stability=[0.9, 0.5],
        boxes=boxes, include_iou_preds=False,
    )
    assert "iou_preds" not in sample  # the load-bearing condition

    evaluator = InstanceSegmentationEvaluator(
        metrics=[
            {"name": "predicted_iou", "key": "pred_iou",
             "iou_thresholds": [0.5]},
            "mask_map",
            "box_map",
        ],
        mask_chunk_size=2,
        match_labels=False,
        gt_mask_source="label_map",
        gt_box_format="xyzxyz",
        gt_boxes_normalized=False,
    )
    evaluator.process(_data_sample(target), outputs=[sample])

    # PredictedIoUEvalMetric never got a push -> empty stream.
    assert evaluator.metrics["pred_iou"]._stream == []
    # MaskMAP did get the pred_masks-derived rows.
    assert len(evaluator.metrics["mask_map"]._stream) == 1

    results = evaluator.evaluate()
    # PredictedIoUEvalMetric still aggregates to its zero-dict (flattened).
    assert results["pred_iou/iou_head_mae"] == 0.0
    assert results["pred_iou/map_rank_score"] == 0.0
    # MaskMAP populated (prop 0 is an exact TP -> AP > 0).
    assert math.isfinite(results["mask_map"]) and results["mask_map"] > 0.0
    # BoxMAP must be > 0: prop 0's box exactly matches the GT box. This guards
    # the class-agnostic box-metric fix — SAM2 preds carry class -1 while GT
    # keeps real labels, so under match_labels=False the evaluator collapses
    # both label sets to a single sentinel before scoring. Without that, the
    # class-aware box AP would score 0 even for a perfect box.
    assert "box_map" in results and results["box_map"] > 0.0


# ---------------------------------------------------------------------------
# 3. Per-class slice alignment with match_labels=True and >1 class
# ---------------------------------------------------------------------------


def test_per_class_slice_alignment_match_labels_true_two_classes():
    """With match_labels=True and 2 classes, the evaluator must slice
    pred_masks / pred_ious / scores by class consistently. We give each class a
    perfectly-matching proposal so per-class true IoU == 1.0 in each bucket and
    the pred_ious stay aligned to the right class slice."""
    # Labelmap with two instances of two different classes.
    label_map = torch.zeros(4, 4, 4, dtype=torch.long)
    label_map[0, :2, :2] = 11      # instance 11 -> class 1
    label_map[3, 2:, 2:] = 22      # instance 22 -> class 2
    target = {
        "label_map": label_map,
        "mask_ids": torch.tensor([11, 22], dtype=torch.long),
        "labels": torch.tensor([1, 2], dtype=torch.long),
        "boxes": torch.zeros(2, 6, dtype=torch.float32),
    }
    gt1 = (label_map == 11)
    gt2 = (label_map == 22)

    # 2 proposals: prop0 matches class-1 instance, prop1 matches class-2.
    pred_masks = torch.zeros(2, 4, 4, 4, dtype=torch.bool)
    pred_masks[0] = gt1
    pred_masks[1] = gt2
    sample = _sam2_sample(
        pred_masks,
        iou_preds=[0.88, 0.77],
        stability=[0.9, 0.8],
        class_ids=[1, 2],           # honest per-proposal class ids
    )

    evaluator = InstanceSegmentationEvaluator(
        metrics=[
            {"name": "predicted_iou", "key": "pred_iou",
             "iou_thresholds": [0.5], "pred_iou_thresholds": [0.5]},
            "mask_map",
        ],
        mask_chunk_size=4,
        match_labels=True,          # honor class labels -> per-class buckets
        gt_mask_source="label_map",
    )
    evaluator.process(_data_sample(target), outputs=[sample])

    # Two class buckets in the stream (class 1 and class 2), each with a single
    # proposal whose true IoU row is [[1.0]] and the right pred_iou.
    stream = {e["class_id"]: e for e in evaluator.metrics["pred_iou"]._stream}
    assert set(stream.keys()) == {1, 2}
    torch.testing.assert_close(
        stream[1]["ious"], torch.tensor([[1.0]]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        stream[1]["pred_ious"], torch.tensor([0.88]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        stream[2]["ious"], torch.tensor([[1.0]]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        stream[2]["pred_ious"], torch.tensor([0.77]), atol=1e-6, rtol=0
    )

    results = evaluator.evaluate()
    # Both classes perfectly matched -> per-class AP 1.0, mean 1.0; calibration
    # MAE = mean(|0.88-1|, |0.77-1|).
    assert results["mask_map"] == pytest.approx(1.0, abs=1e-6)
    assert results["pred_iou/iou_head_mae"] == pytest.approx(
        (abs(0.88 - 1.0) + abs(0.77 - 1.0)) / 2, abs=1e-6
    )


# ---------------------------------------------------------------------------
# 4. DEVICE HARDENING
# ---------------------------------------------------------------------------


def test_device_hardening_all_cpu_does_not_crash():
    """The documented CPU-only trajectory: indices/masks/pred_ious all on CPU.
    The coercion-to-device path (device read from topk_query_indices) must be a
    no-op and not crash. Uses 2 proposals so PredictedIoUEvalMetric's
    calibration block (which requires n >= 2 pooled detections) activates and we
    can assert the hand-computed MAE actually flowed through the coercion."""
    target = _classagnostic_target_single_gt()
    gt_mask = (target["label_map"] == 5)
    # prop 0: exact GT -> true IoU 1.0; prop 1: half -> true IoU 0.5.
    pred_masks = torch.zeros(2, 4, 4, 4, dtype=torch.bool)
    pred_masks[0] = gt_mask
    pred_masks[1, 0, 0, :2] = True
    sample = _sam2_sample(pred_masks, [0.90, 0.40], [0.95, 0.6])
    # Everything is CPU.
    assert sample["topk_query_indices"].device.type == "cpu"
    assert sample["pred_masks"].device.type == "cpu"
    assert sample["iou_preds"].device.type == "cpu"

    evaluator = InstanceSegmentationEvaluator(
        metrics=[
            {"name": "predicted_iou", "key": "pred_iou",
             "iou_thresholds": [0.5]},
            "mask_map",
        ],
        mask_chunk_size=1,
        match_labels=False,
        gt_mask_source="label_map",
    )
    evaluator.process(_data_sample(target), outputs=[sample])
    results = evaluator.evaluate()
    # true_iou = [1.0, 0.5], pred = [0.90, 0.40] -> MAE = mean(0.10, 0.10) = 0.10.
    assert results["pred_iou/iou_head_mae"] == pytest.approx(0.1, abs=1e-6)


def test_device_hardening_masks_on_other_device_branch():
    """Mimic a producer that returns pred_masks on a DIFFERENT device than the
    indexing mask. The evaluator indexes on the masks' own device then moves the
    chunk to the index device. With CUDA we use cuda for masks; without CUDA we
    fall back to asserting the CPU path is robust (still meaningful coverage of
    the .to(src_device) / .to(device) coercion logic, which is a no-op on CPU
    but must not raise). Uses 2 proposals so the calibration block (n >= 2)
    activates."""
    target = _classagnostic_target_single_gt()
    gt_mask = (target["label_map"] == 5)
    pred_masks = torch.zeros(2, 4, 4, 4, dtype=torch.bool)
    pred_masks[0] = gt_mask                     # true IoU 1.0
    pred_masks[1, 0, 0, :2] = True              # true IoU 0.5

    sample = _sam2_sample(pred_masks, [0.85, 0.65], [0.9, 0.7])
    if CUDA:
        # Put masks + boxes + iou_preds on CUDA, indices/scores/class_ids on CPU.
        sample["pred_masks"] = sample["pred_masks"].cuda()
        sample["iou_preds"] = sample["iou_preds"].cuda()
        sample["boxes"] = sample["boxes"].cuda()
        # topk_query_indices stays CPU -> device = CPU; masks indexed on CUDA.

    evaluator = InstanceSegmentationEvaluator(
        metrics=[
            {"name": "predicted_iou", "key": "pred_iou",
             "iou_thresholds": [0.5]},
            "mask_map",
        ],
        mask_chunk_size=1,
        match_labels=False,
        gt_mask_source="label_map",
    )
    # Must not crash regardless of device placement.
    evaluator.process(_data_sample(target), outputs=[sample])
    results = evaluator.evaluate()
    # true_iou = [1.0, 0.5], pred = [0.85, 0.65] -> MAE = mean(0.15, 0.15) = 0.15.
    assert results["pred_iou/iou_head_mae"] == pytest.approx(0.15, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. SAM2.evaluate_step CONTRACT (faked _predict_generate_masks, no GPU)
# ---------------------------------------------------------------------------


def test_evaluate_step_dict_contract_against_real_method():
    """Run the REAL SAM2.evaluate_step by stubbing only the heavy backbone
    call ``_predict_generate_masks`` with a tiny MaskData. Asserts the per-image
    dict has EXACTLY the documented keys with documented shapes/dtypes, batch
    length 1, and that to_numpy() was NOT applied (tensors preserved)."""
    from cell_observatory_platform.models.meta_arch.sam import SAM2
    from cell_observatory_platform.inference.amg import MaskData

    z = y = x = 4
    n = 3
    md = MaskData(
        masks=(torch.rand(n, z, y, x) > 0.5),
        iou_preds=torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32),
        stability_score=torch.tensor([0.6, 0.7, 0.8], dtype=torch.float32),
        boxes=torch.zeros(n, 6, dtype=torch.float32),
    )

    class _Stub:
        # Reuse the unbound real method; supply only the attributes it touches.
        evaluate_step = SAM2.evaluate_step
        # evaluate_step converts the platform layout at the model boundary;
        # the stub borrows the method, so it needs the helper too.
        # staticmethod(): a bare function assigned to a class attribute would
        # re-bind as an instance method and receive self as the first arg.
        _to_model_layout = staticmethod(SAM2._to_model_layout)
        training = False
        iou_prediction_use_sigmoid = True

        def eval(self):
            return self

        def train(self):
            return self

        def _predict_generate_masks(self, vol):
            return md

    stub = _Stub()
    vol = torch.zeros(1, 1, z, y, x, 1)  # (B=1, T=1, Z, Y, X, C=1) platform layout
    data_sample = {"data_tensor": vol}

    with torch.no_grad():
        out = stub.evaluate_step(data_sample)

    assert isinstance(out, list) and len(out) == 1
    d = out[0]
    expected_keys = {
        "mask_source",
        "topk_query_indices", "topk_class_scores", "topk_class_ids",
        "boxes", "eval_frame_size", "pred_masks", "iou_preds",
    }
    assert set(d.keys()) == expected_keys
    # The model DECLARES its mask source; the evaluator no longer sniffs keys.
    assert d["mask_source"] == "direct"
    # Forbidden keys absent.
    assert "mask_embeddings" not in d
    assert "pixel_decoder_output" not in d
    assert "points" not in d
    assert "stability_score" not in d

    assert d["topk_query_indices"].shape == (n,)
    assert d["topk_query_indices"].dtype == torch.long
    torch.testing.assert_close(d["topk_query_indices"], torch.arange(n))

    assert d["topk_class_scores"].shape == (n,)
    assert d["topk_class_scores"].dtype == torch.float32
    # topk_class_scores carries stability_score (NOT iou_preds) -> distinct.
    torch.testing.assert_close(
        d["topk_class_scores"], md["stability_score"]
    )
    assert not torch.equal(d["topk_class_scores"], d["iou_preds"])

    assert d["topk_class_ids"].shape == (n,)
    assert d["topk_class_ids"].dtype == torch.long
    assert torch.all(d["topk_class_ids"] == -1)

    assert d["boxes"].shape == (n, 6)
    assert d["boxes"].dtype == torch.float32

    assert d["pred_masks"].shape == (n, z, y, x)
    assert d["pred_masks"].dtype == torch.bool
    assert d["pred_masks"].device.type == "cpu"

    assert d["iou_preds"].shape == (n,)
    assert d["iou_preds"].dtype == torch.float32
    assert isinstance(d["eval_frame_size"], tuple)
    assert d["eval_frame_size"] == (z, y, x)


def test_evaluate_step_empty_case_shapes():
    """N == 0 must yield correctly-typed zero-leading-dim tensors."""
    from cell_observatory_platform.models.meta_arch.sam import SAM2
    from cell_observatory_platform.inference.amg import MaskData

    z = y = x = 4
    md = MaskData()  # empty

    class _Stub:
        evaluate_step = SAM2.evaluate_step
        # evaluate_step converts the platform layout at the model boundary;
        # the stub borrows the method, so it needs the helper too.
        # staticmethod(): a bare function assigned to a class attribute would
        # re-bind as an instance method and receive self as the first arg.
        _to_model_layout = staticmethod(SAM2._to_model_layout)
        training = False
        iou_prediction_use_sigmoid = True

        def eval(self):
            return self

        def train(self):
            return self

        def _predict_generate_masks(self, vol):
            return md

    stub = _Stub()
    data_sample = {"data_tensor": torch.zeros(1, 1, z, y, x, 1)}  # B=1, T=1
    with torch.no_grad():
        out = stub.evaluate_step(data_sample)
    d = out[0]
    assert d["pred_masks"].shape == (0, z, y, x)
    assert d["pred_masks"].dtype == torch.bool
    assert d["boxes"].shape == (0, 6)
    assert d["iou_preds"].shape == (0,)
    assert d["topk_query_indices"].shape == (0,)
    assert d["topk_class_ids"].shape == (0,)
    assert d["topk_class_scores"].shape == (0,)


def test_evaluate_step_asserts_batch_size_one():
    from cell_observatory_platform.models.meta_arch.sam import SAM2
    from cell_observatory_platform.inference.amg import MaskData

    class _Stub:
        evaluate_step = SAM2.evaluate_step
        # evaluate_step converts the platform layout at the model boundary;
        # the stub borrows the method, so it needs the helper too.
        # staticmethod(): a bare function assigned to a class attribute would
        # re-bind as an instance method and receive self as the first arg.
        _to_model_layout = staticmethod(SAM2._to_model_layout)
        training = False
        iou_prediction_use_sigmoid = True

        def eval(self):
            return self

        def train(self):
            return self

        def _predict_generate_masks(self, vol):
            return MaskData()

    stub = _Stub()
    # B*T == 2 must trip the assert.
    data_sample = {"data_tensor": torch.zeros(2, 1, 4, 4, 4, 1)}
    with pytest.raises(AssertionError, match="single volume"):
        stub.evaluate_step(data_sample)


@pytest.mark.skipif(
    not CUDA,
    reason="Full SAM2.evaluate_step smoke needs CUDA + a real backbone/weights; "
           "skipped on CPU-only. The dict contract is covered by the stubbed test.",
)
def test_evaluate_step_real_backbone_smoke():
    # Intentionally a structural placeholder: building a real SAM2 (image
    # encoder + mask decoder + prompt encoder + criterion) requires a Hydra
    # config and weights not available in the unit-test environment. The
    # stubbed contract test above fully validates the dict shape/dtype/keys
    # without GPU. If a fixture for a tiny real backbone is added later, wire it
    # here.
    pytest.skip(
        "No tiny real-backbone SAM2 fixture available; dict contract covered by "
        "test_evaluate_step_dict_contract_against_real_method."
    )
