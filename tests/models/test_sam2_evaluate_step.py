"""`SAM2.evaluate_step` dict contract, run against the real method by stubbing only
the AMG call (`_predict_generate_masks`); no GPU, no weights."""

import pytest
import torch

from cell_observatory_platform.inference.amg import MaskData
from cell_observatory_platform.models.meta_arch.sam import SAM2


def _stub_sam2(mask_data: MaskData):
    """A SAM2 stand-in that borrows the real ``evaluate_step`` and stubs only the AMG call."""
    class _Stub:
        evaluate_step = SAM2.evaluate_step
        # staticmethod: a bare function on the class would re-bind and receive self.
        _to_model_layout = staticmethod(SAM2._to_model_layout)
        training = False
        iou_prediction_use_sigmoid = True

        def eval(self):
            return self

        def train(self):
            return self

        def _predict_generate_masks(self, vol):
            return mask_data

    return _Stub()


def test_evaluate_step_dict_contract():
    """The per-image dict has EXACTLY the documented keys with the documented
    shapes/dtypes, batch length 1, tensors preserved (no to_numpy()), and
    `topk_class_scores` carries stability_score rather than iou_preds."""
    z = y = x = 4
    n = 3
    md = MaskData(
        masks=(torch.rand(n, z, y, x) > 0.5),
        iou_preds=torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32),
        stability_score=torch.tensor([0.6, 0.7, 0.8], dtype=torch.float32),
        boxes=torch.zeros(n, 6, dtype=torch.float32),
    )
    stub = _stub_sam2(md)
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
    # The model DECLARES its mask source; the evaluator does not sniff keys.
    assert d["mask_source"] == "direct"

    assert d["topk_query_indices"].shape == (n,)
    assert d["topk_query_indices"].dtype == torch.long
    torch.testing.assert_close(d["topk_query_indices"], torch.arange(n))

    assert d["topk_class_scores"].shape == (n,)
    assert d["topk_class_scores"].dtype == torch.float32
    # topk_class_scores carries stability_score (NOT iou_preds) -> distinct.
    torch.testing.assert_close(d["topk_class_scores"], md["stability_score"])
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
    torch.testing.assert_close(d["iou_preds"], md["iou_preds"])
    assert isinstance(d["eval_frame_size"], tuple)
    assert d["eval_frame_size"] == (z, y, x)


def test_evaluate_step_empty_mask_data_yields_zero_length_tensors():
    """N == 0 must yield correctly-typed zero-leading-dim tensors."""
    z = y = x = 4
    stub = _stub_sam2(MaskData())  # empty
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
    assert d["eval_frame_size"] == (z, y, x)


def test_evaluate_step_rejects_more_than_one_volume():
    """AMG encodes one volume at a time: B*T > 1 is rejected up front."""
    stub = _stub_sam2(MaskData())
    data_sample = {"data_tensor": torch.zeros(2, 1, 4, 4, 4, 1)}  # B*T == 2
    with pytest.raises(AssertionError, match="single volume"):
        stub.evaluate_step(data_sample)
