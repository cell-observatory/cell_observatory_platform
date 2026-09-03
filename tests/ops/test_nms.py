import pytest
import torch

pytestmark = pytest.mark.cuda
_C = pytest.importorskip("ops3d._C", reason="ops3d extension not built")
if not torch.cuda.is_available():
    pytest.skip("CUDA device not available", allow_module_level=True)


def test_nms3d_suppresses_high_iou_duplicates():
    # one high-score box duplicated 1000x
    # all IoUs = 1 so only the top-score remains
    iou_thresh = 0.1
    box0 = torch.tensor([0, 0, 0, 100, 100, 100], device="cuda", dtype=torch.float32)
    boxes = box0.unsqueeze(0).repeat(1000, 1)  # (1000, 6)
    scores = torch.cat(
        [
            torch.tensor([0.9], device="cuda"),
            torch.zeros(999, device="cuda"),
        ]
    )
    keep = _C.nms_3d(boxes, scores, iou_thresh)

    # expect exactly one index: [0]
    kept = keep.cpu().tolist()
    assert kept == [0]


def test_nms3d_keeps_non_overlapping_boxes():
    # non-overlapping boxes should both be kept
    iou_thresh = 0.5
    box_a = torch.tensor([0, 0, 0, 10, 10, 10], device="cuda", dtype=torch.float32)
    box_b = torch.tensor([20, 20, 20, 30, 30, 30], device="cuda", dtype=torch.float32)
    boxes = torch.stack([box_a, box_b], dim=0)
    scores = torch.tensor([0.6, 0.7], device="cuda")
    keep = _C.nms_3d(boxes, scores, iou_thresh)

    kept = set(keep.cpu().tolist())
    # both indices 0 and 1 should be present
    assert kept == {0, 1}


def test_nms_nd_splits_trailing_score_column():
    """nms_nd takes dets = [boxes | score], suppresses the lower-scored duplicate,
    keeps the disjoint box, and hands the dets tensor back unchanged."""
    from cell_observatory_platform.models.ops.nms_nd import nms_nd

    dets = torch.tensor([
        [0, 0, 0, 10, 10, 10, 0.6],      # duplicate of row 1, lower score -> suppressed
        [0, 0, 0, 10, 10, 10, 0.9],
        [20, 20, 20, 30, 30, 30, 0.7],   # disjoint -> kept
    ], device="cuda", dtype=torch.float32)
    keep, dets_out = nms_nd(dets, iou_threshold=0.5)
    assert sorted(keep.cpu().tolist()) == [1, 2]
    assert dets_out is dets
