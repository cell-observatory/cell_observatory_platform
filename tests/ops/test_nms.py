import pytest
import torch

try:
    import ops3d._C as _C
except ImportError:
    print("3D NMS op failed to load. Please compile ops3d if needed.")
    pytestmark = pytest.mark.skip(reason="This module is temporarily disabled till we add ops3d to the docker image")


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for ops3d._C.nms_3d")


def test_nms3d_suppresses_high_iou_duplicates():
    # one high‐score box duplicated 1000x
    # all IoUs = 1 so only the top‐score remains
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
    assert len(kept) == 1


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
    assert len(kept) == 2
