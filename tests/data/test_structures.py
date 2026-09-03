"""Unit tests for data.structures box operations."""

import math

import pytest
import torch

from cell_observatory_platform.data.structures import (
    bbox2delta,
    convert_bbox_format,
    delta2bbox,
    generalized_box_iou,
    generalized_box_iou_diag,
    masks_to_boxes_v2,
    validate_bbox_normalization,
)


def _make_valid_boxes(n: int, device=None, dtype=torch.float32, seed=None) -> torch.Tensor:
    """Create boxes in (x1, y1, z1, x2, y2, z2) format with 0 <= x1 < x2 etc."""
    if seed is not None:
        torch.manual_seed(seed)
    # Random corners in [0, 1), ensure x1 < x2
    coords = torch.rand(n, 6, device=device, dtype=dtype)
    x1, y1, z1, x2, y2, z2 = coords.unbind(-1)
    x1, x2 = torch.minimum(x1, x2), torch.maximum(x1, x2)
    y1, y2 = torch.minimum(y1, y2), torch.maximum(y1, y2)
    z1, z2 = torch.minimum(z1, z2), torch.maximum(z1, z2)
    # Avoid degenerate boxes
    x2 = torch.where(x2 > x1 + 1e-4, x2, x1 + 0.1)
    y2 = torch.where(y2 > y1 + 1e-4, y2, y1 + 0.1)
    z2 = torch.where(z2 > z1 + 1e-4, z2, z1 + 0.1)
    return torch.stack([x1, y1, z1, x2, y2, z2], dim=-1)


@pytest.mark.parametrize("n", [1, 5, 23, 100])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_generalized_box_iou_diag_matches_diag_of_full(n, dtype):
    """generalized_box_iou_diag output equals torch.diag(generalized_box_iou) for matched pairs."""
    boxes1 = _make_valid_boxes(n, dtype=dtype, seed=42)
    boxes2 = _make_valid_boxes(n, dtype=dtype, seed=123)

    diag_result = generalized_box_iou_diag(boxes1, boxes2)
    full_result = generalized_box_iou(boxes1, boxes2)
    expected = torch.diag(full_result)

    assert diag_result.shape == (n,)
    assert torch.allclose(diag_result, expected, rtol=1e-5, atol=1e-6), (
        f"n={n}: max diff = {(diag_result - expected).abs().max().item()}"
    )


def test_generalized_box_iou_diag_empty():
    """generalized_box_iou_diag handles empty input."""
    boxes = torch.zeros(0, 6, dtype=torch.float32)
    result = generalized_box_iou_diag(boxes, boxes)
    assert result.shape == (0,)


def test_generalized_box_iou_diag_identical_boxes():
    """Identical boxes yield GIoU = 1.0."""
    boxes = _make_valid_boxes(7, dtype=torch.float32, seed=99)
    result = generalized_box_iou_diag(boxes, boxes)
    assert result.shape == (7,)
    expected = boxes.new_ones(7)
    assert torch.allclose(result, expected, rtol=1e-5, atol=5e-5), f"max diff = {(result - expected).abs().max().item()}"


@pytest.mark.parametrize("n,m", [(10, 5), (5, 10), (3, 7)])
def test_generalized_box_iou_diag_raises_on_mismatched_lengths(n, m):
    """generalized_box_iou_diag raises ValueError when boxes1 and boxes2 have different lengths."""
    boxes1 = _make_valid_boxes(n, dtype=torch.float32, seed=42)
    boxes2 = _make_valid_boxes(m, dtype=torch.float32, seed=123)

    with pytest.raises(ValueError, match=f"got {n} and {m}"):
        generalized_box_iou_diag(boxes1, boxes2)


def test_generalized_box_iou_diag_non_overlapping():
    """Non-overlapping boxes yield negative GIoU."""
    # boxes1: [0,0,0] to [1,1,1], boxes2: [2,2,2] to [3,3,3]
    boxes1 = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    boxes2 = torch.tensor([[2.0, 2.0, 2.0, 3.0, 3.0, 3.0]])
    result = generalized_box_iou_diag(boxes1, boxes2)
    expected = torch.diag(generalized_box_iou(boxes1, boxes2))
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6), f"max diff = {(result - expected).abs().max().item()}"
    assert result.item() < 0, f"result = {result.item()}"


# ---------------------------------------------------------------------------
# Box deltas (cx cy cz w h d proposals <-> corner boxes)
# ---------------------------------------------------------------------------


def test_bbox2delta_delta2bbox_round_trip():
    """bbox2delta encodes (centre offset / proposal size, log size ratio) and
    delta2bbox decodes those deltas back to the ground-truth corner boxes."""
    proposals = torch.tensor([[1.0, 2.0, 3.0, 4.0, 4.0, 2.0], [0.5, 0.5, 0.5, 1.0, 1.0, 1.0]])  # cx cy cz w h d
    gt = torch.tensor([[2.0, 2.5, 2.0, 6.0, 2.0, 3.0], [0.6, 0.4, 0.5, 2.0, 0.5, 1.0]])
    deltas = bbox2delta(proposals, gt)
    assert deltas[0, 0].item() == pytest.approx((2.0 - 1.0) / 4.0)       # dx = (gx - px) / pw
    assert deltas[0, 3].item() == pytest.approx(math.log(6.0 / 4.0))      # dw = log(gw / pw)
    gt_corners = torch.cat([gt[:, :3] - gt[:, 3:] / 2, gt[:, :3] + gt[:, 3:] / 2], dim=-1)
    torch.testing.assert_close(delta2bbox(proposals, deltas), gt_corners, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# masks -> boxes
# ---------------------------------------------------------------------------


def test_masks_to_boxes_v2_returns_half_open_xyz_boxes():
    """Boxes are half-open xyzxyz (max = last foreground index + 1); an empty mask
    yields an all-zero row and an empty stack yields (0, 6)."""
    m = torch.zeros(3, 4, 5, 6, dtype=torch.bool)                # (N, D, H, W)
    m[0, 1:3, 2:4, 0:5] = True                                    # z 1-2, y 2-3, x 0-4
    m[1, 3, 4, 5] = True                                          # single voxel; mask 2 empty
    boxes = masks_to_boxes_v2(m)
    assert boxes.tolist() == [[0, 2, 1, 5, 4, 3], [5, 4, 3, 6, 5, 4], [0, 0, 0, 0, 0, 0]]
    assert masks_to_boxes_v2(torch.zeros(0, 2, 2, 2)).shape == (0, 6)


# ---------------------------------------------------------------------------
# Format conversion and the normalization guard
# ---------------------------------------------------------------------------


def test_convert_bbox_format_zyxzyx_to_normalized_cxcyczwhd():
    """The collator's conversion path: zyxzyx corners -> xyz centre/size divided
    by (W, H, D); the inverse conversion restores the corners; an unsupported
    pair raises."""
    boxes = torch.tensor([[1.0, 2.0, 3.0, 3.0, 6.0, 7.0]])                       # z1 y1 x1 z2 y2 x2
    out = convert_bbox_format(boxes, "zyxzyx", "cxcyczwhd", normalize=True, spatial_size=(8, 8, 4))  # (W, H, D)
    torch.testing.assert_close(out, torch.tensor([[5 / 8, 4 / 8, 2 / 4, 4 / 8, 4 / 8, 2 / 4]]))
    back = convert_bbox_format(out * torch.tensor([8.0, 8, 4, 8, 8, 4]), "cxcyczwhd", "zyxzyx")
    torch.testing.assert_close(back, boxes)
    with pytest.raises(ValueError, match="Unsupported bbox format conversion"):
        convert_bbox_format(boxes, "zyxzyx", "xyzxyz")


def test_validate_bbox_normalization_rejects_normalized_zyxzyx():
    """normalize_bboxes only acts inside the cxcyczwhd conversion; with a zyxzyx
    output it would be a silent no-op, so the combination is rejected."""
    with pytest.raises(ValueError, match="normalize_bboxes"):
        validate_bbox_normalization(True, "zyxzyx")


def test_validate_bbox_normalization_accepts_supported_combinations():
    validate_bbox_normalization(False, "zyxzyx")
    validate_bbox_normalization(True, "cxcyczwhd")
