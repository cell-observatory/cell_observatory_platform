"""Unit tests for data.structures box operations."""

import pytest
import torch

from cell_observatory_platform.data.structures import (
    generalized_box_iou,
    generalized_box_iou_diag,
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
def test_generalized_box_iou_diag_different_lengths(n, m):
    """generalized_box_iou_diag with N!=M matches torch.diag of full matrix (min(N,M) diagonal)."""
    boxes1 = _make_valid_boxes(n, dtype=torch.float32, seed=42)
    boxes2 = _make_valid_boxes(m, dtype=torch.float32, seed=123)

    diag_result = generalized_box_iou_diag(boxes1, boxes2)
    full_result = generalized_box_iou(boxes1, boxes2)
    expected = torch.diag(full_result)

    assert diag_result.shape == (min(n, m),)
    assert torch.allclose(diag_result, expected, rtol=1e-5, atol=1e-6), (
        f"n={n}, m={m}: max diff = {(diag_result - expected).abs().max().item()}"
    )


def test_generalized_box_iou_diag_non_overlapping():
    """Non-overlapping boxes yield negative GIoU."""
    # boxes1: [0,0,0] to [1,1,1], boxes2: [2,2,2] to [3,3,3]
    boxes1 = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    boxes2 = torch.tensor([[2.0, 2.0, 2.0, 3.0, 3.0, 3.0]])
    result = generalized_box_iou_diag(boxes1, boxes2)
    expected = torch.diag(generalized_box_iou(boxes1, boxes2))
    assert torch.allclose(result, expected, rtol=1e-5, atol=1e-6), f"max diff = {(result - expected).abs().max().item()}"
    assert result.item() < 0, f"result = {result.item()}"
