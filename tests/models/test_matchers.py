import pytest
import torch

from cell_observatory_platform.models.layers.matchers import HungarianMatcher

CUDA_AVAILABLE = torch.cuda.is_available()


def _random_boxes(num_boxes: int, device: torch.device) -> torch.Tensor:
    """
    Generate reasonable (cx, cy, cz, w, h, d) boxes in [0, 1].
    """
    centers = torch.rand(num_boxes, 3, device=device) * 0.8 + 0.1  # [0.1, 0.9]
    sizes = torch.rand(num_boxes, 3, device=device) * 0.3 + 0.05  # [0.05, 0.35]
    return torch.cat([centers, sizes], dim=-1)  # (N, 6)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_hungarian_matcher_cls_only():
    device = torch.device("cuda")

    batch_size = 2
    num_queries = 6
    num_classes = 4

    # predictions
    pred_logits = torch.randn(batch_size, num_queries, num_classes, device=device)
    pred_masks = torch.randn(batch_size, num_queries, 8, 8, 8, device=device)
    pred_boxes = _random_boxes(num_queries * batch_size, device).view(batch_size, num_queries, 6)

    # targets: variable number of GT per image
    targets = []
    num_targets_per_image = [3, 5]
    for n_tgt in num_targets_per_image:
        labels = torch.randint(0, num_classes, (n_tgt,), device=device)
        masks = torch.randn(n_tgt, 8, 8, 8, device=device)
        boxes = _random_boxes(n_tgt, device)
        targets.append({"labels": labels, "masks": masks, "boxes": boxes})

    outputs = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks,
        "pred_boxes": pred_boxes,
    }

    matcher = HungarianMatcher(
        cost_classification=1.0,
        cost_mask=0.0,
        cost_mask_dice=0.0,
        cost_box=0.0,
        cost_box_giou=0.0,
        num_points=5,
    )

    matches = matcher(outputs, targets, costs=["cls"])

    assert len(matches) == batch_size
    for (idx_pred, idx_tgt), n_tgt in zip(matches, num_targets_per_image):
        # 1D index tensors
        assert idx_pred.dim() == 1
        assert idx_tgt.dim() == 1
        # same number of matches on both sides
        assert idx_pred.numel() == idx_tgt.numel()
        # Hungarian gives min(num_queries, num_targets) matches
        assert idx_pred.numel() == min(num_queries, n_tgt)
        # indices in valid range
        assert torch.all((idx_pred >= 0) & (idx_pred < num_queries))
        assert torch.all((idx_tgt >= 0) & (idx_tgt < n_tgt))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_hungarian_matcher_with_mask_cost():
    device = torch.device("cuda")

    batch_size = 2
    num_queries = 5
    num_classes = 3
    D = H = W = 10
    num_points = 16

    pred_logits = torch.randn(batch_size, num_queries, num_classes, device=device)
    pred_masks = torch.randn(batch_size, num_queries, D, H, W, device=device)
    pred_boxes = _random_boxes(num_queries * batch_size, device).view(batch_size, num_queries, 6)

    targets = []
    num_targets_per_image = [4, 2]
    for n_tgt in num_targets_per_image:
        labels = torch.randint(0, num_classes, (n_tgt,), device=device)
        masks = torch.randn(n_tgt, D, H, W, device=device)
        boxes = _random_boxes(n_tgt, device)
        targets.append({"labels": labels, "masks": masks, "boxes": boxes})

    outputs = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks,
        "pred_boxes": pred_boxes,
    }

    matcher = HungarianMatcher(
        cost_classification=1.0,
        cost_mask=1.0,
        cost_mask_dice=1.0,
        cost_box=0.0,
        cost_box_giou=0.0,
        num_points=num_points,
    )

    matches = matcher(outputs, targets, costs=["cls", "mask"])

    assert len(matches) == batch_size
    for (idx_pred, idx_tgt), n_tgt in zip(matches, num_targets_per_image):
        assert idx_pred.dim() == 1
        assert idx_tgt.dim() == 1
        assert idx_pred.numel() == idx_tgt.numel()
        assert idx_pred.numel() == min(num_queries, n_tgt)
        assert torch.all((idx_pred >= 0) & (idx_pred < num_queries))
        assert torch.all((idx_tgt >= 0) & (idx_tgt < n_tgt))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_hungarian_matcher_full_costs():
    device = torch.device("cuda")

    batch_size = 2
    num_queries = 7
    num_classes = 5
    D = H = W = 12
    num_points = 20

    pred_logits = torch.randn(batch_size, num_queries, num_classes, device=device)
    pred_masks = torch.randn(batch_size, num_queries, D, H, W, device=device)
    pred_boxes = _random_boxes(num_queries * batch_size, device).view(batch_size, num_queries, 6)

    targets = []
    num_targets_per_image = [3, 6]
    for n_tgt in num_targets_per_image:
        labels = torch.randint(0, num_classes, (n_tgt,), device=device)
        masks = torch.randn(n_tgt, D, H, W, device=device)
        boxes = _random_boxes(n_tgt, device)
        targets.append({"labels": labels, "masks": masks, "boxes": boxes})

    outputs = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks,
        "pred_boxes": pred_boxes,
    }

    matcher = HungarianMatcher(
        cost_classification=1.0,
        cost_mask=1.0,
        cost_mask_dice=1.0,
        cost_box=1.0,
        cost_box_giou=1.0,
        num_points=num_points,
    )

    # exercise all branches: cls + box + mask
    matches = matcher(outputs, targets, costs=["cls", "box", "mask"])

    assert len(matches) == batch_size
    for (idx_pred, idx_tgt), n_tgt in zip(matches, num_targets_per_image):
        assert idx_pred.dim() == 1
        assert idx_tgt.dim() == 1
        assert idx_pred.numel() == idx_tgt.numel()
        assert idx_pred.numel() == min(num_queries, n_tgt)
        assert torch.all((idx_pred >= 0) & (idx_pred < num_queries))
        assert torch.all((idx_tgt >= 0) & (idx_tgt < n_tgt))
