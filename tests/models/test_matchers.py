import pytest
import torch

from cell_observatory_platform.models.layers.matchers import HungarianMatcher


def _random_boxes(num_boxes: int, device: torch.device) -> torch.Tensor:
    """
    Generate reasonable (cx, cy, cz, w, h, d) boxes in [0, 1].
    """
    centers = torch.rand(num_boxes, 3, device=device) * 0.8 + 0.1  # [0.1, 0.9]
    sizes = torch.rand(num_boxes, 3, device=device) * 0.3 + 0.05  # [0.05, 0.35]
    return torch.cat([centers, sizes], dim=-1)  # (N, 6)


@pytest.mark.parametrize(
    "costs,num_queries,num_classes,side,num_points,num_targets_per_image",
    [
        (["cls"], 6, 4, 8, 5, [3, 5]),
        (["cls", "mask"], 5, 3, 10, 16, [4, 2]),
        (["cls", "box", "mask"], 7, 5, 12, 20, [3, 6]),
    ],
    ids=["cls", "cls_mask", "cls_box_mask"],
)
def test_hungarian_matcher_match_structure(costs, num_queries, num_classes, side, num_points, num_targets_per_image):
    """Every cost combination yields one 1-D (pred, tgt) index pair per image of size min(Q, n_tgt), in range."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    batch_size = len(num_targets_per_image)
    D = H = W = side

    pred_logits = torch.randn(batch_size, num_queries, num_classes, device=device)
    pred_masks = torch.randn(batch_size, num_queries, D, H, W, device=device)
    pred_boxes = _random_boxes(num_queries * batch_size, device).view(batch_size, num_queries, 6)

    targets = []
    for n_tgt in num_targets_per_image:
        labels = torch.randint(0, num_classes, (n_tgt,), device=device)
        boxes = _random_boxes(n_tgt, device)
        mask_ids = torch.arange(1, n_tgt + 1, device=device)
        label_map = torch.zeros(D, H, W, dtype=torch.long, device=device)
        targets.append({"labels": labels, "boxes": boxes, "mask_ids": mask_ids, "label_map": label_map})

    outputs = {"pred_logits": pred_logits, "pred_masks": pred_masks, "pred_boxes": pred_boxes}

    use_mask = 1.0 if "mask" in costs else 0.0
    use_box = 1.0 if "box" in costs else 0.0
    matcher = HungarianMatcher(
        cost_classification=1.0,
        cost_mask=use_mask,
        cost_mask_dice=use_mask,
        cost_box=use_box,
        cost_box_giou=use_box,
        num_points=num_points,
    )

    matches = matcher(outputs, targets, costs=costs)

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
        # an assignment never reuses a query or a target
        assert idx_pred.unique().numel() == idx_pred.numel()
        assert idx_tgt.unique().numel() == idx_tgt.numel()


def test_hungarian_matcher_recovers_permuted_identity():
    """One-hot logits, exact boxes and slab masks -> the min-cost assignment is the known permutation."""
    torch.manual_seed(0)
    D, H, W, Q = 6, 4, 4, 3
    labels = torch.tensor([2, 0, 1])                 # target t has class labels[t]
    expect_tgt_for_query = torch.tensor([1, 2, 0])   # query q predicts class q -> target with that label
    boxes = torch.tensor([[0.2, 0.2, 0.2, 0.1, 0.1, 0.1],
                          [0.5, 0.5, 0.5, 0.1, 0.1, 0.1],
                          [0.8, 0.8, 0.8, 0.1, 0.1, 0.1]])
    label_map = torch.zeros(D, H, W, dtype=torch.int32)
    for t in range(3):                                # target t owns z-slab [2t, 2t+2)
        label_map[2 * t : 2 * t + 2] = t + 1
    mask_ids = torch.tensor([1, 2, 3])

    pred_logits = torch.full((1, Q, 3), -10.0)
    pred_boxes = torch.zeros(1, Q, 6)
    pred_masks = torch.full((1, Q, D, H, W), -8.0)
    for q in range(Q):
        t = expect_tgt_for_query[q]
        pred_logits[0, q, q] = 10.0
        pred_boxes[0, q] = boxes[t]
        pred_masks[0, q][label_map == t + 1] = 8.0

    matcher = HungarianMatcher(cost_classification=1.0, cost_mask=1.0, cost_mask_dice=1.0,
                               cost_box=1.0, cost_box_giou=1.0, num_points=32)
    outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes, "pred_masks": pred_masks}
    targets = [{"labels": labels, "boxes": boxes, "mask_ids": mask_ids, "label_map": label_map}]

    (idx_pred, idx_tgt), = matcher(outputs, targets, costs=["cls", "box", "mask"])
    order = torch.argsort(idx_pred)
    assert torch.equal(idx_pred[order], torch.arange(Q))
    assert torch.equal(idx_tgt[order], expect_tgt_for_query)
