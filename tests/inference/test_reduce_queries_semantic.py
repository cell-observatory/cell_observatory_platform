"""Tests for query-to-semantic reduction (Mask2Former inference)."""

import pytest
import torch

from cell_observatory_platform.inference.utils import reduce_queries_to_semantic_map


def test_multiclass_disjoint_queries_produce_distinct_class_maps():
    """Same query must not dominate two classes; assigned-class aggregation separates channels."""
    B, Q, D, H, W = 1, 2, 2, 2, 2
    num_classes = 2
    # Query 0 -> class 0 only; query 1 -> class 1 only (argmax on logits including no-object)
    # Logits: [c0, c1, no_object] — use large margins so argmax is unambiguous
    pred_logits = torch.tensor(
        [
            [
                [5.0, -5.0, -5.0],  # q0: class 0
                [-5.0, 5.0, -5.0],  # q1: class 1
            ]
        ],
        dtype=torch.float32,
    )  # [1, 2, 3]

    # Distinct spatial masks. Use strongly negative logits for "off" voxels so sigmoid ≈ 0;
    # raw 0 logits would give sigmoid(0)=0.5 and falsely activate the whole volume.
    z = torch.full((B, Q, D, H, W), -10.0)
    z[0, 0, :, :, :] = 10.0  # q0: all foreground
    z[0, 1, 0, 0, 0] = 10.0  # q1: single voxel (D,H,W index)
    pred_masks = z

    sem, avg = reduce_queries_to_semantic_map(
        pred_masks, pred_logits, num_classes=num_classes, topk_per_image=2,
        reduction="topk_max",
    )
    assert sem.shape == (B, D, H, W, num_classes)
    # Channel 0 should be bright everywhere (query 0); channel 1 only at one corner
    assert sem[0, :, :, :, 0].min() > 0.5
    assert sem[0, 1:, :, :, 1].max() < 0.01
    assert sem[0, 0, 0, 0, 1] > 0.5


def test_binary_branch_unchanged_shape():
    B, Q, D, H, W = 1, 3, 4, 4, 4
    pred_logits = torch.randn(B, Q, 2)
    pred_masks = torch.randn(B, Q, D, H, W)
    sem, avg = reduce_queries_to_semantic_map(
        pred_masks, pred_logits, num_classes=1, topk_per_image=2,
        reduction="topk_max",
    )
    assert sem.shape == (B, D, H, W, 1)
    assert avg.shape == (B,)  # mean over top-k queries drops the K dim


# ---------------------------------------------------------------------------
# Canonical reduction (default): sums over ALL queries, no-object at channel 0.
# ---------------------------------------------------------------------------

def test_canonical_background_first_layout_and_shapes():
    B, Q, C, D, H, W = 2, 5, 3, 2, 4, 4
    semseg, avg = reduce_queries_to_semantic_map(
        torch.randn(B, Q, D, H, W), torch.randn(B, Q, C + 1), reduction="canonical"
    )
    assert semseg.shape == (B, D, H, W, C + 1)      # channel 0 = background
    assert avg.shape == (B, C)


def test_canonical_confident_class_wins_argmax():
    # One query, overwhelmingly class 1, mask on everywhere -> argmax == 2 (class1 + 1).
    semseg, _ = reduce_queries_to_semantic_map(
        torch.full((1, 1, 1, 2, 2), 10.0),              # sigmoid ~1
        torch.tensor([[[-10.0, 10.0, -10.0]]]),         # [c0, c1, no-object]
        reduction="canonical",
    )
    assert torch.all(semseg.argmax(dim=-1) == 2)


def test_canonical_no_object_yields_background():
    semseg, _ = reduce_queries_to_semantic_map(
        torch.full((1, 1, 1, 2, 2), -10.0),             # sigmoid ~0
        torch.tensor([[[-10.0, -10.0, 10.0]]]),         # no-object confident
        reduction="canonical",
    )
    assert torch.all(semseg.argmax(dim=-1) == 0)


def test_canonical_sums_over_all_queries_not_topk():
    # Two queries each half-confident on class 0 must beat one query on class 1.
    semseg, _ = reduce_queries_to_semantic_map(
        torch.full((1, 3, 1, 1, 1), 10.0),
        torch.tensor([[[2.0, 0.0, -10.0], [2.0, 0.0, -10.0], [0.0, 2.0, -10.0]]]),
        reduction="canonical",
    )
    # class0 channel (index 1) accumulates two queries, class1 (index 2) only one.
    assert semseg[0, 0, 0, 0, 1] > semseg[0, 0, 0, 0, 2]


def test_canonical_binary_and_multiclass_share_one_code_path():
    for c in (1, 4):
        semseg, avg = reduce_queries_to_semantic_map(
            torch.randn(1, 3, 1, 2, 2), torch.randn(1, 3, c + 1), reduction="canonical"
        )
        assert semseg.shape == (1, 1, 2, 2, c + 1)
        assert avg.shape == (1, c)


def test_topk_max_still_returns_legacy_layout():
    # The legacy reduction is unchanged: no background channel.
    B, Q, C, D, H, W = 2, 5, 3, 2, 4, 4
    semseg, avg = reduce_queries_to_semantic_map(
        torch.randn(B, Q, D, H, W), torch.randn(B, Q, C + 1),
        num_classes=C, topk_per_image=2, reduction="topk_max",
    )
    assert semseg.shape == (B, D, H, W, C)
    assert avg.shape == (B, C)


def test_unknown_reduction_raises():
    with pytest.raises(ValueError, match="reduction"):
        reduce_queries_to_semantic_map(
            torch.randn(1, 1, 1, 2, 2), torch.randn(1, 1, 2), reduction="nonsense"
        )
