"""Tests for query-to-semantic reduction (Mask2Former inference)."""

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
        pred_masks, pred_logits, num_classes=num_classes, topk_per_image=2
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
        pred_masks, pred_logits, num_classes=1, topk_per_image=2
    )
    assert sem.shape == (B, D, H, W, 1)
    assert avg.shape == (B,)  # mean over top-k queries drops the K dim
