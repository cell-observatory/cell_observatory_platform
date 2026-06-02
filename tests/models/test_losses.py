import pytest
import torch

import cell_observatory_platform.training.losses as losses_mod
from cell_observatory_platform.models.ops.losses import (
    batch_dice_loss,
    batch_sigmoid_ce_loss,
    calculate_uncertainty,
    dice_loss,
    sigmoid_ce_loss,
    sigmoid_focal_loss,
)


class DummyMatcher(torch.nn.Module):
    """
    Simple matcher that pairs the first T queries with T targets
    for each batch element, where T = num_targets for that element.
    """

    def forward(self, outputs, targets, costs=None):
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        matched = []
        for b in range(batch_size):
            num_targets = len(targets[b]["labels"])
            if num_targets == 0:
                src_idx = torch.empty(0, dtype=torch.int64)
                tgt_idx = torch.empty(0, dtype=torch.int64)
            else:
                src_idx = torch.arange(num_targets, dtype=torch.int64)
                tgt_idx = torch.arange(num_targets, dtype=torch.int64)
            matched.append((src_idx, tgt_idx))
        return matched


# Loss function tests


def test_sigmoid_focal_loss_monotonicity():
    # Perfect prediction: high logit on true class, low on false class
    inputs_good = torch.tensor([[10.0, -10.0]])
    targets = torch.tensor([[1.0, 0.0]])
    num_boxes = 1.0

    # Bad prediction: flipped logits
    inputs_bad = torch.tensor([[-10.0, 10.0]])

    loss_good = sigmoid_focal_loss(inputs_good, targets, num_boxes)
    loss_bad = sigmoid_focal_loss(inputs_bad, targets, num_boxes)

    assert loss_good < loss_bad
    assert loss_good >= 0
    assert loss_bad >= 0


def test_dice_loss_overlap():
    # Simple 2x2 mask, one example
    # GT has ones on diagonal
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # (N=1, D=1, H=2, W=2)

    # Good logits: large positive where target=1, negative where target=0
    inputs_good = torch.tensor([[[[5.0, -5.0], [-5.0, 5.0]]]])
    # Bad logits: flip positives/negatives
    inputs_bad = -inputs_good

    num_masks = 1.0

    loss_good = dice_loss(inputs_good, targets, num_masks)
    loss_bad = dice_loss(inputs_bad, targets, num_masks)

    assert loss_good < loss_bad
    assert loss_good >= 0
    assert loss_bad <= 1.0 + 1e-3  # should be bounded


def test_sigmoid_ce_loss_monotonicity():
    # Binary BCE on two pixels
    targets = torch.tensor([[1.0, 0.0]])
    num_masks = 1.0

    # Good logits: match targets
    inputs_good = torch.tensor([[5.0, -5.0]])
    # Bad logits: flipped
    inputs_bad = -inputs_good

    loss_good = sigmoid_ce_loss(inputs_good, targets, num_masks)
    loss_bad = sigmoid_ce_loss(inputs_bad, targets, num_masks)

    assert loss_good < loss_bad
    assert loss_good >= 0
    assert loss_bad >= 0


def test_batch_dice_loss_diagonal_smaller():
    # Two distinct target masks
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # mask A
            [0.0, 0.0, 0.0, 1.0],  # mask B
        ]
    )  # (M=2, C=4)

    # Build logits that match each target (after sigmoid)
    # Use large magnitude to approximate hard 0/1
    logits = torch.tensor(
        [
            [10.0, -10.0, -10.0, -10.0],  # matches targets[0]
            [-10.0, -10.0, -10.0, 10.0],  # matches targets[1]
        ]
    )  # (N=2, C=4)

    loss_matrix = batch_dice_loss(logits, targets)

    # Diagonal entries should be smaller (better match) than off-diagonals
    assert loss_matrix.shape == (2, 2)
    assert loss_matrix[0, 0] < loss_matrix[0, 1]
    assert loss_matrix[1, 1] < loss_matrix[1, 0]


def test_batch_sigmoid_ce_loss_diagonal_smaller():
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],  # mask A
            [0.0, 0.0, 0.0, 1.0],  # mask B
        ]
    )

    logits = torch.tensor(
        [
            [10.0, -10.0, -10.0, -10.0],  # matches target 0
            [-10.0, -10.0, -10.0, 10.0],  # matches target 1
        ]
    )

    loss_matrix = batch_sigmoid_ce_loss(logits, targets)

    assert loss_matrix.shape == (2, 2)
    # lower loss when prediction matches target
    assert loss_matrix[0, 0] < loss_matrix[0, 1]
    assert loss_matrix[1, 1] < loss_matrix[1, 0]
    assert torch.all(loss_matrix >= 0)


def test_calculate_uncertainty_ordering():
    # logits near zero -> most uncertain (score ~0)
    # large magnitude logits -> more confident (score large negative)
    logits = torch.tensor(
        [
            [[0.0]],  # most uncertain
            [[2.0]],  # less uncertain
            [[-3.0]],  # least uncertain
        ]
    )  # (R=3, 1, 1)

    scores = calculate_uncertainty(logits)
    assert scores.shape == logits.shape

    # scores = -abs(logits)
    # so: 0 > -2 > -3
    assert scores[0, 0, 0] > scores[1, 0, 0]
    assert scores[1, 0, 0] > scores[2, 0, 0]


# DETR_Set_Loss tests


@pytest.mark.gpu
def test_detr_set_loss_basic(monkeypatch):
    # Make world_size=1 and no distributed init for deterministic behavior
    monkeypatch.setattr(losses_mod, "get_world_size", lambda: 1)
    monkeypatch.setattr(losses_mod, "is_torch_dist_initialized", lambda: False)

    device = torch.device("cuda")

    batch_size = 2
    num_queries = 5
    num_classes = 3
    D = H = W = 8

    matcher = DummyMatcher()
    criterion = losses_mod.DETR_Set_Loss(
        num_classes=num_classes,
        matcher=matcher,
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "boxes", "masks"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
        denoise=False,
        semantic_ce_loss=True,
    ).to(device)

    # Outputs
    pred_logits = torch.randn(batch_size, num_queries, num_classes + 1, device=device)
    pred_boxes = torch.rand(batch_size, num_queries, 6, device=device)
    pred_masks = torch.randn(batch_size, num_queries, D, H, W, device=device)

    outputs = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "pred_masks": pred_masks,
    }

    # Targets: 2 images, each with 3 objects
    targets = []
    for _ in range(batch_size):
        labels = torch.randint(0, num_classes, (3,), device=device)
        boxes = torch.rand(3, 6, device=device)
        masks = torch.randint(0, 2, (3, D, H, W), device=device, dtype=torch.float32)
        mask_ids = torch.arange(1, 4, device=device)
        label_map = torch.zeros(D, H, W, dtype=torch.long, device=device)
        targets.append({"labels": labels, "boxes": boxes, "masks": masks, "mask_ids": mask_ids, "label_map": label_map})

    losses = criterion(outputs, targets)

    # Expect main losses present
    assert "loss_ce" in losses
    assert "loss_bbox" in losses
    assert "loss_giou" in losses
    assert "loss_mask" in losses
    assert "loss_dice" in losses

    for v in losses.values():
        assert v.ndim == 0
        assert torch.isfinite(v)


@pytest.mark.gpu
def test_detr_set_loss_focal_branch(monkeypatch):
    # Test the focal-loss classification branch (semantic_ce_loss=False)
    monkeypatch.setattr(losses_mod, "get_world_size", lambda: 1)
    monkeypatch.setattr(losses_mod, "is_torch_dist_initialized", lambda: False)

    device = torch.device("cuda")

    batch_size = 1
    num_queries = 4
    num_classes = 2
    D = H = W = 4

    matcher = DummyMatcher()
    criterion = losses_mod.DETR_Set_Loss(
        num_classes=num_classes,
        matcher=matcher,
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels"],
        num_points=8,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
        denoise=False,
        semantic_ce_loss=False,
    ).to(device)

    pred_logits = torch.randn(batch_size, num_queries, num_classes, device=device)
    pred_boxes = torch.rand(batch_size, num_queries, 6, device=device)
    pred_masks = torch.randn(batch_size, num_queries, D, H, W, device=device)

    outputs = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "pred_masks": pred_masks,
    }

    labels = torch.randint(0, num_classes, (3,), device=device)
    boxes = torch.rand(3, 6, device=device)
    masks = torch.randint(0, 2, (3, D, H, W), device=device, dtype=torch.float32)
    targets = [{"labels": labels, "boxes": boxes, "masks": masks}]

    losses = criterion(outputs, targets)

    assert "loss_ce" in losses
    v = losses["loss_ce"]
    assert v.ndim == 0
    assert torch.isfinite(v)


@pytest.mark.gpu
def test_detr_set_loss_denoise_without_predictions(monkeypatch):
    # denoise=True but no denoise_predictions -> zero-valued *_denoise keys
    monkeypatch.setattr(losses_mod, "get_world_size", lambda: 1)
    monkeypatch.setattr(losses_mod, "is_torch_dist_initialized", lambda: False)

    device = torch.device("cuda")

    batch_size = 1
    num_queries = 4
    num_classes = 2
    D = H = W = 4

    matcher = DummyMatcher()
    criterion = losses_mod.DETR_Set_Loss(
        num_classes=num_classes,
        matcher=matcher,
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "boxes", "masks"],
        num_points=4,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
        denoise=True,
        # denoise_type="seg",
        denoise_losses=["labels", "boxes", "masks"],
        semantic_ce_loss=True,
    ).to(device)

    # classifier outputs foreground + no-object
    pred_logits = torch.randn(batch_size, num_queries, num_classes + 1, device=device)
    pred_boxes = torch.rand(batch_size, num_queries, 6, device=device)
    pred_masks = torch.randn(batch_size, num_queries, D, H, W, device=device)

    outputs = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "pred_masks": pred_masks,
    }

    labels = torch.randint(0, num_classes, (2,), device=device)
    boxes = torch.rand(2, 6, device=device)
    masks = torch.randint(0, 2, (2, D, H, W), device=device, dtype=torch.float32)
    mask_ids = torch.arange(1, 3, device=device)
    label_map = torch.zeros(D, H, W, dtype=torch.long, device=device)
    targets = [{"labels": labels, "boxes": boxes, "masks": masks, "mask_ids": mask_ids, "label_map": label_map}]

    losses = criterion(outputs, targets, denoise_predictions=None)

    # Check that denoise keys exist and are exactly zero
    for key in [
        "loss_bbox_denoise",
        "loss_giou_denoise",
        "loss_ce_denoise",
        "loss_mask_denoise",
        "loss_dice_denoise",
    ]:
        assert key in losses
        assert losses[key].ndim == 0
        assert torch.isfinite(losses[key])
        assert float(losses[key].item()) == 0.0


@pytest.mark.gpu
def test_detr_set_loss_denoise_with_predictions(monkeypatch):
    # denoise=True and valid denoise_predictions -> *_denoise keys non-zero/finate
    monkeypatch.setattr(losses_mod, "get_world_size", lambda: 1)
    monkeypatch.setattr(losses_mod, "is_torch_dist_initialized", lambda: False)

    device = torch.device("cuda")

    batch_size = 1
    num_queries = 4
    num_classes = 2
    D = H = W = 4

    matcher = DummyMatcher()
    criterion = losses_mod.DETR_Set_Loss(
        num_classes=num_classes,
        matcher=matcher,
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "boxes", "masks"],
        num_points=4,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
        denoise=True,
        # denoise_type="seg",
        denoise_losses=["labels", "boxes", "masks"],
        semantic_ce_loss=True,
    ).to(device)

    # Main outputs
    pred_logits = torch.randn(batch_size, num_queries, num_classes + 1, device=device)
    pred_boxes = torch.rand(batch_size, num_queries, 6, device=device)
    pred_masks = torch.randn(batch_size, num_queries, D, H, W, device=device)

    outputs = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "pred_masks": pred_masks,
    }

    # Targets
    labels = torch.randint(0, num_classes, (2,), device=device)
    boxes = torch.rand(2, 6, device=device)
    masks = torch.randint(0, 2, (2, D, H, W), device=device, dtype=torch.float32)
    mask_ids = torch.arange(1, 3, device=device)
    label_map = torch.zeros(D, H, W, dtype=torch.long, device=device)
    targets = [{"labels": labels, "boxes": boxes, "masks": masks, "mask_ids": mask_ids, "label_map": label_map}]

    # Construct synthetic denoise_predictions that match the expected structure
    denoise_queries_per_label = 2
    query_pad_size_per_label = 4
    max_query_pad_size = denoise_queries_per_label * query_pad_size_per_label  # 8
    num_dn_queries = max_query_pad_size

    dn_pred_logits = torch.randn(batch_size, num_dn_queries, num_classes + 1, device=device)
    dn_pred_boxes = torch.rand(batch_size, num_dn_queries, 6, device=device)
    dn_pred_masks = torch.randn(batch_size, num_dn_queries, D, H, W, device=device)

    denoise_predictions = {
        "predicted_denoise_outputs": {
            "pred_logits": dn_pred_logits,
            "pred_boxes": dn_pred_boxes,
            "pred_masks": dn_pred_masks,
            "auxiliary_outputs": [],
        },
        "denoise_target_indices": torch.arange(len(labels), dtype=torch.int64, device=device),
        "max_query_pad_size": max_query_pad_size,
        "denoise_queries_per_label": denoise_queries_per_label,
    }

    losses = criterion(outputs, targets, denoise_predictions=denoise_predictions)

    # Main losses + denoise losses should be present
    for key in ["loss_ce", "loss_bbox", "loss_giou", "loss_mask", "loss_dice"]:
        assert key in losses

    for key in ["loss_ce_denoise", "loss_bbox_denoise", "loss_giou_denoise", "loss_mask_denoise", "loss_dice_denoise"]:
        assert key in losses
        assert losses[key].ndim == 0
        assert torch.isfinite(losses[key])


@pytest.mark.gpu
def test_detr_set_loss_with_aux_and_intermediate(monkeypatch):
    # Exercise auxiliary_outputs and intermediate_outputs branches
    monkeypatch.setattr(losses_mod, "get_world_size", lambda: 1)
    monkeypatch.setattr(losses_mod, "is_torch_dist_initialized", lambda: False)

    device = torch.device("cuda")

    batch_size = 1
    num_queries = 3
    num_classes = 2
    D = H = W = 4

    matcher = DummyMatcher()
    criterion = losses_mod.DETR_Set_Loss(
        num_classes=num_classes,
        matcher=matcher,
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels"],
        num_points=4,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
        denoise=False,
        semantic_ce_loss=True,
    ).to(device)

    # Main outputs
    main_logits = torch.randn(batch_size, num_queries, num_classes + 1, device=device)
    main_boxes = torch.rand(batch_size, num_queries, 6, device=device)
    main_masks = torch.randn(batch_size, num_queries, D, H, W, device=device)

    # One auxiliary layer
    aux_logits = torch.randn(batch_size, num_queries, num_classes + 1, device=device)
    aux_boxes = torch.rand(batch_size, num_queries, 6, device=device)
    aux_masks = torch.randn(batch_size, num_queries, D, H, W, device=device)

    outputs = {
        "pred_logits": main_logits,
        "pred_boxes": main_boxes,
        "pred_masks": main_masks,
        "auxiliary_outputs": [
            {
                "pred_logits": aux_logits,
                "pred_boxes": aux_boxes,
                "pred_masks": aux_masks,
            }
        ],
        # Also add an intermediate_outputs dict to trigger the intermediate branch
        "intermediate_outputs": {
            "pred_logits": main_logits.clone(),
            "pred_boxes": main_boxes.clone(),
            "pred_masks": main_masks.clone(),
        },
    }

    labels = torch.randint(0, num_classes, (2,), device=device)
    boxes = torch.rand(2, 6, device=device)
    masks = torch.randint(0, 2, (2, D, H, W), device=device, dtype=torch.float32)
    targets = [{"labels": labels, "boxes": boxes, "masks": masks}]

    losses = criterion(outputs, targets)

    # Main loss
    assert "loss_ce" in losses

    # Aux layer 0 loss
    assert "loss_ce_0" in losses

    for key in ["loss_ce", "loss_ce_0"]:
        assert losses[key].ndim == 0
        assert torch.isfinite(losses[key])


# -------------------------------------------------------------------------
# DETR_Set_Loss tests
# -------------------------------------------------------------------------


def _make_plain_detr_batch(
    batch_size: int = 2,
    num_queries: int = 5,
    num_classes: int = 3,
    num_targets_per_image: int = 2,
    device: str = "cpu",
):
    """
    Helper to build a small, consistent batch of outputs/targets/indices/num_boxes for DETR_Set_Loss.
    """
    torch.manual_seed(0)

    pred_logits = torch.randn(batch_size, num_queries, num_classes, device=device)
    pred_boxes = torch.rand(batch_size, num_queries, 6, device=device)  # cx,cy,cz,w,h,d

    outputs = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
    }

    targets = []
    for _ in range(batch_size):
        labels = torch.randint(0, num_classes, (num_targets_per_image,), device=device)
        boxes = torch.rand(num_targets_per_image, 6, device=device)
        targets.append({"labels": labels, "boxes": boxes})

    matcher = DummyMatcher()
    indices = matcher(outputs, targets)

    num_boxes = float(sum(len(t["labels"]) for t in targets))
    return outputs, targets, indices, num_boxes


def test_plain_detr_loss_labels_basic_cpu():
    num_classes = 3
    outputs, targets, indices, num_boxes = _make_plain_detr_batch(num_classes=num_classes)

    criterion = losses_mod.PlainDETR_Set_Loss(
        num_classes=num_classes,
        matcher=DummyMatcher(),
        weight_dict={},
        losses=["labels"],
        focal_alpha=0.25,
        reparam=False,
    )

    losses = criterion.loss_labels(outputs, targets, indices, num_boxes)

    assert "loss_ce" in losses
    v = losses["loss_ce"]
    assert v.ndim == 0
    assert torch.isfinite(v)


def test_plain_detr_loss_cardinality_scalar_and_finite():
    num_classes = 3
    outputs, targets, indices, num_boxes = _make_plain_detr_batch(num_classes=num_classes)

    criterion = losses_mod.PlainDETR_Set_Loss(
        num_classes=num_classes,
        matcher=DummyMatcher(),
        weight_dict={},
        losses=["cardinality"],
        focal_alpha=0.25,
        reparam=False,
    )

    losses = criterion.loss_cardinality(outputs, targets, indices, num_boxes)
    assert "cardinality_error" in losses
    v = losses["cardinality_error"]
    assert v.ndim == 0
    assert torch.isfinite(v)


def test_plain_detr_loss_boxes_l1_perfect_match_zero():
    """
    When pred_boxes == target boxes on matched indices, L1 and GIoU losses should be ~0.
    """
    num_classes = 3
    batch_size = 1
    num_queries = 4
    num_targets_per_image = 4

    outputs, targets, indices, num_boxes = _make_plain_detr_batch(
        batch_size=batch_size,
        num_queries=num_queries,
        num_classes=num_classes,
        num_targets_per_image=num_targets_per_image,
    )

    # Force perfect match on the matched queries
    with torch.no_grad():
        for (src_idx, tgt_idx), t in zip(indices, targets):
            outputs["pred_boxes"][0, src_idx] = t["boxes"][tgt_idx]

    criterion = losses_mod.PlainDETR_Set_Loss(
        num_classes=num_classes,
        matcher=DummyMatcher(),
        weight_dict={},
        losses=["boxes"],
        focal_alpha=0.25,
        reparam=False,  # L1 mode
    )

    losses = criterion.loss_boxes(outputs, targets, indices, num_boxes)

    assert "loss_bbox" in losses
    assert "loss_giou" in losses
    assert losses["loss_bbox"].ndim == 0
    assert losses["loss_giou"].ndim == 0
    assert torch.allclose(losses["loss_bbox"], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(losses["loss_giou"], torch.tensor(0.0), atol=1e-5)


def test_plain_detr_loss_boxes_reparam_requires_keys():
    """
    In reparam mode, missing pred_deltas / pred_boxes_old should raise KeyError.
    """
    num_classes = 3
    outputs, targets, indices, num_boxes = _make_plain_detr_batch(num_classes=num_classes)

    criterion = losses_mod.PlainDETR_Set_Loss(
        num_classes=num_classes,
        matcher=DummyMatcher(),
        weight_dict={},
        losses=["boxes"],
        focal_alpha=0.25,
        reparam=True,
    )

    with pytest.raises(KeyError):
        _ = criterion.loss_boxes(outputs, targets, indices, num_boxes)


def test_plain_detr_loss_boxes_reparam_runs_with_required_keys():
    num_classes = 3
    outputs, targets, indices, num_boxes = _make_plain_detr_batch(num_classes=num_classes)

    B, Q, _ = outputs["pred_boxes"].shape
    device = outputs["pred_boxes"].device

    outputs["pred_boxes_old"] = torch.rand(B, Q, 6, device=device)
    outputs["pred_deltas"] = torch.randn(B, Q, 6, device=device)

    criterion = losses_mod.PlainDETR_Set_Loss(
        num_classes=num_classes,
        matcher=DummyMatcher(),
        weight_dict={},
        losses=["boxes"],
        focal_alpha=0.25,
        reparam=True,
    )

    losses = criterion.loss_boxes(outputs, targets, indices, num_boxes)

    assert "loss_bbox" in losses
    assert "loss_giou" in losses
    assert losses["loss_bbox"].ndim == 0
    assert losses["loss_giou"].ndim == 0
    assert torch.isfinite(losses["loss_bbox"]).item()
    assert torch.isfinite(losses["loss_giou"]).item()


@pytest.mark.parametrize(
    "loss_name,expected_key",
    [
        ("labels", "loss_ce"),
        ("boxes", "loss_bbox"),
        ("cardinality", "cardinality_error"),
    ],
)
def test_plain_detr_get_loss_dispatch(loss_name, expected_key):
    num_classes = 3
    outputs, targets, indices, num_boxes = _make_plain_detr_batch(num_classes=num_classes)

    criterion = losses_mod.PlainDETR_Set_Loss(
        num_classes=num_classes,
        matcher=DummyMatcher(),
        weight_dict={},
        losses=[loss_name],
        focal_alpha=0.25,
        reparam=False,
    )

    losses = criterion.get_loss(loss_name, outputs, targets, indices, num_boxes)
    assert expected_key in losses
    assert losses[expected_key].ndim == 0
    assert torch.isfinite(losses[expected_key])


def test_plain_detr_set_loss_forward_with_aux_and_enc_outputs_cpu():
    torch.manual_seed(0)

    batch_size = 2
    num_queries = 5
    num_classes = 3

    outputs, targets, _, _ = _make_plain_detr_batch(
        batch_size=batch_size,
        num_queries=num_queries,
        num_classes=num_classes,
        num_targets_per_image=2,
    )

    # Add aux_outputs: one aux layer with same shapes
    aux_outputs = [
        {
            "pred_logits": outputs["pred_logits"].clone(),
            "pred_boxes": outputs["pred_boxes"].clone(),
        }
    ]
    outputs["aux_outputs"] = aux_outputs

    # Add enc_outputs: encoder head outputs with same shapes
    enc_outputs = {
        "pred_logits": outputs["pred_logits"].clone(),
        "pred_boxes": outputs["pred_boxes"].clone(),
    }
    outputs["enc_outputs"] = enc_outputs

    criterion = losses_mod.PlainDETR_Set_Loss(
        num_classes=num_classes,
        matcher=DummyMatcher(),
        weight_dict={},
        losses=["labels", "boxes", "cardinality"],
        focal_alpha=0.25,
        reparam=False,
    )

    loss_dict = criterion(outputs, targets)

    # Base losses
    assert "loss_ce" in loss_dict
    assert "loss_bbox" in loss_dict
    assert "loss_giou" in loss_dict
    assert "cardinality_error" in loss_dict

    # Aux losses (index 0)
    assert "loss_ce_0" in loss_dict
    assert "loss_bbox_0" in loss_dict
    assert "loss_giou_0" in loss_dict
    assert "cardinality_error_0" in loss_dict

    # Encoder losses
    assert "loss_ce_enc" in loss_dict
    assert "loss_bbox_enc" in loss_dict
    assert "loss_giou_enc" in loss_dict
    assert "cardinality_error_enc" in loss_dict

    for k, v in loss_dict.items():
        assert v.ndim == 0, f"{k} is not scalar"
        assert torch.isfinite(v), f"{k} is not finite"


# -------------------------------------------------------------------------
# MultiStepMultiMasksAndIousLoss regression tests
# -------------------------------------------------------------------------


@pytest.mark.gpu
def test_sample_points_and_labels_yields_binary_labels():
    """`MultiStepMultiMasksAndIousLoss._sample_points_and_labels` must return
    point_labels in strictly {0, 1}. This is the actual call site at
    training/losses.py:1854 that consumed mode="nearest" - if that kwarg ever
    regresses, this test catches it because the labels feeding into focal/dice
    would then carry fractional values.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)

    criterion = losses_mod.MultiStepMultiMasksAndIousLoss(
        input_fmt="TZYXC",
        weight_dict={
            "loss_mask": 20.0,
            "loss_dice": 1.0,
            "loss_iou": 1.0,
            "loss_class": 0.0,
        },
        use_point_sampling=True,
        num_points=64,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
        low_res_multimasks=False,
    ).to(device)

    # src_masks: [N=2, M=1, Z=8, Y=8, X=8] coarse logits (any finite values are fine).
    src_masks = torch.randn(2, 1, 8, 8, 8, device=device)
    # target_masks: [N=2, 1, Z=8, Y=8, X=8] strictly {0, 1}.
    target_masks = torch.zeros(2, 1, 8, 8, 8, device=device, dtype=torch.float32)
    target_masks[0, 0, 2:6, 2:6, 2:6] = 1.0
    target_masks[1, 0, 1:4, 3:7, 2:5] = 1.0

    point_coords_flat, point_labels = criterion._sample_points_and_labels(
        src_masks, target_masks
    )

    # Shape: [N=2, M=1, P=64]
    assert point_labels.shape == (2, 1, 64)
    unique_vals = torch.unique(point_labels)
    assert torch.all((point_labels == 0.0) | (point_labels == 1.0)), (
        "_sample_points_and_labels must emit strictly binary labels (mode="
        f"'nearest' contract); unique values seen: {unique_vals.tolist()}"
    )
