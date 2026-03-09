import pytest
import torch

from cell_observatory_platform.training.losses import FourierLoss, Mask2FormerSetLoss
from cell_observatory_platform.models.layers.patch_embeddings import PatchEmbedding

CUDA_AVAILABLE = torch.cuda.is_available()


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


@pytest.mark.parametrize(
    "input_fmt,input_shape,patch_shape,in_chans",
    [
        # 4D case: TZYXC
        ("TZYXC", (2, 128, 128, 128, 2), (2, 16, 16, 16), 2),
        # 3D case: ZYXC
        ("ZYXC", (128, 128, 128, 2), (16, 16, 16), 2),
    ],
)
def test_fourier_loss_forward_backward(input_fmt, input_shape, patch_shape, in_chans):
    torch.manual_seed(0)
    device = "cuda"
    B = 2

    loss_mod = FourierLoss(
        alpha=0.01,
        fft_loss="l1_masked",
        spatial_loss="l2_masked",
        input_fmt=input_fmt,
        input_shape=input_shape,
        patch_shape=patch_shape,
        embed_dim=256,
        # in_chans=in_chans, # FIXME
    ).to(device)

    full_shape = (B, *input_shape)
    full_targets_img = torch.randn(full_shape, device=device)

    pe = PatchEmbedding(
        input_fmt=input_fmt,
        input_shape=input_shape,
        patch_shape=patch_shape,
        embed_dim=256,
        channels=in_chans,
    )
    full_targets_patches = pe._patchify(full_targets_img)
    
    B_, P, D = full_targets_patches.shape
    assert B_ == B and P > 4, "Test expects more than 4 patches to mask a few."

    full_predictions_patches = (
        (full_targets_patches + 0.01 * torch.randn_like(full_targets_patches)).detach().requires_grad_(True)
    )

    M = 4
    masked_idx = torch.arange(M, device=device).unsqueeze(0).repeat(B, 1)
    masks_grid = torch.zeros((B, P), dtype=torch.float32, device=device)
    masks_grid.scatter_(1, masked_idx, 1.0)

    batch_index = torch.arange(B, device=device).unsqueeze(-1)
    targets_masked = full_targets_patches[batch_index, masked_idx, :]
    predictions_masked = full_predictions_patches[batch_index, masked_idx, :]

    aux = {
        "targets": full_targets_patches,
        "predictions": full_predictions_patches,
        "target_masks": masked_idx,
    }

    print(f"targets_masked.shape: {targets_masked.shape}")
    print(f"predictions_masked.shape: {predictions_masked.shape}")
    print(f"masks_grid.shape: {masks_grid.shape}")
    print(f"full_targets_patches.shape: {full_targets_patches.shape}")
    print(f"full_predictions_patches.shape: {full_predictions_patches.shape}")
    print(f"target_masks.shape: {aux['target_masks'].shape}")

    loss, out = loss_mod(
        targets=targets_masked,
        predictions=predictions_masked,
        num_patches=masks_grid.sum(),
        aux_loss_meta=aux,
    )
    print(f"Loss output: {loss.item()}")

    assert loss.ndim == 0
    assert torch.isfinite(loss), "Loss produced inf/NaN"

    loss.backward()
    assert full_predictions_patches.grad is not None
    assert torch.isfinite(full_predictions_patches.grad).all()
    assert full_predictions_patches.grad.abs().sum() > 0


# Mask2FormerSetLoss tests


def _make_mask2former_batch(
    batch_size: int = 2,
    num_queries: int = 5,
    num_classes: int = 3,
    num_targets_per_image: int = 2,
    D: int = 8,
    H: int = 8,
    W: int = 8,
    device: str = "cpu",
):
    """
    Helper to build a small, consistent batch of outputs/targets/indices/num_masks for Mask2FormerSetLoss.
    """
    torch.manual_seed(0)

    # pred_logits includes num_classes + 1 for the no-object class
    pred_logits = torch.randn(batch_size, num_queries, num_classes + 1, device=device)
    pred_masks = torch.randn(batch_size, num_queries, D, H, W, device=device)

    outputs = {
        "pred_logits": pred_logits,
        "pred_masks": pred_masks,
    }

    targets = []
    for _ in range(batch_size):
        labels = torch.randint(0, num_classes, (num_targets_per_image,), device=device)
        masks = torch.randint(0, 2, (num_targets_per_image, D, H, W), device=device, dtype=torch.float32)
        targets.append({"labels": labels, "masks": masks})

    matcher = DummyMatcher()
    indices = matcher(outputs, targets)

    num_masks = float(sum(len(t["labels"]) for t in targets))
    return outputs, targets, indices, num_masks


def test_mask2former_set_loss_init():
    """Verify initialization stores all parameters correctly."""
    num_classes = 3
    matcher = DummyMatcher()
    loss_weight_dict = {"loss_ce": 1.0, "loss_mask": 1.0, "loss_dice": 1.0}
    no_object_loss_weight = 0.1
    losses = ["labels", "masks"]
    num_points = 16
    oversample_ratio = 3
    importance_sample_ratio = 0.75

    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=matcher,
        loss_weight_dict=loss_weight_dict,
        no_object_loss_weight=no_object_loss_weight,
        losses=losses,
        num_points=num_points,
        oversample_ratio=oversample_ratio,
        importance_sample_ratio=importance_sample_ratio,
    )

    assert criterion.num_classes == num_classes, "num_classes not stored"
    assert criterion.matcher is matcher, "matcher not stored"
    assert criterion.loss_weight_dict == loss_weight_dict, "loss_weight_dict not stored"
    assert criterion.no_object_loss_weight == no_object_loss_weight, "no_object_loss_weight not stored"
    assert criterion.losses == losses, "losses not stored"
    assert criterion.num_points == num_points, "num_points not stored"
    assert criterion.oversample_ratio == oversample_ratio, "oversample_ratio not stored"
    assert criterion.importance_sample_ratio == importance_sample_ratio, "importance_sample_ratio not stored"
    assert criterion.costs == ["cls", "mask"], "costs not set correctly"

    assert hasattr(criterion, "empty_weight"), "empty_weight buffer not registered"
    empty_weight_buffer = criterion.empty_weight  # type: ignore
    assert empty_weight_buffer.shape == (num_classes + 1,), "empty_weight shape incorrect"
    last_element = empty_weight_buffer[num_classes]  # type: ignore
    assert torch.allclose(last_element, torch.tensor(no_object_loss_weight)), "empty_weight last element should equal no_object_loss_weight"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_mask2former_set_loss_forward_basic(monkeypatch):
    """Basic forward pass with labels and masks."""
    monkeypatch.setattr("cell_observatory_platform.training.losses.get_world_size", lambda: 1)
    monkeypatch.setattr("cell_observatory_platform.training.losses.is_torch_dist_initialized", lambda: False)

    device = torch.device("cuda")
    batch_size = 1
    num_queries = 3
    num_classes = 2
    D = H = W = 4

    matcher = DummyMatcher()
    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=matcher,
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "masks"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
    ).to(device)

    outputs, targets, _, _ = _make_mask2former_batch(
        batch_size=batch_size,
        num_queries=num_queries,
        num_classes=num_classes,
        num_targets_per_image=2,
        D=D,
        H=H,
        W=W,
        device=str(device),
    )

    losses = criterion(outputs, targets)

    assert "loss_labels_ce" in losses, "Classification loss missing"
    assert "loss_mask_ce" in losses, "Mask CE loss missing"
    assert "loss_mask_dice" in losses, "Dice loss missing"

    for v in losses.values():
        assert v.ndim == 0, "Loss must be scalar"
        assert torch.isfinite(v), "Loss must be finite"
        assert v >= 0, "Loss must be non-negative"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_mask2former_set_loss_forward_with_auxiliary(monkeypatch):
    """Test auxiliary outputs handling."""
    monkeypatch.setattr("cell_observatory_platform.training.losses.get_world_size", lambda: 1)
    monkeypatch.setattr("cell_observatory_platform.training.losses.is_torch_dist_initialized", lambda: False)

    device = torch.device("cuda")
    batch_size = 1
    num_queries = 3
    num_classes = 2
    D = H = W = 4

    matcher = DummyMatcher()
    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=matcher,
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "masks"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
    ).to(device)

    outputs, targets, _, _ = _make_mask2former_batch(
        batch_size=batch_size,
        num_queries=num_queries,
        num_classes=num_classes,
        num_targets_per_image=2,
        D=D,
        H=H,
        W=W,
        device=str(device),
    )

    aux_outputs = {
        "pred_logits": outputs["pred_logits"].clone(),
        "pred_masks": outputs["pred_masks"].clone(),
    }
    outputs = {**outputs, "auxiliary_outputs": [aux_outputs]}

    losses = criterion(outputs, targets)

    assert "loss_labels_ce" in losses, "Main classification loss missing"
    assert "loss_mask_ce" in losses, "Main mask CE loss missing"
    assert "loss_mask_dice" in losses, "Main dice loss missing"

    assert "loss_labels_ce_0" in losses, "Auxiliary classification loss missing"
    assert "loss_mask_ce_0" in losses, "Auxiliary mask CE loss missing"
    assert "loss_mask_dice_0" in losses, "Auxiliary dice loss missing"

    for v in losses.values():
        assert v.ndim == 0, "Loss must be scalar"
        assert torch.isfinite(v), "Loss must be finite"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_mask2former_set_loss_forward_empty_targets(monkeypatch):
    """Handle empty targets gracefully."""
    monkeypatch.setattr("cell_observatory_platform.training.losses.get_world_size", lambda: 1)
    monkeypatch.setattr("cell_observatory_platform.training.losses.is_torch_dist_initialized", lambda: False)

    device = torch.device("cuda")
    batch_size = 2
    num_queries = 3
    num_classes = 2
    D = H = W = 4

    matcher = DummyMatcher()
    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=matcher,
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "masks"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
    ).to(device)

    outputs, _, _, _ = _make_mask2former_batch(
        batch_size=batch_size,
        num_queries=num_queries,
        num_classes=num_classes,
        num_targets_per_image=0,
        D=D,
        H=H,
        W=W,
        device=str(device),
    )

    targets = [
        {"labels": torch.empty(0, dtype=torch.int64, device=device), "masks": torch.empty(0, D, H, W, device=device)},
        {"labels": torch.empty(0, dtype=torch.int64, device=device), "masks": torch.empty(0, D, H, W, device=device)},
    ]

    losses = criterion(outputs, targets)

    for v in losses.values():
        assert v.ndim == 0, "Loss must be scalar"
        assert torch.isfinite(v), "Loss must be finite"


def test_mask2former_loss_labels_basic():
    """Test loss_labels method directly."""
    num_classes = 3
    outputs, targets, indices, num_masks = _make_mask2former_batch(num_classes=num_classes)

    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=DummyMatcher(),
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "masks"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
    )

    losses = criterion.loss_labels(outputs, targets, indices, num_masks)

    assert "loss_labels_ce" in losses, "Classification loss missing"
    v = losses["loss_labels_ce"]
    assert v.ndim == 0, "Loss must be scalar"
    assert torch.isfinite(v), "Loss must be finite"
    assert v >= 0, "Loss must be non-negative"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_mask2former_loss_masks_basic(monkeypatch):
    """Test loss_masks method directly."""
    monkeypatch.setattr("cell_observatory_platform.training.losses.get_world_size", lambda: 1)
    monkeypatch.setattr("cell_observatory_platform.training.losses.is_torch_dist_initialized", lambda: False)

    device = torch.device("cuda")
    num_classes = 2
    D = H = W = 4
    outputs, targets, indices, num_masks = _make_mask2former_batch(
        num_classes=num_classes, D=D, H=H, W=W, device=str(device)
    )

    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=DummyMatcher(),
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "masks"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
    ).to(device)

    losses = criterion.loss_masks(outputs, targets, indices, num_masks)

    assert "loss_mask_ce" in losses, "Mask CE loss missing"
    assert "loss_mask_dice" in losses, "Dice loss missing"

    for v in losses.values():
        assert v.ndim == 0, "Loss must be scalar"
        assert torch.isfinite(v), "Loss must be finite"
        assert v >= 0, "Loss must be non-negative"


def test_mask2former_get_loss_invalid_name():
    """Test get_loss with invalid loss name."""
    num_classes = 3
    outputs, targets, indices, num_masks = _make_mask2former_batch(num_classes=num_classes)

    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=DummyMatcher(),
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "masks"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
    )

    with pytest.raises(AssertionError, match="do you really want to compute"):
        criterion.get_loss("invalid_loss", outputs, targets, indices, num_masks)


def test_mask2former_forward_missing_pred_logits():
    """Test forward with missing pred_logits key."""
    device = "cpu"
    num_classes = 2
    D = H = W = 4

    matcher = DummyMatcher()
    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=matcher,
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "masks"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
    )

    outputs = {"pred_masks": torch.randn(1, 3, D, H, W, device=device)}
    targets = [{"labels": torch.tensor([0], device=device), "masks": torch.randint(0, 2, (1, D, H, W), dtype=torch.float32, device=device)}]

    with pytest.raises(KeyError):
        criterion(outputs, targets)


def test_mask2former_forward_missing_pred_masks():
    """Test forward with missing pred_masks when masks loss is enabled."""
    device = "cpu"
    num_classes = 2

    matcher = DummyMatcher()
    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=matcher,
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "masks"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
    )

    # pred_logits should have num_classes + 1 for no-object class
    outputs = {"pred_logits": torch.randn(1, 3, num_classes + 1, device=device)}
    targets = [{"labels": torch.tensor([0], device=device), "masks": torch.randint(0, 2, (1, 4, 4, 4), dtype=torch.float32, device=device)}]

    with pytest.raises(KeyError):
        criterion(outputs, targets)


def test_mask2former_loss_labels_missing_labels():
    """Test loss_labels with targets missing labels key."""
    num_classes = 3
    outputs, _, indices, num_masks = _make_mask2former_batch(num_classes=num_classes)

    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=DummyMatcher(),
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "masks"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
    )

    targets = [{"masks": torch.randint(0, 2, (2, 8, 8, 8), dtype=torch.float32)}]

    with pytest.raises(KeyError):
        criterion.loss_labels(outputs, targets, indices, num_masks)


def test_mask2former_loss_masks_missing_masks():
    """Test loss_masks with targets missing masks key."""
    device = "cpu"
    num_classes = 2
    D = H = W = 4
    outputs, _, indices, num_masks = _make_mask2former_batch(num_classes=num_classes, D=D, H=H, W=W, device=str(device))

    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=DummyMatcher(),
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "masks"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
    )

    targets = [{"labels": torch.tensor([0, 1])}]

    with pytest.raises(KeyError):
        criterion.loss_masks(outputs, targets, indices, num_masks)


def test_mask2former_get_query_indices():
    """Test _get_query_indices helper."""
    num_classes = 3
    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=DummyMatcher(),
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "masks"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
    )

    indices = [
        (torch.tensor([0, 1], dtype=torch.int64), torch.tensor([0, 1], dtype=torch.int64)),
        (torch.tensor([2], dtype=torch.int64), torch.tensor([0], dtype=torch.int64)),
    ]

    batch_indices, source_indices = criterion._get_query_indices(indices)

    assert isinstance(batch_indices, torch.Tensor), "batch_indices must be tensor"
    assert isinstance(source_indices, torch.Tensor), "source_indices must be tensor"
    assert batch_indices.shape == (3,), "batch_indices shape incorrect"
    assert source_indices.shape == (3,), "source_indices shape incorrect"
    assert torch.equal(batch_indices, torch.tensor([0, 0, 1])), "batch_indices values incorrect"
    assert torch.equal(source_indices, torch.tensor([0, 1, 2])), "source_indices values incorrect"


def test_mask2former_get_target_class_indices():
    """Test _get_target_class_indices helper."""
    num_classes = 3
    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=DummyMatcher(),
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels", "masks"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
    )

    indices = [
        (torch.tensor([0, 1], dtype=torch.int64), torch.tensor([0, 1], dtype=torch.int64)),
        (torch.tensor([2], dtype=torch.int64), torch.tensor([0], dtype=torch.int64)),
    ]

    batch_indices, target_indices = criterion._get_target_class_indices(indices)

    assert isinstance(batch_indices, torch.Tensor), "batch_indices must be tensor"
    assert isinstance(target_indices, torch.Tensor), "target_indices must be tensor"
    assert batch_indices.shape == (3,), "batch_indices shape incorrect"
    assert target_indices.shape == (3,), "target_indices shape incorrect"
    assert torch.equal(batch_indices, torch.tensor([0, 0, 1])), "batch_indices values incorrect"
    assert torch.equal(target_indices, torch.tensor([0, 1, 0])), "target_indices values incorrect"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_mask2former_forward_single_loss_labels_only(monkeypatch):
    """Test forward with only labels loss."""
    monkeypatch.setattr("cell_observatory_platform.training.losses.get_world_size", lambda: 1)
    monkeypatch.setattr("cell_observatory_platform.training.losses.is_torch_dist_initialized", lambda: False)

    device = torch.device("cuda")
    batch_size = 1
    num_queries = 3
    num_classes = 2
    D = H = W = 4

    matcher = DummyMatcher()
    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=matcher,
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["labels"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
    ).to(device)

    outputs, targets, _, _ = _make_mask2former_batch(
        batch_size=batch_size,
        num_queries=num_queries,
        num_classes=num_classes,
        num_targets_per_image=2,
        D=D,
        H=H,
        W=W,
        device=str(device),
    )

    losses = criterion(outputs, targets)

    assert "loss_labels_ce" in losses, "Classification loss missing"
    assert "loss_mask_ce" not in losses, "Mask CE loss should not be present"
    assert "loss_mask_dice" not in losses, "Dice loss should not be present"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_mask2former_forward_single_loss_masks_only(monkeypatch):
    """Test forward with only masks loss."""
    monkeypatch.setattr("cell_observatory_platform.training.losses.get_world_size", lambda: 1)
    monkeypatch.setattr("cell_observatory_platform.training.losses.is_torch_dist_initialized", lambda: False)

    device = torch.device("cuda")
    batch_size = 1
    num_queries = 3
    num_classes = 2
    D = H = W = 4

    matcher = DummyMatcher()
    criterion = Mask2FormerSetLoss(
        num_classes=num_classes,
        matcher=matcher,
        loss_weight_dict={},
        no_object_loss_weight=0.1,
        losses=["masks"],
        num_points=16,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
    ).to(device)

    outputs, targets, _, _ = _make_mask2former_batch(
        batch_size=batch_size,
        num_queries=num_queries,
        num_classes=num_classes,
        num_targets_per_image=2,
        D=D,
        H=H,
        W=W,
        device=str(device),
    )

    losses = criterion(outputs, targets)

    assert "loss_mask_ce" in losses, "Mask CE loss missing"
    assert "loss_mask_dice" in losses, "Dice loss missing"
    assert "loss_labels_ce" not in losses, "Classification loss should not be present"
