import pytest
import torch

from cell_observatory_platform.models.ops.point_sampling import (
    sample_uncertain_points_and_labelmap_labels,
)
from cell_observatory_platform.models.layers.utils import (
    point_sample,
    point_sample_labelmap_batched,
    get_uncertain_point_coords_with_randomness,
)
from cell_observatory_platform.models.ops.losses import calculate_uncertainty

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_sample_uncertain_points_and_labelmap_labels_shapes():
    device = torch.device("cuda")
    N, Z, Y, X = 4, 6, 8, 10
    P = 32

    src_logits = torch.randn(N, 1, Z, Y, X, device=device)
    labelmap = torch.arange(2 * Z * Y * X, dtype=torch.int32, device=device).reshape(2, Z, Y, X) + 1
    batch_indices = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    instance_ids = torch.tensor(
        [int(labelmap[0, 0, 0, 0].item()), int(labelmap[0, 1, 2, 3].item()),
         int(labelmap[1, 2, 3, 4].item()), int(labelmap[1, 5, 7, 9].item())],
        dtype=torch.int64, device=device,
    )

    coords, labels = sample_uncertain_points_and_labelmap_labels(
        src_logits=src_logits,
        labelmap=labelmap,
        batch_indices=batch_indices,
        instance_ids=instance_ids,
        num_points=P,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    )

    assert coords.shape == (N, P, 3)
    assert labels.shape == (N, P)
    assert torch.all(coords >= 0.0) and torch.all(coords <= 1.0)
    assert labels.dtype == torch.float32


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_sample_uncertain_points_and_labelmap_labels_sentinel_padding():
    # Sentinel id -1 must never match any labelmap voxel: labels are all 0.
    device = torch.device("cuda")
    N, Z, Y, X = 3, 4, 4, 4
    P = 64

    src_logits = torch.randn(N, 1, Z, Y, X, device=device)
    labelmap = torch.randint(0, 5, (1, Z, Y, X), dtype=torch.int32, device=device)
    batch_indices = torch.zeros((N,), dtype=torch.long, device=device)
    instance_ids = torch.tensor([-1, -1, -1], dtype=torch.int64, device=device)

    _, labels = sample_uncertain_points_and_labelmap_labels(
        src_logits=src_logits,
        labelmap=labelmap,
        batch_indices=batch_indices,
        instance_ids=instance_ids,
        num_points=P,
    )

    assert torch.all(labels == 0.0), "sentinel pad id must produce all-zero labels"


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_sample_uncertain_points_and_labelmap_labels_matches_dense_reference():
    # On a non-symmetric labelmap, the sampled labels at the returned coords must
    # equal point_sample(mode='nearest') of the dense (labelmap == id) binary mask.
    # This verifies the shared helper composes the coord-order-fixed primitives
    # correctly end-to-end.
    device = torch.device("cuda")
    Z, Y, X = 3, 5, 7
    labelmap_single = (
        torch.arange(Z * Y * X, dtype=torch.int32, device=device).reshape(Z, Y, X) + 1
    )
    labelmap = labelmap_single.unsqueeze(0)  # [1, Z, Y, X]

    N = 4
    src_logits = torch.randn(N, 1, Z, Y, X, device=device)
    batch_indices = torch.zeros((N,), dtype=torch.long, device=device)
    instance_ids = torch.tensor(
        [int(labelmap_single[0, 0, 0].item()),
         int(labelmap_single[1, 2, 4].item()),
         int(labelmap_single[2, 4, 6].item()),
         int(labelmap_single[1, 3, 5].item())],
        dtype=torch.int64, device=device,
    )

    coords, labels = sample_uncertain_points_and_labelmap_labels(
        src_logits=src_logits,
        labelmap=labelmap,
        batch_indices=batch_indices,
        instance_ids=instance_ids,
        num_points=128,
    )

    for n in range(N):
        ref_input = (labelmap.float() == int(instance_ids[n].item())).float().unsqueeze(1)
        labels_ref = point_sample(
            ref_input, coords[n : n + 1], mode="nearest", align_corners=False,
        ).squeeze(0).squeeze(0)
        assert torch.equal(labels[n], labels_ref), (
            f"row {n}: sampled labels disagree with point_sample(mode='nearest') reference"
        )
