import pytest
import torch

from cell_observatory_platform.models.layers.utils import (
    _max_by_axis,
    batch_tensors,
    compute_unmasked_ratio,
    get_reference_points,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_get_reference_points_shapes_and_range():
    device = torch.device("cuda")

    batch_size = 2
    # realistic 3D pyramid
    shapes = torch.tensor(
        [
            [32, 32, 32],  # level 0
            [16, 16, 16],  # level 1
            [8, 8, 8],  # level 2
        ],
        dtype=torch.long,
        device=device,
    )  # (L, 3)
    num_levels = shapes.shape[0]

    # valid ratios all ones -> no padding
    valid_ratios = torch.ones(batch_size, num_levels, 3, device=device)

    ref_pts = get_reference_points(shapes, valid_ratios, device)

    # S = sum(D*H*W)
    num_tokens_per_level = shapes.prod(dim=1)
    total_tokens = int(num_tokens_per_level.sum().item())

    # shape: (B, S, L, 3)
    assert ref_pts.shape == (batch_size, total_tokens, num_levels, 3)

    # coordinates should lie in [0, 1] after final scaling
    assert torch.all(ref_pts >= 0.0)
    assert torch.all(ref_pts <= 1.0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_point_sample_vector_coords():
    device = torch.device("cuda")

    N, C, D, H, W = 2, 3, 5, 6, 7
    P = 10

    x = torch.ones(N, C, D, H, W, device=device)
    point_coords = torch.rand(N, P, 3, device=device)

    out = point_sample(x, point_coords, mode="bilinear", align_corners=False)

    assert out.shape == (N, C, P)
    assert out.dtype == x.dtype

    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_point_sample_grid_coords():
    device = torch.device("cuda")

    N, C, D, H, W = 1, 2, 4, 4, 4
    Dz, Hy, Wx = 3, 2, 5

    x = torch.randn(N, C, D, H, W, device=device)
    point_coords = torch.rand(N, Dz, Hy, Wx, 3, device=device)

    out = point_sample(x, point_coords, mode="bilinear", align_corners=False)

    assert out.shape == (N, C, Dz, Hy, Wx)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_get_uncertain_point_coords_with_randomness_shapes_and_range():
    device = torch.device("cuda")

    N, C, D, H, W = 2, 1, 8, 8, 8
    coarse_logits = torch.randn(N, C, D, H, W, device=device)

    def uncertainty_fn(logits: torch.Tensor) -> torch.Tensor:
        # logits: (N, C, P) -> return (N, 1, P)
        return -logits.abs().mean(dim=1, keepdim=True)

    num_points = 20
    oversample_ratio = 3
    importance_sample_ratio = 0.7

    coords = get_uncertain_point_coords_with_randomness(
        coarse_logits,
        uncertainty_fn,
        num_points=num_points,
        oversample_ratio=oversample_ratio,
        importance_sample_ratio=importance_sample_ratio,
    )

    # (N, P, 3)
    assert coords.shape == (N, num_points, 3)
    assert torch.all(coords >= 0.0)
    assert torch.all(coords <= 1.0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_get_uncertain_point_coords_with_randomness_all_random():
    device = torch.device("cuda")

    N, C, D, H, W = 1, 1, 4, 4, 4
    coarse_logits = torch.randn(N, C, D, H, W, device=device)

    def uncertainty_fn(logits: torch.Tensor) -> torch.Tensor:
        return -logits.abs().mean(dim=1, keepdim=True)

    num_points = 10
    oversample_ratio = 1
    importance_sample_ratio = 0.0  # all random points

    coords = get_uncertain_point_coords_with_randomness(
        coarse_logits,
        uncertainty_fn,
        num_points=num_points,
        oversample_ratio=oversample_ratio,
        importance_sample_ratio=importance_sample_ratio,
    )

    assert coords.shape == (N, num_points, 3)
    assert torch.all(coords >= 0.0)
    assert torch.all(coords <= 1.0)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_compute_unmasked_ratio_all_valid_and_partial():
    device = torch.device("cuda")

    B, D, H, W = 2, 4, 3, 5

    # case 1: all voxels valid -> all ratios 1
    mask_all_valid = torch.zeros(B, D, H, W, dtype=torch.bool, device=device)
    ratios_all_valid = compute_unmasked_ratio(mask_all_valid)
    assert ratios_all_valid.shape == (B, 3)
    assert torch.allclose(ratios_all_valid, torch.ones_like(ratios_all_valid))

    # case 2: for sample 0, last half of D slices are fully masked
    mask_partial = mask_all_valid.clone()
    mask_partial[0, 2:, :, :] = True  # mask last 2 of 4 slices along D
    ratios_partial = compute_unmasked_ratio(mask_partial)

    # sample 0: D ratio = 2/4, W/H still fully valid
    assert torch.isclose(ratios_partial[0, 0], torch.tensor(1.0, device=device))
    assert torch.isclose(ratios_partial[0, 1], torch.tensor(1.0, device=device))
    assert torch.isclose(ratios_partial[0, 2], torch.tensor(0.5, device=device))

    # sample 1 unchanged (all valid)
    assert torch.allclose(ratios_partial[1], torch.tensor([1.0, 1.0, 1.0], device=device))
