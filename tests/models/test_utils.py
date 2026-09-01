import pytest
import torch

from cell_observatory_platform.models.layers.utils import (
    compute_unmasked_ratio,
    get_reference_points,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    point_sample_labelmap,
    point_sample_labelmap_batched,
    sample_box_points,
    sample_box_points_from_boxes,
    sample_random_points_from_errors,
)


def test_get_reference_points_shapes_and_range():
    """Reference points are (x, y, z) voxel centres in [0, 1], concatenated level by level."""
    device = torch.device("cpu")
    batch_size = 2
    shapes = torch.tensor([[32, 32, 32], [16, 16, 16], [8, 8, 8]], dtype=torch.long, device=device)
    num_levels = shapes.shape[0]
    valid_ratios = torch.ones(batch_size, num_levels, 3, device=device)

    ref_pts = get_reference_points(shapes, valid_ratios, device)

    total_tokens = int(shapes.prod(dim=1).sum().item())
    assert ref_pts.shape == (batch_size, total_tokens, num_levels, 3)
    assert torch.all(ref_pts >= 0.0) and torch.all(ref_pts <= 1.0)
    # voxel centres: first token of level 0 is (0.5/32,)*3 in (x, y, z); second token steps x by 1/32
    assert torch.allclose(ref_pts[0, 0, 0], torch.full((3,), 0.5 / 32))
    assert torch.allclose(ref_pts[0, 1, 0], torch.tensor([1.5 / 32, 0.5 / 32, 0.5 / 32]))
    # level-1 block starts at token 32**3 with spacing 1/16
    assert torch.allclose(ref_pts[0, 32 ** 3, 0], torch.full((3,), 0.5 / 16))
    # identical across batch when valid_ratios are all ones
    assert torch.equal(ref_pts[0], ref_pts[1])


def test_point_sample_vector_coords():
    device = torch.device("cpu")

    N, C, D, H, W = 2, 3, 5, 6, 7
    P = 10

    x = torch.ones(N, C, D, H, W, device=device)
    point_coords = torch.rand(N, P, 3, device=device)

    out = point_sample(x, point_coords, mode="bilinear", align_corners=False)

    assert out.shape == (N, C, P)
    assert out.dtype == x.dtype

    # Interpolating an all-ones volume is 1.0 up to matmul precision: under
    # TF32 (which an earlier test in the batch may leave enabled) grid_sample
    # can overshoot by ~1e-5, so the bound needs an epsilon -- exact <= 1.0
    # made this test order-dependent within the suite.
    assert torch.all(out >= -1e-4)
    assert torch.all(out <= 1.0 + 1e-4)


def test_point_sample_grid_coords():
    device = torch.device("cpu")

    N, C, D, H, W = 1, 2, 4, 4, 4
    Dz, Hy, Wx = 3, 2, 5

    x = torch.randn(N, C, D, H, W, device=device)
    point_coords = torch.rand(N, Dz, Hy, Wx, 3, device=device)

    out = point_sample(x, point_coords, mode="bilinear", align_corners=False)

    assert out.shape == (N, C, Dz, Hy, Wx)


def test_get_uncertain_point_coords_with_randomness_shapes_and_range():
    device = torch.device("cpu")

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


def test_get_uncertain_point_coords_with_randomness_all_random():
    device = torch.device("cpu")

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


@pytest.mark.parametrize("align_corners", [False, True])
def test_point_sample_labelmap_batched_matches_grid_sample_nearest(align_corners):
    # Regression test for the labelmap coord-order fix.
    # point_sample_labelmap_batched must follow the same grid_sample convention
    # as point_sample: coords[..., 0]=x (W), [..., 1]=y (H), [..., 2]=z (D).
    # A non-symmetric labelmap is required so a swapped order would be detected.
    device = torch.device("cpu")
    Z, Y, X = 3, 5, 7

    labelmap_single = (
        torch.arange(Z * Y * X, dtype=torch.int32, device=device).reshape(Z, Y, X) + 1
    )
    labelmap = labelmap_single.unsqueeze(0)  # [1, Z, Y, X]

    torch.manual_seed(0)
    K = 256
    coords = torch.rand(1, K, 3, device=device)

    target_id = int(labelmap_single[1, 2, 4].item())
    batch_indices = torch.zeros((1,), dtype=torch.long, device=device)
    instance_ids = torch.tensor([target_id], dtype=torch.int64, device=device)

    labels_helper = point_sample_labelmap_batched(
        labelmap=labelmap,
        point_coords=coords,
        batch_indices=batch_indices,
        instance_ids=instance_ids,
        align_corners=align_corners,
    ).squeeze(0)  # [K]

    ref_input = (labelmap.float() == target_id).float().unsqueeze(1)  # [1, 1, Z, Y, X]
    labels_ref = point_sample(
        ref_input, coords, mode="nearest", align_corners=align_corners
    ).squeeze(0).squeeze(0)  # [K]

    assert torch.equal(labels_helper, labels_ref), (
        f"point_sample_labelmap_batched disagrees with point_sample(mode='nearest') "
        f"on a non-symmetric labelmap (align_corners={align_corners})."
    )


@pytest.mark.parametrize("align_corners", [False, True])
def test_point_sample_labelmap_matches_grid_sample_nearest(align_corners):
    # Same regression contract for the single-batch matcher variant.
    device = torch.device("cpu")
    Z, Y, X = 3, 5, 7

    labelmap_single = (
        torch.arange(Z * Y * X, dtype=torch.int32, device=device).reshape(Z, Y, X) + 1
    )

    torch.manual_seed(1)
    K = 256
    coords = torch.rand(1, K, 3, device=device)

    target_ids = torch.tensor(
        [
            int(labelmap_single[0, 0, 0].item()),
            int(labelmap_single[1, 2, 4].item()),
            int(labelmap_single[2, 4, 6].item()),
        ],
        dtype=torch.int64,
        device=device,
    )

    labels_helper = point_sample_labelmap(
        labelmap_single=labelmap_single,
        point_coords=coords,
        instance_ids=target_ids,
        align_corners=align_corners,
    )  # [M, K]

    refs = []
    for tid in target_ids.tolist():
        ref_input = (labelmap_single.float() == tid).float().unsqueeze(0).unsqueeze(0)
        labels_ref = point_sample(
            ref_input, coords, mode="nearest", align_corners=align_corners
        ).squeeze(0).squeeze(0)
        refs.append(labels_ref)
    labels_ref = torch.stack(refs, dim=0)  # [M, K]

    assert torch.equal(labels_helper, labels_ref), (
        f"point_sample_labelmap disagrees with point_sample(mode='nearest') "
        f"on a non-symmetric labelmap (align_corners={align_corners})."
    )


def test_compute_unmasked_ratio_all_valid_and_partial():
    device = torch.device("cpu")

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


# ----------------------------------------------------------------------------- #
# sample_box_points_from_boxes
# ----------------------------------------------------------------------------- #


def test_sample_box_points_from_boxes_xyzxyz_noise_zero_matches_corners():
    # With noise=0, output points must equal the box corners exactly (clamped
    # to image bounds). Labels must be [top_left_label, bottom_right_label].
    device = torch.device("cpu")
    Z, Y, X = 10, 20, 30
    boxes = torch.tensor(
        [[1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
         [10.0, 11.0, 5.0, 12.0, 13.0, 8.0]],
        device=device,
    )
    pts, lbls = sample_box_points_from_boxes(
        boxes=boxes, box_format="xyzxyz", image_shape=(Z, Y, X), noise=0.0,
    )
    assert pts.shape == (2, 2, 3)
    assert lbls.shape == (2, 2)
    # Row 0: top-left (1, 2, 3), bottom-right (5, 6, 7)
    assert torch.equal(
        pts[0], torch.tensor([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
    )
    assert torch.equal(lbls[0], torch.tensor([2, 3], dtype=torch.int32))


def test_sample_box_points_from_boxes_zyxzyx_format_conversion():
    # zyxzyx input must map to identical xyz output as the xyzxyz input case.
    device = torch.device("cpu")
    Z, Y, X = 10, 20, 30
    boxes_zyx = torch.tensor(
        [[3.0, 2.0, 1.0, 7.0, 6.0, 5.0]],  # (z1, y1, x1, z2, y2, x2)
        device=device,
    )
    boxes_xyz = torch.tensor(
        [[1.0, 2.0, 3.0, 5.0, 6.0, 7.0]],  # (x1, y1, z1, x2, y2, z2)
        device=device,
    )
    pts_zyx, _ = sample_box_points_from_boxes(
        boxes=boxes_zyx, box_format="zyxzyx", image_shape=(Z, Y, X), noise=0.0,
    )
    pts_xyz, _ = sample_box_points_from_boxes(
        boxes=boxes_xyz, box_format="xyzxyz", image_shape=(Z, Y, X), noise=0.0,
    )
    assert torch.equal(pts_zyx, pts_xyz)


def test_sample_box_points_from_boxes_cxcyczwhd_format_conversion():
    # cxcyczwhd input must map to identical xyz output as the xyzxyz input case.
    device = torch.device("cpu")
    Z, Y, X = 10, 20, 30
    # Box centered at (3, 4, 5) (xyz) with size (2, 4, 6) (whd)
    # -> x1=2, y1=2, z1=2, x2=4, y2=6, z2=8
    boxes_c = torch.tensor([[3.0, 4.0, 5.0, 2.0, 4.0, 6.0]], device=device)
    boxes_x = torch.tensor([[2.0, 2.0, 2.0, 4.0, 6.0, 8.0]], device=device)
    pts_c, _ = sample_box_points_from_boxes(
        boxes=boxes_c, box_format="cxcyczwhd", image_shape=(Z, Y, X), noise=0.0,
    )
    pts_x, _ = sample_box_points_from_boxes(
        boxes=boxes_x, box_format="xyzxyz", image_shape=(Z, Y, X), noise=0.0,
    )
    assert torch.equal(pts_c, pts_x)


def test_sample_box_points_from_boxes_clamps_to_image_bounds():
    # Boxes outside the image must be clamped to [0, dim-1].
    device = torch.device("cpu")
    Z, Y, X = 10, 20, 30
    boxes = torch.tensor(
        [[-5.0, -5.0, -5.0, 100.0, 100.0, 100.0]],
        device=device,
    )
    pts, _ = sample_box_points_from_boxes(
        boxes=boxes, box_format="xyzxyz", image_shape=(Z, Y, X), noise=0.0,
    )
    assert torch.equal(pts[0, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.equal(pts[0, 1], torch.tensor([X - 1.0, Y - 1.0, Z - 1.0]))


def test_sample_box_points_from_boxes_valid_zeros_pad_rows():
    # Pad rows (valid=False) must get all-zero points and label 0.
    device = torch.device("cpu")
    Z, Y, X = 10, 20, 30
    boxes = torch.tensor(
        [[1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
         [10.0, 11.0, 5.0, 12.0, 13.0, 8.0]],
        device=device,
    )
    valid = torch.tensor([True, False])
    pts, lbls = sample_box_points_from_boxes(
        boxes=boxes, box_format="xyzxyz", image_shape=(Z, Y, X), noise=0.0,
        valid=valid,
    )
    assert torch.equal(lbls[0], torch.tensor([2, 3], dtype=torch.int32))
    assert torch.equal(lbls[1], torch.tensor([0, 0], dtype=torch.int32))
    assert torch.all(pts[1] == 0.0)
    assert not torch.all(pts[0] == 0.0)


def test_sample_box_points_from_boxes_noise_bounded():
    # With noise>0, perturbed corners must still lie inside the image and
    # within noise_bound pixels of the original (per-axis).
    torch.manual_seed(0)
    device = torch.device("cpu")
    Z, Y, X = 50, 100, 200
    boxes = torch.tensor(
        [[10.0, 20.0, 30.0, 40.0, 50.0, 45.0]] * 10,
        device=device,
    )
    pts, _ = sample_box_points_from_boxes(
        boxes=boxes, box_format="xyzxyz", image_shape=(Z, Y, X),
        noise=0.1, noise_bound=20,
    )
    assert torch.all(pts[..., 0] >= 0) and torch.all(pts[..., 0] <= X - 1)
    assert torch.all(pts[..., 1] >= 0) and torch.all(pts[..., 1] <= Y - 1)
    assert torch.all(pts[..., 2] >= 0) and torch.all(pts[..., 2] <= Z - 1)


def test_sample_box_points_from_boxes_rejects_bad_shape():
    boxes = torch.zeros((3, 4))  # last dim should be 6
    with pytest.raises(ValueError):
        sample_box_points_from_boxes(
            boxes=boxes, box_format="xyzxyz", image_shape=(4, 5, 6),
        )


# ----------------------------------------------------------------------------- #
# sample_box_points (mask -> noised corner prompts)
# ----------------------------------------------------------------------------- #


def test_sample_box_points_noise_keeps_batch_shape_and_bounds():
    """Each of the B boxes gets its own 2 corners (no [B,B,6] broadcast), every
    corner moves at most min(noise*extent, noise_bound) per axis from the
    half-open xyzxyz corner, and stays inside the image (0..size-1)."""
    torch.manual_seed(0)
    masks = torch.zeros(3, 1, 8, 8, 8)
    masks[0, 0, 1:4, 2:5, 3:6] = 1      # corners (x,y,z) = (3,2,1) .. (6,5,4), extent 3
    masks[1, 0, 0:2, 0:2, 0:2] = 1      # (0,0,0) .. (2,2,2), extent 2
    masks[2, 0, 4:8, 4:8, 4:8] = 1      # (4,4,4) .. (8,8,8), extent 4
    coords, labels = sample_box_points(
        input_fmt="ZYXC", masks=masks, noise=0.9, noise_bound=2
    )
    assert coords.shape == (3, 2, 3) and labels.shape == (3, 2)
    assert torch.equal(labels, torch.tensor([[2, 3]] * 3, dtype=labels.dtype))

    corners = torch.tensor([
        [[3., 2., 1.], [6., 5., 4.]],
        [[0., 0., 0.], [2., 2., 2.]],
        [[4., 4., 4.], [8., 8., 8.]],
    ])
    # min(0.9 * extent, 2): 2.7->2, 1.8, 3.6->2
    max_delta = torch.tensor([2.0, 1.8, 2.0]).view(3, 1, 1)
    lo = (corners - max_delta).clamp(0, 7)
    hi = (corners + max_delta).clamp(0, 7)
    assert torch.all(coords >= lo - 1e-6) and torch.all(coords <= hi + 1e-6), (coords, lo, hi)
    # noise was actually applied (seeded, so deterministic)
    assert not torch.equal(coords, corners.clamp(0, 7))


# ----------------------------------------------------------------------------- #
# sample_random_points_from_errors
# ----------------------------------------------------------------------------- #


def _sample_from_errors(gt, pred, num_pt=8):
    return sample_random_points_from_errors("ZYXC", gt, pred, num_pt=num_pt)


def test_sample_random_points_from_errors_fp_only_gives_negative_clicks():
    """With only a false-positive voxel, every click is negative and sits on it (x, y, z)."""
    gt = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)
    pred = torch.zeros_like(gt)
    pred[0, 0, 1, 2, 3] = True  # single FP voxel
    points, labels = _sample_from_errors(gt, pred)
    assert (labels == 0).all()
    assert (points == torch.tensor([3.0, 2.0, 1.0])).all()


def test_sample_random_points_from_errors_fn_only_gives_positive_clicks():
    """With only a false-negative voxel, every click is positive and sits on it."""
    gt = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)
    gt[0, 0, 2, 1, 0] = True  # single FN voxel (pred empty)
    pred = torch.zeros_like(gt)
    points, labels = _sample_from_errors(gt, pred)
    assert (labels == 1).all()
    assert (points == torch.tensor([0.0, 1.0, 2.0])).all()


def test_sample_random_points_from_errors_no_errors_samples_background():
    """A perfect prediction yields negative clicks drawn from the background."""
    gt = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)
    gt[0, 0, :2] = True
    pred = gt.clone()
    points, labels = _sample_from_errors(gt, pred, num_pt=16)
    assert (labels == 0).all()
    for p in points[0]:
        x, y, z = (int(v) for v in p)
        assert not gt[0, 0, z, y, x]


def test_sample_random_points_from_errors_mixed_points_land_in_their_pools():
    """Positive clicks land on FN voxels, negative clicks on FP voxels, and with
    one voxel in each pool neither label monopolizes the sample."""
    torch.manual_seed(0)
    gt = torch.zeros(2, 1, 4, 4, 4, dtype=torch.bool)
    pred = torch.zeros_like(gt)
    gt[:, 0, 0, 0, 0] = True                    # FN (pred misses it)
    pred[:, 0, 3, 3, 3] = True                  # FP
    points, labels = _sample_from_errors(gt, pred, num_pt=32)
    assert points.shape == (2, 32, 3) and labels.shape == (2, 32)
    for b in range(2):
        for p, l in zip(points[b], labels[b]):
            x, y, z = (int(v) for v in p)
            if l == 1:
                assert gt[b, 0, z, y, x] and not pred[b, 0, z, y, x]
            else:
                assert pred[b, 0, z, y, x] and not gt[b, 0, z, y, x]
    assert 0 < int(labels.sum()) < labels.numel()
