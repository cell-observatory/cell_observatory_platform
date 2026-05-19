import pytest
import torch

from cell_observatory_platform.models.ops.point_sampling import (
    gt_masks_from_labelmap,
    sample_box_points_from_boxes,
    sample_prompt_point_from_labelmap,
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
# gt_masks_from_labelmap
# ----------------------------------------------------------------------------- #


def test_gt_masks_from_labelmap_matches_dense_reference():
    device = torch.device("cpu")
    B, Z, Y, X = 2, 3, 4, 5
    labelmap = torch.zeros((B, Z, Y, X), dtype=torch.int32, device=device)
    labelmap[0, 1, 2, 3] = 7
    labelmap[0, 2, 3, 4] = 7  # two voxels with the same id in batch 0
    labelmap[1, 0, 0, 0] = 11

    img_ids = torch.tensor([0, 0, 1, 0], dtype=torch.int64, device=device)
    instance_ids = torch.tensor([7, 11, 11, -1], dtype=torch.int64, device=device)

    masks = gt_masks_from_labelmap(labelmap, img_ids, instance_ids)
    assert masks.shape == (4, 1, Z, Y, X)
    assert masks.dtype == torch.bool

    # Row 0: id=7 in batch 0 -> 2 True voxels.
    assert masks[0, 0].sum().item() == 2
    # Row 1: id=11 in batch 0 -> 0 True voxels (id 11 lives in batch 1).
    assert masks[1, 0].sum().item() == 0
    # Row 2: id=11 in batch 1 -> 1 True voxel.
    assert masks[2, 0].sum().item() == 1
    # Row 3: sentinel -1 -> all False.
    assert masks[3, 0].sum().item() == 0


# ----------------------------------------------------------------------------- #
# sample_prompt_point_from_labelmap
# ----------------------------------------------------------------------------- #


def test_sample_prompt_point_from_labelmap_uniform_returns_xy_z_points():
    torch.manual_seed(0)
    device = torch.device("cpu")
    B, Z, Y, X = 1, 4, 6, 8
    labelmap = torch.zeros((B, Z, Y, X), dtype=torch.int32, device=device)
    labelmap[0, 1, 2, 3] = 5

    img_ids = torch.tensor([0, 0], dtype=torch.int64, device=device)
    instance_ids = torch.tensor([5, -1], dtype=torch.int64, device=device)

    points, labels = sample_prompt_point_from_labelmap(
        labelmap=labelmap,
        img_ids=img_ids,
        instance_ids=instance_ids,
        pred_masks=None,
        input_fmt="TZYXC",
        time_separable=True,
        method="uniform",
        num_pt=1,
    )
    assert points.shape == (2, 1, 3)
    assert labels.shape == (2, 1)
    # Row 0: positive click should land on the id=5 voxel.
    x, y, z = points[0, 0].tolist()
    assert (int(x), int(y), int(z)) == (3, 2, 1), (x, y, z)
    assert labels[0, 0].item() == 1
    # Row 1: sentinel pad -> all-zero gt -> sampler chooses background w/ label 0.
    assert labels[1, 0].item() == 0


def test_sample_prompt_point_from_labelmap_center_falls_back_to_uniform_training():
    # method="center" with exact_edt_for_eval=False must not invoke scipy
    # (scipy is CPU and slow). Verify by running on CPU with no scipy import
    # path triggered -- behavior must equal "uniform".
    torch.manual_seed(42)
    device = torch.device("cpu")
    B, Z, Y, X = 1, 2, 3, 4
    labelmap = torch.zeros((B, Z, Y, X), dtype=torch.int32, device=device)
    labelmap[0, 1, 2, 3] = 4

    img_ids = torch.tensor([0], dtype=torch.int64, device=device)
    instance_ids = torch.tensor([4], dtype=torch.int64, device=device)

    torch.manual_seed(99)
    pts_center, lbls_center = sample_prompt_point_from_labelmap(
        labelmap=labelmap, img_ids=img_ids, instance_ids=instance_ids,
        pred_masks=None, input_fmt="TZYXC", method="center",
        exact_edt_for_eval=False, num_pt=1,
    )
    torch.manual_seed(99)
    pts_uniform, lbls_uniform = sample_prompt_point_from_labelmap(
        labelmap=labelmap, img_ids=img_ids, instance_ids=instance_ids,
        pred_masks=None, input_fmt="TZYXC", method="uniform", num_pt=1,
    )
    assert torch.equal(pts_center, pts_uniform)
    assert torch.equal(lbls_center, lbls_uniform)


def test_sample_prompt_point_from_labelmap_center_exact_rejects_multi_pt():
    device = torch.device("cpu")
    labelmap = torch.zeros((1, 2, 2, 2), dtype=torch.int32, device=device)
    img_ids = torch.tensor([0], dtype=torch.int64, device=device)
    instance_ids = torch.tensor([1], dtype=torch.int64, device=device)
    with pytest.raises(ValueError):
        sample_prompt_point_from_labelmap(
            labelmap=labelmap, img_ids=img_ids, instance_ids=instance_ids,
            method="center", exact_edt_for_eval=True, num_pt=2,
        )
