"""`Crop`: window selection, declared-target walk, box adjustment, image_sizes
bookkeeping, dtype preservation."""

import pytest
import torch

from cell_observatory_platform.data.transforms.crop import Crop


_DENSE_3D = {"kind": "dense", "layout": "ZYXC", "role": "input", "has_time": False}


def _crop_sample(data, data_types=None, targets=None, **meta):
    """Wrap a tensor in the dict contract Crop consumes.

    `data_types` defaults to a single dense ZYXC input; targets (Form S, a
    per-sample list of dicts) and any other metainfo keys are optional.
    """
    m = {"data_types": data_types if data_types is not None else {"data_tensor": _DENSE_3D}, **meta}
    if targets is not None:
        m["targets"] = targets
    return {"data_tensor": data, "metainfo": m}


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------


def test_center_crop_takes_the_centered_window():
    """crop_type="center" takes the window starting at (full - crop) // 2 on every
    cropped axis; with no prior image_sizes the whole crop is valid content."""
    c = Crop(target_spatial_shape=(2, 4, 4), crop_dims="ZYX", crop_type="center")
    x = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(1, 4, 8, 8, 1)
    out = c(_crop_sample(x, {"data_tensor": _DENSE_3D}))
    assert torch.equal(out["data_tensor"], x[:, 1:3, 2:6, 2:6, :])   # offsets (4-2)//2, (8-4)//2, (8-4)//2
    assert out["metainfo"]["image_sizes"].tolist() == [[2, 4, 4]]    # no prior sizes -> all valid


def test_yx_crop_dims_leave_z_alone():
    """Axes not named in crop_dims keep their full extent."""
    c = Crop(target_spatial_shape=(4, 4, 4), crop_dims="YX", crop_type="start")
    x = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(1, 4, 8, 8, 1)
    out = c(_crop_sample(x, {"data_tensor": _DENSE_3D}))
    assert torch.equal(out["data_tensor"], x[:, :, :4, :4, :])


# ---------------------------------------------------------------------------
# Declared-target walk (data_types drives which fields are cropped and how)
# ---------------------------------------------------------------------------


def test_crops_declared_semantic_maps_in_lockstep_with_image():
    """A declared semantic_masks stack (N, Z, Y, X) is cropped with the same
    window as the image and keeps its integer dtype."""
    c = Crop(
        target_spatial_shape=(2, 4, 4),
        crop_dims="ZYX",
        crop_type="start",
        dtype="float32",
    )
    x = torch.arange(4 * 8 * 8, dtype=torch.float32).reshape(1, 4, 8, 8, 1)
    stack = torch.arange(2 * 4 * 8 * 8, dtype=torch.int32).reshape(2, 4, 8, 8)
    out = c(
        _crop_sample(
            x,
            {
                "data_tensor": _DENSE_3D,
                "semantic_maps": {"kind": "semantic_masks", "layout": "ZYXC",
                                  "role": "target"},
            },
            targets=[{"semantic_maps": stack}],
        )
    )
    got = out["metainfo"]["targets"][0]["semantic_maps"]
    assert got.shape == (2, 2, 4, 4)
    assert got.dtype == torch.int32
    assert torch.equal(got, stack[:, :2, :4, :4])
    # image cropped identically
    assert torch.equal(out["data_tensor"], x[:, :2, :4, :4, :])


def test_crops_label_map_and_filters_boxes_with_aligned_fields():
    """The label map is windowed; boxes fully outside the crop are dropped and the
    per-instance fields aligned with them (mask_ids, labels) are filtered by the
    same validity mask."""
    c = Crop(
        target_spatial_shape=(2, 4, 4),
        crop_dims="ZYX",
        crop_type="start",
        bbox_format="zyxzyx",
        dtype="float32",
    )
    x = torch.zeros(1, 4, 8, 8, 1)
    lm = torch.arange(4 * 8 * 8, dtype=torch.int32).reshape(4, 8, 8)
    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 2.0, 3.0, 3.0],   # inside crop
            [3.0, 5.0, 5.0, 4.0, 8.0, 8.0],   # fully outside -> filtered
        ]
    )
    out = c(
        _crop_sample(
            x,
            {
                "data_tensor": _DENSE_3D,
                "label_map": {"kind": "instance_masks", "layout": "ZYXC",
                              "role": "target"},
                "boxes": {"kind": "boxes", "layout": "zyxzyx", "role": "target"},
            },
            targets=[{
                "label_map": lm,
                "boxes": boxes,
                "mask_ids": torch.tensor([7, 9]),
                "labels": torch.tensor([0, 1]),
            }],
        )
    )
    tgt = out["metainfo"]["targets"][0]
    assert torch.equal(tgt["label_map"], lm[:2, :4, :4])
    assert tgt["boxes"].shape == (1, 6)
    # aligned per-instance fields filtered with the same valid mask
    assert tgt["mask_ids"].tolist() == [7]
    assert tgt["labels"].tolist() == [0]


def test_shifts_cxcyczwhd_boxes_by_crop_offset():
    """cxcyczwhd boxes are shifted by the crop offset (zero for a start crop, so
    a box inside the window is returned unchanged)."""
    c = Crop(
        target_spatial_shape=(2, 4, 4),
        crop_dims="ZYX",
        crop_type="start",
        bbox_format="cxcyczwhd",
        dtype="float32",
    )
    x = torch.zeros(1, 4, 8, 8, 1)
    # Box centered at (x=2, y=2, z=1), size (2, 2, 2): inside the (2,4,4) crop.
    boxes = torch.tensor([[2.0, 2.0, 1.0, 2.0, 2.0, 2.0]])
    out = c(
        _crop_sample(
            x,
            {
                "data_tensor": _DENSE_3D,
                "boxes": {"kind": "boxes", "layout": "cxcyczwhd", "role": "target"},
            },
            targets=[{"boxes": boxes}],
        )
    )
    got = out["metainfo"]["targets"][0]["boxes"]
    assert torch.allclose(got, boxes)


def test_renormalizes_normalized_boxes_to_crop_window():
    """boxes_normalized=True: coordinates normalized to the pre-crop base are
    renormalized to the crop window."""
    c = Crop(
        target_spatial_shape=(2, 4, 4),
        crop_dims="ZYX",
        crop_type="start",
        bbox_format="zyxzyx",
        boxes_normalized=True,
        dtype="float32",
    )
    x = torch.zeros(1, 4, 8, 8, 1)
    # In absolute voxels: (z 0-2, y 0-4, x 0-4) on an (4, 8, 8) base.
    boxes = torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]])
    out = c(
        _crop_sample(
            x,
            {
                "data_tensor": _DENSE_3D,
                "boxes": {"kind": "boxes", "layout": "zyxzyx", "role": "target"},
            },
            targets=[{"boxes": boxes}],
        )
    )
    got = out["metainfo"]["targets"][0]["boxes"]
    # Absolute (0,0,0)-(2,4,4) inside the (2,4,4) crop -> fills it -> all 1.0.
    assert torch.allclose(got, torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]))


# ---------------------------------------------------------------------------
# image_sizes bookkeeping
# ---------------------------------------------------------------------------


def test_image_sizes_keep_trailing_padding_after_crop():
    """image_sizes becomes min(old_valid - offset, crop) per axis, not the crop
    size: a crop window can retain trailing buffer padding."""
    c = Crop(
        target_spatial_shape=(4, 8, 8),
        crop_dims="ZYX",
        crop_type="center",
        dtype="float32",
    )
    x = torch.zeros(1, 8, 16, 16, 1)
    # center crop: offsets ((8-4)//2, (16-8)//2, (16-8)//2) = (2, 4, 4)
    image_sizes = torch.tensor([[5, 12, 9]])  # valid content per axis
    out = c(
        _crop_sample(
            x,
            {"data_tensor": _DENSE_3D},
            targets=[],
            image_sizes=image_sizes,
        )
    )
    # min(5-2, 4)=3, min(12-4, 8)=8, min(9-4, 8)=5
    assert out["metainfo"]["image_sizes"].tolist() == [[3, 8, 5]]


def test_image_sizes_scale_with_fallback_resize():
    """An input smaller than the target forces the resize fallback; the valid
    extent scales by final / crop on that axis."""
    c = Crop(
        target_spatial_shape=(8, 4, 4),
        crop_dims="ZYX",
        crop_type="start",
        dtype="float32",
    )
    x = torch.zeros(1, 4, 8, 8, 1)     # Z=4 < target 8 -> crop (4,4,4) then resize
    image_sizes = torch.tensor([[2, 8, 8]])
    out = c(
        _crop_sample(
            x,
            {"data_tensor": _DENSE_3D},
            targets=[],
            image_sizes=image_sizes,
        )
    )
    # crop = (4,4,4), offsets 0; valid after crop = (2,4,4); resize to
    # (8,4,4) scales Z by 2 -> (4,4,4).
    assert out["data_tensor"].shape[1:4] == (8, 4, 4)
    assert out["metainfo"]["image_sizes"].tolist() == [[4, 4, 4]]


# ---------------------------------------------------------------------------
# dtype preservation (the preprocessor's float32 count intermediate)
# ---------------------------------------------------------------------------


def test_crop_keeps_float32_input_dtype():
    """The configured dtype does not narrow the dense tensor: a float32 input stays
    float32 through a pure crop."""
    c = Crop(target_spatial_shape=(2, 4, 4), crop_dims="ZYX",
             crop_type="start", dtype="bfloat16")
    out = c(_crop_sample(torch.rand(1, 4, 8, 8, 1, dtype=torch.float32)))
    assert out["data_tensor"].dtype == torch.float32


def test_crop_resize_keeps_float32_input_dtype():
    """The crop_resize mode also preserves the incoming float32 dtype."""
    c = Crop(target_spatial_shape=(8, 8, 8), crop_dims="ZYX",
             crop_type="start", dtype="bfloat16",
             mode_probs={"crop_resize": 1.0})
    out = c(_crop_sample(torch.rand(1, 4, 4, 4, 1, dtype=torch.float32)))
    assert out["data_tensor"].dtype == torch.float32
    assert tuple(out["data_tensor"].shape[1:4]) == (8, 8, 8)


def test_crop_keeps_counts_bit_exact():
    """Counts that bf16 cannot represent survive the transform bit-exactly."""
    x = torch.full((1, 4, 8, 8, 1), 40001.0, dtype=torch.float32)
    c = Crop(target_spatial_shape=(2, 4, 4), crop_dims="ZYX",
             crop_type="start", dtype="bfloat16")
    out = c(_crop_sample(x))
    assert (out["data_tensor"] == 40001.0).all()


# ---------------------------------------------------------------------------
# Fail-loud on non-Form-S targets
# ---------------------------------------------------------------------------


def test_rejects_form_d_targets():
    """A Form-D role dict (DeepCopyInputsAsTargets clones) cannot be warped
    coherently and is rejected with a TypeError naming the form."""
    c = Crop(target_spatial_shape=(2, 4, 4), crop_dims="ZYX", crop_type="start")
    s = _crop_sample(torch.rand(1, 4, 8, 8, 1),
                     targets={"denoising": torch.rand(1, 4, 8, 8, 1)})
    with pytest.raises(TypeError, match="Form-D role dict"):
        c(s)


def test_rejects_non_dict_target_elements():
    """Form-S elements must be dicts; a bare tensor element is rejected."""
    c = Crop(target_spatial_shape=(2, 4, 4), crop_dims="ZYX", crop_type="start")
    s = _crop_sample(torch.rand(1, 4, 8, 8, 1), targets=[torch.rand(4, 8, 8, 1)])
    with pytest.raises(TypeError, match="Form-S"):
        c(s)
