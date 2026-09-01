import pytest

import torch
import torch.nn.functional as F

# NOTE: ChannelDropout (data/transforms/channel_dropout.py) and MultiCrop3D
# (data/transforms/multicrop.py) are commented out at the source; crop.py exports
# `Crop`, not `Crop3D`. TestChannelDropout / TestCrop3D / TestMultiCrop3D below are
# commented out to match. Re-enable together with the source modules.
# from cell_observatory_platform.data.transforms.channel_dropout import ChannelDropout
# from cell_observatory_platform.data.transforms.crop import Crop3D
# from cell_observatory_platform.data.transforms.multicrop import MultiCrop3D
from cell_observatory_platform.data.transforms.resize import Resize
from cell_observatory_platform.data.transforms.probabilistic_choice import ProbabilisticChoice
from cell_observatory_platform.data.transforms.utils import (

    parse_target_shape_range,
    resize_boxes,
    resize_label_map,
    resize_masks,
    resize_tensor_3d,
    sample_target_shape,
    stack_metainfo,
)


def _sample(data: torch.Tensor, has_time: bool = False, **meta) -> dict:
    """Wrap a tensor in the dict contract the transforms now require.

    Layout is no longer a constructor arg -- transforms read
    metainfo["data_types"]["data_tensor"]["has_time"] to dispatch 3D vs 4D.
    Mirrors what RayPreprocessor.forward injects before running transforms.
    """
    return {
        "data_tensor": data,
        "metainfo": {"data_types": {"data_tensor": {"has_time": has_time}}, **meta},
    }



# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestParseTargetShapeRange:
    def test_fixed(self):
        mn, mx, is_random = parse_target_shape_range((4, 8, 8))
        assert mn == (4, 8, 8)
        assert mx == (4, 8, 8)
        assert is_random is False

    def test_range(self):
        mn, mx, is_random = parse_target_shape_range(((2, 4, 4), (4, 8, 8)))
        assert mn == (2, 4, 4)
        assert mx == (4, 8, 8)
        assert is_random is True


def test_sample_target_shape_fixed():
    shape = sample_target_shape((4, 8, 8), (4, 8, 8))
    assert shape == (4, 8, 8)


def test_sample_target_shape_range():
    for _ in range(20):
        shape = sample_target_shape((2, 4, 4), (6, 12, 12))
        assert 2 <= shape[0] <= 6
        assert 4 <= shape[1] <= 12
        assert 4 <= shape[2] <= 12


def test_resize_tensor_3d_shape():
    x = torch.randn(2, 4, 8, 8, 3)
    out, factors = resize_tensor_3d(x, (2, 4, 4))
    assert out.shape == (2, 2, 4, 4, 3)
    assert abs(factors[0] - 0.5) < 1e-6
    assert abs(factors[1] - 0.5) < 1e-6
    assert abs(factors[2] - 0.5) < 1e-6


def test_resize_tensor_3d_rejects_nearest_mode():
    """Plain "nearest" is edge-aligned and would misregister image and GT; the
    error points at the supported "nearest-exact" mode."""
    x = torch.rand(1, 7, 9, 5, 2)
    with pytest.raises(ValueError, match="nearest-exact"):
        resize_tensor_3d(x, (3, 4, 2), mode="nearest")


class TestNearestExactResize:
    """resize_label_map / resize_masks: nearest-exact, id-preserving GT resampling."""

    def test_matches_f_interpolate_nearest_exact(self):
        lm = torch.randint(0, 50, (7, 9, 11), dtype=torch.int32)
        ref = (
            F.interpolate(
                lm.float().unsqueeze(0).unsqueeze(0),
                size=(3, 4, 5),
                mode="nearest-exact",
            )
            .squeeze(0)
            .squeeze(0)
            .to(torch.int32)
        )
        got = resize_label_map(lm, (3, 4, 5))
        assert got.dtype == torch.int32
        assert torch.equal(got, ref)

    def test_masks_keep_dtype_and_large_ids(self):
        """16777217 is not representable in float32: an index-gather path keeps
        such ids exact where a float round-trip would corrupt them."""
        big = 16_777_217
        stack = torch.zeros((1, 2, 2, 2), dtype=torch.int64)
        stack[0, 0, 0, 0] = big
        out = resize_masks(stack, (2, 2, 2))
        assert out.dtype == torch.int64
        assert out[0, 0, 0, 0].item() == big

    def test_bool_masks_supported(self):
        m = torch.zeros((2, 4, 4, 4), dtype=torch.bool)
        m[0, 0, 0, 0] = True
        out = resize_masks(m, (2, 2, 2))
        assert out.dtype == torch.bool
        assert out.shape == (2, 2, 2, 2)


def test_resize_boxes_zyxzyx():
    boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0, 6.0, 9.0]])
    out = resize_boxes(boxes, (2.0, 0.5, 1.0), "zyxzyx")
    expected = torch.tensor([[2.0, 1.0, 3.0, 8.0, 3.0, 9.0]])
    assert torch.allclose(out, expected)


def test_resize_boxes_cxcyczwhd():
    boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0, 6.0, 8.0]])
    out = resize_boxes(boxes, (2.0, 0.5, 3.0), "cxcyczwhd")
    expected = torch.tensor([[3.0, 1.0, 6.0, 12.0, 3.0, 16.0]])
    assert torch.allclose(out, expected)


class TestStackMetainfo:
    def test_empty(self):
        assert stack_metainfo([]) == {}

    def test_lists_concat(self):
        m1 = {"targets": [{"id": 1}]}
        m2 = {"targets": [{"id": 2}]}
        merged = stack_metainfo([m1, m2])
        assert len(merged["targets"]) == 2

    def test_tensors_cat(self):
        m1 = {"sizes": torch.tensor([[1, 2, 3]])}
        m2 = {"sizes": torch.tensor([[4, 5, 6]])}
        merged = stack_metainfo([m1, m2])
        assert merged["sizes"].shape == (2, 3)

    def test_scalars_to_list(self):
        m1 = {"mode": "crop"}
        m2 = {"mode": "resize"}
        merged = stack_metainfo([m1, m2])
        assert merged["mode"] == ["crop", "resize"]


# ---------------------------------------------------------------------------
# ChannelDropout
# ---------------------------------------------------------------------------


# class TestChannelDropout:
#     def test_tensor_input(self):
#         cd = ChannelDropout(p_apply=1.0, keep_ratio=0.5, min_keep=1, seed=42)
#         x = torch.randn(2, 4, 8, 8, 6)
#         out = cd(_sample(x))["data_tensor"]
#         assert out.ndim == 5
#         assert out.shape[-1] <= 6
#         assert out.shape[-1] >= 1

#     def test_dict_input(self):
#         cd = ChannelDropout(p_apply=1.0, keep_ratio=(0.3, 0.7), min_keep=1, seed=42)
#         data = {"data_tensor": torch.randn(2, 4, 8, 8, 10)}
#         out = cd(data)
#         assert "data_tensor" in out
#         assert "metainfo" in out
#         info = out["metainfo"]["channel_dropout"]
#         assert info["applied"] is True
#         assert info["C_out"] <= 10

#     def test_no_apply(self):
#         cd = ChannelDropout(p_apply=0.0, keep_ratio=0.5, seed=42)
#         data = {"data_tensor": torch.randn(1, 4, 8, 8, 5)}
#         out = cd(data)
#         assert out["data_tensor"].shape[-1] == 5

#     def test_6d_input(self):
#         cd = ChannelDropout(p_apply=1.0, keep_ratio=0.5, min_keep=1, seed=42)
#         x = torch.randn(2, 3, 4, 8, 8, 6)
#         out = cd(_sample(x))["data_tensor"]
#         assert out.ndim == 6
#         assert out.shape[-1] >= 1


# # ---------------------------------------------------------------------------
# # Resize
# # ---------------------------------------------------------------------------


class TestResize:
    def test_tensor_input(self):
        r = Resize(target_spatial_shape=(2, 4, 4))
        x = torch.randn(1, 4, 8, 8, 2)
        out = r(_sample(x))["data_tensor"]
        assert out.shape == (1, 2, 4, 4, 2)

    def test_dict_input(self):
        r = Resize(target_spatial_shape=(2, 4, 4))
        data = {
            **_sample(torch.randn(1, 4, 8, 8, 2)),
        }
        out = r(data)
        assert out["data_tensor"].shape == (1, 2, 4, 4, 2)

    def test_dict_with_targets(self):
        """Which target fields get resized (and how) is driven by data_types:
        stacked semantic masks and the instance label map are nearest-exact
        resampled (dtype kept), boxes are scaled by target / source per axis."""
        r = Resize(target_spatial_shape=(2, 4, 4), bbox_format="zyxzyx")
        masks = torch.ones(3, 4, 8, 8, dtype=torch.bool)
        label_map = torch.randint(0, 5, (4, 8, 8), dtype=torch.int32)
        boxes = torch.tensor([[0.0, 2.0, 4.0, 4.0, 8.0, 8.0], [1.0, 1.0, 1.0, 3.0, 5.0, 7.0]])
        data = {
            "data_tensor": torch.randn(1, 4, 8, 8, 2),
            "metainfo": {
                "data_types": {
                    "data_tensor": {"kind": "dense", "layout": "ZYXC", "role": "input", "has_time": False},
                    "masks": {"kind": "semantic_masks", "layout": "ZYXC", "role": "target"},
                    "label_map": {"kind": "instance_masks", "layout": "ZYXC", "role": "target"},
                    "boxes": {"kind": "boxes", "layout": "zyxzyx", "role": "target"},
                },
                "targets": [{"masks": masks, "label_map": label_map, "boxes": boxes}],
            },
        }
        tgt = r(data)["metainfo"]["targets"][0]
        assert tgt["masks"].shape == (3, 2, 4, 4) and tgt["masks"].dtype == torch.bool and tgt["masks"].all()
        ref = F.interpolate(label_map[None, None].float(), size=(2, 4, 4), mode="nearest-exact")[0, 0].to(torch.int32)
        assert torch.equal(tgt["label_map"], ref)
        torch.testing.assert_close(tgt["boxes"], boxes * 0.5)            # every axis halves

    def test_random_range(self):
        r = Resize(target_spatial_shape=((2, 4, 4), (4, 8, 8)))
        x = torch.randn(1, 4, 8, 8, 2)
        out = r(_sample(x))["data_tensor"]
        assert 2 <= out.shape[1] <= 4
        assert 4 <= out.shape[2] <= 8

    def test_tzyxc_resizes_each_frame_independently(self):
        """TZYXC folds T into the batch: every frame is resized exactly as the
        corresponding single-frame ZYXC call would."""
        r = Resize(target_spatial_shape=(2, 4, 4))
        x = torch.rand(1, 3, 4, 8, 8, 2)                                   # (B, T, Z, Y, X, C)
        out = r(_sample(x, has_time=True))["data_tensor"]
        assert out.shape == (1, 3, 2, 4, 4, 2)
        for t in range(3):
            ref = F.interpolate(x[:, t].permute(0, 4, 1, 2, 3), size=(2, 4, 4),
                                mode="trilinear", align_corners=False).permute(0, 2, 3, 4, 1)
            torch.testing.assert_close(out[:, t], ref)

    def test_crop_to_valid_resizes_only_the_content_region(self):
        """crop_to_valid (default) resizes the valid region only, so trailing
        buffer padding is never interpolated into the content; image_sizes and
        padding_mask then describe a fully valid output. With crop_to_valid=False
        the full buffer (padding included) is resized."""
        x = torch.rand(1, 4, 8, 8, 1)
        x[:, :, 4:, :, :] = 0                                              # trailing Y padding
        pm = torch.zeros(1, 4, 8, 8, dtype=torch.bool)
        pm[:, :, 4:, :] = True
        sizes = torch.tensor([[4, 4, 8]])
        out = Resize(target_spatial_shape=(4, 8, 8))(_sample(x.clone(), image_sizes=sizes, padding_mask=pm.clone()))
        ref = F.interpolate(x[:, :, :4, :, :].permute(0, 4, 1, 2, 3), size=(4, 8, 8),
                            mode="trilinear", align_corners=False).permute(0, 2, 3, 4, 1)
        torch.testing.assert_close(out["data_tensor"], ref)               # padding never interpolated in
        assert out["metainfo"]["image_sizes"].tolist() == [[4, 8, 8]]
        assert not out["metainfo"]["padding_mask"].any()
        full = Resize(target_spatial_shape=(4, 8, 8), crop_to_valid=False)(_sample(x.clone(), image_sizes=sizes))
        assert (full["data_tensor"][:, :, 4:, :, :] == 0).all()           # without the crop, padding stays

    def test_normalized_boxes_renormalized_from_buffer_to_valid(self):
        """boxes_normalized=True: crop_to_valid changes the normalization base
        from the buffer to the valid region, so boxes scale by buffer / valid
        per axis (the resize itself leaves normalized coords invariant)."""
        r = Resize(target_spatial_shape=(4, 4, 4), bbox_format="zyxzyx", boxes_normalized=True)
        boxes = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.25, 0.5]])            # normalized to the (4, 8, 8) buffer
        data = {"data_tensor": torch.zeros(1, 4, 8, 8, 1), "metainfo": {
            "data_types": {
                "data_tensor": {"kind": "dense", "layout": "ZYXC", "role": "input", "has_time": False},
                "boxes": {"kind": "boxes", "layout": "zyxzyx", "role": "target"}},
            "image_sizes": torch.tensor([[4, 4, 4]]),                      # valid = half of Y and X
            "targets": [{"boxes": boxes}]}}
        got = r(data)["metainfo"]["targets"][0]["boxes"]
        torch.testing.assert_close(got, torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.5, 1.0]]))   # x (8/4) on Y, X

    def test_keeps_float32_input_dtype(self):
        """The configured dtype does not narrow the dense tensor: float32 in, float32 out."""
        r = Resize(target_spatial_shape=(2, 4, 4), dtype="bfloat16")
        out = r(_sample(torch.rand(1, 4, 8, 8, 1, dtype=torch.float32)))
        assert out["data_tensor"].dtype == torch.float32

    def test_keeps_label_map_int_dtype(self):
        """An integer label map keeps its dtype through the nearest-exact resize."""
        lm = torch.randint(0, 5, (1, 4, 8, 8), dtype=torch.int32)
        meta = {
            "data_types": {
                "data_tensor": {"has_time": False},
                "label_map": {"kind": "instance_masks", "layout": "ZYXC",
                              "role": "target"},
            },
            "targets": [{"label_map": lm[0]}],
        }
        r = Resize(target_spatial_shape=(2, 4, 4), dtype="bfloat16")
        out = r({"data_tensor": torch.rand(1, 4, 8, 8, 1), "metainfo": meta})
        assert out["metainfo"]["targets"][0]["label_map"].dtype == torch.int32

    def test_rejects_form_d_targets(self):
        """A Form-D role dict (DeepCopyInputsAsTargets clones) cannot be warped
        coherently and is rejected with a TypeError naming the form."""
        r = Resize(target_spatial_shape=(2, 4, 4))
        s = _sample(torch.rand(1, 4, 8, 8, 1),
                    targets={"denoising": torch.rand(1, 4, 8, 8, 1)})
        with pytest.raises(TypeError, match="Form-D role dict"):
            r(s)

    def test_padding_mask_resized_when_no_crop_ran(self):
        """When no crop-to-valid ran the full padded buffer was resized, so the
        padding mask is resized alongside it (provenance survives) rather than blanked."""
        r = Resize(target_spatial_shape=(4, 4, 4), mode="trilinear", align_corners=False)
        pm = torch.zeros(1, 8, 8, 8, dtype=torch.bool)
        pm[0, :, 4:, :] = True                       # half the buffer is padding
        out = r._resize_metainfo(
            {"padding_mask": pm}, [(0.5, 0.5, 0.5)], (4, 4, 4),
            [(8, 8, 8)], torch.zeros(1, 8, 8, 8, 1), False,
            cropped_to_valid=False,
        )
        out_pm = out["padding_mask"]
        assert out_pm.shape == (1, 4, 4, 4)
        assert out_pm.any(), "padding provenance must survive a full-buffer resize"
        assert out_pm[0, :, 2:, :].all() and not out_pm[0, :, :2, :].any()

    def test_padding_mask_blanked_after_crop_to_valid(self):
        """When crop-to-valid removed the padding before the resize, the content
        fills the target shape and the mask is reset to all-valid."""
        r = Resize(target_spatial_shape=(4, 4, 4), mode="trilinear", align_corners=False)
        pm = torch.zeros(1, 8, 8, 8, dtype=torch.bool)
        pm[0, :, 4:, :] = True
        out = r._resize_metainfo(
            {"padding_mask": pm}, [(0.5, 0.5, 0.5)], (4, 4, 4),
            [(8, 4, 8)], torch.zeros(1, 8, 8, 8, 1), False,
            cropped_to_valid=True,
        )
        assert not out["padding_mask"].any()
        assert out["padding_mask"].shape == (1, 4, 4, 4)


# ---------------------------------------------------------------------------
# Crop3D
# ---------------------------------------------------------------------------


# class TestCrop3D:
#     def test_center_crop(self):
#         c = Crop3D(
#             input_format="ZYXC",
#             target_spatial_shape=(4, 8, 8),
#             crop_dims="ZYX",
#             crop_type="center",
#         )
#         data = {"data_tensor": torch.randn(1, 8, 16, 16, 2)}
#         out = c(data)
#         assert out["data_tensor"].shape[1:4] == torch.Size([4, 8, 8])

#     def test_random_crop(self):
#         c = Crop3D(
#             input_format="ZYXC",
#             target_spatial_shape=(4, 8, 8),
#             crop_dims="ZYX",
#             crop_type="random",
#         )
#         data = {"data_tensor": torch.randn(1, 8, 16, 16, 2)}
#         out = c(data)
#         assert out["data_tensor"].shape[1:4] == torch.Size([4, 8, 8])

#     def test_yx_only_crop(self):
#         c = Crop3D(
#             input_format="ZYXC",
#             target_spatial_shape=(8, 8, 8),
#             crop_dims="YX",
#             crop_type="center",
#         )
#         data = {"data_tensor": torch.randn(1, 8, 16, 16, 2)}
#         out = c(data)
#         z_out = out["data_tensor"].shape[1]
#         assert z_out == 8  # Z untouched
#         assert out["data_tensor"].shape[2] == 8
#         assert out["data_tensor"].shape[3] == 8


# # ---------------------------------------------------------------------------
# # MultiCrop3D
# # ---------------------------------------------------------------------------


# class TestMultiCrop3D:
#     def test_two_streams(self):
#         global_crop = Crop3D(
#             input_format="ZYXC",
#             target_spatial_shape=(4, 8, 8),
#             crop_dims="ZYX",
#             crop_type="random",
#         )
#         local_crop = Crop3D(
#             input_format="ZYXC",
#             target_spatial_shape=(2, 4, 4),
#             crop_dims="ZYX",
#             crop_type="random",
#         )
#         mc = MultiCrop3D(
#             crop_transforms=[global_crop, local_crop],
#             names=["global", "local"],
#             counts=[2, 3],
#         )
#         B = 1
#         data = {"data_tensor": torch.randn(B, 8, 16, 16, 2), "metainfo": {}}
#         out = mc(data)
#         assert isinstance(out["data_tensor"], dict)
#         assert out["data_tensor"]["global"].shape[0] == B * 2
#         assert out["data_tensor"]["local"].shape[0] == B * 3
#         assert out["metainfo"]["global"]["n_crops"] == 2
#         assert out["metainfo"]["local"]["n_crops"] == 3


# # ---------------------------------------------------------------------------
# # ProbabilisticChoice
# # ---------------------------------------------------------------------------


class TestProbabilisticChoice:
    def test_always_pick_first(self):
        t1 = Resize(target_spatial_shape=(2, 4, 4))
        t2 = Resize(target_spatial_shape=(4, 8, 8))
        pc = ProbabilisticChoice(transforms=[t1, t2], probs=[1.0, 0.0])
        data = _sample(torch.randn(1, 4, 8, 8, 2))
        out = pc(data)
        assert out["data_tensor"].shape == (1, 2, 4, 4, 2)

    def test_always_pick_second(self):
        t1 = Resize(target_spatial_shape=(2, 4, 4))
        t2 = Resize(target_spatial_shape=(4, 8, 8))
        pc = ProbabilisticChoice(transforms=[t1, t2], probs=[0.0, 1.0])
        data = _sample(torch.randn(1, 4, 8, 8, 2))
        out = pc(data)
        assert out["data_tensor"].shape == (1, 4, 8, 8, 2)

    def test_bad_probs(self):
        with pytest.raises(ValueError, match="probs must sum to 1"):
            ProbabilisticChoice(
                transforms=[Resize(target_spatial_shape=(2, 4, 4))],
                probs=[0.5],
            )
