import pytest

import torch

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


def test_resize_masks_shape():
    masks = torch.ones(3, 4, 8, 8)
    out = resize_masks(masks, (2, 4, 4))
    assert out.shape == (3, 2, 4, 4)


def test_resize_label_map_shape():
    lm = torch.randint(0, 5, (4, 8, 8))
    out = resize_label_map(lm, (2, 4, 4))
    assert out.shape == (2, 4, 4)


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
        r = Resize(
            target_spatial_shape=(2, 4, 4),
            bbox_format="zyxzyx",
        )
        masks = torch.ones(3, 4, 8, 8, dtype=torch.bool)
        label_map = torch.randint(0, 5, (4, 8, 8))
        boxes = torch.rand(3, 6)
        data = {
            "data_tensor": torch.randn(1, 4, 8, 8, 2),
            "metainfo": {
                # Which target fields get resized (and how) is driven by data_types,
                # not by their presence in the target dict -- see
                # Resize._target_field_specs (resize.py:307).
                "data_types": {
                    "data_tensor": {"kind": "dense", "layout": "ZYXC",
                                    "role": "input", "has_time": False},
                    # stacked (N, Z, Y, X) -> resize_masks; a single (Z, Y, X)
                    # labelmap is "instance_masks" -> resize_label_map.
                    "masks": {"kind": "semantic_masks", "layout": "ZYXC", "role": "target"},
                    "label_map": {"kind": "instance_masks", "layout": "ZYXC", "role": "target"},
                    "boxes": {"kind": "boxes", "layout": "zyxzyx", "role": "target"},
                },
                "targets": [{"masks": masks, "label_map": label_map, "boxes": boxes}],
            },
        }
        out = r(data)
        tgt = out["metainfo"]["targets"][0]
        assert tgt["masks"].shape == (3, 2, 4, 4)
        assert tgt["label_map"].shape == (2, 4, 4)
        assert tgt["boxes"].shape == (3, 6)

    def test_random_range(self):
        r = Resize(target_spatial_shape=((2, 4, 4), (4, 8, 8)))
        x = torch.randn(1, 4, 8, 8, 2)
        out = r(_sample(x))["data_tensor"]
        assert 2 <= out.shape[1] <= 4
        assert 4 <= out.shape[2] <= 8


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
