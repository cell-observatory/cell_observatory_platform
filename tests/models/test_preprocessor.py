import pytest
import torch

from cell_observatory_platform.data.utils import resize_mask
from cell_observatory_platform.models.layers.preprocessor import (
    ChannelSplitPreprocessor,
    DaliPreprocessor,
    InstanceSegmentationPreprocessor,
    RayPreprocessor,
    TorchPreprocessor,
    UpsamplePreprocessor,
)

BATCH = 32
TIME = 16
DEPTH = 128
HEIGHT = 128
WIDTH = 128
CHANNELS = 2

FMT_2D = "TYXC"
FMT_3D = "TZYXC"

SHAPE_2D = (BATCH, TIME, HEIGHT, WIDTH, CHANNELS)
SHAPE_3D = (BATCH, TIME, DEPTH, HEIGHT, WIDTH, CHANNELS)

# ---- helpers ----


def _dummy_mask_generator(batch_size: int):
    masks = torch.ones(batch_size, 1, dtype=torch.bool)
    ctx = torch.zeros(batch_size, 1, dtype=torch.bool)
    tgt = torch.ones(batch_size, 1, dtype=torch.bool)
    orig_idx = torch.arange(batch_size)
    channels_to_mask = [0]
    return masks, ctx, tgt, orig_idx, channels_to_mask, None


def _delta_psf_2d(h: int, w: int) -> torch.Tensor:
    psf = torch.zeros((h, w), dtype=torch.float32)
    psf[h // 2, w // 2] = 1.0
    return psf


def _delta_psf_3d(d: int, h: int, w: int) -> torch.Tensor:
    psf = torch.zeros((d, h, w), dtype=torch.float32)
    psf[d // 2, h // 2, w // 2] = 1.0
    return psf


# ---- ---- ----


# -------------------------
# TorchPreprocessor tests
# -------------------------


def test_torch_preprocessor_no_mask_stacks_list_and_keeps_dtype():
    B, C = 3, 4
    tensors = [torch.zeros(C, dtype=torch.float32) for _ in range(B)]
    sample = {"data_tensor": tensors, "metainfo": {"foo": 123}}

    proc = TorchPreprocessor(dtype=torch.float32, with_masking=False, mask_generator=None)
    out = proc(sample, data_time=0.01)

    assert "data_tensor" in out and "metainfo" in out
    assert out["data_tensor"].shape == (B, C)
    assert out["data_tensor"].dtype == torch.float32
    assert out["metainfo"]["foo"] == 123


def test_torch_preprocessor_with_mask_includes_mask_info():
    B, C = 2, 5
    sample = {"data_tensor": torch.ones(B, C, dtype=torch.float32), "metainfo": {"a": 7}}

    proc = TorchPreprocessor(dtype=torch.float32, with_masking=True, mask_generator=_dummy_mask_generator)
    out = proc(sample, data_time=0.02)

    meta = out["metainfo"]
    for k in ("masks", "context_masks", "target_masks", "original_patch_indices", "channels_to_mask"):
        assert k in meta and isinstance(meta[k], list) and len(meta[k]) == 1

    assert isinstance(meta["preprocess_time"], float)
    assert isinstance(meta["masking_time"], float)
    assert isinstance(meta["data_time"], float) and meta["data_time"] == 0.02
    assert meta["a"] == 7


# -------------------------
# DaliPreprocessor tests
# -------------------------


def test_dali_preprocessor_no_mask_minimal():
    B, C = 2, 3
    inputs = torch.zeros(B, C, dtype=torch.float16)
    dali_sample = ({"data_tensor": inputs, "get_item_time": 0.123},)

    proc = DaliPreprocessor(dtype=torch.float16, with_masking=False, mask_generator=None)
    out = proc(dali_sample, data_time=0.0)

    assert torch.equal(out["data_tensor"], inputs)
    assert out["metainfo"] == {}


def test_dali_preprocessor_with_mask_includes_fields_and_timings():
    B, C = 2, 4
    inputs = torch.ones(B, C, dtype=torch.bfloat16)
    dali_sample = ({"data_tensor": inputs, "get_item_time": 0.5},)

    proc = DaliPreprocessor(dtype=torch.bfloat16, with_masking=True, mask_generator=_dummy_mask_generator)
    out = proc(dali_sample, data_time=0.25)

    meta = out["metainfo"]
    for k in ("masks", "context_masks", "target_masks", "original_patch_indices", "channels_to_mask"):
        assert k in meta and isinstance(meta[k], list) and len(meta[k]) == 1
    assert isinstance(meta["preprocess_time"], float)
    assert isinstance(meta["masking_time"], float)
    assert meta["get_item_time"] == 0.5
    assert meta["data_time"] == 0.25


# -------------------------
# RayPreprocessor tests
# -------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for RayPreprocessor tests")
def test_ray_preprocessor_transform_and_masking_on_cuda():
    B, C = 2, 3
    inputs = torch.ones(B, C, dtype=torch.float32, device="cuda")
    sample = {"data_tensor": inputs, "metainfo": {"k": "v"}}

    def add_five(x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda
        return x + 5

    proc = RayPreprocessor(
        dtype=torch.float32,
        with_masking=True,
        mask_generator=_dummy_mask_generator,
        transforms_list=[add_five],
    )

    out = proc(sample, data_time=0.33)
    data = out["data_tensor"]
    meta = out["metainfo"]

    assert data.is_cuda
    assert torch.allclose(data, inputs + 5)

    for k in ("masks", "context_masks", "target_masks", "original_patch_indices", "channels_to_mask"):
        assert k in meta and isinstance(meta[k], list) and len(meta[k]) == 1

    assert isinstance(meta["preprocess_time"], float)
    assert isinstance(meta["masking_time"], float)
    assert isinstance(meta["transform_time"], float)
    assert meta["data_time"] == 0.33


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for RayPreprocessor tests")
def test_ray_preprocessor_no_mask_returns_empty_meta():
    inputs = torch.zeros(2, 2, dtype=torch.float32, device="cuda")
    sample = {"data_tensor": inputs, "metainfo": {}}

    proc = RayPreprocessor(dtype=torch.float32, with_masking=False, mask_generator=_dummy_mask_generator)
    out = proc(sample, data_time=0.0)

    assert out["data_tensor"].is_cuda
    assert out["metainfo"] == {}


def test_requires_c_last_in_input_format():
    with pytest.raises(AssertionError):
        ChannelSplitPreprocessor(
            patch_shape=(1, 2, 4, 4, 1, None),
            transforms_list=[],
            with_masking=False,
            mask_generator=None,
            dtype=torch.float32,
            input_format="TYXZ",
            input_shape=(BATCH, TIME, HEIGHT, WIDTH, DEPTH),
        )


def test_requires_xy_axes_present():
    with pytest.raises(ValueError):
        ChannelSplitPreprocessor(
            dtype=torch.float32,
            patch_shape=(1, 2, 4, None),
            transforms_list=[],
            with_masking=False,
            mask_generator=None,
            input_format="TZYC",
            input_shape=(BATCH, TIME, DEPTH, HEIGHT, CHANNELS),
        )


@pytest.mark.parametrize(
    "fmt, full_shape, axial_patch_size",
    [
        (FMT_2D, SHAPE_2D, None),
        (FMT_3D, SHAPE_3D, 4),
    ],
)
def test_spatial_dims_and_indices(fmt, full_shape, axial_patch_size):
    expected_axis_index = {ax: idx + 1 for idx, ax in enumerate(fmt)}
    axis_to_size = dict(zip(fmt, full_shape[1:]))

    expected_spatial_dims = tuple(expected_axis_index[ax] for ax in fmt if ax in ("Z", "Y", "X"))
    expected_axial_shape = axis_to_size.get("Z")
    expected_lateral_shape = (axis_to_size["Y"], axis_to_size["X"])
    expected_spatial_shape = (
        (expected_axial_shape, *expected_lateral_shape) if expected_axial_shape is not None else expected_lateral_shape
    )

    if "Z" in fmt:
        patch_shape = (1, axial_patch_size, 4, 4, None)
    else:
        patch_shape = (1, 4, 4, None)

    pp = ChannelSplitPreprocessor(
        dtype=torch.float32,
        patch_shape=patch_shape,
        transforms_list=[],
        with_masking=False,
        mask_generator=None,
        input_format=fmt,
        input_shape=full_shape[1:],
    )

    assert pp.input_format == fmt
    assert pp.input_shape == full_shape[1:]
    assert pp.axis_index == expected_axis_index
    assert pp.spatial_dims == expected_spatial_dims

    assert pp.channel_idx == expected_axis_index.get("C")
    assert pp.time_idx == expected_axis_index.get("T")
    assert pp.z_idx == expected_axis_index.get("Z")
    assert pp.y_idx == expected_axis_index.get("Y")
    assert pp.x_idx == expected_axis_index.get("X")

    assert pp.axial_shape == expected_axial_shape
    assert pp.lateral_shape == expected_lateral_shape
    assert pp.spatial_shape == expected_spatial_shape
    assert pp.channels == axis_to_size.get("C")
    assert pp.timepoints == axis_to_size.get("T")


def test_channel_split_keeps_dim_and_means():
    B, T, Y, X, C = SHAPE_2D
    x = torch.zeros(SHAPE_2D, dtype=torch.float32)
    for c in range(C):
        x[..., c] = float(c)
    pp = ChannelSplitPreprocessor(
        dtype=torch.float32,
        patch_shape=(1, 4, 4, None),
        transforms_list=[],
        with_masking=False,
        mask_generator=None,
        input_format=FMT_2D,
        input_shape=SHAPE_2D[1:],
    )
    out = pp.forward({"data_tensor": x, "metainfo": {}}, data_time=0.0)
    y = out["data_tensor"]
    assert y.shape == (B, T, Y, X, 1)
    assert torch.allclose(y.mean(), torch.tensor((C - 1) / 2, dtype=torch.float32))


def test_resize_mask_broadcast_3d_with_time_and_channel():
    spatial = _delta_psf_3d(DEPTH, HEIGHT, WIDTH)
    mask = resize_mask(
        spatial,
        input_format=FMT_3D,
        channels=CHANNELS,
        timepoints=TIME,
        axial_shape=DEPTH,
        lateral_shape=(HEIGHT, WIDTH),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert mask.ndim == len(FMT_3D)
    assert mask.shape[FMT_3D.index("Z")] == DEPTH
    assert mask.shape[FMT_3D.index("Y")] == HEIGHT
    assert mask.shape[FMT_3D.index("X")] == WIDTH
