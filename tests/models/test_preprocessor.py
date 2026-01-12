import pytest
import torch

from cell_observatory_platform.data.utils import resize_mask
from cell_observatory_platform.models.layers.preprocessor import (
    ChannelSplitPreprocessor,
    DenoisingPreprocessor,
    InstanceSegmentationPreprocessor,
    RayPreprocessor,
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
# RayPreprocessor tests
# -------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for RayPreprocessor tests")
def test_ray_preprocessor_transform_and_masking_on_cuda():
    B, T, Y, X, C = 2, 1, 4, 4, 3
    inputs = torch.ones((B, T, Y, X, C), dtype=torch.float32, device="cuda")
    sample = {"data_tensor": inputs, "metainfo": {"k": "v"}}

    def add_five(x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda
        return x + 5

    proc = RayPreprocessor(
        dtype=torch.float32,
        with_masking=True,
        input_format="TYXC",
        input_shape=(T, Y, X, C),
        patch_shape=(1, 4, 4, None),
        mask_generator=_dummy_mask_generator,
        transforms_list=[add_five],
    )

    out = proc(sample, data_time=0.33)
    data = out["data_tensor"]
    meta = out["metainfo"]

    assert data.is_cuda
    assert data.shape == inputs.shape
    assert torch.allclose(data, inputs + 5)

    for k in ("masks", "context_masks", "target_masks", "original_patch_indices", "channels_to_mask"):
        assert k in meta
        assert isinstance(meta[k], list)
        assert len(meta[k]) == 1

    assert isinstance(meta["preprocess_time"], float)
    assert isinstance(meta["masking_time"], float)
    assert isinstance(meta["transform_time"], float)
    assert meta["data_time"] == 0.33

    assert meta["tokens_per_batch"] == B * proc.seq_len


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for RayPreprocessor tests")
def test_ray_preprocessor_no_mask_returns_empty_meta():
    B, T, Y, X, C = 2, 1, 4, 4, 2
    inputs = torch.zeros((B, T, Y, X, C), dtype=torch.float32, device="cuda")
    sample = {"data_tensor": inputs, "metainfo": {}}

    proc = RayPreprocessor(
        dtype=torch.float32,
        with_masking=False,
        input_format="TYXC",
        input_shape=(T, Y, X, C),
        patch_shape=(1, 4, 4, None),
        mask_generator=_dummy_mask_generator,
        transforms_list=None,
    )

    out = proc(sample, data_time=0.0)
    data = out["data_tensor"]
    meta = out["metainfo"]

    assert data.is_cuda
    assert data.shape == inputs.shape
    assert torch.allclose(data, inputs)

    for k in ("masks", "context_masks", "target_masks", "original_patch_indices", "channels_to_mask", "patches_used"):
        assert k not in meta

    assert isinstance(meta["preprocess_time"], float)
    assert meta["masking_time"] == -1.0
    assert isinstance(meta["transform_time"], float)
    assert meta["data_time"] == 0.0
    assert meta["tokens_per_batch"] == B * proc.seq_len


def test_requires_c_last_in_input_format():
    with pytest.raises(ValueError):
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


# -------------------------
# DenoisingPreprocessor tests
# -------------------------


def test_denoising_preprocessor_init():
    """Test initialization with float and tuple parameters."""
    # Float parameters
    proc = DenoisingPreprocessor(
        quantum_efficiency=0.82,
        electrons_per_count=0.22,
        sigma_background_noise=40.0,
        mean_background_offset=100.0,
        dtype=torch.float32,
        patch_shape=(1, 4, 4, None),
        transforms_list=[],
        with_masking=False,
        mask_generator=None,
        input_format=FMT_2D,
        input_shape=SHAPE_2D[1:],
    )
    assert proc.quantum_efficiency == 0.82
    assert proc.electrons_per_count == 0.22
    assert proc.sigma_background_noise == 40.0
    assert proc.mean_background_offset == 100.0

    # Tuple parameters
    proc = DenoisingPreprocessor(
        quantum_efficiency=(0.7, 0.9),
        electrons_per_count=(0.2, 0.3),
        sigma_background_noise=(30, 50),
        mean_background_offset=(80, 120),
        dtype=torch.float32,
        patch_shape=(1, 4, 4, None),
        transforms_list=[],
        with_masking=False,
        mask_generator=None,
        input_format=FMT_2D,
        input_shape=SHAPE_2D[1:],
    )
    assert proc.quantum_efficiency == (0.7, 0.9)
    assert proc.electrons_per_count == (0.2, 0.3)


def test_denoising_preprocessor_init_invalid_params():
    """Test that invalid parameter values raise ValueError."""
    with pytest.raises(ValueError, match="quantum_efficiency must be a float or tuple of two floats"):
        DenoisingPreprocessor(
            quantum_efficiency="invalid",
            electrons_per_count=0.22,
            sigma_background_noise=40.0,
            mean_background_offset=100.0,
            dtype=torch.float32,
            patch_shape=(1, 4, 4, None),
            transforms_list=[],
            with_masking=False,
            mask_generator=None,
            input_format=FMT_2D,
            input_shape=SHAPE_2D[1:],
        )

    with pytest.raises(ValueError, match="electrons_per_count must be a float or tuple of two floats"):
        DenoisingPreprocessor(
            quantum_efficiency=0.82,
            electrons_per_count=(0.1,),  # wrong length
            sigma_background_noise=40.0,
            mean_background_offset=100.0,
            dtype=torch.float32,
            patch_shape=(1, 4, 4, None),
            transforms_list=[],
            with_masking=False,
            mask_generator=None,
            input_format=FMT_2D,
            input_shape=SHAPE_2D[1:],
        )


def test_denoising_preprocessor_noise_addition():
    """Test noise addition with float and tuple parameters."""
    # Fixed parameters
    proc = DenoisingPreprocessor(
        quantum_efficiency=0.82,
        electrons_per_count=0.22,
        sigma_background_noise=40.0,
        mean_background_offset=100.0,
        dtype=torch.float32,
        patch_shape=(1, 4, 4, None),
        transforms_list=[],
        with_masking=False,
        mask_generator=None,
        input_format=FMT_2D,
        input_shape=SHAPE_2D[1:],
        seed=42,
    )

    B, T, Y, X, C = 2, 1, 8, 8, 2
    inputs = torch.ones((B, T, Y, X, C), dtype=torch.float32) * 100.0

    noisy_inputs, time_taken = proc._add_noise(inputs)

    assert not torch.allclose(inputs, noisy_inputs, atol=1e-6)
    assert noisy_inputs.shape == inputs.shape
    assert torch.all(noisy_inputs >= 0)  # After clipping
    assert isinstance(time_taken, float) and time_taken >= 0

    # Tuple parameters - check per-batch variation
    proc = DenoisingPreprocessor(
        quantum_efficiency=(0.7, 0.9),
        electrons_per_count=(0.2, 0.3),
        sigma_background_noise=(30, 50),
        mean_background_offset=(80, 120),
        dtype=torch.float32,
        patch_shape=(1, 4, 4, None),
        transforms_list=[],
        with_masking=False,
        mask_generator=None,
        input_format=FMT_2D,
        input_shape=SHAPE_2D[1:],
        seed=42,
    )

    B, T, Y, X, C = 4, 1, 8, 8, 2
    inputs = torch.ones((B, T, Y, X, C), dtype=torch.float32) * 100.0
    noisy_inputs, _ = proc._add_noise(inputs)

    # Different batch elements should have different noise patterns
    batch_0_noise = noisy_inputs[0] - inputs[0]
    batch_1_noise = noisy_inputs[1] - inputs[1]
    assert not torch.allclose(batch_0_noise, batch_1_noise, atol=1e-6)


def test_denoising_preprocessor_forward():
    """Test forward pass produces noisy inputs and clean targets."""
    proc = DenoisingPreprocessor(
        quantum_efficiency=0.82,
        electrons_per_count=0.22,
        sigma_background_noise=40.0,
        mean_background_offset=100.0,
        dtype=torch.float32,
        patch_shape=(1, 4, 4, None),
        transforms_list=[],
        with_masking=False,
        mask_generator=None,
        input_format=FMT_2D,
        input_shape=SHAPE_2D[1:],
        seed=42,
    )

    B, T, Y, X, C = 2, 1, 8, 8, 2
    inputs = torch.ones((B, T, Y, X, C), dtype=torch.float32) * 100.0
    sample = {"data_tensor": inputs, "metainfo": {}}

    output = proc(sample, data_time=0.1)

    assert "data_tensor" in output and "metainfo" in output
    noisy_inputs = output["data_tensor"]
    targets = output["metainfo"]["targets"][0]

    assert not torch.allclose(inputs, noisy_inputs, atol=1e-6)
    assert noisy_inputs.shape == inputs.shape
    assert targets.shape == inputs.shape
    assert not torch.allclose(targets, noisy_inputs, atol=1e-6)

    meta = output["metainfo"]
    assert isinstance(meta["preprocess_time"], float)
    assert isinstance(meta["transform_time"], float)
    assert meta["data_time"] == 0.1


def test_denoising_preprocessor_with_transforms():
    """Test forward pass applies transforms before noise."""
    def add_constant(x: torch.Tensor) -> torch.Tensor:
        return x + 50.0

    proc = DenoisingPreprocessor(
        quantum_efficiency=0.82,
        electrons_per_count=0.22,
        sigma_background_noise=40.0,
        mean_background_offset=100.0,
        dtype=torch.float32,
        patch_shape=(1, 4, 4, None),
        transforms_list=[add_constant],
        with_masking=False,
        mask_generator=None,
        input_format=FMT_2D,
        input_shape=SHAPE_2D[1:],
        seed=42,
    )

    B, T, Y, X, C = 2, 1, 8, 8, 2
    inputs = torch.ones((B, T, Y, X, C), dtype=torch.float32) * 100.0
    sample = {"data_tensor": inputs, "metainfo": {}}

    output = proc(sample, data_time=0.1)

    targets = output["metainfo"]["targets"][0]
    expected_targets = inputs + 50.0
    assert torch.allclose(targets, expected_targets, atol=1e-5)


def test_denoising_preprocessor_reproducibility():
    """Test that same seed produces same noise, different seeds produce different noise."""
    B, T, Y, X, C = 2, 1, 8, 8, 2
    inputs = torch.ones((B, T, Y, X, C), dtype=torch.float32) * 100.0
    sample = {"data_tensor": inputs, "metainfo": {}}

    proc1 = DenoisingPreprocessor(
        quantum_efficiency=0.82,
        electrons_per_count=0.22,
        sigma_background_noise=40.0,
        mean_background_offset=100.0,
        dtype=torch.float32,
        patch_shape=(1, 4, 4, None),
        transforms_list=[],
        with_masking=False,
        mask_generator=None,
        input_format=FMT_2D,
        input_shape=SHAPE_2D[1:],
        seed=42,
    )

    proc2 = DenoisingPreprocessor(
        quantum_efficiency=0.82,
        electrons_per_count=0.22,
        sigma_background_noise=40.0,
        mean_background_offset=100.0,
        dtype=torch.float32,
        patch_shape=(1, 4, 4, None),
        transforms_list=[],
        with_masking=False,
        mask_generator=None,
        input_format=FMT_2D,
        input_shape=SHAPE_2D[1:],
        seed=42,
    )

    proc3 = DenoisingPreprocessor(
        quantum_efficiency=0.82,
        electrons_per_count=0.22,
        sigma_background_noise=40.0,
        mean_background_offset=100.0,
        dtype=torch.float32,
        patch_shape=(1, 4, 4, None),
        transforms_list=[],
        with_masking=False,
        mask_generator=None,
        input_format=FMT_2D,
        input_shape=SHAPE_2D[1:],
        seed=123,
    )

    output1 = proc1(sample, data_time=0.0)
    output2 = proc2(sample, data_time=0.0)
    output3 = proc3(sample, data_time=0.0)

    # Same seed produces same results
    assert torch.allclose(output1["data_tensor"], output2["data_tensor"], atol=1e-6)
    # Different seed produces different results
    assert not torch.allclose(output1["data_tensor"], output3["data_tensor"], atol=1e-6)
