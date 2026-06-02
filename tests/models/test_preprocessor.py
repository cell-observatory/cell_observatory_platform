import pytest
import torch

from cell_observatory_platform.data.utils import resize_mask
from cell_observatory_platform.data.transforms.noise import MixedPoissonGaussianNoise
from cell_observatory_platform.data.transforms.make_targets import DeepCopyInputsAsTargets
from cell_observatory_platform.data.transforms.psf import ConvolveWithPSF
from cell_observatory_platform.models.layers.preprocessor import (
    ChannelSplitPreprocessor,
    DenoisingPreprocessor,
    InstanceSegmentationPreprocessor,
    RayPreprocessor,
    UpsamplePreprocessor,
)


def _metrics_by_name(meta: dict) -> dict[str, dict]:
    """Index the metrics list by metric_name for easier assertions."""
    return {r["metric_name"]: r for r in meta.get("metrics", [])}


def _expected_timing_keys() -> set[str]:
    return {"data_time", "preprocess_time", "transform_time", "masking_time"}


BATCH = 32
TIME = 16
DEPTH = 128
HEIGHT = 128
WIDTH = 128
CHANNELS = 2

FMT_3D = "TZYXC"
SHAPE_3D = (BATCH, TIME, DEPTH, HEIGHT, WIDTH, CHANNELS)

# ---- helpers ----


def _dummy_mask_generator(batch_size: int):
    masks = torch.ones(batch_size, 1, dtype=torch.bool)
    ctx = torch.zeros(batch_size, 1, dtype=torch.long)
    tgt = torch.ones(batch_size, 1, dtype=torch.long)
    orig_idx = torch.arange(batch_size)
    channels_to_mask = [0]
    return {
        "masks": masks,
        "context_masks": ctx,
        "target_masks": tgt,
        "original_patch_indices": orig_idx,
        "channels_to_mask": channels_to_mask,
    }


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
    B, T, Z, Y, X, C = 2, 1, 1, 4, 4, 3
    inputs = torch.ones((B, T, Z, Y, X, C), dtype=torch.float32, device="cuda")
    sample = {"data_tensor": inputs, "metainfo": {"k": "v"}}

    def add_five(x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda
        return x + 5

    proc = RayPreprocessor(
        dtype=torch.float32,
        with_masking=True,
        input_format="TZYXC",
        input_shape=(T, Z, Y, X, C),
        patch_shape=(1, 1, 4),
        mask_generator=_dummy_mask_generator,
        transforms_list=[add_five],
    )

    out = proc(sample, data_time=0.33, idx=0)
    data = out["data_tensor"]
    meta = out["metainfo"]

    assert data.is_cuda
    assert data.shape == inputs.shape
    assert torch.allclose(data, inputs + 5)

    for k in ("masks", "context_masks", "target_masks", "original_patch_indices", "channels_to_mask"):
        assert k in meta
        assert isinstance(meta[k], list)
        assert len(meta[k]) == 1

    # Timing keys moved out of direct metainfo into metainfo["metrics"]
    for k in _expected_timing_keys():
        assert k not in meta, f"Direct timing field {k!r} should have moved into metainfo['metrics']"
    metrics = _metrics_by_name(meta)
    assert _expected_timing_keys().issubset(metrics.keys())
    for k in _expected_timing_keys():
        assert metrics[k]["category"] == "timing"
        assert metrics[k]["reduce_method"] == ["median", "max", "min"]
        assert isinstance(metrics[k]["value"], float)
    assert metrics["data_time"]["value"] == 0.33
    assert metrics["masking_time"]["value"] >= 0  # masking ran, must be non-negative
    assert metrics["transform_time"]["value"] >= 0  # transforms ran, must be non-negative


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for RayPreprocessor tests")
def test_ray_preprocessor_no_mask_returns_empty_meta():
    B, T, Z, Y, X, C = 2, 1, 1, 4, 4, 2
    inputs = torch.zeros((B, T, Z, Y, X, C), dtype=torch.float32, device="cuda")
    sample = {"data_tensor": inputs, "metainfo": {}}

    proc = RayPreprocessor(
        dtype=torch.float32,
        with_masking=False,
        input_format="TZYXC",
        input_shape=(T, Z, Y, X, C),
        patch_shape=(1, 1, 4),
        mask_generator=_dummy_mask_generator,
        transforms_list=None,
    )

    out = proc(sample, data_time=0.0, idx=0)
    data = out["data_tensor"]
    meta = out["metainfo"]

    assert data.is_cuda
    assert data.shape == inputs.shape
    assert torch.allclose(data, inputs)

    for k in ("masks", "context_masks", "target_masks", "original_patch_indices", "channels_to_mask", "patches_used"):
        assert k not in meta

    for k in _expected_timing_keys():
        assert k not in meta
    metrics = _metrics_by_name(meta)
    assert _expected_timing_keys().issubset(metrics.keys())
    assert metrics["data_time"]["value"] == 0.0
    assert metrics["masking_time"]["value"] == -1.0  # masking disabled
    assert metrics["transform_time"]["value"] == -1.0  # no transforms configured
    assert isinstance(metrics["preprocess_time"]["value"], float)


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
    [(FMT_3D, SHAPE_3D, 4)],
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
        patch_shape = (1, axial_patch_size, 4)
    else:
        patch_shape = (1, 4)

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
        denoising_type="microscopy",
        transforms_list=[
            DeepCopyInputsAsTargets(),
            ConvolveWithPSF(
                psf=_delta_psf_3d(DEPTH//2, HEIGHT//2, WIDTH//2),
                pad_type="zero",
                input_format="ZYXC",
                input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS - 1),
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
                psf_pixel_size_um=(1.0, 1.0, 1.0),
            ), 
            MixedPoissonGaussianNoise(
                quantum_efficiency=0.82,
                electrons_per_count=0.22,
                sigma_background_noise=40.0,
                mean_background_offset=100.0,
                seed=42,
                )
        ],
        dtype=torch.float32,
        patch_shape=(1, 4, 4, 4),
        with_masking=False,
        mask_generator=None,
        input_format="ZYXC",
        input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS),
        mask_channel_idx=-1,
    )
    assert len(proc.transforms) == 3, "Expected three transforms: DeepCopyInputsAsTargets, ConvolveWithPSF, MixedPoissonGaussianNoise"
    assert isinstance(proc.transforms[0], DeepCopyInputsAsTargets), "Expected first transform to be DeepCopyInputsAsTargets"
    assert isinstance(proc.transforms[1], ConvolveWithPSF), "Expected second transform to be ConvolveWithPSF"
    assert isinstance(proc.transforms[2], MixedPoissonGaussianNoise), "Expected third transform to be MixedPoissonGaussianNoise"
    assert proc.transforms[2].quantum_efficiency == 0.82, "Expected quantum efficiency to be 0.82"
    assert proc.transforms[2].electrons_per_count == 0.22, "Expected electrons per count to be 0.22"
    assert proc.transforms[2].sigma_background_noise == 40.0, "Expected sigma background noise to be 40.0"
    assert proc.transforms[2].mean_background_offset == 100.0, "Expected mean background offset to be 100.0"
    assert proc.transforms[2].seed == 42, "Expected seed to be 42"

def test_denoising_preprocessor_init_invalid_params():
    """Test that invalid parameter values raise ValueError."""
    with pytest.raises(ValueError, match="quantum_efficiency must be a float or tuple of two floats"):
        DenoisingPreprocessor(
            denoising_type="microscopy",
            transforms_list=[
                DeepCopyInputsAsTargets(), 
                ConvolveWithPSF(
                    psf=_delta_psf_3d(DEPTH//2, HEIGHT//2, WIDTH//2),
                    pad_type="zero",
                    input_format="ZYXC",
                    input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS - 1),
                    input_pixel_size_um=(1.0, 1.0, 1.0),
                    psf_format="ZYX",
                    psf_pixel_size_um=(1.0, 1.0, 1.0),
                ), 
                MixedPoissonGaussianNoise(
                    quantum_efficiency="invalid",
                    electrons_per_count=0.22,
                    sigma_background_noise=40.0,
                    mean_background_offset=100.0,
                    seed=42,
                )
            ],
            dtype=torch.float32,
            patch_shape=(1, 4, 4, 4),
            with_masking=False,
            mask_generator=None,
            input_format="ZYXC",
            input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS),
            mask_channel_idx=-1,
        )


def test_denoising_preprocessor_noise_addition():
    """Test noise addition with float and tuple parameters."""
    # Fixed parameters
    proc = DenoisingPreprocessor(
        denoising_type="microscopy",
        transforms_list=[
            DeepCopyInputsAsTargets(),
            ConvolveWithPSF(
                psf=_delta_psf_3d(DEPTH//2, HEIGHT//2, WIDTH//2),
                pad_type="zero",
                input_format="ZYXC",
                input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS - 1),
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
                psf_pixel_size_um=(1.0, 1.0, 1.0),
            ), 
            MixedPoissonGaussianNoise(
                quantum_efficiency=0.82,
                electrons_per_count=0.22,
                sigma_background_noise=40.0,
                mean_background_offset=100.0,
                seed=42,
            )
        ],
        dtype=torch.float32,
        patch_shape=(1, 4, 4, 4),
        with_masking=False,
        mask_generator=None,
        input_format="ZYXC",
        input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS),
        mask_channel_idx=-1,
    )

    inputs = torch.ones((BATCH, DEPTH, HEIGHT, WIDTH, CHANNELS), dtype=torch.float32) * 100.0
    inputs_clone = inputs.clone()
    inputs_clone = inputs_clone[..., :-1] # Remove mask channel
    sample = {"data_tensor": inputs, "metainfo": {}}
    noisy_inputs = proc(sample, data_time=0.0, idx=0)["data_tensor"]
    assert not torch.allclose(inputs_clone, noisy_inputs, atol=1e-6), "Noised inputs are the same as original inputs"
    assert noisy_inputs.shape == inputs_clone.shape, "Noised inputs have different shape than original inputs"
    assert torch.all(noisy_inputs >= 0), "Noised inputs are not in valid range [0, 65535]"

    # Tuple parameters - check per-batch variation
    proc = DenoisingPreprocessor(
        denoising_type="microscopy",
        transforms_list=[
            DeepCopyInputsAsTargets(),
            ConvolveWithPSF(
                psf=_delta_psf_3d(DEPTH//2, HEIGHT//2, WIDTH//2),
                pad_type="zero",
                input_format="ZYXC",
                input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS - 1),
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
                psf_pixel_size_um=(1.0, 1.0, 1.0),
            ), 
            MixedPoissonGaussianNoise(
            quantum_efficiency=(0.7, 0.9),
            electrons_per_count=(0.2, 0.3),
            sigma_background_noise=(30, 50),
            mean_background_offset=(80, 120),
            seed=42,
            ),
        ],
        dtype=torch.float32,
        patch_shape=(1, 4, 4, 4),
        with_masking=False,
        mask_generator=None,
        input_format="ZYXC",
        input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS),
        mask_channel_idx=-1,
    )

    
    inputs = torch.ones((BATCH, DEPTH, HEIGHT, WIDTH, CHANNELS), dtype=torch.float32) * 100.0
    inputs_clone = inputs.clone()
    inputs_clone = inputs_clone[..., :-1] # Remove mask channel
    targets_expected = proc.pe_patchify(inputs_clone, channels = CHANNELS - 1)
    sample = {"data_tensor": inputs, "metainfo": {}}
    noisy_inputs = proc(sample, data_time=0.0, idx=0)["data_tensor"]

    # Different batch elements should have different noise patterns
    batch_0_noise = noisy_inputs[0] - inputs_clone[0]
    batch_1_noise = noisy_inputs[1] - inputs_clone[1]
    assert not torch.allclose(batch_0_noise, batch_1_noise, atol=1e-6), "Different batch elements have the same noised output"


def test_denoising_preprocessor_forward():
    """Test forward pass produces noisy inputs and clean targets."""
    proc = DenoisingPreprocessor(
        denoising_type="microscopy",
        transforms_list=[
            DeepCopyInputsAsTargets(),
            ConvolveWithPSF(
                psf=_delta_psf_3d(DEPTH//2, HEIGHT//2, WIDTH//2),
                pad_type="zero",
                input_format="ZYXC",
                input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS - 1),
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
                psf_pixel_size_um=(1.0, 1.0, 1.0),
            ), 
            MixedPoissonGaussianNoise(
            quantum_efficiency=0.82,
            electrons_per_count=0.22,
            sigma_background_noise=40.0,
            mean_background_offset=100.0,
            seed=42,
            ),
        ],
        dtype=torch.float32,
        patch_shape=(1, 4, 4, 4),
        with_masking=False,
        mask_generator=None,
        input_format="ZYXC",
        input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS),
        mask_channel_idx=-1,
    )

    inputs = torch.ones((BATCH, DEPTH, HEIGHT, WIDTH, CHANNELS), dtype=torch.float32) * 100.0
    inputs_clone = inputs.clone()
    inputs_clone = inputs_clone[..., :-1] # Remove mask channel
    expected_targets = proc.pe_patchify(inputs_clone, channels = CHANNELS - 1)
    sample = {"data_tensor": inputs, "metainfo": {}}

    output = proc(sample, data_time=0.1, idx=0)

    assert "data_tensor" in output and "metainfo" in output, "data_tensor and metainfo are not in output"
    noisy_inputs = output["data_tensor"]
    targets = output["metainfo"]["targets"][0]

    assert not torch.allclose(inputs_clone, noisy_inputs, atol=1e-6), "Noised inputs are the same as original inputs"
    assert noisy_inputs.shape == inputs_clone.shape, f"Noised inputs have unexpected shape: noisy_inputs.shape={noisy_inputs.shape}, inputs_clone.shape={inputs_clone.shape}"
    assert targets.shape == expected_targets.shape, f"Targets have unexpected shape: targets.shape={targets.shape}, expected_targets.shape={expected_targets.shape}"
    assert not torch.allclose(targets, proc.pe_patchify(noisy_inputs, channels = CHANNELS - 1), atol=1e-6), "Targets are the same as noised inputs"

    meta = output["metainfo"]
    for k in _expected_timing_keys():
        assert k not in meta, f"Timing field {k!r} should now live in metainfo['metrics']"
    metrics = _metrics_by_name(meta)
    assert _expected_timing_keys().issubset(metrics.keys())
    assert isinstance(metrics["preprocess_time"]["value"], float), "preprocess_time is not a float"
    assert isinstance(metrics["transform_time"]["value"], float), "transform_time is not a float"
    assert metrics["data_time"]["value"] == 0.1, "data_time was modified"
    assert metrics["data_time"]["category"] == "timing"
    assert metrics["data_time"]["reduce_method"] == ["median", "max", "min"]


def test_denoising_preprocessor_reproducibility():
    """Test that same seed produces same noise, different seeds produce different noise."""
    inputs = torch.ones((BATCH, DEPTH, HEIGHT, WIDTH, CHANNELS), dtype=torch.float32) * 100.0
    sample1 = {"data_tensor": inputs.clone(), "metainfo": {}}
    sample2 = {"data_tensor": inputs.clone(), "metainfo": {}}
    sample3 = {"data_tensor": inputs.clone(), "metainfo": {}}

    proc1 = DenoisingPreprocessor(
        denoising_type="microscopy",
        transforms_list=[
            DeepCopyInputsAsTargets(),
            ConvolveWithPSF(
                psf=_delta_psf_3d(DEPTH//2, HEIGHT//2, WIDTH//2),
                pad_type="zero",
                input_format="ZYXC",
                input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS - 1),
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
                psf_pixel_size_um=(1.0, 1.0, 1.0),
            ), 
            MixedPoissonGaussianNoise(
                quantum_efficiency=0.82,
                electrons_per_count=0.22,
                sigma_background_noise=40.0,
                mean_background_offset=100.0,
                seed=42,
            ),
        ],
        dtype=torch.float32,
        patch_shape=(1, 4, 4, 4),
        with_masking=False,
        mask_generator=None,
        input_format="ZYXC",
        input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS),
        mask_channel_idx=-1,
    )

    proc2 = DenoisingPreprocessor(
        denoising_type="microscopy",
        transforms_list=[
            DeepCopyInputsAsTargets(),
            ConvolveWithPSF(
                psf=_delta_psf_3d(DEPTH//2, HEIGHT//2, WIDTH//2),
                pad_type="zero",
                input_format="ZYXC",
                input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS - 1),
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
                psf_pixel_size_um=(1.0, 1.0, 1.0),
            ), 
            MixedPoissonGaussianNoise(
            quantum_efficiency=0.82,
            electrons_per_count=0.22,
            sigma_background_noise=40.0,
            mean_background_offset=100.0,
            seed=42,
            ),
        ],
        dtype=torch.float32,
        patch_shape=(1, 4, 4, 4),
        with_masking=False,
        mask_generator=None,
        input_format="ZYXC",
        input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS),
        mask_channel_idx=-1,
    )

    proc3 = DenoisingPreprocessor(
        denoising_type="microscopy",
        transforms_list=[
            DeepCopyInputsAsTargets(),
            ConvolveWithPSF(
                psf=_delta_psf_3d(DEPTH//2, HEIGHT//2, WIDTH//2),
                pad_type="zero",
                input_format="ZYXC",
                input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS - 1),
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
                psf_pixel_size_um=(1.0, 1.0, 1.0),
            ), 
            MixedPoissonGaussianNoise(
            quantum_efficiency=0.82,
            electrons_per_count=0.22,
            sigma_background_noise=40.0,
            mean_background_offset=100.0,
            seed=123,
        )],
        dtype=torch.float32,
        patch_shape=(1, 4, 4, 4),
        with_masking=False,
        mask_generator=None,
        input_format="ZYXC",
        input_shape=(DEPTH, HEIGHT, WIDTH, CHANNELS),
        mask_channel_idx=-1,
    )

    output1 = proc1(sample1, data_time=0.0, idx=0)
    output2 = proc2(sample2, data_time=0.0, idx=0)
    output3 = proc3(sample3, data_time=0.0, idx=0)

    # Same seed produces same results
    assert torch.allclose(output1["data_tensor"], output2["data_tensor"], atol=1e-6), "Same seed produces different results"
    # Different seed produces different results
    assert not torch.allclose(output1["data_tensor"], output3["data_tensor"], atol=1e-6), "Different seed produces same results"
