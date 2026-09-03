import numpy as np
import pytest
import torch

from cell_observatory_platform.data.transforms.noise import MixedPoissonGaussianNoise
from cell_observatory_platform.data.transforms.make_targets import DeepCopyInputsAsTargets
from cell_observatory_platform.data.transforms.psf import ConvolveWithPSF
from cell_observatory_platform.models.layers.preprocessor import (
    ChannelSplitPreprocessor,
    DenoisingPreprocessor,
    RayPreprocessor,
    _channel_mapping_from_meta,
    _object_type_names_from_meta,
    partition_channels,
)


def _metrics_by_name(meta: dict) -> dict[str, dict]:
    """Index the metrics list by metric_name for easier assertions."""
    return {r["metric_name"]: r for r in meta.get("metrics", [])}


def _expected_timing_keys() -> set[str]:
    return {"data_time", "preprocess_time", "transform_time", "masking_time"}


BATCH = 2
TIME = 2
DEPTH = 16
HEIGHT = 16
WIDTH = 16
CHANNELS = 2          # 1 signal channel + 1 instance-seg channel (stripped via channel_mapping)
SIGNAL_CHANNELS = CHANNELS - 1

FMT_3D = "TZYXC"
SHAPE_3D = (BATCH, TIME, DEPTH, HEIGHT, WIDTH, CHANNELS)
DENOISE_FMT = "ZYXC"
DENOISE_SHAPE = (DEPTH, HEIGHT, WIDTH, CHANNELS)          # (Z, Y, X, C) passed as input_shape
DENOISE_BATCH_SHAPE = (BATCH, DEPTH, HEIGHT, WIDTH, CHANNELS)
DENOISE_PATCH = (4, 4, 4, None)                           # (axial, lateral, lateral): 16/4 = 4 tokens per axis -> 64 patches
DENOISE_NUM_PATCHES = 4 * 4 * 4
DENOISE_PIXELS_PER_PATCH = 4 * 4 * 4 * SIGNAL_CHANNELS
PSF_SHAPE = (DEPTH // 2, HEIGHT // 2, WIDTH // 2)         # 8^3 delta PSF

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


def _denoise_transforms(seed: int = 42, **noise_kwargs) -> list:
    noise = dict(quantum_efficiency=0.82, electrons_per_count=0.22,
                 sigma_background_noise=40.0, mean_background_offset=100.0, seed=seed)
    noise.update(noise_kwargs)
    return [
        DeepCopyInputsAsTargets(),
        ConvolveWithPSF(
            psf=_delta_psf_3d(*PSF_SHAPE), pad_type="zero",
            input_shape=(DEPTH, HEIGHT, WIDTH, SIGNAL_CHANNELS),
            input_pixel_size_um=(1.0, 1.0, 1.0), psf_format="ZYX", psf_pixel_size_um=(1.0, 1.0, 1.0),
        ),
        MixedPoissonGaussianNoise(**noise),
    ]


def _make_denoiser(transforms: list) -> DenoisingPreprocessor:
    return DenoisingPreprocessor(
        denoising_type="microscopy", max_channel_count=SIGNAL_CHANNELS, min_channel_count=SIGNAL_CHANNELS,
        transforms_list=transforms, dtype=torch.float32, patch_shape=DENOISE_PATCH,
        with_masking=False, mask_generator=None, input_format=DENOISE_FMT, input_shape=DENOISE_SHAPE,
    )


def _denoise_sample() -> tuple[torch.Tensor, dict]:
    inputs = torch.ones(DENOISE_BATCH_SHAPE, dtype=torch.float32) * 100.0
    sample = {"data_tensor": inputs, "metainfo": {"channel_mapping": {1: "instance_masks"}}}
    return inputs[..., :SIGNAL_CHANNELS].clone(), sample


# ---- ---- ----


# -------------------------
# RayPreprocessor tests
# -------------------------


@pytest.mark.cuda
def test_ray_preprocessor_transform_and_masking_on_cuda():
    B, T, Z, Y, X, C = 2, 1, 1, 4, 4, 3
    inputs = torch.ones((B, T, Z, Y, X, C), dtype=torch.float32, device="cuda")
    sample = {"data_tensor": inputs, "metainfo": {"k": "v"}}

    def add_five(sample: dict) -> dict:
        # Transforms receive the full {data_tensor, metainfo} sample, not a bare tensor --
        # forward() injects metainfo["data_types"] and real transforms (Resize/crop_to_valid)
        # read image_sizes from it. Mirrors _CropLastX in test_preprocessor_sam2.py.
        x = sample["data_tensor"]
        assert x.is_cuda
        assert "data_types" in sample["metainfo"], (
            "forward() must inject data_types into metainfo before transforms run"
        )
        sample["data_tensor"] = x + 5
        return sample

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


@pytest.mark.cuda
def test_ray_preprocessor_no_mask_returns_empty_meta():
    B, T, Z, Y, X, C = 2, 1, 1, 4, 4, 2
    inputs = torch.zeros((B, T, Z, Y, X, C), dtype=torch.float32, device="cuda")
    sample = {"data_tensor": inputs, "metainfo": {"channel_mapping": {1: "instance_masks"}}}

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


@pytest.mark.parametrize(
    "fmt,shape",
    [("TYXZ", (TIME, HEIGHT, WIDTH, DEPTH)),        # C not last
     ("TZYC", (TIME, DEPTH, HEIGHT, CHANNELS))],    # X missing
    ids=["c_not_last", "missing_x"],
)
def test_unknown_layout_raises(fmt, shape):
    """An unsupported layout string fails at construction with the layout named in the error."""
    # RayPreprocessor.__init__ calls calc_num_patches -> get_patch_sizes before any
    # C-last / Y-X validation; an unsupported layout must fail there with a clear message.
    with pytest.raises(ValueError, match="Unknown dataset layout order"):
        ChannelSplitPreprocessor(
            patch_shape=(1, 4, 4), transforms_list=[], with_masking=False, mask_generator=None,
            dtype=torch.float32, input_format=fmt, input_shape=shape,
            max_channel_count=1, min_channel_count=1,
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
        # `channels` is DB-sourced (max_channel_count), no longer derived from
        # input_shape[channel_idx]; recon preprocessors now require it explicitly.
        max_channel_count=axis_to_size["C"],
        min_channel_count=axis_to_size["C"],
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


# -------------------------
# DenoisingPreprocessor tests
# -------------------------


def test_denoising_preprocessor_forward():
    """Forward emits a noisy signal tensor and a patchified clean clone as the denoising target."""
    proc = _make_denoiser(_denoise_transforms())
    clean, sample = _denoise_sample()

    output = proc(sample, data_time=0.1, idx=0)
    noisy = output["data_tensor"]
    targets = output["metainfo"]["targets"]["denoising"]   # Form-D role, patchified clean clone

    assert noisy.shape == clean.shape
    assert not torch.allclose(clean, noisy, atol=1e-6)
    assert torch.all(noisy >= 0)
    # 64 patches of 4*4*4 voxels * 1 channel
    assert targets.shape == (BATCH, DENOISE_NUM_PATCHES, DENOISE_PIXELS_PER_PATCH)
    assert torch.allclose(targets, torch.full_like(targets, 100.0))        # clean clone taken BEFORE noise
    assert not torch.allclose(targets, proc.pe_patchify(noisy, channels=SIGNAL_CHANNELS), atol=1e-6)

    meta = output["metainfo"]
    for k in _expected_timing_keys():
        assert k not in meta
    metrics = _metrics_by_name(meta)
    assert _expected_timing_keys().issubset(metrics)
    assert metrics["data_time"]["value"] == 0.1
    assert metrics["data_time"]["category"] == "timing"
    assert metrics["data_time"]["reduce_method"] == ["median", "max", "min"]
    assert isinstance(metrics["transform_time"]["value"], float)


def test_denoising_tuple_params_vary_noise_per_batch():
    """Tuple noise parameters are drawn per sample, so batch elements get different noise fields."""
    proc = _make_denoiser(_denoise_transforms(
        quantum_efficiency=(0.7, 0.9), electrons_per_count=(0.2, 0.3),
        sigma_background_noise=(30, 50), mean_background_offset=(80, 120),
    ))
    clean, sample = _denoise_sample()
    noisy = proc(sample, data_time=0.0, idx=0)["data_tensor"]
    assert noisy.shape == clean.shape
    # per-sample parameter draws -> different noise fields per batch element
    assert not torch.allclose(noisy[0] - clean[0], noisy[1] - clean[1], atol=1e-6)


def test_denoising_preprocessor_reproducibility():
    """Same seed reproduces the noise field exactly; a different seed does not."""
    outs = {}
    for name, seed in (("a42", 42), ("b42", 42), ("c123", 123)):
        _, sample = _denoise_sample()
        outs[name] = _make_denoiser(_denoise_transforms(seed=seed))(sample, data_time=0.0, idx=0)["data_tensor"]
    assert torch.allclose(outs["a42"], outs["b42"], atol=1e-6)
    assert not torch.allclose(outs["a42"], outs["c123"], atol=1e-6)


# -------------------------
# partition_channels
# -------------------------


def test_partition_channels_routes_family_member_to_targets():
    """A `<family>_<name>` role (semantic_masks_golgi) belongs to the declared
    family and must partition as a target, never as model input."""
    cm = {0: "membrane", 1: "cytosol", 2: "semantic_masks_golgi"}
    p = partition_channels(cm, 3, frozenset({"semantic_masks"}))
    assert p.input_idxs == [0, 1]
    assert p.targets_by_role == {"semantic_masks_golgi": [2]}
    assert p.dropped_idxs == []


def test_partition_channels_drops_unconsumed_object_role():
    """An object role that no declared target family consumes is dropped, not fed
    to the model as input."""
    cm = {0: "membrane", 1: "semantic_masks_golgi"}
    p = partition_channels(cm, 2, frozenset())  # recon task: no targets
    assert p.input_idxs == [0]
    assert p.dropped_idxs == [1]


def test_partition_channels_rejects_non_object_role_in_target_family():
    """A role matching a declared target family that is NOT an object role is a
    config/DB contradiction: raise, never silently become model input."""
    cm = {0: "membrane", 1: "custom_target_foo"}
    with pytest.raises(ValueError, match="matches a target family"):
        partition_channels(cm, 2, frozenset({"custom_target"}))


# -------------------------
# _channel_mapping_from_meta
# -------------------------


class TestChannelMappingFromMeta:
    def test_dict_passthrough(self):
        assert _channel_mapping_from_meta(
            {"channel_mapping": {1: "instance_masks"}}
        ) == {1: "instance_masks"}

    def test_homogeneous_list_takes_first(self):
        cm = [{"0": "a"}, {"0": "a"}]
        assert _channel_mapping_from_meta({"channel_mapping": cm}) == {"0": "a"}

    def test_heterogeneous_batch_raises(self):
        cm = [{"0": "a"}, {"0": "b"}]
        with pytest.raises(ValueError, match="not homogeneous"):
            _channel_mapping_from_meta({"channel_mapping": cm})

    def test_json_string_rows_parsed(self):
        cm = np.array(['{"0": "membrane", "1": "instance_masks"}'] * 2, dtype=object)
        out = _channel_mapping_from_meta({"channel_mapping": cm})
        assert out == {"0": "membrane", "1": "instance_masks"}

    def test_missing_returns_none(self):
        assert _channel_mapping_from_meta({}) is None
        assert _channel_mapping_from_meta({"channel_mapping": None}) is None


# -------------------------
# DenoisingPreprocessor: target precision
# -------------------------


def test_denoising_targets_keep_uint16_counts_exact():
    """The clean denoising target is snapshotted from a float32 intermediate, so
    adjacent uint16 counts (40000/40001 alias in bf16) stay distinguishable; the
    model input narrows to the configured dtype only at the end of forward."""
    Z = Y = X = 8
    pp = DenoisingPreprocessor(
        denoising_type="microscopy",
        transforms_list=[DeepCopyInputsAsTargets()],
        with_masking=False,
        mask_generator=None,
        patch_shape=(1, 4, 4, 4),
        dtype="bfloat16",
        input_format="ZYXC",
        input_shape=(Z, Y, X, 2),
        max_channel_count=1,
        min_channel_count=1,
    )
    inputs = torch.zeros((1, Z, Y, X, 2), dtype=torch.uint16)
    inputs[0, 0, 0, 0, 0] = 40000
    inputs[0, 0, 0, 1, 0] = 40001
    sample = {
        "data_tensor": inputs,
        "metainfo": {"channel_mapping": {1: "instance_masks"}},
    }
    out = pp(sample, data_time=0.0, idx=0)

    targets = out["metainfo"]["targets"]["denoising"]   # Form-D role, no wrap
    assert targets.dtype == torch.float32
    vals = set(torch.unique(targets).tolist())
    assert 40000.0 in vals and 40001.0 in vals, (
        "exact counts lost before the loss target -- the float32 "
        f"intermediate is broken (unique={sorted(vals)[-4:]})"
    )
    assert out["data_tensor"].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# _object_type_names_from_meta
# ---------------------------------------------------------------------------


def test_object_type_names_read_from_metainfo():
    """The catalog rides in metainfo, not a constructor kwarg: the registry
    splats the preprocessor's config node into its __init__, so a key only
    SemanticSegmentationPreprocessor accepts would be an unexpected kwarg for the
    other seven."""
    assert _object_type_names_from_meta({"object_type_names": {1: "cell"}}) == {1: "cell"}


def test_object_type_names_tolerates_the_per_sample_convention():
    """Ray hands batched columns through as per-sample sequences; the catalog is
    batch-level, so row 0 is the whole answer."""
    assert _object_type_names_from_meta(
        {"object_type_names": [{"2": "nucleus"}, {"2": "nucleus"}]}
    ) == {2: "nucleus"}


def test_object_type_names_absent_is_empty():
    """Empty is not silently tolerated downstream -- build_semantic_targets
    raises on a legend class it cannot resolve."""
    assert _object_type_names_from_meta({}) == {}
    assert _object_type_names_from_meta(None) == {}


# --------------------------------------------------------------------------- #
# uint16 -> fp32 lift gate
#
# An fp32 intermediate is kept only when a transform declares reads_raw_counts;
# otherwise uint16 is cast once, straight to the model dtype. Transforms opt in
# themselves, so the check must find one nested inside a wrapper too.
# --------------------------------------------------------------------------- #

from cell_observatory_platform.models.layers.preprocessor import _reads_raw_counts


class _Plain:
    """A transform with no opinion about counts (Crop, Resize, ...)."""


class _NeedsCounts:
    reads_raw_counts = True


class _Wrapper:
    """Holds children in `.transforms`, as ProbabilisticChoice does."""

    def __init__(self, *children):
        self.transforms = list(children)


@pytest.mark.parametrize("transforms", [None, [], [_Plain()], [_Plain(), _Plain()]])
def test_no_transform_reads_raw_counts(transforms):
    assert _reads_raw_counts(transforms) is False


def test_a_declared_transform_is_detected():
    assert _reads_raw_counts([_Plain(), _NeedsCounts()]) is True


def test_a_nested_transform_is_detected():
    """The flat-scan regression: a sensor-model transform inside a
    ProbabilisticChoice needs the exact counts just as much as a top-level one,
    and a scan that missed it would quantize silently."""
    assert _reads_raw_counts([_Plain(), _Wrapper(_Plain(), _NeedsCounts())]) is True


def test_two_levels_of_nesting_are_detected():
    assert _reads_raw_counts([_Wrapper(_Wrapper(_NeedsCounts()))]) is True


def test_an_empty_wrapper_is_not_a_false_positive():
    assert _reads_raw_counts([_Wrapper()]) is False


def test_the_real_sensor_transforms_declare_themselves():
    """The gate is only correct if the transforms that need counts opt in."""
    from cell_observatory_platform.data.transforms.noise import MixedPoissonGaussianNoise
    from cell_observatory_platform.data.transforms.psf import ConvolveWithPSF

    assert MixedPoissonGaussianNoise.reads_raw_counts is True
    assert ConvolveWithPSF.reads_raw_counts is True


def test_normalize_does_not_claim_raw_counts():
    """Measured, not assumed: normalizing after the cast computes its statistics
    from the same quantized values the model sees, and on real data the z-scores
    differ by ~0.0035 RMS. See docs/analysis/2026-08-24-uint16-fp32-lift-gate.md."""
    from cell_observatory_platform.data.transforms.normalize import Normalize

    assert getattr(Normalize, "reads_raw_counts", False) is False
