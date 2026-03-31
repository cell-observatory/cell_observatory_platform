import numpy as np
import pytest
import torch
import torch.nn.functional as F_nn

from pathlib import Path

from cell_observatory_platform.data.transforms.noise import (
    CytosolicHaze,
    GaussianBlur3d,
    MixedPoissonGaussianNoise,
)


SEED = 42

# -----------------------------------------------------------------------------
# Helper functions from https://github.com/cell-observatory/synthetic_data_generation_pipelines/blob/f3b4d13eb5c322d35d4992ec69873f234782d03f/src_code/membrane_simulation_pipeline/semi_synthetic_membrane_sim_cubes.py
# to test the noise transform for correctness against the original implementation

def photons2electrons(image, quantum_efficiency=.82):
    return image * quantum_efficiency


def electrons2photons(image, quantum_efficiency=.82):
    return image / quantum_efficiency


def electrons2counts(image, electrons_per_count=.22):
    return image / electrons_per_count


def counts2electrons(image, electrons_per_count=.22):
    return image * electrons_per_count


def photons2counts(image, quantum_efficiency=.82, electrons_per_count=.22):
    return electrons2counts(photons2electrons(image))


def counts2photons(image, quantum_efficiency=.82, electrons_per_count=.22):
    return electrons2photons(counts2electrons(image))


def randuniform(var, rng: np.random.Generator):
    if np.isscalar(var):
        return var
    else:
        return rng.uniform(*var)

def normal_noise(mean: float, sigma: float, size: tuple, rng: np.random.Generator) -> np.array:
    mean = randuniform(mean, rng)
    sigma = randuniform(sigma, rng)
    return rng.normal(loc=mean, scale=sigma, size=size).astype(np.float32)

def poisson_noise(image: np.ndarray, rng: np.random.Generator) -> np.array:
    image = np.nan_to_num(image, nan=0)
    return rng.poisson(lam=image).astype(np.float32) - image

def noise_ref(image, mean_background_offset=100, sigma_background_noise=40, quantum_efficiency=.82, electrons_per_count=.22):
    """
    Args:
        image: noise-free image in photons
        mean_background_offset: camera background offset
        sigma_background_noise: read noise from the camera
        quantum_efficiency: quantum efficiency of the camera
        electrons_per_count: conversion factor to go from electrons to counts
    Returns:
        noisy image in counts
    """
    rng = np.random.default_rng(SEED)
    image = photons2electrons(image, quantum_efficiency=quantum_efficiency)
    sigma_background_noise *= electrons_per_count  # electrons;  40 counts = 40 * .22 electrons per count
    shot_noise = poisson_noise(image, rng=rng)   # shot noise in electrons
    dark_read_noise = normal_noise(mean=0, sigma=sigma_background_noise, size=image.shape, rng=rng)  # dark image in electrons
    print("image[0]\n", image[0])
    print("shot_noise[0]\n", shot_noise[0])
    print("dark_read_noise[0]\n", dark_read_noise[0])

    image += shot_noise + dark_read_noise
    image = electrons2counts(image, electrons_per_count=electrons_per_count)

    # image += mean_background_offset    # add camera offset (camera offset in counts)
    image[image < 0] = 0
    return image.astype(np.float32)

def noise_efficient(image, mean_background_offset=100, sigma_background_noise=40, quantum_efficiency=.82, electrons_per_count=.22):
    rng = np.random.default_rng(SEED)
    original_dtype = image.dtype
    image_batch = image.astype(np.float32)
    qe = quantum_efficiency
    epc = electrons_per_count
    mean_offset = mean_background_offset
    sigma_bg = sigma_background_noise
    # sensor pipeline with noise: photons → noisy counts
    # 1. Convert photons → electrons
    image_batch *= qe

    # 2. Compute shot noised electrons (Poisson thinned by QE) 
    # Shot noise alone should be done in photon space (e.g. photons arrival ~ Poisson(irradiance))
    # However, we actually want to sample from the total random process (photon arrival AND detection).
    # Because photon arrival is a hidden variable
    # we can sample from the marginal distribution of detection as a Poisson thinning process.        
    # Where detected photons = Poisson(n_photons_arrived * QE) = Poisson(electrons)
    # https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/14%3A_The_Poisson_Process/14.05%3A_Thinning_and_Superpositon
    photons_detected = rng.poisson(image_batch)
    # 3. Compute dark/read noise (Gaussian) in electron space
    dark_read_noise = rng.normal(
        loc=0,
        scale=sigma_bg * epc,
        size=image_batch.shape, 
    )
    # ) * sigma_bg * epc
    print("image_batch[0]")
    print(image_batch[0])
    print("photons_detected[0]")
    print(photons_detected[0])
    print("dark_read_noise[0]")
    print(dark_read_noise[0])
    # 4. electrons = detected photons + dark/read noise
    image_batch = photons_detected + dark_read_noise
    
    # 5. Convert electrons → counts
    image_batch /= epc
    
    # Add camera background offset
    # image_batch += mean_offset
    
    # Clip to valid range [0, 65535] for uint16
    # TODO: Should we clamp to 0-65535 or just min=0?
    image_batch = np.clip(image_batch, a_min=0, a_max=65535)
    return image_batch.astype(original_dtype)

# -----------------------------------------------------------------------------

def _scipy_gaussian_filter_3d_ref(arr: np.ndarray, sigma: float, radius: int) -> np.ndarray:
    """Reference 3D Gaussian blur via scipy. arr shape (Z,Y,X).
    Uses constant (zero) padding to match GaussianBlur3d."""
    from scipy.ndimage import gaussian_filter1d

    out = arr.astype(np.float64)
    for axis in (0, 1, 2):
        out = gaussian_filter1d(
            out, sigma=sigma, axis=axis, mode="constant", cval=0.0, radius=radius
        )
    return out


def _gaussian_1d_kernel_for_ref(sigma: float, kernel_size: int, device, dtype) -> torch.Tensor:
    """Build normalized 1D Gaussian kernel (same formula as noise._gaussian_kernel_1d)."""
    k = kernel_size
    half = (k - 1) / 2.0
    x = torch.arange(k, device=device, dtype=dtype) - half
    kernel = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel  # (k,)


def _full_3d_gaussian_conv_ref(
    x: torch.Tensor, sigma: float, kernel_size: int
) -> torch.Tensor:
    """Reference: one 3D conv with kernel = outer product of 1D Gaussians.
    Mathematically equivalent to separable 3x 1D conv when 1D kernels are normalized."""
    B, C, Z, Y, X = x.shape
    device, dtype = x.device, x.dtype
    pad = kernel_size // 2
    k_1d = _gaussian_1d_kernel_for_ref(sigma, kernel_size, device, dtype)  # (k,)
    # Outer product: K3d[z,y,x] = k_1d[z] * k_1d[y] * k_1d[x]; sum(K3d) = 1
    K_3d = (
        k_1d.view(kernel_size, 1, 1)
        * k_1d.view(1, kernel_size, 1)
        * k_1d.view(1, 1, kernel_size)
    )  # (k,k,k)
    # conv3d weight: (C_out, C_in, kZ, kY, kX); groups=C -> (C, 1, k, k, k)
    w = K_3d.unsqueeze(0).unsqueeze(0).expand(C, 1, kernel_size, kernel_size, kernel_size)
    x_pad = F_nn.pad(
        x, (pad, pad, pad, pad, pad, pad), mode="constant", value=0.0
    )  # (left_x, right_x, left_y, right_y, left_z, right_z)
    out = F_nn.conv3d(x_pad, w, groups=C)
    return out


# -----------------------------------------------------------------------------
# MixedPoissonGaussianNoise tests
# -----------------------------------------------------------------------------

def test_mixed_poisson_gaussian_noise_correctness_vs_ref():
    inputs = torch.full((2, 1, 4, 4), 12.0)
    kwargs = dict(
        quantum_efficiency=0.62,
        electrons_per_count=2.5,
        sigma_background_noise=0.2,
        mean_background_offset=4.0,
    )
    # noise = MixedPoissonGaussianNoise(**kwargs)
    # out = noise(inputs)
    ref = noise_ref(inputs.numpy(), **kwargs)
    efficient_out = noise_efficient(inputs.numpy(), **kwargs)
    # assert efficient_out == ref, "Noise is not the same as reference"
    assert np.allclose(efficient_out, ref), "Noise is not the same as reference"

def test_mixed_poisson_gaussian_noise_reproducible_with_seed():
    inputs = torch.full((2, 1, 4, 4), 12.0)
    kwargs = dict(
        quantum_efficiency=0.62,
        electrons_per_count=2.5,
        sigma_background_noise=0.2,
        mean_background_offset=4.0,
        seed=123,
    )
    noise_a = MixedPoissonGaussianNoise(**kwargs)
    noise_b = MixedPoissonGaussianNoise(**kwargs)
    in_a = inputs.clone()
    in_b = inputs.clone()
    out_a = noise_a(in_a)
    out_b = noise_b(in_b)

    assert torch.allclose(out_a, out_b), "Noise is not reproducible with seed"
    assert out_a.shape == inputs.shape, "Output shape is not the same as input shape"
    assert torch.any(out_a != inputs), "Output is not different from input"
    assert torch.all((out_a >= 0) & (out_a <= 65535)), "Output is not in valid range [0, 65535]"

def test_mixed_poisson_gaussian_noise_tuple_parameters():
    inputs = torch.full((2, 1, 4, 4), 12.0)
    kwargs = dict(
        quantum_efficiency=(0.62, 0.82),
        electrons_per_count=(2.5, 3.5),
        sigma_background_noise=(0.2, 0.4),
        mean_background_offset=(4.0, 6.0),
        seed=123,
    )
    noise_a = MixedPoissonGaussianNoise(**kwargs)
    noise_b = MixedPoissonGaussianNoise(**kwargs)
    in_a = inputs.clone()
    in_b = inputs.clone()
    out_a = noise_a(in_a)
    out_b = noise_b(in_b)

    assert torch.any(out_a[0] != out_a[1]), "Noise is not different for different batch elements"
    print(out_a[0])
    print(out_b[0])
    assert torch.allclose(out_a, out_b), "Noise is not reproducible with tuple parameters"
    assert out_a.shape == inputs.shape, "Output shape is not the same as input shape"
    assert torch.any(out_a != inputs), "Noise not added; output is not different from input"
    assert torch.all((out_a >= 0) & (out_a <= 65535)), "Output is not in valid range [0, 65535]"

def test_mixed_poisson_gaussian_noise_returns_background_for_zero_signal():
    inputs = torch.zeros((1, 1, 2, 2))
    noise = MixedPoissonGaussianNoise(
        quantum_efficiency=1.0,
        electrons_per_count=1.0,
        sigma_background_noise=0.0,
        mean_background_offset=7.0,
        seed=0,
    )

    out = noise(inputs)

    assert torch.allclose(out, torch.full_like(inputs, 7.0)), "Output is not the same as expected background"


def test_mixed_poisson_gaussian_noise_updates_data_tensor_in_dict_only():
    inputs = torch.zeros((1, 1, 3, 3))
    targets = torch.ones((1, 1, 3, 3))
    noise = MixedPoissonGaussianNoise(
        quantum_efficiency=1.0,
        electrons_per_count=1.0,
        sigma_background_noise=0.0,
        mean_background_offset=2.0,
        seed=0,
    )
    batch = {"data_tensor": inputs.clone(), "metainfo": {"targets": [targets.clone()]}}

    out = noise(batch)

    assert out["data_tensor"].shape == inputs.shape, "Output data tensor shape is not the same as input shape"
    assert torch.allclose(out["metainfo"]["targets"][0], targets), "Output targets are not the same as input targets"
    assert torch.all(out["data_tensor"] == 2.0), "Output data tensor is not the same as expected background"


# -----------------------------------------------------------------------------
# GaussianBlur3d tests
# -----------------------------------------------------------------------------


class TestGaussianBlur3dInit:
    """Tests for GaussianBlur3d initialization."""

    def test_init_stores_kernel_size(self):
        blur = GaussianBlur3d(kernel_size=7)
        assert blur.kernel_size == 7

    def test_init_odd_kernel_size(self):
        blur5 = GaussianBlur3d(kernel_size=5)
        blur7 = GaussianBlur3d(kernel_size=7)
        assert blur5.kernel_size == 5
        assert blur7.kernel_size == 7


class TestGaussianBlur3dShape:
    """Tests that GaussianBlur3d preserves input shape."""

    def test_single_sigma_preserves_shape(self):
        blur = GaussianBlur3d(kernel_size=7)
        x = torch.randn(2, 3, 8, 16, 16)
        out = blur(x, sigma=1.0)
        assert out.shape == x.shape

    def test_per_batch_sigma_preserves_shape(self):
        blur = GaussianBlur3d(kernel_size=7)
        x = torch.randn(2, 3, 8, 16, 16)
        sigma = torch.tensor([1.0, 1.5])
        out = blur(x, sigma=sigma)
        assert out.shape == x.shape

    def test_handles_small_spatial_dims(self):
        blur = GaussianBlur3d(kernel_size=5)
        x = torch.randn(1, 1, 4, 4, 4)
        out = blur(x, sigma=1.0)
        assert out.shape == x.shape


class TestGaussianBlur3dSingleSigma:
    """Tests for the single-sigma branch."""

    def test_deterministic(self):
        blur = GaussianBlur3d(kernel_size=7)
        torch.manual_seed(42)
        x = torch.randn(2, 1, 8, 8, 8)
        out1 = blur(x, sigma=1.0)
        out2 = blur(x, sigma=1.0)
        torch.testing.assert_close(out1, out2)

    def test_blurs_smoothes(self):
        blur = GaussianBlur3d(kernel_size=7)
        torch.manual_seed(42)
        x = torch.randn(1, 1, 12, 12, 12)
        out = blur(x, sigma=1.0)
        assert out.var() < x.var()

    def test_correctness_vs_scipy(self):
        kernel_size = 7
        radius = (kernel_size - 1) // 2
        blur = GaussianBlur3d(kernel_size=kernel_size)
        torch.manual_seed(123)
        x = torch.randn(1, 1, 10, 12, 14, dtype=torch.float64)
        out = blur(x, sigma=1.0)
        ref = _scipy_gaussian_filter_3d_ref(x[0, 0].numpy(), sigma=1.0, radius=radius)
        np.testing.assert_allclose(
            out[0, 0].numpy(), ref, rtol=1e-4, atol=1e-4, err_msg="GaussianBlur3d single-sigma vs scipy mismatch"
        )


class TestGaussianBlur3dPerBatchSigma:
    """Tests for the per-batch-sigma branch."""

    def test_different_sigmas_produce_different_blur(self):
        blur = GaussianBlur3d(kernel_size=7)
        torch.manual_seed(42)
        x = torch.randn(2, 1, 8, 8, 8)
        sigma = torch.tensor([1.0, 2.0])
        out = blur(x, sigma=sigma)
        assert not torch.allclose(out[0], out[1])

    def test_same_sigma_per_element_matches_single_sigma(self):
        blur = GaussianBlur3d(kernel_size=7)
        torch.manual_seed(42)
        x = torch.randn(2, 1, 8, 8, 8)
        sigma_per_batch = torch.tensor([1.0, 1.0])
        out_per_batch = blur(x, sigma=sigma_per_batch)
        out_single = blur(x, sigma=1.0)
        torch.testing.assert_close(out_per_batch, out_single, rtol=1e-5, atol=1e-5)

    def test_correctness_vs_scipy_per_batch(self):
        kernel_size = 7
        radius = (kernel_size - 1) // 2
        blur = GaussianBlur3d(kernel_size=kernel_size)
        torch.manual_seed(123)
        x = torch.randn(2, 1, 8, 10, 12, dtype=torch.float64)
        sigma = torch.tensor([1.0, 1.5], dtype=torch.float64)
        out = blur(x, sigma=sigma)
        ref = np.empty_like(x.numpy())
        for b in range(2):
            ref[b, 0] = _scipy_gaussian_filter_3d_ref(
                x[b, 0].numpy(), sigma=float(sigma[b]), radius=radius
            )
        np.testing.assert_allclose(
            out.numpy(), ref, rtol=1e-4, atol=1e-4, err_msg="GaussianBlur3d per-batch vs scipy mismatch"
        )


class TestGaussianBlur3dVsFull3dConv:
    """Correctness of separable implementation vs a single 3D conv with outer-product kernel.
    Catches normalization or kernel-construction bugs."""

    def test_full_3d_kernel_sums_to_one(self):
        """Outer product of normalized 1D kernels should sum to 1 (no double normalization)."""
        for sigma in (0.5, 1.0, 1.5):
            k_1d = _gaussian_1d_kernel_for_ref(sigma, 7, torch.device("cpu"), torch.float64)
            one = torch.tensor(1.0, device=k_1d.device, dtype=k_1d.dtype)
            assert torch.allclose(k_1d.sum(), one), "1D kernel should be normalized"
            K_3d = (
                k_1d.view(7, 1, 1) * k_1d.view(1, 7, 1) * k_1d.view(1, 1, 7)
            )
            assert torch.allclose(
                K_3d.sum(), one, rtol=1e-9, atol=1e-9
            ), "3D outer-product kernel should sum to 1"

    def test_single_sigma_separable_matches_full_3d_conv(self):
        """GaussianBlur3d (separable 1D convs) should match one 3D conv with same kernel."""
        kernel_size = 7
        sigma = 1.0
        blur = GaussianBlur3d(kernel_size=kernel_size)
        torch.manual_seed(123)
        x = torch.randn(1, 2, 8, 10, 12, dtype=torch.float64)
        out_separable = blur(x, sigma=sigma)
        out_full_3d = _full_3d_gaussian_conv_ref(x, sigma, kernel_size)
        torch.testing.assert_close(
            out_separable,
            out_full_3d,
            rtol=1e-9,
            atol=1e-9,
            msg="Separable 3D blur should match full 3D conv (same kernel, no normalizing factor error)",
        )

    def test_single_sigma_matches_full_3d_conv_float32(self):
        """Same as above in float32 (typical training dtype)."""
        kernel_size = 5
        sigma = 0.8
        blur = GaussianBlur3d(kernel_size=kernel_size)
        torch.manual_seed(456)
        x = torch.randn(2, 1, 6, 8, 8, dtype=torch.float32)
        out_separable = blur(x, sigma=sigma)
        out_full_3d = _full_3d_gaussian_conv_ref(x, sigma, kernel_size)
        torch.testing.assert_close(
            out_separable,
            out_full_3d,
            rtol=1e-5,
            atol=1e-5,
            msg="Separable vs full 3D conv mismatch in float32",
        )


class TestGaussianBlur3dEdgeCases:
    """Tests for edge cases and error handling."""

    def test_sigma_shape_mismatch_raises(self):
        blur = GaussianBlur3d(kernel_size=7)
        x = torch.randn(2, 1, 8, 8, 8)
        sigma = torch.tensor([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="sigma must be scalar or shape"):
            blur(x, sigma=sigma)

    def test_single_sigma_accepts_scalar_and_0dim_tensor(self):
        blur = GaussianBlur3d(kernel_size=7)
        x = torch.randn(1, 1, 6, 6, 6)
        out_float = blur(x, sigma=1.0)
        out_int = blur(x, sigma=1)
        out_tensor = blur(x, sigma=torch.tensor(1.0))
        torch.testing.assert_close(out_float, out_int)
        torch.testing.assert_close(out_float, out_tensor)


# -----------------------------------------------------------------------------
# CytosolicHaze unit tests (not redundant)
# -----------------------------------------------------------------------------


class TestCytosolicHazeInit:
    """Tests for CytosolicHaze initialization."""

    def test_init_stores_parameters(self):
        haze = CytosolicHaze(membrane_enhancement_factor=1.5, haze_sigma=1.0)
        assert haze.membrane_enhancement_factor == 1.5
        assert haze.haze_sigma == 1.0

    def test_init_computes_kernel_size(self):
        haze = CytosolicHaze(membrane_enhancement_factor=1.0, haze_sigma=1.0)
        assert haze.kernel_size % 2 == 1
        assert haze.kernel_size >= 6 * 1.0 + 1
        haze_tuple = CytosolicHaze(membrane_enhancement_factor=1.0, haze_sigma=(1.0, 2.0))
        assert haze_tuple.kernel_size % 2 == 1
        assert haze_tuple.kernel_size >= 6 * 2.0 + 1

    def test_init_unsupported_input_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported input_format"):
            CytosolicHaze(membrane_enhancement_factor=1.0, haze_sigma=1.0, input_format="XYZ")


class TestCytosolicHazeShape:
    """Tests that CytosolicHaze preserves input shape."""

    def test_preserves_shape_zyxc(self):
        haze = CytosolicHaze(membrane_enhancement_factor=1.2, haze_sigma=1.0, seed=42, input_format="ZYXC")
        x = torch.randn(2, 8, 12, 12, 2)
        out = haze(x)
        assert out.shape == x.shape

    def test_preserves_shape_czxy(self):
        haze = CytosolicHaze(membrane_enhancement_factor=1.2, haze_sigma=1.0, seed=42, input_format="CZXY")
        x = torch.randn(2, 2, 8, 12, 12)
        out = haze(x)
        assert out.shape == x.shape

    def test_preserves_shape_with_time(self):
        haze = CytosolicHaze(membrane_enhancement_factor=1.2, haze_sigma=1.0, seed=42, input_format="TZYXC")
        x = torch.randn(1, 2, 6, 10, 10, 2)
        out = haze(x)
        assert out.shape == x.shape


class TestCytosolicHazeReproducibility:
    """Tests for reproducibility and single vs tuple sigma."""

    def test_reproducible_with_seed(self):
        haze_a = CytosolicHaze(membrane_enhancement_factor=1.2, haze_sigma=1.0, seed=42, input_format="ZYXC")
        haze_b = CytosolicHaze(membrane_enhancement_factor=1.2, haze_sigma=1.0, seed=42, input_format="ZYXC")
        x = torch.randn(1, 8, 12, 12, 1)
        torch.testing.assert_close(haze_a(x), haze_b(x))

    def test_reproducible_with_tuple_haze_sigma(self):
        haze_a = CytosolicHaze(membrane_enhancement_factor=1.2, haze_sigma=(0.5, 1.5), seed=123, input_format="ZYXC")
        haze_b = CytosolicHaze(membrane_enhancement_factor=1.2, haze_sigma=(0.5, 1.5), seed=123, input_format="ZYXC")
        x = torch.randn(2, 8, 12, 12, 1)
        torch.testing.assert_close(haze_a(x), haze_b(x))

    def test_tuple_haze_sigma_different_per_batch(self):
        haze = CytosolicHaze(membrane_enhancement_factor=1.2, haze_sigma=(0.5, 1.5), seed=42, input_format="ZYXC")
        x = torch.randn(2, 8, 12, 12, 1)
        out = haze(x)
        assert not torch.allclose(out[0], out[1])


class TestCytosolicHazeCorrectness:
    """Tests correctness vs GaussianBlur3d formula."""

    def test_formula_with_factor_one(self):
        haze = CytosolicHaze(membrane_enhancement_factor=1.0, haze_sigma=1.0, seed=42, input_format="ZYXC")
        blur = GaussianBlur3d(kernel_size=haze.kernel_size)
        torch.manual_seed(123)
        x = torch.randn(1, 8, 10, 12, 1, dtype=torch.float64)
        out = haze(x)
        x_channel_first = x.permute(0, 4, 1, 2, 3)
        blur_ref = blur(x_channel_first, sigma=1.0)
        blur_ref_zyxc = blur_ref.permute(0, 2, 3, 4, 1)
        expected = x + blur_ref_zyxc
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_membrane_enhancement_factor_applied(self):
        haze = CytosolicHaze(membrane_enhancement_factor=2.0, haze_sigma=0.5, seed=42, input_format="ZYXC")
        blur = GaussianBlur3d(kernel_size=haze.kernel_size)
        x = torch.ones(1, 6, 8, 8, 1, dtype=torch.float64) * 10.0
        out = haze(x)
        x_channel_first = x.permute(0, 4, 1, 2, 3)
        blur_ref = blur(x_channel_first, sigma=0.5)
        blur_ref_zyxc = blur_ref.permute(0, 2, 3, 4, 1)
        expected = 2.0 * x + blur_ref_zyxc
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


class TestCytosolicHazeDictInput:
    """Tests for dict input handling."""

    def test_updates_data_tensor_only(self):
        haze = CytosolicHaze(membrane_enhancement_factor=1.2, haze_sigma=1.0, seed=42, input_format="ZYXC")
        data_tensor = torch.randn(1, 6, 8, 8, 1)
        targets = torch.ones(1, 6, 8, 8, 1)
        batch = {"data_tensor": data_tensor.clone(), "metainfo": {"targets": [targets.clone()]}}
        out = haze(batch)
        assert out["data_tensor"].shape == data_tensor.shape
        torch.testing.assert_close(out["metainfo"]["targets"][0], targets)
        assert torch.any(out["data_tensor"] != data_tensor)

    def test_missing_data_tensor_raises(self):
        haze = CytosolicHaze(membrane_enhancement_factor=1.0, haze_sigma=1.0, input_format="ZYXC")
        with pytest.raises(KeyError, match="expects 'data_tensor'"):
            haze({"metainfo": {"targets": [torch.zeros(1, 4, 4, 4, 1)]}})

    def test_missing_targets_raises(self):
        haze = CytosolicHaze(membrane_enhancement_factor=1.0, haze_sigma=1.0, input_format="ZYXC")
        with pytest.raises(KeyError, match="expects 'targets'"):
            haze({"data_tensor": torch.zeros(1, 4, 4, 4, 1), "metainfo": {}})


class TestCytosolicHazeEdgeCases:
    """Tests for edge cases and error handling."""

    def test_wrong_input_type_raises(self):
        haze = CytosolicHaze(membrane_enhancement_factor=1.0, haze_sigma=1.0, input_format="ZYXC")
        with pytest.raises(TypeError, match="expects torch.Tensor or dict"):
            haze([1, 2, 3])
        with pytest.raises(TypeError, match="expects torch.Tensor or dict"):
            haze(None)

    def test_restores_original_dtype(self):
        haze = CytosolicHaze(membrane_enhancement_factor=1.2, haze_sigma=1.0, seed=42, input_format="ZYXC")
        x = torch.randint(0, 100, (1, 6, 8, 8, 1), dtype=torch.uint16)
        out = haze(x)
        assert out.dtype == torch.uint16

    def test_handles_small_spatial(self):
        haze = CytosolicHaze(membrane_enhancement_factor=1.2, haze_sigma=0.5, seed=42, input_format="ZYXC")
        x = torch.randn(1, 1, 4, 4, 4)
        out = haze(x)
        assert out.shape == x.shape


# -----------------------------------------------------------------------------
# CytosolicHaze visualization test (optional, skipped by default)
# -----------------------------------------------------------------------------

@pytest.mark.skip(reason="optional visualization")
def test_cytosolichaze_visualize_sphere():
    """Generate a batch of hollow 3D spheres with different radii, apply CytosolicHaze, and save orthoslice visualizations.
    To run: temporarily remove the @pytest.mark.skip decorator, then:
    pytest tests/data/transforms/test_noise.py::test_cytosolichaze_visualize_sphere -v -s
    Output saved to tests/data/transforms/cytosolichaze_vis_output/
    """
    Z, Y, X = 24, 32, 32
    centers = (Z / 2 - 0.5, Y / 2 - 0.5, X / 2 - 0.5)
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")

    radii = [
        min(Z, Y, X) / 4 - 4,         # Small sphere
        min(Z, Y, X) / 4,             # Medium sphere
        min(Z, Y, X) / 4 + 4,         # Large sphere
    ]
    thickness = 3

    batch_spheres = []
    for ri in radii:
        ro = ri
        ri_inner = ro - thickness
        r = np.sqrt((zz - centers[0]) ** 2 + (yy - centers[1]) ** 2 + (xx - centers[2]) ** 2)
        sphere = ((r <= ro) & (r > ri_inner)).astype(np.float32) * 100.0
        batch_spheres.append(sphere)

    # Stack as batch (B, Z, Y, X, C=1) shape
    x = np.stack(batch_spheres, axis=0)[..., np.newaxis]
    x = torch.from_numpy(x)

    vis_dir = Path(__file__).resolve().parent / "cytosolichaze_vis_output"
    vis_dir.mkdir(parents=True, exist_ok=True)

    haze = CytosolicHaze(
        membrane_enhancement_factor=(0.8, 1.2),
        haze_sigma=(0.8, 1.5),
        seed=42,
        input_format="ZYXC",
        visualization_dir=str(vis_dir),
    )
    out = haze(x)

    assert out.shape == x.shape
    assert torch.any(out != x)
    assert out.dtype == x.dtype

