import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.fft import next_fast_len
from skimage.io import imsave

from cell_observatory_platform.data.transforms.psf import ConvolveWithPSF


def create_gaussian_psf(shape: tuple[int, ...], sigma: float = 1.0) -> torch.Tensor:
    """Create a simple 3D Gaussian PSF for testing."""
    z, y, x = shape
    zc, yc, xc = z // 2, y // 2, x // 2
    zz, yy, xx = torch.meshgrid(
        torch.arange(z), torch.arange(y), torch.arange(x), indexing="ij"
    )
    psf = torch.exp(
        -((zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2) / (2 * sigma**2)
    )
    return psf / psf.sum()  # Normalize


class TestConvolveWithPSFInitialization:
    """Tests for ConvolveWithPSF initialization."""

    def test_init_with_tensor_psf(self):
        """Test initialization with a tensor PSF."""
        psf = create_gaussian_psf((5, 5, 5))
        input_shape = (8, 16, 16, 1)  # ZYXC

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_centered=True,
        )

        assert transform.otf is not None

    def test_init_with_file_path(self):
        """Test initialization with a file path to a PSF image."""
        psf_array = create_gaussian_psf((5, 5, 5)).numpy().astype(np.float32)
        input_shape = (8, 16, 16, 1)  # ZYXC

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            imsave(f.name, psf_array)
            psf_path = Path(f.name)

            transform = ConvolveWithPSF(
                psf=psf_path,
                pad_type="reflect",
                input_format="ZYXC",
                input_shape=input_shape,
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
                psf_centered=True,
            )

            assert transform.otf is not None
            psf_path.unlink()  # Clean up

    def test_init_unsupported_input_format_raises(self):
        """Test that unsupported input format raises ValueError."""
        psf = create_gaussian_psf((5, 5, 5))

        with pytest.raises(ValueError, match="Unsupported input_format"):
            ConvolveWithPSF(
                psf=psf,
                pad_type="reflect",
                input_format="YXC",  # Missing Z
                input_shape=(16, 16, 1),
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
            )

    def test_init_unsupported_psf_format_raises(self):
        """Test that unsupported PSF format raises ValueError."""
        psf = create_gaussian_psf((5, 5, 5))

        with pytest.raises(ValueError, match="Unsupported psf_format"):
            ConvolveWithPSF(
                psf=psf,
                pad_type="reflect",
                input_format="ZYXC",
                input_shape=(8, 16, 16, 1),
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="XYZ",  # Unsupported
            )

    def test_init_invalid_psf_type_raises(self):
        """Test that invalid PSF type raises ValueError."""
        with pytest.raises(ValueError, match="PSF must be a path or a tensor"):
            ConvolveWithPSF(
                psf=[1, 2, 3],  # type: ignore[arg-type]  # Invalid type intentional
                pad_type="reflect",
                input_format="ZYXC",
                input_shape=(8, 16, 16, 1),
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
            )

    def test_init_non_3d_psf_raises(self):
        """Test that 2D or 4D PSF raises ValueError with message about 3D shape."""
        # Test 2D PSF
        psf_2d = torch.rand(5, 5)
        with pytest.raises(ValueError, match="Expected PSF to be 3D"):
            ConvolveWithPSF(
                psf=psf_2d,
                pad_type="reflect",
                input_format="ZYXC",
                input_shape=(8, 16, 16, 1),
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
            )

        # Test 4D PSF
        psf_4d = torch.rand(2, 5, 5, 5)
        with pytest.raises(ValueError, match="Expected PSF to be 3D"):
            ConvolveWithPSF(
                psf=psf_4d,
                pad_type="reflect",
                input_format="ZYXC",
                input_shape=(8, 16, 16, 1),
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
            )


class TestConvolveWithPSFConvolution:
    """Tests for PSF convolution functionality."""

    def test_convolve_tensor_preserves_shape(self):
        """Test that convolution preserves the input tensor shape."""
        psf = create_gaussian_psf((5, 5, 5))
        input_shape = (8, 16, 16, 1)  # ZYXC without batch
        data = torch.rand(2, *input_shape)  # Add batch dimension

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
        )

        output = transform(data)
        assert isinstance(output, torch.Tensor)
        assert output.shape == data.shape, "Output shape should match input shape"

    def test_convolve_tensor_preserves_shape_non_zyxc(self):
        """Non-ZYXC layouts are currently unsupported."""
        psf = create_gaussian_psf((5, 5, 5))
        input_shape = (1, 8, 16, 16)  # CZYX without batch
        data = torch.rand(2, *input_shape)  # Add batch dimension

        with pytest.raises(ValueError, match="Unsupported input_format"):
            ConvolveWithPSF(
                psf=psf,
                pad_type="reflect",
                input_format="CZYX",
                input_shape=input_shape,
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
            )

    def test_convolve_tensor_preserves_dtype(self):
        """Test that convolution preserves the original dtype."""
        psf = create_gaussian_psf((5, 5, 5))
        input_shape = (8, 16, 16, 1)
        data = torch.rand(2, *input_shape, dtype=torch.float16)

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
        )

        output = transform(data)
        assert isinstance(output, torch.Tensor)
        assert output.dtype == data.dtype, "Output dtype should match input dtype"

    def test_convolve_delta_function_returns_psf(self):
        """
        Test that convolving an interior delta function approximates the PSF.

        Note: With reflect padding, the effective impulse response depends on distance to the
        boundary. So we place the delta sufficiently far from the edge and compare a local
        window around the impulse to the PSF.
        """
        psf = create_gaussian_psf((11, 11, 11), sigma=2.0)
        input_shape = (21, 21, 21, 1)

        # Create a delta function at the center
        data = torch.zeros(1, *input_shape)
        center = 10
        data[0, center, center, center, 0] = 1.0

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
            psf_centered=True,
        )

        output = transform(data)

        # The local window around the impulse should approximate the PSF (normalized).
        half = psf.shape[0] // 2
        output_spatial = output[0, :, :, :, 0][
            center - half : center + half + 1,
            center - half : center + half + 1,
            center - half : center + half + 1,
        ]
        psf_normalized = psf / psf.max()
        output_normalized = output_spatial / output_spatial.max()

        assert torch.allclose(
            output_normalized, psf_normalized, atol=1e-5
        ), "Convolving delta with PSF should approximate the PSF"

    def test_convolve_uniform_input_remains_uniform(self):
        """Test that convolving uniform input with normalized PSF stays uniform."""
        psf = create_gaussian_psf((5, 5, 5))  # Already normalized
        input_shape = (16, 16, 16, 1)
        uniform_value = 42.0
        data = torch.full((1, *input_shape), uniform_value)

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
        )

        output = transform(data)

        # With reflect padding, a uniform input stays uniform everywhere.
        interior = output[0, :, :, :, 0]
        assert torch.allclose(
            interior, torch.full_like(interior, uniform_value), atol=1e-4
        ), "Uniform input should remain uniform after convolution with normalized PSF"

    def test_convolution_does_not_wrap_around_edges(self):
        """
        Regression test: FFT-based circular convolution wraps energy from one edge to the opposite edge.
        With reflect-padding + crop, we should not see wraparound at the far edge.
        """
        psf = create_gaussian_psf((7, 7, 7), sigma=1.2)
        input_shape = (16, 16, 16, 1)
        data = torch.zeros(1, *input_shape)
        data[0, 0, 0, 0, 0] = 1.0  # impulse at a corner

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
            psf_centered=True,
        )

        output = transform(data)[0, :, :, :, 0]

        # The far corner should be ~0 (no wraparound from the impulse at [0,0,0]).
        assert output[-1, -1, -1].abs().item() < 1e-6

    def test_convolve_non_5d_tensor_raises(self):
        """Verify 4D input tensor raises ValueError about 5D BZYXC."""
        psf = create_gaussian_psf((5, 5, 5))
        input_shape = (8, 16, 16, 1)

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
        )

        # 4D tensor without batch dimension
        data_4d = torch.rand(*input_shape)
        with pytest.raises(ValueError, match="Expected 5D input tensor"):
            transform(data_4d)

    def test_convolve_clamps_negative_values(self):
        """Verify output has no negative values after convolution."""
        psf = create_gaussian_psf((5, 5, 5))
        input_shape = (8, 16, 16, 1)

        # Create data with values that might produce small negatives due to FFT ringing
        data = torch.rand(1, *input_shape) * 0.1
        # Add an impulse near edge to potentially trigger ringing
        data[0, 0, 0, 0, 0] = 10.0

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="zero",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
        )

        output = transform(data)
        assert isinstance(output, torch.Tensor)
        assert (output >= 0).all(), "Output should have no negative values after clamping"

    def test_convolve_multichannel_preserves_shape(self):
        """Verify multi-channel (C>1) convolution preserves shape."""
        psf = create_gaussian_psf((5, 5, 5))
        input_shape = (8, 16, 16, 3)  # 3 channels
        data = torch.rand(2, *input_shape)

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
        )

        output = transform(data)
        assert isinstance(output, torch.Tensor)
        assert output.shape == data.shape, "Multi-channel output shape should match input shape"

    def test_convolve_multichannel_independent(self):
        """Verify each channel is convolved independently."""
        psf = create_gaussian_psf((5, 5, 5))
        input_shape = (8, 16, 16, 2)  # 2 channels
        input_shape_single = (8, 16, 16, 1)

        # Create data where only one channel has signal
        data = torch.zeros(1, *input_shape)
        data[0, 4, 8, 8, 0] = 1.0  # Impulse in channel 0 only

        transform_multi = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
        )

        transform_single = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape_single,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
        )

        output_multi = transform_multi(data)

        # Channel 1 should remain zero (no cross-channel leakage)
        assert torch.allclose(
            output_multi[0, :, :, :, 1],
            torch.zeros_like(output_multi[0, :, :, :, 1]),
            atol=1e-6,
        ), "Channel 1 should remain zero (no cross-channel leakage)"

        # Channel 0 should match single-channel convolution
        data_single = data[..., 0:1]  # Extract channel 0
        output_single = transform_single(data_single)
        assert torch.allclose(
            output_multi[0, :, :, :, 0],
            output_single[0, :, :, :, 0],
            atol=1e-5,
        ), "Channel 0 should match single-channel convolution result"


class TestConvolveWithPSFDictInput:
    """Tests for dict input handling."""

    def test_convolve_dict_updates_data_tensor(self):
        """Test that dict input updates 'data_tensor' key."""
        psf = create_gaussian_psf((5, 5, 5))
        input_shape = (8, 16, 16, 1)
        data = torch.rand(2, *input_shape)

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
        )

        batch: dict = {"data_tensor": data.clone(), "other_key": "preserved"}
        output = transform(batch)
        assert isinstance(output, dict)
        assert "data_tensor" in output
        assert output["other_key"] == "preserved", "Other keys should be preserved"
        assert output["data_tensor"].shape == data.shape

    def test_convolve_dict_missing_data_tensor_raises(self):
        """Test that dict without 'data_tensor' raises KeyError."""
        psf = create_gaussian_psf((5, 5, 5))

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=(8, 16, 16, 1),
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
        )

        with pytest.raises(KeyError, match="data_tensor"):
            transform({"wrong_key": torch.rand(1, 8, 16, 16, 1)})

    def test_convolve_invalid_type_raises(self):
        """Test that invalid input type raises TypeError."""
        psf = create_gaussian_psf((5, 5, 5))

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=(8, 16, 16, 1),
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
        )

        with pytest.raises(TypeError, match="expects torch.Tensor or dict"):
            transform("invalid_input")  # type: ignore[arg-type]


class TestConvolveWithPSFCentering:
    """Tests for PSF centering behavior."""

    def test_centered_otf_matches_explicit_ifftshift_rfftn(self):
        """Test that centered PSF uses ifftshift before rfftn when building OTF (even-sized PSF)."""
        psf = create_gaussian_psf((8, 8, 8), sigma=1.5)
        input_shape = (8, 8, 8, 1)
        input_spatial_shape = input_shape[:3]

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
            psf_centered=True,
        )

        # Match the implementation: use next_fast_len for optimal FFT sizes
        spatial_shape_padded = tuple(
            input_spatial_shape[i] + psf.shape[i] - 1 for i in range(3)
        )
        common_real_space_shape = tuple(int(next_fast_len(s)) for s in spatial_shape_padded)

        # Compute kernel padding to reach common_real_space_shape
        # NOTE: Match the implementation - pad_before rounds up for correct ifftshift centering
        kernel_padding = []
        for fft_size, kernel_size in zip(common_real_space_shape, psf.shape):
            diff = fft_size - kernel_size
            pad_before = (diff + 1) // 2  # rounds up
            pad_after = diff // 2  # rounds down
            kernel_padding.append((pad_before, pad_after))
        kernel_padding_flat = tuple(
            val for pair in reversed(kernel_padding) for val in pair
        )
        psf_padded = torch.nn.functional.pad(
            psf, pad=kernel_padding_flat, mode="constant", value=0
        )
        expected = torch.fft.rfftn(
            torch.fft.ifftshift(psf_padded, dim=(0, 1, 2)),
            s=common_real_space_shape,
            dim=(0, 1, 2),
        )
        got = transform.otf.squeeze()
        assert expected.shape == got.shape
        assert torch.allclose(got, expected, atol=1e-6), "OTF should match explicit ifftshift+rfftn"

    def test_psf_uncentered_raises(self):
        """psf_centered=False is currently unsupported."""
        psf = create_gaussian_psf((7, 7, 7))
        input_shape = (16, 16, 16, 1)

        with pytest.raises(NotImplementedError, match="psf_centered=False"):
            ConvolveWithPSF(
                psf=psf,
                pad_type="reflect",
                input_format="ZYXC",
                input_shape=input_shape,
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
                psf_centered=False,
            )


class TestConvolveWithPSFPixelSize:
    def test_psf_resampling_identity_when_pixel_sizes_equal(self):
        psf = create_gaussian_psf((7, 7, 7), sigma=1.4)
        input_shape = (16, 16, 16, 1)
        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_centered=True,
        )
        assert transform.psf.shape == psf.shape
        assert torch.allclose(transform.psf, psf, atol=1e-6)
        assert torch.isclose(transform.psf.sum(), torch.tensor(1.0), atol=1e-6)

    def test_psf_resampling_changes_shape_when_pixel_sizes_differ(self):
        psf = create_gaussian_psf((7, 7, 7), sigma=1.4)
        input_shape = (16, 16, 16, 1)
        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
            psf_pixel_size_um=(2.0, 1.0, 1.0),  # coarser PSF sampling in Z → should upsample Z by ~2x
            psf_centered=True,
        )
        assert transform.psf.shape == (14, 7, 7)
        assert torch.isclose(transform.psf.sum(), torch.tensor(1.0), atol=1e-6)

    def test_psf_resampling_downsamples_when_psf_finer(self):
        """Verify PSF downsamples when PSF pixel size < input pixel size."""
        psf = create_gaussian_psf((14, 14, 14), sigma=2.8)
        input_shape = (16, 16, 16, 1)
        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="reflect",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
            psf_pixel_size_um=(0.5, 0.5, 0.5),  # finer PSF sampling → should downsample by ~2x
            psf_centered=True,
        )
        assert transform.psf.shape == (7, 7, 7)
        assert torch.isclose(transform.psf.sum(), torch.tensor(1.0), atol=1e-6)


class TestConvolveWithPSFPadding:
    """Tests for padding behavior."""

    def test_zero_padding_preserves_shape(self):
        """Verify pad_type='zero' produces correct output shape."""
        psf = create_gaussian_psf((5, 5, 5))
        input_shape = (8, 16, 16, 1)
        data = torch.rand(2, *input_shape)

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="zero",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
        )

        output = transform(data)
        assert isinstance(output, torch.Tensor)
        assert output.shape == data.shape, "Output shape should match input shape"

    def test_zero_padding_uniform_input(self):
        """Confirm zero padding on uniform input (edge values differ from reflect)."""
        psf = create_gaussian_psf((5, 5, 5))
        input_shape = (16, 16, 16, 1)
        uniform_value = 42.0
        data = torch.full((1, *input_shape), uniform_value)

        transform = ConvolveWithPSF(
            psf=psf,
            pad_type="zero",
            input_format="ZYXC",
            input_shape=input_shape,
            input_pixel_size_um=(1.0, 1.0, 1.0),
            psf_pixel_size_um=(1.0, 1.0, 1.0),
            psf_format="ZYX",
        )

        output = transform(data)

        # With zero padding, edges should have lower values than interior
        # because the PSF sees zeros outside the boundary.
        interior = output[0, 4:-4, 4:-4, 4:-4, 0]
        edge = output[0, 0, 0, 0, 0]
        assert interior.mean() > edge, "Interior should have higher values than edges with zero padding"

    def test_invalid_pad_type_raises(self):
        """Verify invalid pad_type raises ValueError."""
        psf = create_gaussian_psf((5, 5, 5))

        with pytest.raises(ValueError, match="Unsupported pad_type"):
            ConvolveWithPSF(
                psf=psf,
                pad_type="invalid_padding",
                input_format="ZYXC",
                input_shape=(8, 16, 16, 1),
                input_pixel_size_um=(1.0, 1.0, 1.0),
                psf_pixel_size_um=(1.0, 1.0, 1.0),
                psf_format="ZYX",
            )

