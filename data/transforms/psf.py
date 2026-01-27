from typing import Any


import torch
from pathlib import Path
from os import PathLike
from skimage.io import imread
from scipy.fft import next_fast_len
import logging

logger = logging.getLogger(__name__)

class ConvolveWithPSF:
    def __init__(
        self, 
        psf: torch.Tensor | PathLike[str],
        pad_type: str,
        input_format: str,
        input_shape: tuple[int, ...],
        input_pixel_size_um: tuple[float, float, float],
        psf_format: str,
        psf_pixel_size_um: tuple[float, float, float],
        *,
        psf_centered: bool = True,
        visualization_dir: str | None = None,
    ):
        # NOTE: Always normalize spatial dimensions to ZYX for consistency
        # between input and PSF.
        if input_format == "ZYXC":
            z_idx = input_format.index("Z")
            y_idx = input_format.index("Y")
            x_idx = input_format.index("X")
            self.data_spatial_dim_indices = (z_idx, y_idx, x_idx)
            self.data_spatial_dim_indices_batched = (z_idx + 1, y_idx + 1, x_idx + 1) # +1 to account for batch dimension
            self.input_spatial_shape = (input_shape[z_idx], input_shape[y_idx], input_shape[x_idx])
        else:
            raise ValueError(f"Unsupported input_format {input_format}")
        if psf_format == "ZYX":
            z_idx = psf_format.index("Z")
            y_idx = psf_format.index("Y")
            x_idx = psf_format.index("X")
            self.psf_spatial_dim_indices = (z_idx, y_idx, x_idx)
        else:
            raise ValueError(f"Unsupported psf_format {psf_format}")
        if isinstance(psf, PathLike) or isinstance(psf, str):
            psf = torch.from_numpy(imread(Path(psf)))
        elif isinstance(psf, torch.Tensor):
            pass
        else:
            raise ValueError(f"PSF must be a path or a tensor, got {type(psf)}")
        
        if not psf_centered:
            raise NotImplementedError("psf_centered=False is not supported (expects a centered PSF).")
        
        self.visualization_dir: Path | None = (
            Path(visualization_dir) if visualization_dir is not None else None
        )
        if self.visualization_dir is not None:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        if pad_type not in ["reflect", "zero"]:
            raise ValueError(f"Unsupported pad_type {pad_type}")
        self.pad_type = pad_type


        # PSF preprocessing
        # Convert to float32 for FFT precision
        psf = psf.to(dtype=torch.float32)
        if psf.ndim != 3:
            raise ValueError(f"Expected PSF to be 3D (ZYX); got shape {tuple(psf.shape)}")

        # DEBUG: Plot PSF
        if self.visualization_dir is not None:
            self._plot_psf_orthoslices(
                psf, 
                title="Input PSF", 
                filename="psf_0_input.png",
                display="abs",
            )

        # Resize PSF to match input pixel sampling size while preserving physical extent:
        # new_shape[i] ~= old_shape[i] * (psf_pixel_size_um[i] / input_pixel_size_um[i]).
        scale_zyx = tuple(psf_pixel_size_um[i] / input_pixel_size_um[i] for i in range(3))
        target_shape = tuple(max(1, int(round(psf.shape[i] * scale_zyx[i]))) for i in range(3))
        print(f"\n\nDEBUG ConvolveWithPSF: psf.shape={psf.shape}, target_shape={target_shape}, scale_zyx={scale_zyx}")
        if target_shape != tuple(psf.shape):
            psf = psf.view(1, 1, *psf.shape)
            psf = torch.nn.functional.interpolate(
                psf,
                size=target_shape,
                mode="trilinear",
                align_corners=False,
            )[0, 0]
        # Normalize PSF to unit sum
        psf = psf / psf.sum()
        # Keep processed PSF for tests/inspection.
        self.psf = psf
        
        # DEBUG: Plot PSF
        if self.visualization_dir is not None:
            self._plot_psf_orthoslices(
                self.psf, 
                title="Normalized & Resized PSF", 
                filename="psf_1_normalized_resized.png",
                display="abs",
            )

        # We perform convolution via FFT on a reflected extension of the image to avoid
        # circular wraparound artifacts and reduce ringing near borders. Concretely, we:
        #   - reflect-pad the input by floor(psf_size/2) on each side (per spatial axis),
        #   - do FFT-conv at the padded size,
        #   - crop back to the original size.
        # This yields a "same"-shaped output with reflect boundary conditions.
        
        # The common padded real-space spatial shape used for FFT convolution.
        spatial_shape_padded = tuple(
            spatial_shape + kernel_shape - 1
            for spatial_shape, kernel_shape 
            in zip(self.input_spatial_shape, psf.shape)
        )
        # Optimal FFT sizes for performance (uses FFTW-style optimal lengths)
        common_real_space_shape = tuple(int(next_fast_len(s)) for s in spatial_shape_padded)
        self.common_real_space_shape = common_real_space_shape
        # Compute optimal padding for sample to make it the same size as the FFT shape
        sample_padding = []
        for fft_size, sample_size in zip(common_real_space_shape, self.input_spatial_shape):
            diff = fft_size - sample_size
            pad_before = diff // 2
            pad_after = diff - pad_before  # Handle odd differences correctly
            sample_padding.append((pad_before, pad_after))
        self.sample_padding: list[tuple[int, int]] = sample_padding
        # NOTE: We unpack the reversed padding tuple to a list of integers for F.pad.
        self.sample_padding_reversed_unpacked: list[int] = [
            val for pair 
            in reversed(sample_padding) 
            for val in pair
            ]

        # Zero-pad the centered PSF symmetrically to reach common_real_space_shape.
        # NOTE: For ifftshift to move the PSF center to index 0, the center must be at N//2.
        # For even N with odd PSF size, we need more padding before than after.
        kernel_padding = []
        for fft_size, kernel_size in zip(common_real_space_shape, psf.shape):
            diff = fft_size - kernel_size
            pad_before = (diff + 1) // 2  # rounds up
            pad_after = diff // 2  # rounds down
            kernel_padding.append((pad_before, pad_after))
        # NOTE: F.pad uses reverse dimension order: (X_l, X_r, Y_l, Y_r, Z_l, Z_r).
        # That is why we reverse the padding tuple.
        kernel_padding = reversed(kernel_padding)
        kernel_padding = tuple(val for pair in kernel_padding for val in pair)
        psf_padded = torch.nn.functional.pad(psf, pad=kernel_padding, mode="constant", value=0)
        
        # DEBUG: Plot PSF
        if self.visualization_dir is not None:
            self._plot_psf_orthoslices(
                psf_padded, 
                title="Padded & Centered PSF", 
                filename="psf_2_padded_centered.png",
                display="abs",
            )
        

        # FFT expects spatial origin at 0,0,0: shift the PADDED CENTERED PSF
        psf_shifted = torch.fft.ifftshift(
            psf_padded, 
            dim=self.psf_spatial_dim_indices,
        )

        if self.visualization_dir is not None:
            # NOTE: After ifftshift, a centered PSF peak moves to index 0 along each spatial axis.
            self._plot_psf_orthoslices(
                psf_shifted,
                title="Shifted & Padded PSF",
                filename="psf_3_shifted_padded.png",
                slices=(0, 0, 0),
                display="abs",
            )
        
        # Compute OTF using rfftn (real-space FFT) to save on memory overhead
        # which is identical because psf is real-valued.
        # NOTE: Use common_real_space_shape to match the FFT size used for data.
        otf = torch.fft.rfftn(
            psf_shifted, 
            s=common_real_space_shape,
            dim=self.psf_spatial_dim_indices,
        )
        
        # Compute shape to broadcast OTF to data dimensions (with batch dim)
        # Use actual OTF sizes instead of -1 since view() only allows one inferred dim
        broadcast_shape = [1] * (len(input_shape) + 1)  # +1 for batch dimension
        for i, spatial_dim_idx in enumerate(self.data_spatial_dim_indices_batched):
            broadcast_shape[spatial_dim_idx] = otf.shape[i]
        
        # Broadcast OTF to data dimensions (with batch dim)
        self.otf = otf.view(*broadcast_shape).contiguous()
        
        
        if self.visualization_dir is not None:
            # For visualization we prefer a full fftn (not rfftn) so the X frequency axis is
            # symmetric around DC after fftshift. This avoids misleading half-spectrum plots.
            otf_full = torch.fft.fftn(
                psf_shifted,
                s=common_real_space_shape,
                dim=self.psf_spatial_dim_indices,
            )
            self._plot_otf_orthoslices(
                otf_full,
                voxel_size_um=None,
                title="OTF orthoslices (log10|.|)",
                filename="otf.png",
                log=True,
            )
        



    def _convolve_with_psf(self, data: torch.Tensor) -> torch.Tensor:
        """
        Convolve data with PSF.
        """
        # Move OTF to same device as data
        self.otf = self.otf.to(device=data.device)

        # DEBUG: Plot data before and after convolution
        if self.visualization_dir is not None:
            logger.warning(f"Visualization directory set to {self.visualization_dir}. Original image batch will be cloned and images will be saved to this directory.")
            original_data = data.clone()

        # Save original dtype to restore after convolution
        dtype = data.dtype

        # Convert data to float32 for FFT precision
        if data.dtype != torch.float32:
            data = data.to(dtype=torch.float32)
        # FFT is much faster if the data is contiguous
        data = data.contiguous()

        # Save original for checking after convolution
        original_shape = tuple(data.shape)

        # Reflect-pad spatial dims (Z, Y, X) to avoid circular convolution wraparound.
        if data.ndim != 5:
            raise ValueError(f"Expected 5D input tensor (BZYXC); got shape {tuple(data.shape)}")

        # BZYXC -> BCZYX for F.pad.
        data = data.permute(0, 4, 1, 2, 3)
        # NOTE: F.pad uses reverse order: (Xl, Xr, Yl, Yr, Zl, Zr)
        if self.pad_type == "reflect":
            data = torch.nn.functional.pad(data, pad=self.sample_padding_reversed_unpacked, mode="reflect")
        elif self.pad_type == "zero":
            data = torch.nn.functional.pad(data, pad=self.sample_padding_reversed_unpacked, mode="constant", value=0)
        else:
            raise ValueError(f"Unsupported pad_type {self.pad_type}")

        # BCZYX -> BZYXC.
        data = data.permute(0, 2, 3, 4, 1)
        assert tuple(data.shape[1:4]) == tuple(self.common_real_space_shape), (
            "\nConvolveWithPSF: Padded data spatial shape does not match common real space shape"
            + f"\n\tPadded data shape: {tuple(data.shape)} (spatial dims: {tuple(data.shape[1:4])})"
            + f"\n\tCommon real space shape: {tuple(self.common_real_space_shape)}"
        )

        # FFT, multiply by OTF, inverse FFT
        data = torch.fft.rfftn(
            data, 
            s=self.common_real_space_shape, 
            dim=self.data_spatial_dim_indices_batched
        )
        data = data * self.otf
        data = torch.fft.irfftn(
            data, 
            s=self.common_real_space_shape,
            dim=self.data_spatial_dim_indices_batched
        )
        
        # Compile list of slices to crop back to original size
        crop_slices = [slice(None)] * data.ndim
        for padding, spatial_dim_idx in zip(self.sample_padding, self.data_spatial_dim_indices_batched):
            crop_slices[spatial_dim_idx] = slice(padding[0], -padding[1])

        # Crop back to original shape
        data = data[tuple(crop_slices)]
        assert data.shape == original_shape, (
            "\nConvolveWithPSF: Processed data shape does not match original shape"
            + f"\n\tProcessed data shape: {tuple(data.shape)}"
            + f"\n\tOriginal data shape: {tuple(original_shape)}"
        )
        
        # Clamp negative values
        data = torch.clamp(data, min=0)

        # Restore original dtype
        data = data.to(dtype=dtype)

        # DEBUG: Plot data before and after convolution
        if self.visualization_dir is not None:
            assert original_data is not None
            self._plot_data_before_and_after_convolution(
                original_data=original_data, 
                data=data.clone(),
            )
        return data

    def __call__(self, data: torch.Tensor | dict) -> torch.Tensor | dict:
        if isinstance(data, torch.Tensor):
            return self._convolve_with_psf(data)
        elif isinstance(data, dict):
            if "data_tensor" not in data:
                raise KeyError("ConvolveWithPSF expects 'data_tensor' in dict.")
            data["data_tensor"] = self._convolve_with_psf(data["data_tensor"])
            return data
        raise TypeError(f"ConvolveWithPSF expects torch.Tensor or dict, got {type(data)}")
    
    def _plot_data_before_and_after_convolution(self, original_data: torch.Tensor, data: torch.Tensor) -> None:
        """
        Plot original data and data after convolution with PSF.
        """
        import matplotlib.pyplot as plt
        import torch
        import numpy as np

        original_data_numpy = original_data.real.squeeze().detach().cpu().numpy()
        data_numpy = data.real.squeeze().detach().cpu().numpy()
        ndim = original_data_numpy.ndim
        if ndim != 3:
            raise ValueError(f"Unsupported ndim {ndim}, expected 3D data")

        zc, yc, xc = [s // 2 for s in original_data_numpy.shape]

        pmin_org, pmax_org = np.percentile(original_data_numpy, (1, 99))

        fig, ax = plt.subplots(2, 3, figsize=(15, 10))

        # Original data orthoslices (top row)
        im00 = ax[0, 0].imshow(original_data_numpy[zc, :, :], vmin=pmin_org, vmax=pmax_org, cmap="gray")
        ax[0, 0].set_title(f"Original (XY) z={zc}")
        plt.colorbar(im00, ax=ax[0, 0], fraction=0.046, pad=0.04)
        im01 = ax[0, 1].imshow(original_data_numpy[:, yc, :], vmin=pmin_org, vmax=pmax_org, cmap="gray")
        ax[0, 1].set_title(f"Original (XZ) y={yc}")
        plt.colorbar(im01, ax=ax[0, 1], fraction=0.046, pad=0.04)
        im02 = ax[0, 2].imshow(original_data_numpy[:, :, xc], vmin=pmin_org, vmax=pmax_org, cmap="gray")
        ax[0, 2].set_title(f"Original (YZ) x={xc}")
        plt.colorbar(im02, ax=ax[0, 2], fraction=0.046, pad=0.04)

        # Convolved data orthoslices (bottom row)
        im10 = ax[1, 0].imshow(data_numpy[zc, :, :], vmin=pmin_org, vmax=pmax_org, cmap="gray")
        ax[1, 0].set_title(f"After PSF (XY) z={zc}")
        plt.colorbar(im10, ax=ax[1, 0], fraction=0.046, pad=0.04)
        im11 = ax[1, 1].imshow(data_numpy[:, yc, :], vmin=pmin_org, vmax=pmax_org, cmap="gray")
        ax[1, 1].set_title(f"After PSF (XZ) y={yc}")
        plt.colorbar(im11, ax=ax[1, 1], fraction=0.046, pad=0.04)
        im12 = ax[1, 2].imshow(data_numpy[:, :, xc], vmin=pmin_org, vmax=pmax_org, cmap="gray")
        ax[1, 2].set_title(f"After PSF (YZ) x={xc}")
        plt.colorbar(im12, ax=ax[1, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        assert self.visualization_dir is not None
        plt.savefig(self.visualization_dir / "data_orthoslices_before_and_after_psf.png")
        plt.close(fig)
    
    def _plot_psf_orthoslices(
        self,
        image: torch.Tensor,
        title: str = "Volume",
        filename: str = "output.png",
        *,
        slices: tuple[int, int, int] | None = None,
        display: str = "abs",
    ) -> None:
        """
        Plot orthogonal slices (XY, XZ, YZ) from a 2D or 3D tensor/array.
        """
        import matplotlib.pyplot as plt
        import torch
        import numpy as np

        img = image.detach()
        if display == "abs":
            img = img.abs()
        elif display == "real":
            img = img.real
        elif display == "logabs":
            img = torch.log1p(img.abs())
        else:
            raise ValueError(f"Unsupported display mode: {display}")

        arr = img.squeeze().cpu().numpy()
        if arr.ndim != 3:
            raise ValueError(
                f"Array for orthoslices must be 3D after squeezing singleton dims; got shape {arr.shape}."
            )
        arr_flat = arr.flatten()
        vmin, vmax = np.percentile(arr_flat[arr_flat != 0], (1, 99))
        print(f"DEBUG _plot_psf_orthoslices: arr.min={arr.min():.6f}, arr.max={arr.max():.6f}, vmin={vmin}, vmax={vmax}")

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        if slices is None:
            zc, yc, xc = [s // 2 for s in arr.shape]
        else:
            zc, yc, xc = slices

        im0 = ax[0].imshow(arr[zc, :, :], cmap="gray", vmin=vmin, vmax=vmax)
        ax[0].set_title(f"{title} (XY) z={zc}")
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        im1 = ax[1].imshow(arr[:, yc, :], cmap="gray", vmin=vmin, vmax=vmax)
        ax[1].set_title(f"{title} (XZ) y={yc}")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        im2 = ax[2].imshow(arr[:, :, xc], cmap="gray", vmin=vmin, vmax=vmax)
        ax[2].set_title(f"{title} (YZ) x={xc}")
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        assert self.visualization_dir is not None
        plt.savefig(self.visualization_dir / filename)
        plt.close(fig)

    def _plot_otf_orthoslices(
        self,
        otf: torch.Tensor,
        *,
        voxel_size_um: tuple[float, float, float] | None,
        title: str = "OTF orthoslices",
        filename: str = "otf.png",
        center: tuple[int, int, int] | None = None,
        log: bool = True,
        gamma: float = 1.0,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = "magma",
    ) -> None:
        """
        Plot OTF orthoslices (XY, XZ, YZ) through the DC-centered frequency volume.

        Notes:
        - We fftshift the magnitude so DC is centered for visualization.
        - If `voxel_size_um=(dz, dy, dx)` is provided, axes are labeled in cycles/µm.
          Otherwise axes are in frequency bins (index units).
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        if self.visualization_dir is None:
            return

        mag = otf.detach().abs()
        mag = torch.fft.fftshift(mag, dim=self.psf_spatial_dim_indices)

        arr = mag.squeeze().cpu().numpy()
        if arr.ndim != 3:
            raise ValueError(
                f"OTF array for orthoslices must be 3D after squeezing singleton dims; got shape {arr.shape}."
            )

        nz, ny, nx = arr.shape
        if center is None:
            cz, cy, cx = nz // 2, ny // 2, nx // 2
        else:
            cz, cy, cx = center

        # Build axes: either physical frequency units (cycles/µm) or index coordinates.
        if voxel_size_um is not None:
            dz, dy, dx = voxel_size_um
            fz = np.fft.fftshift(np.fft.fftfreq(nz, d=dz))
            fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
            fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
            xlab, ylab, zlab = "fX (cycles/µm)", "fY (cycles/µm)", "fZ (cycles/µm)"
            xname, yname, zname = "fX", "fY", "fZ"
        else:
            fz = np.arange(nz) - (nz // 2)
            fy = np.arange(ny) - (ny // 2)
            fx = np.arange(nx) - (nx // 2)
            xlab, ylab, zlab = "fX (bins)", "fY (bins)", "fZ (bins)"
            xname, yname, zname = "fX", "fY", "fZ"

        def _prep(img: np.ndarray) -> np.ndarray:
            out = img.astype(np.float32, copy=False)
            if log:
                out = np.log10(out + 1e-12)
            if gamma != 1.0:
                out = np.sign(out) * (np.abs(out) ** gamma)
            return out

        xy = _prep(arr[cz, :, :])
        xz = _prep(arr[:, cy, :])
        yz = _prep(arr[:, :, cx])

        if vmin is None or vmax is None:
            stack = np.concatenate([xy.ravel(), xz.ravel(), yz.ravel()])
            if vmin is None:
                vmin = float(np.percentile(stack, 1))
            if vmax is None:
                vmax = float(np.percentile(stack, 99))

        # Use constrained_layout so colorbar doesn't overlap the last subplot.
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        fig.suptitle(title)

        im0 = ax[0].imshow(
            xy,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[fx[0], fx[-1], fy[0], fy[-1]],
            aspect="auto",
            interpolation="nearest",
        )
        ax[0].set_title(f"XY ({zname}≈0)")
        ax[0].set_xlabel(xlab)
        ax[0].set_ylabel(ylab)

        ax[1].imshow(
            xz,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[fx[0], fx[-1], fz[0], fz[-1]],
            aspect="auto",
            interpolation="nearest",
        )
        ax[1].set_title(f"XZ ({yname}≈0)")
        ax[1].set_xlabel(xlab)
        ax[1].set_ylabel(zlab)

        ax[2].imshow(
            yz,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[fy[0], fy[-1], fz[0], fz[-1]],
            aspect="auto",
            interpolation="nearest",
        )
        ax[2].set_title(f"YZ ({xname}≈0)")
        ax[2].set_xlabel(ylab)
        ax[2].set_ylabel(zlab)

        # Let constrained_layout manage spacing; avoid tight_layout() which can
        # cause the colorbar to overlap subplots depending on Matplotlib version.
        cbar = fig.colorbar(im0, ax=ax, shrink=0.9, pad=0.02)
        cbar.set_label("log10|OTF|" if log else "|OTF|")

        plt.savefig(self.visualization_dir / filename)
        plt.close(fig)

