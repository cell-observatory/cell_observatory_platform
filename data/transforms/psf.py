import torch
from pathlib import Path
from os import PathLike
from skimage.io import imread

class ConvolveWithPSF:
    def __init__(
        self, 
        psf: torch.Tensor | PathLike[str],
        input_format: str,
        input_shape: tuple[int, ...],
        psf_format: str,
        psf_centered: bool = True,
    ):
        # NOTE: Always normalize spatial dimensions to ZYX for consistency
        # between input and PSF.
        if input_format in ("ZYXC"):
            z_idx = input_format.index("Z")
            y_idx = input_format.index("Y")
            x_idx = input_format.index("X")
            self.data_spatial_dim_indices = (z_idx, y_idx, x_idx)
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

        self.psf = psf.to(dtype=torch.float32)
        

        # Compute shape to broadcast OTF to data dimensions
        broadcast_shape = [1] * len(input_shape) # Used for broadcasting OTF to data dimensions
        for spatial_idx in self.data_spatial_dim_indices:
            broadcast_shape[spatial_idx] = -1
        broadcast_shape = [1] + broadcast_shape # add batch dimension

        # FFT Expects PSF to have spatial orgin at 0,0,0. Shift if necessary.
        # ifftshift shifts (dim[0]//2, dim[1]//2, dim[2]//2) -> (0,0,0)
        if psf_centered:
            psf_shifted = torch.fft.ifftshift(
                self.psf,
                dim=self.psf_spatial_dim_indices
            )
        else:
            psf_shifted = self.psf
            
        # Get the spatial size of the sample to be convolved in real space
        spatial_sizes = tuple(
            input_shape[d] 
            for d in self.data_spatial_dim_indices
        )
        
        # Compute OTF, broadcast to data dimensions
        self.otf = torch.fft.rfftn(
            psf_shifted, 
            s=spatial_sizes,
            dim=self.psf_spatial_dim_indices,
        ).view(*broadcast_shape)
    
    def _convolve_with_psf(self, data: torch.Tensor) -> torch.Tensor:
        """
        Convolve data with PSF.
        """
        dtype = data.dtype
        # Move OTF to same device as data
        self.otf = self.otf.to(device=data.device)
        # Convert data to float32 for FFT precision
        if data.dtype != torch.float32:
            data = data.to(dtype=torch.float32)
        # Get real space size of data for inverse FFT
        spatial_sizes = tuple(data.shape[d] for d in self.data_spatial_dim_indices)
        # FFT, multiply by OTF, inverse FFT
        data = torch.fft.rfftn(data, s=spatial_sizes, dim=self.data_spatial_dim_indices)
        data = data * self.otf
        data = torch.fft.irfftn(data, s=spatial_sizes, dim=self.data_spatial_dim_indices)
        # Return to original dtype
        return data.to(dtype=dtype)

    def __call__(self, data: torch.Tensor | dict) -> torch.Tensor | dict:
        if isinstance(data, torch.Tensor):
            return self._convolve_with_psf(data)
        elif isinstance(data, dict):
            if "data_tensor" not in data:
                raise KeyError("ConvolveWithPSF expects 'data_tensor' in dict.")
            data["data_tensor"] = self._convolve_with_psf(data["data_tensor"])
            return data
        raise TypeError(f"ConvolveWithPSF expects torch.Tensor or dict, got {type(data)}")
    
    def _plot_orthoslices(self, arr, title="Volume", filename="output.png"):
        """
        Plot orthogonal slices (XY, XZ, YZ) from a 2D or 3D tensor/array.
        """
        import matplotlib.pyplot as plt
        import torch

        # Convert to numpy if torch
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()

        arr = arr.squeeze()
        ndim = arr.ndim

        fig, ax = plt.subplots(1, 3 if ndim == 3 else 1, figsize=(12 if ndim == 3 else 4, 4))
        if ndim == 2:
            if isinstance(ax, (list, tuple)):
                ax = ax[0]
            ax.imshow(arr, cmap="gray")
            ax.set_title(title + " (XY)")
        elif ndim == 3:
            zc, yc, xc = [s // 2 for s in arr.shape]
            ax[0].imshow(arr[zc, :, :], cmap="gray")
            ax[0].set_title(f"{title} (XY) z={zc}")
            ax[1].imshow(arr[:, yc, :], cmap="gray")
            ax[1].set_title(f"{title} (XZ) y={yc}")
            ax[2].imshow(arr[:, :, xc], cmap="gray")
            ax[2].set_title(f"{title} (YZ) x={xc}")
        else:
            raise ValueError("Array for orthoslices must be 2D or 3D.")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)

    def plot_psf(self):
        """
        Plot orthogonal slices of PSF.
        """
        self._plot_orthoslices(self.psf, title="PSF", filename="psf.png")

    def plot_otf(self):
        """
        Plot orthogonal slices of OTF.
        """
        self._plot_orthoslices(
            torch.abs(self.otf), 
            title="OTF", 
            filename="otf.png"
        )