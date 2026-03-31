import torch
import torch.nn as nn
import torch.nn.functional as F_nn
from pathlib import Path
import logging
from typing import Literal
logger = logging.getLogger(__name__)


class MixedPoissonGaussianNoise:
    def __init__(
        self, 
        quantum_efficiency: float | tuple[float, float], 
        electrons_per_count: float | tuple[float, float], 
        sigma_background_noise: int| tuple[int, int] | float | tuple[float, float], 
        mean_background_offset: int| tuple[int, int] | float | tuple[float, float],
        seed: int | None = None,
        *,
        visualization_dir: str | None = None,
    ):
        """
        Adds realistic mixed Poisson-Gaussian noise to the input data.
        
        Assumes that the input data is expressed as incident photons (counts).
        
        Args:
            quantum_efficiency: float or tuple[float, float] representing quantum efficiency of the camera
            electrons_per_count: float or tuple[float, float] representing the conversion factor from electrons to counts
            sigma_background_noise: float or tuple[float, float] representing read noise from the camera in counts
            mean_background_offset: float or tuple[float, float] representing the camera background offset in counts

        If tuple, sample uniformly from the range [min, max] giving a random value for each batch element.

        Takes clean images in counts and applies realistic sensor noise model:
                
        sensor pipeline with noise (to generate noisy counts):
        1. Convert photons → electrons using quantum_efficiency
        2. Add shot noise (Poisson) in electron space
        3. Add dark/read noise (Gaussian) in electron space
        4. Convert electrons → counts using electrons_per_count
        5. Add camera background offset (in counts)
        6. Clip to valid uint16 range [0, 65535]

        Poisson noise represents the shot noise modeled as a Poisson distribution 
        where noise is proportional to the product of the intensity of the light and 
        the quantum efficiency of the camera.
        
        Gaussian noise represents the read noise modeled as a Gaussian distribution 
        (dark/read noise from the camera electronics).
        """
        if not isinstance(quantum_efficiency, (float, int)) and not (isinstance(quantum_efficiency, (tuple, list)) and len(quantum_efficiency) == 2):
            raise ValueError("quantum_efficiency must be a float or tuple of two floats")
        if not isinstance(electrons_per_count, (float, int)) and not (isinstance(electrons_per_count, (tuple, list)) and len(electrons_per_count) == 2):
            raise ValueError("electrons_per_count must be a float or tuple of two floats")
        if not isinstance(sigma_background_noise, (float, int)) and not (isinstance(sigma_background_noise, (tuple, list)) and len(sigma_background_noise) == 2):
            raise ValueError("sigma_background_noise must be a float or tuple of two floats")
        if not isinstance(mean_background_offset, (float, int)) and not (isinstance(mean_background_offset, (tuple, list)) and len(mean_background_offset) == 2):
            raise ValueError("mean_background_offset must be a float or tuple of two floats")
        

        self.quantum_efficiency = quantum_efficiency
        self.electrons_per_count = electrons_per_count
        self.sigma_background_noise = sigma_background_noise
        self.mean_background_offset = mean_background_offset
        self.seed = seed
        self._generators: dict[torch.device, torch.Generator] = {}
        
        self.visualization_dir = Path(visualization_dir) if visualization_dir is not None else None
        if self.visualization_dir is not None:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)

    def _get_generator(self, device: torch.device) -> torch.Generator:
        """Get or create a generator for the specified device."""
        if device not in self._generators:
            gen = torch.Generator(device=device)
            if self.seed is not None:
                gen.manual_seed(self.seed)
            self._generators[device] = gen
        return self._generators[device]

    def __call__(self, data: torch.Tensor | dict) -> torch.Tensor | dict:            
        if isinstance(data, torch.Tensor):
            return self._add_noise(data)
        if isinstance(data, dict):
            if "data_tensor" not in data:
                raise KeyError("MixedPoissonGaussianNoise expects 'data_tensor' in dict.")
            if "metainfo" not in data:
                raise KeyError("MixedPoissonGaussianNoise expects 'metainfo' in dict.")
            if "targets" not in data["metainfo"]:
                raise KeyError("MixedPoissonGaussianNoise expects 'targets' in metainfo.")
            data["data_tensor"] = self._add_noise(data["data_tensor"])
            return data
        raise TypeError(f"MixedPoissonGaussianNoise expects torch.Tensor or dict, got {type(data)}")
    
    def _add_noise(self, image_batch: torch.Tensor) -> torch.Tensor:
        # Save original dtype of image_batch
        original_dtype = image_batch.dtype
        
        # Convert to float32 for numerical precision
        image_batch = image_batch.to(dtype=torch.float32)

        if self.visualization_dir is not None:
            logger.warning(f"Visualization directory set to {self.visualization_dir}. Original image batch will be cloned and images will be saved to this directory.")
            original_image_batch = image_batch.clone()
            

        B = image_batch.shape[0]
        device = image_batch.device
        rng = self._get_generator(device)
        
        # Sample parameters for each batch element (image) if parameters are tuples
        if isinstance(self.quantum_efficiency, tuple):
            qe = torch.empty(B, device=device)
            qe.uniform_(*self.quantum_efficiency, generator=rng)
        else:
            qe = torch.full((B,), self.quantum_efficiency, device=device)
            
        if isinstance(self.electrons_per_count, tuple):
            epc = torch.empty(B, device=device)
            epc.uniform_(*self.electrons_per_count, generator=rng)
        else:
            epc = torch.full((B,), self.electrons_per_count, device=device)
            
        if isinstance(self.sigma_background_noise, tuple):
            sigma_bg = torch.empty(B, device=device)
            sigma_bg.uniform_(*self.sigma_background_noise, generator=rng)
        else:
            sigma_bg = torch.full((B,), self.sigma_background_noise, device=device)
            
        if isinstance(self.mean_background_offset, tuple):
            mean_offset = torch.empty(B, device=device)
            mean_offset.uniform_(*self.mean_background_offset, generator=rng)
        else:
            mean_offset = torch.full((B,), self.mean_background_offset, device=device)

        # Broadcast to match inputs shape e.g. (B, T, Z, Y, X, C) => (B, 1, 1, 1, 1, 1)
        qe = qe.view(-1, *[1] * (image_batch.ndim - 1))
        epc = epc.view(-1, *[1] * (image_batch.ndim - 1))
        sigma_bg = sigma_bg.view(-1, *[1] * (image_batch.ndim - 1))
        mean_offset = mean_offset.view(-1, *[1] * (image_batch.ndim - 1))

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
        photons_detected = torch.poisson(image_batch, generator=rng)
        
        # 3. Compute dark/read noise (Gaussian) in electron space
        dark_read_noise = torch.randn(
            image_batch.shape, 
            device=device, 
            generator=rng
        ) * sigma_bg * epc

        # 4. electrons = detected photons + dark/read noise
        image_batch = photons_detected + dark_read_noise
        
        # 5. Convert electrons → counts
        image_batch /= epc
        
        # Add camera background offset
        image_batch += mean_offset
        
        # Clip to valid range [0, 65535] for uint16
        # TODO: Should we clamp to 0-65535 or just min=0?
        image_batch = torch.clamp(image_batch, min=0, max=65535)
        
        if self.visualization_dir is not None:
            noise_params = {
                "quantum_efficiency": self.quantum_efficiency,
                "electrons_per_count": self.electrons_per_count,
                "sigma_background_noise": self.sigma_background_noise,
                "mean_background_offset": self.mean_background_offset,
            }
            self.plot_noise(original_image_batch, image_batch, noise_params)
            
        # Restore original dtype
        image_batch = image_batch.to(dtype=original_dtype)
        return image_batch
    
    def plot_noise(
        self, 
        image_batch: torch.Tensor, 
        image_batch_noised: torch.Tensor,
        noise_params: dict,
    ) -> None:
        """
        Plot the noise added to the image.
        """
        import matplotlib.pyplot as plt
        import torch
        from pathlib import Path
        
        # Assume B, Z, Y, X, C        
        B, Z, Y, X, C = image_batch.shape
        # Convert to numpy if torch
        if isinstance(image_batch, torch.Tensor):
            image_batch_numpy = image_batch.detach().cpu().numpy()
        if isinstance(image_batch_noised, torch.Tensor):
            image_batch_noised_numpy = image_batch_noised.detach().cpu().numpy()

        # noise_params_str = "_".join([f"{k}-{v[0].item():.2f}" for k, v in noise_params.items()])
        
        # Plot central orthoslices (XY, XZ, YZ) for each channel
        # Layout: 2 rows (clean/noised) × (3 * C) columns (XY, XZ, YZ per channel)
        n_cols = 3 * C
        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8), squeeze=False)
        
        for c in range(C):
            col_base = c * 3
            # XY plane (Z slice)
            axes[0, col_base].imshow(image_batch_numpy[0, Z//2, :, :, c], cmap="gray")
            axes[1, col_base].imshow(image_batch_noised_numpy[0, Z//2, :, :, c], cmap="gray")
            axes[0, col_base].set_title(f"Ch {c} XY (clean)")
            axes[1, col_base].set_title(f"Ch {c} XY (noised)")
            
            # XZ plane (Y slice)
            axes[0, col_base + 1].imshow(image_batch_numpy[0, :, Y//2, :, c], cmap="gray", aspect="auto")
            axes[1, col_base + 1].imshow(image_batch_noised_numpy[0, :, Y//2, :, c], cmap="gray", aspect="auto")
            axes[0, col_base + 1].set_title(f"Ch {c} XZ (clean)")
            axes[1, col_base + 1].set_title(f"Ch {c} XZ (noised)")
            
            # YZ plane (X slice)
            axes[0, col_base + 2].imshow(image_batch_numpy[0, :, :, X//2, c], cmap="gray", aspect="auto")
            axes[1, col_base + 2].imshow(image_batch_noised_numpy[0, :, :, X//2, c], cmap="gray", aspect="auto")
            axes[0, col_base + 2].set_title(f"Ch {c} YZ (clean)")
            axes[1, col_base + 2].set_title(f"Ch {c} YZ (noised)")
            
            for i in range(3):
                axes[0, col_base + i].axis("off")
                axes[1, col_base + i].axis("off")
        
        plt.tight_layout()
        plt.savefig(self.visualization_dir / f"noise_plot.png")
        plt.close(fig)



def _gaussian_kernel_1d(
    sigma: torch.Tensor,
    kernel_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build normalized 1D Gaussian kernels.

    Args:
        sigma: scalar tensor or (B,) tensor of sigmas.
        kernel_size: odd integer.
        device, dtype: for the output tensor.

    Returns:
        (1, k) if sigma is scalar, or (B, k) if sigma is (B,).
    """
    k = kernel_size
    half = (k - 1) / 2.0
    x = torch.arange(k, device=device, dtype=dtype) - half  # (k,)
    # sigma -> (N, 1) for broadcast; N = 1 or B
    s = sigma.view(-1, 1).to(dtype)
    kernel = torch.exp(-x.unsqueeze(0) ** 2 / (2.0 * s ** 2))  # (N, k)
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    return kernel


class GaussianBlur3d(nn.Module):
    """
    Separable 3D Gaussian blur (three 1D convolutions along Z, Y, X).

    Complexity is O(3k) per voxel instead of O(k^3) for a full 3D kernel.
    Supports a single sigma (one kernel shared across the whole batch/channel)
    or per-batch sigma (one kernel per batch element).
    Expects input (B, C, Z, Y, X).
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self._pad = self.kernel_size // 2

    def forward(self, x: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, Z, Y, X)
            sigma: scalar (same kernel for all) or (B,) tensor (one kernel per batch element)
        """
        B, C = x.shape[:2]
        k = self.kernel_size
        device = x.device
        dtype = x.dtype
        pad = self._pad

        if isinstance(sigma, (int, float)) or (isinstance(sigma, torch.Tensor) and sigma.ndim == 0):
            return self._blur_single_sigma(x, float(sigma), B, C, k, pad, device, dtype)

        sigma_b = sigma if isinstance(sigma, torch.Tensor) else torch.as_tensor(sigma, device=device, dtype=dtype)
        if sigma_b.shape != (B,):
            raise ValueError(f"sigma must be scalar or shape (B,), got {sigma_b.shape}")
        return self._blur_per_batch_sigma(x, sigma_b, B, C, k, pad, device, dtype)

    # -- single sigma: one kernel, depthwise over all B*C channels ---------

    @staticmethod
    def _blur_single_sigma(
        x: torch.Tensor,
        sigma_val: float,
        B: int, C: int, k: int, pad: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        sigma_t = torch.tensor([sigma_val], device=device, dtype=dtype)
        g = _gaussian_kernel_1d(sigma_t, k, device, dtype)  # (1, k)
        G = B * C
        # Weight shapes for each axis (groups=G applies same kernel per channel)
        wZ = g.view(1, 1, k, 1, 1).expand(G, 1, k, 1, 1).contiguous()
        wY = g.view(1, 1, 1, k, 1).expand(G, 1, 1, k, 1).contiguous()
        wX = g.view(1, 1, 1, 1, k).expand(G, 1, 1, 1, k).contiguous()

        x_flat = x.reshape(1, G, *x.shape[2:])  # (1, B*C, Z, Y, X)
        x_flat = F_nn.pad(x_flat, (0, 0, 0, 0, pad, pad), mode="constant", value=0.0)
        x_flat = F_nn.conv3d(x_flat, wZ, groups=G)
        x_flat = F_nn.pad(x_flat, (0, 0, pad, pad, 0, 0), mode="constant", value=0.0)
        x_flat = F_nn.conv3d(x_flat, wY, groups=G)
        x_flat = F_nn.pad(x_flat, (pad, pad, 0, 0, 0, 0), mode="constant", value=0.0)
        x_flat = F_nn.conv3d(x_flat, wX, groups=G)
        return x_flat.view(B, C, *x_flat.shape[2:])

    # -- per-batch sigma: one kernel per batch element, grouped conv --------

    @staticmethod
    def _blur_per_batch_sigma(
        x: torch.Tensor,
        sigma: torch.Tensor,
        B: int, C: int, k: int, pad: int,
        device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        gs = _gaussian_kernel_1d(sigma, k, device, dtype)  # (B, k)
        G = B * C
        # Each batch element b gets gs[b]; repeat C times for the C channels
        # gs_rep (B*C, k) where gs_rep[b*C + c] = gs[b]
        gs_rep = gs.unsqueeze(1).expand(B, C, k).reshape(G, k)

        wZ = gs_rep.view(G, 1, k, 1, 1)
        wY = gs_rep.view(G, 1, 1, k, 1)
        wX = gs_rep.view(G, 1, 1, 1, k)

        x_flat = x.reshape(1, G, *x.shape[2:])  # (1, B*C, Z, Y, X)
        x_flat = F_nn.pad(x_flat, (0, 0, 0, 0, pad, pad), mode="constant", value=0.0)
        x_flat = F_nn.conv3d(x_flat, wZ, groups=G)
        x_flat = F_nn.pad(x_flat, (0, 0, pad, pad, 0, 0), mode="constant", value=0.0)
        x_flat = F_nn.conv3d(x_flat, wY, groups=G)
        x_flat = F_nn.pad(x_flat, (pad, pad, 0, 0, 0, 0), mode="constant", value=0.0)
        x_flat = F_nn.conv3d(x_flat, wX, groups=G)
        return x_flat.view(B, C, *x_flat.shape[2:])


class CytosolicHaze:
    def __init__(
        self,
        membrane_enhancement_factor: float | tuple[float, float],
        haze_sigma: float | tuple[float, float],
        seed: int | None = None,
        input_format: Literal["ZYXC", "CZXY", "TCZYX", "TZYXC"] = "ZYXC",
        # TODO: Add units and electrons_per_count
        # units: Literal["counts", "photons"] = "counts",
        # electrons_per_count: float | tuple[float, float] = 1.0,
        *,
        visualization_dir: str | None = None,
    ):
        self.membrane_enhancement_factor = membrane_enhancement_factor
        # TODO: Add units and electrons_per_count
        # if units not in ["counts", "photons"]:
        #     raise ValueError(f"Unsupported units {units}")
        # self.units = units
        # self.electrons_per_count = electrons_per_count
        self.input_format = input_format.upper()
        if self.input_format not in ["ZYXC", "CZXY", "TCZYX", "TZYXC"]:
            raise ValueError(f"Unsupported input_format {self.input_format}")
        C = self.input_format.index("C") + 1 # +1 for batch dimension
        if C == len(self.input_format):
            self.channel_last = True
        else:
            self.channel_last = False
        if "T" in self.input_format:
            T = self.input_format.index("T") + 1  # +1 for batch dimension
            self.has_time = True
        else:
            T = None
            self.has_time = False
        if self.channel_last:
            Z = self.input_format.index("Z") + 1  # +1 for batch dimension
            Y = self.input_format.index("Y") + 1
            X = self.input_format.index("X") + 1
            if self.has_time:
                self.to_channel_first_shape = (T, C, Z, Y, X)
                T, C, Z, Y, X = 1, 2, 3, 4, 5
                self.to_channel_last_shape = (T, Z, Y, X, C)
            else:
                self.to_channel_first_shape = (C, Z, Y, X)
                C, Z, Y, X = 1, 2, 3, 4
                self.to_channel_last_shape = (Z, Y, X, C)
        else:
            self.to_channel_first_shape = None
            self.to_channel_last_shape = None
        self.haze_sigma = haze_sigma
        if isinstance(haze_sigma, tuple):
            max_haze_sigma = max(haze_sigma)
        else:
            max_haze_sigma = haze_sigma
        kernel_size = 6 * max_haze_sigma + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd
        self.kernel_size = int(kernel_size)
        self._blur = GaussianBlur3d(self.kernel_size)

        self.seed = seed
        self._generators: dict[torch.device, torch.Generator] = {}
        self.visualization_dir = Path(visualization_dir) if visualization_dir is not None else None
        if self.visualization_dir is not None:
            self.visualization_dir.mkdir(parents=True, exist_ok=True)

    def _get_generator(self, device: torch.device) -> torch.Generator:
        """Get or create a generator for the specified device."""
        if device not in self._generators:
            gen = torch.Generator(device=device)
            if self.seed is not None:
                gen.manual_seed(self.seed)
            self._generators[device] = gen
        return self._generators[device]

    def __call__(self, data: torch.Tensor | dict) -> torch.Tensor | dict:
        if isinstance(data, torch.Tensor):
            return self._add_haze(data)
        if isinstance(data, dict):
            if "data_tensor" not in data:
                raise KeyError("CytosolicHaze expects 'data_tensor' in dict.")
            if "metainfo" not in data:
                raise KeyError("CytosolicHaze expects 'metainfo' in dict.")
            if "targets" not in data["metainfo"]:
                raise KeyError("CytosolicHaze expects 'targets' in metainfo.")
            data["data_tensor"] = self._add_haze(data["data_tensor"])
            return data
        raise TypeError(f"CytosolicHaze expects torch.Tensor or dict, got {type(data)}")

    def _add_haze(self, image_batch: torch.Tensor) -> torch.Tensor:
        # Save original dtype of image_batch
        original_dtype = image_batch.dtype
        
        # Convert to float32 for numerical precision
        image_batch = image_batch.to(dtype=torch.float32)
        
        if self.visualization_dir is not None:
            logger.warning(f"Visualization directory set to {self.visualization_dir}. Original image batch will be cloned and images will be saved to this directory.")
            original_image_batch = image_batch.clone()  
        
        B = image_batch.shape[0]
        device = image_batch.device
        rng = self._get_generator(device)
        
        if isinstance(self.membrane_enhancement_factor, tuple):
            membrane_enhancement_factor = torch.empty(B, device=device)
            membrane_enhancement_factor.uniform_(*self.membrane_enhancement_factor, generator=rng)
            membrane_enhancement_factor = membrane_enhancement_factor.view(B, 1, 1, 1, 1)
        else:
            membrane_enhancement_factor = self.membrane_enhancement_factor

        # Sample haze sigma for each batch element (image) if parameters are tuples
        if isinstance(self.haze_sigma, tuple):
            haze_sigma = torch.empty(B, device=device)
            haze_sigma.uniform_(*self.haze_sigma, generator=rng)
        else:
            haze_sigma = self.haze_sigma  # scalar -> single kernel over batch/channel

        # Channel-first (B, C, Z, Y, X) for blur module
        if self.channel_last:
            to_first = self.to_channel_first_shape
            assert to_first is not None
            x = image_batch.permute(0, *to_first)
        else:
            x = image_batch
        
        if self.has_time:
            T, C, Z, Y, X = x.shape[1:]
            # Squash time -> channel dimension
            # The rationale here is that each experiment would have a fixed
            # haze sigma for all timepoints, so we can group the timepoints together
            # and apply the same kernel to all timepoints.
            x = x.reshape(B, T * C, Z, Y, X)

        # Single sigma -> one kernel over entire batch/channel; per-batch sigma -> grouped conv
        x = self._blur(x, haze_sigma)

        if self.has_time:
            # Unsquash time <-> channel dimension (reuse T, C, Z, Y, X from above)
            x = x.reshape(B, T, C, Z, Y, X)

        # Enhance the membrane intensity before adding haze
        # We found this to be necessary to match the intensity profile of the real membrane data.
        image_batch = image_batch * membrane_enhancement_factor

        # Add the haze to the image
        if self.channel_last:
            to_last = self.to_channel_last_shape
            assert to_last is not None
            image_batch += x.permute(0, *to_last)
        else:
            image_batch += x


        if self.visualization_dir is not None:
            haze_sigma_plot = haze_sigma if isinstance(haze_sigma, torch.Tensor) else torch.full((B,), haze_sigma, device=device)
            membrane_enhancement_factor_plot = membrane_enhancement_factor if isinstance(membrane_enhancement_factor, torch.Tensor) else torch.full((B,), membrane_enhancement_factor, device=device)
            self.plot_haze(original_image_batch, image_batch, membrane_enhancement_factor_plot, haze_sigma_plot)
        
        # Restore original dtype
        image_batch = image_batch.to(dtype=original_dtype)
        return image_batch
    
    def plot_haze(
        self,
        image_batch: torch.Tensor,
        image_batch_hazy: torch.Tensor,
        membrane_enhancement_factor: torch.Tensor,
        haze_sigma: torch.Tensor,
    ) -> None:
        """
        Plot the haze added to the image.
        """
        import matplotlib.pyplot as plt
        import torch
        import numpy as np

        # Assume B, Z, Y, X, C        
        B, Z, Y, X, C = image_batch.shape
        # Convert to numpy if torch
        if isinstance(image_batch, torch.Tensor):
            image_batch_numpy = image_batch.detach().cpu().numpy()
        if isinstance(image_batch_hazy, torch.Tensor):
            image_batch_hazy_numpy = image_batch_hazy.detach().cpu().numpy()
        if isinstance(membrane_enhancement_factor, torch.Tensor):
            membrane_enhancement_factor_numpy = membrane_enhancement_factor.squeeze().detach().cpu().numpy()
        if isinstance(haze_sigma, torch.Tensor):
            haze_sigma_numpy = haze_sigma.squeeze().detach().cpu().numpy()
        for b in range(B):
            # Plot central orthoslices (XY, XZ, YZ) for each channel
            # Layout: 2 rows (clean/hazy) × (3 * C) columns (XY, XZ, YZ per channel)
            n_cols = 3 * C
            fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8), squeeze=False)
            
            for c in range(C):
                col_base = c * 3
                # XY plane (Z slice)
                axes[0, col_base].imshow(image_batch_numpy[b, Z//2, :, :, c], cmap="gray")
                axes[1, col_base].imshow(image_batch_hazy_numpy[b, Z//2, :, :, c], cmap="gray")
                axes[0, col_base].set_title(f"Ch {c} XY (clean)")
                axes[1, col_base].set_title(f"Ch {c} XY (hazy)")
                
                # XZ plane (Y slice)
                axes[0, col_base + 1].imshow(image_batch_numpy[b, :, Y//2, :, c], cmap="gray", aspect="auto")
                axes[1, col_base + 1].imshow(image_batch_hazy_numpy[b, :, Y//2, :, c], cmap="gray", aspect="auto")
                axes[0, col_base + 1].set_title(f"Ch {c} XZ (clean)")
                axes[1, col_base + 1].set_title(f"Ch {c} XZ (hazy)")
                
                # YZ plane (X slice)
                axes[0, col_base + 2].imshow(image_batch_numpy[b, :, :, X//2, c], cmap="gray", aspect="auto")
                axes[1, col_base + 2].imshow(image_batch_hazy_numpy[b, :, :, X//2, c], cmap="gray", aspect="auto")
                axes[0, col_base + 2].set_title(f"Ch {c} YZ (clean)")
                axes[1, col_base + 2].set_title(f"Ch {c} YZ (hazy)")
                
                for i in range(3):
                    axes[0, col_base + i].axis("off")
                    axes[1, col_base + i].axis("off")
            
            plt.tight_layout()
            plt.savefig(self.visualization_dir / f"haze_plot_b{b}_membrane_enhancement_factor_{membrane_enhancement_factor_numpy[b].item():.2f}_haze_sigma_{haze_sigma_numpy[b].item():.2f}.png")
            plt.close(fig)