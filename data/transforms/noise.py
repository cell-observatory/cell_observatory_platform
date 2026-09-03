import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MixedPoissonGaussianNoise:
    # Models the sensor in absolute counts (Poisson variance == mean), so the
    # count magnitude is a parameter, not just a scale: it needs the exact
    # uint16 values rather than a bfloat16 approximation of them. The
    # preprocessor keeps an fp32 intermediate when any transform sets this.
    reads_raw_counts = True

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

        UNITS CONTRACT: the INPUT is interpreted as incident
        PHOTONS; the OUTPUT is camera COUNTS. The pipeline applies a
        deterministic sensor gain on top of the stochastic terms:

            E[output] ~= (quantum_efficiency / electrons_per_count) * photons
                         + mean_background_offset

        (with the shipped task config, ~3.73x + 100). A clean target snapshotted
        BEFORE this transform therefore lives in the PHOTON domain while the
        noised output lives in the COUNT domain -- see the denoising
        preprocessor yaml for the deliberate training contract built on this.

        Args:
            quantum_efficiency: float or tuple[float, float] representing quantum efficiency of the camera
            electrons_per_count: float or tuple[float, float] representing the conversion factor from electrons to counts
            sigma_background_noise: float or tuple[float, float] representing read noise from the camera in counts
            mean_background_offset: float or tuple[float, float] representing the camera background offset in counts

        If tuple, sample uniformly from the range [min, max] giving a random value for each batch element.

        Sensor pipeline (photons in -> noisy counts out):
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
                # Fold the rank in: a bare shared ${seed} gives every rank the
                # SAME noise stream (step-for-step correlated augmentation
                # across DDP replicas). Lazy import: transforms are built
                # inside Ray workers where the distributed context exists;
                # outside one (unit tests) rank falls back to 0.
                try:
                    from cell_observatory_platform.utils.context import process_rank
                    rank = int(process_rank())
                except Exception:
                    rank = 0
                gen.manual_seed(self.seed + rank)
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
        
        # Convert to float32 for numerical precision. copy=True is load-bearing:
        # when the input is already float32, .to() would return the SAME tensor
        # and the in-place ops below (*=, +=, /=) would mutate the caller's
        # tensor (e.g. the clean denoising target / a reused buffer slot).
        image_batch = image_batch.to(dtype=torch.float32, copy=True)

        if self.visualization_dir is not None:
            logger.warning(f"Visualization directory set to {self.visualization_dir}. Original image batch will be cloned and images will be saved to this directory.")
            original_image_batch = image_batch.clone()
            

        B = image_batch.shape[0]
        device = image_batch.device
        rng = self._get_generator(device)
        
        # Sample parameters for each batch element (image) if parameters are tuples
        if isinstance(self.quantum_efficiency, (tuple, list)):
            qe = torch.empty(B, device=device)
            qe.uniform_(*self.quantum_efficiency, generator=rng)
        else:
            qe = torch.full((B,), self.quantum_efficiency, device=device)
            
        if isinstance(self.electrons_per_count, (tuple, list)):
            epc = torch.empty(B, device=device)
            epc.uniform_(*self.electrons_per_count, generator=rng)
        else:
            epc = torch.full((B,), self.electrons_per_count, device=device)
            
        if isinstance(self.sigma_background_noise, (tuple, list)):
            sigma_bg = torch.empty(B, device=device)
            sigma_bg.uniform_(*self.sigma_background_noise, generator=rng)
        else:
            sigma_bg = torch.full((B,), self.sigma_background_noise, device=device)
            
        if isinstance(self.mean_background_offset, (tuple, list)):
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

