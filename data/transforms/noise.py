import torch

class MixedPoissonGaussianNoise:
    def __init__(
        self, 
        quantum_efficiency: float | tuple[float, float], 
        electrons_per_count: float | tuple[float, float], 
        sigma_background_noise: int| tuple[int, int] | float | tuple[float, float], 
        mean_background_offset: int| tuple[int, int] | float | tuple[float, float],
        seed: int | None = None,
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
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

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
        B = image_batch.shape[0]
        device = image_batch.device
        
        # Sample parameters for each batch element (image) if parameters are tuples
        if isinstance(self.quantum_efficiency, tuple):
            qe = torch.empty(B, device=device)
            qe.uniform_(*self.quantum_efficiency, generator=self.rng)
        else:
            qe = torch.full((B,), self.quantum_efficiency, device=device)
            
        if isinstance(self.electrons_per_count, tuple):
            epc = torch.empty(B, device=device)
            epc.uniform_(*self.electrons_per_count, generator=self.rng)
        else:
            epc = torch.full((B,), self.electrons_per_count, device=device)
            
        if isinstance(self.sigma_background_noise, tuple):
            sigma_bg = torch.empty(B, device=device)
            sigma_bg.uniform_(*self.sigma_background_noise, generator=self.rng)
        else:
            sigma_bg = torch.full((B,), self.sigma_background_noise, device=device)
            
        if isinstance(self.mean_background_offset, tuple):
            mean_offset = torch.empty(B, device=device)
            mean_offset.uniform_(*self.mean_background_offset, generator=self.rng)
        else:
            mean_offset = torch.full((B,), self.mean_background_offset, device=device)

        # Broadcast to match inputs shape e.g. (B, T, Y, X, C) => (B, 1, 1, 1, 1)
        qe = qe.view(-1, *[1] * (image_batch.ndim - 1))
        epc = epc.view(-1, *[1] * (image_batch.ndim - 1))
        sigma_bg = sigma_bg.view(-1, *[1] * (image_batch.ndim - 1))
        mean_offset = mean_offset.view(-1, *[1] * (image_batch.ndim - 1))

        # sensor pipeline with noise: photons → noisy counts
        # 1. Convert photons → electrons
        image_batch *= qe

        # 2. Compute shot noise (Poisson) in electron space
        shot_noise = torch.poisson(image_batch, generator=self.rng)
        
        # 3. Compute dark/read noise (Gaussian) in electron space
        dark_read_noise = torch.randn(
            image_batch.shape, 
            device=device, 
            generator=self.rng
        ) * sigma_bg * epc

        # 4. Add shot noise and dark/read noise to electrons
        image_batch += shot_noise + dark_read_noise
        
        # 5. Convert electrons → counts
        image_batch /= epc
        
        # Add camera background offset
        image_batch += mean_offset
        
        # Clip to valid range [0, 65535] for uint16
        # TODO: Should we clamp to 0-65535 or just min=0?
        image_batch = torch.clamp(image_batch, min=0, max=65535)
        
        return image_batch
