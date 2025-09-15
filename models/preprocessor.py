import time
import torch
from typing import Dict, Tuple

from omegaconf import DictConfig
from hydra.utils import instantiate, get_method

from data.data_types import TORCH_DTYPES


class TorchPreprocessor(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, with_masking: bool, mask_generator, **kwargs):
        super().__init__()  
        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        self.with_masking = with_masking
        self.mask_generator = mask_generator

    def forward(self, data_sample: dict, data_time: float) -> dict:
        """
        Preprocess the input data sample to maintain uniform data
        layout accross data loaders.
        """
        preprocess_time = time.time()

        inputs, meta = data_sample['data_tensor'], data_sample['metainfo']
        # inputs, meta = data_sample['data_tensor'], {}

        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)

        # TODO: this is relatively slow on GPU, consider moving to CPU
        #        or skipping this check
        # if torch.isnan(inputs).all() or torch.isinf(inputs).all():
        #     raise ValueError(f"Invalid training data")

        assert inputs.dtype == self.dtype, f"{inputs.dtype} != {self.dtype}"

        if self.with_masking:
            masking_time = time.time()
            masks, context_masks, target_masks, \
            original_patch_indices, channels_to_mask = self.mask_generator(inputs.shape[0])
            masking_time = time.time() - masking_time

            return {
                'data_tensor': inputs,
                'metainfo': {
                    'masks': [masks] if self.with_masking else None,
                    'context_masks': [context_masks] if self.with_masking else None,
                    'target_masks': [target_masks] if self.with_masking else None,
                    'original_patch_indices': [original_patch_indices] if self.with_masking else None,
                    'channels_to_mask': [channels_to_mask] if self.with_masking else None,
                    'preprocess_time': time.time() - preprocess_time,
                    'data_time': data_time,
                    'masking_time': masking_time,
                    **meta
                }
            }
        else:
            return {
                'data_tensor': inputs,
                'metainfo': meta
            }


class DaliPreprocessor(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, with_masking: bool, mask_generator, **kwargs):
        super().__init__()
        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        self.with_masking = with_masking
        self.mask_generator = mask_generator

    def forward(self, data_sample: Tuple[Dict[str, torch.Tensor]], data_time: float) -> dict:
        """
        Preprocess the input data sample to maintain uniform data
        layout accross data loaders.
        """
        preprocess_time = time.time()
        inputs = data_sample[0]['data_tensor']

        # TODO: this is relatively slow on GPU, consider moving to CPU
        #        or skipping this check
        # if torch.isnan(inputs).all() or torch.isinf(inputs).all():
        #     raise ValueError(f"Invalid training data")

        assert inputs.dtype == self.dtype, f"{inputs.dtype} != {self.dtype}"

        if self.with_masking:
            masking_time = time.time()
            masks, context_masks, target_masks, \
            original_patch_indices, channels_to_mask = self.mask_generator(inputs.shape[0])
            masking_time = time.time() - masking_time

            return {
                'data_tensor': inputs,
                "metainfo": {
                    'masks': [masks] if self.with_masking else None,
                    'context_masks': [context_masks] if self.with_masking else None,
                    'target_masks': [target_masks] if self.with_masking else None,
                    'original_patch_indices': [original_patch_indices] if self.with_masking else None,
                    'channels_to_mask': [channels_to_mask] if self.with_masking else None,
                    'data_time': data_time,
                    'get_item_time': data_sample[0].get('get_item_time', None),
                    'preprocess_time': time.time() - preprocess_time,
                    'masking_time': masking_time,
                }
            }
        else:
            return {
                'data_tensor': inputs,
                'metainfo': {}
            }


class RayPreprocessor(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, with_masking: bool,  mask_generator, **kwargs):
        super().__init__()  
        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        self.with_masking = with_masking
        self.mask_generator = mask_generator
        self.transforms = []
        for t in kwargs.get("transforms_list", []):
            if isinstance(t, DictConfig):
                # not yet instantiated
                self.transforms.append(instantiate(t))
            elif isinstance(t, str):
                # a dotted‑path string
                self.transforms.append(get_method(t))
            else:
                # already an instantiated callable object
                self.transforms.append(t)

    def forward(self, data_sample: dict, data_time: float) -> dict:
        """
        Preprocess the input data sample to maintain uniform data
        layout accross data loaders.
        """
        preprocess_time = time.time()

        if isinstance(data_sample['data_tensor'], list):
            inputs = [t.to("cuda", non_blocking=True) for t in data_sample['data_tensor']]
            inputs = torch.cat(inputs, dim=0)
        else:
            inputs = data_sample['data_tensor'].to("cuda", non_blocking=True)
        
        if inputs.dtype != self.dtype:
            # ray.logger.warning(f"Casting inputs to {self.dtype}")
            inputs = inputs.to(self.dtype)
            
        meta = data_sample['metainfo']
        
        # skipping checks for NaN/Inf values
        # if torch.isnan(inputs).all() or torch.isinf(inputs).all():
        #     raise ValueError(f"Invalid training data")

        if self.transforms is not None:
            transform_t0 = time.time()
            for transform in self.transforms:
                inputs = transform(inputs)
            transform_time = time.time() - transform_t0

        assert inputs.dtype == self.dtype, f"{inputs.dtype} != {self.dtype}"
        
        if self.with_masking:
            masking_time = time.time()
            masks, context_masks, target_masks, \
            original_patch_indices, channels_to_mask = self.mask_generator(inputs.shape[0])
            masking_time = time.time() - masking_time

            return {
                'data_tensor': inputs,
                'metainfo': {
                    'masks': [masks] if self.with_masking else None,
                    'context_masks': [context_masks] if self.with_masking else None,
                    'target_masks': [target_masks] if self.with_masking else None,
                    'original_patch_indices': [original_patch_indices] if self.with_masking else None,
                    'channels_to_mask': [channels_to_mask] if self.with_masking else None,
                    'preprocess_time': time.time() - preprocess_time,
                    'data_time': data_time,
                    'masking_time': masking_time,
                    'transform_time': transform_time if self.transforms is not None else -1,
                    **meta
                }
            }
        else:
            return {
                'data_tensor': inputs,
                'metainfo': {}
            }