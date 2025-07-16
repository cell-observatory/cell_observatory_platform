import torch
from typing import Dict, Tuple

from data.data_types import TORCH_DTYPES

class TorchPreprocessor(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, with_masking: bool, mask_generator):
        super().__init__()  
        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        self.with_masking = with_masking
        self.mask_generator = mask_generator

    def forward(self, data_sample: dict) -> dict:
        """
        Preprocess the input data sample to maintain uniform data
        layout accross data loaders.
        """
        inputs, meta = data_sample['data_tensor'], data_sample['metainfo']

        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)

        if torch.isnan(inputs).all() or torch.isinf(inputs).all():
            raise ValueError(f"Invalid training data")

        assert inputs.dtype == self.dtype, f"{inputs.dtype} != {self.dtype}"

        if self.with_masking:
            masks, context_masks, target_masks, \
            original_patch_indices, channels_to_mask = self.mask_generator(inputs.shape[0])

            return {
                'data_tensor': inputs,
                'metainfo': {
                    'masks': [masks] if self.with_masking else None,
                    'context_masks': [context_masks] if self.with_masking else None,
                    'target_masks': [target_masks] if self.with_masking else None,
                    'original_patch_indices': [original_patch_indices] if self.with_masking else None,
                    'channels_to_mask': [channels_to_mask] if self.with_masking else None,
                    **meta
                }
            }
        else:
            return {
                'data_tensor': inputs,
                'metainfo': meta
            }


class DaliPreprocessor(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, with_masking: bool, mask_generator):
        super().__init__()
        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        self.with_masking = with_masking
        self.mask_generator = mask_generator

    def forward(self, data_sample: Tuple[Dict[str, torch.Tensor]]) -> dict:
        """
        Preprocess the input data sample to maintain uniform data
        layout accross data loaders.
        """
        inputs = data_sample[0]['data_tensor']

        if torch.isnan(inputs).all() or torch.isinf(inputs).all():
            raise ValueError(f"Invalid training data")

        assert inputs.dtype == self.dtype, f"{inputs.dtype} != {self.dtype}"

        if self.with_masking:
            masks, context_masks, target_masks, \
            original_patch_indices, channels_to_mask = self.mask_generator(inputs.shape[0])

            return {
                'data_tensor': inputs,
                "metainfo": {
                    'masks': [masks] if self.with_masking else None,
                    'context_masks': [context_masks] if self.with_masking else None,
                    'target_masks': [target_masks] if self.with_masking else None,
                    'original_patch_indices': [original_patch_indices] if self.with_masking else None,
                    'channels_to_mask': [channels_to_mask] if self.with_masking else None,
                }
            }
        else:
            return {
                'data_tensor': inputs,
                'metainfo': {}
            }
