from typing import Optional

import torch


def get_input_data(model, inputs, device: Optional[torch.device] = None):
    input_data = ({"data_tensor": torch.randn(*inputs, device=device), "metainfo": {}},)
    return input_data


def get_masked_input_data(model, inputs, device: Optional[torch.device] = None):
    n_patches = model.get_num_patches()
    context_len = int(n_patches * (1 - model.mask_ratio))
    context_idx = torch.arange(context_len, dtype=torch.long, device=device).unsqueeze(0)
    target_idx  = torch.arange(context_len, n_patches, dtype=torch.long, device=device).unsqueeze(0)

    meta = {
        "masks": [torch.ones(n_patches, dtype=torch.long, device=device).unsqueeze(0)],
        "context_masks": [context_idx],
        "target_masks": [target_idx],
        "original_patch_indices": [torch.arange(n_patches, dtype=torch.long, device=device)],
    }

    # summary() will unpack the input data but the fwd function in 
    # JEPA and MAE models expects a dict hence we wrap the input data
    # in a tuple with a single dict element
    input_data = ({"data_tensor": torch.randn(*inputs, device=device), "metainfo": meta},)
    return input_data