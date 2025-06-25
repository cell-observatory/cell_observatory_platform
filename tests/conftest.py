from pathlib import Path
import pytest

import torch

@pytest.fixture(scope="session")
def kargs():
    repo = Path.cwd()
    kargs = dict(
        repo=repo,
        outdir=repo/'pretrained_models',
        input_shape=64,
        modes=15,
        batch_size=512,
        hidden_size=768,
        patches=32,
        heads=16,
        repeats=4,
        opt='lamb',
        lr=5e-4,
        wd=5e-5,
        ld=None,
        ema=(.998, 1.),
        epochs=5,
        warmup=1,
        cooldown=1,
        clip_grad=.5,
        fixedlr=False,
        dropout=0.1,
        fixed_dropout_depth=False,
        amp='fp16',
        finetune=None,
        profile=False,
        workers=1,
        gpu_workers=1,
        cpu_workers=8,
    )
    return kargs


def get_input_data(model, inputs):
    input_data = ({"data_tensor": torch.randn(*inputs), "metainfo": {}},)
    return input_data


def get_masked_input_data(model, inputs):
    n_patches = model.get_num_patches()
    context_len = int(n_patches * (1 - model.mask_ratio))
    context_idx = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    target_idx  = torch.arange(context_len, n_patches, dtype=torch.long).unsqueeze(0)

    meta = {
        "masks":                [torch.ones(n_patches, dtype=torch.long).unsqueeze(0)],
        "context_masks":        [context_idx],
        "target_masks":         [target_idx],
        "original_patch_indices": [torch.arange(n_patches, dtype=torch.long)],
    }

    # summary() will unpack the input data but the fwd function in 
    # JEPA and MAE models expects a dict hence we wrap the input data
    # in a tuple with a single dict element
    input_data = ({"data_tensor": torch.randn(*inputs), "metainfo": meta},)
    return input_data