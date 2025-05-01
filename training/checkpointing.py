import os
import logging
import warnings
from pathlib import Path

from typing import Dict, Optional, Union, Literal
from collections import OrderedDict

from omegaconf import DictConfig

import torch
from torch.optim import Optimizer

from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from data.data_types import TORCH_DTYPES

def _strip_prefix(state_dict: dict, prefix: str = "module.") -> dict:
    """
    If the keys in `state_dict` are all prefixed with `prefix`, remove it.
    Otherwise return the dict unchanged.
    """
    # do all keys start with the prefix?
    if all(key.startswith(prefix) for key in state_dict.keys()):
        new_state = OrderedDict()
        for key, value in state_dict.items():
            new_key = key[len(prefix):]
            new_state[new_key] = value
        return new_state
    else:
        return state_dict


def _add_prefix(state_dict: Dict[str, torch.Tensor],
                prefix: str = "module.") -> Dict[str, torch.Tensor]:
    """
    If none of the keys start with `prefix`, add it to every key.
    Otherwise return the dict unchanged.
    """
    if all(k.startswith(prefix) for k in state_dict):
        return state_dict  # already has the prefix
    return OrderedDict((f"{prefix}{k}", v) for k, v in state_dict.items() if not k.startswith(prefix))


def load_checkpoint(
    model_engine: DeepSpeedEngine,
    opt: Optional[Optimizer],
    config: DictConfig,
    logger: logging.Logger,
    checkpointdir: Optional[Union[str, Path]],
    ckpt_suffix: str = "best",
    dtype: Optional[Literal["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"]] = None,
):
    logger.info(f"Loading pretrained model @ resumed checkpoint -> {checkpointdir}")

    if dtype is not None:
        try:
            target_dtype = TORCH_DTYPES[dtype]
        except KeyError:
            raise ValueError(f"Unsupported dtype '{dtype}'. Valid: {list(TORCH_DTYPES)}")

    if getattr(config.deepspeed.zero_optimization, "stage", None) == 3:
        # load_checkpoint returns a tuple (load_dir, client_states) with load_dir
        # not being None if the checkpoint was loaded successfully
        load_path, _ = model_engine.load_checkpoint(checkpointdir, tag=f"{ckpt_suffix}_model")
        logger.info(f"DeepSpeed.load_checkpoint returned load_path={load_path!r}")
        if load_path is None:
            raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {checkpointdir}")
        if dtype is not None:
            model_engine.module.to(target_dtype)
        return

    model_path = Path(os.path.join(checkpointdir, f"{ckpt_suffix}_model.bin"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_state = torch.load(model_path, map_location="cpu")

    ckpt_has_module = any(k.startswith("module.") for k in model_state)
    model_expects_mod = any(k.startswith("module.") for k in model_engine.state_dict())

    if ckpt_has_module and not model_expects_mod:
        # Loading a DDP checkpoint into a non‑DDP model
        model_state = _strip_prefix(model_state, prefix="module.")
    elif not ckpt_has_module and model_expects_mod:
        # Loading a non‑DDP checkpoint into a DDP‑wrapped model
        model_state = _add_prefix(model_state, prefix="module.")

    model_engine.load_state_dict(model_state)

    if dtype is not None:
        module = getattr(model_engine, "module", model_engine)
        module.to(target_dtype)

    if opt is not None:
        optimizer_path = Path(os.path.join(checkpointdir, f"{ckpt_suffix}_optimizer.bin"))
        if optimizer_path.exists():
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            opt.load_state_dict(optimizer_state)
        else:
            warnings.warn(FileNotFoundError(f"Optimizer file not found: {optimizer_path}"))


# useful utility function to convert a deepspeed zero stage 3 checkpoint to a standard checkpoint
def convert_zero_checkpoint(checkpoint_path: str, output_dir: str, tag: str = "best_model", ckpt_prefix: str = "best"):
    state = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path, tag)
    output_path = os.path.join(output_dir, f"{ckpt_prefix}_model.bin")
    torch.save(state, output_path)