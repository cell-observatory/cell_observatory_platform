"""
Torch-native parallelization for the TorchNativeTrainer: applies, in order,

  1. low-precision linear swap (FP8 / MX)      -- parallelism/quantize.py
  2. activation checkpointing                  -- training/helpers.py (existing)
  3. per-block torch.compile                   -- training/helpers.py (existing)
  4. FSDP2 ``fully_shard`` per block + root    -- this module

The ordering is load-bearing (quantize before AC/compile so the swapped
modules are what gets wrapped/traced; compile before fully_shard so dynamo
never sees the FSDP pre/post hooks). 
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)

from cell_observatory_platform.parallelism.quantize import build_quantize_converter
from cell_observatory_platform.training.helpers import (
    apply_activation_checkpointing,
    apply_compile,
    get_model_optimizations_node,
    yield_transformer_stacks,
)

logger = logging.getLogger("ray")
logger.setLevel(logging.INFO)

TORCH_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_reshard_after_forward(policy: str) -> bool:
    """Map the config policy string to fully_shard's bool (no PP: default=True)."""
    match policy:
        case "always" | "default":
            return True
        case "never":
            return False
        case _:
            raise ValueError(
                f"Invalid fsdp_reshard_after_forward policy: {policy!r}. "
                "Valid: default, always, never."
            )


def _fsdp_module_blocks(opt_node: DictConfig):
    """(module_fqn, block_names) pairs driving per-block fully_shard.

    Uses the dedicated ``fsdp.modules`` list when present, else falls back to
    the ``torch_compile.modules`` list (the stacks are the same modules).
    """
    fsdp_node = opt_node.get("fsdp", None)
    if fsdp_node is not None and fsdp_node.get("modules", None) is not None:
        return fsdp_node.modules
    return opt_node.torch_compile.modules


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    module_blocks,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
    reshard_after_forward: bool = True,
) -> None:
    """fully_shard each transformer block, then the root module.

    The root wrap claims every parameter not owned by a block group (embeddings,
    heads, norms), so the model is fully sharded with one all-gather group per
    transformer block plus one for the remainder.
    """
    fsdp_kwargs = {
        "mesh": dp_mesh,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        ),
    }
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    n_blocks = 0
    for stack_fqn, stack in yield_transformer_stacks(module_blocks, model):
        for i, block in enumerate(stack):
            # skip resharding the last block of each stack: FSDP would
            # immediately prefetch it back for backward (torchtitan trick)
            is_last = i == len(stack) - 1
            fully_shard(
                block,
                **fsdp_kwargs,
                reshard_after_forward=reshard_after_forward and not is_last,
            )
            n_blocks += 1
    if n_blocks == 0:
        raise ValueError(
            "apply_fsdp discovered no transformer blocks — check the "
            "optimizations.models '(module_fqn, block_names)' lists."
        )
    fully_shard(model, **fsdp_kwargs, reshard_after_forward=reshard_after_forward)
    logger.info(
        f"[parallelize] fully_shard applied: {n_blocks} transformer blocks + root "
        f"(mesh={dp_mesh.shape}, param_dtype={param_dtype}, reduce_dtype={reduce_dtype})"
    )


def parallelize(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    cfg: DictConfig,
) -> Tuple[nn.Module, Optional[object]]:
    """quantize -> AC -> compile -> fully_shard. Returns (model, converter|None)."""
    train_cfg = cfg.parallelism.training
    opt_node = get_model_optimizations_node(cfg)

    converter = build_quantize_converter(cfg.parallelism)
    if converter is not None:
        model = converter.convert(model)

    if opt_node.activation_checkpoint.enable:
        logger.info("[parallelize] Applying activation checkpointing...")
        apply_activation_checkpointing(opt_node, model)

    if opt_node.torch_compile.enable:
        if opt_node.torch_compile.range != "block_based":
            raise ValueError(
                "TorchNativeTrainer only supports torch_compile.range="
                "'block_based' (whole-model compile would have to run after "
                f"fully_shard); got {opt_node.torch_compile.range!r}."
            )
        logger.info("[parallelize] Applying torch.compile per block...")
        model = apply_compile(opt_node, model)

    apply_fsdp(
        model,
        dp_mesh,
        module_blocks=_fsdp_module_blocks(opt_node),
        param_dtype=TORCH_DTYPE_MAP[train_cfg.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[train_cfg.mixed_precision_reduce],
        cpu_offload=train_cfg.get("enable_cpu_offload", False),
        reshard_after_forward=_resolve_reshard_after_forward(
            train_cfg.get("fsdp_reshard_after_forward", "default")
        ),
    )
    return model, converter
