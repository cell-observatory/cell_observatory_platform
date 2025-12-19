from logging import getLogger

import torch
import torch.nn as nn

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._composable.replicate import replicate
from torch.distributed.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
)
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    PrepareModuleOutput,
    parallelize_module,
)

from torchtitan.distributed import ParallelDims
from torchtitan.config.job_config import JobConfig, Parallelism, Compile
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp

from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_platform.training.helpers import (
    apply_activation_checkpointing,
    apply_compile,
)

from cell_observatory_platform.parallelism.utils import (
    replicate_parameter_on_mesh,
    replicate_module_params_on_mesh,
    distribute_param_on_mesh
)

logger = getLogger(__name__)


def PARALLELIZE(
    model: nn.Module,
    parallel_dims: ParallelDims,
    cfg,
):
    """
    Apply, in order:
      1. Tensor + sequence (context) parallelism (TP + CP)
      2. Activation checkpointing
      3. torch.compile
      4. FSDP2 or DDP fallback if FSDP disabled.
    """

    world_mesh = parallel_dims.world_mesh

    # ------------------------------------------------------------------
    # 0. Parallelism CFG Checks
    # ------------------------------------------------------------------
    
    check_parallelism_cfg(cfg, model, parallel_dims)

    # ------------------------------------------------------------------
    # 2. Tensor + Sequence Parallelism
    # ------------------------------------------------------------------    

    # FIXME: not supported yet on this branch

    # ------------------------------------------------------------------
    # 2. Activation checkpointing
    # ------------------------------------------------------------------

    if getattr(cfg.optimizations.models.activation_checkpoint, "enable"):
        apply_activation_checkpointing(cfg, model)

    # ------------------------------------------------------------------
    # 3. torch.compile (block-based or full)
    # ------------------------------------------------------------------

    if getattr(cfg.optimizations.models.torch_compile, "enable"):
        model = apply_compile(cfg, model)

    # ------------------------------------------------------------------
    # 4. FSDP2 OR DDP fallback
    # ------------------------------------------------------------------
    
    if parallel_dims.fsdp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

        param_dtype = TORCH_DTYPES[cfg.parallelism.training.mixed_precision_param].value
        reduce_dtype = TORCH_DTYPES[cfg.parallelism.training.mixed_precision_reduce].value

        apply_fsdp(
            model=model,
            dp_mesh=dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            pp_enabled=False,  # we do not currently support pipeline parallelism
            cpu_offload=getattr(cfg.parallelism.training, "enable_cpu_offload"),
            reshard_after_forward_policy=getattr(
                cfg.parallelism.training,
                "fsdp_reshard_after_forward",
            ),
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP (hierarchical FSDP) to model.")
        else:
            logger.info("Applied FSDP2 to model.")

        if parallel_dims.cp_enabled:
            logger.info("Context Parallel active via dp_shard_cp axis in dp mesh.")

        if getattr(cfg.parallelism.training, "enable_cpu_offload"):
            logger.info("Applied CPU offloading for model parameters.")

    elif parallel_dims.dp_replicate_enabled:
        dp_mesh = world_mesh["dp_replicate"]
        if dp_mesh.ndim > 1:
            raise RuntimeError("DDP path expects a 1D dp_replicate mesh.")
        apply_ddp(
            model=model,
            dp_mesh=dp_mesh,
            enable_compile=getattr(cfg.optimizations.models.torch_compile, "enable"),
        )

    return model


def check_parallelism_cfg(cfg, model: nn.Module, parallel_dims: ParallelDims):
    """
    Perform safety checks on the parallelism configuration after applying all parallelism strategies.
    """
    # ensure that TP is only enabled if FSDP is also enabled
    if getattr(cfg.parallelism.training, "tensor_parallel_degree", 1) > 1:
        if not parallel_dims.fsdp_enabled:
            raise ValueError(
                "Tensor Parallelism requires FSDP to be enabled. "
                "Please enable FSDP in the configuration."
            )
    # ensure that drop_path is off if TP is on or compile is on
    if getattr(cfg.parallelism.training, "tensor_parallel_degree", 1) > 1 or \
       getattr(cfg.optimizations.models.torch_compile, "enable", False):
        if hasattr(model, "drop_path_rate") and model.drop_path_rate > 0.0:
            raise ValueError(
                "drop_path is not supported with Tensor Parallelism. "
                "Please set drop_path_rate to 0.0 when using TP."
            )
    # ensure that fourier loss is not used with loss parallelism
    if not getattr(cfg.parallelism.training, "disable_loss_parallel"):
        if isinstance(model.loss_fn, nn.Module) and hasattr(model.loss_fn, "loss_type") \
            and model.loss_fn.loss_type == "fourier_loss":
            raise ValueError(
                "Fourier loss is not compatible with loss parallelism. "
                "Please set disable_loss_parallel to True when using Fourier loss."
            )
    # ensure Ray Dataloader does not use gRPC callbacks with TP
    if getattr(cfg.parallelism.training, "tensor_parallel_degree", 1) > 1:
        if cfg.datasets.callback_strategy == "grpc":
            raise ValueError(
                "gRPC callbacks in Ray Dataloader are not compatible with Tensor Parallelism. "
                "Please set callback_strategy to 'queue' or do not use async mode."
            )
    # we do not currently support TP with mixed ROPE encodings
    if getattr(cfg.parallelism.training, "tensor_parallel_degree", 1) > 1:
        if hasattr(model.backbone, "rope_pos_enc") and model.backbone.rope_pos_enc:
            raise ValueError(
                "Tensor Parallelism is not currently supported with ROPE positional encodings."
            )


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """Apply FSDP2 to MAE."""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy={reshard_after_forward_policy}"
            )

    # ---- Encoder: patch embedding + transformer blocks ----
    fully_shard(
        model.masked_encoder.patch_embedding,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
    for block in model.masked_encoder.encoder.transformer_blocks:
        fully_shard(
            block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # ---- Decoder: projections + transformer blocks ----
    fully_shard(
        [model.masked_decoder.patch_projection, model.masked_decoder.output_projection],
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
    for block in model.masked_decoder.encoder.transformer_blocks:
        fully_shard(
            block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    fully_shard(model, **fsdp_config)
    logger.info("Applied FSDP2 to MaskedAutoEncoder.")


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
):
    if enable_compile:
        torch._dynamo.config.optimize_ddp = "ddp_optimizer"
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)
    logger.info("Applied composable DDP to MAE.")
