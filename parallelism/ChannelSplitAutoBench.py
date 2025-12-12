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
    # 1. Tensor + sequence parallel
    # ------------------------------------------------------------------
    
    if parallel_dims.tp_enabled:
        disable_loss_parallel = getattr(cfg.parallelism.training, "disable_loss_parallel")
        loss_parallel = not disable_loss_parallel

        apply_tp(
            model=model,
            tp_mesh=world_mesh["tp"],
            loss_parallel=loss_parallel,
        )
        job_config = JobConfig(
            parallelism=Parallelism(
                # === Data Parallelism ===
                data_parallel_replicate_degree=cfg.parallelism.training.data_parallel_replicate_degree,
                data_parallel_shard_degree=cfg.parallelism.training.data_parallel_shard_degree,
                fsdp_reshard_after_forward=cfg.parallelism.training.fsdp_reshard_after_forward,

                # === Tensor Parallelism ===
                tensor_parallel_degree=cfg.parallelism.training.tensor_parallel_degree,
                enable_async_tensor_parallel=cfg.parallelism.training.enable_async_tp,
                disable_loss_parallel=cfg.parallelism.training.disable_loss_parallel,

                # === Pipeline Parallelism ===
                pipeline_parallel_degree=cfg.parallelism.training.pipeline_parallel_degree,
                pipeline_parallel_schedule=cfg.parallelism.training.get("pipeline_parallel_schedule"),
                pipeline_parallel_microbatch_size=cfg.parallelism.training.get("pipeline_parallel_microbatch_size"),

                # === Context Parallelism ===
                context_parallel_degree=cfg.parallelism.training.context_parallel_degree,
                context_parallel_rotate_method=cfg.parallelism.training.context_parallel_rotate_method,

                # === Expert Parallelism ===
                expert_parallel_degree=cfg.parallelism.training.expert_parallel_degree,
                expert_tensor_parallel_degree=cfg.parallelism.training.expert_tensor_parallel_degree,

                # === Misc ===
                enable_compiled_autograd=cfg.parallelism.training.get("enable_compiled_autograd", False),
            ),

            compile=Compile(
                enable=cfg.optimizations.models.torch_compile.enable,
                components=cfg.optimizations.models.torch_compile.modules
            ),
        )
        maybe_enable_async_tp(job_config, world_mesh["tp"])

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


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
):
    """
    Apply tensor + sequence parallelism.
    """

    rowwise_parallel = RowwiseParallel
    colwise_parallel = ColwiseParallel
    prepare_module_input = PrepareModuleInput
    prepare_module_output = PrepareModuleOutput
    prepare_module_input_output = PrepareModuleInputOutput

    # --------------------------------------------------------------
    # 1. Encoder patch embedding: RowwiseParallel on proj
    # --------------------------------------------------------------
    
    parallelize_module(
        model.backbone,
        tp_mesh,
        {
            "": prepare_module_input_output(
                input_layouts=(Replicate()),
                desired_input_layouts=(Replicate()),
                output_layouts=(Shard(1), Replicate()),
                desired_output_layouts=(Shard(1), Replicate()),
                # TODO: consider change to False
                use_local_output=True,
            ),
            "patch_embedding": PrepareModuleInputOutput(
                input_layouts=(Replicate(),),
                desired_input_layouts=(Replicate(),),
                output_layouts=(Replicate(), Replicate()),
                desired_output_layouts=(Replicate(), Replicate()),
                use_local_output=True,
            ),
            "norm": SequenceParallel(),
        },
    )

    # --------------------------------------------------------------
    # 2. Decoder projections: patch_projection & output_projection
    # --------------------------------------------------------------
    
    layer_plan = {
        # input: x with layout Shard(1) from the encoder
        "": prepare_module_input(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Shard(-1),),
            use_local_output=True,
        ),
        # final projection row-parallel on the feature dim
        "last_layer": rowwise_parallel(
            output_layouts=Shard(-1) if loss_parallel else Replicate(),
        ),
    }

    # Attach colwise_parallel to every Linear inside decoder.mlp
    for name, module in model.decoder.mlp.named_children():
        if isinstance(module, nn.Linear):
            # e.g. "mlp.0", "mlp.2", "mlp.4", ...
            layer_plan[f"mlp.{name}"] = colwise_parallel(input_layouts=Shard(-1))

    parallelize_module(
        model.decoder,
        tp_mesh,
        layer_plan,
    )

    # --------------------------------------------------------------
    # 3. Transformer blocks (shared recipe for encoder+decoder)
    # --------------------------------------------------------------

    def _tp_transformer_block(block: nn.Module, is_first_block: bool = False):
        """
        TP recipe for `Transformer` block:
        """
        layer_plan = {}
        # NOTE: we need to prepare the module input for the first block.
        # Hence, prepare_module_input: Replicate -> Shard(1) on sequence dim.
        # Only needed for the very first block in the stack. This is very important
        # and will cause erroneous sequence duplication if missed. See:
        # https://github.com/pytorch/pytorch/torch/distributed/tensor/parallel/style.py.
        # Only needed if input projections are not sharded on sequence dim.
        if is_first_block:
            layer_plan[""] = prepare_module_input(
                input_layouts=(Replicate(),),
                desired_input_layouts=(Shard(1),),
                # TODO: we keep local output here since the residual path
                #       is addition with output from sharded but Tensor output.
                #       Consider changing to False and looking into drop_path/Id.
                use_local_output=True,
            )
        else:
            layer_plan[""] = prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Shard(1),),
                use_local_output=True,
            )

        layer_plan.update(
            {
                "norm1": SequenceParallel(),
                "norm2": SequenceParallel(),

                "att": prepare_module_input(
                    # (x, masks=None, return_attention=False)
                    input_layouts=(Shard(1), None),
                    desired_input_layouts=(Replicate(), None),
                    use_local_output=True if block.rope_pos_enc else False,
                ),
                "att.wq": colwise_parallel(),
                "att.wk": colwise_parallel(),
                "att.wv": colwise_parallel(),
                "att.proj": rowwise_parallel(output_layouts=Shard(1)),

                # "mlp.fc1": colwise_parallel(),
                # "mlp.fc2": rowwise_parallel(output_layouts=Shard(1)),
            }
        )

        layer_plan["mlp"] = prepare_module_input(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
            use_local_output=True,
        )

        layer_plan["mlp.fc1"] = ColwiseParallel()
        layer_plan["mlp.fc2"] = RowwiseParallel(
            output_layouts=Shard(1),
        )

        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    # Encoder stack
    for i, block in enumerate(model.backbone.encoder.transformer_blocks):
        _tp_transformer_block(block, is_first_block=(i == 0))

    # Finally, replicate unsharded modules. This is necessary to ensure
    # that the mesh for all modules is set correctly.
    # --------------------------------------------------------------
    # 4. Put specific remaining params on the tp_mesh
    # --------------------------------------------------------------

    enc = model.backbone

    # ------- Backbone side -------

    # final encoder norm
    replicate_module_params_on_mesh(getattr(enc, "norm", None), tp_mesh)

    # abs sincos positional embedding (if enabled)
    # if getattr(enc, "abs_sincos_enc", False):
    #     replicate_module_params_on_mesh(getattr(enc, "pos_embedding", None), tp_mesh)

    # encoder patch projection: PatchEmbedding.proj (Linear)
    if hasattr(enc, "patch_embedding") and hasattr(enc.patch_embedding, "proj"):
        replicate_module_params_on_mesh(enc.patch_embedding.proj, tp_mesh)

    # ---- Rope Attention params (if applicable) ----

    if enc.rope_pos_enc and enc.rope_mixed:
        for enc_block in enc.encoder.transformer_blocks:
            att = enc_block.att
            if hasattr(att, "freqs"):
                distribute_param_on_mesh(
                    module=att,
                    param_name="freqs",
                    mesh=tp_mesh,
                    placements=[Shard(1)],  # shard over head dimension
                )

    # ----- Loss fn params (if applicable) -----
    if hasattr(model, "loss_fn") and isinstance(model.loss_fn, nn.Module):
        # NOTE: assumes no loss parallelism which is currently required
        #       for Fourier loss (only nn.Module loss fn supported currently).
        parallelize_module(
            model.loss_fn,
            tp_mesh,
            {
                "": prepare_module_input(
                    # (targets, predictions, masks, aux_loss_meta)
                    input_layouts=(Replicate(), Replicate(), Replicate(), None),
                    desired_input_layouts=(Replicate(), Replicate(), Replicate(), None),
                    use_local_output=True,
                )
            },
        )

    print(f"[DEBUG] Applied Tensor + Sequence Parallelism "
        f"(loss_parallel={loss_parallel}).")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply FSDP2.
    """
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
        model.backbone.patch_embedding,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )

    for block in model.backbone.encoder.transformer_blocks:
        fully_shard(
            block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # ---- Decoder: LinearHead ----

    fully_shard(
        model.decoder,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )

    # ---- Root module (hierarchical FSDP) ----
    fully_shard(
        model,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )

    logger.info("Applied FSDP2 to AutoBench model (backbone + linear head).")


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
):
    if enable_compile:
        torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied composable DDP.")