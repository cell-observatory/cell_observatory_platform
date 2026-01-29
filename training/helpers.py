import copy
import itertools
import logging
import math
import os
import random
from collections import defaultdict
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Iterable

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.functional as F
import torch.nn as nn
import ujson
from omegaconf import DictConfig, open_dict
from timm.layers.weight_init import trunc_normal_
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torchinfo import summary
from torchtitan.components.checkpoint import CheckpointManager

logger = logging.getLogger("ray")
logger.setLevel(logging.INFO)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


def record_dataset_len(config, num_train_rows: int, num_val_rows: int):
    bs = config.clusters.batch_size_per_gpu

    def steps_from_rows(n_rows: int):
        return int(n_rows / bs)

    steps_per_epoch = steps_from_rows(num_train_rows)
    val_steps_per_epoch = steps_from_rows(num_val_rows) if num_val_rows > 0 else None

    with open_dict(config):
        config.runtime = {
            "train_steps_per_epoch": steps_per_epoch,
            "val_steps_per_epoch": val_steps_per_epoch,
            "n_train_rows": num_train_rows,
            "n_val_rows": num_val_rows,
        }


def _infer_steps_per_epoch(config, type: str = "train"):
    if config.datasets.dataset._target_.endswith("PretrainDatasourceRay"):
        if type == "train":
            return config.runtime.get("train_steps_per_epoch")
        elif type == "val":
            return config.runtime.get("val_steps_per_epoch")
    else:
        raise TypeError(
            f"Cannot infer steps/epoch for loader. "
            f"Extend the _infer_steps_per_epoch function to handle this type."
        )


def get_steps_per_epoch(
    config: DictConfig, 
    gradient_accumulation_steps: int = 1,
):
    with_validation_loop = isinstance(config.datasets.split, float) and config.datasets.split > 0
    # TODO: double check correctness
    steps_per_epoch = _infer_steps_per_epoch(config,type="train")
    val_steps_per_epoch = _infer_steps_per_epoch(config,
                                                 type="val") if with_validation_loop else None
    logger.info(
        f"Steps per epoch: {steps_per_epoch}, "
        f"Validation steps per epoch: {val_steps_per_epoch}"
    )

    if steps_per_epoch is None or steps_per_epoch <= 0:
        raise ValueError(
            f"Steps per epoch is None or <= 0. Cannot proceed with training."
        )
    
    if (val_steps_per_epoch is None or val_steps_per_epoch <= 0) and with_validation_loop:
        raise ValueError("Validation Dataloader is provided but validation steps per epoch is None or <= 0.")
    
    if gradient_accumulation_steps > 1:
        if steps_per_epoch % gradient_accumulation_steps != 0:
            logger.warning(
                "steps_per_epoch (%d) not divisible by gradient_accumulation_steps (%d); "
                "last %d microbatches of the epoch will be dropped.",
                steps_per_epoch,
                gradient_accumulation_steps,
                steps_per_epoch % gradient_accumulation_steps,
            )
        steps_per_epoch = steps_per_epoch // gradient_accumulation_steps
        if val_steps_per_epoch is not None:
            if val_steps_per_epoch % gradient_accumulation_steps != 0:
                logger.warning(
                    "val_steps_per_epoch (%d) not divisible by gradient_accumulation_steps (%d); "
                    "last %d microbatches of the epoch will be dropped.",
                    val_steps_per_epoch,
                    gradient_accumulation_steps,
                    val_steps_per_epoch % gradient_accumulation_steps,
                )
            val_steps_per_epoch = val_steps_per_epoch // gradient_accumulation_steps

        logger.info(
            f"Adjusted steps per epoch for gradient accumulation steps "
            f"{gradient_accumulation_steps}: {steps_per_epoch}, "
            f"Validation steps per epoch: {val_steps_per_epoch}"
        )

    return steps_per_epoch, val_steps_per_epoch


def load_model_from_ckpt(cfg: DictConfig, checkpoint_manager: CheckpointManager):
    """
    Load ONLY the model weights from checkpoint directory,
    leaving optimizer/lr_schedulers/train_state/etc. unchanged.
    """
    ckpt_dir = cfg.checkpoint.checkpoint_manager.pretrained_checkpointdir

    if not ckpt_dir:
        raise ValueError("pretrained_checkpointdir is empty in config.")
    if not os.path.isdir(ckpt_dir):
        raise ValueError(f"pretrained_checkpointdir does not exist: {ckpt_dir}")

    ckpt_tag = cfg.checkpoint.checkpoint_manager.get("ckpt_tag", None)
    if not ckpt_tag:
        raise ValueError("ckpt_tag is not specified in config.")

    checkpoint_id = checkpoint_manager._create_checkpoint_id(ckpt_tag, folder=ckpt_dir)
    logger.info(f"[Trainer] Loading *model only* from pretrained checkpoint {checkpoint_id}")

    state_dict = checkpoint_manager.states[MODEL].state_dict()
    checkpoint_manager.dcp_load(
        state_dict=state_dict,
        checkpoint_id=checkpoint_id,
        # TODO: add support for from_hf, from_quantized
        from_hf=False,
        from_quantized=False,
    )


def load_trainer_state_dict_from_checkpoint(
        checkpoint_manager: CheckpointManager,
        resume_dir: str | Path,
        step: int = -1
):
    """
    Load only train_state from checkpoint directory,
    without loading model / optimizer / lr_schedulers.
    """
    resume_dir = Path(resume_dir)
    if not resume_dir.is_dir():
        raise FileNotFoundError(f"Resume directory {resume_dir} does not exist")

    # Temporarily point the manager at the resume directory
    original_folder = checkpoint_manager.folder
    checkpoint_manager.folder = str(resume_dir)

    try:
        # If step == -1 find latest step in this folder
        if step == -1:
            step = checkpoint_manager._find_load_step()
            if step == -1:
                raise FileNotFoundError(f"No checkpoints found in {resume_dir}.")

        checkpoint_id = checkpoint_manager._create_checkpoint_id(step)

        train_state_obj = checkpoint_manager.states[TRAIN_STATE]
        states_to_load = {TRAIN_STATE: train_state_obj}
        dcp.load(states_to_load, checkpoint_id=str(checkpoint_id))

        train_state_sd: Dict[str, Any] = train_state_obj.state_dict()
        return train_state_sd, step

    finally:
        checkpoint_manager.folder = original_folder


# NOTE: For DeepSpeed backend, we store best loss, starting epoch and step
#       with checkpoint manager in client state
#       resume model state is most useful when restarting a 
#       job from an earlier checkpoint.
#       With resume_model_state only the checkpoint directory
#       and the checkpoint tag need be specified whereafter
#       any checkpoint with corresponding iter, epoch, best_loss
#       will be loaded from the checkpoint directory.
def resume_model_state(config: DictConfig, checkpoint_manager):
    if config.backend.upper() == "DEEPSPEED":
        assert config.checkpoint.checkpoint_manager.resume_checkpointdir is not None and \
            Path(config.checkpoint.checkpoint_manager.resume_checkpointdir).is_dir(), \
            f"Checkpoint directory does not exist: {config.checkpoint.checkpoint_manager.resume_checkpointdir}" \
            f"Checkpoint directory must be populated " \
            f"with a valid checkpoint to resume training."
        
        ckpt_path, client_state = checkpoint_manager.load()

        # get metadata from client state
        best_loss = client_state["best_loss"]
        starting_epoch, starting_iter = client_state["epoch"], client_state["iter"]
        epochs_left = config.schedulers.epochs - starting_epoch

        if epochs_left <= 0:
            logger.error(
                f"No epochs left to train. Starting epoch {starting_epoch} "
                f"exceeds total epochs {config.schedulers.epochs}."
            )
        
        if config.checkpoint.checkpoint_manager.resume_checkpointdir != \
            config.checkpoint.checkpoint_manager.save_checkpointdir:
            logger.warning(
                f"Checkpoint resume directory {config.checkpoint.checkpoint_manager.resume_checkpointdir} "
                f"does not match new save checkpoint directory {config.checkpoint.checkpoint_manager.save_checkpointdir}. "
                "New checkpoints will NOT be saved to the previous checkpoint directory!"
            )
            Path(config.checkpoint.checkpoint_manager.save_checkpointdir).mkdir(exist_ok=True, parents=True)
        
        if not Path(config.loggers.logdir).exists():
            logger.warning(
                f"Log directory {config.loggers.logdir} does not exist. "
                f"Creating new log directory. New logs from starting epoch {starting_epoch} "
                f"will not contain any previous training run data!"
            )
            Path(config.loggers.logdir).mkdir(exist_ok=True, parents=True)

        return best_loss, starting_iter, starting_epoch

    elif config.backend.upper() == "TORCHTITAN":
        train_state_sd, step = load_trainer_state_dict_from_checkpoint(
            checkpoint_manager=checkpoint_manager,
            resume_dir=config.checkpoint.checkpoint_manager.resume_checkpointdir,
            step=config.checkpoint.checkpoint_manager.save_step,
        )

        if config.checkpoint.checkpoint_manager.resume_checkpointdir != \
            config.checkpoint.checkpoint_manager.save_checkpointdir:
            logger.warning(
                f"Checkpoint resume directory {config.checkpoint.checkpoint_manager.resume_checkpointdir} "
                f"does not match new save checkpoint directory {config.checkpoint.checkpoint_manager.save_checkpointdir}. "
                "New checkpoints will NOT be saved to the previous checkpoint directory!"
            )
            Path(config.checkpoint.checkpoint_manager.save_checkpointdir).mkdir(exist_ok=True, parents=True)

        start_iter = int(train_state_sd.get("iteration", 0))
        start_epoch = int(train_state_sd.get("epoch", 0))

        if not Path(config.loggers.logdir).exists():
            logger.warning(
                f"Log directory {config.loggers.logdir} does not exist. "
                f"Creating new log directory. New logs from starting epoch {start_epoch} "
                f"will not contain any previous training run data!"
            )
            Path(config.loggers.logdir).mkdir(exist_ok=True, parents=True)

        best_metric = train_state_sd.get("best_metric", float("inf"))
        best_metric_epoch = train_state_sd.get("best_metric_epoch", start_epoch)
        best_metric_iter = train_state_sd.get("best_metric_iter", start_iter)

        epochs_left = config.schedulers.epochs - start_epoch
        if epochs_left <= 0:
            raise ValueError(
                f"No epochs left to train. Starting epoch {start_epoch} "
                f"exceeds total epochs {config.schedulers.epochs}."
            )
        if best_metric == float("inf") or best_metric_epoch == 0 or best_metric_iter == 0:
            logger.warning(
                f"Best metric not found in checkpoint or best_metric_epoch/iter is 0. "
            )

        return best_metric, start_iter, start_epoch

    else:
        raise ValueError(f"Unsupported backend: {config.backend}") 


def resume_run(trainer, config: DictConfig):
    Path(config.paths.outdir).mkdir(exist_ok=True, parents=True)
    if config.paths.resume_checkpointdir:
        best_loss, iter, epoch = resume_model_state(config, checkpoint_manager=trainer.checkpoint_manager)
        trainer.event_recorder.resume(iter=iter, epoch=epoch)

    else:
        epoch, iter, best_loss = 0, 0, np.inf

        Path(config.loggers.logdir).mkdir(exist_ok=True, parents=True)
        Path(config.checkpoint.checkpoint_manager.save_checkpointdir).mkdir(exist_ok=True, parents=True)

        logger.info(f"Output dir: {config.paths.outdir}")
        logger.info(f"Log dir: {config.loggers.logdir}")
        logger.info(f"Checkpoint save dir: {config.checkpoint.checkpoint_manager.save_checkpointdir}")
    
    return best_loss, iter, epoch


def summarize_model(
    model: nn.Module,
    batch_size: int,
    logdir: Path | str,
    inputs: Optional[tuple] = None,
    input_data: Optional[dict] = None,
):
    logdir = Path(logdir)

    model_logbook = {}
    model_stats = summary(
        model=model,
        input_size=inputs if input_data is None else None,
        input_data=input_data,
        depth=5,
        col_width=25,
        col_names=["kernel_size", "output_size", "num_params"],
        row_settings=["var_names"],
        verbose=0,
        mode='eval'
    )
    train_stats = summary(
        model=model,
        input_size=inputs if input_data is None else None,
        input_data=input_data,
        depth=5,
        col_width=25,
        col_names=["kernel_size", "output_size", "num_params"],
        row_settings=["var_names"],
        verbose=1,
        mode="train",
    )

    with (logdir / "model.log").open("w") as f:
        f.write(str(model_stats))

    model_logbook["training_batch_size"] = batch_size
    model_logbook["input_bytes"] = model_stats.total_input
    model_logbook["total_params"] = model_stats.total_params
    model_logbook["trainable_params"] = model_stats.trainable_params
    model_logbook["param_bytes"] = model_stats.total_param_bytes

    model_logbook["eval_macs"] = model_stats.total_mult_adds
    model_logbook["training_macs"] = train_stats.total_mult_adds

    model_logbook["forward_pass_bytes"] = model_stats.total_output_bytes
    model_logbook["forward_backward_pass_bytes"] = train_stats.total_output_bytes

    model_logbook["eval_model_bytes"] = model_logbook["param_bytes"] + model_logbook["forward_pass_bytes"]
    model_logbook["training_model_bytes"] = model_logbook["param_bytes"] + model_logbook["forward_backward_pass_bytes"]

    model_logbook["eval_bytes"] = model_logbook["input_bytes"] + model_logbook["eval_model_bytes"]
    model_logbook["training_bytes"] = model_logbook["input_bytes"] + model_logbook["training_model_bytes"]

    model_logbook["layers"] = {}
    for layer in train_stats.summary_list:
        if layer.is_leaf_layer:
            model_logbook["layers"][f"{layer.class_name}_{layer.var_name}"] = {
                "macs": layer.macs,
                "params": max(layer.num_params, 0),
                "param_bytes": layer.param_bytes,
                "forward_pass_bytes": layer.output_bytes,
                "forward_backward_pass_bytes": layer.output_bytes * 2,  # x2 for gradients
                "output_shape": layer.output_size,
            }

    with (logdir / "model_logbook.json").open("w") as f:
        ujson.dump(model_logbook, f, indent=4, sort_keys=False, ensure_ascii=False, escape_forward_slashes=False)


def log_data_timings(
    trainer,
    idx,
    data_sample: dict,
    loss_dict: dict,
    type: str = "train",
):
    assert data_sample is not None, "data_sample is None"
    assert data_sample['metainfo'] is not None, "data_sample['metainfo'] is None"

    data_time = data_sample['metainfo'].get('data_time', None)
    if data_time is not None:
        trainer.event_recorder.put_scalars(
            scope="step",
            prefix="val_" if type == "val" else None,
            data_time=data_time,
            reduce_method=["median", "max", "min"]
        )

    get_item_time = data_sample['metainfo'].get('get_item_time', None)
    if get_item_time is not None:
        trainer.event_recorder.put_scalars(
            scope="step",
            prefix="val_" if type == "val" else None,
            get_item_time=get_item_time.mean().item(),
            reduce_method=["median", "max", "min"]
        )
    
    preprocess_time = data_sample['metainfo'].get('preprocess_time', None)
    if preprocess_time is not None:
        trainer.event_recorder.put_scalars(
            scope="step",
            prefix="val_" if type == "val" else None,
            preprocess_time=preprocess_time,
            reduce_method=["median", "max", "min"]
        )

    masking_time = data_sample['metainfo'].get('masking_time', None)
    if masking_time is not None:
        trainer.event_recorder.put_scalars(
            scope="step",
            prefix="val_" if type == "val" else None,
            masking_time=masking_time,
            reduce_method=["median", "max", "min"]
        )
    
    collate_time = data_sample['metainfo'].get('collate_time', None)
    if collate_time is not None:
        trainer.event_recorder.put_scalars(
            scope="step",
            prefix="val_" if type == "val" else None,
            collate_time=collate_time,
            reduce_method=["median", "max", "min"]
        )
    
    slice_time = data_sample['metainfo'].get('slice_time', None)
    if slice_time is not None:
        trainer.event_recorder.put_scalars(
            scope="step",
            prefix="val_" if type == "val" else None,
            slice_time=slice_time.mean().item() if \
                isinstance(slice_time, torch.Tensor) else np.mean(slice_time),
            reduce_method=["median", "max", "min"]
        )

    transform_time = data_sample['metainfo'].get('transform_time', None)
    if transform_time is not None:
        trainer.event_recorder.put_scalars(
            scope="step",
            prefix="val_" if type == "val" else None,
            transform_time=transform_time,
            reduce_method=["median", "max", "min"]
        )

    advanced_metrics = data_sample.get('advanced_metrics', None)
    if advanced_metrics is not None:
        for k, v in advanced_metrics.items():
            trainer.event_recorder.put_scalars(
                scope="step",
                prefix="val_" if type == "val" else None,
                **{k: (v.item() if torch.is_tensor(v) else v)}
            )

    if type == "train":
        trainer.event_recorder.put_scalars(
            scope="step",
            **{k: (v.item() if torch.is_tensor(v) else v)
            for k, v in loss_dict.items()
            }
        )
    elif type == "val":
        trainer.event_recorder.put_scalars(
            scope="step",
            prefix="val_",
            **{k: (v.item() if torch.is_tensor(v) else v)
            for k, v in loss_dict.items()
            }
        )


def get_input_data(inputs, device: Optional[torch.device] = 'cuda'):
    input_data = ({"data_tensor": torch.randn(*inputs, device=device), "metainfo": {}},)
    return input_data


def get_masked_input_data(model, inputs, device: Optional[torch.device] = 'cuda', mask_ratio: float = 0.75):
    n_patches = model.get_num_patches()
    context_len = int(n_patches * (1 - mask_ratio))
    context_idx = torch.arange(context_len, dtype=torch.long, device=device).unsqueeze(0)
    target_idx  = torch.arange(context_len, n_patches, dtype=torch.long, device=device).unsqueeze(0)

    meta = {
        "masks": [torch.ones(n_patches, dtype=torch.long, device=device).unsqueeze(0)],
        "context_masks": [context_idx],
        "target_masks": [target_idx],
        "original_patch_indices": [torch.arange(n_patches, dtype=torch.long, device=device)],
        "patches_used": [torch.arange(n_patches, dtype=torch.long, device=device).unsqueeze(0).expand(inputs[0],-1)],
    }

    # summary() will unpack the input data but the fwd function in
    # JEPA and MAE models expects a dict hence we wrap the input data
    # in a tuple with a single dict element
    input_data = ({"data_tensor": torch.randn(*inputs, device=device), "metainfo": meta},)
    return input_data


def enable_optimizations(cfg: DictConfig):
    # torch backend optimization flags for training
    if cfg.optimizations.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if cfg.optimizations.cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    if cfg.optimizations.cudnn_enabled:
        torch.backends.cudnn.enabled = True
    if cfg.optimizations.cudnn_allow_tf32:
        torch.backends.cudnn.allow_tf32 = True
    if cfg.optimizations.deterministic:
        torch.use_deterministic_algorithms(mode=True)


# many of the below functions are based on:
# https://github.com/pytorch/torchtitan/main/torchtitan/models/llama3/infra/parallelize.py
def apply_activation_checkpointing(cfg: DictConfig, model: nn.Module):
    """Apply activation checkpointing to the model."""
    num_blocks = apply_ac_over_discovered_stacks(cfg.activation_checkpoint, model)
    logger.info(f"Applied activation checkpointing to {num_blocks} transformer blocks.")


# TODO: ensure set is exhaustive
# identify all the MM functions that we want to count
# for activation checkpointing
_MM_FUNCS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.matmul.default,
}
# for selective op activation checkpointing
_save_list = {
    *tuple(_MM_FUNCS),
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
}


def _as_stack(
    path_str: str,
    parent,
    block_names: str
) -> Tuple[str, nn.ModuleList]:
    if not hasattr(parent, block_names):
        raise TypeError(
            f"Config path '{path_str}' resolved to {type(parent).__name__}, "
            f"which has no '.{block_names}' attribute. "
            f"Expected a *parent* module containing a '{block_names}' ModuleList."
        )
    stack = getattr(parent, block_names)
    if not isinstance(stack, nn.ModuleList):
        raise TypeError(
            f"Attribute '{block_names}' on '{path_str}' is "
            f"{type(stack).__name__}; expected nn.ModuleList."
        )
    return f"{path_str}.{block_names}", stack


def yield_transformer_stacks(
    cfg_module_blocks: Sequence[Tuple[str, Union[str, Sequence[str]]]],
    model: nn.Module,
):
    for module_fqn, block_names in cfg_module_blocks:
        submod = attrgetter(module_fqn)(model)
        # allow either a single block name or a list/tuple of block names per module
        if isinstance(block_names, str):
            block_names = [block_names]
        for bn in block_names:
            stack_fqn, stack = _as_stack(module_fqn, submod, block_names=bn)
            yield (stack_fqn, stack)
    return


def apply_ac_over_discovered_stacks(cfg, model: nn.Module):
    cfg_modules = cfg.get("modules", None)
    cfg_module_blocks: List[Tuple[str, Union[str, Sequence[str]]]] = []
    for entry in cfg_modules:
        if isinstance(entry, str) or not isinstance(entry, Sequence) or len(entry) != 2:
            raise ValueError(
                "Activation checkpointing 'modules' entries must be "
                "(module_fqn, block_names_or_list)."
            )
        cfg_module_blocks.append((entry[0], entry[1]))

    wrapped = 0
    for stack_fqn, stack in yield_transformer_stacks(cfg_module_blocks, model):
        for i, block in enumerate(stack):
            wrapped_block = _apply_ac_to_module(
                module=block,
                act_ckpt_mode=cfg.mode,
                selective_ac_option=cfg.selective_ac_option,
                per_op_sac_force_recompute_mm_shapes_by_fqns=\
                    cfg.per_op_sac_force_recompute_mm_shapes_by_fqns,
                base_fqn=f"{stack_fqn}.{i}",
                mm_recompute_frac=cfg.mm_recompute_frac,
            )
            stack[i] = wrapped_block
            wrapped += 1
    return wrapped


# args info from: https://github.com/pytorch/pytorch/aten/src/ATen/native/native_functions.yaml
def _rhs_shape_for(func, args):
    # return the (K, N) rhs shape
    if func == torch.ops.aten.mm.default:
        # func: mm(Tensor self, Tensor mat2) -> Tensor
        return tuple(args[1].shape)
    if func == torch.ops.aten.addmm.default:
        # func: addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, ...)
        # addmm(input, mat1[M,K], mat2[K,N], beta, alpha)
        return tuple(args[2].shape)
    if func in (torch.ops.aten.matmul.default, torch.ops.aten.bmm.default):
        # matmul: func: matmul(Tensor self, Tensor other) -> Tensor
        # bmm: func: bmm(Tensor self, Tensor mat2) -> Tensor
        # (..., M, K) @ (..., K, N)
        return tuple(args[1].shape[-2:])
    return None


def _apply_ac_to_module(
    module: nn.Module,
    act_ckpt_mode: str,
    base_fqn: Optional[str] = None,
    selective_ac_option: Optional[Union[str, int]] = None,
    per_op_sac_force_recompute_mm_shapes_by_fqns: Optional[List[str]] = None,
    mm_recompute_frac: Optional[int] = 2,
):
    valid_ac_modes = ("full", "selective")
    if act_ckpt_mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {act_ckpt_mode}. Valid modes: {valid_ac_modes}"
        )

    if act_ckpt_mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

    assert act_ckpt_mode == "selective", f"{act_ckpt_mode}"

    use_op_sac = selective_ac_option == "op"
    use_layer_sac = isinstance(selective_ac_option, (str, int)) \
        and str(selective_ac_option).isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )

    if use_op_sac:
        from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

        mm_recompute_shapes, per_op_act_ckpt_fqns = set(), []
        # True if len(per_op_sac_force_recompute_mm_shapes_by_fqns) > 0 or
        # per_op_sac_force_recompute_mm_shapes_by_fqns is not None
        if per_op_sac_force_recompute_mm_shapes_by_fqns:
            for module_fqn, submod in module.named_modules():
                fqn = module_fqn
                if base_fqn is not None:
                    fqn = f"{base_fqn}.{module_fqn}"

                if not any(
                    filter_fqn in fqn
                    for filter_fqn in per_op_sac_force_recompute_mm_shapes_by_fqns
                ):
                    continue

                if not isinstance(submod, nn.Linear):
                    raise ValueError(
                        "per_op_sac_force_recompute_mm_shapes_by_fqns expected to match "
                        f"a nn.Linear, but got: {submod}"
                    )

                # use rhs shapes to identify the mm ops to recompute
                out_f, in_f = submod.weight.shape
                mm_recompute_shapes.add((in_f, out_f))

                logger.info(
                    f"Selective op AC force recompute mm shape for {fqn}: "
                    f"{(in_f, out_f)}"
                )

                per_op_act_ckpt_fqns.append(fqn)

            logger.info(
                f"Activation checkpointing summary:     \n"
                f"Selective op AC mode: {act_ckpt_mode} \n"
                f"Selective op AC option: {selective_ac_option} \n"
                f"Selective op AC force recompute functions: {per_op_act_ckpt_fqns} \n"
                f"Selective op AC force recomputing mms with rhs shapes {mm_recompute_shapes}"
            )

        assert mm_recompute_frac is not None and mm_recompute_frac > 0, \
            f"mm_recompute_frac must be a positive integer, got: {mm_recompute_frac}"

        def _get_custom_policy(meta, mm_recompute_frac, mm_recompute_shapes):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"

                is_mm = func in _MM_FUNCS
                if is_mm:
                    rhs = _rhs_shape_for(func, args)
                    # force-recompute if rhs matches a targeted Linear
                    if rhs is not None and rhs in mm_recompute_shapes:
                        # PREFER_XXX may be overridden
                        # but MUST_XXX is always respected
                        return CheckpointPolicy.PREFER_RECOMPUTE
                    meta[mm_count_key] += 1

                # saves output of all compute ops, except every mm_recompute_frac mm
                to_save = func in _save_list and not (
                    is_mm and meta[mm_count_key] % mm_recompute_frac == 0
                )
                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta,
                                                                           mm_recompute_frac,
                                                                           mm_recompute_shapes))

        # selective checkpointing of mm ops as well every mm_recompute_frac-th
        # mm op in the module
        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )

    elif use_layer_sac:
        # checkpoint every `selective_ac_option` of the modules passed to this function
        ac_freq = int(selective_ac_option)
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        else:
            return module
        
    else:
        raise ValueError(
            f"Invalid selective AC option: {selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )


def apply_compile(cfg: DictConfig, model: nn.Module):
    cfg_compile = cfg.torch_compile
    if cfg_compile.range == "full":
        logger.info("Applying torch.compile to the whole model.")
        model = torch.compile(
            model,
            dynamic=cfg_compile.dynamic,
            mode=cfg_compile.mode,
            fullgraph=False,  # DS causes graph breaks -> keep False here
        )
        # mark whole-model compilation so printer can tag the root
        setattr(model, "_is_compiled", True)
        setattr(model, "_compiled_fqns", set(["<whole_model>"]))
    elif cfg_compile.range == "block_based":
        num_blocks_compiled = _apply_compile_over_discovered_stacks(cfg_compile, model)
        logger.info(f"Applied torch.compile to {num_blocks_compiled} transformer blocks.")
    else:
        raise ValueError(
            f"Invalid torch compile mode: {cfg_compile.mode}. "
            "Valid modes: 'full' or 'block_based'"
        )

    return model


def _apply_compile_over_discovered_stacks(cfg, model: nn.Module):
    """
    Apply torch.compile to each Transformer block, and record which blocks were compiled
    so the tree printer can show [TC].
    """
    cfg_modules = cfg.get("modules")

    num_blocks_compiled = 0
    compiled_fqns = set()

    if cfg_modules is None:
        raise ValueError(
            "torch_compile config must specify "
            "'modules' (list[(module, block_names)])."
        )

    cfg_module_blocks: List[Tuple[str, Union[str, Sequence[str]]]] = []
    for entry in cfg_modules:
            if isinstance(entry, str) or not isinstance(entry, Sequence) or len(entry) != 2:
                raise ValueError(
                    "torch_compile 'modules' entries must be "
                    "(module_fqn, block_names_or_list)."
                )
            cfg_module_blocks.append((entry[0], entry[1]))

    for stack_fqn, stack in yield_transformer_stacks(cfg_module_blocks, model):
        for i, block in enumerate(stack):
            compiled = torch.compile(
                block,
                fullgraph=getattr(cfg, "blockbased_fullgraph", False),
                dynamic=getattr(cfg, "dynamic", None),
                mode=getattr(cfg, "mode", None),
                backend=getattr(cfg, "backend", None),
            )
            # mark and re-register
            setattr(compiled, "_is_compiled", True)  # helpful heuristic for printer
            stack[i] = compiled
            compiled_fqns.add(f"{stack_fqn}.{i}")
            num_blocks_compiled += 1

    # stash a summary for the printer
    setattr(model, "_compiled_fqns", compiled_fqns)
    return num_blocks_compiled


def get_model_optimizations_node(
    cfg: Optional[DictConfig],
    models_path: str = "optimizations.models",
    leaf_keys: Iterable[str] = ("activation_checkpoint", "torch_compile"),
    max_depth: int = 42,
):
    """
    Return the DictConfig node that contains model-specific optimization settings,
    even if cfg.optimizations.models is nested through multiple single-key dict layers.

    It will:
      1) Resolve cfg.<models_path>
      2) If that node already contains any of `leaf_keys`, return it (legacy layout).
      3) Otherwise, repeatedly "peel" one level deeper IF the node has exactly one
         non-internal key, until it finds a node containing any of `leaf_keys`.
      4) Raise if it encounters ambiguity (0 or >1 keys) before reaching a leaf.
    """
    if cfg is None:
        return None

    # Walk down dotted path to cfg.optimizations.models
    node = cfg
    for part in models_path.split("."):
        if node is None or not hasattr(node, "get"):
            return None
        node = node.get(part, None)
        if node is None:
            return None

    def has_leaf(n: DictConfig) -> bool:
        return any(n.get(k, None) is not None for k in leaf_keys)

    # If already at leaf (legacy), return immediately
    if has_leaf(node):
        return node

    # Otherwise peel through single-key dict layers
    cur = node
    for _ in range(max_depth):
        if cur is None or not hasattr(cur, "keys"):
            return None

        if has_leaf(cur):
            return cur

        keys = [k for k in cur.keys() if not str(k).startswith("_")]
        if len(keys) != 1:
            raise ValueError(
                f"Expected exactly one key while peeling cfg.{models_path}, "
                f"but got keys={keys}. (Reached node={cur})"
            )
        cur = cur[keys[0]]

    raise ValueError(
        f"Exceeded max_depth={max_depth} while peeling cfg.{models_path}. "
        f"Likely a cycle or unexpected structure."
    )


# helpers for printing model structure with
# activation checkpointing to ensure correctness
def unwrap_checkpoint(m):
    if isinstance(m, CheckpointWrapper):
        inner = getattr(m, "_checkpoint_wrapped_module", None)
        if inner is None:
            inner = getattr(m, "module", None)
        return True, m, inner
    return False, None, m


def print_model_tree_with_opt(model, max_depth=99, treat_wrappers_as_leaves=False):
    compiled_fqns = getattr(model, "_compiled_fqns", set())

    def is_compiled(mod, fqn):
        return getattr(mod, "_is_compiled", False) or (fqn in compiled_fqns)

    def walk(mod, prefix="", fqn="", depth=0):
        is_ckpt, wrap, inner = unwrap_checkpoint(mod)
        label = mod.__class__.__name__
        tags = []

        if is_ckpt:
            inner_name = inner.__class__.__name__ if inner is not None else "?"
            label = f"{label} -> {inner_name}"
            tags.append("AC")
        if is_compiled(mod, fqn):
            tags.append("TC")

        tag_str = ("  [" + ",".join(tags) + "]") if tags else ""
        print(prefix + label + tag_str)

        if depth >= max_depth:
            return
        if is_ckpt and treat_wrappers_as_leaves:
            return

        children = list(mod.named_children())
        for i, (name, child) in enumerate(children):
            last = i == (len(children) - 1)
            branch = "└─ " if last else "├─ "
            indent = "   " if last else "│  "
            child_fqn = f"{fqn}.{name}" if fqn else name
            print(prefix + branch + f"{name}: ", end="")
            walk(child, prefix + indent, child_fqn, depth + 1)

    walk(model)


# from: https://github.com/rwightman/timm/timm/models/_manipulate.py
def named_apply(
        fn: Callable,
        module: nn.Module, name='',
        depth_first: bool = True,
        include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def init_weights(model: nn.Module, weight_init_type: str):
    logger = logging.getLogger(__name__)
    
    # ------------------------------------------------------------------
    # Common alias resolution helpers
    # ------------------------------------------------------------------
    def _resolve_alias(root: nn.Module, paths):
        """
        paths: list[tuple[str, ...]]

        Returns (obj, path) where obj is the resolved attribute chain,
        or (None, None) if none match.
        """
        for chain in paths:
            obj = root
            ok = True
            for name in chain:
                if not hasattr(obj, name):
                    ok = False
                    break
                obj = getattr(obj, name)
            if ok:
                return obj, chain
        return None, None

    # Alias tables
    PATCH_EMBED_WEIGHT_ALIASES = [
        ("masked_encoder", "patch_embedding", "proj", "weight"),
        ("masked_encoder", "patch_embed", "proj", "weight"),
        ("backbone", "patch_embedding", "proj", "weight"),
        ("backbone", "patch_embed", "proj", "weight"),
        ("encoder", "patch_embedding", "proj", "weight"),
        ("encoder", "patch_embed", "proj", "weight"),
    ]

    TOKEN_PARAM_ALIASES = [
        ("masked_decoder", "token_param"),
        ("decoder", "token_param"),
        ("backbone", "token_param"),
        ("encoder", "token_param"),
    ]

    INPUT_ENCODER_ALIASES = [
        ("input_encoder",),
        ("masked_encoder",),
        ("backbone",),
        ("encoder",),
    ]

    TARGET_PREDICTOR_ALIASES = [
        ("target_predictor",),
        ("decoder",),
    ]

    # ------------------------------------------------------------------
    # MAE
    # ------------------------------------------------------------------
    
    if weight_init_type == "mae":
        # MAE model init utility function adapted from:
        # https://github.com/facebookresearch/mae/main/models_mae.py
        def _mae_init_weights(m):
            if isinstance(m, nn.Linear):
                # use xavier_uniform following official JAX ViT
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            # NOTE: we follow the initialization scheme of MAE here however
            #       we might consider doing as timm does and
            #       include the option to initalize with init_weights
            #       function of module if it exists and then call
            #       named_apply(_mae_init_weights) afterwards in which
            #       case we'd include the below commented code
            # elif hasattr(m, 'init_weights'):
            #     m.init_weights()

        patch_w, patch_path = _resolve_alias(model, PATCH_EMBED_WEIGHT_ALIASES)
        if patch_w is None:
            raise ValueError(
                "MAE init: could not locate patch embedding weight. "
                f"Tried aliases: {PATCH_EMBED_WEIGHT_ALIASES}"
            )
        torch.nn.init.xavier_uniform_(patch_w.view(patch_w.shape[0], -1))

        token_param, token_path = _resolve_alias(model, TOKEN_PARAM_ALIASES)
        if token_param is not None:
            torch.nn.init.normal_(token_param, std=0.02)
        else:
            logger.debug(
                "MAE init: token_param not found (aliases: %s); skipping token init.",
                TOKEN_PARAM_ALIASES,
            )

        # initialize nn.Linear and nn.LayerNorm
        model.apply(_mae_init_weights)

    # ------------------------------------------------------------------
    # VJEPA
    # ------------------------------------------------------------------

    elif weight_init_type == "vjepa":
        # helpers from:
        # https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py
        def _vjepa_fix_init_weight(enc_model: nn.Module):
            def rescale(param, layer_id):
                param.div_(math.sqrt(2.0 * layer_id))

            if not hasattr(enc_model, "encoder") or not hasattr(
                enc_model.encoder, "transformer_blocks"
            ):
                raise ValueError(
                    "VJEPA init: expected an encoder with `encoder.transformer_blocks` "
                    f"on {enc_model.__class__.__name__}"
                )

            for layer_id, layer in enumerate(enc_model.encoder.transformer_blocks):
                rescale(layer.att.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        def _vjepa_init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=model.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=model.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=model.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        decoder_token, dec_path = _resolve_alias(model, [("masked_decoder", "token_param")] + TOKEN_PARAM_ALIASES)
        if decoder_token is None:
            raise ValueError(
                "VJEPA init: could not locate decoder token_param. "
                f"Tried aliases: {[('masked_decoder', 'token_param')] + TOKEN_PARAM_ALIASES}"
            )
        trunc_normal_(decoder_token, std=model.init_std)

        # Initialize all Linear/Norm/Conv
        model.apply(_vjepa_init_weights)

        # Required: input encoder
        input_encoder, input_path = _resolve_alias(model, INPUT_ENCODER_ALIASES)
        if input_encoder is None:
            raise ValueError(
                "VJEPA init: could not locate input encoder. "
                f"Tried aliases: {INPUT_ENCODER_ALIASES}"
            )
        _vjepa_fix_init_weight(input_encoder)

        # Required: target predictor
        target_predictor, tp_path = _resolve_alias(model, TARGET_PREDICTOR_ALIASES)
        if target_predictor is None:
            raise ValueError(
                "VJEPA init: could not locate target predictor. "
                f"Tried aliases: {TARGET_PREDICTOR_ALIASES}"
            )
        _vjepa_fix_init_weight(target_predictor)

    # ------------------------------------------------------------------
    # VJEPA2
    # ------------------------------------------------------------------
    
    elif weight_init_type == "vjepa2":
        # helpers from:
        # https://github.com/facebookresearch/vjepa2/main/src/models/vision_transformer.py
        def _vjepa2_init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=model.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            # NOTE: technically vjepa2 only applies the below
            #       to input encoder and not target predictor
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=model.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=model.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        def _vjepa2_rescale_blocks(enc_model: nn.Module):
            def rescale(param, layer_id):
                param.div_(math.sqrt(2.0 * layer_id))

            if not hasattr(enc_model, "encoder") or not hasattr(
                enc_model.encoder, "transformer_blocks"
            ):
                raise ValueError(
                    "VJEPA2 init: expected an encoder with `encoder.transformer_blocks` "
                    f"on {enc_model.__class__.__name__}"
                )

            for layer_id, layer in enumerate(enc_model.encoder.transformer_blocks):
                rescale(layer.att.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        model.apply(_vjepa2_init_weights)

        input_encoder, input_path = _resolve_alias(model, INPUT_ENCODER_ALIASES)
        if input_encoder is None:
            raise ValueError(
                "VJEPA2 init: could not locate input encoder. "
                f"Tried aliases: {INPUT_ENCODER_ALIASES}"
            )
        _vjepa2_rescale_blocks(input_encoder)

        target_predictor, tp_path = _resolve_alias(model, TARGET_PREDICTOR_ALIASES)
        if target_predictor is not None and hasattr(target_predictor, "encoder"):
            _vjepa2_rescale_blocks(target_predictor)

    # ------------------------------------------------------------------
    # ViT-Adapter style init
    # ------------------------------------------------------------------

    elif weight_init_type == "vit_adapter":
        def _vit_adapter_init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                fan_out = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                fan_out = m.kernel_size[0] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

        if not (hasattr(model, "up") and hasattr(model, "spatial_prior_module") and hasattr(model, "interactions")):
            raise ValueError(
                "vit_adapter init: expected model to have attributes "
                "`up`, `spatial_prior_module`, and `interactions`."
            )
        model.up.apply(_vit_adapter_init_weights)
        model.spatial_prior_module.apply(_vit_adapter_init_weights)
        model.interactions.apply(_vit_adapter_init_weights)
        torch.nn.init.normal_(model.level_embed)

    else:
        raise ValueError(f"Unknown weight initialization type: {weight_init_type}")


def get_data_dim(layout_order: str) -> int:
    if layout_order == "TZYXC":
        return 4
    elif layout_order == "ZYXC":
        return 3
    elif layout_order == "YXC":
        return 2
    elif layout_order == "TYXC":
        return 3
    else:
        raise ValueError(f"Unknown dataset layout order: {layout_order}")


def get_patch_sizes(input_format: str, patch_shape: List[int]):
    if input_format == "TZYXC":
        # temporal, axial, lateral
        return (patch_shape[0], patch_shape[1], patch_shape[2])
    
    elif input_format == "TYXC":
        # temporal, lateral
        return (patch_shape[0], None, patch_shape[1])
    
    elif input_format == "ZYXC":
        # axial, lateral
        return (None, patch_shape[0], patch_shape[1])

    elif input_format == "YXC":
        # lateral only
        return (None, None, patch_shape[0])

    elif input_format == "XC":
        # lateral only (1D)
        return (None, None, patch_shape[0])

    else:
        raise ValueError(f"Unknown dataset layout order: {input_format}")


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_image_sizes(
    input_format: str,
    input_shape: Tuple[int, ...],
    batch_size: int,
    metadata: Dict[str, Any],
    device: Optional[torch.device] = None,
):
    """
    Get image sizes and a 3D padding mask for each sample in the batch.

    Args:
        input_format (str): Input format string (e.g. "TZYXC", "ZYXC").
        input_shape (tuple): Shape of the input (no batch), matching input_format.
        batch_size (int): Number of samples in the batch.
        metadata (dict): Batch metadata; each key maps to a 1D array of
                         length `batch_size` (e.g. "y_size", "x_size", ...).
        device (torch.device, optional): Device on which to allocate the padding
                         mask. If None, uses CPU.

    Returns:
        image_sizes:        list[tuple], per-sample "current" sizes
        image_sizes_padded: list[tuple], per-sample sizes including any padding
        orig_image_sizes:   list[tuple], per-sample original sizes (or image_sizes)
        padding_mask:       torch.BoolTensor of shape [B, Z, Y, X] or [B, Y, X]
                            True = padded voxel, False = valid voxel.
    """
    if input_format == "TZYXC":
        ax_names = ("time", "z", "y", "x")
    elif input_format == "ZYXC":
        ax_names = ("z", "y", "x")
    elif input_format == "TCZYX":
        ax_names = ("time", "channel", "z", "y", "x")
    elif input_format == "CZYX":
        ax_names = ("channel", "z", "y", "x")
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

    # TODO: consider how to generalize to spacetime

    # Build a 3D padding mask [B, Z, Y, X] or [B, Y, X]
    # We only care about spatial volume axes for DETR-style masks.
    spatial_axes = [ax for ax in ("Z", "Y", "X") if ax in input_format]

    # map axis -> full size from input_shape
    axis_to_size = dict(zip(input_format, input_shape))
    full_sizes = {ax: int(axis_to_size[ax]) for ax in spatial_axes}

    # spatial mask shape (Z, Y, X) or (Y, X)
    spatial_shape = tuple(full_sizes[ax] for ax in spatial_axes)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_sizes: List[Tuple[int, ...]] = []
    for i in range(batch_size):
        spatial_dims = [int(metadata[f"{ax}_size"][i]) for ax in ax_names]
        image_sizes.append(tuple(spatial_dims))

    image_sizes_padded: List[Tuple[int, ...]] = [spatial_shape] * batch_size

    # use orig_* sizes only if *all* are present
    if all(f"orig_{ax}_size" in metadata for ax in ax_names):
        orig_image_sizes: List[Tuple[int, ...]] = []
        for i in range(batch_size):
            spatial_dims = [int(metadata[f"orig_{ax}_size"][i]) for ax in ax_names]
            orig_image_sizes.append(tuple(spatial_dims))
    else:
        orig_image_sizes = image_sizes_padded

    padding_mask = torch.zeros(
        (batch_size, *spatial_shape),
        dtype=torch.bool,
        device=device,
    )

    # metadata keys for sizes: z_size, y_size, x_size
    size_keys = {ax: f"{ax.lower()}_size" for ax in spatial_axes}

    for b in range(batch_size):
        # actual sizes along each spatial axis (default: full size if missing)
        actual = {}
        for ax in spatial_axes:
            key = size_keys[ax]
            if key in metadata:
                actual[ax] = int(metadata[key][b])
            else:
                actual[ax] = full_sizes[ax]

        # Mark padded voxels as True
        # We want: padded if index >= actual[ax] along ANY spatial axis.
        if spatial_axes == ["Z", "Y", "X"]:
            Z_full, Y_full, X_full = full_sizes["Z"], full_sizes["Y"], full_sizes["X"]
            z_lim, y_lim, x_lim = actual["Z"], actual["Y"], actual["X"]

            if z_lim < Z_full:
                padding_mask[b, z_lim:, :, :] = True
            if y_lim < Y_full:
                padding_mask[b, :, y_lim:, :] = True
            if x_lim < X_full:
                padding_mask[b, :, :, x_lim:] = True

        elif spatial_axes == ["Y", "X"]:
            Y_full, X_full = full_sizes["Y"], full_sizes["X"]
            y_lim, x_lim = actual["Y"], actual["X"]

            if y_lim < Y_full:
                padding_mask[b, y_lim:, :] = True
            if x_lim < X_full:
                padding_mask[b, :, x_lim:] = True

        else:
            raise ValueError(f"Unsupported spatial_axes combination: {spatial_axes}")

    return image_sizes, orig_image_sizes, image_sizes_padded, padding_mask


def set_global_seed(seed: int):
    # Python built-in RNG
    random.seed(seed)
    # Numpy RNG
    np.random.seed(seed)
    # Torch CPU RNG
    torch.manual_seed(seed)
    # Torch CUDA RNG (all devices)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def aggregate_microbatch_losses(
    loss_dicts: Sequence[dict],
    gradient_accumulation_steps: int
) -> dict:
    """
    Aggregate scalar losses across microbatches.
    - Sums scalar entries (tensors or floats/ints) across all loss_dicts
    - Averages by gradient_accumulation_steps
    - Returns a new dict with detached CPU tensors
    - Ensures 'step_loss' exists and is a tensor
    """
    agg: dict[str, torch.Tensor] = {}
    denom = float(gradient_accumulation_steps)

    for d in loss_dicts:
        for k, v in d.items():
            if torch.is_tensor(v):
                if v.numel() != 1:
                    continue
                val = v.detach().float().cpu()
            elif isinstance(v, (float, int)):
                val = torch.tensor(float(v))
            else:
                continue

            if k in agg:
                agg[k] = agg[k] + val
            else:
                agg[k] = val

    # Average over microbatches
    for k in agg:
        agg[k] = agg[k] / denom

    if "step_loss" not in agg:
        raise KeyError(
            "Expected 'step_loss' in loss_dicts when aggregating microbatch losses."
        )

    if not torch.is_tensor(agg["step_loss"]):
        agg["step_loss"] = torch.as_tensor(agg["step_loss"])

    return agg


def configure_torch_comm_env(comm_config):
    def _warn_overwrite_env(env, val):
        if env in os.environ:
            logger.warning(
                f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config."
            )
        os.environ[env] = val

    TRACE_BUFFER_SIZE = "TORCH_FR_BUFFER_SIZE"
    TRACE_FILE = "TORCH_FR_DUMP_TEMP_FILE"
    DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
    ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
    SKIP_CLEANUP = "3"

    # FlightRecorder is incompatible with =1 mode where watchdog aborts work, must use =3 (skipcleanup)
    # to get flight recorder dumps. See https://github.com/pytorch/pytorch/issues/121055
    # This could be done only when flight recorder is enabled, but its nice 
    # to be consistent to avoid subtle behavior differences
    _warn_overwrite_env(ASYNC_ERROR_HANDLING, SKIP_CLEANUP)

    # enable torch nccl flight recorder in the mode that would dump files if timeout is detected
    _warn_overwrite_env(TRACE_BUFFER_SIZE, str(comm_config.trace_buf_size))
    if comm_config.trace_buf_size > 0:
        # dump on timeout by default if trace buffer is enabled
        _warn_overwrite_env(DUMP_ON_TIMEOUT, "1")
        dump_dir = os.path.join(comm_config.dump_base_folder, comm_config.save_traces_folder)
        prefix = comm_config.save_traces_file_prefix
        os.makedirs(dump_dir, exist_ok=True)
        _warn_overwrite_env(TRACE_FILE, f"{dump_dir}/{prefix}")


def get_nparams_and_flops(
    model: nn.Module,
    data_sample: dict,
    seq_len: int,
) -> dict:
    """
    Convenience helper to get parameter counts and MACs/FLOPs
    for the given model + example batch, using torchinfo.summary.
    """

    if isinstance(data_sample, dict):
        input_data = (data_sample,)
    elif isinstance(data_sample, (list, tuple)):
        input_data = data_sample
    else:
        input_data = (data_sample,)

    eval_stats = summary(
        model=model,
        input_data=input_data,
        depth=3,
        col_width=25,
        col_names=["num_params", "mult_adds"],
        row_settings=["var_names"],
        verbose=0,
        mode="eval",
    )

    train_stats = summary(
        model=model,
        input_data=input_data,
        depth=3,
        col_width=25,
        col_names=["num_params", "mult_adds"],
        row_settings=["var_names"],
        verbose=0,
        mode="train",
    )

    total_params = int(eval_stats.total_params)
    trainable_params = int(eval_stats.trainable_params)

    eval_macs = int(eval_stats.total_mult_adds or 0)
    training_macs = int(train_stats.total_mult_adds or 0)

    # FLOPs ≈ 2 * MACs (mult + add)
    eval_flops = 2 * eval_macs
    training_flops = 2 * training_macs

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "eval_macs": eval_macs,
        "training_macs": training_macs,
        "eval_flops": eval_flops,
        "training_flops": training_flops,
        "training_flops_per_token": training_flops // seq_len
    }


def get_dense_model_nparams_and_flops(
    num_layers: int,
    num_heads: int,
    head_dims: int,
    model: nn.Module,
    seq_len: int,
) -> tuple[int, float]:
    """
    Args:
        num_layers: The number of layers in the model.
        num_heads: The number of attention heads in the model.
        model: nn.Module representing the model.
        head_dims: The sum of qk and v head dimensions.
        seq_len: The sequence length in training configs.

    Returns:
        Tuple of (nparams, num_flops_per_token):
            nparams: Total number of model parameters.
            num_flops_per_token: Estimated number of floating point operations per token.
    """
    nparams = sum(p.numel() for p in model.parameters())
    nparams_embedding = sum(
        sum(p.numel() for p in m.parameters())
        for m in model.children()
        if isinstance(m, nn.Embedding)
    )

    # Reasoning behind the factor of 6 for the self-attention part of the formula:
    # 1. each self-attention has 2 matmul (attention scores and value aggregation,
    #    combined in head_dims, counted as 1) in the forward and 4 (counted as 2)
    #    in the backward                                                      (3)
    # 2. the flash attention does 1 more matmul recomputation in the backward
    #    but recomputation should not be counted in calculating MFU           (+0)
    # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
    # 4. we follow the convention and do not account for sparsity in causal attention
    num_flops_per_token = (
        6 * (nparams - nparams_embedding)
        + 6 * num_layers * num_heads * head_dims * seq_len
    )

    return nparams, num_flops_per_token


HASH_COLS = ["prepared_id", "tile_name", "z_start", "y_start", "x_start", "time_start"]


def df_signature_polars(df_pd) -> int:
    df_pl = pl.from_pandas(df_pd[HASH_COLS])
    # UInt64 per row
    row_hashes = df_pl.hash_rows(seed=0)
    # Reduce to one scalar
    h = row_hashes.to_numpy()
    sig_u64 = np.bitwise_xor.reduce(h)
    # convert to a signed int64
    return int(sig_u64.view(np.int64))


def assert_same_db_hash_across_ranks(local_hash: int, group=None):
    if group is None:
        group = dist.group.WORLD
    hash_tensor = torch.tensor([local_hash], dtype=torch.int64, device="cuda")
    dist.all_reduce(hash_tensor, op=dist.ReduceOp.BXOR, group=group)
    if hash_tensor.item() != 0:
        raise RuntimeError(
            f"[RANK {dist.get_rank()}] Database hash mismatch across ranks "
            f"(xor != 0) - shards / filters not identical!"
        )
