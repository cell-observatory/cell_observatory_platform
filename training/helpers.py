import math
import ujson
import logging
from pathlib import Path
from operator import attrgetter
from collections import defaultdict
from typing import Optional, List, Union, Callable, Tuple

import numpy as np

import torch
import torch.nn as nn
from torchinfo import summary
from timm.layers.weight_init import trunc_normal_
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointWrapper,
)

from omegaconf import DictConfig, open_dict

import ray

logger = logging.getLogger("ray")
logger.setLevel(logging.INFO)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


def record_dataset_len(config, num_train_steps: int, num_val_steps: int):
    bs = config.clusters.batch_size_per_gpu
    world_size = ray.train.get_context().get_world_size()
    drop_last = bool(getattr(config.datasets, "drop_last_policy"))

    def steps_from_rows(n_rows: int):
        if drop_last:
            min_rows_per_worker = n_rows // world_size
            return (min_rows_per_worker // bs)
        else:
            return math.ceil(n_rows / (world_size * bs))

    steps_per_epoch = steps_from_rows(num_train_steps)
    val_steps_per_epoch = steps_from_rows(num_val_steps) if num_val_steps > 0 else None
    
    with open_dict(config):
        config.runtime = {"train_steps_per_epoch": steps_per_epoch, 
                       "val_steps_per_epoch": val_steps_per_epoch,
                       "n_train_rows": num_train_steps,
                       "n_val_rows": num_val_steps}


def _infer_steps_per_epoch(config, loader, batch_size, type: str = "train"):    
    if config.datasets.dataset._target_.endswith("PretrainDatasetDali") or \
        config.datasets.dataset._target_.endswith("PretrainDataset"):
        return len(loader)
    
    elif config.datasets.dataset._target_.endswith("PretrainDatasourceRay"):
        if type == "train":
            return config.runtime.get("train_steps_per_epoch")
        elif type == "val":
            return config.runtime.get("val_steps_per_epoch")
    
    else:
        raise TypeError(
            f"Cannot infer steps/epoch for loader type {type(loader)}. "
            f"Extend the _infer_steps_per_epoch function to handle this type."
        )


def get_steps_per_epoch(train_dataloader, val_dataloader, config: DictConfig):
    # TODO: double check correctness
    steps_per_epoch = _infer_steps_per_epoch(config, 
                                             train_dataloader, 
                                             config.clusters.batch_size_per_gpu, 
                                             type="train")
    val_steps_per_epoch = _infer_steps_per_epoch(config, 
                                                 val_dataloader, 
                                                 config.clusters.batch_size_per_gpu, 
                                                 type="val") if val_dataloader else None
    logger.info(
        f"Steps per epoch: {steps_per_epoch}, "
        f"Validation steps per epoch: {val_steps_per_epoch}"
    )
    return steps_per_epoch, val_steps_per_epoch


# NOTE: we store best loss, starting epoch and starting step
#       with checkpoint manager in client state
#       resume model state is most useful when restarting a 
#       job from an earlier checkpoint to sidestep training 
#       instabilities.
#       with resume_model_state only the checkpoint directory
#       and the checkpoint tag need be specified whereafter
#       any checkpoint with corresponding iter, epoch, best_loss
#       will be loaded from the checkpoint directory.
def resume_model_state(config: DictConfig, checkpoint_manager):
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


def resume_run(trainer, config: DictConfig):
    Path(config.paths.outdir).mkdir(exist_ok=True, parents=True)
    if config.paths.resume_checkpointdir:
        best_loss, iter, epoch = resume_model_state(config, 
                                    checkpoint_manager=trainer.checkpoint_manager)        
        trainer.event_recorder.resume(
            iter=iter, 
            epoch=epoch
        )

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
    logdir: Path|str,
    inputs: Optional[tuple] = None,
    input_data: Optional[dict] = None
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
        mode='train'
    )

    with (logdir / 'model.log').open('w') as f:
        f.write(str(model_stats))

    model_logbook['training_batch_size'] = batch_size
    model_logbook['input_bytes'] = model_stats.total_input
    model_logbook['total_params'] = model_stats.total_params
    model_logbook['trainable_params'] = model_stats.trainable_params
    model_logbook['param_bytes'] = model_stats.total_param_bytes

    model_logbook['eval_macs'] = model_stats.total_mult_adds
    model_logbook['training_macs'] = train_stats.total_mult_adds

    model_logbook['forward_pass_bytes'] = model_stats.total_output_bytes
    model_logbook['forward_backward_pass_bytes'] = train_stats.total_output_bytes

    model_logbook['eval_model_bytes'] = model_logbook['param_bytes'] \
        + model_logbook['forward_pass_bytes']
    model_logbook['training_model_bytes'] = model_logbook['param_bytes'] \
        + model_logbook['forward_backward_pass_bytes']

    model_logbook['eval_bytes'] = model_logbook['input_bytes'] + \
        model_logbook['eval_model_bytes']
    model_logbook['training_bytes'] = model_logbook['input_bytes'] + \
        model_logbook['training_model_bytes']

    model_logbook['layers'] = {}
    for layer in train_stats.summary_list:
        if layer.is_leaf_layer:
            model_logbook['layers'][f'{layer.class_name}_{layer.var_name}'] = {
                'macs': layer.macs,
                'params': max(layer.num_params, 0),
                'param_bytes': layer.param_bytes,
                'forward_pass_bytes': layer.output_bytes,
                'forward_backward_pass_bytes': layer.output_bytes * 2, # x2 for gradients
                'output_shape': layer.output_size,
            }

    with (logdir / 'model_logbook.json').open('w') as f:
        ujson.dump(
            model_logbook,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )


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

    if type == "train":
        trainer.event_recorder.put_scalars(
            scope="step",
            **{k: (v.item() if torch.is_tensor(v) else v)
            for k, v in loss_dict.items()
            }
        )


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


def enable_optimizations(cfg: DictConfig, model: nn.Module):
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

    # activation checkpointing 
    if cfg.optimizations.activation_checkpoint.enabled:
        apply_activation_checkpointing(cfg, model)
    
    # torch compile optimizations
    if cfg.optimizations.torch_compile.enabled:
        model = apply_compile(model, cfg)

    print_model_tree_with_opt(model, treat_wrappers_as_leaves=False) # sanity check
    return model


# many of the below functions are based on: 
# https://github.com/pytorch/torchtitan/main/torchtitan/models/llama3/infra/parallelize.py
def apply_activation_checkpointing(cfg: DictConfig, model: nn.Module):
    """Apply activation checkpointing to the model."""
    num_blocks = apply_ac_over_discovered_stacks(cfg, model)
    logger.info(f"Applied activation checkpointing to {num_blocks} transformer blocks.")


# TODO: we should consider consolidating all these types of objects
# to identify which models currently support automated activation checkpointing
# where the user does not need to specify the modules
_ac_ckpt_supported_models_list = ["MaskedAutoEncoder", "JEPA"]
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


def _as_stack(path_str: str, 
              submod, 
              blocks_nomenclature: str = "transformer_blocks"
) -> Tuple[str, nn.ModuleList]:
    """
    Normalize a object into a (stack_fqn, stack_container) pair.
    Accept ModuleList/Sequential directly, or parent modules that own a
    transformer blocks ModuleList.
    """
    # direct container
    if isinstance(submod, (nn.ModuleList, nn.Sequential)):
        return path_str, submod

    # parent form: grab its blocks named `blocks_nomenclature`
    if hasattr(submod, blocks_nomenclature) and \
        isinstance(getattr(submod, blocks_nomenclature), nn.ModuleList):
        return f"{path_str}.{blocks_nomenclature}", getattr(submod, blocks_nomenclature)

    raise TypeError(
        f"Config path '{path_str}' resolved to {type(submod).__name__}; "
        f"expected ModuleList/Sequential or a module with '.{blocks_nomenclature}'."
    )


# TODO: break `yield_transformer_stacks_for_act_ckpt` and 
#       `yield_transformer_stacks_for_compile` into one helper function
#        and two smaller functions that yield the stacks for each case to
#        avoid code duplication

def yield_transformer_stacks_for_act_ckpt(cfg, model: nn.Module):
    # if user provided explicit modules, honor them
    cfg_modules = getattr(cfg.optimizations.activation_checkpoint, "modules", None)
    blocks_name = getattr(cfg.optimizations.activation_checkpoint, "blocks_nomenclature", "transformer_blocks")
    if cfg_modules:
        for module_fqn in cfg_modules:
            try:
                submod = attrgetter(module_fqn)(model)
            except AttributeError as e:
                raise AttributeError(f"Could not resolve '{module_fqn}' on model: {e}") from e
            stack_fqn, stack = _as_stack(module_fqn, submod, blocks_nomenclature=blocks_name)
            yield (stack_fqn, stack)
        return
    
    # if user does not provide explicit modules, we auto-discover
    # transformer stacks in the model. 
    # since our current activation checkpointing implementation
    # is based on correctly identifying transformer blocks
    # we catch model class names that are not supported here
    # if model.__class__.__name__ is not in _ac_ckpt_supported_models_list 
    # then the user should extend these functions to support their model
    # or explicitly provide the modules to apply activation checkpointing
    # via the config.
    if model.__class__.__name__ not in _ac_ckpt_supported_models_list:
        raise ValueError(
            f"Model {model.__class__.__name__} is not supported for activation checkpointing. "
            f"Supported models: {_ac_ckpt_supported_models_list}"
        )

    # fallback auto-discovery 
    # MAE
    if hasattr(model, "masked_encoder") and hasattr(model.masked_encoder, "encoder"):
        if hasattr(model.masked_encoder.encoder, blocks_name):
            yield (f"masked_encoder.encoder.{blocks_name}", getattr(model.masked_encoder.encoder, blocks_name))
    if hasattr(model, "masked_decoder") and hasattr(model.masked_decoder, "encoder"):
        if hasattr(model.masked_decoder.encoder, blocks_name):
            yield (f"masked_decoder.encoder.{blocks_name}",
                   getattr(model.masked_decoder.encoder, blocks_name))
    # JEPA
    if hasattr(model, "input_encoder") and hasattr(model.input_encoder, "encoder"):
        if hasattr(model.input_encoder.encoder, blocks_name):
            yield (f"input_encoder.encoder.{blocks_name}",
                   getattr(model.input_encoder.encoder, blocks_name))
    if hasattr(model, "target_predictor") and hasattr(model.target_predictor, "encoder"):
        if hasattr(model.target_predictor.encoder, blocks_name):
            yield (f"target_predictor.encoder.{blocks_name}",
                   getattr(model.target_predictor.encoder, blocks_name))


def yield_transformer_stacks_for_compile(cfg: DictConfig, model: nn.Module):
    # honor explicit module paths if provided
    cfg_modules = getattr(cfg.optimizations.torch_compile, "modules", None)
    blocks_name = getattr(cfg.optimizations.torch_compile, "blocks_nomenclature", "transformer_blocks")
    if cfg_modules:
        for module_fqn in cfg_modules:
            submod = attrgetter(module_fqn)(model)  # raises if missing
            stack_fqn, stack = _as_stack(module_fqn, submod, blocks_nomenclature=blocks_name)
            yield (stack_fqn, stack)
        return

    # fallback auto-discovery
    # MAE
    if hasattr(model, "masked_encoder") and hasattr(model.masked_encoder, "encoder"):
        if hasattr(model.masked_encoder.encoder, blocks_name):
            yield (f"masked_encoder.encoder.{blocks_name}", getattr(model.masked_encoder.encoder, blocks_name))
    if hasattr(model, "masked_decoder") and hasattr(model.masked_decoder, "encoder"):
        if hasattr(model.masked_decoder.encoder, blocks_name):
            yield (f"masked_decoder.encoder.{blocks_name}", getattr(model.masked_decoder.encoder, blocks_name))
    # JEPA
    if hasattr(model, "input_encoder") and hasattr(model.input_encoder, "encoder"):
        if hasattr(model.input_encoder.encoder, blocks_name):
            yield (f"input_encoder.encoder.{blocks_name}", getattr(model.input_encoder.encoder, blocks_name))
    if hasattr(model, "target_predictor") and hasattr(model.target_predictor, "encoder"):
        if hasattr(model.target_predictor.encoder, blocks_name):
            yield (f"target_predictor.encoder.{blocks_name}", getattr(model.target_predictor.encoder, blocks_name))
    # also compile the frozen target encoder
    if hasattr(model, "target_encoder") and hasattr(model.target_encoder, "encoder"):
        if hasattr(model.target_encoder.encoder, blocks_name):
            yield (f"target_encoder.encoder.{blocks_name}", getattr(model.target_encoder.encoder, blocks_name))


def apply_ac_over_discovered_stacks(cfg, model: nn.Module):
    wrapped = 0
    for stack_fqn, stack in yield_transformer_stacks_for_act_ckpt(cfg, model):
        for i, block in enumerate(stack):
            wrapped_block = _apply_ac_to_module(
                module=block,
                act_ckpt_mode=cfg.optimizations.activation_checkpoint.mode,
                selective_ac_option=cfg.optimizations.activation_checkpoint.selective_ac_option,
                per_op_sac_force_recompute_mm_shapes_by_fqns=\
                    cfg.optimizations.activation_checkpoint.per_op_sac_force_recompute_mm_shapes_by_fqns,
                base_fqn=f"{stack_fqn}.{i}",
                mm_recompute_frac=cfg.optimizations.activation_checkpoint.mm_recompute_frac,
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
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

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


def apply_compile(model: nn.Module, cfg: DictConfig):
    if cfg.optimizations.torch_compile.range == "full":
        logger.info("Applying torch.compile to the whole model.")
        model = torch.compile(
            model,
            dynamic=cfg.optimizations.torch_compile.dynamic,
            mode=cfg.optimizations.torch_compile.mode,
            fullgraph=False,  # DS causes graph breaks -> keep False here
        )
        # mark whole-model compilation so printer can tag the root
        setattr(model, "_is_compiled", True)
        setattr(model, "_compiled_fqns", set(["<whole_model>"]))
    elif cfg.optimizations.torch_compile.range == "block_based":
        num_blocks_compiled = _apply_compile_over_discovered_stacks(cfg, model)
        logger.info(f"Applied torch.compile to {num_blocks_compiled} transformer blocks.")
    else:
        raise ValueError(
            f"Invalid torch compile mode: {cfg.optimizations.torch_compile.mode}. "
            "Valid modes: 'full' or 'block_based'"
        )
    
    return model


def _apply_compile_over_discovered_stacks(cfg, model: nn.Module):
    """
    Apply torch.compile to each Transformer block, and record which blocks were compiled
    so the tree printer can show [TC].
    """
    num_blocks_compiled = 0
    compiled_fqns = set()

    for stack_fqn, stack in yield_transformer_stacks_for_compile(cfg, model):
        for i, block in enumerate(stack):
            compiled = torch.compile(
                block,
                fullgraph=getattr(cfg.optimizations.torch_compile, "blockbased_fullgraph", False),
                dynamic=getattr(cfg.optimizations.torch_compile, "dynamic", False),
                mode=getattr(cfg.optimizations.torch_compile, "mode", "default"),
            )
            # mark and re-register
            setattr(compiled, "_is_compiled", True)  # helpful heuristic for printer
            stack[i] = compiled
            compiled_fqns.add(f"{stack_fqn}.{i}")
            num_blocks_compiled += 1

    # stash a summary for the printer
    setattr(model, "_compiled_fqns", compiled_fqns)
    return num_blocks_compiled


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
    if weight_init_type == "mae":
        # MAE model init utility function adapted from:
        # https://github.com/facebookresearch/mae/main/models_mae.py
        def _mae_init_weights(m):
            if isinstance(m, nn.Linear):
                # use xavier_uniform following official JAX ViT
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
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

         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = model.masked_encoder.patch_embedding.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively 
        # normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(model.masked_decoder.token_param, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        model.apply(_mae_init_weights)

    if weight_init_type == "vjepa":
        # helpers from: 
        # https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py
        def _vjepa_fix_init_weight(model):
            def rescale(param, layer_id):
                param.div_(math.sqrt(2.0 * layer_id))

            for layer_id, layer in enumerate(model.encoder.transformer_blocks):
                # important to match names with timm MLP and our Attention module
                rescale(layer.att.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        def _vjepa_init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=model.init_std)
                if isinstance(m, nn.Linear) and m.bias is not None:
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

        # NOTE: IJEPA does initlization of encoder and decoder separately
        #       however they are intialized the same way hence we
        #       just do a single named_apply over the whole model
        trunc_normal_(model.masked_decoder.token_param, std=model.init_std)
        model.apply(_vjepa_init_weights)
        # rescale blocks for input encoder and target predictor
        _vjepa_fix_init_weight(model.input_encoder)
        _vjepa_fix_init_weight(model.target_predictor)

    elif weight_init_type == "vjepa2":
        # helpers from:
        # https://github.com/facebookresearch/vjepa2/main/src/models/vision_transformer.py
        def _vjepa2_init_weights(self, m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=self.init_std)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            # NOTE: technically vjepa2 only applies the below
            #       to input encoder and not target predictor
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=self.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        def _vjepa2_rescale_blocks(model):
            def rescale(param, layer_id):
                param.div_(math.sqrt(2.0 * layer_id))

            for layer_id, layer in enumerate(model.encoder.transformer_blocks):
                rescale(layer.att.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

        model.apply(_vjepa2_init_weights)
        _vjepa2_rescale_blocks(model.input_encoder)
        _vjepa2_rescale_blocks(model.target_predictor)

    else:
        raise ValueError(f"Unknown weight initialization type: {weight_init_type}")