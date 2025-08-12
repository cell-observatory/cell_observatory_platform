import sys
import math
import ujson
import logging
from pathlib import Path
from functools import wraps
from operator import attrgetter
from typing import Optional, Tuple, Iterator

import numpy as np

import torch
import torch.nn as nn
from torchinfo import summary
from torch.optim.lr_scheduler import LinearLR
from timm.scheduler import create_scheduler_v2

from omegaconf import DictConfig

from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.lamb import FusedLamb
from deepspeed.runtime.lr_schedules import WarmupCosineLR
from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint

import ray

from training.loggers import WandBEventWriter

logger = logging.getLogger("ray")
logger.setLevel(logging.INFO)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


def get_lr_scheduler(
    opt: torch.optim.Optimizer,
    steps_per_epoch: int,
    config: DictConfig,
    decay: str = 'cosine'
):
    if config.schedulers.fixedlr:
        scheduler = LinearLR(
            opt,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=config.schedulers.epochs,
        )
        logger.info(f"Training steps: [{steps_per_epoch * config.schedulers.epochs}]")
    else:
        decay_epochs = config.schedulers.epochs - (config.schedulers.warmup + config.schedulers.cooldown)
        total_steps = config.schedulers.epochs * steps_per_epoch
        warmup_steps = config.schedulers.warmup * steps_per_epoch
        cooldown_steps = config.schedulers.cooldown * steps_per_epoch
        decay_steps = total_steps - (warmup_steps + cooldown_steps)

        cos_min_lr = config.schedulers.cos_min_ratio * config.optimizers.lr
        warmup_min_lr = config.schedulers.warmup_min_ratio * config.optimizers.lr

        logger.info('-'*80)
        logger.info(
            f"Epochs: {config.schedulers.epochs} = "
            f"[{config.schedulers.warmup} warmup + {decay_epochs} decay + {config.schedulers.cooldown} cooldown]\n"
            f"Steps: {total_steps} = "
            f"[{warmup_steps} warmup + {decay_steps} decay + {cooldown_steps} cooldown]\n"
            f"LR: {config.optimizers.lr} = [{warmup_min_lr=},  {cos_min_lr=}]"
        )
        logger.info('-'*80)

        scheduler, num_epochs = create_scheduler_v2(
            optimizer=opt,
            sched=decay,
            num_epochs=config.schedulers.epochs,
            warmup_epochs=config.schedulers.warmup,
            cooldown_epochs=config.schedulers.cooldown,
            decay_epochs=decay_epochs,
            min_lr=cos_min_lr,
            warmup_lr=warmup_min_lr,
        )

    return scheduler


def get_optimizer(
    params,
    config: DictConfig,
    optimizer: str,
    steps_per_epoch: int,
    deepspeed_scheduler: bool = False
):
    if optimizer == 'adamw':
        opt = FusedAdam(
            params,
            lr=config.optimizers.lr,
            weight_decay=config.optimizers.wd,
            betas=(0.9, 0.99),
            eps=1e-08,
        )
    elif optimizer == 'lamb':
        opt = FusedLamb(
            params,
            lr=config.optimizers.lr,
            weight_decay=config.optimizers.wd,
            betas=(0.9, 0.99),
            eps=1e-08,
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    if deepspeed_scheduler:
        decay_epochs = config.schedulers.epochs - (config.schedulers.warmup + config.schedulers.cooldown)
        total_steps = config.schedulers.epochs * steps_per_epoch
        warmup_steps = config.schedulers.warmup * steps_per_epoch
        decay_steps = total_steps - warmup_steps

        cos_min_lr = config.schedulers.cos_min_ratio * config.optimizers.lr
        warmup_min_lr = config.schedulers.warmup_min_ratio * config.optimizers.lr

        logger.info('-'*80)
        logger.info(
            f"Epochs: {config.schedulers.epochs} = "
            f"[{config.schedulers.warmup} warmup + {decay_epochs} decay + NA cooldown]\n"
            f"Steps: {total_steps} = "
            f"[{warmup_steps} warmup + {decay_steps} decay + NA cooldown]\n"
            f"LR: {config.optimizers.lr} = [{warmup_min_lr=},  {cos_min_lr=}]"
        )
        logger.info('-'*80)

        scheduler = WarmupCosineLR(
            optimizer=opt,
            total_num_steps=total_steps,
            warmup_num_steps=warmup_steps,
            warmup_min_ratio=config.schedulers.warmup_min_ratio,
            cos_min_ratio=config.schedulers.cos_min_ratio,
            warmup_type=config.schedulers.cos_miwarmup_type,
        )

        return opt, scheduler
    else:
        return opt, None


def _infer_steps_per_epoch(loader, batch_size, type: str = "train"):
    if loader is None:
        return None

    # PyTorch/DALI
    try:
        return len(loader)
    except TypeError:
        pass

    # Ray Dataset iterator or wrapped Ray Dataset iterator for auto
    # transfer to GPU (see cell_observatory_platform/data/datasets/pretrain_dataset_ray.py)
    if isinstance(loader, ray.data.iterator._IterableFromIterator) or \
        isinstance(loader.data_iter, Iterator):
        if type == "train":
            dataset = ray.train.get_dataset_shard("train")
            rows = dataset._base_dataset.count()
        elif type == "val":
            dataset = ray.train.get_dataset_shard("val")
            rows = dataset._base_dataset.count()
        else:
            raise ValueError(f"Unknown dataset type: {type}")
        return math.ceil(rows / batch_size)

    raise TypeError(
        f"Cannot infer steps/epoch for loader type {type(loader)}. "
        f"Extend the _infer_steps_per_epoch function to handle this type."
    )


def get_steps_per_epoch(train_dataloader, val_dataloader, config: DictConfig):
    # TODO: double check correctness
    steps_per_epoch = _infer_steps_per_epoch(train_dataloader, 
                                             config.clusters.batch_size_per_gpu, 
                                             type="train")
    val_steps_per_epoch = _infer_steps_per_epoch(val_dataloader, 
                                                 config.clusters.batch_size_per_gpu, 
                                                 type="val") if val_dataloader else None
    logger.info(
        f"Steps per epoch: {steps_per_epoch}, "
        f"Validation steps per epoch: {val_steps_per_epoch}"
    )
    return steps_per_epoch, val_steps_per_epoch


# TODO: we store best loss, starting epoch and starting step
#       with checkpoint manager in client state
#       resume model state is most useful when restarting a 
#       job from an earlier checkpoint to sidestep training 
#       instabilities.
#       with resume_model_state only the checkpoint directory
#       and the checkpoint tag need be specified whereafter
#       any checkpoint with corresponding iter, epoch, best_loss
#       will be loaded from the checkpoint directory.
#       see: https://arxiv.org/pdf/2204.02311 for
#       strategies to resume training after training 
#       instabilities
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


def activation_checkpoint(cfg, model):
    """Wrap listed sub-modules with activation-checkpointing."""
    def wrap_forward(forward):
        @wraps(forward)
        def wrapper(*args, **kwargs):
            return checkpoint(forward, use_reentrant=False, *args, **kwargs)
        wrapper._is_ckpt_wrapped = True
        return wrapper

    for mod_name in cfg.optimizations.activation_checkpoint.modules:
        mod = attrgetter(mod_name)(model)
        if not getattr(mod.forward, "_is_ckpt_wrapped", False):
            mod.forward = wrap_forward(mod.forward)


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
        torch.use_deterministic_algorithms(True)

    # activation checkpointing 
    if cfg.optimizations.activation_checkpoint.enabled:
        logger.info(
            f"Enabling activation checkpointing for modules: \
                {cfg.optimizations.activation_checkpoint.modules}"
        )
        activation_checkpoint(cfg, model)
        # TODO: add deepspeed checkpointing config logic
        # deepspeed.checkpointing.configure(**cfg.deepspeed.checkpointing)
    
    # torch compile optimizations
    if cfg.optimizations.torch_compile.enabled:
        model = torch.compile(model,
                            dynamic = cfg.optimizations.torch_compile.dynamic,
                            # options: "default", "reduce-overhead", "max-autotune"
                            # reduced overhead is for small models
                            # max-autotune takes long but gives largest speedup
                            # see: https://pytorch.org/get-started/pytorch-2-x/#user-experience
                            mode = cfg.optimizations.torch_compile.mode,
                            # DeepSpeed generates graph breaks
                            # hence we must disable fullgraph
                            fullgraph = False)

    return model


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
        )

    masking_time = data_sample['metainfo'].get('masking_time', None)
    if masking_time is not None:
        trainer.event_recorder.put_scalars(
            scope="step",
            prefix="val_" if type == "val" else None,
            masking_time=masking_time,
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

    if type == "train":
        trainer.event_recorder.put_scalars(
            scope="step",
            **{k: (v.item() if torch.is_tensor(v) else v)
            for k, v in loss_dict.items()
            }
        )

    if idx == 0 and trainer.event_writers_list is not None:
        try:
            wandb_writer = next(
                w for w in trainer.event_writers_list.writers
                if isinstance(w, WandBEventWriter)
            )
        except StopIteration:   
            return


        # keys expected by the writer (from Hydra/YAML config)
        expected_step_keys  = set(wandb_writer.step_scalar_keys)
        expected_epoch_keys = set(wandb_writer.epoch_scalar_keys)
        # filter out keys that are not relevant for the current type of loop
        if type == "train":
            expected_step_keys  = {k for k in expected_step_keys  if not k.startswith(("val_", "test_"))}
            expected_epoch_keys = {k for k in expected_epoch_keys if not k.startswith(("val_", "test_"))}
        elif type == "val":
            expected_step_keys  = {k for k in expected_step_keys  if k.startswith("val_")}
            expected_epoch_keys = {k for k in expected_epoch_keys if k.startswith("val_")}

        # keys that have actually been recorded so far by the recorder
        recorded_step_keys = set(trainer.event_recorder.get_step_scalars().keys())
        recorded_epoch_keys = set(trainer.event_recorder.get_epoch_scalars().keys())

        unexpected_step = recorded_step_keys  - expected_step_keys
        missing_step = expected_step_keys  - recorded_step_keys
        unexpected_ep = recorded_epoch_keys - expected_epoch_keys
        # it's hard to guard against missing epoch keys since they 
        # are not logged until after the epoch ends but this doesn't
        # really matter since the logger will throw an error
        # anyways at the end of the epoch if the keys are missing
        missing_ep = expected_epoch_keys - recorded_epoch_keys

        assert not unexpected_step, (
            f"[WandB] step-scalar(s) {sorted(unexpected_step)} were logged "
            f"but are not listed in WandBEventWriter.step_scalar_keys"
        )
        assert not unexpected_ep, (
            f"[WandB] epoch-scalar(s) {sorted(unexpected_ep)} were logged "
            f"but are not listed in WandBEventWriter.epoch_scalar_keys"
        )
        assert not missing_step, (
            f"[WandB] step-scalar key(s) {sorted(missing_step)} were declared "
            f"in WandBEventWriter.step_scalar_keys but never logged "
            f"in the first iteration"
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
