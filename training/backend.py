import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchinfo import summary
from torch.optim.lr_scheduler import LinearLR
from timm.scheduler import create_scheduler_v2
from typing import Optional, Union

from deepspeed import initialize, DeepSpeedEngine
from deepspeed.runtime.lr_schedules import WarmupCosineLR
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.lamb import FusedLamb

import ray.train.torch as raytorch
from ray.train import Checkpoint, report, get_context, get_checkpoint

from omegaconf import DictConfig, OmegaConf

import logging
import ujson
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import nullcontext

from data import ao_dataset
from training import masking
from training.checkpointing import load_checkpoint
from training.earlystopping import EarlyStoppingCallback
from training.registry import build_dependency_graph_and_instantiate

logger = logging.getLogger("ray")
logger.setLevel(logging.DEBUG)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


def is_main_process():
    return get_context().get_world_rank() == 0

def process_rank():
    return get_context().get_world_rank()


def summarize_model(model: nn.Module, inputs: tuple, batch_size: int, logdir: Path):
    model_logbook = {}
    model_stats = summary(
        model=model,
        input_size=(1, *inputs[1:]),
        depth=5,
        col_width=25,
        col_names=["kernel_size", "output_size", "num_params"],
        row_settings=["var_names"],
        verbose=0,
        mode='eval'
    )
    train_stats = summary(
        model=model,
        input_size=inputs,
        depth=5,
        col_width=25,
        col_names=["kernel_size", "output_size", "num_params"],
        row_settings=["var_names"],
        verbose=1,
        mode='train'
    )

    with Path(logdir / 'model.log').open('w') as f:
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

    model_logbook['eval_model_bytes'] = model_logbook['param_bytes'] + model_logbook['forward_pass_bytes']
    model_logbook['training_model_bytes'] = model_logbook['param_bytes'] + model_logbook['forward_backward_pass_bytes']

    model_logbook['eval_bytes'] = model_logbook['input_bytes'] + model_logbook['eval_model_bytes']
    model_logbook['training_bytes'] = model_logbook['input_bytes'] + model_logbook['training_model_bytes']

    model_logbook['layers'] = {}
    for layer in train_stats.summary_list:
        if layer.is_leaf_layer:
            model_logbook['layers'][f'{layer.class_name}_{layer.var_name}'] = {
                'macs': layer.macs,
                'params': max(layer.num_params, 0),
                'param_bytes': layer.param_bytes,
                'forward_pass_bytes': layer.output_bytes,
                'forward_backward_pass_bytes': layer.output_bytes * 2,  # x2 for gradients
                'output_shape': layer.output_size,
            }

    with Path(logdir / 'model_logbook.json').open('w') as f:
        ujson.dump(
            model_logbook,
            f,
            indent=4,
            sort_keys=False,
            ensure_ascii=False,
            escape_forward_slashes=False
        )


def restore_model(config: DictConfig):
    outdir = Path(config.outdir)
    try:  # check if model already exists
        checkpoints = [
            d for d in outdir.rglob('checkpoint*')
            if d.is_dir() and (
                    any((d / 'best_model').glob('*model_states.pt'))
                    or any((d / 'latest_model').glob('*model_states.pt'))
                    or (d / 'best_model.bin').exists()
            )
        ]
        checkpoints.sort(key=os.path.getctime)
        logger.info(f"Available checkpoints: {checkpoints}")

        logdir = Path(config.logdir)
        logger.info(f"{logdir / 'logbook.csv'}: {(logdir / 'logbook.csv').exists()}")
        training_history = pd.read_csv(logdir / 'logbook.csv', header=0, index_col=0).dropna(axis=0, how='any')
        logger.info(f"Training history\n{training_history}")

        latest_checkpoint = checkpoints[-1]
        starting_epoch = training_history.index.values[-1]

        overall_step = 0
        best_loss = training_history.loc[starting_epoch, 'loss']
        logger.info(f"Restoring from {latest_checkpoint} epoch {starting_epoch} with loss {best_loss}")

        starting_epoch += 1
        step_logbook = {}
        epoch_logbook = training_history.to_dict(orient='index')
        epoch_left = config.schedulers.epochs - starting_epoch
        logger.info(epoch_logbook)

        logger.info(f"Epochs left {epoch_left}")
        restored = True

        if epoch_left == 0:
            return

    except Exception as e:
        restored = False
        latest_checkpoint = None
        logger.warning(e)
        logger.warning(f"No model found in {config.outdir}")
        best_loss, overall_step, starting_epoch = np.inf, 0, 0
        step_logbook, epoch_logbook = {}, {}

    return restored, latest_checkpoint, best_loss, overall_step, starting_epoch, step_logbook, epoch_logbook


def get_lr_scheduler(opt: torch.optim.Optimizer, steps_per_epoch: int, config: DictConfig, decay: str = 'cosine'):
    if config.fixedlr:
        scheduler = LinearLR(
            opt,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=config.epochs,
        )
        logger.info(f"Training steps: [{steps_per_epoch * config.epochs}]")
    else:
        decay_epochs = config.epochs - (config.warmup + config.cooldown)
        total_steps = config.epochs * steps_per_epoch
        warmup_steps = config.warmup * steps_per_epoch
        cooldown_steps = config.cooldown * steps_per_epoch
        decay_steps = total_steps - (warmup_steps + cooldown_steps)

        logger.info(
            f"Training [epochs: {config.epochs} = total_steps: {total_steps}, "
            f"warmup: {config.warmup} = warmup_steps: {warmup_steps}, "
            f"cooldown: {config.cooldown} = cooldown_steps: {cooldown_steps}, "
            f"decay_epochs: {decay_epochs}," 
            f"decay_steps: {decay_steps}]"
        )

        scheduler, num_epochs = create_scheduler_v2(
            optimizer=opt,
            sched=decay,
            num_epochs=config.epochs,
            warmup_epochs=config.warmup,
            cooldown_epochs=config.cooldown,
            decay_epochs=decay_epochs,
            min_lr=1e-8,
            warmup_lr=1e-8,
        )

    return scheduler


def get_optimizer(params, config: DictConfig, optimizer: str, steps_per_epoch: int, deepspeed_scheduler: bool = False):
    if optimizer == 'adamw':
        opt = FusedAdam(
            params,
            lr=config.lr,
            weight_decay=config.wd,
            betas=config.betas,
            eps=config.eps,
        )
    elif optimizer == 'lamb':
        opt = FusedLamb(
            params,
            lr=config.lr,
            weight_decay=config.wd,
            betas=config.betas,
            eps=config.eps,
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    if deepspeed_scheduler:
        decay_epochs = config.epochs - (config.warmup + config.cooldown)
        total_steps = config.epochs * steps_per_epoch
        warmup_steps = config.warmup * steps_per_epoch
        decay_steps = total_steps - warmup_steps

        logger.info(
            f"Training [epochs: {config.epochs}, steps_per_epoch: {steps_per_epoch} = total_steps: {total_steps}, "
            f"warmup: {config.warmup} = warmup_steps: {warmup_steps}, decay_epochs: {decay_epochs},"
            f"decay_steps: {decay_steps}]"
        )

        scheduler = WarmupCosineLR(
            optimizer=opt,
            total_num_steps=total_steps,
            warmup_num_steps=warmup_steps,
            warmup_min_ratio=0.0,
            cos_min_ratio=0.0001,
            warmup_type='linear',
        )

        return opt, scheduler
    else:
        return opt


def _load_checkpoint(
    latest_checkpoint: Optional[Union[str, Path]],
    model_engine: DeepSpeedEngine,
    opt: Optional[Optimizer],
    config: DictConfig,
    logger: logging.Logger
):
    # we load checkpoints in the following order:
    # 1. from Ray's checkpoint manager
    # 2. from the load_checkpointdir specified in the config
    # 3. from the latest checkpoint found in the outdir (current run dir)

    checkpoint = get_checkpoint()
    if checkpoint is not None:
        checkpointdir = checkpoint.as_directory()
        load_checkpoint(model_engine, opt, config, logger, checkpointdir)

    elif config.load_checkpointdir is not None:
        checkpointdir = Path(config.load_checkpointdir)
        load_checkpoint(model_engine, opt, config, logger, checkpointdir)

    elif latest_checkpoint is not None:
        checkpointdir = Path(latest_checkpoint)
        load_checkpoint(model_engine, opt, config, logger, checkpointdir)

    else:
        warnings.warn("No checkpoint found. Starting from scratch.")


def supervised(config: DictConfig):
    restored, latest_checkpoint, best_loss, overall_step, starting_epoch, step_logbook, epoch_logbook = restore_model(config)

    if config.datasets.split:
        train_dataloader, val_dataloader = ao_dataset.collect_dataset(config)
    else:
        train_dataloader = ao_dataset.collect_dataset(config)
        val_dataloader = None

    print(config.clusters.num_workers)
    steps_per_epoch = int(np.ceil(len(train_dataloader)  / config.clusters.num_workers))

    train_dataloader = raytorch.prepare_data_loader(train_dataloader)

    model = build_dependency_graph_and_instantiate(config.models)

    summarize_model(
        model=model,
        inputs=(config.batch_size, *config.models.input_shape[1:]),
        batch_size=config.batch_size,
        logdir=config.logdir,
    )

    opt = get_optimizer(
        params=model.parameters(),
        config=config.optimizers,
        optimizer=config.optimizers.opt,
        steps_per_epoch=steps_per_epoch
    )

    scheduler = get_lr_scheduler(
        opt=opt,
        config=config.schedulers,
        steps_per_epoch=steps_per_epoch
    )

    model, opt, _, _ = initialize(
        model=model,
        optimizer=opt,
        config=OmegaConf.to_container(config.deepspeed, resolve=True)
    )

    if restored or config.load_checkpointdir is not None:
        _load_checkpoint(
            model_engine=model,
            opt=opt,
            config=config,
            logger=logger,
            latest_checkpoint=latest_checkpoint,
        )

    ray_context = get_context()
    loss_fn = nn.MSELoss(reduction='sum')
    loss_nans = 0

    with torch.autograd.set_detect_anomaly(True, check_nan=False):
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(skip_first=1, warmup=1, active=3, repeat=2, wait=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
        ) if config.profile else nullcontext() as profiler:

            for epoch in range(starting_epoch, config.schedulers.epochs):
                if ray_context.get_world_size() > 1:
                    train_dataloader.sampler.set_epoch(epoch)

                epoch_time = time.time()
                loss = 0.0
                step_times, step_utilization, step_vram = [], [], []

                for step, (inputs, targets) in enumerate(train_dataloader):
                    step_time = time.time()
                    lr_value = opt.param_groups[0]["lr"]

                    outputs = model(inputs)
                    step_loss = loss_fn(outputs, targets)

                    if torch.isnan(step_loss):
                        loss_nans += 1
                        logger.warning(f"Step loss is {step_loss} for {step=} in {epoch=}")

                        if loss_nans > 5:
                            raise Exception(f"Step loss is {step_loss} for {step=} in {epoch=}")

                    model.backward(step_loss)
                    model.step()
                    scheduler.step(epoch)

                    cuda_util = torch.cuda.utilization()
                    cuda_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    loss += step_loss.detach().float()
                    overall_step += 1
                    step_timer = time.time() - step_time

                    step_times.append(step_timer)
                    step_utilization.append(cuda_util)
                    step_vram.append(cuda_vram)

                    step_logbook[overall_step] = {
                        "step_loss": step_loss.detach().float(),
                        "step_lr": lr_value,
                        "step_timer": step_timer,
                        "cuda_vram": cuda_vram,
                        "step_utilization": cuda_util,
                    }

                mem_log = torch.cuda.memory_summary()
                logger.info(mem_log)

                if is_main_process():
                    with (Path(config.logdir) / 'memory.log').open('w') as f:
                        f.write(str(mem_log))

                loss = loss.item() / steps_per_epoch
                step_timer_avg = np.mean(step_times)
                epoch_timer = time.time() - epoch_time
                remaining_epochs = config.schedulers.epochs - (epoch + 1)
                eta = epoch_timer * remaining_epochs / 3600
                cuda_utilization_avg = np.mean(step_utilization)
                cuda_memory_allocated_avg = np.mean(step_vram)
                max_cuda_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)

                logger.info(f"│ training_epoch: {epoch + 1}/{config.schedulers.epochs}")
                logger.info(f"│ epoch_loss: {loss:.4g}")
                logger.info(f"│ epoch_lr: {lr_value:.4g}")
                logger.info(f"│ cuda_utilization: {cuda_utilization_avg:.0f}%")
                logger.info(f"│ cuda_memory_allocated: {cuda_memory_allocated_avg:.4g} GB")
                logger.info(f"│ max_cuda_memory_allocated: {max_cuda_memory_allocated:.4g} GB")
                logger.info(f"│ step_timer: {step_timer_avg * 1000:.0f}ms")
                logger.info(f"│ epoch_timer: {epoch_timer:.0f}s")
                logger.info(f"│ ETA: {eta:.2f}h")

                epoch_logbook[epoch] = {
                    "loss": loss,
                    "lr": lr_value,
                    "cuda_utilization": cuda_utilization_avg,
                    "cuda_memory_allocated": cuda_memory_allocated_avg,
                    "max_cuda_memory_allocated": max_cuda_memory_allocated,
                    "step_timer": step_timer_avg,
                    "epoch_timer": epoch_timer,
                    "eta": eta,
                }

                if loss < best_loss:
                    best_loss = loss
                    model.save_checkpoint(config.checkpointdir, tag="best_model")

                if is_main_process():
                    pd.DataFrame.from_dict(epoch_logbook, orient='index').to_csv(Path(config.logdir) / 'logbook.csv')
                    pd.DataFrame.from_dict(step_logbook, orient='index').to_csv(Path(config.logdir) / 'steplogbook.csv')

                    logger.info(epoch_logbook[epoch])

                    if config.profile:
                        profiler.step()

                # save model weights for latest model every checkpoint_update_interval epochs
                if (epoch + 1) % config.checkpoint_update_interval == 0:
                    model.save_checkpoint(config.checkpointdir, tag="latest_model")

                # update latest model every checkpoint_update_interval epochs and best model
                checkpoint = Checkpoint.from_directory(config.checkpointdir) if is_main_process() else None
                # report must be called on all ranks, else hangs
                report(metrics=epoch_logbook[epoch], checkpoint=checkpoint)


def pixel_reconstruction(config: DictConfig):
    restored, latest_checkpoint, best_loss, overall_step, starting_epoch, step_logbook, epoch_logbook = restore_model(config)

    if config.datasets.split:
        train_dataloader, val_dataloader = ao_dataset.collect_dataset(config)
    else:
        train_dataloader = ao_dataset.collect_dataset(config)
        val_dataloader = None

    print(config.clusters.num_workers)
    steps_per_epoch = int(np.ceil(len(train_dataloader)  / (config['gpu_workers'] * config['workers'])))

    train_dataloader = raytorch.prepare_data_loader(train_dataloader)

    model = build_dependency_graph_and_instantiate(config.models)

    summarize_model(
        model=model,
        inputs=(config.batch_size, *config.models.input_shape[1:]),
        batch_size=config.batch_size,
        logdir=config.logdir,
    )

    opt = get_optimizer(
        params=model.parameters(),
        config=config.optimizers,
        optimizer=config.optimizers.opt,
        steps_per_epoch=steps_per_epoch
    )

    scheduler = get_lr_scheduler(
        opt=opt,
        config=config.schedulers,
        steps_per_epoch=steps_per_epoch
    )

    model, opt, _, _ = initialize(
        model=model,
        optimizer=opt,
        config=OmegaConf.to_container(config.deepspeed, resolve=True)
    )

    if restored or config.load_checkpointdir is not None:
        _load_checkpoint(
            model_engine=model,
            opt=opt,
            config=config,
            logger=logger,
            latest_checkpoint=latest_checkpoint,
        )

    ray_context = get_context()
    loss_nans = 0
    with torch.autograd.set_detect_anomaly(True, check_nan=False):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(skip_first=1, warmup=1, active=3, repeat=2, wait=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ) if config['profile'] else nullcontext() as profiler:

            for epoch in range(starting_epoch, config['epochs']):

                if ray_context.get_world_size() > 1:
                    train_dataloader.sampler.set_epoch(epoch)

                epoch_time = time.time()
                loss = 0.
                step_times, step_utilization, step_vram = [], [], []

                for step, (inputs, targets) in enumerate(train_dataloader):
                    step_time = time.time()
                    lr_value = opt.param_groups[0]["lr"]

                    step_loss = model(inputs)

                    if torch.isnan(step_loss):
                        loss_nans += 1
                        logger.warning(f"Step loss is {step_loss} for {step=} in {epoch=}")

                        if loss_nans > 5:
                            raise Exception(f"Step loss is {step_loss} for {step=} in {epoch=}")

                    model.backward(step_loss)
                    model.step()
                    scheduler.step(epoch)

                    cuda_util = torch.cuda.utilization()
                    cuda_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    loss += step_loss.detach().float()
                    overall_step += 1
                    step_timer = time.time() - step_time

                    step_times.append(step_timer)
                    step_utilization.append(cuda_util)
                    step_vram.append(cuda_vram)

                    step_logbook[overall_step] = {
                        "step_loss": step_loss.detach().float(),
                        "step_lr": lr_value,
                        "step_timer": step_timer,
                        "cuda_vram": cuda_vram,
                        "step_utilization": cuda_util,
                    }

                mem_log = torch.cuda.memory_summary()
                logger.info(mem_log)

                if is_main_process():
                    with (Path(config.logdir) / 'memory.log').open('w') as f:
                        f.write(str(mem_log))

                loss = loss.item() / steps_per_epoch
                step_timer_avg = np.mean(step_times)
                epoch_timer = time.time() - epoch_time
                remaining_epochs = config.schedulers.epochs - (epoch + 1)
                eta = epoch_timer * remaining_epochs / 3600
                cuda_utilization_avg = np.mean(step_utilization)
                cuda_memory_allocated_avg = np.mean(step_vram)
                max_cuda_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)

                logger.info(f"│ training_epoch: {epoch + 1}/{config.schedulers.epochs}")
                logger.info(f"│ epoch_loss: {loss:.4g}")
                logger.info(f"│ epoch_lr: {lr_value:.4g}")
                logger.info(f"│ cuda_utilization: {cuda_utilization_avg:.0f}%")
                logger.info(f"│ cuda_memory_allocated: {cuda_memory_allocated_avg:.4g} GB")
                logger.info(f"│ max_cuda_memory_allocated: {max_cuda_memory_allocated:.4g} GB")
                logger.info(f"│ step_timer: {step_timer_avg * 1000:.0f}ms")
                logger.info(f"│ epoch_timer: {epoch_timer:.0f}s")
                logger.info(f"│ ETA: {eta:.2f}h")

                epoch_logbook[epoch] = {
                    "loss": loss,
                    "lr": lr_value,
                    "cuda_utilization": cuda_utilization_avg,
                    "cuda_memory_allocated": cuda_memory_allocated_avg,
                    "max_cuda_memory_allocated": max_cuda_memory_allocated,
                    "step_timer": step_timer_avg,
                    "epoch_timer": epoch_timer,
                    "eta": eta,
                }

                if loss < best_loss:
                    best_loss = loss
                    model.save_checkpoint(config.checkpointdir, tag="best_model")

                if is_main_process():
                    pd.DataFrame.from_dict(epoch_logbook, orient='index').to_csv(Path(config.logdir) / 'logbook.csv')
                    pd.DataFrame.from_dict(step_logbook, orient='index').to_csv(Path(config.logdir) / 'steplogbook.csv')

                    logger.info(epoch_logbook[epoch])

                    if config.profile:
                        profiler.step()

                # save model weights for latest model every checkpoint_update_interval epochs
                if (epoch + 1) % config.checkpoint_update_interval == 0:
                    model.save_checkpoint(config.checkpointdir, tag="latest_model")

                # update latest model every checkpoint_update_interval epochs and best model
                checkpoint = Checkpoint.from_directory(config.checkpointdir) if is_main_process() else None
                # report must be called on all ranks, else hangs
                report(metrics=epoch_logbook[epoch], checkpoint=checkpoint)


#TODO hydra config integration
def joint_embedding_prediction(config: DictConfig):
    restored, latest_checkpoint, best_loss, overall_step, starting_epoch, step_logbook, epoch_logbook = restore_model(config)

    collate_fn = masking.MaskCollator(
        input_shape=config['inputs'],
        lateral_patch_size=config['patches'] if type(config['patches']) == int else config['patches'][0],
        axial_patch_size=1,
        lateral_range=(0.2, 0.8),
        axial_range=(1.0, 1.0),
        num_blocks=8,
        patchify_scheme='blocks',
    )

    train_dataloader = ao_dataset.collect_dataset(
        config['dataset'],
        metadata=False,
        modes=config['pmodes'],
        distribution=config['distribution'],
        embedding=config['embedding'],
        samplelimit=config['samplelimit'],
        max_amplitude=config['max_amplitude'],
        photons_range=(config['min_photons'], config['max_photons']),
        cpu_workers=config['cpu_workers'],
        gpu_workers=config['gpu_workers'],
        model_input_shape=config['inputs'],
        batch_size=config['batch_size'],
        dtype=_DTYPES[config['amp']],
        # collate_fn=collate_fn
    )
    steps_per_epoch = int(np.ceil(len(train_dataloader)  / (config['gpu_workers'] * config['workers'])))

    train_dataloader = raytorch.prepare_data_loader(train_dataloader)

    if config['network'].startswith('jepa'):
        model = JEPA(
            model_template=config['network'],
            input_shape=config['inputs'],
            embed_dim=config['hidden_size'],
            lateral_patch_size=config['patches'] if type(config['patches']) == int else config['patches'][0],
            axial_patch_size=1,
            num_heads=config['heads'] if type(config['heads']) == int else config['heads'][0],
            depth=config['repeats'] if type(config['repeats']) == int else config['repeats'][0],
            proj_drop_rate=config['dropout'],
            fixed_dropout_depth=config['fixed_dropout_depth'],
        )
        block = Encoder
    else:
        raise Exception(f'Network "{config["network"]}" is unknown.')

    summarize_model(
        model=model,
        inputs=config['inputs'],
        batch_size=config['batch_size'],
        logdir=config['logdir'],
    )

    opt = get_optimizer(
        params=model.parameters(),
        config=config,
        optimizer=config['opt'],
        steps_per_epoch=steps_per_epoch
    )

    scheduler = get_lr_scheduler(
        opt=opt,
        config=config,
        steps_per_epoch=steps_per_epoch
    )

    # linearly increase lr from ema[0] to ema[1]
    total_steps = config['epochs'] * steps_per_epoch
    ema_scheduler = (
        config['ema'][0] + i * (config['ema'][1]-config['ema'][0]) / total_steps
        for i in range(total_steps+1)
    )

    model, opt, _, _ = initialize(
        model=model,
        optimizer=opt,
        config=config.deepspeed,
    )

    if config['finetune'] is not None:
        logger.info(f"Finetuning pretrained model @ {config['finetune']}")
        model_state = torch.load(config['finetune'].glob("*model.bin"))
        model.load_state_dict(model_state)

        optimizer_state = torch.load(config['finetune'].glob("*optimizer.bin"))
        opt.load_state_dict(optimizer_state)

    elif restored:
        checkpoint = get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpointdir:
                logger.info(f"Loading pretrained model @ {latest_checkpoint} -> {checkpointdir}")

                model_state = torch.load(config['checkpointdir'] / f"best_model.bin")
                model.load_state_dict(model_state)

                optimizer_state = torch.load(config['checkpointdir'] / f"best_optimizer.bin")
                opt.load_state_dict(optimizer_state)

    es = EarlyStoppingCallback(min_delta=1e-4, patience=10)
    ray_context = get_context()
    loss_nans = 0

    with torch.autograd.set_detect_anomaly(True, check_nan=False):

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(skip_first=1, warmup=1, active=3, repeat=2, wait=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ) if config['profile'] else nullcontext() as profiler:

            for epoch in range(starting_epoch, config['epochs']):

                if ray_context.get_world_size() > 1:
                    train_dataloader.sampler.set_epoch(epoch)

                epoch_time = time.time()
                loss = 0.

                step_times, step_utilization, step_vram = [], [], []
                for step, batch in enumerate(train_dataloader):
                    inputs, zernikes = batch

                    step_time = time.time()
                    lr = opt.param_groups[0]["lr"]

                    step_loss = model(inputs)

                    if torch.isnan(step_loss):
                        loss_nans += 1
                        logger.warning(f"Step loss is {step_loss} for {step=} in {epoch=}")

                        if loss_nans > 5:
                            raise Exception(f"Step loss is {step_loss} for {step=} in {epoch=}")

                    model.backward(step_loss)
                    model.step()
                    scheduler.step(epoch)

                    model.ema_update(beta=next(ema_scheduler))

                    cuda_util = torch.cuda.utilization()
                    cuda_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

                    loss += step_loss.detach().float()

                    overall_step += 1
                    step_timer = time.time() - step_time

                    step_times.append(step_timer)
                    step_utilization.append(cuda_util)
                    step_vram.append(cuda_vram)

                    step_logbook[overall_step] = {
                        "step_loss": step_loss,
                        "step_lr": lr,
                        "step_timer": step_timer,
                        "cuda_vram": cuda_vram,
                        "step_utilization": cuda_util,
                    }

                mem_log = torch.cuda.memory_summary()
                logger.info(mem_log)
                with Path(config['logdir'] / 'memory.log').open('w') as f:
                    f.write(str(mem_log))

                loss = loss.item() / steps_per_epoch
                step_timer = np.mean(step_times)
                epoch_timer = time.time() - epoch_time
                remaining_epochs = config['epochs'] - (epoch + 1)
                eta = epoch_timer * remaining_epochs / 3600
                cuda_utilization = np.mean(step_utilization)
                cuda_memory_allocated = np.mean(step_vram)
                max_cuda_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)

                logger.info(f"│ training_epoch:                 \t {epoch+1}/{config['epochs']}")
                logger.info(f"│ epoch_loss:                     \t {loss:.4g}")
                logger.info(f"│ epoch_lr:                       \t {lr:.4g}")
                logger.info(f"│ cuda_utilization:               \t {cuda_utilization:.0f}%")
                logger.info(f"│ cuda_memory_allocated:          \t {cuda_memory_allocated:.4g} GB")
                logger.info(f"│ max_cuda_memory_allocated:      \t {max_cuda_memory_allocated:.4g} GB")
                logger.info(f"│ step_timer:                     \t {step_timer * 1000:.0f}ms")
                logger.info(f"│ epoch_timer:                    \t {epoch_timer:.0f}s")
                logger.info(f"│ ETA:                            \t {eta:.2f}h")

                epoch_logbook[epoch] = {
                    "loss": loss,
                    "epoch_lr": lr,
                    "cuda_utilization": cuda_utilization,
                    "cuda_memory_allocated": cuda_memory_allocated,
                    "max_cuda_memory_allocated": max_cuda_memory_allocated,
                    "step_timer": step_timer,
                    "epoch_timer": epoch_timer,
                }
                df = pd.DataFrame.from_dict(epoch_logbook, orient='index')
                df.to_csv(config['logdir'] / 'logbook.csv')

                df = pd.DataFrame.from_dict(step_logbook, orient='index')
                df.to_csv(config['logdir'] / 'steplogbook.csv')

                with config['outdir'] / 'checkpoints' as checkpointdir:
                    if loss < best_loss:
                        best_loss = loss
                        torch.save(model.state_dict(), config['checkpointdir'] / f"best_model.bin")
                        torch.save(opt.state_dict(), config['checkpointdir'] / f"best_optimizer.bin")

                    checkpoint = Checkpoint.from_directory(checkpointdir)
                    report(metrics=epoch_logbook[epoch], checkpoint=checkpoint)

                if is_main_process():
                    logger.info(epoch_logbook[epoch])

                if config['profile']:
                    profiler.step()

            with config['outdir'] / 'checkpoints' as checkpointdir:
                torch.save(model.state_dict(), config['checkpointdir'] / f"last_model.bin")
                torch.save(opt.state_dict(), config['checkpointdir'] / f"last_optimizer.bin")

                checkpoint = Checkpoint.from_directory(checkpointdir)
                report(metrics=epoch_logbook[epoch], checkpoint=checkpoint)
