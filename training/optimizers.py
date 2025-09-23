import logging

from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.lamb import FusedLamb
from deepspeed.runtime.lr_schedules import WarmupCosineLR

from omegaconf import DictConfig

logger = logging.getLogger("ray")
logger.setLevel(logging.INFO)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


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