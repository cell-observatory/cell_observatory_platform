"""
Partly adapted from:
https://github.com/pytorch/torchtitan/torchtitan/components/lr_scheduler.py
https://github.com/facebookresearch/dinov3/dinov3/train/cosine_lr_scheduler.py
"""

import math
import copy
import logging
import functools
from typing import Dict, List, Any, Callable, Iterator

import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR

from timm.scheduler import create_scheduler_v2

from omegaconf import DictConfig, OmegaConf

from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from cell_observatory_platform.training.optimizers import OptimizersContainer

logger = logging.getLogger("ray")
logger.setLevel(logging.INFO)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


def get_param_groups(config, model: nn.Module) -> List[Dict]:
    """Build optimizer parameter groups.

    Delegates to ``model.get_param_groups(weight_decay, **kw)`` when the model
    implements that method.  Otherwise falls back to a universal decay /
    no-decay split via ``_default_param_groups``.
    """
    weight_decay = float(config.optimizers.wd)

    if hasattr(model, "get_param_groups"):
        extra_cfg = getattr(config.optimizers, "get_param_groups_extra", None)
        extra = OmegaConf.to_container(extra_cfg, resolve=True) if extra_cfg else {}
        if not isinstance(extra, dict):
            extra = {}
        return model.get_param_groups(weight_decay=weight_decay, **extra)

    logger.warning(
        "%s does not implement get_param_groups(); "
        "falling back to default decay/no-decay split.",
        model.__class__.__name__,
    )
    return _default_param_groups(model, weight_decay)


_NO_WD_KEYWORDS = ("bias", "pos_embedding", "cls_token", "token_param", "level_embed")


def _default_param_groups(model: nn.Module, weight_decay: float) -> List[Dict]:
    """Universal fallback: separate decay vs no-decay params.

    No-decay: all 1-d params (biases, norms) and known special params.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or any(kw in name for kw in _NO_WD_KEYWORDS):
            no_decay.append(p)
        else:
            decay.append(p)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def get_schedulers(
    opt: torch.optim.Optimizer,
    steps_per_epoch: int,
    config: DictConfig,
    decay: str = 'cosine'
):
    if config.schedulers.type == "fixedlr":
        scheduler = LinearLR(
            opt,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=config.schedulers.epochs,
        )
        logger.info(f"Training steps: [{steps_per_epoch * config.schedulers.epochs}]")

    elif config.schedulers.type == "warmup_stable_decay":
        scheduler = WarmupStableDecaySchedule(
            optimizer=opt,
            warmup_steps=config.schedulers.warmup * steps_per_epoch,
            anneal_steps=config.schedulers.cooldown * steps_per_epoch,
            T_max=config.schedulers.epochs * steps_per_epoch,
            start_lr=config.schedulers.warmup_min_ratio * config.optimizers.lr,
            ref_lr=config.optimizers.lr,
            final_lr=config.schedulers.final_lr_ratio * config.optimizers.lr,
            update_type=config.schedulers.update_type,
        )

    elif config.schedulers.type == "cosine":
        decay_epochs = config.schedulers.epochs - (config.schedulers.warmup + config.schedulers.cooldown)
        total_steps = config.schedulers.epochs * steps_per_epoch
        warmup_steps = config.schedulers.warmup * steps_per_epoch
        cooldown_steps = config.schedulers.cooldown * steps_per_epoch
        decay_steps = total_steps - (warmup_steps + cooldown_steps)

        cos_min_lr = config.schedulers.cos_min_ratio * config.optimizers.lr
        warmup_min_lr = config.schedulers.warmup_min_ratio * config.optimizers.lr

        logger.info("-" * 80)
        logger.info(
            f"Epochs: {config.schedulers.epochs} = "
            f"[{config.schedulers.warmup} warmup + {decay_epochs} decay + {config.schedulers.cooldown} cooldown]\n"
            f"Steps: {total_steps} = "
            f"[{warmup_steps} warmup + {decay_steps} decay + {cooldown_steps} cooldown]\n"
            f"LR: {config.optimizers.lr} = [{warmup_min_lr=},  {cos_min_lr=}]"
        )
        logger.info("-" * 80)

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
        scheduler.update_type = config.schedulers.update_type

    else:
        raise NotImplementedError(f"Unknown scheduler: {config.schedulers.type}")

    if config.schedulers.wd_scheduler.enabled:
        wd_scheduler = CosineWeightDecaySchedule(
            optimizer=opt,
            ref_wd=config.schedulers.wd_scheduler.ref_wd,
            T_max=config.schedulers.epochs * steps_per_epoch,
            final_wd=config.schedulers.wd_scheduler.final_wd
        )

        _hook_is_registered = False
        for hook in list(config.hooks.hooks_list):
            if hook._target_.endswith("WeightDecayScheduleHook"):
                _hook_is_registered = True
                break
        if not _hook_is_registered:
            raise ValueError("WeightDecayScheduleHook not found in "
                             "config.hooks.hooks_list but wd_scheduler.enabled is True")

    else:
        wd_scheduler = None

    return scheduler, wd_scheduler


# from: https://github.com/facebookresearch/vjepa2/blob/main/src/utils/schedulers.py
class WarmupStableDecaySchedule(object):
    def __init__(self, 
                 optimizer, 
                 warmup_steps, 
                 anneal_steps, 
                 T_max, 
                 start_lr, 
                 ref_lr, 
                 final_lr=0.0,
                 update_type="step"
):
        self._step = 0.0
        self.optimizer = optimizer
        
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        
        self.anneal_steps = anneal_steps
        self.warmup_steps = warmup_steps
        
        self.T_max = T_max - warmup_steps - anneal_steps
        self.update_type = update_type

    def step(self, epoch):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        
        elif self._step < self.T_max + self.warmup_steps:
            new_lr = self.ref_lr
        
        else:
            _step = self._step - (self.T_max + self.warmup_steps)
            progress = float(_step) / float(max(1, self.anneal_steps))
            new_lr = self.ref_lr + progress * (self.final_lr - self.ref_lr)

        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
            if "lr_scale" in group:
                group["lr"] *= group["lr_scale"]

        return new_lr
    

# from: https://github.com/facebookresearch/vjepa2/blob/main/src/utils/schedulers.py
class CosineWeightDecaySchedule(object):
    def __init__(self, 
                 optimizer, 
                 ref_wd, 
                 T_max, 
                 final_wd=0.0
):
        self._step = 0.0
        self.optimizer = optimizer

        self.ref_wd = ref_wd
        self.final_wd = final_wd

        self.T_max = T_max

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ("WD_exclude" not in group) or not group["WD_exclude"]:
                group["weight_decay"] = new_wd
        
        return new_wd


# --- TorchTitan Support --- #


class WDSchedulersContainer(Stateful):
    """Container for multiple weight-decay schedulers.

    This class wraps one `CosineWeightDecaySchedule` per optimizer into a single
    object, mirroring `LRSchedulersContainer` for LR.

    **Limitations**
    We assume all weight-decay schedulers are identical (same config). We only
    store the state of the first one and broadcast it to the others on load.
    """

    schedulers: list[CosineWeightDecaySchedule]

    def __init__(
        self,
        optimizers: OptimizersContainer,
        ref_wd: float,
        final_wd: float,
        training_steps: int,
    ) -> None:
        assert len(optimizers) > 0, (
            "Must have at least one optimizer to create WDSchedulersContainer"
        )
        if training_steps <= 0:
            raise ValueError(
                f"training_steps must be positive, got {training_steps}"
            )

        self.schedulers = [
            CosineWeightDecaySchedule(
                optimizer=optimizer,
                ref_wd=ref_wd,
                T_max=training_steps,
                final_wd=final_wd,
            )
            for optimizer in optimizers
        ]

    def __iter__(self) -> Iterator[CosineWeightDecaySchedule]:
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self) -> None:
        """Advance all underlying weight-decay schedulers by one step."""
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        """Save state from the first scheduler and reuse for all.

        The only quantity we really care about is the current step counter.
        """
        if not self.schedulers:
            return {}
        return {"_step": copy.deepcopy(self.schedulers[0]._step)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the same state into all schedulers."""
        if not self.schedulers:
            return
        step = float(state_dict.get("_step", 0.0))
        for scheduler in self.schedulers:
            scheduler._step = step


def build_wd_schedulers(
    optimizers: OptimizersContainer,
    wd_scheduler_config: dict,
    training_steps: int,
) -> WDSchedulersContainer | None:
    """Create a WDSchedulersContainer for the given optimizers and config.

    Args:
        optimizers (OptimizersContainer): The corresponding optimizers whose
            `param_groups` will get their `weight_decay` updated.
        wd_scheduler_config (dict): Weight-decay scheduler config, expected keys:
            - "enabled": bool
            - "ref_wd": float
            - "final_wd": float
            - "type": str (currently only "cosine" is supported)
        training_steps (int): Total number of training steps (T_max for the schedule).

    Returns:
        WDSchedulersContainer | None: The constructed container, or None if
            the scheduler is disabled.
    """
    if not wd_scheduler_config.get("enabled", False):
        return None

    sched_type = wd_scheduler_config.get("type", "cosine")
    if sched_type != "cosine":
        raise NotImplementedError(
            f"Unknown weight-decay scheduler type: {sched_type}"
        )

    ref_wd = float(wd_scheduler_config["ref_wd"])
    final_wd = float(wd_scheduler_config.get("final_wd", 0.0))

    if training_steps <= 0:
        raise ValueError(
            f"training_steps must be positive, got {training_steps}"
        )

    return WDSchedulersContainer(
        optimizers=optimizers,
        ref_wd=ref_wd,
        final_wd=final_wd,
        training_steps=training_steps,
    )


class LRSchedulersContainer(Stateful):
    """Container for multiple learning rate schedulers.

    This class is used to wrap multiple LRSchedulers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.lr_scheduler.LRScheduler``. The design concept is the same as
    ``OptimizersContainer``. This class currently only supports ``LambdaLR``.

    **Limitations**
    This class assumes all the lr schedulers are the same. There is no easy way to support
    resharding for multiple different LRSchedulers because LRScheduler.state_dict() is not
    resharding friendly. Therefore, the limitation is used to allow TorchTitan to support
    lr scheduler resharding.

    Args:
        optimizers (OptimizersContainer): The corresponding optimizers for the lr_schedulers.
    """

    schedulers: list[LRScheduler]

    def __init__(self, optimizers: OptimizersContainer, lr_lambda: Callable) -> None:
        assert (
            len(optimizers) > 0
        ), "Must have at least one optimizer to create LRScheduler"

        self.schedulers = [LambdaLR(optimizer, lr_lambda) for optimizer in optimizers]

    def __iter__(self) -> Iterator[LRScheduler]:
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self) -> None:
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        # While there may be multiple schedulers, we only save the first one because
        # the state_dict is the same for all. See the limitations section in the
        # docstring.
        return self.schedulers[0].state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # Load the same state_dict for all schedulers. The key value we're concerned
        # within ``LRScheduler.state_dict()`` is ``last_epoch``, which is an integer
        # that is immutable. As long as ``training.steps`` and ``lr_scheduler.warmup_steps``
        # in ``job_config`` remain unchanged when resuming from a checkpoint, this
        # approach is safe. We call ``copy()`` here to ensure extra safety.
        for scheduler in self.schedulers:
            scheduler.load_state_dict(copy.deepcopy(state_dict))


def linear_warmup_stable_decay(
    current_step: int,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    lr_decay_type: str,
    min_lr_factor: float,
):
    """
    Computes linear warmup followed by stable learning rate for a while,
    then some type of decay.

    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor `curr_adjustment` ranging from 1 to 0
    to adjust the learning rate to create the desired schedule.

    We offer three types of learning rate decay schedules:
    1. `linear`: decays linearly from 1 to 0 over the decay period.
    2. `sqrt`: decays as 1 minus the square root of the decay progress.
    3. `cosine`: follows a cosine curve, decaying according to the values of the half-period of the cosine function.

    If `min_lr_factor` is specified, the decay range is scaled from 1 to `min_lr_factor`
    to ensure the learning rate does not drop below this minimum value.
    """
    warmup_stable_steps = warmup_steps + stable_steps
    if current_step < warmup_steps:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        assert (
            warmup_steps != 0
        ), "warmup_steps must not be zero to reach this branch"
        curr_adjustment = float(current_step / warmup_steps)
    elif current_step < warmup_stable_steps:
        curr_adjustment = 1.0
    else:
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        assert decay_steps != 0, "decay_steps must not be zero to reach this branch"
        progress = float(current_step - warmup_stable_steps) / decay_steps

        if lr_decay_type == "linear":
            curr_adjustment = 1 - progress
        elif lr_decay_type == "sqrt":
            curr_adjustment = 1 - math.sqrt(progress)
        elif lr_decay_type == "cosine":
            curr_adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
        curr_adjustment = min_lr_factor + (1 - min_lr_factor) * curr_adjustment
    return curr_adjustment


def warmup_decay_cooldown_min(
    current_step: int,
    warmup_steps: int,
    decay_steps: int,
    cooldown_steps: int,
    lr_decay_type: str,
    warmup_min_factor: float,
    min_lr_factor: float,
):
    """
    Multiplicative LR factor schedule (DeepSpeed/timm-like):

      warmup:   warmup_min_factor -> 1.0
      decay:    1.0 -> min_lr_factor    (linear / sqrt / cosine)
      cooldown: hold min_lr_factor
    """
    # warmup
    if current_step < warmup_steps:
        t = float(current_step + 1) / float(warmup_steps)
        return warmup_min_factor + t * (1.0 - warmup_min_factor)

    # decay
    decay_start = warmup_steps
    decay_end = warmup_steps + decay_steps
    if current_step < decay_end:
        if decay_steps == 0:
            return min_lr_factor
        t = float(current_step - decay_start + 1) / float(decay_steps)
        if lr_decay_type == "linear":
            f = 1.0 - t
        elif lr_decay_type == "sqrt":
            f = 1.0 - math.sqrt(t)
        elif lr_decay_type == "cosine":
            f = 0.5 * (1.0 + math.cos(math.pi * t))
        else:
            raise ValueError(f"Unknown lr_decay_type: {lr_decay_type}")
        return min_lr_factor + (1.0 - min_lr_factor) * f

    return min_lr_factor


def build_lr_schedulers(
    optimizers,
    lr_scheduler_config: dict,
    training_steps: int,
    steps_per_epoch: int,
) -> LRSchedulersContainer:
    """
    Supports two schedules via lr_scheduler_config["schedule"]:
      1) "linear_warmup_stable_decay" (existing TorchTitan behavior)
      2) "warmup_decay_cooldown_min"  (DeepSpeed/timm-like ratios + cooldown hold)
    """
    schedule = lr_scheduler_config.get("schedule")

    warmup_steps = int(lr_scheduler_config["warmup"] * steps_per_epoch)
    cooldown_steps = int(lr_scheduler_config.get("cooldown", 0) * steps_per_epoch)

    # clamp warmup to training_steps
    if warmup_steps > training_steps:
        logger.warning(
            f"Warmup steps ({warmup_steps}) exceed total training steps ({training_steps}). "
            f"Adjusting warmup steps to {training_steps}."
        )
        warmup_steps = training_steps

    # decay_steps
    if lr_scheduler_config.get("decay_epochs", None) is not None:
        decay_steps = int(round(lr_scheduler_config["decay_epochs"] * steps_per_epoch))
    else:
        if schedule == "warmup_decay_cooldown_min":
            decay_steps = training_steps - warmup_steps - cooldown_steps
        else:
            decay_steps = training_steps - warmup_steps

    # clamp the trio to fit into training_steps
    warmup_steps = min(warmup_steps, training_steps)
    remaining = training_steps - warmup_steps
    decay_steps = max(0, min(decay_steps, remaining))
    remaining -= decay_steps
    cooldown_steps = max(0, min(cooldown_steps, remaining))

    lr_decay_type = lr_scheduler_config["decay_type"]
    min_lr_factor = float(lr_scheduler_config["cos_min_ratio"])

    if schedule == "linear_warmup_stable_decay":
        # Add a virtual last step to prevent the learning rate from dropping to 0
        stable_steps = training_steps + 1 - warmup_steps - decay_steps
        lr_lambda = functools.partial(
            linear_warmup_stable_decay,
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
            lr_decay_type=lr_decay_type,
            min_lr_factor=min_lr_factor,
        )

    elif schedule == "warmup_decay_cooldown_min":
        warmup_min_factor = float(lr_scheduler_config.get("warmup_min_ratio"))
        lr_lambda = functools.partial(
            warmup_decay_cooldown_min,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            cooldown_steps=cooldown_steps,
            lr_decay_type=lr_decay_type,
            warmup_min_factor=warmup_min_factor,
            min_lr_factor=min_lr_factor,
        )

    else:
        raise ValueError(f"Unknown lr scheduler schedule: {schedule}")

    return LRSchedulersContainer(optimizers, lr_lambda)


class CosineScheduler(object):
    def __init__(
        self,
        base_value,
        final_value,
        total_iters,
        warmup_iters=0,
        start_warmup_value=0,
        freeze_iters=0,
        trunc_extra=0.0,
    ):
        super().__init__()

        self.total_iters = total_iters
        self.final_value = np.float64(final_value)

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        if trunc_extra == 0.0:
            iters = np.arange(total_iters - warmup_iters - freeze_iters)
            schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        else:
            cosine_steps = total_iters - warmup_iters - freeze_iters
            iters = np.linspace(0, np.pi, int((1 + trunc_extra) * cosine_steps))[:cosine_steps]
            schedule = np.cos(iters)
            schedule = (schedule + 1) / 2
            schedule = (schedule - schedule[-1]) / (1 - schedule[-1])
            schedule = schedule * (base_value - final_value) + final_value

        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule), dtype=np.float64)

        assert len(self.schedule) == self.total_iters, "Schedule length does not match total iterations"

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def linear_warmup_cosine_decay(
    start: float,
    peak: float,
    end: float,
    warmup_iterations: int,
    total_iterations: int,
    cosine_iterations: int | None = None,
) -> np.ndarray:
    """
    Create a learning rate schedule with linear warmup, a cosine, and an optional constant part in the end.

    Args:
        start (float): Initial learning rate.
        peak (float): Learning rate after linear warmup.
        end (float): Final learning rate after cosine.
        warmup_iterations (int): Number of iterations for linear warmup.
        total_iterations (int): Total number of iterations for the schedule.
        cosine_iterations (int | None): Number of iterations for cosine.
            If None, cosine part will be over remaining iterations after warmup.
    Returns:
        np.ndarray: Learning rate schedule as a numpy array.
    """
    linear = np.linspace(start, peak, warmup_iterations, endpoint=False)

    if cosine_iterations is None:
        cosine_iterations = total_iterations - warmup_iterations
    cosine = np.cos(np.linspace(0, np.pi, cosine_iterations))
    cosine = (cosine + 1) / 2
    cosine = (peak - end) * cosine + end

    remaining_iterations = total_iterations - cosine_iterations - warmup_iterations
    assert remaining_iterations >= 0
    
    constant = np.full((remaining_iterations,), fill_value=end)
    return np.concatenate([linear, cosine, constant])