"""
Partly adapted from:
https://github.com/pytorch/torchtitan/torchtitan/components/lr_scheduler.py
"""

import re
import math
import json
import copy
import logging
import functools
from typing import Dict, List, Any, Callable, Iterator

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR

from timm.scheduler import create_scheduler_v2

from omegaconf import DictConfig

from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from cell_observatory_platform.training.optimizers import OptimizersContainer

logger = logging.getLogger("ray")
logger.setLevel(logging.INFO)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


def get_param_groups(config, model: nn.Module) -> List[Dict]:
    if not getattr(config.optimizers, "param_group_split_mode", False):
        return model.parameters()

    # adapted from: https://github.com/facebookresearch/mae/blob/main/util/lr_decay.py
    # FIXME: remove unused config parameters and simplify
    if config.optimizers.param_group_split_mode == "mae":
        enc_layer_decay = float(getattr(config.optimizers, "layer_decay"))
        dec_layer_decay = float(getattr(config.optimizers, "decoder_layer_decay"))
        weight_decay = float(getattr(config.optimizers, "wd"))
        no_wd_list = tuple(getattr(config.optimizers, "no_weight_decay_list"))

        ALWAYS_NO_WD_SUFFIX = ("pos_embedding", "cls_token", "token_param")

        enc_L = model.masked_encoder.get_num_layers()
        dec_L = model.masked_decoder.get_num_layers()

        enc_scales = [enc_layer_decay ** (enc_L - i) for i in range(enc_L + 1)]
        dec_scales = [dec_layer_decay ** (dec_L - i) for i in range(dec_L + 1)]

        def _layer_id_from_name(suffix: str, L: int) -> int:
            if suffix.startswith(("patch_embedding", "pos_embedding", "cls_token",
                                  "token_param", "patch_projection")):
                return 0
            m = re.search(r"transformer_blocks\.(\d+)", suffix)
            if m:
                return int(m.group(1)) + 1
            if "output_projection" in suffix or suffix.startswith("norm"):
                return L
            return L

        def _is_no_wd(name: str, p) -> bool:
            if p.ndim == 1:
                return True
            for pat in no_wd_list:
                if name == pat or name.endswith(pat):
                    return True
            for pat in ALWAYS_NO_WD_SUFFIX:
                if pat in name:
                    return True
            return False

        param_groups, param_group_names = {}, {}

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue

            if n.startswith("masked_encoder."):
                side, suffix, L, scales = "enc", n[len("masked_encoder."):], enc_L, enc_scales
            elif n.startswith("masked_decoder."):
                side, suffix, L, scales = "dec", n[len("masked_decoder."):], dec_L, dec_scales
            else:
                raise ValueError(f"Parameter {n} not under masked_encoder/decoder")

            decay_tag = "no_decay" if _is_no_wd(n, p) else "decay"
            wd = 0.0 if decay_tag == "no_decay" else weight_decay

            lid = _layer_id_from_name(suffix, L)
            lr_scale = scales[lid]

            gname = f"{side}_layer_{lid}_{decay_tag}"
            if gname not in param_groups:
                param_groups[gname] = {"lr_scale": lr_scale, "weight_decay": wd, "params": []}
                param_group_names[gname] = {"lr_scale": lr_scale, "weight_decay": wd, "params": []}

            param_groups[gname]["params"].append(p)
            param_group_names[gname]["params"].append(n)

        print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

        return list(param_groups.values())

    # from: https://github.com/facebookresearch/ijepa/main/src/helper.py
    elif config.optimizers.param_group_split_mode == "vjepa":
        param_groups = [
            {
                'params': (p for n, p in model.input_encoder.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in model.target_predictor.named_parameters()
                        if ('bias' not in n) and (len(p.shape) != 1))
            }, {
                'params': (p for n, p in model.input_encoder.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }, {
                'params': (p for n, p in model.target_predictor.named_parameters()
                        if ('bias' in n) or (len(p.shape) == 1)),
                'WD_exclude': True,
                'weight_decay': 0
            }
        ]
        return param_groups

    # from: https://github.com/facebookresearch/vjepa2/app/vjepa/utils.py#L228
    elif config.optimizers.param_group_split_mode == "vjepa2":
        zero_init_bias_wd = config.optimizers.zero_init_bias_wd

        param_groups = [
            {"params": (p for n, p in model.input_encoder.named_parameters() \
                        if ("bias" not in n) and (len(p.shape) != 1))},
            {"params": (p for n, p in model.target_predictor.named_parameters() \
                        if ("bias" not in n) and (len(p.shape) != 1))},
            {
                "params": (p for n, p in model.input_encoder.named_parameters() \
                            if ("bias" in n) or (len(p.shape) == 1)),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
            },
            {
                "params": (p for n, p in model.target_predictor.named_parameters() \
                           if ("bias" in n) or (len(p.shape) == 1)),
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
            },
        ]
        return param_groups

    else:
        raise NotImplementedError(
            f"Unknown param_group_split_mode: \
                                  {config.optimizers.param_group_split_mode}"
        )


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