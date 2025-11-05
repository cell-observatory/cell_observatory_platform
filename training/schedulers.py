import re
import math
import json
import logging
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR

from timm.scheduler import create_scheduler_v2

from omegaconf import DictConfig

logger = logging.getLogger("ray")
logger.setLevel(logging.INFO)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


def get_param_groups(
    config,
    model: nn.Module
) -> List[Dict]:
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
        raise NotImplementedError(f"Unknown param_group_split_mode: \
                                  {config.optimizers.param_group_split_mode}")


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
                 update_type='step'
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