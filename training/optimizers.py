"""
Partly adapted from:
https://github.com/pytorch/torchtitan/torchtitan/components/optimizer.py
"""

from __future__ import annotations   # lazy annotations: torchtitan types stay import-free

import logging

from omegaconf import DictConfig

import functools
from typing import Any, Generic, Iterator, TypeVar

import torch
import torch.nn as nn
from torch.optim import Optimizer, Muon
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful

from timm.optim.lion import Lion

from cell_observatory_platform.utils.registry import REGISTRY

logger = logging.getLogger("ray")
logger.setLevel(logging.INFO)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


# --- Registry -------------------------------------------------------------- #
# Each optimizer is a config-selected swap point (`config.optimizers.name`). The
# factory reads the `optimizers` sub-config node and receives `params=` from the
# caller.

@REGISTRY.register("optimizer", "adamw")
def _build_adamw(cfg, *, params):
    from deepspeed.ops.adam import FusedAdam  # lazy: see module docstring note

    return FusedAdam(
        params,
        lr=cfg.lr,
        weight_decay=cfg.wd,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
    )


@REGISTRY.register("optimizer", "adamw_torch")
def _build_adamw_torch(cfg, *, params):
    # NOTE: sometimes DeepSpeed's fused AdamW has issues, so we fall back to
    #       torch's implementation.
    return torch.optim.AdamW(
        params,
        lr=cfg.lr,
        weight_decay=cfg.wd,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
        fused=True,
    )


@REGISTRY.register("optimizer", "lamb")
def _build_lamb(cfg, *, params):
    from deepspeed.ops.lamb import FusedLamb  # lazy: see module docstring note

    return FusedLamb(
        params,
        lr=cfg.lr,
        weight_decay=cfg.wd,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
    )


@REGISTRY.register("optimizer", "lion")
def _build_lion(cfg, *, params):
    return Lion(
        params,
        lr=cfg.lr,
        weight_decay=cfg.wd,
        betas=tuple(cfg.betas),
    )


@REGISTRY.register("optimizer", "lamb_torch")
def _build_lamb_torch(cfg, *, params):
    # torch-native LAMB (no DeepSpeed FusedLamb dependency); plain-torch ops,
    # so it also works on FSDP2 DTensor parameters.
    from timm.optim import Lamb  # lazy: keep timm.optim.lamb off actor imports

    return Lamb(
        params,
        lr=cfg.lr,
        weight_decay=cfg.wd,
        betas=tuple(cfg.betas),
        eps=cfg.eps,
    )


def get_optimizer(
    params,
    config: DictConfig,
    optimizer: str,
    steps_per_epoch: int,
):
    # NOTE: `muon` is not supported fully yet — it is intentionally left
    #       unregistered, so `name: muon` fails loud at build (was the else-branch):
    #     Muon(params, lr=config.optimizers.lr, weight_decay=config.optimizers.wd,
    #          eps=config.optimizers.eps,
    #          adjust_lr_fn=config.optimizers.get("adjust_lr_fn", None))
    opt = REGISTRY.build("optimizer", optimizer, config.optimizers, params=params)
    return opt, None


# --- TorchTitan Support --- #


T = TypeVar("T", bound=Optimizer)


class OptimizersContainer(Optimizer, Stateful, Generic[T]):
    """A container for multiple optimizers.

    This class is used to wrap multiple optimizers into a single object that can be
    used to reduce the complexity of the training loop. This mimics the behavior of
    ``torch.optim.Optimizer``. This class currently only supports ``Adam`` and ``AdamW``.

    **Limitations**
    This class assumes that all the optimizers are the same type and have the same
    configurations. With this assumption, TorchTitan can support lr scheduler resharding
    (e.g., loading a checkpoint with a different number of GPUs and/or different
    parallelization strategy). Note that ``get_optimizer_state_dict`` already enables the
    resharding for the optimizer state but not for the lr scheduler state, hence the limitation.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_kwargs (Dict[str, Any]): Keyword arguments for the optimizers.
        param_groups (List[dict] | None): Pre-split parameter groups (e.g. the
            decay/no-decay split from ``schedulers.get_param_groups``). Only
            supported with a single model part; when omitted, all
            ``requires_grad`` parameters form one group.
    """

    optimizers: list[T]
    model_parts: list[nn.Module]

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        param_groups: list[dict] | None = None,
    ) -> None:
        if param_groups is not None and len(model_parts) != 1:
            raise ValueError(
                "param_groups is only supported with a single model part "
                f"(got {len(model_parts)})."
            )
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        for model in self.model_parts:
            if param_groups is not None:
                params = param_groups
                all_params.extend(p for g in param_groups for p in g["params"])
            else:
                params = [p for p in model.parameters() if p.requires_grad]
                all_params.extend(params)
            self.optimizers.append(optimizer_cls(params, **optimizer_kwargs))
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

    def __getitem__(self, idx: int) -> T:
        # hooks address the wrapped optimizers positionally
        # (e.g. WeightDecayScheduleHook reads trainer.optimizers[0].param_groups)
        return self.optimizers[idx]

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {
            k: v
            for sd in map(func, self.model_parts, self.optimizers)
            for k, v in sd.items()
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model_parts, self.optimizers))

    def _validate_length(self, expected_length: int) -> None:
        assert expected_length == len(self.optimizers), (
            "Must pass one optimizer per model part."
        )

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)


def build_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: DictConfig,
    param_groups: list[dict] | None = None,
) -> OptimizersContainer:
    """Create an OptimizersContainer from the repo's ``optimizers`` config node.

    Consumes the same schema the DeepSpeed path uses
    (``configs/optimizers/*.yaml``: name/lr/wd/betas/eps), plus an optional
    ``implementation`` in {"fused", "foreach", "for-loop"} (default "fused";
    ignored for optimizers without those kwargs).

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_config (DictConfig): ``cfg.optimizers`` node.
        param_groups (List[dict] | None): Pre-split parameter groups from
            ``schedulers.get_param_groups`` (decay/no-decay, layer decay).
    """
    name = optimizer_config.name
    optimizer_kwargs: dict[str, Any] = {
        "lr": optimizer_config.lr,
        "betas": tuple(optimizer_config.betas),
        "eps": optimizer_config.eps,
        "weight_decay": optimizer_config.wd,
    }

    optimizer_classes = {
        "adam_torch": torch.optim.Adam,
        "adamw_torch": torch.optim.AdamW,
    }
    if name in optimizer_classes:
        implementation = optimizer_config.get("implementation", "fused")
        if implementation not in ("fused", "foreach", "for-loop"):
            raise ValueError(
                f"Unknown optimizer implementation: {implementation!r}. "
                "Valid: fused, foreach, for-loop."
            )
        optimizer_kwargs["fused"] = implementation == "fused"
        optimizer_kwargs["foreach"] = implementation == "foreach"
        optimizer_cls = optimizer_classes[name]
    elif name == "lamb_torch":
        from timm.optim import Lamb  # lazy: see module docstring note

        optimizer_cls = Lamb
    elif name == "lion_torch":
        optimizer_cls = Lion
        optimizer_kwargs.pop("eps")
    else:
        raise NotImplementedError(
            f"Optimizer {name!r} is not supported on the torch-native path. "
            "Valid: adam_torch, adamw_torch, lamb_torch, lion_torch."
        )

    return OptimizersContainer(
        model_parts, optimizer_cls, optimizer_kwargs, param_groups=param_groups
    )

# --------------------------------------------------------------------------- #
# PARKED (planned return; do NOT delete): deepspeed-managed LR scheduler branch of get_optimizer
# --------------------------------------------------------------------------- #
# # (was the `if deepspeed_scheduler:` branch of get_optimizer; note the
# #  `cos_miwarmup_type` typo -- should be `warmup_type` -- if resurrected)
#     if deepspeed_scheduler:
#         decay_epochs = config.schedulers.epochs - (config.schedulers.warmup + config.schedulers.cooldown)
#         total_steps = config.schedulers.epochs * steps_per_epoch
#         warmup_steps = config.schedulers.warmup * steps_per_epoch
#         decay_steps = total_steps - warmup_steps
#
#         cos_min_lr = config.schedulers.cos_min_ratio * config.optimizers.lr
#         warmup_min_lr = config.schedulers.warmup_min_ratio * config.optimizers.lr
#
#         logger.info("-" * 80)
#         logger.info(
#             f"Epochs: {config.schedulers.epochs} = "
#             f"[{config.schedulers.warmup} warmup + {decay_epochs} decay + NA cooldown]\n"
#             f"Steps: {total_steps} = "
#             f"[{warmup_steps} warmup + {decay_steps} decay + NA cooldown]\n"
#             f"LR: {config.optimizers.lr} = [{warmup_min_lr=},  {cos_min_lr=}]"
#         )
#         logger.info("-" * 80)
#
#         scheduler = WarmupCosineLR(
#             optimizer=opt,
#             total_num_steps=total_steps,
#             warmup_num_steps=warmup_steps,
#             warmup_min_ratio=config.schedulers.warmup_min_ratio,
#             cos_min_ratio=config.schedulers.cos_min_ratio,
#             warmup_type=config.schedulers.cos_miwarmup_type,
#         )
#
#         return opt, scheduler
#     else:
#         return opt, None
