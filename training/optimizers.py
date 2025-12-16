"""
Partly adapted from:
https://github.com/pytorch/torchtitan/torchtitan/components/optimizer.py
"""

import logging

from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.lamb import FusedLamb
from deepspeed.runtime.lr_schedules import WarmupCosineLR

from omegaconf import DictConfig

import functools
from typing import Any, Generic, Iterator, TypeVar

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import Optimizer
from torch.optim import Muon

from torchtitan.distributed import ParallelDims
from torchtitan.components.ft import FTManager, has_torchft

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
    if optimizer == "adamw":
        opt = FusedAdam(
            params,
            lr=config.optimizers.lr,
            weight_decay=config.optimizers.wd,
            betas=tuple(config.optimizers.betas),
            eps=config.optimizers.eps,
        )
    elif optimizer == "lamb":
        opt = FusedLamb(
            params,
            lr=config.optimizers.lr,
            weight_decay=config.optimizers.wd,
            betas=(0.9, 0.99),
            eps=1e-08,
        )
    elif optimizer == "muon":
        opt = Muon(
            params,
            lr=config.optimizers.lr,
            weight_decay=config.optimizers.wd,
            betas=tuple(config.optimizers.betas),
            eps=config.optimizers.eps,
            adjust_lr_fn=config.optimizers.get("adjust_lr_fn", None),
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

        logger.info("-" * 80)
        logger.info(
            f"Epochs: {config.schedulers.epochs} = "
            f"[{config.schedulers.warmup} warmup + {decay_epochs} decay + NA cooldown]\n"
            f"Steps: {total_steps} = "
            f"[{warmup_steps} warmup + {decay_steps} decay + NA cooldown]\n"
            f"LR: {config.optimizers.lr} = [{warmup_min_lr=},  {cos_min_lr=}]"
        )
        logger.info("-" * 80)

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
        name (str): Name of the optimizers.
    """

    optimizers: list[T]
    model_parts: list[nn.Module]

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        for model in self.model_parts:
            params = [p for p in model.parameters() if p.requires_grad]
            self.optimizers.append(optimizer_cls(params, **optimizer_kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

    def __iter__(self) -> Iterator[T]:
        return iter(self.optimizers)

    def __len__(self) -> int:
        return len(self.optimizers)

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
            "Must pass one optimizer per model part or per param if "
            "using OptimizersInBackwardContainer."
        )

    def _post_init(
        self, all_params: list[nn.Parameter], optimizer_kwargs: dict[str, Any]
    ) -> None:
        # We need to call Optimizer.__init__() to initialize some necessary optimizer
        # functionality such as hooks.
        Optimizer.__init__(self, all_params, optimizer_kwargs)


class OptimizersInBackwardContainer(OptimizersContainer):
    """OptimizersContainer for executing ``optim.step()`` in backward pass.

    This class extend ``OptimizersContainer`` to support optimizer step in
    backward pass. ``step()`` and ``zero_grad()`` are no-op in this class.
    Instead, ``register_post_accumulate_grad_hook`` is used to register a hook to
    execute these methods when the gradient is accumulated.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.model_parts = model_parts

        optim_dict = {}
        for model in self.model_parts:
            for p in model.parameters():
                if p.requires_grad:
                    optim_dict[p] = optimizer_cls([p], **optimizer_kwargs)
                all_params.append(p)

        def optim_hook(param) -> None:
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = list(optim_dict.values())

        self._validate_length(
            sum(len(list(model.parameters())) for model in self.model_parts)
        )
        self._post_init(all_params, optimizer_kwargs)

    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass
    

def build_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: dict,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    """Create a OptimizersContainer for the given model parts and job config.

    This function creates a ``OptimizersContainer`` for the given model parts.
    ``optimizer_config`` should define the correct optimizer name and parameters.
    This function currently supports creating ``OptimizersContainer`` and
    ``OptimizersInBackwardContainer``.

    Args:
        model_parts (List[nn.Module]): List of model parts to be optimized.
        optimizer_config (OptimizerConfig): Optimizer config containing the optimizer name and parameters.
        parallel_dims (ParallelDims): Parallel dimensions for the model.
    """
    optim_in_bwd = optimizer_config["early_step_in_backward"]
    if optim_in_bwd:
        if parallel_dims.ep_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Expert Parallel."
            )
        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Optimizers in backward is not supported with Pipeline Parallel."
            )
        if ft_manager and ft_manager.enabled:
            raise NotImplementedError(
                "TorchFT is not supported with optimizers in backward."
            )

    name = optimizer_config["name"]
    lr = optimizer_config["lr"]
    beta1 = optimizer_config["beta1"]
    beta2 = optimizer_config["beta2"]
    eps = optimizer_config["eps"]
    weight_decay = optimizer_config["wd"]

    optim_implementation = optimizer_config["implementation"]
    assert optim_implementation in ["fused", "foreach", "for-loop"]

    fused = optim_implementation == "fused"
    foreach = optim_implementation == "foreach"

    optimizer_kwargs = {
        "lr": lr,
        "betas": (beta1, beta2),
        "eps": eps,
        "weight_decay": weight_decay,
        "fused": fused,
        "foreach": foreach,
    }

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {name} not added.")
    optimizer_cls = optimizer_classes[name]

    if optim_in_bwd:
        return OptimizersInBackwardContainer(
            model_parts, optimizer_cls, optimizer_kwargs
        )

    if ft_manager and ft_manager.enabled:
        raise NotImplementedError(
            "TorchFT is not yet supported with multiple optimizers."
        )

    return OptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)