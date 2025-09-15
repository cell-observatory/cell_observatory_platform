import os
import warnings
warnings.filterwarnings("ignore")

import logging
from enum import Enum
from typing import Optional

import ray
from ray.train import get_context
from contextlib import contextmanager, nullcontext

import torch
from torch import distributed as dist

logger = logging.getLogger("ray")
logger.setLevel(logging.DEBUG)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


class OpMap(Enum):
    """
    Map of supported reduce operations.
    """
    SUM = dist.ReduceOp.SUM
    MAX = dist.ReduceOp.MAX
    MIN = dist.ReduceOp.MIN
    # Use SUM for mean, divide 
    # by world size later
    MEAN = dist.ReduceOp.SUM 


def is_main_process():
    return process_rank() == 0


def is_ray_initialized() -> bool:
    return ray.is_initialized()


def is_torch_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_context_manager():
    if is_ray_initialized():
        return get_context()
    elif is_torch_dist_initialized():
        return nullcontext()
    else:
        raise NotImplementedError


def process_rank() -> int:
    """
    Return the global rank, falling back to 0
    for single-process runs.
    Works for:
      - Ray Train via `ray.train.get_context()`
      - Plain torchrun/DDP - via `torch.distributed`
      - Local debugging - rank 0
    """
    try:
        return get_context().get_world_rank()
    except RuntimeError:
        pass
    if is_torch_dist_initialized():
        return dist.get_rank()
    else:
        return os.getenv("LOCAL_RANK", 0)


def get_world_size() -> int:
    """
    Return the global world size, falling 
    back to 1 for single-process runs.
    Works for:
      - Ray Train via `ray.train.get_context()`
      - Plain torchrun/DDP - via `torch.distributed`
      - Local debugging - world size 1
    """
    try:
        return get_context().get_world_size()
    except RuntimeError:
        pass
    if is_torch_dist_initialized():
        return dist.get_world_size()
    else:
        return 1


def barrier(device_ids: Optional[int] = None) -> None:
    """
    Global synchronisation:
      - Ray Train barrier when available
      - torch.distributed.barrier() NCCL backend 
        if available
      - fallback to cpu/Gloo otherwise 
      - No-op in single-process mode
    """
    try:
        ctx = get_context()
        if hasattr(ctx, "barrier"):
            ctx.barrier()
            return
    except RuntimeError:
        pass
    if is_torch_dist_initialized():
        dist.barrier(device_ids=[device_ids]) if \
            device_ids is not None else dist.barrier()
        return
    return


def gather_and_reduce(tensor: torch.Tensor, reduce_op: str = 'mean'):
    if not is_torch_dist_initialized():
        return tensor.clone()

    if reduce_op.upper() not in OpMap.__members__:
        raise ValueError(f"Unsupported op: {reduce_op}")

    dist.all_reduce(tensor, op=OpMap[reduce_op.upper()].value)
    if reduce_op == "mean":
        tensor /= get_world_size()
    return tensor


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)