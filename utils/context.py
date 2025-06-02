import warnings
warnings.filterwarnings("ignore")

import logging
import ray
from ray.train import get_context
from contextlib import contextmanager, nullcontext
from torch import distributed as dist

logger = logging.getLogger("ray")
logger.setLevel(logging.DEBUG)
logging.getLogger("ray.train._internal.checkpoint_manager").setLevel(logging.INFO)


def is_main_process():
    return get_context().get_world_rank() == 0

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
    Return the global rank, falling back to 0 for single-process runs.
    Works for:
      - Ray Train via `ray.train.get_context()`
      - Plain torchrun/DDP - via `torch.distributed`
      - Local debugging - rank 0
    """
    if is_ray_initialized():
        return get_context().get_world_rank()
    elif is_torch_dist_initialized():
        return dist.get_rank()
    else:
        return 0


def get_world_size() -> int:
    """
    Return the global world size, falling back to 1 for single-process runs.
    Works for:
      - Ray Train via `ray.train.get_context()`
      - Plain torchrun/DDP - via `torch.distributed`
      - Local debugging - world size 1
    """
    if is_ray_initialized():
        return get_context().get_world_size()
    elif is_torch_dist_initialized():
        return dist.get_world_size()
    else:
        return 1


def barrier() -> None:
    """
    Global synchronisation:
      - Ray Train barrier when available
      - torch.distributed.barrier() otherwise
      - No-op in single-process mode
    """
    if is_ray_initialized():
        ctx = get_context()
        if hasattr(ctx, "barrier"):
            ctx.barrier()
            return
    if is_torch_dist_initialized():
        dist.barrier()


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