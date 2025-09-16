import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import warnings
warnings.filterwarnings("ignore")

import pynvml as nvml
# from numa import schedule, memory

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


# ---------------- Distributed helpers ----------------


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


def node_id() -> str:
    if not ray.is_initialized():
        raise RuntimeError("Ray not initialized")

    rc = ray.get_runtime_context()
    if hasattr(rc, "get_node_id"):
        nid = rc.get_node_id()
    elif hasattr(rc, "node_id"):
        nid = rc.node_id
    else:
        raise NotImplementedError("Unable to get node ID from Ray runtime context")
    
    return nid


def get_context_manager():
    if is_ray_initialized():
        return get_context()
    elif is_torch_dist_initialized():
        return nullcontext()
    else:
        raise NotImplementedError


def local_rank() -> int:
    try:
        return get_context().get_local_rank()
    except Exception:
        pass
    if is_torch_dist_initialized():
        return int(os.getenv("LOCAL_RANK", "0"))
    return 0


def process_rank() -> int:
    try:
        return get_context().get_world_rank()
    except Exception:
        pass
    if is_torch_dist_initialized():
        return dist.get_rank()
    return 0


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


def get_visible_devices():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is not None, \
        "CUDA_VISIBLE_DEVICES not set"
    devices = os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
    return [int(d) for d in devices if d.isdigit()]


# ---------------- NVML helpers ----------------


_NVML_INIT = False
def _nvml_init():
    global _NVML_INIT
    if not _NVML_INIT:
        try:
            nvml.nvmlInit()
            _NVML_INIT = True
        except Exception:
            _NVML_INIT = False


def _nvml_handle_for_torch_index(torch_idx: int):
    _nvml_init()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return nvml.nvmlDeviceGetHandleByIndex(torch_idx)
    tokens = [t.strip() for t in cvd.split(",") if t.strip()]
    tok = tokens[torch_idx]
    if tok.isdigit():
        return nvml.nvmlDeviceGetHandleByIndex(int(tok))
    else:
        raise ValueError(f"Invalid CUDA_VISIBLE_DEVICES token: {tok}")


def _pci_bus_id(handle) -> str:
    # see: https://docs.nvidia.com/deploy/nvml-api/structnvmlPciInfo__t.html#structnvmlPciInfo__t
    pci = nvml.nvmlDeviceGetPciInfo(handle)
    domain, bb, dd_func = pci.busId.split(":", 2)
    domain = f"{int(domain, 16):04x}"
    return f"{domain}:{bb}:{dd_func}".lower()


def _nvml_cpu_affinity_cpus(handle) -> List[int]:
    _nvml_init()
    ncpu = os.cpu_count()
    cpus = []
    mask = nvml.nvmlDeviceGetCpuAffinity(handle, ncpu)
    for word_i, word in enumerate(mask):
        w = int(word)
        for b in range(64):
            if w & (1 << b):
                c = word_i * 64 + b
                if c < ncpu:
                    cpus.append(c)
    return sorted(set(cpus))


# ---------------- sysfs parsing ----------------


def _parse_range_list(s: str) -> List[int]:
    # "0-3,8,10-11" -> [0,1,2,3,8,10,11]
    out = []
    s = (s or "").strip()
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except Exception:
        return None


def _numa_from_sysfs(pci_bus_id: str) -> Optional[int]:
    p = Path(f"/sys/bus/pci/devices/{pci_bus_id}/numa_node")
    s = _read_text(p)
    if s is None:
        return None
    try:
        v = int(s)
        return v if v >= 0 else None
    except Exception:
        return None


def _local_cpulist_for_dev(pci_bus_id: str) -> List[int]:
    base = Path(f"/sys/bus/pci/devices/{pci_bus_id}")
    s = _read_text(base / "local_cpulist")
    if not s:
        return []
    return _parse_range_list(s)


def list_numa_nodes_with_gpus(device_count: int, strict: bool = True) -> list[int]:
    nodes = set()
    for ti in range(device_count):
        try:
            info = torch_gpu_to_numa(ti)
            n = info.get("numa_node")
            if n is not None and n >= 0:
                nodes.add(int(n))
        except Exception:
            pass

    if strict:
        online = set(list_numa_nodes())
        nodes &= online

    return sorted(nodes)

def list_numa_nodes() -> List[int]:
    base = Path("/sys/devices/system/node")
    online = _read_text(base / "online")
    if online:
        return sorted(_parse_range_list(online))
    # fallback: scan node directories
    nodes = []
    for p in base.glob("node[0-9]*"):
        try:
            nodes.append(int(p.name[4:]))
        except Exception:
            pass
    return sorted(nodes)


def _cpus_for_node(node: int) -> List[int]:
    p = Path(f"/sys/devices/system/node/node{node}/cpulist")
    s = _read_text(p)
    if not s:
        return []
    return _parse_range_list(s)


def _cpu_to_node_map() -> Dict[int, int]:
    m = {}
    for n in list_numa_nodes():
        for c in _cpus_for_node(n):
            m[c] = n
    return m


def _infer_nodes_from_cpus(cpus: List[int]) -> List[Tuple[int, int]]:
    cpu2node = _cpu_to_node_map()
    counts = {}
    for c in cpus:
        n = cpu2node.get(c)
        if n is not None:
            counts[n] = counts.get(n, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------- public API ----------------


def torch_gpu_to_numa(torch_idx: int) -> Dict:
    h = _nvml_handle_for_torch_index(torch_idx)
    pci = _pci_bus_id(h)

    # 1) sysfs numa_node
    numa = _numa_from_sysfs(pci)
    if numa is not None:
        return {"torch_index": torch_idx, 
                "pci_bus_id": pci, 
                "numa_node": numa, 
                "method": "sysfs.numa_node"
        }

    # 2) sysfs local_cpulist
    loc_cpus = _local_cpulist_for_dev(pci)
    if loc_cpus:
        candidates = _infer_nodes_from_cpus(loc_cpus)
        if candidates:
            best, _ = candidates[0]
            return {
                "torch_index": torch_idx,
                "pci_bus_id": pci,
                "numa_node": best,
                "method": "sysfs.local_cpulist",
                "candidates": candidates,
            }

    # 3) NVML CPU affinity
    aff_cpus = _nvml_cpu_affinity_cpus(h)
    if aff_cpus:
        candidates = _infer_nodes_from_cpus(aff_cpus)
        if candidates:
            best, _ = candidates[0]
            return {
                "torch_index": torch_idx,
                "pci_bus_id": pci,
                "numa_node": best,
                "method": "nvml.cpu_affinity",
                "candidates": candidates,
            }

    raise RuntimeError(f"Cannot determine NUMA node for torch GPU index {torch_idx}, PCI {pci}")


def bind_current_process_to_node(node: int):
    if node is None:
        raise RuntimeError("Cannot bind: NUMA node is None")
    target = set(_cpus_for_node(node))
    if not target:
        raise RuntimeError(f"Node {node} has no CPUs")
    allowed = set(os.sched_getaffinity(0))
    chosen = sorted(target & allowed)
    if not chosen:
        raise RuntimeError(f"Node {node} CPUs not in current cpuset; allowed={sorted(allowed)}")
    os.sched_setaffinity(0, chosen)
    return node


def pin_to_numa_node(gpu_id: int) -> Optional[int]:
    info = torch_gpu_to_numa(gpu_id)
    n = info.get("numa_node")
    if n is not None:
        bind_current_process_to_node(n)
    else:
        raise RuntimeError(f"Cannot pin to NUMA node for GPU {gpu_id}, info: {info}")
    return n


# ---------------- ---------------- ----------------