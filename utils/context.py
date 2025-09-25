import os
import re
import ctypes
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from multiprocessing import shared_memory

import warnings
warnings.filterwarnings("ignore")

import pynvml as nvml
from numa import schedule, memory, info

import logging
from enum import Enum
from typing import Optional

import ray
from ray.train import get_context
from contextlib import contextmanager, nullcontext

import cupy as cp

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


def get_local_world_size() -> int:
    try:
        return get_context().get_local_world_size()
    except RuntimeError:
        pass
    if is_torch_dist_initialized():
        return int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    else:
        return 1


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


def get_node_process_group():
    assert dist.is_initialized()
    g_rank = process_rank()
    l_rank = local_rank()
    l_size = get_local_world_size()
    base = g_rank - l_rank
    ranks = list(range(base, base + l_size))
    return dist.new_group(ranks=ranks)


def get_local_numa_nodes(worker_numa_node: int):
    if not dist.is_initialized():
        # single-process case
        return [worker_numa_node]

    pg = get_node_process_group()
    lrank = local_rank()
    lsize = get_local_world_size()

    dev = torch.device(f"cuda:{lrank}")
    worker_numa = torch.tensor([worker_numa_node], dtype=torch.int32, device=dev)
    buf = torch.empty(lsize, dtype=torch.int32, device=dev)

    # gather each local rank's NUMA id
    dist.all_gather_into_tensor(buf, worker_numa, group=pg)

    if lrank == 0:
        buf = buf.cpu().tolist()
        buf = {i: n for i, n in enumerate(buf)}
        return buf
    else:
        return None


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
    # NOTE: NVML idx and CUDA idx may not be correlated hence
    #       we use PCIBusId
    device_PCIBusId = cp.cuda.runtime.deviceGetPCIBusId(torch_idx)
    # prefer v2 if available
    for fn_name in ("nvmlDeviceGetHandleByPciBusId_v2", "nvmlDeviceGetHandleByPciBusId"):
        pci_info_func = getattr(nvml, fn_name, None)
        if pci_info_func is None:
            continue
        handle = pci_info_func(device_PCIBusId)
        if handle is None:
            raise RuntimeError(f"NVML handle not found for PCI {device_PCIBusId}")
        return handle


_BUSID_RE = re.compile(
    r'^(?:(?P<domain>[0-9A-Fa-f]{4,8}):)?(?P<bus>[0-9A-Fa-f]{2}):(?P<dev>[0-9A-Fa-f]{2})(?:\.(?P<func>[0-7]))?$'
)


def _decode_c_buffer(x):
    if isinstance(x, bytes):
        s = x.decode("utf-8", "ignore")
    elif isinstance(x, str):
        s = x
    elif isinstance(x, (ctypes.Array,)):
        s = ctypes.string_at(x).decode("utf-8", "ignore")
    else:
        s = str(x)
    return s.split("\x00", 1)[0].strip()


def _format_from_fields(domain: int, bus: int, dev: int, func: int | None) -> str:
    d = f"{int(domain):04x}"
    b = f"{int(bus):02x}"
    de = f"{int(dev):02x}"
    f = str(func) if func is not None else "0"
    return f"{d}:{b}:{de}.{f}"


def _try_struct_fields(pci) -> str | None:
    if all(hasattr(pci, attr) for attr in ("domain", "bus", "device")):
        func = None
        # try to recover function number from busId if present
        raw = getattr(pci, "busId", None)
        if raw is not None:
            m = _BUSID_RE.match(_decode_c_buffer(raw))
            if m and m.group("func"):
                func = int(m.group("func"))
        return _format_from_fields(pci.domain, pci.bus, pci.device, func)
    return None


def _pci_bus_id(handle) -> str:
    # 1) prefer v3 if available
    for fn_name in ("nvmlDeviceGetPciInfo_v3", "nvmlDeviceGetPciInfo"):
        pci_info_func = getattr(nvml, fn_name, None)
        if pci_info_func is None:
            continue
        pci = pci_info_func(handle)
        s = _try_struct_fields(pci)
        if s:
            return s.lower()

        # 2) fallback: parse busId text
        if hasattr(pci, "busId"):
            raw = _decode_c_buffer(pci.busId)
            m = _BUSID_RE.match(raw)
            if not m:
                raise RuntimeError(f"Unrecognized NVML busId format: {raw!r}")
            domain = int(m.group("domain"), 16) if m.group("domain") else 0
            bus = int(m.group("bus"), 16)
            dev = int(m.group("dev"), 16)
            func = int(m.group("func")) if m.group("func") else 0
            return _format_from_fields(domain, bus, dev, func).lower()

    # if none of the getters exist (very old NVML), this will run:
    raise RuntimeError("No NVML pci info function available")


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


def _cpu_to_node_map() -> Dict[int, int]:
    m = {}
    for n in list_numa_nodes():
        for c in cpus_for_node(n):
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


def cpus_for_node(node: int) -> List[int]:
    p = Path(f"/sys/devices/system/node/node{node}/cpulist")
    s = _read_text(p)
    if not s:
        return []
    return _parse_range_list(s)


def read_numa_distance_row(node: int) -> Optional[List[int]]:
    # /sys/devices/system/node/nodeX/distance -> "10 20 20 10 ..."
    p = Path(f"/sys/devices/system/node/node{node}/distance")
    s = _read_text(p)
    if not s:
        return None
    try:
        return [int(x) for x in s.split()]
    except Exception:
        return None


# TODO: This function must be called from a process that 
#       has visibility of a CUDA capable device, else
#       will throw. There may be a way to get the 
#       correct NVML handle without this requirement
#       however since the CUDA idx may not always 
#       correlate with NVML idx this will require care.
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

    raise RuntimeError(f"Cannot determine NUMA node for torch GPU index {torch_idx}, PCI {pci}")


def bind_current_process_to_node(node: int):
    if node is None:
        raise RuntimeError("Cannot bind: NUMA node is None")
    target = set(cpus_for_node(node))
    if not target:
        raise RuntimeError(f"Node {node} has no CPUs")
    allowed = set(os.sched_getaffinity(0))
    chosen = sorted(target & allowed)
    if not chosen:
        raise RuntimeError(f"Node {node} CPUs not in current cpuset; allowed={sorted(allowed)}")
    try:
        schedule.run_on_nodes(node)
        memory.set_membind_nodes(node)
    except Exception as e:
        logger.warning(f"numa.schedule/memory bind failed: {e}")
        os.sched_setaffinity(0, chosen)
    return node


def unlink_shared_memory():
    # TODO: change naming scheme of shared memory segments
    paths = Path("/dev/shm").glob("psm_*")
    for p in paths:
        name = p.name
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


# ---------------- ---------------- ----------------