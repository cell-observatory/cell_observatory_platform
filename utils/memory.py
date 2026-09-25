from __future__ import annotations

import os
import psutil
import datetime as _dt
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import ray
import torch


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def read_proc_meminfo(keys: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                k = parts[0].rstrip(":")
                if k in keys:
                    # meminfo is kB
                    out[k] = int(parts[1]) * 1024
    except Exception:
        pass
    return out


def bytes_gb(x: Optional[int]) -> str:
    if x is None:
        return "NA"
    return f"{x / 1e9:.3f} GB"


def statvfs_usage(path: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    try:
        st = os.statvfs(path)
        total = st.f_frsize * st.f_blocks
        free = st.f_frsize * st.f_bfree
        used = total - free
        return total, used, free
    except Exception:
        return None, None, None


def top_dir_entries_by_size(path: str, n: int = 25) -> str:
    try:
        entries = []
        with os.scandir(path) as it:
            for e in it:
                try:
                    st = e.stat(follow_symlinks=False)
                    entries.append((st.st_size, e.path))
                except Exception:
                    continue
        entries.sort(key=lambda x: x[0])
        tail = entries[-n:]
        lines = [f"{size:>12}  {p}" for size, p in tail]
        return "\n".join(lines) if lines else "(none)"
    except FileNotFoundError:
        return "(missing)"
    except Exception as e:
        return f"(error: {e})"


def _run_cmd(cmd: Sequence[str], timeout_s: float = 10.0) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout_s)
        return out.strip()
    except Exception as e:
        return f"(error running {cmd!r}: {e})"


def ray_memory_summary(stats_only: bool = True) -> str:
    try:
        from ray._private import internal_api
        try:
            return internal_api.memory_summary(stats_only=stats_only)
        except TypeError:
            return internal_api.memory_summary()
    except Exception:
        return _run_cmd(["ray", "memory", "--stats-only"])


def ray_resources_summary() -> str:
    try:
        if not ray.is_initialized():
            return "(ray not initialized)"
        cluster = ray.cluster_resources()
        avail = ray.available_resources()
        return f"cluster_resources={cluster}\navailable_resources={avail}"
    except Exception as e:
        return f"(error: {e})"


def torch_cuda_summary() -> str:
    try:
        if not torch.cuda.is_available():
            return "(cuda not available)"
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)
        max_alloc = torch.cuda.max_memory_allocated(dev)
        max_reserved = torch.cuda.max_memory_reserved(dev)
        return (
            f"cuda_device={dev}\n"
            f"allocated={bytes_gb(alloc)} reserved={bytes_gb(reserved)}\n"
            f"max_allocated={bytes_gb(max_alloc)} max_reserved={bytes_gb(max_reserved)}"
        )
    except Exception as e:
        return f"(error: {e})"


def process_summary() -> str:
    pid = os.getpid()
    try:
        p = psutil.Process(pid)
        rss = p.memory_info().rss
        vms = p.memory_info().vms
        num_fds = getattr(p, "num_fds", lambda: None)()
        num_threads = p.num_threads()
        return (
            f"pid={pid} rss={bytes_gb(rss)} vms={bytes_gb(vms)} "
            f"threads={num_threads} fds={num_fds}"
        )
    except Exception as e:
        return f"(error: {e})"


def top_processes_rss(n: int = 15) -> str:
    return _run_cmd(["bash", "-lc", f"ps -eo pid,rss,comm --sort=-rss | head -n {n+1}"])


def check_inference_capacity(
    output_base_path: Union[Path, str],
    estimated_bytes: int,
    min_free_disk_gb: float = 10.0,
    min_free_ram_gb: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Pre-flight check for inference: verify disk and optionally RAM capacity.

    Returns (ok, message). If ok is False, message explains the failure.
    """
    path = str(output_base_path)
    if not path or path == ".":
        path = os.getcwd()
    path = os.path.abspath(path)

    total, used, free = statvfs_usage(path)
    if free is not None:
        min_free_bytes = int(min_free_disk_gb * 1e9)
        if free < estimated_bytes:
            return (
                False,
                f"Insufficient disk space: need {estimated_bytes / 1e9:.1f} GB, "
                f"only {free / 1e9:.1f} GB free at {path}",
            )
        if free < min_free_bytes:
            return (
                False,
                f"Insufficient disk space: need at least {min_free_disk_gb} GB free, "
                f"only {free / 1e9:.1f} GB free at {path}",
            )
    else:
        return False, f"Cannot stat filesystem at {path}"

    if min_free_ram_gb is not None:
        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / 1e9
            if available_gb < min_free_ram_gb:
                return (
                    False,
                    f"Insufficient RAM: need at least {min_free_ram_gb} GB free, "
                    f"only {available_gb:.1f} GB available",
                )
        except Exception as e:
            return False, f"Cannot check RAM: {e}"

    return True, "OK"