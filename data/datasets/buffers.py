from __future__ import annotations
import sys
import atexit
import asyncio
import contextlib
import ctypes
import logging
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Buffer

from omegaconf import DictConfig, open_dict

import numpy as np
from dataclasses import dataclass, field
import ray
from ray.actor import ActorHandle, ActorProxy
import torch
from collections import defaultdict
import queue
from threading import Lock
from multiprocessing import resource_tracker, shared_memory
import time
import re

from cell_observatory_platform.data.data_types import NUMPY_DTYPES, TORCH_DTYPES
from cell_observatory_platform.utils.context import (
    local_rank, 
    node_id,
    bind_current_process_to_node
)

logger = logging.getLogger(__name__)

# Non-greedy pool_name + anchored end: a greedy `.*` would swallow a literal
# "_numa_<n>_rank_<n>" inside a pool name and mis-parse the trailing fields.
BUFFER_NAME_REGEX = re.compile(r"host_pinned_shm_buffer_(?P<pool_name>.+?)_numa_(?P<numa_node>\d+)_rank_(?P<global_rank>\d+)$")


def attach_shared_memory(name: str) -> shared_memory.SharedMemory:
    """Attach to owner-managed shared memory without registering unlink cleanup.

    Python's resource tracker treats every ``SharedMemory(name=...)`` attach as
    something this process may clean up at exit. In this codebase the
    HostMemoryBuffer actor owns unlinking, while loaders/collators only borrow
    the segment. Unregister attachers so a short-lived Ray worker cannot unlink
    the owner's segment out from under later workers.
    """
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        pass

    shm = shared_memory.SharedMemory(name=name)
    resource_tracker.unregister(shm._name, "shared_memory")
    return shm

def parse_buffer_name(buffer_name: str) -> Dict[str, str]:
    match = BUFFER_NAME_REGEX.match(buffer_name)
    if match is None:
        raise ValueError(f"Invalid buffer name: {buffer_name}")
    return match.groupdict()

def get_buffer_name(pool_name: str, numa_node: int, global_rank: int) -> str:
    return f"host_pinned_shm_buffer_{pool_name}_numa_{numa_node}_rank_{global_rank}"

def slot_info_to_view(slot_info: Dict[str, Any], shm: shared_memory.SharedMemory) -> np.ndarray:
    """
    Convert slot info to a view of the slot.
    """
    batch_shape = slot_info["batch_shape"]
    dtype = slot_info["dtype"]
    dtype = np.dtype(
        NUMPY_DTYPES[dtype].value
        if isinstance(dtype, str)
        else dtype
    )
    buffer = shm.buf
    offset = slot_info["slot"] * slot_info["slot_bytes"]
    return np.ndarray(batch_shape, dtype=dtype, buffer=buffer, offset=offset)


class DeviceMemoryBuffer:
    def __init__(self,
                 name: str,
                 capacity: int, 
                 input_shape: tuple,
                 batch_size: int, 
                 dtype: str,
                 device_idx: int
    ):
        self.name = name
        self.capacity = int(capacity)
        self.batch_shape = (int(batch_size), *tuple(input_shape))
        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        self.device = torch.device(f"cuda:{device_idx}")
        torch.cuda.set_device(self.device)

        with torch.cuda.device(self.device):
            self.device_buffers = [
                torch.empty(self.batch_shape, dtype=self.dtype, device=self.device).contiguous()
                for _ in range(self.capacity)
            ]

        self._free = queue.SimpleQueue()
        for i in range(self.capacity):
            self._free.put(i)

    def get_free(self) -> int:
        return self._free.get()

    def put_free(self, slot: int) -> None:
        self._free.put(int(slot))


@ray.remote(namespace="buffers", lifetime="detached", num_cpus=0)
class HostMemoryBuffer:
    def __init__(self, 
                 numa_node: int,
                 name: str,
                 capacity: int, 
                 input_shape: tuple,
                 batch_size: int, 
                 dtype: str = "uint16", 
                 pin_numa_node: bool = True
):
        if pin_numa_node:
            bind_current_process_to_node(numa_node)

        self.actor_name = name
        self.cap = int(capacity)
        self.batch_shape = (int(batch_size), *tuple(input_shape))
        self.dtype = np.dtype(NUMPY_DTYPES[dtype].value if isinstance(dtype, str) else dtype)
        self.slot_bytes = int(np.prod(self.batch_shape)) * self.dtype.itemsize

        total_bytes = self.cap * self.slot_bytes
        self._shm = shared_memory.SharedMemory(create=True, size=total_bytes)
        self.name = self._shm.name

        self._try_get_free_drops = 0
        self._occupied_slots = 0
        self._metrics_enabled = False
        self._metrics: Dict[str, float | int | list[float | int]] = {
            "get_free_wait_time_ms": [],
            "put_free_wait_time_ms": [],
            "try_get_free_wait_time_ms": [],
            "try_get_free_drops": [],
            "occupied_slots": [],
        }

        self.free = asyncio.Queue(self.cap)
        for i in range(self.cap):
            self.free.put_nowait(i)
        # Slots currently checked OUT of the free queue. put_free of a slot not
        # in this set is a double free (or a free of a never-issued slot) and
        # raises instead of silently over-filling the queue.
        self._outstanding: set = set()

        atexit.register(self._cleanup)

    def _cleanup(self):
        if getattr(self, "_cleaned", False):
            return
        self._cleaned = True
        try:
            self._shm.close()
            self._shm.unlink()
        except FileNotFoundError:
            pass

    def release(self) -> None:
        """Unlink the segment from inside the owner. atexit does not run under
        ``ray.kill``, so teardown must be an explicit remote call (see
        BufferManager.remove_buffer)."""
        self._cleanup()

    def free_count(self) -> int:
        """Number of slots currently in the free queue (== capacity when idle)."""
        return self.free.qsize()

    async def get_free(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = await self.free.get()
        t1 = time.perf_counter()
        self._outstanding.add(int(slot))
        if self._metrics_enabled:
            self._metrics["get_free_wait_time_ms"].append((t1 - t0) * 1000)
            self._occupied_slots += 1
            self._metrics["occupied_slots"].append(self._occupied_slots)
        return {
            "slot": slot,
            "name": self.name,
            "actor_name": self.actor_name,
            "slot_bytes": self.slot_bytes,
            "batch_shape": self.batch_shape,
            "dtype": self.dtype,
            "capacity": self.cap,
        }

    def try_get_free(self) -> Optional[Dict[str, Any]]:
        """Non-blocking get. Returns slot info dict or None if queue empty."""
        t0 = time.perf_counter()
        try:
            slot = self.free.get_nowait()
        except asyncio.QueueEmpty:
            # The drop counter must live here: after the early return below it
            # was dead code and pool-exhaustion drops were never counted.
            self._try_get_free_drops += 1
            if self._metrics_enabled:
                self._metrics["try_get_free_drops"].append(self._try_get_free_drops)
            return None
        t1 = time.perf_counter()
        self._outstanding.add(int(slot))
        if self._metrics_enabled:
            self._metrics["try_get_free_wait_time_ms"].append((t1 - t0) * 1000)
            self._occupied_slots += 1
            self._metrics["occupied_slots"].append(self._occupied_slots)
        return {
            "slot": slot,
            "name": self.name,
            "actor_name": self.actor_name,
            "slot_bytes": self.slot_bytes,
            "batch_shape": self.batch_shape,
            "dtype": self.dtype,
            "capacity": self.cap,
        }

    async def put_free(self, slot: int):
        slot = int(slot)
        if slot not in self._outstanding:
            # Double free (or free of a never-issued slot): the queue would
            # exceed capacity / hand the same slot to two writers.
            raise RuntimeError(
                f"put_free({slot}) on pool {self.actor_name!r}: slot is not "
                f"outstanding (double free?). outstanding={sorted(self._outstanding)}"
            )
        self._outstanding.discard(slot)
        t0 = time.perf_counter()
        await self.free.put(slot)
        t1 = time.perf_counter()
        if self._metrics_enabled:
            self._metrics["put_free_wait_time_ms"].append((t1 - t0) * 1000)
            self._occupied_slots -= 1
        return True

    def get_config(self):
        return {"capacity": self.cap, "name": self.name, "slot_bytes": self.slot_bytes,
                "batch_shape": self.batch_shape, "dtype": self.dtype}

    def enable_metrics_collection(self) -> None:
        self._metrics_enabled = True

    def disable_metrics_collection(self) -> None:
        self._metrics_enabled = False

    def get_metrics(self) -> Dict[str, float | int | list[float | int]]:
        metrics = self._metrics.copy()
        metrics["capacity"] = self.cap
        metrics["slot_bytes"] = self.slot_bytes
        self._metrics = {
            "get_free_wait_time_ms": [],
            "put_free_wait_time_ms": [],
            "try_get_free_wait_time_ms": [],
            "try_get_free_drops": [],
            "occupied_slots": [],
        }
        return metrics



def set_buffers(
    local_rank: int,
    global_rank: int,
    numa_node: int,
    dtype: str,
    batch_size: int,
    input_shape: tuple,
    buffer_type: str,
    buffer_capacity: int,
    pin_numa_node: bool,
    node_id: int,
    pool_name: str,
    max_concurrent_calls: int = 256,
) -> Tuple[ActorHandle[HostMemoryBuffer], Dict[str, Any]]:
    """
    Set up a memory buffer actor for a given pool.

    Idempotent: if an actor with the same name already exists in the namespace,
    returns it and its config instead of creating a new one. The existing
    buffer's config must match the requested capacity, batch_shape, and dtype.
    """
    if buffer_type == "host_memory":
        name = get_buffer_name(pool_name, numa_node, global_rank)
        namespace = f"buffers_node_{node_id}"
        expected_batch_shape = (int(batch_size), *tuple(input_shape))

        try:
            buffer: ActorHandle[HostMemoryBuffer] = ray.get_actor(name, namespace=namespace)
            buffer_cfg = ray.get(buffer.get_config.remote())
            # Validate existing buffer matches requested config
            if (buffer_cfg["capacity"] != buffer_capacity
                    or buffer_cfg["batch_shape"] != expected_batch_shape
                    or buffer_cfg["dtype"] != np.dtype(NUMPY_DTYPES[dtype].value if isinstance(dtype, str) else dtype)):
                raise ValueError(
                    f"Existing buffer '{name}' config does not match request: "
                    f"capacity {buffer_cfg['capacity']} vs {buffer_capacity}, "
                    f"batch_shape {buffer_cfg['batch_shape']} vs {expected_batch_shape}, "
                    f"dtype {buffer_cfg['dtype']} vs {dtype}"
                )
            # Detached actors survive driver crashes WITH their queue state: a
            # previous run that died with slots checked out leaves a depleted
            # free queue, and every save on top of it silently no-ops
            # (block_on_save: false) or deadlocks (true). Refuse to reuse it.
            free_count = ray.get(buffer.free_count.remote())
            if free_count != buffer_capacity:
                raise RuntimeError(
                    f"reusing pool {name!r} with a depleted free queue "
                    f"({free_count}/{buffer_capacity} slots free; a previous run "
                    f"crashed with slots checked out); kill the actor or reset "
                    f"the queue before reuse."
                )
            ray.logger.info(
                f"Reusing existing shared memory buffer actor '{name}' "
                f"on NUMA node {numa_node}, local rank {local_rank}, node id {node_id}, global rank {global_rank}."
            )
            return buffer, buffer_cfg
        except ValueError as e:
            if "config does not match request" in str(e):
                raise
            # Actor not found; create it below

        ray.logger.info(f"Global rank {global_rank} creating {pool_name} host buffer actor "
                    f"on local rank {local_rank} and NUMA node {numa_node}")

        scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=node_id,
            soft=False,
        )
        buffer = HostMemoryBuffer.options(
            name=name,
            namespace=namespace,
            lifetime="detached",
            # Allow for concurrent get/put calls
            max_concurrency=max_concurrent_calls,
            scheduling_strategy=scheduling_strategy,
        ).remote(
            name=name,
            dtype=dtype,
            capacity=buffer_capacity,
            batch_size=batch_size,
            input_shape=input_shape,
            pin_numa_node=pin_numa_node,
            numa_node=numa_node,
        )
        buffer_cfg = ray.get(buffer.get_config.remote())

        ray.logger.info(f"Shared memory buffer actor '{name}' "
                    f"on NUMA node {numa_node} and local rank {local_rank} "
                    f"and node id {node_id} and global rank {global_rank} "
                    f"with capacity {buffer_capacity} and batch shape "
                    f"{(batch_size, *input_shape)} set up.")

        return buffer, buffer_cfg

    else:
        raise ValueError(f"Unsupported buffer type: {buffer_type}")
    

def get_buffers(
    type: str, 
    global_rank: int, 
    local_rank: int, 
    node_id: int, 
    pool_name: str,
    numa_node: int = None,
) -> ActorProxy[HostMemoryBuffer]:
    """
    Get a memory buffer actor for a given pool.
    """
    if type == "host_memory":
        name = f"host_pinned_shm_buffer_{pool_name}_numa_{numa_node}_rank_{global_rank}"
        return ray.get_actor(name, namespace=f"buffers_node_{node_id}")
    else:
        raise ValueError(f"Unsupported buffer type: {type}")


def get_slot_bytes(shape: tuple[int, ...], dtype: str) -> int:
    return (
        int(np.prod(shape)) * np.dtype(NUMPY_DTYPES[dtype].value 
        if isinstance(dtype, str) else dtype).itemsize
    )


def dense_buffer_formats(output_metadata, layout: Optional[str]) -> Dict[str, str]:
    """``{name -> data_format}`` for every DENSE buffered tensor, save or viz.

    Dense membership comes from ``save_tensors[name].annotation_type == "dense"``
    for saved tensors, plus ``tensor_info[name].kind == "dense"`` for viz-only
    dense tensors (e.g. ``data_tensor``, declared in the inference config).
    Keying restore/enlarge/crop off save_tensors alone left viz-only dense
    tensors at processed shape while predictions were restored to the original
    frame -- misaligned overlays (round-3 H-I3).
    """
    tensor_info = output_metadata.get("tensor_info", {}) or {}
    save_tensors = output_metadata.get("save_tensors", {}) or {}
    visualize_tensors = list(output_metadata.get("visualize_tensors", []) or [])

    dense_formats: Dict[str, str] = {}
    for name, meta in (save_tensors.items() if hasattr(save_tensors, "items") else []):
        if hasattr(meta, "get") and meta.get("annotation_type") == "dense":
            dense_formats[name] = str(meta.get("data_format", layout)).upper()
    for name in visualize_tensors:
        if name in dense_formats:
            continue
        info = tensor_info.get(name)
        if info is not None and hasattr(info, "get") and info.get("kind") == "dense":
            dense_formats[name] = str(info.get("data_format", layout)).upper()
    return dense_formats


def _resize_dense_output_shapes_for_restore(output_metadata, layout: Optional[str], stats) -> None:
    """Grow dense save/viz buffer shapes to the FULL ORIGINAL TILE.

    Tile-mode inference resizes inputs down to the model's ``train_shape`` and then
    restores predictions back up to the original tile size. The SHM save/viz slots
    are sized from ``tensor_info[name].shape`` (the model's processed shape after
    the merge), which would be too small for the restored full-tile outputs and
    overflow the raw device->host memcpy. Here we enlarge the spatial dims of each
    DENSE buffer tensor to the DB per-table maxima (``stats.max_{z,y,x}_size``),
    which bounds the largest tile across the batch. A no-op when the model already
    runs at full tile (maxima <= current shape).

    Identifies dense tensors via :func:`dense_buffer_formats` (save-dense via
    ``annotation_type``, viz-only dense via ``tensor_info[...].kind``) and uses
    their ``data_format`` to locate the Z/Y/X axes in the shape tuple.

    Mutates ``output_metadata`` IN PLACE (the caller passes the live config node, so
    the enlarged shapes are visible to every downstream consumer of that node).
    """
    tensor_info = output_metadata.get("tensor_info", {}) or {}
    buffer_tensors = list(output_metadata.get("buffer_tensors", []) or [])

    axis_max = {
        "Z": int(getattr(stats, "max_z_size", 0) or 0),
        "Y": int(getattr(stats, "max_y_size", 0) or 0),
        "X": int(getattr(stats, "max_x_size", 0) or 0),
    }

    dense_formats = dense_buffer_formats(output_metadata, layout)

    # Buffer tensors that are declared dense MUST be resizable: a dense tensor
    # missing from tensor_info would keep its (too small) processed shape and
    # overflow the raw device->host memcpy at restore time. Sparse buffer
    # tensors (no dense_formats entry) legitimately pass through untouched.
    unmapped = [n for n in buffer_tensors if n in dense_formats and n not in tensor_info]
    if unmapped:
        raise ValueError(
            f"dense buffer tensors {unmapped} have no tensor_info entry; their "
            f"SHM slots cannot be resized for full-tile restore and the raw "
            f"memcpy would overflow. Declare them in the model's "
            f"output_metadata.tensor_info."
        )

    # open_dict only applies to DictConfig (the live config node); plain dicts
    # (tests, programmatic callers) are already writable.
    ctx = open_dict(output_metadata) if isinstance(output_metadata, DictConfig) \
        else contextlib.nullcontext()
    with ctx:
        for name in buffer_tensors:
            fmt = dense_formats.get(name)
            if fmt is None:
                # Not declared dense (sparse/aux tensor): nothing to grow.
                continue
            shape = [int(s) for s in tensor_info[name]["shape"]]
            for axis, max_value in axis_max.items():
                if axis in fmt and max_value > 0:
                    idx = fmt.index(axis)
                    if idx < len(shape):
                        shape[idx] = max(shape[idx], max_value)
            tensor_info[name]["shape"] = shape


def init_output_memory_pools(
    buffer_manager: BufferManager,
    output_metadata: Dict[str, Any],
    batch_size: int,
    save: Optional[bool] = False,
    viz: Optional[bool] = False,
    save_buffer_capacity: Optional[int] = None,
    viz_buffer_capacity: Optional[int] = None,
    pin_numa_node: Optional[bool] = True,
    layout: Optional[str] = None,
    restore_stats: Optional[Any] = None,
) -> None:
    """
    Initialize output memory pools for save and viz.

    NOTE: tensor_info is configured in the model meta-arch config
    whereas save_tensors and visualize_tensors are configured in the inference config.

    output_metadata should be structured as follows:
    {
        "tensor_info": {
            "tensor_name_1": {
                "shape": tuple,
                "dtype": str,
            },
            "tensor_name_2": {
                "shape": tuple,
                "dtype": str,
            },
        },
        "save_tensors": [
            "tensor_name_1",
            "tensor_name_2",
            ...
        ],
        "visualize_tensors": [
            "tensor_name_1",
            "tensor_name_2",
            ...
        ],
        "buffer_tensors": [
            "tensor_name_1",
            "tensor_name_2",
            ...
        ],
    }
    """
    if save and save_buffer_capacity is None:
        raise ValueError("save_buffer_capacity must be provided if save is True")
    if viz and viz_buffer_capacity is None:
        raise ValueError("viz_buffer_capacity must be provided if viz is True")
    if not save and not viz:
        raise ValueError("at least one of save or viz must be True")

    # Tile-mode inference restores predictions to the full original tile; enlarge the
    # dense buffer shapes (in place) before sizing the pools so the SHM slots fit. The
    # in-place mutation of output_metadata is intentional: the caller passes the live
    # config node, so the enlarged shapes are seen by every downstream consumer (the
    # inferencer_worker + save/viz workers built from that same node). No-op / skipped
    # when restore_stats is None (model already runs at full tile).

    # Snapshot the PROCESSED (pre-enlarge) shapes: viz renders the pre-restore,
    # processed-resolution copies, so viz slots keep the processed shape while the
    # save pools grow to the full-tile restore shape.
    processed_shapes = {
        name: list(output_metadata["tensor_info"][name]["shape"])
        for name in (output_metadata.get("buffer_tensors") or ())
        if name in (output_metadata.get("tensor_info") or {})
    }

    if restore_stats is not None:
        _resize_dense_output_shapes_for_restore(output_metadata, layout, restore_stats)

    for name in (output_metadata.get("buffer_tensors") or ()):
        try:
            tensor_shape = output_metadata["tensor_info"][name]["shape"]
            tensor_dtype = output_metadata["tensor_info"][name]["dtype"]
        except KeyError as e:
            raise ValueError(f"Tensor info for {name} not found in output_metadata: {e}")

        viz_tensor_shape = processed_shapes.get(name, tensor_shape)

        if name in output_metadata["save_tensors"]:
            buffer_manager.set_buffer(
                pool_name=f"{name}_save",
                batch_size=batch_size,
                input_shape=tensor_shape,
                dtype=tensor_dtype,
                buffer_type="host_memory",
                buffer_capacity=save_buffer_capacity,
                pin_numa_node=pin_numa_node,
            )
        if name in output_metadata["visualize_tensors"]:
            buffer_manager.set_buffer(
                pool_name=f"{name}_viz",
                batch_size=batch_size,
                input_shape=viz_tensor_shape,
                dtype=tensor_dtype,
                buffer_type="host_memory",
                buffer_capacity=viz_buffer_capacity,
                pin_numa_node=pin_numa_node,
            )


class BufferManager:
    """
    Per-rank memory manager. Owns save_output pool (wraps HostMemoryBuffer).
    preproc_input, postproc_input: not implemented (deferred).
    viz_output: stub (alloc returns None when empty; non-blocking).

    Serialization: implements ``__getstate__``/``__setstate__`` so the manager
    can be passed to Ray actors.  Shared-memory handles and CUDA pinned
    pointers are process-local and are re-attached on the remote side.
    Deserialized copies are non-owning: they will close their local shm
    handles on exit but never kill the underlying ``HostMemoryBuffer`` actors.
    """

    def __init__(
        self,
        local_rank: int,
        global_rank: int,
        node_id: int,
        numa_node: int,
        rank_memory_budget_gb,
        max_concurrent_calls: int,
        safety_margin: float = 0.05,
    ):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.node_id = node_id
        self.numa_node = numa_node
        self.rank_memory_budget_gb = rank_memory_budget_gb
        self.max_concurrent_calls = max_concurrent_calls
        self._is_owner = True

        self._buffer_actors: Dict[str, ActorHandle[HostMemoryBuffer]] = {}
        self._buffer_cfgs: Dict[str, Dict[str, Any]] = {}
        self._buffer_shms: Dict[str, shared_memory.SharedMemory] = {}
        self._pinned_ptrs: Dict[str, int] = {}
        self._free_refs: List[Any] = []   # outstanding put_free refs (see free_slot)
        self._current_memory_usage_bytes = 0
        self._max_memory_usage_bytes = int(rank_memory_budget_gb * 2**30 * (1 - safety_margin))

        atexit.register(self.shutdown)

    # -- Serialization --------------------------------------------------------

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # SharedMemory handles, CUDA pointers, and pending free refs are process-local
        del state["_buffer_shms"]
        del state["_pinned_ptrs"]
        state["_free_refs"] = []
        state["_is_owner"] = False
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._pinned_ptrs = {}
        self._buffer_shms = {}
        self._free_refs = []
        for pool_name, cfg in list(self._buffer_cfgs.items()):
            try:
                shm = attach_shared_memory(cfg["name"])
            except FileNotFoundError:
                # The pickled segment name is stale (the owner recreated the
                # pool since this manager was serialized). Revalidate against
                # the LIVE actor rather than attaching blind; a dead actor
                # makes ray.get raise loudly instead of a confusing ENOENT.
                actor = self._buffer_actors.get(pool_name)
                if actor is None:
                    raise RuntimeError(
                        f"pool {pool_name!r}: shared memory segment "
                        f"{cfg['name']!r} is gone and no actor handle is "
                        f"available to revalidate against."
                    )
                fresh = ray.get(actor.get_config.remote(), timeout=30)
                self._buffer_cfgs[pool_name] = fresh
                shm = attach_shared_memory(fresh["name"])
            self._buffer_shms[pool_name] = shm
        atexit.register(self._cleanup_shms)

    def pin_buffers(self) -> None:
        """Page-lock all registered shared memory buffers with cudaHostRegister.

        Must be called from a process that owns a CUDA context (e.g. the
        inference worker).  Mirrors the pattern used in CollatorActor for H2D.
        """
        # cupy is imported lazily: pinning only happens in CUDA-owning workers,
        # while this module is imported by every loader/collator/driver process.
        from cupy.cuda import runtime as cudart

        for pool_name, shm in self._buffer_shms.items():
            if pool_name in self._pinned_ptrs:
                continue
            base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
            cudart.hostRegister(base_ptr, shm.size, 0)
            self._pinned_ptrs[pool_name] = base_ptr
            logger.info(f"Pinned shared memory for pool {pool_name} ({shm.size} bytes)")

    def _unpin_ptr(self, pool_name: str, ptr: int) -> None:
        """cudaHostUnregister one pinned base pointer (lazy cupy import)."""
        from cupy.cuda import runtime as cudart

        cudart.hostUnregister(ptr)
        logger.info(f"Unpinned shared memory for pool {pool_name}")

    def unpin_buffers(self) -> None:
        """Unregister all page-locked shared memory buffers."""
        # getattr: __del__ can run on a partially-constructed instance
        for pool_name, ptr in list(getattr(self, "_pinned_ptrs", {}).items()):
            try:
                self._unpin_ptr(pool_name, ptr)
            except Exception as e:
                logger.exception(f"Failed to unpin buffer {pool_name}: {e}")
        if hasattr(self, "_pinned_ptrs"):
            self._pinned_ptrs.clear()

    def _cleanup_shms(self) -> None:
        """Clean up the BufferManager."""
        if not hasattr(self, "_buffer_shms"):
            return  # partially-constructed instance (__del__ during tests/init failure)
        self.unpin_buffers()
        for pool_name, buffer_shm in self._buffer_shms.items():
            try:
                buffer_shm.close()
            except Exception as e:
                logger.exception(f"Failed to close buffer shared memory {pool_name}: {e}")
        try:
            self._buffer_shms.clear()
        except Exception as e:
            logger.exception(f"Failed to clear buffer shared memory: {e}")
        try:
            self._buffer_actors.clear()
        except Exception as e:
            logger.exception(f"Failed to clear buffer actors: {e}")
        try:
            self._buffer_cfgs.clear()
        except Exception as e:
            logger.exception(f"Failed to clear buffer configs: {e}")

    def __del__(self) -> None:
        """Clean up the BufferManager."""
        self._cleanup_shms()

    def set_buffer(
        self,
        pool_name: str,
        batch_size: int,
        input_shape: tuple,
        dtype: str,
        buffer_type: str,
        buffer_capacity: int,
        pin_numa_node: bool,
    ) -> Tuple[ActorHandle[HostMemoryBuffer], Dict[str, Any]]:
        """
        Set a buffer for a given pool.
        """
        if pool_name in self._buffer_actors:
            raise ValueError(f"Pool {pool_name} already exists. Use get_buffer instead.")

        slot_bytes = get_slot_bytes((int(batch_size), *tuple(input_shape)), dtype)
        total_bytes = slot_bytes * buffer_capacity
        if total_bytes + self._current_memory_usage_bytes > self._max_memory_usage_bytes:
            raise ValueError(
                f"Allocating pool {pool_name!r} ({total_bytes} bytes) would exceed the memory budget:"
                f"current usage {self._current_memory_usage_bytes} + {total_bytes} > max {self._max_memory_usage_bytes}"
            )

        try:
            buffer_dtype = NUMPY_DTYPES[dtype].value if isinstance(dtype, str) else dtype
            buffer_actor, buffer_cfg = set_buffers(
                local_rank=self.local_rank,
                global_rank=self.global_rank,
                numa_node=self.numa_node,
                dtype=buffer_dtype,
                batch_size=batch_size,
                input_shape=input_shape,
                buffer_type=buffer_type,
                buffer_capacity=buffer_capacity,
                pin_numa_node=pin_numa_node,
                node_id=self.node_id,
                pool_name=pool_name,
                max_concurrent_calls=self.max_concurrent_calls,
            )
            self._buffer_actors[pool_name] = buffer_actor
            self._buffer_cfgs[pool_name] = buffer_cfg
            self._buffer_shms[pool_name] = attach_shared_memory(buffer_cfg["name"])
        except Exception as e:
            raise RuntimeError(f"Failed to set buffer for pool {pool_name}: {e}")
        
        self._current_memory_usage_bytes += total_bytes
    
        return buffer_actor, buffer_cfg

    def get_buffer(self, pool_name: str) -> ActorProxy[HostMemoryBuffer]:
        """
        Get a buffer for a given pool.
        """
        if pool_name in self._buffer_actors:
            return self._buffer_actors[pool_name]
        else:
            buffer_actor = get_buffers(
                type="host_memory",
                global_rank=self.global_rank,
                local_rank=self.local_rank,
                node_id=self.node_id,
                pool_name=pool_name,
                numa_node=self.numa_node,
            )
            buffer_cfg = ray.get(buffer_actor.get_config.remote())
            additional_memory_usage_bytes = get_slot_bytes(buffer_cfg["batch_shape"], buffer_cfg["dtype"]) * buffer_cfg["capacity"]
            if additional_memory_usage_bytes + self._current_memory_usage_bytes > self._max_memory_usage_bytes:
                raise ValueError(f"Additional memory usage {additional_memory_usage_bytes} exceeds max memory usage {self._max_memory_usage_bytes}")
            self._buffer_actors[pool_name] = buffer_actor
            self._buffer_cfgs[pool_name] = buffer_cfg
            self._buffer_shms[pool_name] = attach_shared_memory(buffer_cfg["name"])
            self._current_memory_usage_bytes += additional_memory_usage_bytes
            return buffer_actor
    
    def remove_buffer(self, pool_name: str) -> None:
        """
        Remove a buffer for a given pool. Owner instances also tear down the
        underlying detached actor (release the segment, then kill).
        """
        if pool_name not in self._buffer_actors:
            raise ValueError(f"Pool {pool_name} does not exist. Use set_buffer instead.")
        self._current_memory_usage_bytes -= get_slot_bytes(self._buffer_cfgs[pool_name]["batch_shape"], self._buffer_cfgs[pool_name]["dtype"]) * self._buffer_cfgs[pool_name]["capacity"]
        # Unregister CUDA pinning BEFORE closing/unlinking the segment: the
        # driver must release the page-lock while the mapping still exists
        # (unregistering an unmapped VA is swallowed but leaves stale pinning
        # records if the range is remapped in-process). pop() keeps this
        # idempotent against _cleanup_shms' bulk unpin fallback.
        ptr = self._pinned_ptrs.pop(pool_name, None)
        if ptr is not None:
            try:
                self._unpin_ptr(pool_name, ptr)
            except Exception as e:
                logger.warning(f"cudaHostUnregister failed for {pool_name}: {e}")
        self._buffer_shms.pop(pool_name).close()
        self._buffer_cfgs.pop(pool_name)
        actor = self._buffer_actors.pop(pool_name)
        if self._is_owner:
            # Detached actors outlive drivers; without this the segment AND the
            # depleted free-queue state leak into the next run (silent no-op
            # saves / deadlocks). Unlink from inside first -- atexit does not
            # run under ray.kill.
            try:
                ray.get(actor.release.remote(), timeout=30)
            except Exception as e:
                logger.warning(f"release() failed for pool {pool_name!r}: {e}")
            try:
                ray.kill(actor, no_restart=True)
            except Exception as e:
                logger.warning(f"ray.kill failed for pool {pool_name!r}: {e}")
    
    def slot_info_to_view(self, slot_info: Dict[str, Any]) -> np.ndarray:
        """
        Convert slot info to a view of the slot.
        """
        pool_name = parse_buffer_name(slot_info["actor_name"])["pool_name"]
        self.get_buffer(pool_name)
        return slot_info_to_view(slot_info, self._buffer_shms[pool_name])

    def free_slot(self, slot_info: Dict[str, Any]) -> None:
        """
        Free a slot (non-blocking). 
        """
        try:
            pool_name = parse_buffer_name(slot_info["actor_name"])["pool_name"]
            buffer_actor = self.get_buffer(pool_name)
            ref = buffer_actor.put_free.remote(slot_info["slot"])
            self._free_refs.append(ref)
        except Exception as e:
            logger.error(
                f"Failed to free slot {slot_info['slot']} for pool "
                f"{slot_info.get('actor_name', slot_info.get('name'))}: {e}"
            )
            return
        # Reap completed frees without blocking. Errors are logged (not raised):
        # free_slot runs inside teardown finally-loops that must free EVERY
        # remaining slot; drain_free_refs() is the raising reap point.
        if len(self._free_refs) >= 64:
            done, self._free_refs = ray.wait(
                self._free_refs, num_returns=len(self._free_refs), timeout=0
            )
            for ref in done:
                try:
                    ray.get(ref)
                except Exception as e:
                    logger.error(f"put_free failed (double-free?): {e}")

    def drain_free_refs(self) -> None:
        """Blocking reap of every outstanding put_free ref; RAISES on the first
        failed free (double-free etc.) -- call from shutdown/finalize/tests to
        make producer-side free bugs observable instead of background log noise."""
        refs, self._free_refs = self._free_refs, []
        if refs:
            ray.get(refs)

    def enable_metrics_collection(self) -> None:
        """Enable metrics collection for the BufferManager."""
        for pool_name, buffer_actor in self._buffer_actors.items():
            ray.logger.info(f"[BufferManager] Enabling metrics collection for pool {pool_name}")
            ray.get(buffer_actor.enable_metrics_collection.remote())

    def disable_metrics_collection(self) -> None:
        """Disable metrics collection for the BufferManager."""
        for pool_name, buffer_actor in self._buffer_actors.items():
            ray.logger.info(f"[BufferManager] Disabling metrics collection for pool {pool_name}")
            ray.get(buffer_actor.disable_metrics_collection.remote())

    def get_metrics(self) -> Dict[str, Dict[str, float | int | list[float | int]]]:
        """Get the metrics for the BufferManager.

        Best-effort: a buffer actor may already be torn down (e.g. ``ray.kill``
        during test cleanup or shutdown, which skips its ``atexit`` handlers), so
        a dead/unreachable actor is skipped with a warning rather than raising.
        Callers (the metrics hook and :meth:`shutdown`) must not crash — and
        ``shutdown`` must still proceed to release resources — when only a subset
        of pools is alive.
        """
        metrics = {}
        for pool_name, buffer_actor in self._buffer_actors.items():
            try:
                metrics[pool_name] = ray.get(buffer_actor.get_metrics.remote())
            except Exception as e:
                ray.logger.warning(
                    f"[BufferManager] metrics unavailable for pool {pool_name} "
                    f"(actor dead/unreachable): {e}"
                )
        return metrics
    
    def log_metrics_at_shutdown(self) -> None:
        """Log BufferManager metrics at shutdown."""
        metrics = self.get_metrics()
        for pool_name, pool_metrics in metrics.items():
            ray.logger.info(
                f"[BufferManager] Shutdown metrics for pool {pool_name}: {pool_metrics}"
            )
            
    def shutdown(self) -> None:
        """Log final metrics and release resources.

        Owner instances tear down the underlying Ray actors.
        Non-owner (deserialized) copies only close local shm handles.
        """
        if not self._is_owner:
            self._cleanup_shms()
            return
        try:
            # Reap outstanding frees BEFORE killing the actors: a failed free
            # (double-free) surfaces here instead of dying with the actor.
            try:
                self.drain_free_refs()
            except Exception as e:
                logger.error(f"Outstanding slot frees failed during shutdown: {e}")
            self.log_metrics_at_shutdown()
            for pool_name in list(self._buffer_actors.keys()):
                self.remove_buffer(pool_name)
        except Exception as e:
            logger.exception(f"Exception occurred while shutting down BufferManager (rank {self.global_rank}, node {self.node_id}, numa {self.numa_node}): {e}")
        finally:
            self._cleanup_shms()
