from __future__ import annotations
import sys
import atexit
import asyncio
import logging
from typing import Any, Dict, Optional, Tuple
from typing_extensions import Buffer

import cupy as cp
import numpy as np
from dataclasses import dataclass, field
import ray
from ray.actor import ActorHandle, ActorProxy
import torch
from collections import defaultdict
import queue
from threading import Lock
from multiprocessing import shared_memory
import time

from cell_observatory_platform.data.data_types import NUMPY_DTYPES, TORCH_DTYPES
from cell_observatory_platform.utils.context import (
    local_rank, 
    node_id,
    bind_current_process_to_node
)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def slot_info_to_view(slot_info: Dict[str, Any]) -> np.ndarray:
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
    buffer = shared_memory.SharedMemory(name=slot_info["name"]).buf
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
        self._metrics: Dict[str, float | int] = {
            "get_free_count": 0.0,
            "get_free_wait_time_s": 0.0,
            "put_free_count": 0.0,
            "put_free_wait_time_s": 0.0,
            "try_get_free_count": 0.0,
            "try_get_free_wait_time_s": 0.0,
            "try_get_free_drops": 0,
            "in_use_current": 0,
            "capacity": self.cap,
            "slot_bytes": self.slot_bytes,
        }

        self.free = asyncio.Queue(self.cap)
        for i in range(self.cap):
            self.free.put_nowait(i)

        atexit.register(self._cleanup)

    def _cleanup(self):
        try: 
            self._shm.close()
            self._shm.unlink()
        except FileNotFoundError: 
            pass

    async def get_free(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        slot = await self.free.get()
        t1 = time.perf_counter()
        self._metrics["get_free_count"] += 1
        self._metrics["get_free_wait_time_s"] += t1 - t0
        self._metrics["in_use_current"] += 1
        return {"slot": slot, "name": self.name, "slot_bytes": self.slot_bytes,
                "batch_shape": self.batch_shape, "dtype": self.dtype, "capacity": self.cap}

    def try_get_free(self) -> Optional[Dict[str, Any]]:
        """Non-blocking get. Returns slot info dict or None if queue empty."""
        t0 = time.perf_counter()
        try:
            slot = self.free.get_nowait()
        except asyncio.QueueEmpty:
            return None
        t1 = time.perf_counter()
        self._metrics["try_get_free_count"] += 1
        self._metrics["try_get_free_wait_time_s"] += t1 - t0
        if slot is None:
            self._metrics["try_get_free_drops"] += 1
        else:
            self._metrics["in_use_current"] += 1
        return {"slot": slot, "name": self.name, "slot_bytes": self.slot_bytes,
                "batch_shape": self.batch_shape, "dtype": self.dtype, "capacity": self.cap}

    async def put_free(self, slot: int):
        t0 = time.perf_counter()
        await self.free.put(int(slot))
        t1 = time.perf_counter()
        self._metrics["put_free_count"] += 1
        self._metrics["put_free_wait_time_s"] += t1 - t0
        self._metrics["in_use_current"] -= 1
        return True

    def get_config(self):
        return {"capacity": self.cap, "name": self.name, "slot_bytes": self.slot_bytes,
                "batch_shape": self.batch_shape, "dtype": self.dtype}

    def get_metrics(self) -> Dict[str, float | int]:
        return self._metrics.copy()

    def clear_metrics(self) -> None:
        self._metrics = {
            "get_free_count": 0.0,
            "get_free_wait_time_s": 0.0,
            "put_free_count": 0.0,
            "put_free_wait_time_s": 0.0,
            "try_get_free_count": 0.0,
            "try_get_free_wait_time_s": 0.0,
            "try_get_free_drops": 0,
            "in_use_current": 0,
            "capacity": self.cap,
            "slot_bytes": self.slot_bytes,
        }


def set_buffers(
    local_rank: int,
    global_rank: int,
    numa_node: int,
    dtype: str,
    batch_size: tuple,
    input_shape: tuple,
    buffer_type: str,
    buffer_capacity: int,
    pin_to_numa_node: bool,
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
        name = f"host_pinned_shm_buffer_{pool_name}_numa_{numa_node}_rank_{global_rank}"
        namespace = f"buffers_node_{node_id}"
        expected_batch_shape = (int(batch_size[0]), *tuple(input_shape))

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
            pin_numa_node=pin_to_numa_node,
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


def init_output_memory_pools(
    buffer_manager: BufferManager, 
    output_metadata: Dict[str, Any], 
    batch_size: int,
    save: Optional[bool] = False,
    viz: Optional[bool] = False,
    save_buffer_capacity: Optional[int] = None,
    viz_buffer_capacity: Optional[int] = None,
    pin_to_numa_node: Optional[bool] = True,
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
    }
    """
    if save and save_buffer_capacity is None:
        raise ValueError("save_buffer_capacity must be provided if save is True")
    if viz and viz_buffer_capacity is None:
        raise ValueError("viz_buffer_capacity must be provided if viz is True")
    if not save and not viz:
        raise ValueError("at least one of save or viz must be True")
    
    for name in output_metadata["save_tensors"]:
        try:
            tensor_shape = output_metadata["tensor_info"][name]["shape"]
            tensor_dtype = output_metadata["tensor_info"][name]["dtype"]
        except KeyError as e:
            raise ValueError(f"Tensor info for {name} not found in output_metadata: {e}")
        buffer_manager.set_buffer(
            pool_name=f"{name}_save",
            batch_size=batch_size,
            input_shape=tensor_shape,
            dtype=tensor_dtype,
            buffer_type="host_memory",
            buffer_capacity=save_buffer_capacity,
            pin_to_numa_node=pin_to_numa_node,
        )
    for name in output_metadata["visualize_tensors"]:
        try:
            tensor_shape = output_metadata["tensor_info"][name]["shape"]
            tensor_dtype = output_metadata["tensor_info"][name]["dtype"]
        except KeyError as e:
            raise ValueError(f"Tensor info for {name} not found in output_metadata: {e}")
        buffer_manager.set_buffer(
            pool_name=f"{name}_viz",
            batch_size=batch_size,
            input_shape=tensor_shape,
            dtype=tensor_dtype,
            buffer_type="host_memory",
            buffer_capacity=viz_buffer_capacity,
            pin_to_numa_node=pin_to_numa_node,
        )



class BufferManager:
    """
    Per-rank memory manager. Owns save_output pool (wraps HostMemoryBuffer).
    preproc_input, postproc_input: not implemented (deferred).
    viz_output: stub (alloc returns None when empty; non-blocking).
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

        self._buffer_actors: Dict[str, ActorHandle[HostMemoryBuffer]] = {}
        
        self._current_memory_usage_bytes = 0
        self._max_memory_usage_bytes = int(rank_memory_budget_gb * 2**30 * (1 - safety_margin))

        atexit.register(self.shutdown)

    def set_buffer(
        self,
        pool_name: str,
        batch_size: int,
        input_shape: tuple,
        dtype: str,
        buffer_type: str,
        buffer_capacity: int,
        pin_to_numa_node: bool,
    ) -> Tuple[ActorHandle[HostMemoryBuffer], Dict[str, Any]]:
        """
        Set a buffer for a given pool.
        """
        if pool_name in self._buffer_actors:
            raise ValueError(f"Pool {pool_name} already exists")

        slot_bytes = get_slot_bytes(input_shape, dtype)
        total_bytes = slot_bytes * buffer_capacity
        if total_bytes > self._max_memory_usage_bytes:
            raise ValueError(f"Total bytes {total_bytes} exceeds max memory usage {self._max_memory_usage_bytes}")

        try:
            buffer_dtype = NUMPY_DTYPES[dtype].value if isinstance(dtype, str) else dtype
            buffer_actor, buffer_cfg = set_buffers(
                local_rank=self.local_rank,
                global_rank=self.global_rank,
                numa_node=self.numa_node,
                dtype=buffer_dtype,
                batch_size=(batch_size,),
                input_shape=input_shape,
                buffer_type=buffer_type,
                buffer_capacity=buffer_capacity,
                pin_to_numa_node=pin_to_numa_node,
                node_id=self.node_id,
                pool_name=pool_name,
                max_concurrent_calls=self.max_concurrent_calls,
            )
            self._buffer_actors[pool_name] = buffer_actor
            
        except Exception as e:
            raise RuntimeError(f"Failed to set buffer for pool {pool_name}: {e}")
        
        self._current_memory_usage_bytes += total_bytes
    
        return buffer_actor, buffer_cfg

    def get_buffer(self, pool_name: str) -> ActorProxy[HostMemoryBuffer]:
        """
        Get a buffer for a given pool.
        """
        return get_buffers(
            type="host_memory",
            global_rank=self.global_rank,
            local_rank=self.local_rank,
            node_id=self.node_id,
            pool_name=pool_name,
            numa_node=self.numa_node,
        )
    
    def free_slot(self, slot_info: Dict[str, Any]) -> None:
        """
        Free a slot.
        """
        buffer_actor = ray.get_actor(slot_info["name"], namespace=f"buffers_node_{self.node_id}")
        buffer_actor.put_free.remote(slot_info["slot"])
    

    def get_metrics(self) -> Dict[str, Dict[str, float | int]]:
        """Get the metrics for the BufferManager."""
        metrics = {}
        for pool_name, buffer_actor in self._buffer_actors.items():
            metrics[pool_name] = ray.get(buffer_actor.get_metrics.remote())
        return metrics
    
    def clear_metrics(self) -> None:
        """Flush the metrics for the BufferManager."""
        for pool_name, buffer_actor in self._buffer_actors.items():
            buffer_actor.clear_metrics.remote()

    def log_metrics_at_shutdown(self) -> None:
        """Log BufferManager metrics at shutdown."""
        metrics = self.get_metrics()
        for pool_name, pool_metrics in metrics.items():
            ray.logger.info(
                f"BufferManager shutdown metrics for pool {pool_name}: {pool_metrics}"
            )
            
    def shutdown(self) -> None:
        """Log final metrics and kill the underlying Ray actors."""
        self.log_metrics_at_shutdown()
