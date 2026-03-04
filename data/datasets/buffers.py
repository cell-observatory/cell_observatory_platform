import sys
import atexit
import asyncio
import logging

import cupy as cp
import numpy as np

import ray
import torch

import queue
from multiprocessing import shared_memory

from cell_observatory_platform.data.data_types import NUMPY_DTYPES, TORCH_DTYPES
from cell_observatory_platform.utils.context import (local_rank, 
                           node_id,
                           bind_current_process_to_node)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

    async def get_free(self):
        slot = await self.free.get()
        return {"slot": slot, "name": self.name, "slot_bytes": self.slot_bytes,
                "batch_shape": self.batch_shape, "dtype": self.dtype, "capacity": self.cap}

    async def put_free(self, slot: int):
        await self.free.put(int(slot))
        return True

    def get_config(self):
        return {"capacity": self.cap, "name": self.name, "slot_bytes": self.slot_bytes,
                "batch_shape": self.batch_shape, "dtype": self.dtype}


def set_buffers(local_rank: int,
                global_rank: int,
                numa_node: int,
                dtype: str,
                batch_size: tuple,
                input_shape: tuple,
                buffer_type: str,
                buffer_capacity: int,
                pin_to_numa_node: bool,
                node_id: int,
                max_concurrent_calls: int = 256
):
    if buffer_type == "host_memory":
        name = f"host_pinned_shm_buffer_numa_{numa_node}_rank_{global_rank}"

        ray.logger.info(f"Global rank {global_rank} creating host buffer actor "
                    f"on local rank {local_rank} and NUMA node {numa_node}")

        scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=node_id,
            soft=False,
        )
        buffer = HostMemoryBuffer.options(
            name=name,
            namespace=f"buffers_node_{node_id}",
            lifetime="detached",
            # allow concurrent get/put calls
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
    

def get_buffers(type: str, 
                global_rank: int, 
                local_rank: int, 
                node_id: int, 
                numa_node: int = None
):
    if type == "host_memory":
        name = f"host_pinned_shm_buffer_numa_{numa_node}_rank_{global_rank}"
        return ray.get_actor(name, namespace=f"buffers_node_{node_id}")
    else:
        raise ValueError(f"Unsupported buffer type: {type}")


def set_output_buffers(
    local_rank: int,
    global_rank: int,
    numa_node: int,
    batch_size: int,
    output_shape: tuple,
    dtype: str = "float16",
    buffer_capacity: int = 8,
    pin_to_numa_node: bool = True,
    node_id: int = 0,
    max_concurrent_calls: int = 256,
):
    """
    Create OutputBuffer for inference: shared memory for model outputs
    (batch_size, T, Z, Y, X, C_out). Metadata (roi, tile_name, coords, etc.)
    is passed separately with each slot reference.
    """
    name = f"output_shm_buffer_numa_{numa_node}_rank_{global_rank}"
    input_shape = output_shape

    ray.logger.info(
        f"Global rank {global_rank} creating output buffer actor "
        f"on local rank {local_rank} and NUMA node {numa_node}"
    )

    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id,
        soft=False,
    )
    buffer = HostMemoryBuffer.options(
        name=name,
        namespace=f"buffers_node_{node_id}",
        lifetime="detached",
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

    ray.logger.info(
        f"Output buffer actor '{name}' on NUMA node {numa_node} "
        f"with capacity {buffer_capacity} and batch shape {(batch_size, *input_shape)}"
    )

    return buffer, buffer_cfg


def get_output_buffers(
    global_rank: int,
    local_rank: int,
    node_id: int,
    numa_node: int,
):
    name = f"output_shm_buffer_numa_{numa_node}_rank_{global_rank}"
    return ray.get_actor(name, namespace=f"buffers_node_{node_id}")