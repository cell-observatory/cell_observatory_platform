import os
import sys
import time
import logging
from typing import List, Optional, Dict, Any, Callable

import ctypes
import cupy as cp
import numpy as np
from cupy.cuda import runtime as cudart

import tensorstore as ts
from multiprocessing import shared_memory

from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

from data.io import read_zarr
from data.data_types import (NUMPY_DTYPES, 
                             TENSORSTORE_DTYPES, 
                             TORCH_DTYPES)

import ray
import pyarrow as pa

import torch
from torch.utils.data import random_split

from utils.context import (process_rank, 
                           local_rank,
                           get_world_size,
                           pin_to_numa_node)
from utils.profiling import pprof_func, pprof_class
from data.datasets.buffers import get_buffers, DeviceMemoryBuffer

from training.helpers import record_dataset_len

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -------- -------- Collators -------- --------


@pprof_class
class CollatorActor:
    def __init__(self, 
                 batch_size: int,
                 input_shape: tuple,
                 device_buffer_capacity: int,
                 dtype: str,
                 buffer_dtype: str,
                 pin_numa_node: bool,
                 pin_pages: bool,
                 debug: bool = False
    ):
        self.batch_size = batch_size
        self.input_shape = tuple(input_shape)
        self.device_buffer_capacity = device_buffer_capacity

        if pin_numa_node:
            pin_to_numa_node(local_rank())

        self.out_dtype = TORCH_DTYPES[dtype].value \
            if isinstance(dtype, str) else dtype
        self.buffer_dtype = NUMPY_DTYPES[buffer_dtype].value \
            if isinstance(buffer_dtype, str) else buffer_dtype

        self.host_buffer_actor = get_buffers(type="host_memory", 
                                             local_rank = local_rank(), 
                                             global_rank=process_rank())
        cfg = ray.get(self.host_buffer_actor.get_config.remote())
        self.slot_bytes = int(cfg["slot_bytes"])
        self.batch_shape = tuple(cfg["batch_shape"])
        self.capacity = int(cfg["capacity"])

        self._shm = shared_memory.SharedMemory(name=cfg["name"])
        
        self.pin_pages = pin_pages
        if pin_pages:
            base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm.buf))
            self.host_buffer_ptr = base_ptr
            cp.cuda.runtime.hostRegister(base_ptr, self.slot_bytes * self.capacity, 0)
            self._pinned = True
        else:
            self._pinned = False

        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.copy_stream = torch.cuda.Stream(device=self.device)
        
        self.device_buffer = DeviceMemoryBuffer(
            name=f"device_buffer_rank_{process_rank()}",
            capacity=self.device_buffer_capacity,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            dtype=dtype
        )

        logger.info(f"CollatorActor on rank {process_rank()} using host shared memory buffer "
                    f"with name {cfg['name']} and capacity {cfg['capacity']} and HostMemoryBuffer "
                    f"with pin_pages={self._pinned}")

        self.debug = debug

    def __del__(self):
        try:
            if getattr(self, "_pinned", False) and self.host_buffer_ptr is not None:
                cp.cuda.runtime.hostUnregister(self.host_buffer_ptr)
            if hasattr(self, "_shm"):
                self._shm.close()
        except Exception:
            pass

    def copy_h2d(self, dst, src):
        assert src.flags["C_CONTIGUOUS"], "src must be contiguous"
        # __array_interface__ protocol: data field is a
        #  2-tuple whose first argument is a Python integer that points
        # to the data-area storing the array contents
        # see: https://numpy.org/doc/stable/reference/arrays.interface.html
        src_ptr = ctypes.c_void_p(src.__array_interface__["data"][0])
        # see: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html
        dst_ptr = ctypes.c_void_p(dst.data_ptr())
        # cupy function handle: 
        # cupy.cuda.runtime.memcpyAsync(intptr_t dst, intptr_t src, size_t size, int kind, intptr_t stream)
        cudart.memcpyAsync(dst_ptr.value, src_ptr.value, src.nbytes,
                                 cudart.memcpyHostToDevice, self.copy_stream.cuda_stream)

    def __call__(self, batch):        
        host_buffer_idx = int(batch["buffer_idx"][0])
        h_view = np.ndarray(self.batch_shape, dtype=self.buffer_dtype,
                            buffer=self._shm.buf, offset=host_buffer_idx * self.slot_bytes)

        device_buffer_idx = self.device_buffer.get_free()
        dst_device = self.device_buffer.device_buffers[device_buffer_idx]

        with torch.cuda.stream(self.copy_stream):
            self.copy_h2d(dst=dst_device, src=h_view)

        metainfo = {
            "host_buffer_idx": host_buffer_idx,
            "device_buffer_idx": device_buffer_idx
        }

        if self.debug:
            # NOTE: for testing only, put_free(idx) otherwise called by hooks in 
            #       training loop, see training/hooks.py:FreeDeviceBufferHook
            ray.get(self.host_buffer_actor.put_free.remote(host_buffer_idx))
            self.device_buffer.put_free(device_buffer_idx)

        return {"data_tensor": dst_device, "metainfo": metainfo}


# -------- -------- Loader Actors -------- --------


@pprof_class
class LoaderActor:
    def __init__(self, 
                 local_rank: int,
                 global_rank: int,
                 batch_size: int,
                 input_layout: str,
                 context_spec: Dict[str, Any],
                 dtype: str = "fp16",
                 buffer_dtype: str = "uint16",
                 pin_numa_node: bool = True,
                 with_batched_api: bool = True,
                 channels_subset: Optional[List[int]] = None
):
        if pin_numa_node:
            pin_to_numa_node(local_rank)

        # input data layout
        self.channels_subset = list(channels_subset) \
            if channels_subset is not None else None
        self.input_layout = input_layout.upper()
        self.batch_size = batch_size

        # dtypes
        self.dtype = TENSORSTORE_DTYPES[dtype].value \
            if isinstance(dtype, str) else dtype
        if self.dtype == TENSORSTORE_DTYPES.bf16.value:
            self.dtype = TENSORSTORE_DTYPES.fp16.value
        self.buffer_dtype = NUMPY_DTYPES[buffer_dtype].value \
            if isinstance(buffer_dtype, str) else buffer_dtype

        # tensorstore 
        self.with_batched_api = with_batched_api
        self.ctx = ts.Context(context_spec)
        self._handles= {}

        # memory buffer 
        self.buffer_actor = get_buffers(type="host_memory", 
                                        global_rank=global_rank, 
                                        local_rank=local_rank)
        cfg = ray.get(self.buffer_actor.get_config.remote())
        self.slot_bytes = int(cfg["slot_bytes"])
        self.batch_shape = tuple(cfg["batch_shape"])
        self._shm = shared_memory.SharedMemory(name=cfg["name"])

        logger.info(f"LoaderActor on rank {global_rank} using shared memory buffer "
                    f"with name {cfg['name']} and capacity {cfg['capacity']}")

    def _slice_hypercube(self, data_tensor, meta: Dict[str, Any], ts_batch=None):
        t = slice(meta["time_start"], meta["time_start"] + meta["time_size"])
        z = slice(meta["z_start"], meta["z_start"] + meta["cube_size"])
        y = slice(meta["y_start"], meta["y_start"] + meta["cube_size"])
        x = slice(meta["x_start"], meta["x_start"] + meta["cube_size"])

        if self.channels_subset is not None:
            view = data_tensor[t, z, y, x, self.channels_subset]
        else:
            c = slice(0, meta["channel_size"])
            view = data_tensor[t, z, y, x, c]

        return view

    def _get_handle(self, path: str):
        h = self._handles.get(path)
        if h is None:
            h = read_zarr(path, dtype=self.dtype, context=self.ctx)
            self._handles[path] = h
        return h
    
    def __call__(self, batch):
        buffer = ray.get(self.buffer_actor.get_free.remote())
        dst = np.ndarray(self.batch_shape, 
                         dtype=self.buffer_dtype,
                         buffer=self._shm.buf,
                         offset=buffer["slot"] * self.slot_bytes)

        write_futs = []
        with ts.Batch() as b:
            for i in range(self.batch_size):
                p = os.path.join(batch["server_folder"][i],
                                batch["output_folder"][i],
                                batch["tile_name"][i])
                meta = {
                    "time_start":  batch["time_start"][i],
                    "time_size": batch["time_size"][i],
                    "z_start": batch["z_start"][i],
                    "y_start": batch["y_start"][i],
                    "x_start": batch["x_start"][i],
                    "cube_size": batch["cube_size"][i],
                    "channel_size": batch["channel_size"][i],
                }
                src_view = self._slice_hypercube(self._get_handle(p), meta=meta, ts_batch=b)
                write_futs.append(ts.array(dst[i]).write(src_view))

        for f in write_futs:
            f.result()

        batch["buffer_name"] = np.array([buffer["name"]] * self.batch_size)
        batch["buffer_idx"] = np.full((self.batch_size,), buffer["slot"], dtype=np.int32)
        return batch


# -------- -------- dataset helpers / API -------- -------- --------


def set_data_context(cfg: DictConfig):
    ctx = ray.data.DataContext.get_current()
    ctx.use_arrow_tensor_v2 = cfg.datasets.use_arrow_tensor_v2
    ctx.execution_options.locality_with_output = cfg.datasets.locality_with_output


def get_context_spec(cfg: DictConfig) -> Dict[str, Any]:
    ts_ctx = OmegaConf.to_container(cfg.datasets.context, resolve=True)
    ctx_spec = {k: v for k, v in ts_ctx.items() if v is not None}
    return ctx_spec


def get_dataset_ray(cfg: DictConfig,
                    indices: Optional[List[int]],
                    database: Optional[Any] = None
):
    if cfg.datasets.channels_subset is not None:
        num_channels = cfg.datasets.input_shape[cfg.dataset_layout_order.index("C")]
        assert len(list(cfg.datasets.channels_subset)) == num_channels, \
            f"channels_subset length {len(cfg.datasets.channels_subset)} " \
            f"does not match number of channels {num_channels} in input_shape {cfg.datasets.input_shape}"

    set_data_context(cfg)
    ctx_spec = get_context_spec(cfg)

    table = pa.table(database.hypercubes_dataframe.iloc[indices] \
                        if indices is not None else database.hypercubes_dataframe)
    
    dataset = ray.data.from_arrow(table)
    dataset = dataset.split(n = get_world_size(), equal = True)[process_rank()]
    
    if cfg.datasets.drop_last_policy:
        B = cfg.clusters.batch_size_per_gpu
        dataset = dataset.limit((dataset.count() // B) * B)

    dataset_len = dataset.count()
    
    dataset = dataset.repartition(
        target_num_rows_per_block=cfg.datasets.rows_per_block,
        shuffle=False
    )
    dataset = dataset.map_batches(
        LoaderActor,
        batch_size=cfg.clusters.batch_size_per_gpu,
        batch_format="numpy",
        fn_constructor_kwargs={
            "batch_size": cfg.clusters.batch_size_per_gpu,
            "context_spec": ctx_spec,
            "with_batched_api": cfg.datasets.with_batched_api,
            "dtype": cfg.dataset_dtype,
            "buffer_dtype": cfg.storage_dtype,
            "pin_numa_node": cfg.datasets.pin_numa_node,
            "input_layout": cfg.datasets.dataset.input_layout.value,
            "channels_subset": cfg.datasets.channels_subset,
            "local_rank": local_rank(),
            "global_rank": process_rank()
        },
        concurrency=(cfg.datasets.num_actors_min, 
                        cfg.datasets.num_actors_max)
    )

    return dataset, dataset_len


def get_dataloader_ray(cfg: DictConfig,
                       batch_size: int,
                       collate_fn: Optional[Callable],
                       drop_last: bool = True
):
    db = instantiate(cfg.datasets.databases)
    dataset_len = len(db.hypercubes_dataframe)

    if cfg.datasets.split is not None:
        val_size = round(dataset_len * cfg.datasets.split)
        train_subset, val_subset = random_split(
            range(dataset_len),
            lengths=[dataset_len - val_size, val_size]
        )
        train_indices, val_indices = train_subset.indices, val_subset.indices

        train_dataset, train_dataset_len = get_dataset_ray(cfg, indices=train_indices, database=db)
        val_dataset, val_dataset_len = get_dataset_ray(cfg, indices=val_indices, database=db)

        record_dataset_len(cfg, train_dataset_len, val_dataset_len)

        train_dataloader = train_dataset.iterator()._iter_batches(
            batch_size=batch_size,
            _finalize_fn=collate_fn,
            batch_format="numpy"
        )
        val_dataloader = val_dataset.iterator()._iter_batches(
            batch_size=batch_size,
            _finalize_fn=collate_fn,
            batch_format="numpy"
        )
        return train_dataloader, val_dataloader

    else:
        train_dataset, train_dataset_len = get_dataset_ray(cfg, indices=None, database=db)
        record_dataset_len(cfg, train_dataset_len, 0)

        train_dataloader = train_dataset.iterator()._iter_batches(
            batch_size=batch_size,
            _finalize_fn=collate_fn,
            batch_format="numpy"
        )
        return train_dataloader, None