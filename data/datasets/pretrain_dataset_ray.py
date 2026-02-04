import ctypes
import logging
import os
import sys
from multiprocessing import shared_memory
from queue import Queue
from threading import Thread
from typing import Any, Callable, Dict, List, Literal, Optional

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import ray
import tensorstore as ts
import torch
import ujson
from cupy.cuda import runtime as cudart
from hydra.utils import get_method, instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import random_split

from cell_observatory_platform.data.data_types import NUMPY_DTYPES, TENSORSTORE_DTYPES, TORCH_DTYPES
from cell_observatory_platform.data.datasets.buffers import DeviceMemoryBuffer, get_buffers
from cell_observatory_platform.data.io import read_zarr
from cell_observatory_platform.data.structures import convert_bbox_format, mask_ids_to_masks
from cell_observatory_platform.inference.utils import tile_owner
from cell_observatory_platform.training.helpers import get_data_dim, get_image_sizes, record_dataset_len, df_signature_polars
from cell_observatory_platform.utils.context import (
    bind_current_process_to_node,
    get_world_size,
    local_rank,
    node_id,
    process_rank,
    torch_gpu_to_numa,
)
from cell_observatory_platform.utils.profiling import pprof_class, pprof_func

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -------- -------- Collators -------- --------


@pprof_class
class FinetuneCollatorActor:
    """
    Collator Actor for finetune training:
      - Read hypercubes from shared host buffer.
      - On CPU:
          * split off mask channel,
          * build per-instance binary masks and boxes from mask_bbox_dict,
          * compute image_sizes / orig_image_sizes / padding_mask,
          * optionally apply Resize() (image + masks + boxes + padding_mask).
      - Copy resized image tensor into DeviceMemoryBuffer on GPU.
      - Move targets to GPU.

    Output:
      {
        "data_tensor": dst_device,
        "metainfo": {
           "host_buffer_idx": ...,
           "device_buffer_idx": ...,
           <all metadata columns>,
           "image_sizes": (B, 3) tensor on GPU,
           "orig_image_sizes": (B, 3) tensor on GPU,
           "padding_mask": (B, Z, Y, X) tensor on GPU,
           "targets": List[Dict[str, Tensor]]  # on GPU
        }
      }
    """

    def __init__(
        self,
        batch_size: int,
        input_shape: tuple,
        device_buffer_capacity: int,
        dtype: str,
        buffer_dtype: str,
        pin_numa_node: bool,
        pin_pages: bool,
        node_id: int,
        columns: List[str] = [
            "x_start",
            "y_start",
            "z_start",
            "time_start",
            "channel_size",
            "z_size",
            "y_size",
            "x_size",
            "time_size",
            "server_folder",
            "output_folder",
            "tile_name",
            "prepared_id",
            "mask_bbox_dict",
        ],
        input_format: Literal["ZYXC", "TZYXC"] = "ZYXC",
        mask_idx: int = -1,
        bbox_data_format: str = "zyxzyx",
        bbox_output_format: str = "zyxzyx",
        transforms_list: Optional[List[DictConfig]] = None,
        use_masks: bool = False,
        generate_binary_masks: bool = False,
        require_targets: bool = True,
        expect_mask_channel: bool = True,
        # with_resize: bool = False,
        debug: bool = False,
        normalize_bboxes: bool = False,
        async_device_copy: bool = False,
    ):
        self.columns = columns
        self.debug_device_idx = debug_device_idx

        self.node_id = node_id
        self.local_rank = local_rank()
        self.global_rank = process_rank()

        self.batch_size = batch_size
        # shape for the device buffer (after resize, without mask channel)
        self.input_shape = tuple(input_shape)
        self.spatial_shape = self._get_spatial_shape(self.input_shape, input_format)
        self.device_buffer_capacity = device_buffer_capacity

        self.input_format = input_format.upper()
        if self.input_format != "ZYXC":
            raise NotImplementedError(f"FinetuneCollatorActor currently assumes ZYXC, got {self.input_format}")

        self.mask_idx = mask_idx
        self.bbox_data_format = bbox_data_format
        self.bbox_output_format = bbox_output_format

        self.numa_node = torch_gpu_to_numa(self.local_rank)["numa_node"]
        if pin_numa_node:
            bind_current_process_to_node(self.numa_node)

        self.out_dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        self.buffer_dtype = NUMPY_DTYPES[buffer_dtype].value if isinstance(buffer_dtype, str) else buffer_dtype

        self.host_buffer_actor = get_buffers(
            type="host_memory",
            numa_node=self.numa_node,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            node_id=self.node_id,
        )
        cfg = ray.get(self.host_buffer_actor.get_config.remote())
        self.slot_bytes = int(cfg["slot_bytes"])
        self.batch_shape = tuple(cfg["batch_shape"])
        self.capacity = int(cfg["capacity"])
        self._shm = shared_memory.SharedMemory(name=cfg["name"])

        # original input shape (without batch) from host buffer
        # e.g. (Z_raw, Y_raw, X_raw, C_full)
        self.raw_input_shape = self.batch_shape[1:]

        self.pin_pages = pin_pages
        if pin_pages:
            base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm.buf))
            self.host_buffer_ptr = base_ptr
            cp.cuda.runtime.hostRegister(base_ptr, self.slot_bytes * self.capacity, 0)
            self._pinned = True
        else:
            self._pinned = False

        idx = self._get_device_index()
        torch.cuda.set_device(idx)
        self.device = torch.device(f"cuda:{idx}")
        with cp.cuda.Device(self.device.index):
            self.cp_stream = cp.cuda.Stream(non_blocking=True)
        # torch stream wrapping the same underlying CUDA stream
        self.copy_stream = torch.cuda.ExternalStream(int(self.cp_stream.ptr), device=self.device)

        # Device buffer for resized images (no mask channel)
        self.device_buffer = DeviceMemoryBuffer(
            name=f"device_buffer_rank_{self.global_rank}",
            capacity=self.device_buffer_capacity,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            dtype=buffer_dtype,
            device_idx=idx,
        )

        self.transforms = []
        for t in transforms_list or []:
            if isinstance(t, DictConfig):
                # not yet instantiated
                self.transforms.append(instantiate(t))
            elif isinstance(t, str):
                # a dotted‑path string
                self.transforms.append(get_method(t))
            else:
                # already an instantiated callable object
                self.transforms.append(t)

        # TODO: deprecate
        # self.with_resize = with_resize
        # if self.with_resize:
        #     ray.logger.info(f"FinetuneCollatorActor on rank {self.global_rank} using Resize transform")
        #     # CPU-side pinned resize buffer: shape matches final GPU input
        #     # input_shape is (Z_new, Y_new, X_new, C_no_mask)
        #     self.resize_buffer = torch.empty(
        #         (self.batch_size, *self.input_shape),
        #         dtype=self.out_dtype,
        #         pin_memory=True,
        #     )

        self.use_masks = use_masks
        self.generate_binary_masks = generate_binary_masks
        self.require_targets = require_targets
        self.expect_mask_channel = expect_mask_channel
        self.normalize_bboxes = normalize_bboxes

        ray.logger.info(
            f"FinetuneCollatorActor on rank {self.global_rank} and Numa Node {self.numa_node} "
            f"using host shared memory buffer with pin_numa_node={pin_numa_node} "
            f"with local rank {self.local_rank} and node id {self.node_id} "
            f"with name {cfg['name']} and capacity {cfg['capacity']} and HostMemoryBuffer "
            f"with pin_pages={self._pinned} and ray.get_gpu_ids()={ray.get_gpu_ids()} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"torch_dev={torch.cuda.current_device()} "
            f"cupy_dev={cp.cuda.runtime.getDevice()} "
            f"torch_count={torch.cuda.device_count()}"
        )

        self.debug = debug
        self.async_device_copy = async_device_copy

    def _get_spatial_shape(self, input_shape: tuple, input_format: str) -> tuple:
        input_format = input_format.upper()
        if input_format == "ZYXC":
            return input_shape[:-1]
        else:
            raise NotImplementedError(f"Unsupported input_format: {input_format}")

    def _get_input_shape(self, input_shape: tuple, input_format: str) -> tuple:
        # TODO: generalize this beyond last channel mask removal
        input_format = input_format.upper()
        if input_format == "ZYXC":
            # remove mask channel: (Z, Y, X, C_full) -> (Z, Y, X, C_full-1)
            *spatial, channels = input_shape
            return tuple([*spatial, channels - 1])
        else:
            raise NotImplementedError(f"Unsupported input_format: {input_format}")

    def _get_device_index(self) -> int:
        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            return int(gpu_ids[0])
        # Fallback for debug mode (running outside Ray Train workers)
        elif self.debug_device_idx is not None:
            return self.debug_device_idx
        else:
            raise RuntimeError("No GPUs assigned to this worker by Ray")

    def __del__(self):
        try:
            if getattr(self, "_pinned", False) and getattr(self, "host_buffer_ptr", None) is not None:
                cp.cuda.runtime.hostUnregister(self.host_buffer_ptr)
            if hasattr(self, "_shm"):
                self._shm.close()
        except Exception:
            pass

    def _get_masks(self, inputs: torch.Tensor):
        """
        inputs: (B, Z, Y, X, C_full)
        returns:
          inputs_wo_mask: (B, Z, Y, X, C_full-1)
          masks_labelmap: (B, Z, Y, X)
        """
        assert inputs.ndim == 5, f"Expected (B, Z, Y, X, C), got {inputs.shape}"
        B, Z, Y, X, C = inputs.shape

        if C < 2:
            raise ValueError(f"Expected at least 2 channels (image + mask), got C={C}")

        # For zero-copy we *require* the mask to be the last channel
        if self.mask_idx not in (-1, C - 1):
            raise ValueError(
                f"For zero-copy split, mask_idx must be -1 or C-1; " f"got mask_idx={self.mask_idx}, C={C}."
            )

        masks = inputs[..., -1].clone()  # (B, Z, Y, X), view
        return inputs, masks

    def _build_targets(
        self,
        masks_labelmap: torch.Tensor,  # (B, Z, Y, X) on CPU
        mask_bbox_dict_batch: List[str],
    ):
        """
        Build per-sample targets from labelmap + mask_bbox_dict.
        If self.use_masks is False, no binary masks are constructed and
        the "masks" key is omitted entirely from the targets.
        """
        # TODO: add check for different input formats
        B, Zm, Ym, Xm = masks_labelmap.shape
        spatial_shape = (Zm, Ym, Xm)
        device = masks_labelmap.device

        mask_ids_batch: List[List[int]] = []
        bboxes_batch: List[torch.Tensor] = []

        for raw in mask_bbox_dict_batch:
            instances = ujson.loads(raw)

            ids: List[int] = []
            boxes: List[List[float]] = []

            for cell_id_str, bbox in instances.items():
                ids.append(int(cell_id_str))
                if isinstance(bbox, (list, tuple)) and len(bbox) == 6:
                    if self.bbox_data_format == "zyxzyx":
                        zmin, ymin, xmin, zmax, ymax, xmax = bbox
                    elif self.bbox_data_format == "xyzxyz":
                        xmin, ymin, zmin, xmax, ymax, zmax = bbox
                    else:
                        raise ValueError(f"Unsupported bbox_data_format={self.bbox_data_format}")
                elif isinstance(bbox, dict):
                    zmin = bbox.get("zmin")
                    ymin = bbox.get("ymin")
                    xmin = bbox.get("xmin")
                    zmax = bbox.get("zmax")
                    ymax = bbox.get("ymax")
                    xmax = bbox.get("xmax")
                else:
                    continue

                if None in (zmin, ymin, xmin, zmax, ymax, xmax):
                    continue

                boxes.append([zmin, ymin, xmin, zmax, ymax, xmax])

            mask_ids_batch.append(ids)

            if boxes:
                bboxes_batch.append(torch.as_tensor(boxes, device=device, dtype=torch.float32))
            else:
                bboxes_batch.append(torch.zeros((0, 6), device=device, dtype=torch.float32))

        if self.use_masks and self.generate_binary_masks:
                binary_masks_batch = mask_ids_to_masks(
                    batch_size=B,
                    spatial_shape=spatial_shape,
                    mask_ids_batch=mask_ids_batch,
                    masks=masks_labelmap,
                    device=device,
                )
        else:
            binary_masks_batch = [None] * B

        if self.bbox_data_format != self.bbox_output_format:
            bboxes_batch = [
                convert_bbox_format(b, 
                    self.bbox_data_format, 
                    self.bbox_output_format, 
                    self.normalize_bboxes, 
                    # spatial shape is (Z, Y, X), need (X, Y, Z)
                    self.spatial_shape[::-1]) for b in bboxes_batch
            ]

        targets: List[Dict[str, Any]] = []
        for b, (ids, bm, boxes) in enumerate(zip(mask_ids_batch, binary_masks_batch, bboxes_batch)):
            mask_ids_tensor = torch.as_tensor(ids, device=device, dtype=torch.long)
            labels = torch.zeros(len(ids), device=device, dtype=torch.long)
            t: Dict[str, Any] = {
                "boxes": boxes,
                "mask_ids": mask_ids_tensor,
                "labels": labels,
            }
            if self.use_masks:
                if self.generate_binary_masks:
                    t["masks"] = bm
                else:
                    t["label_map"] = masks_labelmap[b]
            targets.append(t)
        
        return targets

    def _copy_h2d(self, dst: torch.Tensor, src: torch.Tensor):
        src_ptr = ctypes.c_void_p(src.data_ptr())
        dst_ptr = ctypes.c_void_p(dst.data_ptr())

        cudart.memcpyAsync(
            dst_ptr.value,
            src_ptr.value,
            src.numel() * src.element_size(),
            cudart.memcpyHostToDevice,
            int(self.cp_stream.ptr),
        )

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        batch: Ray batch containing at least:
          - "buffer_idx"
          - "mask_bbox_dict"
          - columns listed in self.columns (z_size, y_size, x_size, etc.)
        """
        with torch.cuda.device(self.device.index), cp.cuda.Device(self.device.index):
            host_buffer_idx = int(batch["buffer_idx"][0])
            h_view = np.ndarray(
                self.batch_shape,
                dtype=self.buffer_dtype,
                buffer=self._shm.buf,
                offset=host_buffer_idx * self.slot_bytes,
            )

            inputs_full = torch.from_numpy(h_view)
            # In inference we may have no labelmap channel at all.
            if self.expect_mask_channel:
                inputs, masks_labelmap = self._get_masks(inputs_full)
            else:
                inputs, masks_labelmap = inputs_full, None

            meta_cpu: Dict[str, Any] = {}
            for k in self.columns:
                if k in batch:
                    meta_cpu[k] = batch[k]

            # Build targets only when requested (training). For inference, produce empty targets
            # so downstream transforms that expect `metainfo["targets"]` still work.
            if self.require_targets:
                if "mask_bbox_dict" not in meta_cpu:
                    raise KeyError("FinetuneCollatorActor expects 'mask_bbox_dict' in columns when require_targets=True.")
                mask_bbox_dict_batch = list(meta_cpu["mask_bbox_dict"])
                targets_cpu = self._build_targets(
                    masks_labelmap=masks_labelmap,
                    mask_bbox_dict_batch=mask_bbox_dict_batch,
                )
            else:
                B = inputs.shape[0]
                device = torch.device("cpu")
                targets_cpu = []
                for _ in range(B):
                    targets_cpu.append(
                        {
                            "boxes": torch.zeros((0, 6), device=device, dtype=torch.float32),
                            "mask_ids": torch.zeros((0,), device=device, dtype=torch.long),
                            "labels": torch.zeros((0,), device=device, dtype=torch.long),
                        }
                    )

            image_sizes, orig_image_sizes, image_sizes_padded, padding_mask = get_image_sizes(
                input_format=self.input_format,
                input_shape=self.raw_input_shape,
                batch_size=self.batch_size,
                metadata=meta_cpu,
                device=torch.device("cpu"),
            )
            meta_cpu["image_sizes"] = torch.as_tensor(image_sizes)
            meta_cpu["orig_image_sizes"] = torch.as_tensor(orig_image_sizes)
            meta_cpu["image_sizes_padded"] = torch.as_tensor(image_sizes_padded)
            meta_cpu["padding_mask"] = torch.as_tensor(padding_mask)

            sample_cpu = {
                "data_tensor": inputs,
                "metainfo": {
                    **meta_cpu,
                    "targets": targets_cpu,
                    # "resize_buffer": self.resize_buffer if self.with_resize else None,
                },
            }

            if self.transforms:
                for t in self.transforms:
                    sample_cpu = t(sample_cpu)
                inputs_transformed = sample_cpu["data_tensor"]
                metainfo_transformed = sample_cpu["metainfo"]
            else:
                inputs_transformed = inputs
                metainfo_transformed = {
                    **meta_cpu,
                    "targets": targets_cpu,
                }

            device_buffer_idx = self.device_buffer.get_free()
            dst_device = self.device_buffer.device_buffers[device_buffer_idx]

            with torch.cuda.stream(self.copy_stream):
                self._copy_h2d(dst=dst_device, src=inputs_transformed)

                if self.async_device_copy:

                    def _release_buffer_on_done(stream, error_status, user_data):
                        actor_reference = user_data["actor"]
                        hb_idx = user_data["host_buffer_idx"]
                        try:
                            actor_reference.put_free.remote(hb_idx)
                        except Exception as e:
                            logger.exception(f"put_free failed for {hb_idx}: {e}")

                    with self.cp_stream:
                        self.cp_stream.add_callback(
                            _release_buffer_on_done,
                            {
                                "actor": self.host_buffer_actor,
                                "host_buffer_idx": host_buffer_idx,
                            },
                        )

            if not self.async_device_copy:
                # Force the copy to complete
                torch.cuda.synchronize(self.device)
                # Synchronously free host slot
                ray.get(self.host_buffer_actor.put_free.remote(host_buffer_idx))

            else:
                # tells caching allocator & scheduler on training stream
                # that dst_device is owned by copy_stream
                torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
                dst_device.record_stream(self.copy_stream)

            # NOTE: could probably be one recursive function that handles all tensors
            metainfo: Dict[str, Any] = {
                "host_buffer_idx": host_buffer_idx,
                "device_buffer_idx": device_buffer_idx,
            }
            for k, v in metainfo_transformed.items():
                if k in ("targets", "resize_buffer"):
                    continue
                if torch.is_tensor(v):
                    metainfo[k] = v.to(self.device, non_blocking=True)
                else:
                    metainfo[k] = v

            targets_gpu: List[Dict[str, Any]] = []
            for tgt in metainfo_transformed["targets"]:
                t_out: Dict[str, Any] = {}
                for tk, tv in tgt.items():
                    if torch.is_tensor(tv):
                        t_out[tk] = tv.to(self.device, non_blocking=True)
                    else:
                        t_out[tk] = tv
                targets_gpu.append(t_out)
            metainfo["targets"] = targets_gpu

            if self.debug:
                # NOTE: for testing only, put_free(idx) otherwise called by hooks in
                #       training loop, see training/hooks.py:FreeDeviceBufferHook
                if self.async_device_copy:
                    ray.get(self.host_buffer_actor.put_free.remote(host_buffer_idx))
                self.device_buffer.put_free(device_buffer_idx)

            return {"data_tensor": dst_device, "metainfo": metainfo}


@pprof_class
class CollatorActor:
    def __init__(
        self,
        batch_size: int,
        input_shape: tuple,
        device_buffer_capacity: int,
        dtype: str,
        buffer_dtype: str,
        pin_numa_node: bool,
        pin_pages: bool,
        node_id: int,
        callback_strategy: Literal["grpc", "queue"] = "grpc",
        async_device_copy: bool = False,
        columns: List[str] = [
            # metadata columns to keep from the original dataframe
            "x_start",
            "y_start",
            "z_start",
            "time_start",
            "channel_size",
            "z_size",
            "y_size",
            "x_size",
            "time_size",
            "server_folder",
            "output_folder",
            "tile_name",
            "prepared_id",
        ],
        debug: bool = False,
        debug_device_idx: Optional[int] = None,
    ):
        self.columns = columns
        self.debug_device_idx = debug_device_idx

        self.node_id = node_id
        self.local_rank = local_rank()
        self.global_rank = process_rank()

        self.batch_size = batch_size
        self.input_shape = tuple(input_shape)
        self.device_buffer_capacity = device_buffer_capacity

        self.numa_node = torch_gpu_to_numa(self.local_rank)["numa_node"]
        if pin_numa_node:
            bind_current_process_to_node(self.numa_node)

        self.out_dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        self.buffer_dtype = NUMPY_DTYPES[buffer_dtype].value if isinstance(buffer_dtype, str) else buffer_dtype

        self.host_buffer_actor = get_buffers(
            type="host_memory",
            numa_node=self.numa_node,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            node_id=self.node_id,
        )
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

        idx = self._get_device_index()
        torch.cuda.set_device(idx)
        self.device = torch.device(f"cuda:{idx}")
        with cp.cuda.Device(self.device.index):
            self.cp_stream = cp.cuda.Stream(non_blocking=True)
        # wrap the same stream for torch ops
        self.copy_stream = torch.cuda.ExternalStream(int(self.cp_stream.ptr), device=self.device)

        self.device_buffer = DeviceMemoryBuffer(
            name=f"device_buffer_rank_{self.global_rank}",
            capacity=self.device_buffer_capacity,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            dtype=buffer_dtype,
            device_idx=idx,
        )

        ray.logger.info(
            f"CollatorActor on rank {self.global_rank} and Numa Node {self.numa_node} "
            f"using host shared memory buffer with pin_numa_node={pin_numa_node} "
            f"with local rank {self.local_rank} and node id {self.node_id} "
            f"with name {cfg['name']} and capacity {cfg['capacity']} and HostMemoryBuffer "
            f"with pin_pages={self._pinned} and ray.get_gpu_ids()={ray.get_gpu_ids()} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"torch_dev={torch.cuda.current_device()} "
            f"cupy_dev={cp.cuda.runtime.getDevice()} "
            f"torch_count={torch.cuda.device_count()}"
        )

        self.debug = debug
        self.async_device_copy = async_device_copy
        if callback_strategy not in ("grpc", "queue"):
            raise ValueError(f"Unknown callback_strategy={callback_strategy}")
        self.callback_strategy = callback_strategy

        if self.callback_strategy == "queue":
            self._pending_frees = Queue()
            self._free_thread = Thread(target=self._free_worker, daemon=True)
            self._free_thread.start()

    def _free_worker(self):
        torch.cuda.set_device(self.device)
        while True:
            event, host_buffer_idx = self._pending_frees.get()
            event.synchronize()
            try:
                self.host_buffer_actor.put_free.remote(host_buffer_idx)
            except Exception as e:
                logger.exception(f"put_free failed for {host_buffer_idx}: {e}")

    def _get_device_index(self) -> int:
        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            return int(gpu_ids[0])
        # Fallback for debug mode (running outside Ray Train workers)
        elif self.debug_device_idx is not None:
            return self.debug_device_idx
        else:
            raise RuntimeError("No GPUs assigned to this worker by Ray")

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
        cudart.memcpyAsync(dst_ptr.value, src_ptr.value, src.nbytes, cudart.memcpyHostToDevice, int(self.cp_stream.ptr))

    def __call__(self, batch):
        with torch.cuda.device(self.device.index), cp.cuda.Device(self.device.index):
            host_buffer_idx = int(batch["buffer_idx"][0])
            h_view = np.ndarray(
                self.batch_shape,
                dtype=self.buffer_dtype,
                buffer=self._shm.buf,
                offset=host_buffer_idx * self.slot_bytes,
            )

            device_buffer_idx = self.device_buffer.get_free()
            dst_device = self.device_buffer.device_buffers[device_buffer_idx]

            with torch.cuda.stream(self.copy_stream):
                self.copy_h2d(dst=dst_device, src=h_view)

                if self.async_device_copy:

                    if self.callback_strategy == "grpc":

                        def _release_buffer_on_done(stream, error_status, user_data):
                            actor_reference = user_data["actor"]
                            host_buffer_idx = user_data["host_buffer_idx"]
                            # runs after all prior ops in stream
                            try:
                                actor_reference.put_free.remote(host_buffer_idx)
                            except Exception as e:
                                logger.exception(f"put_free failed for {host_buffer_idx}: {e}")

                        with self.cp_stream:
                            self.cp_stream.add_callback(
                                _release_buffer_on_done,
                                {"actor": self.host_buffer_actor, "host_buffer_idx": host_buffer_idx},
                            )

                    else:
                        event = torch.cuda.Event(enable_timing=False)
                        event.record(self.copy_stream)

            if not self.async_device_copy:
                # Force the copy to complete
                torch.cuda.synchronize(self.device)
                # Synchronously free host slot
                ray.get(self.host_buffer_actor.put_free.remote(host_buffer_idx))

            else:
                # tells caching allocator & scheduler on training stream
                # that dst_device is owned by copy_stream
                torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
                dst_device.record_stream(self.copy_stream)
                if self.callback_strategy == "queue":
                    self._pending_frees.put((event, host_buffer_idx))

            metainfo = {
                "host_buffer_idx": host_buffer_idx,
                "device_buffer_idx": device_buffer_idx,
            }
            for k in self.columns:
                if k in batch:
                    metainfo[k] = batch[k]

            if self.debug:
                # NOTE: for testing only, put_free(idx) otherwise called by hooks in
                #       training loop, see training/hooks.py:FreeDeviceBufferHook
                if self.async_device_copy:
                    ray.get(self.host_buffer_actor.put_free.remote(host_buffer_idx))
                self.device_buffer.put_free(device_buffer_idx)

            return {"data_tensor": dst_device, "metainfo": metainfo}


# -------- -------- Loader Actors -------- --------


@pprof_class
class LoaderActor:
    def __init__(
        self,
        dim: int,
        input_format: str,
        node_id: int,
        local_rank: int,
        global_rank: int,
        numa_node: int,
        batch_size: int,
        input_layout: str,
        context_spec: Dict[str, Any],
        dtype: str = "fp16",
        buffer_dtype: str = "uint16",
        pin_numa_node: bool = True,
        with_batched_api: bool = True,
        channels_subset: Optional[List[int]] = None,
        pad_mode: Literal["zero"] = "zero",
    ):
        self.dim = dim
        self.input_format = input_format.upper()
        self.pad_mode = pad_mode

        self.node_id, self.local_rank, self.global_rank = node_id, local_rank, global_rank
        self.driver_process_numa_node = numa_node
        if pin_numa_node:
            self.actor_scheduler = ray.get_actor(
                f"numa_node_affinity_scheduler_node_{self.node_id}", namespace="schedulers"
            )
            self.numa_node = ray.get(self.actor_scheduler.schedule_actor_for_gpu.remote(local_rank))
            ray.logger.info(f"Binding LoaderActor on rank {global_rank} to NUMA node {self.numa_node}")
            bind_current_process_to_node(self.numa_node)

        # input data layout
        self.channels_subset = list(channels_subset) if channels_subset is not None else None
        self.input_layout = input_layout.upper()
        self.batch_size = batch_size

        # dtypes
        self.dtype = TENSORSTORE_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        if self.dtype == TENSORSTORE_DTYPES.bf16.value:
            # ray.logger.warning(
            #     "Using fp16 for PyArrow, Collator will cast data to bf16"
            # )
            self.dtype = TENSORSTORE_DTYPES.fp16.value

        self.buffer_dtype = NUMPY_DTYPES[buffer_dtype].value if isinstance(buffer_dtype, str) else buffer_dtype

        # tensorstore
        self._handles = {}
        self.ctx = ts.Context(context_spec)
        self.with_batched_api = with_batched_api

        # memory buffer
        self.buffer_actor = get_buffers(
            type=f"host_memory",
            node_id=self.node_id,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            numa_node=self.driver_process_numa_node,
        )

        cfg = ray.get(self.buffer_actor.get_config.remote())
        self.slot_bytes = int(cfg["slot_bytes"])
        self.batch_shape = tuple(cfg["batch_shape"])
        self._shm = shared_memory.SharedMemory(name=cfg["name"])

        ray.logger.info(
            f"LoaderActor on global rank {self.global_rank} and numa node {self.driver_process_numa_node} "
            f"using shared memory buffer and placed on numa node {self.numa_node} "
            f"with local rank {self.local_rank} and node id {self.node_id} "
            f"with name {cfg['name']} and capacity {cfg['capacity']}"
        )

    def __del__(self):
        try:
            actor_scheduler = ray.get_actor(f"numa_node_affinity_scheduler_node_{self.node_id}", namespace="schedulers")
            actor_scheduler.free.remote(self.numa_node)
        except Exception:
            pass

    def _slice_hypercube(self, data_tensor, meta: Dict[str, Any], ts_batch=None):
        t = slice(meta["time_start"], meta["time_start"] + meta["time_size"])
        z = slice(meta["z_start"], meta["z_start"] + meta["z_size"])
        y = slice(meta["y_start"], meta["y_start"] + meta["y_size"])
        x = slice(meta["x_start"], meta["x_start"] + meta["x_size"])

        if self.channels_subset is not None:
            if self.input_format == "ZYXC" or self.input_format == "TZYXC":
                view = data_tensor[t, z, y, x, self.channels_subset]
            else:
                raise NotImplementedError(f"Channel subsetting not implemented for input format {self.input_format}")
        else:
            if self.input_format == "ZYXC" or self.input_format == "TZYXC":
                c = slice(0, meta["channel_size"])
                view = data_tensor[t, z, y, x, c]
            else:
                raise NotImplementedError(f"Input format {self.input_format} not implemented")

        if self.dim == 3:
            if self.input_format == "ZYXC":
                view = view[meta["time_start"], ...]
            else:
                raise NotImplementedError(f"Input format {self.input_format} not implemented for 3D data")

        return view

    def _get_handle(self, path: str):
        h = self._handles.get(path)
        if h is None:
            h = read_zarr(path, dtype=self.dtype, context=self.ctx, cast=False)
            self._handles[path] = h
        return h

    def __call__(self, batch):
        buffer = ray.get(self.buffer_actor.get_free.remote())
        dst = np.ndarray(
            self.batch_shape, dtype=self.buffer_dtype, buffer=self._shm.buf, offset=buffer["slot"] * self.slot_bytes
        )

        write_futs = []
        with ts.Batch() as b:
            for i in range(self.batch_size):
                p = os.path.join(
                    batch["server_folder"][i],
                    batch["output_folder"][i],
                    batch["tile_name"][i],
                )
                meta = {
                    "time_start": batch["time_start"][i],
                    "time_size": batch["time_size"][i],
                    "z_start": batch["z_start"][i],
                    "y_start": batch["y_start"][i],
                    "x_start": batch["x_start"][i],
                    "z_size": batch["z_size"][i],
                    "y_size": batch["y_size"][i],
                    "x_size": batch["x_size"][i],
                    "channel_size": batch["channel_size"][i],
                }
                src_view = self._slice_hypercube(self._get_handle(p), meta=meta, ts_batch=b)

                if self.dim == 3:
                    if self.input_format == "ZYXC":
                        tz, ty, tx, tc = src_view.shape
                        dst_slice = (slice(0, tz), slice(0, ty), slice(0, tx), slice(0, tc))
                    else:
                        raise NotImplementedError(f"Input format {self.input_format} not implemented for 3D data")
                else:
                    if self.input_format == "TZYXC":
                        tt, tz, ty, tx, tc = src_view.shape
                        dst_slice = (slice(0, tt), slice(0, tz), slice(0, ty), slice(0, tx), slice(0, tc))
                    else:
                        raise NotImplementedError(f"Input format {self.input_format} not implemented for 4D data")

                write_futs.append(ts.array(dst[i][dst_slice]).write(src_view))

                # NOTE: pad the tail after write, currently we only support zero padding
                if self.pad_mode == "zero":
                    if self.dim == 3:
                        if self.input_format == "ZYXC":
                            B, Z, Y, X, C = self.batch_shape
                            if tz < Z:
                                dst[i][tz:, :, :, :].fill(0)
                            if ty < Y:
                                dst[i][:tz, ty:, :, :].fill(0)
                            if tx < X:
                                dst[i][:tz, :ty, tx:, :].fill(0)
                            if tc < C:
                                dst[i][:tz, :ty, :tx, tc:].fill(0)
                        else:
                            raise NotImplementedError(f"Input format {self.input_format} not implemented for 3D data")
                    else:
                        if self.input_format == "TZYXC":
                            B, T, Z, Y, X, C = self.batch_shape
                            # NOTE: broadcast last valid slice along time axis
                            if tt < T:
                                dst[i][tt:T, ...] = dst[i][tt - 1, ...]
                            if tz < Z:
                                dst[i][:tt, tz:, :, :, :].fill(0)
                            if ty < Y:
                                dst[i][:tt, :tz, ty:, :, :].fill(0)
                            if tx < X:
                                dst[i][:tt, :tz, :ty, tx:, :].fill(0)
                            if tc < C:
                                dst[i][:tt, :tz, :ty, :tx, tc:].fill(0)
                        else:
                            raise NotImplementedError(f"Input format {self.input_format} not implemented for 4D data")
                else:
                    raise NotImplementedError(f"Pad mode {self.pad_mode} not implemented")

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
    ctx._enable_actor_pool_on_exit_hook = True
    # ctx.execution_options.preserve_order = cfg.datasets.preserve_order


def get_context_spec(cfg: DictConfig) -> Dict[str, Any]:
    ts_ctx = OmegaConf.to_container(cfg.datasets.context, resolve=True)
    ctx_spec = {k: v for k, v in ts_ctx.items() if v is not None}
    return ctx_spec


def partition_indices_for_inference(
    df: pd.DataFrame,
    world_size: int,
    batch_size: int,
    drop_last_policy: bool,
    roi_col: str = "prepared_id",
    tile_col: str = "tile_name",
) -> list[list[int]]:
    total = len(df)
    num_samples_per_rank = total // world_size

    if drop_last_policy:
        num_samples_per_rank = (num_samples_per_rank // batch_size) * batch_size

    # round-robin assignment if not enough samples
    if num_samples_per_rank == 0:
        rows_per_rank = [[] for _ in range(world_size)]
        for i, idx in enumerate(df.index.tolist()):
            rows_per_rank[i % world_size].append(int(idx))
        return rows_per_rank

    df_sub = df.iloc[: world_size * num_samples_per_rank]

    df_row_by_rank = df_sub.apply(lambda r: tile_owner(int(r[roi_col]), str(r[tile_col]), world_size), axis=1)
    idxs = df_sub.index.to_numpy()

    df_rank_to_row = {r: [] for r in range(world_size)}
    for i, own in zip(idxs, df_row_by_rank.to_numpy()):
        df_rank_to_row[int(own)].append(int(i))

    rows_per_rank = [[] for _ in range(world_size)]
    row_remainders = []
    for r in range(world_size):
        locality_matched_samples = df_rank_to_row[r][:num_samples_per_rank]
        rank_row_remainders = df_rank_to_row[r][num_samples_per_rank:]
        rows_per_rank[r].extend(locality_matched_samples)
        row_remainders.extend(rank_row_remainders)

    for r in range(world_size):
        non_locality_matched_rows = num_samples_per_rank - len(rows_per_rank[r])
        if non_locality_matched_rows > 0:
            rows_per_rank[r].extend(row_remainders[:non_locality_matched_rows])
            row_remainders = row_remainders[non_locality_matched_rows:]

    assert all(
        len(x) == num_samples_per_rank for x in rows_per_rank
    ), "Not all ranks have equal size data shards after partitioning."

    return rows_per_rank


def shuffle_table(table: pa.Table, seed: int) -> pa.Table:
    n = table.num_rows
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return table.take(pa.array(perm, type=pa.int64()))


def get_dataset_ray(
    cfg: DictConfig,
    seed: Optional[int],
    indices: Optional[List[int]],
    database: Optional[Any] = None,
    columns: list = [
        # metadata columns to keep from the original dataframe
        # adding more columns may slow down collate
        "x_start",
        "y_start",
        "z_start",
        "time_start",
        "channel_size",
        "z_size",
        "y_size",
        "x_size",
        "time_size",
        "server_folder",
        "output_folder",
        "tile_name",
        "prepared_id",
        "mask_bbox_dict",
    ],
    dp_degree: Optional[int] = None,
    dp_rank: Optional[int] = None,
    shuffle: bool = False,
    drop_last: bool = True,
):
    if seed is not None and not shuffle:
        raise ValueError("Seed provided but shuffle is False.")

    if cfg.datasets.channels_subset is not None:
        # NOTE: this always works because dataset_layout_order is 1-1 matched
        num_channels = cfg.datasets.input_shape[cfg.dataset_layout_order.index("C")]
        assert len(list(cfg.datasets.channels_subset)) == num_channels, (
            f"channels_subset length {len(cfg.datasets.channels_subset)} "
            f"does not match number of channels {num_channels} in input_shape {cfg.datasets.input_shape}"
        )

    set_data_context(cfg)
    ctx_spec = get_context_spec(cfg)

    print(database.hypercubes_dataframe)
    print(f"Trying to get dataset with \ncolumns: {columns} from \n{database.hypercubes_dataframe.columns}")
    base_df = database.hypercubes_dataframe[columns]
    if indices is not None:
        base_df = base_df.iloc[indices]

    # sort dataframe for consistent sharding across TP/CP/PP ranks
    base_df = base_df.sort_values(
        ["prepared_id", "tile_name", "z_start", "y_start", "x_start", "time_start"]
    ).reset_index(drop=True)

    # TODO: consider checking dataframe consistency before sharding
    local_db_hash = df_signature_polars(base_df)
    print(f"Dataset dataframe signature hash on rank {process_rank()}: {local_db_hash}")
    # assert_same_db_hash_across_ranks(local_db_hash)

    if dp_degree is not None and dp_rank is not None:
        ws, rk = dp_degree, dp_rank
    else:
        ws, rk = get_world_size(), process_rank()

    if cfg.job_type == "predict":
        per_rank_indices = partition_indices_for_inference(
            df=base_df,
            world_size=ws,
            batch_size=cfg.clusters.batch_size_per_gpu,
            drop_last_policy=drop_last,
            roi_col="prepared_id",
            tile_col="tile_name",
        )
        local_idx = per_rank_indices[rk]
        local_df = base_df.loc[local_idx]

        ray.logger.info(f"Rank {rk} assigned dataframe: {local_df}")
        ray.logger.info(f"Rank {rk} dataframe unique tiles: {local_df['tile_name'].nunique()}")

        table = pa.table(local_df)
        dataset = ray.data.from_arrow(table)

        dataset_len = len(local_df)

    else:
        table = pa.table(base_df)
        if shuffle:
            table = shuffle_table(table, seed=seed)
        
        n = table.num_rows
        n_shard = (n // ws) * ws
        table = table.slice(0, n_shard)

        shard_len = n_shard // ws
        local_table = table.slice(rk * shard_len, shard_len)

        if drop_last:
            B = cfg.clusters.batch_size_per_gpu
            keep = (local_table.num_rows // B) * B
            local_table = local_table.slice(0, keep)

        dataset = ray.data.from_arrow(local_table)
        dataset_len = local_table.num_rows

    # NOTE: this is necessary to avoid slow startup
    dataset = dataset.repartition(target_num_rows_per_block=cfg.datasets.rows_per_block, shuffle=False)

    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id(),
        soft=False,
    )
    dataset = dataset.map_batches(
        LoaderActor,
        scheduling_strategy=scheduling_strategy,
        num_cpus=1 / cfg.datasets.actor_oversub_factor,
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
            "global_rank": process_rank(),
            "node_id": node_id(),
            "numa_node": torch_gpu_to_numa(local_rank())["numa_node"],
            "dim": get_data_dim(cfg.dataset_layout_order),
            "input_format": cfg.dataset_layout_order,
        },
        concurrency=(cfg.datasets.num_actors_min, cfg.datasets.num_actors_max),
    )

    return dataset, dataset_len


def get_dataloader_ray(
    cfg: DictConfig,
    batch_size: int,
    collate_fn: Optional[Callable],
    epoch: int = 0,
    drop_last: bool = True,
    database: Optional[Any] = None,
    dp_degree: Optional[int] = None,
    dp_rank: Optional[int] = None,
):
    assert hasattr(cfg, "seed"), "cfg.seed is required for Ray Dataloader."

    if database is None:
        db = instantiate(cfg.datasets.databases)
    else:
        db = database

    database_df = db.hypercubes_dataframe
    dataset_len = len(db.hypercubes_dataframe)

    if cfg.datasets.split is not None and 0.0 < float(cfg.datasets.split) < 1.0:
        g = torch.Generator()
        g.manual_seed(int(cfg.seed))

        val_size = round(dataset_len * cfg.datasets.split)
        train_subset, val_subset = random_split(
            range(dataset_len), lengths=[dataset_len - val_size, val_size], generator=g
        )
        train_indices, val_indices = train_subset.indices, val_subset.indices

        train_dataset, train_dataset_len = get_dataset_ray(
            cfg,
            indices=train_indices,
            database=db,
            columns=list(cfg.datasets.columns),
            dp_degree=dp_degree,
            dp_rank=dp_rank,
            seed=int(cfg.seed) + int(epoch),
            shuffle=True,
            drop_last=drop_last,
        )
        val_dataset, val_dataset_len = get_dataset_ray(
            cfg,
            indices=val_indices,
            database=db,
            columns=list(cfg.datasets.columns),
            dp_degree=dp_degree,
            dp_rank=dp_rank,
            seed=None,
            shuffle=False,
            drop_last=drop_last,
        )

        record_dataset_len(cfg, train_dataset_len, val_dataset_len)

        train_dataloader = train_dataset.iterator()._iter_batches(
            batch_size=batch_size, _finalize_fn=collate_fn, batch_format="numpy"
        )
        val_dataloader = val_dataset.iterator()._iter_batches(
            batch_size=batch_size, _finalize_fn=collate_fn, batch_format="numpy"
        )
        return train_dataloader, val_dataloader, database_df

    else:
        train_dataset, train_dataset_len = get_dataset_ray(
            cfg, indices=None, database=db, columns=list(cfg.datasets.columns), 
            dp_degree=dp_degree, dp_rank=dp_rank, 
            seed=int(cfg.seed) + int(epoch), shuffle=True, drop_last=drop_last
        )
        record_dataset_len(cfg, train_dataset_len, 0)

        train_dataloader = train_dataset.iterator()._iter_batches(
            batch_size=batch_size, _finalize_fn=collate_fn, batch_format="numpy"
        )
        return train_dataloader, None, database_df
