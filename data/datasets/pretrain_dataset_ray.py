import os
import sys
import ctypes
import logging
from typing import Any, Callable, Dict, List, Literal, Optional

from queue import Queue
from threading import Thread

import ray

import cupy as cp
import numpy as np
import pyarrow as pa
import tensorstore as ts

import torch
import ujson

from cupy.cuda import runtime as cudart
from omegaconf import DictConfig, OmegaConf
 
from cell_observatory_platform.data.databases.local_metadata_store import (
    MappedTable,
    MappedTableDescriptor,
    SampleIndexPlanner,
)
from cell_observatory_platform.data.io import read_zarr
from cell_observatory_platform.data.datasets.buffers import DeviceMemoryBuffer, attach_shared_memory, get_buffers
from cell_observatory_platform.data.structures import convert_bbox_format
from cell_observatory_platform.data.data_types import NUMPY_DTYPES, TENSORSTORE_DTYPES, TORCH_DTYPES
from cell_observatory_platform.data.datasets.utils import resolve_channel_localization_indices
from cell_observatory_platform.training.helpers import get_data_dim, get_image_sizes, record_dataset_len
from cell_observatory_platform.utils.context import (
    bind_current_process_to_node,
    get_world_size,
    local_rank,
    node_id,
    process_rank,
    ray_assigned_gpu_to_torch_ordinal,
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
          * build per-instance targets from annotations_metadata,
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
            "channel_mapping",
            "annotations_metadata",
        ],
        input_format: Literal["ZYXC", "TZYXC"] = "ZYXC",
        bbox_data_format: str = "zyxzyx",
        bbox_output_format: str = "zyxzyx",
        require_targets: bool = True,
        # with_resize: bool = False,
        debug: bool = False,
        debug_device_idx: Optional[int] = None,
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
        if self.input_format not in ["ZYXC", "TZYXC"]:
            raise NotImplementedError(f"FinetuneCollatorActor currently assumes ZYXC, got {self.input_format}")

        self.bbox_data_format = bbox_data_format
        self.bbox_output_format = bbox_output_format

        self.numa_node = torch_gpu_to_numa(self._get_device_index())["numa_node"]
        if pin_numa_node:
            bind_current_process_to_node(self.numa_node)

        self.out_dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        self.buffer_dtype = NUMPY_DTYPES[buffer_dtype].value if isinstance(buffer_dtype, str) else buffer_dtype

        self.host_buffer_actor = get_buffers(
            type="host_memory",
            pool_name="loader",
            numa_node=self.numa_node,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            node_id=self.node_id,
        )
        cfg = ray.get(self.host_buffer_actor.get_config.remote())
        self.slot_bytes = int(cfg["slot_bytes"])
        self.batch_shape = tuple(cfg["batch_shape"])
        self.capacity = int(cfg["capacity"])
        self._shm = attach_shared_memory(cfg["name"])

        # original input shape (without batch) from host buffer
        # e.g. (Z_raw, Y_raw, X_raw, C_full)
        self.raw_input_shape = self.batch_shape[1:]

        self.pin_pages = pin_pages
        if pin_pages:
            base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm.buf))
            self.host_buffer_ptr = base_ptr
            size = self.slot_bytes * self.capacity
            if size > 0:
                try:
                    cp.cuda.runtime.hostRegister(base_ptr, size, 0)
                    self._pinned = True
                except cudart.CUDARuntimeError as e:
                    logger.warning(
                        "hostRegister failed (%s), proceeding without pinned host memory", e
                    )
                    self._pinned = False
            else:
                self._pinned = False
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

        # Labelmap ownership lives entirely on the model preprocessor (GPU).
        # The collator never clones, splits, or materializes the labelmap; the
        # integer labelmap simply rides on the last channel of data_tensor and
        # is transferred to VRAM in the single H2D copy. The preprocessor then
        # splits it off (int32, before the dtype cast), applies transforms, and
        # builds per-instance binary masks. The collator only emits lightweight
        # per-target metadata (boxes / mask_ids / labels).
        self.require_targets = require_targets
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

    @staticmethod
    def _parse_annotations_metadata(raw: object) -> list[dict[str, Any]]:
        if hasattr(raw, "as_py"):
            raw = raw.as_py()
        if raw is None:
            return []
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            payload = raw.strip()
            if not payload or payload.lower() == "null":
                return []
            raw = ujson.loads(payload)

        if not isinstance(raw, list):
            raise ValueError(
                f"annotations_metadata must be a JSON list of objects, got {type(raw).__name__}"
            )

        return [item for item in raw if isinstance(item, dict)]

    def _get_spatial_shape(self, input_shape: tuple, input_format: str) -> tuple:
        input_format = input_format.upper()
        if input_format == "ZYXC":
            return input_shape[:-1]
        elif input_format == "TZYXC":
            # spatial shape is (Z, Y, X)
            return input_shape[1:-1]
        else:
            raise NotImplementedError(f"Unsupported input_format: {input_format}")

    def _get_input_shape(self, input_shape: tuple, input_format: str) -> tuple:
        # TODO: generalize this beyond last channel mask removal
        input_format = input_format.upper()
        if input_format == "ZYXC":
            # remove mask channel: (Z, Y, X, C_full) -> (Z, Y, X, C_full-1)
            *spatial, channels = input_shape
            return tuple([*spatial, channels - 1])
        elif input_format == "TZYXC":
            # remove mask channel: (T, Z, Y, X, C_full) -> (T, Z, Y, X, C_full-1)
            *spatial, channels = input_shape
            return tuple([*spatial, channels - 1])
        else:
            raise NotImplementedError(f"Unsupported input_format: {input_format}")

    def _get_device_index(self) -> int:
        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            return ray_assigned_gpu_to_torch_ordinal(gpu_ids)
        # Fallback for debug mode (running outside Ray Train workers)
        elif self.debug_device_idx is not None:
            ray.logger.warning(f"Using debug device index {self.debug_device_idx}. If not debugging this could lead to unexpected behavior.")
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

    def _build_targets(
        self,
        annotations_metadata_batch: List[object],
    ):
        """
        Build per-sample targets (boxes / mask_ids / labels) from
        annotations_metadata. The labelmap is NOT touched here: it rides on the
        data_tensor channel to VRAM and is split off + transformed + turned into
        per-instance binary masks by the model preprocessor (single source).
        """
        if self.bbox_data_format != "zyxzyx":
            raise ValueError(
                f"annotations_metadata provides bbox_zyxzyx, so bbox_data_format must be 'zyxzyx', got {self.bbox_data_format}"
            )

        device = torch.device("cpu")

        mask_ids_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []
        bboxes_batch: List[torch.Tensor] = []

        for raw in annotations_metadata_batch:
            annotations = self._parse_annotations_metadata(raw)

            ids: List[int] = []
            labels: List[int] = []
            boxes: List[List[float]] = []

            for annotation in annotations:
                seg_id = annotation.get("local_segmentation_id")
                bbox = annotation.get("bbox_zyxzyx")
                if seg_id is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 6:
                    continue

                ids.append(int(seg_id))
                # NOTE: default to 0 if cell_type_id is not present
                labels.append(int(annotation.get("cell_type_id") or 0))
                boxes.append([float(value) for value in bbox])

            mask_ids_batch.append(ids)
            labels_batch.append(labels)

            if boxes:
                box_tensor = torch.as_tensor(boxes, device=device, dtype=torch.float32)
            else:
                box_tensor = torch.zeros((0, 6), device=device, dtype=torch.float32)

            if self.bbox_output_format != "zyxzyx":
                box_tensor = convert_bbox_format(
                    box_tensor,
                    "zyxzyx",
                    self.bbox_output_format,
                    self.normalize_bboxes,
                    self.spatial_shape[::-1],
                )
            bboxes_batch.append(box_tensor)

        targets: List[Dict[str, Any]] = []
        for ids, labels, boxes in zip(mask_ids_batch, labels_batch, bboxes_batch):
            t: Dict[str, Any] = {
                "boxes": boxes,
                "mask_ids": torch.as_tensor(ids, device=device, dtype=torch.long),
                "labels": torch.as_tensor(labels, device=device, dtype=torch.long),
            }
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
          - "annotations_metadata"
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

            # The full tensor (image channels + any labelmap channel) rides to
            # VRAM untouched; the preprocessor owns labelmap extraction.
            inputs = torch.from_numpy(h_view)

            meta_cpu: Dict[str, Any] = {}
            for k in self.columns:
                if k in batch:
                    meta_cpu[k] = batch[k]
            if "batch_size_actual" in batch:
                bsa = batch["batch_size_actual"]
                meta_cpu["batch_size_actual"] = int(np.asarray(bsa).ravel()[0])
            if "valid_mask" in batch:
                meta_cpu["valid_mask"] = batch["valid_mask"]

            # Build targets only when requested (training). For inference, produce empty targets
            # so downstream transforms that expect `metainfo["targets"]` still work.
            if self.require_targets:
                if "annotations_metadata" not in meta_cpu:
                    raise KeyError(
                        "FinetuneCollatorActor expects 'annotations_metadata' in columns when require_targets=True."
                    )
                annotations_metadata_batch = list(meta_cpu["annotations_metadata"])
                targets_cpu = self._build_targets(
                    annotations_metadata_batch=annotations_metadata_batch,
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

            # No CPU transforms: augmentation is owned entirely by the GPU
            # preprocessor (single source of truth for spatiotemporal ops).
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

        self.numa_node = torch_gpu_to_numa(self._get_device_index())["numa_node"]
        if pin_numa_node:
            bind_current_process_to_node(self.numa_node)

        self.out_dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        self.buffer_dtype = NUMPY_DTYPES[buffer_dtype].value if isinstance(buffer_dtype, str) else buffer_dtype

        self.host_buffer_actor = get_buffers(
            type="host_memory",
            pool_name="loader",
            numa_node=self.numa_node,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            node_id=self.node_id,
        )
        cfg = ray.get(self.host_buffer_actor.get_config.remote())
        self.slot_bytes = int(cfg["slot_bytes"])
        self.batch_shape = tuple(cfg["batch_shape"])
        self.capacity = int(cfg["capacity"])
        self._shm = attach_shared_memory(cfg["name"])

        self.pin_pages = pin_pages
        if pin_pages:
            base_ptr = ctypes.addressof(ctypes.c_char.from_buffer(self._shm.buf))
            self.host_buffer_ptr = base_ptr
            size = self.slot_bytes * self.capacity
            if size > 0:
                try:
                    cp.cuda.runtime.hostRegister(base_ptr, size, 0)
                    self._pinned = True
                except cudart.CUDARuntimeError as e:
                    logger.warning(
                        "hostRegister failed (%s), proceeding without pinned host memory", e
                    )
                    self._pinned = False
            else:
                self._pinned = False
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
            return ray_assigned_gpu_to_torch_ordinal(gpu_ids)
        # Fallback for debug mode (running outside Ray Train workers)
        elif self.debug_device_idx is not None:
            ray.logger.warning(f"Using debug device index {self.debug_device_idx}. If not debugging this could lead to unexpected behavior.")
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
            if "batch_size_actual" in batch:
                bsa = batch["batch_size_actual"]
                metainfo["batch_size_actual"] = int(np.asarray(bsa).ravel()[0])
            if "valid_mask" in batch:
                metainfo["valid_mask"] = batch["valid_mask"]

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
        selected_channel_localizations: Optional[List[str]] = None,
        pad_mode: Literal["zero"] = "zero",
        last_batch_policy: str = "drop",
        save_mode: Optional[Literal["overwrite", "append"]] = None,
    ):
        self.dim = dim
        self.input_format = input_format.upper()
        self.pad_mode = pad_mode
        self.last_batch_policy = last_batch_policy
        self.save_mode = save_mode

        self.node_id, self.local_rank, self.global_rank = node_id, local_rank, global_rank
        self.driver_process_numa_node = numa_node
        self.numa_node = numa_node  # default; replaced by scheduler when pin_numa_node
        if pin_numa_node:
            self.actor_scheduler = ray.get_actor(
                f"numa_node_affinity_scheduler_node_{self.node_id}", namespace="schedulers"
            )
            self.numa_node = ray.get(self.actor_scheduler.schedule_actor_for_gpu.remote(local_rank))
            ray.logger.info(f"Binding LoaderActor on rank {global_rank} to NUMA node {self.numa_node}")
            bind_current_process_to_node(self.numa_node)

        # input data layout
        self.selected_channel_localizations = (
            list(selected_channel_localizations) if selected_channel_localizations is not None else None
        )
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
            type="host_memory",
            pool_name="loader",
            node_id=self.node_id,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            numa_node=self.driver_process_numa_node,
        )

        cfg = ray.get(self.buffer_actor.get_config.remote())
        self.slot_bytes = int(cfg["slot_bytes"])
        self.batch_shape = tuple(cfg["batch_shape"])
        self._shm = attach_shared_memory(cfg["name"])

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

        if self.selected_channel_localizations is not None:
            channel_indices = resolve_channel_localization_indices(
                meta.get("channel_mapping"),
                self.selected_channel_localizations,
            )
            if self.input_format == "ZYXC" or self.input_format == "TZYXC":
                view = data_tensor[t, z, y, x, channel_indices]
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
        actual_len = len(batch["server_folder"])
        if actual_len > self.batch_size:
            raise ValueError(
                f"Batch has {actual_len} elements but batch_size is {self.batch_size}. "
                "Partitioning should ensure batch_size is never exceeded."
            )

        buffer = ray.get(self.buffer_actor.get_free.remote())
        dst = np.ndarray(
            self.batch_shape, dtype=self.buffer_dtype, buffer=self._shm.buf, offset=buffer["slot"] * self.slot_bytes
        )

        # When last_batch_policy == "pad" and batch is partial, pad with zeros
        need_pad = self.last_batch_policy == "pad" and actual_len < self.batch_size
        if need_pad:
            dst.fill(0)

        write_futs = []
        with ts.Batch() as b:
            for i in range(actual_len):
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
                if "channel_mapping" in batch:
                    meta["channel_mapping"] = batch["channel_mapping"][i]
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
        # Ray map_batches requires dict values to be list or ndarray (not Python int).
        batch["batch_size_actual"] = np.full((self.batch_size,), actual_len, dtype=np.int64)
        batch["valid_mask"] = np.array([i < actual_len for i in range(self.batch_size)], dtype=bool)

        if self.save_mode == "append_channel":
            batch["existing_zarr_path"] = np.array(
                [
                    os.path.join(
                        batch["server_folder"][i],
                        batch["output_folder"][i],
                        batch["tile_name"][i],
                    )
                    for i in range(actual_len)
                ],
                dtype=object,
            )

        if need_pad:
            for k, v in list(batch.items()):
                if k in ("buffer_name", "buffer_idx", "batch_size_actual", "valid_mask"):
                    continue
                if hasattr(v, "__len__") and len(v) == actual_len and actual_len < self.batch_size:
                    if isinstance(v, np.ndarray):
                        if v.dtype.kind in ("U", "S", "O"):
                            pad_val = v[-1] if actual_len > 0 else ""
                            batch[k] = np.concatenate([v, np.array([pad_val] * (self.batch_size - actual_len))])
                        else:
                            batch[k] = np.pad(
                                v, (0, self.batch_size - actual_len), mode="edge"
                            )
                    else:
                        pad_val = v[-1] if actual_len > 0 else None
                        batch[k] = list(v) + [pad_val] * (self.batch_size - actual_len)

        return batch


# -------- -------- dataset helpers / API -------- -------- --------


def set_data_context(cfg: DictConfig):
    ctx = ray.data.DataContext.get_current()
    ctx.use_arrow_tensor_v2 = cfg.datasets.use_arrow_tensor_v2
    ctx.execution_options.locality_with_output = cfg.datasets.locality_with_output
    ctx._enable_actor_pool_on_exit_hook = True
    # ctx.execution_options.preserve_order = cfg.datasets.preserve_order


def get_context_spec(cfg: DictConfig) -> Dict[str, Any]:
    """Build a :class:`tensorstore.Context` JSON spec from ``cfg.datasets.context``.

    **Stripping:** Keys whose value is ``None`` (Hydra ``null``) are **omitted**, so TensorStore
    uses its **default** resource for that id (see TensorStore "Context framework" docs).

    **TensorStore (google.github.io/tensorstore/context.html):**

    - ``cache_pool``: ``{"total_bytes_limit": N}`` — LRU cache soft cap in bytes. Default ``N`` is
      ``0`` (no cached bytes). Raising this can reuse decoded chunks across reads (memory tradeoff).
    - ``data_copy_concurrency``: ``{"limit": <int>}`` or ``{"limit": "shared"}`` — cap on CPU cores
      used for encode/decode/copy; ``"shared"`` uses all host cores/threads.
    - ``file_io_concurrency``: ``{"limit": <int>}`` — cap on concurrent file I/O ops (same pattern).

    **Ray Data (this file, ``map_batches``):** Parallelism across batches is mostly
    ``num_actors_min`` / ``num_actors_max`` (separate process pool), not these TensorStore limits.

    LoaderActor passes ``ctx_spec`` to ``tensorstore.Context`` and ``read_zarr(..., context=...)``.
    """
    ts_ctx = OmegaConf.to_container(cfg.datasets.context, resolve=True)
    ctx_spec = {k: v for k, v in ts_ctx.items() if v is not None}
    return ctx_spec


def _build_loader_dataset(
    cfg: DictConfig,
    local_table: pa.Table,
    ctx_spec: Dict[str, Any],
    selected_channel_localizations: Optional[List[str]] = None,
):
    dataset = ray.data.from_arrow(local_table)
    dataset = dataset.repartition(target_num_rows_per_block=cfg.datasets.rows_per_block, shuffle=False)

    scheduling_strategy = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=node_id(),
        soft=False,
    )
    if "inference" in cfg.keys():
        save_mode = cfg.inference.save_worker.save_mode
    else:
        save_mode = None
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
            "selected_channel_localizations": selected_channel_localizations,
            "local_rank": local_rank(),
            "global_rank": process_rank(),
            "node_id": node_id(),
            "numa_node": torch_gpu_to_numa(local_rank())["numa_node"],
            "dim": get_data_dim(cfg.dataset_layout_order),
            "input_format": cfg.dataset_layout_order,
            "last_batch_policy": cfg.datasets.last_batch_policy,
            "save_mode": save_mode,
        },
        compute=ray.data.ActorPoolStrategy(
            min_size=cfg.datasets.num_actors_min,
            max_size=cfg.datasets.num_actors_max,
        ),
    )
    return dataset


def get_dataset_ray(
    cfg: DictConfig,
    seed: Optional[int],
    indices: Optional[List[int]],
    sample_store_desc: MappedTableDescriptor,
    columns: Optional[List[str]] = None,
    dp_degree: Optional[int] = None,
    dp_rank: Optional[int] = None,
    selected_channel_localizations: Optional[List[str]] = None,
    shuffle: bool = False,
    last_batch_policy: str = "drop",
):
    if seed is not None and not shuffle:
        raise ValueError("Seed provided but shuffle is False.")

    set_data_context(cfg)
    ctx_spec = get_context_spec(cfg)

    store = MappedTable(sample_store_desc)
    sample_table = store.table()

    ray.logger.info(
        "[DATASET] source=%s rows=%s fingerprint=%s rank=%s",
        sample_store_desc.sample_table.source_key,
        sample_store_desc.stats.num_rows,
        sample_store_desc.stats.ordering_fingerprint,
        process_rank(),
    )

    if indices is not None:
        base_row_ids = np.asarray(indices, dtype=np.int64)
    else:
        base_row_ids = np.arange(sample_table.num_rows, dtype=np.int64)

    if dp_degree is not None and dp_rank is not None:
        ws, rk = dp_degree, dp_rank
    else:
        ws, rk = get_world_size(), process_rank()

    planner = SampleIndexPlanner(world_size=ws, rank=rk)
    local_row_ids = planner.plan_epoch(
        base_row_ids=base_row_ids,
        seed=seed,
        shuffle=shuffle,
        batch_size=cfg.clusters.batch_size_per_gpu,
        last_batch_policy=last_batch_policy,
    )

    local_table = sample_table.take(pa.array(local_row_ids, type=pa.int64()))
    if columns:
        selected_columns = [c for c in columns if c in local_table.column_names]
        local_table = local_table.select(selected_columns)

    dataset = _build_loader_dataset(
        cfg,
        local_table,
        ctx_spec,
        selected_channel_localizations=selected_channel_localizations,
    )
    return dataset, local_table.num_rows


def get_dataloader_ray(
    cfg: DictConfig,
    batch_size: int,
    collate_fn: Optional[Callable],
    epoch: int = 0,
    last_batch_policy: str = "drop",
    sample_store_desc: Optional[MappedTableDescriptor] = None,
    dp_degree: Optional[int] = None,
    dp_rank: Optional[int] = None,
    selected_channel_localizations: Optional[List[str]] = None,
):
    assert hasattr(cfg, "seed"), "cfg.seed is required for Ray Dataloader."
    if sample_store_desc is None:
        raise ValueError("sample_store_desc is required.")

    dataset_len = int(sample_store_desc.stats.num_rows)
    train_indices, val_indices = SampleIndexPlanner.split_train_val(
        num_rows=dataset_len,
        split_fraction=cfg.datasets.split,
        seed=int(cfg.seed),
    )

    if len(val_indices) > 0:
        train_dataset, train_dataset_len = get_dataset_ray(
            cfg=cfg,
            seed=int(cfg.seed) + int(epoch),
            indices=train_indices.tolist(),
            sample_store_desc=sample_store_desc,
            columns=None,
            dp_degree=dp_degree,
            dp_rank=dp_rank,
            selected_channel_localizations=selected_channel_localizations,
            shuffle=True,
            last_batch_policy=last_batch_policy,
        )
        val_dataset, val_dataset_len = get_dataset_ray(
            cfg=cfg,
            seed=None,
            indices=val_indices.tolist(),
            sample_store_desc=sample_store_desc,
            columns=None,
            dp_degree=dp_degree,
            dp_rank=dp_rank,
            selected_channel_localizations=selected_channel_localizations,
            shuffle=False,
            last_batch_policy=last_batch_policy,
        )

        record_dataset_len(cfg, train_dataset_len, val_dataset_len)

        train_dataloader = train_dataset.iterator()._iter_batches(
            batch_size=batch_size, _finalize_fn=collate_fn, batch_format="numpy"
        )
        val_dataloader = val_dataset.iterator()._iter_batches(
            batch_size=batch_size, _finalize_fn=collate_fn, batch_format="numpy"
        )
        return train_dataloader, val_dataloader, None

    train_dataset, train_dataset_len = get_dataset_ray(
        cfg=cfg,
        seed=int(cfg.seed) + int(epoch),
        indices=train_indices.tolist() if len(train_indices) > 0 else None,
        sample_store_desc=sample_store_desc,
        columns=None,
        dp_degree=dp_degree,
        dp_rank=dp_rank,
        selected_channel_localizations=selected_channel_localizations,
        shuffle=True,
        last_batch_policy=last_batch_policy,
    )
    record_dataset_len(cfg, train_dataset_len, 0)

    train_dataloader = train_dataset.iterator()._iter_batches(
        batch_size=batch_size, _finalize_fn=collate_fn, batch_format="numpy"
    )
    return train_dataloader, None, None