import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable, Callable, Literal

import numpy as np

import tensorstore as ts

from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import instantiate, get_method

from data.io import read_zarr, load_hypercubes_dataframe
from data.data_types import NUMPY_DTYPES, TENSORSTORE_DTYPES, TORCH_DTYPES

import ray
from ray.data.block import Block, BlockMetadata
from ray.data.datasource import Datasource, ReadTask
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder

import torch

import pyarrow as pa

from data.data_shapes import MULTICHANNEL_HYPERCUBE
from utils.profiling import pprof_func, pprof_class


@pprof_class
class PinnedTensorCollator:
    """
    Convert FixedShapeTensorArray/FixedSizeListArray/ArrowTensorArray 
    column in Arrow to a pinned Torch tensor using one host-side copy. 
    """

    def __init__(self, 
                 dtype: str, 
                 sample_shape: List[int], 
                 pin_memory: bool = True,
                 impl_type: Literal["FixedShapeTensorArray", 
                                    "FixedSizeListArray"] = "FixedSizeListArray",
                 copy_to_pinned_array: bool = False # to stack and copy data to pinned memory
    ):
        self.impl_type = impl_type
        self.pin_memory = pin_memory
        self.sample_shape = sample_shape
        self.dtype = TORCH_DTYPES[dtype].value
        self.copy_to_pinned_array = copy_to_pinned_array

    def _create_pinned_mem_array(self, batch_size: int, pin_memory: bool) -> None:
        shape = (batch_size, *self.sample_shape)
        data_tensor = torch.empty(shape,
                                  dtype=self.dtype,
                                  pin_memory=pin_memory)
        return data_tensor

    def _copy_arrow_tensor_array(self,
                                 chunks_arr,
                                 pin_memory: bool) -> torch.Tensor:
        if self.copy_to_pinned_array:
            arr = self._create_pinned_mem_array(
                batch_size=chunks_arr.num_chunks,
                pin_memory=pin_memory
            )
        else:
            arr = []

        offset = 0
        for item in chunks_arr.chunks:           
            item = torch.from_numpy(item.to_numpy())

            if item.dtype != self.dtype:
                # ray.logger.warning(f"Casting colatted data to {self.dtype}")
                item = item.to(self.dtype)

            if self.copy_to_pinned_array:
                # copy to pinned memory (should be only real copy)
                arr[offset:offset + item.shape[0]].copy_(item, non_blocking=True)
            else:
                arr.append(item)

            offset += item.shape[0]

        return arr

    def _copy_arrow_list_array(self,
                      chunks_arr: pa.ChunkedArray,
                      pin_memory: bool) -> torch.Tensor:
        if self.copy_to_pinned_array:
            arr = self._create_pinned_mem_array(
            batch_size=chunks_arr.num_chunks,
            pin_memory=pin_memory
            )
        
        else:
            arr = []

        offset = 0
        for chunk in chunks_arr.chunks:
            M = len(chunk)
            flat = chunk.values.to_numpy(zero_copy_only=True)
            np_view = flat.reshape(M, *self.sample_shape)
            tensor = torch.from_numpy(np_view)

            if tensor.dtype != self.dtype:
                # ray.logger.warning(f"Casting colatted data to {self.dtype}")
                tensor = tensor.to(self.dtype)

            if self.copy_to_pinned_array:
                # copy to pinned memory (should be only real copy)
                arr[offset:offset+M].copy_(tensor, non_blocking=True)
            else:
                arr.append(tensor)

            offset += M

        return arr

    def __call__(self, batch: pa.Table | pa.RecordBatch) -> Dict[str, Any]:
        t0 = time.time()

        chunks_arr = batch.column('data_tensor')
        if self.impl_type == "FixedShapeTensorArray":
            data_tensor = self._copy_arrow_tensor_array(chunks_arr, pin_memory=self.pin_memory)
        elif self.impl_type == "FixedSizeListArray":
            data_tensor = self._copy_arrow_list_array(chunks_arr, pin_memory=self.pin_memory)
        else:
            raise ValueError(f"Unsupported impl_type: {self.impl_type}")
        
        # assert data_tensor.dtype == self.dtype, f"{data_tensor.dtype=} != {self.dtype=}" 

        meta = {name: batch.column(name).to_pylist()
                for name in batch.schema.names
                if name != 'data_tensor'}
        meta['collate_time'] = time.time() - t0

        return {"data_tensor": data_tensor, "metainfo": meta}


# -------- -------- -------- v1 -------- -------- --------


def _slice_hypercube(data_tensor, 
                     meta: Dict[str, Any], 
                     channels_subset: Optional[List[int]]
) -> np.ndarray:
    t = slice(meta["time_start"], meta["time_start"] + meta["time_size"])
    z = slice(meta["z_start"], meta["z_start"] + meta["cube_size"])
    y = slice(meta["y_start"], meta["y_start"] + meta["cube_size"])
    x = slice(meta["x_start"], meta["x_start"] + meta["cube_size"])

    if channels_subset is None:
        c = slice(0, meta["channel_size"])
        view = data_tensor[t, z, y, x, c]
    else:
        view = data_tensor[t, z, y, x, channels_subset]

    return view.read().result()


def _load_cube(meta: Dict[str, Any], dtype, channels_subset) -> np.ndarray:
    handle = read_zarr(
        os.path.join(meta["server_folder"], meta["output_folder"], meta["tile_name"]),
        dtype=dtype,
    )
    cube = _slice_hypercube(handle, meta, channels_subset)
    del handle
    return cube


def _read_block(records: List[Dict[str, Any]], 
                timing: bool, 
                dtype, 
                channels_subset
) -> Iterable[Block]:
    builder = DelegatingBlockBuilder()
    for meta in records:
        t0 = time.time() if timing else None
        img_tensor = _load_cube(meta, dtype, channels_subset)

        # TODO: numpy support for ImageList
        # img_sample = ImageList(
        #     torch.from_numpy(img_tensor),
        #     layout=self.input_layout,
        #     image_sizes=[img_tensor.shape]
        # )
        # NOTE:
        # (1) if we support numpy in ImageList, we can use:
        # builder.add({**data_sample.to_dict()})
        if timing:
            builder.add({"data_tensor": img_tensor, "slice_time": time.time() - t0})
        else:
            builder.add({"data_tensor": img_tensor})

    block = builder.build()
    yield block


# based on: https://github.com/ray-project/ray/python/ray/data/datasource/file_based_datasource.py
class PretrainDatasourceRay(Datasource):
    """
    Ray Datasource that reads one Zarr hypercube per block.

    Each CSV row must contain:
        server_folder, output_folder, tile_name,
        time_start, time_size,
        z_start, y_start, x_start, cube_size,
        channel_size
    """

    def __init__(self,
         hypercubes_dataframe_path: Path,
         input_layout: MULTICHANNEL_HYPERCUBE,
         server_folder_path: Optional[Path] = None,
         dtype: TENSORSTORE_DTYPES = TENSORSTORE_DTYPES.fp16,
         indices: Optional[List[int]] = None,
         time: bool = True,
         max_rois: Optional[int] = None,
         max_tiles: Optional[int] = None,
         max_hypercubes: Optional[int] = None,
         hpf_list: Optional[Iterable[int]] = None,
         roi_list: Optional[Iterable[int]] = None,
         tile_list: Optional[Iterable[str]] = None,
         occupancy_threshold: Optional[float] = None,
         channels_subset: Optional[List[int]] = None
    ):
        self.input_layout = input_layout

        self.channels_subset = list(channels_subset) \
            if channels_subset is not None else None

        hypercubes_dataframe_path = Path(hypercubes_dataframe_path)
        if not hypercubes_dataframe_path.exists():
            raise FileNotFoundError(hypercubes_dataframe_path)

        self.server_folder_path = str(server_folder_path) if server_folder_path else None
        self.hypercubes_dataframe, self.hypercubes_dataframe_config = load_hypercubes_dataframe(
            hypercubes_dataframe_path=hypercubes_dataframe_path,
            server_folder_path=server_folder_path,
            max_rois=max_rois,
            max_tiles=max_tiles,
            max_hypercubes=max_hypercubes,
            hpf_list=hpf_list,
            roi_list=roi_list,
            tile_list=tile_list,
            occupancy_threshold=occupancy_threshold
        )

        if indices is not None and len(indices) > 0:
            self.hypercubes_dataframe = self.hypercubes_dataframe.iloc[indices].reset_index(drop=True)

        self._hypercubes_records: List[Dict[str, Any]] = self.hypercubes_dataframe.to_dict(orient="records")
        self._dtype = TENSORSTORE_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        # pre-compute bytes / cube for size estimates
        self._bytes_per_cube = self._compute_bytes_per_record(record=self._hypercubes_records[0], dtype=self._dtype)

        self.time = time

    def _compute_bytes_per_record(self, record: Dict[str, Any], dtype: TENSORSTORE_DTYPES) -> int:
        if self.channels_subset is not None:
            voxels = (
                record["time_size"] * len(self.channels_subset) * record["cube_size"] ** 3
            )
        else:
            voxels = (
                    record["time_size"] * record["channel_size"] * record["cube_size"] ** 3
            )
        if dtype == ts.float16 or dtype == ts.bfloat16:
            return voxels * 2
        elif dtype == ts.float32:
            return voxels * 4
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def get_name(self) -> str:
        return "PretrainHypercube"

    def estimate_inmemory_data_size(self) -> int:
        return self._bytes_per_cube * len(self._hypercubes_records)

    # get_read_tasks returns a list of ReadTask objects, each containing a
    # ReadTask which is a class that wraps a read task function with associated
    # metadata. the read task function returns an iterable which yields blocks of data.
    # blocks may be built using a DelegatingBlockBuilder, which allows for passing
    # rows of data to the block builder followed by a build() call.
    def get_read_tasks(self, parallelism: int) -> List[ReadTask]:
        # parallelism is user configured or inferred by Ray
        parallelism = min(parallelism, len(self._hypercubes_records))
        splits = np.array_split(self._hypercubes_records, parallelism)

        tasks: List[ReadTask] = []
        for shard in splits:
            if len(shard) == 0:
                continue

            # avoid big serialization
            shard_ref = ray.put(list(shard))
            dtype = self._dtype
            timing = self.time
            channels_subset = self.channels_subset

            def _make_read_task(records_ref=shard_ref, 
                                dtype=dtype, 
                                timing=timing, 
                                channels_subset=channels_subset
            ):
                return _read_block(records=ray.get(records_ref), 
                                   timing=timing, 
                                   dtype=dtype, 
                                   channels_subset=channels_subset
                )

            # NOTE: we have seen issues before with Ray's fifo_bundle_queue
            #       where we get:
            #       `AssertionError: Expected the total size of
            #       objects in the queue to be non-negative, but
            #       got -134217744 bytes instead.`
            #       if this persists, consider setting size_bytes to None
            #       or debug further
            shard_size = sum(self._compute_bytes_per_record(r, self._dtype) for r in shard)

            meta = BlockMetadata(
                num_rows=len(shard),
                size_bytes=shard_size,
                input_files=None,
                exec_stats=None,
                schema=None
            )
            tasks.append(ReadTask(_make_read_task, meta))

        return tasks
    

# -------- -------- -------- v2 -------- -------- --------


@pprof_class
class RayLoaderActor:
    """
    Ray actor that loads hypercubes from a Arrow Table.
    Used for Ray Data v2.
    """
    def __init__(self, 
                 context_spec: Dict[str, Any],
                 input_layout: str,
                 with_batched_api: bool = True, 
                 dtype: str = "fp16",
                 impl_type: Literal["FixedShapeTensorArray", 
                                    "FixedSizeListArray"] = "FixedSizeListArray",
                 channels_subset: Optional[List[int]] = None
):
        self.impl_type = impl_type
        self.ctx = ts.Context(context_spec)
        self.channels_subset = list(channels_subset) \
            if channels_subset is not None else None
        self.input_layout = input_layout.upper()
        self.with_batched_api = with_batched_api
        self.dtype = TENSORSTORE_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        if self.dtype == TENSORSTORE_DTYPES.bf16.value:
            # ray.logger.warning(
            #     "Using fp16 for PyArrow, Collator will cast data to bf16"
            # )
            self.dtype = TENSORSTORE_DTYPES.fp16.value

        self._handles = {}  # lazy loading of handles

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

        return view.read(batch=ts_batch, order="C")

    def _get_handle(self, path: str):
        h = self._handles.get(path)
        if h is None:
            h = read_zarr(path, dtype=self.dtype, context=self.ctx)
            self._handles[path] = h
        return h
    
    def _np_as_strided_view(self, array: np.ndarray) -> np.ndarray:
        return np.lib.stride_tricks.as_strided(
            array,
            shape=(1, *array.shape),
            strides=(array.nbytes, *array.strides),
        )
    
    def _get_sample_size(self, record) -> int:
        if self.input_layout == 'TZYXC':
            if self.channels_subset is not None:
                return int(record["time_size"] * record["cube_size"] ** 3 * len(self.channels_subset))
            else:
                return int(record["time_size"] * record["cube_size"] ** 3 * record["channel_size"])
        else:
            #TODO: add support for other layouts
            raise ValueError(f"Unsupported dataset layout order: {self.input_layout}")

    def __call__(self, batch):
        read_time = time.perf_counter()
        records = batch.to_pylist()

        futs = []
        if self.with_batched_api:
            with ts.Batch() as b:
                for i, r in enumerate(records):
                    p = os.path.join(r["server_folder"], r["output_folder"], r["tile_name"])
                    futs.append(self._slice_hypercube(self._get_handle(p), r, ts_batch=b))
        else:
            for i, r in enumerate(records):
                p = os.path.join(r["server_folder"], r["output_folder"], r["tile_name"])
                futs.append(self._slice_hypercube(self._get_handle(p), r, ts_batch=None))

        arrays = [f.result() for f in futs]

        # NOTE: FixedSizeListArray seems to be faster 
        #       based on initial tests
        if self.impl_type == "FixedSizeListArray":            
            chunks = []
            for a in arrays:
                flat = a.reshape(-1)
                values = pa.array(flat)
                fs1 = pa.FixedSizeListArray.from_arrays(values, 
                                                        list_size=self._get_sample_size(records[0]))
                chunks.append(fs1)

            col = pa.chunked_array(chunks)
        
        elif self.impl_type == "FixedShapeTensorArray":
            chunks  = [pa.FixedShapeTensorArray.from_numpy_ndarray(self._np_as_strided_view(a)) 
                        for a in arrays]
            col = pa.chunked_array(chunks)
        
        else:
            raise ValueError(f"Unsupported impl_type: {self.impl_type}")
        
        read_time = time.perf_counter() - read_time
        read_time_col = pa.array([read_time] * batch.num_rows, type=pa.float64())

        batch = batch.append_column("data_tensor", col)
        return batch.append_column("read_time", read_time_col)


# -------- -------- --------  -------- -------- --------


def get_dataset_ray(cfg: DictConfig, 
                    indices: Optional[List[int]], 
                    database: Optional[Any] = None
):
    if cfg.datasets.channels_subset is not None:
        num_channels = cfg.datasets.input_shape[cfg.dataset_layout_order.index("C")]
        assert len(list(cfg.datasets.channels_subset)) == num_channels, \
            f"channels_subset length {len(cfg.datasets.channels_subset)} " \
            f"does not match number of channels {num_channels} in input_shape {cfg.datasets.input_shape}"

    if not cfg.datasets.ray_data_v2:
        datasource = instantiate(
            cfg.datasets.dataset,
            hypercubes_dataframe_path=cfg.datasets.databases.hypercubes_dataframe_path,
            server_folder_path=cfg.paths.server_folder_path,
            dtype=cfg.dataset_dtype,
            input_layout=cfg.datasets.dataset.input_layout,
            indices=indices,
            max_rois=cfg.datasets.max_rois,
            max_tiles=cfg.datasets.max_tiles,
            max_hypercubes=cfg.datasets.max_hypercubes,
            hpf_list=cfg.datasets.hpf_list,
            roi_list=cfg.datasets.roi_list,
            tile_list=cfg.datasets.tile_list,
            occupancy_threshold=cfg.datasets.occupancy_threshold,
            channels_subset=cfg.datasets.channels_subset
        )

        dataset = ray.data.read_datasource(datasource, 
                                        #    ray_remote_args={
                                        #          "num_cpus": cfg.datasets.ray_remote_args.num_cpus}
                                        )

        # set Data Context for the dataset
        ctx = ray.data.DataContext.get_current()
        # ctx.execution_options.locality_with_output = cfg.datasets.locality_with_output
        ctx.use_arrow_tensor_v2 = cfg.datasets.use_arrow_tensor_v2

        # we include transforms here for completion but if data loading
        # is performance critical, the transforms may also be applied
        # in the preprocessor on device
        transforms = []
        for t in cfg.datasets.transforms.transforms_list:
            if isinstance(t, DictConfig):
                transforms.append(instantiate(t))
            else:
                transforms.append(get_method(t))

        for transform in transforms:
            dataset = dataset.map_batches(transform, 
                                        batch_size=cfg.clusters.batch_size_per_gpu)
        return dataset
    
    else:
        # set Data Context for the dataset
        ray_ctx = ray.data.DataContext.get_current()
        # ray_ctx.execution_options.locality_with_output = cfg.datasets.locality_with_output
        ray_ctx.use_arrow_tensor_v2 = cfg.datasets.use_arrow_tensor_v2

        ts_ctx = OmegaConf.to_container(cfg.datasets.context, resolve=True)
        # remove None values from context spec
        ctx_spec = {k: v for k, v in ts_ctx.items() if v is not None}

        table = pa.table(database.hypercubes_dataframe.iloc[indices] \
                         if indices is not None else database.hypercubes_dataframe)
        
        # convert to Ray Dataset
        dataset = ray.data.from_arrow(table).repartition(
            target_num_rows_per_block=cfg.datasets.rows_per_block,
            shuffle=False
        )

        dataset = dataset.map_batches(
            RayLoaderActor,
            batch_size=cfg.clusters.batch_size_per_gpu,
            batch_format="pyarrow",
            fn_constructor_kwargs={
                "context_spec": ctx_spec,
                "with_batched_api": cfg.datasets.with_batched_api,
                "dtype": cfg.dataset_dtype,
                "impl_type": cfg.datasets.impl_type,
                "input_layout": cfg.datasets.dataset.input_layout.value,
                "channels_subset": cfg.datasets.channels_subset
            },
            # consider fractional values for jobs with much I/O, i.e.
            # num_cpus=0.5,
            concurrency=(cfg.datasets.num_actors_min, 
                         cfg.datasets.num_actors_max),
        )

        return dataset


def get_dataloader_ray(dataset: ray.data.Dataset,
                       batch_size: int,
                       drop_last: bool = True,
                       collate_fn: Optional[Callable] = None,
                       prefetch_factor: int = None,
                       auto_transfer: bool = False
):
    # we use _iter_batches instead of iter_torch_batches
    # to avoid a costly conversion to torch tensors that
    # the current implementation of iter_torch_batches does
    if auto_transfer:
        wrapped_loader = _WrappedRayDataLoader(
            dataset._iter_batches(
                batch_size=batch_size,
                prefetch_batches=prefetch_factor,
                _collate_fn=collate_fn,
                batch_format="pyarrow"
            ),
            device=torch.cuda.current_device(),
            tensor_keys=("data_tensor",)
        )
        return wrapped_loader
    else:
        return dataset._iter_batches(
            batch_size=batch_size,
            prefetch_batches=prefetch_factor,
            drop_last=drop_last,
            _collate_fn=collate_fn,
            batch_format="pyarrow"
        )


# based on: _WrappedDataLoader in ray/train/torch/train_loop_utils.py
class _WrappedRayDataLoader:
    """
    Wrap any iterator that yields batches with Torch tensors.
    Each call prefetches the NEXT batch to GPU on its own
    stream while the default stream works on the current batch.
    """

    def __init__(self, data_iter, device, tensor_keys=("data_tensor",)):
        self.device, self.data_iter = device, iter(data_iter)

        self.tensor_keys = (tensor_keys
                            if isinstance(tensor_keys, (tuple, list))
                            else (tensor_keys,))
        self._stream = torch.cuda.Stream(device=device)

        self._next_batch = None
        # TODO: move this later in the training logic?
        self._prefetch_next_batch()

    def _to_cuda(self, batch):
        """Non-recursive helper that only touches the listed keys."""
        with torch.cuda.stream(self._stream):
            for k in self.tensor_keys:
                batch[k] = batch[k].to(self.device, non_blocking=True)
        return batch

    def _prefetch_next_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self._next_batch = None
            return
        self._next_batch = self._to_cuda(batch)

    def __iter__(self):
        return self

    def __next__(self):
        if self._next_batch is None:
            raise StopIteration

        batch = self._next_batch

        # Reference:
        # https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html
        # The training stream (current) needs to wait until
        # the memory copy stream finishes.
        torch.cuda.current_stream().wait_stream(self._stream)

        # When a tensor is used by CUDA streams different from
        # its original allocator, we need to call `record_stream`
        # to inform the allocator of all these streams. Otherwise,
        # the tensor might be freed once it is no longer used by
        # the creator stream.
        for k in self.tensor_keys:
            batch[k].record_stream(torch.cuda.current_stream())

        # prefetch the NEXT batch while we are computing on the current one
        # is being processed
        self._prefetch_next_batch()
        return batch