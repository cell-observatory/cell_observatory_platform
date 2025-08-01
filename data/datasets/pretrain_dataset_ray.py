import os
import gc
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable, Callable

import ujson
import numpy as np
import pandas as pd

from omegaconf import DictConfig
from hydra.utils import instantiate, get_method

from data.io import read_zarr
from data.data_types import NUMPY_DTYPES, TENSORSTORE_DTYPES

import ray
from ray.data.block import Block, BlockMetadata
from ray.data.datasource import Datasource, ReadTask
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder

import tensorstore as ts

import torch

from data.data_shapes import MULTICHANNEL_HYPERCUBE


def arrow_pinned_to_mem(chunked, reuse_buf=None):
    pinned = torch.empty((32, 16, 128, 128, 128, 2),
                             dtype=torch.float16,
                             pin_memory=True)
    
    offset = 0
    for item in chunked.chunks:
        item =  torch.from_numpy(item.to_numpy())
        pinned[offset:offset+item.shape[0]].copy_(item, non_blocking=True)
        offset += item.shape[0]
        
    return pinned


def arrow_chunked_to_pinned(chunked, reuse_buf=None):
    arr = chunked.combine_chunks()
    np_arr = arr.to_numpy()
    if reuse_buf is None or reuse_buf.shape != np_arr.shape:
        pinned = torch.empty(np_arr.shape,
                             dtype=torch.float16,
                             pin_memory=True)
    else:
        pinned = reuse_buf

    pinned.copy_(torch.from_numpy(np_arr))
    return pinned

def pyaarrow_chunks_to_torch2(chunks):
    """ https://github.com/ray-project/ray/issues/50128 """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinned = []
    batch_size = 0
    for item in chunks.chunks:
        item = item.to_numpy()
        item = torch.from_numpy(item)
        # item = item.pin_memory()
        pinned.append(item)
        batch_size += item.shape[0]
    shape = list(pinned[0].shape)
    shape[0] = batch_size
    tensor = torch.zeros(torch.Size(shape), device='cpu', dtype=torch.float16)
    offset = 0
    for item in pinned:
        tensor[offset:offset+item.shape[0]].copy_(item, non_blocking=True)
        offset += item.shape[0]
    return tensor


def base_collate_fn_ray(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the dict-of-arrays produced by Ray's `iter_torch_batches`
    into the structure expected by the rest of the training loop:
        {
            "data_tensor":  Tensor,
            "metainfo":     { <all other cols> }
        }
    """
    t0 = time.time()
    np_img = batch.column("data_tensor")
    # data_tensor = torch.as_tensor(np_img).pin_memory()
    data_tensor = arrow_pinned_to_mem(np_img)

    metainfo = {'slice_time': torch.zeros((1, 2))}
    # for k, v in batch.items():
    #     if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
    #         metainfo[k] = torch.as_tensor(v)
    #     else:
    #         metainfo[k] = v

    metainfo["collate_time"] = time.time() - t0

    # TODO: temporary logic for moving data to GPU, after we figure out all disk->cpu
    #       bottlenecks, we should augment this logic to move data to GPU in a separate stream
    #       with one batch prefetching
    return {"data_tensor": data_tensor, "metainfo": metainfo}


def _slice_hypercube(data_tensor, meta: Dict[str, Any]) -> np.ndarray:
    t = slice(meta["time_start"], meta["time_start"] + meta["time_size"])
    c = slice(0, meta["channel_size"])
    z = slice(meta["z_start"], meta["z_start"] + meta["cube_size"])
    y = slice(meta["y_start"], meta["y_start"] + meta["cube_size"])
    x = slice(meta["x_start"], meta["x_start"] + meta["cube_size"])
    return data_tensor[t, z, y, x, c].read().result()


def _load_cube(meta: Dict[str, Any], dtype) -> np.ndarray:
    handle = read_zarr(
        os.path.join(meta["server_folder"], meta["output_folder"], meta["tile_name"]),
        dtype=dtype,
    )
    cube = _slice_hypercube(handle, meta)
    del handle
    return cube


def _read_block(records: List[Dict[str, Any]], timing: bool, dtype) -> Iterable[Block]:
    builder = DelegatingBlockBuilder()
    for meta in records:
        t0 = time.time() if timing else None
        img_tensor = _load_cube(meta, dtype)

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
    ):
        self.input_layout = input_layout
        
        hypercubes_dataframe_path = Path(hypercubes_dataframe_path)
        if not hypercubes_dataframe_path.exists():
            raise FileNotFoundError(hypercubes_dataframe_path)
        
        self.server_folder_path = str(server_folder_path) if server_folder_path else None
        self.hypercubes_dataframe, self.hypercubes_dataframe_config = self._process_tables(hypercubes_dataframe_path)

        if indices is not None:
            self.hypercubes_dataframe = self.hypercubes_dataframe.iloc[indices].reset_index(drop=True)

        self._hypercubes_records: List[Dict[str, Any]] = self.hypercubes_dataframe.to_dict(orient="records")
        self._dtype = TENSORSTORE_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        # pre-compute bytes / cube for size estimates
        self._bytes_per_cube = self._compute_bytes_per_cube(self._hypercubes_records, self._dtype)

        self.time = time

    def _compute_bytes_per_cube(self, records: List[Dict[str, Any]], dtype: TENSORSTORE_DTYPES) -> int:
        sample = records[0]
        voxels = (
            sample["time_size"] * sample["channel_size"] * sample["cube_size"] ** 3
        )
        if dtype == ts.float16:
            return voxels * 2
        elif dtype == ts.float32:
            return voxels * 4
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def _process_tables(self, hypercubes_dataframe_path) -> tuple[pd.DataFrame, Dict]:
        if not hypercubes_dataframe_path.exists():
            raise FileNotFoundError(f"{hypercubes_dataframe_path} does not exist")

        hypercubes = pd.read_csv(hypercubes_dataframe_path, index_col=0, header=0)
        with open(hypercubes_dataframe_path.with_suffix('.json'), 'r') as f:
            configs = ujson.load(f)

        if self.server_folder_path is not None:
            hypercubes['server_folder'] = self.server_folder_path
        return hypercubes, configs

    def get_name(self) -> str:
        return "PretrainHypercube"

    def estimate_inmemory_data_size(self) -> int:
        return self._bytes_per_cube * len(self._hypercubes_records)

    # get_read_tasks returns a list of ReadTask objects, each containing a 
    # ReadTask which is a class that wraps a read task function with associated
    # metadata. the read task function returns an iterable which yields blocks of data.
    # blocks may be built using a DelegatingBlockBuilder, which allows for passing
    # rows of data to the block builder followed by a build() call. in general, this leverages
    # a API call to example Pandas (or other backend) which generates a DataFrame from dicts of 
    # data and concats tables as needed. 
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

            def _make_read_task(records_ref=shard_ref, dtype=dtype, timing=timing):
                return _read_block(ray.get(records_ref), timing, dtype)

            meta = BlockMetadata(
                num_rows=len(shard),
                size_bytes=self._bytes_per_cube * len(shard),
                schema=None,
                input_files=None,
                exec_stats=None,
            )
            tasks.append(ReadTask(_make_read_task, meta))

        return tasks


def get_dataset_ray(cfg: DictConfig, indices: Optional[List[int]]):
        datasource = instantiate(cfg.datasets.dataset,
                              hypercubes_dataframe_path=cfg.datasets.databases.hypercubes_dataframe_path,
                              server_folder_path=cfg.paths.server_folder_path,
                              dtype=cfg.dataset_dtype,
                              input_layout=cfg.datasets.dataset.input_layout,
                              indices=indices)
        
        dataset = ray.data.read_datasource(datasource)

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
                                          batch_size=cfg.clusters.batch_size_per_gpu, 
                                          batch_format="pandas")
        return dataset


def get_dataloader_ray(dataset: ray.data.Dataset,
                       batch_size: int,
                       drop_last: bool = False,
                       collate_fn: Optional[Callable] = None,
                       prefetch_factor: int = None):
    # return dataset.iter_torch_batches(
    #     batch_size=batch_size,
    #     prefetch_batches=prefetch_factor,
    #     drop_last=drop_last,
    #     collate_fn=collate_fn
    # )
    return dataset._iter_batches(
        batch_size=batch_size, 
        prefetch_batches=prefetch_factor,
        _collate_fn=collate_fn,
        batch_format=None
    )