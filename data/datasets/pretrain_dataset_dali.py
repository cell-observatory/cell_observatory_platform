import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Callable, Optional, Literal

import pandas as pd
import ujson

from hydra.utils import get_method

import torch

import nvidia.dali as dali
from nvidia.dali import pipeline_def, fn, types

from data.io import read_zarr
from utils.context import process_rank, get_world_size
from data.data_types import TENSORSTORE_DTYPES, NUMPY_DTYPES, TORCH_DTYPES, DALI_DTYPES

# based on: 
# https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
# examples/general/data_loading/parallel_external_source.html
class PretrainDatasetDali:
    def __init__(
        self,
        input_layout,
        hypercubes_dataframe_path,
        batch_size: int,
        dtype: NUMPY_DTYPES | TENSORSTORE_DTYPES | TORCH_DTYPES | DALI_DTYPES | str = NUMPY_DTYPES.fp16,
        time: Optional[bool] = True,
        transforms: Optional[Callable] = None,
        indices: Optional[list[int]] = None
    ):
        self.input_layout = input_layout
        self.dtype = DALI_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        self.hypercubes_dataframe_path = Path(hypercubes_dataframe_path)
        self.hypercubes_dataframe, self.hypercubes_dataframe_config = self._process_tables(self.hypercubes_dataframe_path)

        if indices is not None:
            self.hypercubes_dataframe = self.hypercubes_dataframe.iloc[indices].reset_index(drop=True)

        self.paths = {
            os.path.join(sf, of, tn)
            for sf, of, tn in 
            zip(self.hypercubes_dataframe["server_folder"], 
                self.hypercubes_dataframe["output_folder"], 
                self.hypercubes_dataframe["tile_name"]
            )
        }

        self._zarr_handles_data = {
            p: read_zarr(p, dtype=dtype)
            for p in self.paths
        }

        self.bs = batch_size
        self.shard_id = process_rank()
        self.num_shards = get_world_size()

        self.shard_size = len(self.hypercubes_dataframe) // self.num_shards
        self.shard_offset = self.shard_size * self.shard_id

        # if the shard size is not divisible by the batch size, the last
        # incomplete batch will be omitted.        
        self.full_iterations = self.shard_size // batch_size
        self.perm = None  # permutation of indices
        self.last_seen_epoch = (
            # so that we don't have to recompute the `self.perm` for every sample
            None
        )

        self._build_index()

        self.time = time

        self.transforms = [get_method(t) for t in transforms] if transforms is not None else []

    def _process_tables(self, hypercubes_dataframe_path) -> tuple[pd.DataFrame, Dict]:
        if not hypercubes_dataframe_path.exists():
            raise FileNotFoundError(f"{hypercubes_dataframe_path} does not exist")

        hypercubes = pd.read_csv(hypercubes_dataframe_path, index_col=0, header=0)
        with open(hypercubes_dataframe_path.with_suffix('.json'), 'r') as f:
            configs = ujson.load(f)

        return hypercubes, configs
    
    def _build_index(self) -> None:
        # convert df into a list of Python dicts
        self._index = self.hypercubes_dataframe.to_dict(orient="records")

    def _slice_hypercube(self, data_tensor, meta: Dict[str, Any]) -> Tuple[int]:
        t, c = slice(meta["time_start"], meta["time_start"] + meta["time_size"]), slice(0, meta["channel_size"])
        z = slice(meta["z_start"], meta["z_start"] + meta["cube_size"])
        y = slice(meta["y_start"], meta["y_start"] + meta["cube_size"])
        x = slice(meta["x_start"], meta["x_start"] + meta["cube_size"])
        return data_tensor[t, z, y, x, c].read().result()

    def _load_sample(self, meta: Dict[str, Any]) -> np.ndarray | Dict[str, Any]:
        """Read raw image crop into memory."""
        data_tensor = self._zarr_handles_data[
            os.path.join(meta["server_folder"], meta["output_folder"], meta["tile_name"])
        ]
        img = self._slice_hypercube(data_tensor, meta)
        return img

    def __call__(self, sample_info):
        if self.time:
            start_time = time.time()
        if sample_info.iteration >= self.full_iterations:
            # indicate end of the epoch
            raise StopIteration
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            self.perm = np.random.default_rng(seed=42 + sample_info.epoch_idx)
            self.perm = self.perm.permutation(len(self.hypercubes_dataframe))

        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]
        sample = self._index[sample_idx]
        data = self._load_sample(sample)

        # TODO: add support for timing data load with DALI
        if self.time:
            data_time = np.array(time.time() - start_time, dtype=np.float32)
            return data, data_time
        else:
            return data


@pipeline_def
def pretrain_dataset_pipeline(dataset):
    vols = fn.external_source(
        source=dataset,
        num_outputs=2 if dataset.time else 1,
        batch=False,
        parallel=True,
        device="cpu",
        dtype=[dataset.dtype, types.FLOAT] if dataset.time else dataset.dtype,
        ndim=[dataset.input_layout.ndim, 0] if dataset.time else dataset.input_layout.ndim,
    )
    vol = vols[0].gpu()

    # apply transforms if any
    for transform in dataset.transforms:
        vol = transform(vol, dataset.dtype)

    if dataset.time:
        return vol, vols[1]
    else:
        return vol
    
