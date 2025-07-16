import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import ujson

import nvidia.dali as dali
from nvidia.dali import pipeline_def, fn, types

from data.io import read_zarr
from data.data_types import DALI_DTYPES, TENSORSTORE_DTYPES
from utils.context import process_rank, get_world_size


# based on: 
# https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
# examples/general/data_loading/parallel_external_source.html
class PretrainDatasetDali:
    def __init__(self, 
                 input_layout,
                 hypercubes_dataframe_path, 
                 server_folder_path,
                 batch_size: int,
                 ndim: int = 5,
                 target_dtype: str = "float16",
    ):
        self.ndim = ndim
        self.input_layout = input_layout
        self.target_dtype = target_dtype

        self.server_folder_path = server_folder_path
        self.hypercubes_dataframe_path = Path(hypercubes_dataframe_path)
        self.hypercubes_dataframe, self.hypercubes_dataframe_config = self._process_tables(self.hypercubes_dataframe_path)

        self.paths = {
            os.path.join(sf, of, tn)
            for sf, of, tn in 
            zip(self.hypercubes_dataframe["server_folder"], 
                self.hypercubes_dataframe["output_folder"], 
                self.hypercubes_dataframe["tile_name"]
            )
        }

        self._zarr_handles_data = {
            p: read_zarr(p, dtype=TENSORSTORE_DTYPES[self.target_dtype] \
                            if isinstance(self.target_dtype, str) else self.target_dtype)
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

    def _process_tables(self, hypercubes_dataframe_path) -> tuple[pd.DataFrame, Dict]:
        if not hypercubes_dataframe_path.exists():
            raise FileNotFoundError(f"{hypercubes_dataframe_path} does not exist")

        hypercubes = pd.read_csv(hypercubes_dataframe_path, index_col=0, header=0)
        with open(hypercubes_dataframe_path.with_suffix('.json'), 'r') as f:
            configs = ujson.load(f)

        if self.server_folder_path is not None:
            hypercubes['server_folder'] = self.server_folder_path
        return hypercubes, configs
    
    def _build_index(self) -> None:
        # convert df into a list of Python dicts
        self._index = self.hypercubes_dataframe.to_dict(orient="records")

    def _slice_hypercube(self, data_tensor, meta: Dict[str, Any]) -> Tuple[int]:
        t, c = slice(meta["time_start"], meta["time_start"] + meta["time_size"]), slice(0, meta["channel_size"])
        z = slice(meta["z_start"]-28, meta["z_start"] + meta["cube_size"]-28)
        y = slice(meta["y_start"], meta["y_start"] + meta["cube_size"])
        x = slice(meta["x_start"]-14, meta["x_start"] + meta["cube_size"]-14)
        return data_tensor[t, z, y, x, c].read().result()

    def _load_sample(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Read raw image crop into memory."""
        data_tensor = self._zarr_handles_data[
            os.path.join(meta["server_folder"], meta["output_folder"], meta["tile_name"])
        ]
        img = self._slice_hypercube(data_tensor, meta)
        # dali expects the data to have a batch dimension
        batch_img = np.expand_dims(img, axis=0)
        return batch_img

    def __call__(self, sample_info):
        if sample_info.iteration >= self.full_iterations:
            # indicate end of the epoch
            raise StopIteration
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            self.perm = np.random.default_rng(seed=42 + sample_info.epoch_idx)
            self.perm = self.perm.permutation(len(self.hypercubes_dataframe))

        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]
        sample = self._index[sample_idx]
        return self._load_sample(sample)


@pipeline_def
def pretrain_dataset_pipeline(dataset):
    target_dtype = DALI_DTYPES[dataset.target_dtype].value \
        if isinstance(dataset.target_dtype, str) else dataset.target_dtype

    vols = fn.external_source(
        source=dataset,
        num_outputs=1,
        batch=False,
        parallel=True,
        device="cpu",
        dtype=target_dtype,
        ndim=dataset.ndim,
    )
    vol = vols[0].gpu()
    
    # TODO: (1) make this more robust to 
    #       different input shapes
    #       (2) should we do this ourselves,
    #       or let DALI handle it?
    vol_f32 = fn.cast(vol, dtype=types.FLOAT)
    vol_norm = fn.normalize(
        vol_f32,
        axes=[1, 2, 3, 4],
        batch=True
    )
    vol_out = fn.cast(vol_norm, dtype=target_dtype)
    return vol_out