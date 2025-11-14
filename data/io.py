import sys
import time
import logging
from pathlib import Path
from typing import Tuple, Literal, Optional, Iterable, Dict, Any

import re
import ujson
import inspect
import functools

import torch

import numpy as np
import tensorstore as ts

from tifffile import imwrite
from tifffile import TiffFile
from skimage.io import imread, imsave

import pandas as pd
import polars as pl

from cell_observatory_platform.data.data_types import TENSORSTORE_DTYPES, NUMPY_DTYPES, TORCH_DTYPES

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_npy(image_path: str, dtype: NUMPY_DTYPES | str = None) -> np.ndarray:
    if isinstance(image_path, torch.Tensor):
        path = Path(str(image_path.numpy(), "utf-8"))
    else:
        path = Path(str(image_path))

    if path.suffix == '.npy':
        with np.load(path) as arr:
            img = arr
    elif path.suffix == '.npz':
        with np.load(path) as data:
            img = data['arr_0']
    else:
        raise NotImplementedError

    if np.isnan(np.sum(img)):
        logger.error("NaN!")

    if dtype is not None:
        dtype = NUMPY_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        img = img.astype(dtype)
    return img


def read_tiff(image_path: str, dtype: NUMPY_DTYPES | str = None) -> np.ndarray:
    """ Read a TIFF file and return the data as a NumPy array """
    img = imread(image_path)

    if np.isnan(np.sum(img)):
        logger.error("NaN!")

    if dtype is not None:
        dtype = NUMPY_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        img = img.astype(dtype)
    return img


def read_zarr(
    image_path: str,
    zarr_driver: str = "zarr3",
    dtype: Optional[TENSORSTORE_DTYPES | str] = None,
    context: ts.Context | None = None,
    cast: bool = False
) -> np.ndarray:
    """ Read a Zarr file and return the data as a NumPy array """
    spec = {
        "driver": zarr_driver,
        "kvstore": {"driver": "file", "path": image_path},
        "dtype": ts.uint16,
    }
    dtype = TENSORSTORE_DTYPES[dtype].value if isinstance(dtype, str) else dtype
    ds = ts.open(spec, context=context, read=True).result()
    if cast:
        return ts.cast(ds, dtype)
    else:
        return ds


def read_file(
    image_path: str | Path,
    dtype: NUMPY_DTYPES | TENSORSTORE_DTYPES | TORCH_DTYPES | str = None,
) -> np.ndarray:
    """ Infer the file format of the image based on its extension """
    image_path = str(image_path)

    if image_path.endswith(".zarr"):
        return read_zarr(image_path, dtype=dtype)

    elif image_path.endswith(".tiff") or image_path.endswith(".tif"):
        return read_tiff(image_path, dtype=dtype)

    elif image_path.endswith(".npy"):
        return read_npy(image_path, dtype=dtype)

    else:
        raise ValueError(f"Unsupported file format for {image_path}")


def save_file(image_path: str, data: np.ndarray, **kwargs) -> None:
    """ Save a NumPy array to a file based on its extension """
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    image_path = str(image_path)

    if image_path.endswith(".zarr"):
        save_zarr(image_path, data, **kwargs)
    elif image_path.endswith(".tiff") or image_path.endswith(".tif"):
        save_tiff(image_path, data, **kwargs)
    else:
        raise ValueError(f"Unsupported file format for {image_path}")


# NOTE: taken from ml-data-cell_observatory_platform
def create_zarr_spec(data_shape,
                    zarr_version, 
                    path, 
                    input_format, 
                    shard_cube_shape, 
                    chunk_shape,
                    dtype: str = 'uint16'
):
    # NOTE: currently zarr saving format assumes time dimension is present 
    #       always, we should consider changing this in the future
    if zarr_version == 'zarr3':
        if input_format == 'TZYXC':
            num_timepoints_per_image, num_channels = data_shape[0], data_shape[-1]
            shard_shape = [num_timepoints_per_image, shard_cube_shape[0], shard_cube_shape[1], shard_cube_shape[2], num_channels]
            chunk_shape = [1, chunk_shape[0], chunk_shape[1], chunk_shape[2], num_channels]
        
        elif input_format == "TCZYX":
            num_timepoints_per_image, num_channels = data_shape[0], data_shape[1]
            shard_shape = [num_timepoints_per_image, num_channels, shard_cube_shape[0], shard_cube_shape[1], shard_cube_shape[2]]
            chunk_shape = [1, num_channels, chunk_shape[0], chunk_shape[1], chunk_shape[2]]

        elif input_format == 'ZYXC':
            num_timepoints_per_image, num_channels = data_shape[0], data_shape[-1]
            shard_shape = [num_timepoints_per_image, shard_cube_shape[0], shard_cube_shape[1], shard_cube_shape[2], num_channels]
            chunk_shape = [1, chunk_shape[0], chunk_shape[1], chunk_shape[2], num_channels]

        elif input_format == "CZYX":
            num_timepoints_per_image, num_channels = data_shape[0], data_shape[1]
            shard_shape = [num_timepoints_per_image, num_channels, shard_cube_shape[0], shard_cube_shape[1], shard_cube_shape[2]]
            chunk_shape = [1, num_channels, chunk_shape[0], chunk_shape[1], chunk_shape[2]]
    
        else:
            raise ValueError(f"Unsupported data shape length: {len(data_shape)}")

        zarr_spec = {
            'driver': zarr_version,
            'kvstore': {
                'driver': 'file',
                'path': path
            },
            'metadata': {
                'data_type': str(dtype),
                'shape': data_shape,
                'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': shard_shape}},
                'codecs': [{
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": chunk_shape,
                        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                                   {"name": "blosc", "configuration": {
                                       "cname": "zstd", "clevel": 1, "blocksize": 0, "shuffle": "shuffle"}}],
                        "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}],
                        "index_location": "end"
                    }
                }],
                'fill_value': 0,
            },
            'create': True,
            'delete_existing': True
        }
    else:
        zarr_spec = {
            'driver': zarr_version,
            'kvstore': {
                'driver': 'file',
                'path': path
            },
            'metadata': {
                'dtype': '<u2',
                'shape': data_shape,
                'chunks': chunk_shape,
                'compressor': {'blocksize': 0, 'clevel': 1, 'cname': 'zstd', 'id': 'blosc', 'shuffle': 1},
                'fill_value': 0,
                'order': 'C'
            },
            'create': True,
            'delete_existing': True
        }
    return zarr_spec


def save_zarr(
    image_path: str,
    data: np.ndarray,
    shard_cube_shape: Tuple[int, int, int],
    chunk_shape: Tuple[int, int, int],
    input_format: str,
    zarr_driver: str = "zarr3",
    dtype: str = 'uint16'
) -> None:
    zarr_spec = create_zarr_spec(
        data_shape=data.shape,
        zarr_version=zarr_driver,
        path=image_path,
        input_format=input_format,
        shard_cube_shape=shard_cube_shape,
        chunk_shape=chunk_shape,
        dtype=dtype
    )

    ds = ts.open(zarr_spec).result()
    ds[:] = data


def save_tiff(image_path: str, data: np.ndarray, axes: str, with_fiji: bool = False) -> None:
    if with_fiji:
        data = np.ascontiguousarray(data)
        image_path = str(image_path).replace('.tif', '.ome.tif')
        imwrite(
            image_path,
            data,
            ome=True,
            metadata={"axes": axes},
            bigtiff=True,
            photometric="minisblack"
        )
    else:
        imsave(image_path, data)


def get_shape_from_file_tiff(image_path: str) -> tuple:
    path = Path(image_path)
    with TiffFile(str(path)) as tif:
        # series[0] is the first image series (e.g. the main image)
        # .shape might be (Z,Y,X), (C,Z,Y,X), (T,Z,Y,X) or (T,C,Z,Y,X), etc.
        return tif.series[0].shape


def record_init(fn):
    """
    Decorator for __init__ methods.  Captures every arg/kwarg you passed
    (with defaults) into _init_args.
    """
    sig = inspect.signature(fn)
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        init_args = {
            name: value
            for name, value in bound.arguments.items()
            if name != "self"
        }
        self._init_args = init_args
        return fn(self, *args, **kwargs)
    return wrapper

def filter_hypercubes_dataframe_storage_server(
    df: pl.DataFrame, server_folder_path: str | None = None
) -> pl.DataFrame:
    _INT_TYPES = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
    def _coerce_bool_in(df: pl.DataFrame, col: str) -> pl.DataFrame:
        dt = df.schema.get(col)

        if dt == pl.Boolean:
            expr = pl.col(col).fill_null(False)

        elif dt in _INT_TYPES:
            expr = (pl.col(col) != 0).fill_null(False)

        else:
            expr = (
                pl.col(col).cast(pl.Utf8)
                .str.strip_chars()
                .str.to_lowercase()
                .is_in(["t", "true", "1"])
                .fill_null(False)
            )

        return df.with_columns(expr.alias(col))

    if server_folder_path is None or str(server_folder_path).startswith("/clusterfs"):
        flag = "exists"
        df = _coerce_bool_in(df, flag).filter(pl.col(flag))
        return df

    if str(server_folder_path).startswith("/groups"):
        flag = "exists_prfs"
    elif str(server_folder_path).startswith("/aws") or str(server_folder_path).startswith("/workspace/CellObservatoryData"):
        flag = "exists_aws"
    elif str(server_folder_path).startswith("/lustre"):
        flag = "exists_oak"
    else:
        raise ValueError(f"Unknown server_folder_path: {server_folder_path}")

    df = (
        _coerce_bool_in(df, flag)
        .filter(pl.col(flag))
        .with_columns(pl.lit(str(server_folder_path)).alias("server_folder"))
    )
    return df

def apply_hypercubes_dataframe_selections(
    df: pl.DataFrame,
    max_rois: int | None = None,
    max_tiles: int | None = None,
    max_hypercubes: int | None = None,
    hpf_list: list[int] | None = None,
    roi_list: list[int] | None = None,
    tile_list: list[str] | None = None,
    timepoint_list: list[int] | None = None,
) -> pl.DataFrame:
    logger.info( 
        f"\nApplied selections:\n" 
        f"hpf_list={hpf_list}\n" 
        f"roi_list={roi_list}\n" 
        f"tile_list={tile_list}\n" 
        f"timepoint_list={timepoint_list}\n" 
        f"max_rois={max_rois}\n" 
        f"max_tiles={max_tiles}\n" 
        f"max_hypercubes={max_hypercubes}" 
    )

    def _to_list_or_none(x):
        if x is None or len(list(x)) == 0:
            return None
        else:
            return list(x)

    rois  = _to_list_or_none(roi_list)
    tiles = _to_list_or_none(tile_list)
    hpfs = _to_list_or_none(hpf_list)
    tps = _to_list_or_none(timepoint_list)

    conds = []
    if rois is not None and "prepared_id" in df.columns:
        conds.append(pl.col("prepared_id").is_in(rois))
    if tiles is not None and "tile_name" in df.columns:
        conds.append(pl.col("tile_name").is_in(tiles))
    if tps is not None and "time_start" in df.columns:
        conds.append(pl.col("time_start").is_in(tps))

    if conds:
        cond = conds[0]
        for c in conds[1:]:
            cond = cond & c
        df = df.filter(cond)

    if hpfs is not None and "hpf" in df.columns:
        df = df.filter(pl.col("hpf").is_in(hpfs))

    if max_rois is not None and "prepared_id" in df.columns:
        keep_rois = (
            df.select(pl.col("prepared_id").unique())
              .select(pl.col("prepared_id").head(max_rois))
              .to_series().to_list()
        )
        df = df.filter(pl.col("prepared_id").is_in(keep_rois))

    if max_tiles is not None and "tile_name" in df.columns:
        keep_tiles = (
            df.select(pl.col("tile_name").unique())
              .select(pl.col("tile_name").head(max_tiles))
              .to_series().to_list()
        )
        df = df.filter(pl.col("tile_name").is_in(keep_tiles))

    if max_hypercubes is not None:
        df = df.head(max_hypercubes)

    return df

def compute_df_stats(df: pl.DataFrame) -> pl.DataFrame:
    def _parse_string_col(expr: pl.Expr) -> pl.Expr:
        return (
            expr.cast(pl.Utf8)
            .str.strip_chars()
            .str.replace_all(r'^[\[\{\(]\s*', '', literal=False)
            .str.replace_all(r'\s*[\]\}\)]$', '', literal=False)
            .str.replace_all("\n", " ", literal=True)
            .str.replace_all('"', "", literal=True)
            .str.replace_all("'", "", literal=True)
            .str.replace_all(r"[,\s]+", " ", literal=False)
            .str.strip_chars()
            .str.extract_all(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')
            .list.eval(pl.element().cast(pl.Float64))
        )

    def _parse_occupancy_expr(colname: str, dtypes: Dict[str, pl.DataType]) -> pl.Expr:
        dt = dtypes[colname]
        if isinstance(dt, pl.List) and dt.inner in (pl.Float32, pl.Float64):
            return pl.col(colname).cast(pl.List(pl.Float64))
        if dt == pl.Utf8:
            return _parse_string_col(pl.col(colname))
        raise TypeError(f"Unsupported dtype for {colname}: {dt!r}")

    ch0 = _parse_occupancy_expr("occupancy_ratios_ch_0", df.schema)
    ch1 = _parse_occupancy_expr("occupancy_ratios_ch_1", df.schema)

    return df.with_columns(
        ch0.list.min().alias("min_occupancy_ratios_ch_0"),
        ch0.list.mean().alias("mean_occupancy_ratios_ch_0"),
        ch0.list.median().alias("med_occupancy_ratios_ch_0"),
        ch1.list.min().alias("min_occupancy_ratios_ch_1"),
        ch1.list.mean().alias("mean_occupancy_ratios_ch_1"),
        ch1.list.median().alias("med_occupancy_ratios_ch_1"),
    )

def apply_occupancy_threshold(
    df: pl.DataFrame,
    occupancy_threshold: float | None = 0.0,
    occupancy_threshold_filter_type: Literal['min_all', 'min_ch0', 'min_ch1'] = 'min_ch0'
) -> pl.DataFrame:
    t = 0.0 if occupancy_threshold is None else float(occupancy_threshold)
    df = compute_df_stats(df)

    if occupancy_threshold_filter_type == 'min_all':
        mask = (
            (pl.col("min_occupancy_ratios_ch_0") >= t) &
            (pl.col("min_occupancy_ratios_ch_1") >= t)
        )
    elif occupancy_threshold_filter_type == 'min_ch0':
        mask = (pl.col("min_occupancy_ratios_ch_0") >= t)
    elif occupancy_threshold_filter_type == 'min_ch1':
        mask = (pl.col("min_occupancy_ratios_ch_1") >= t)
    else:
        raise ValueError(occupancy_threshold_filter_type)

    return df.filter(mask)

def apply_hypercubes_dataframe_filters(
    df: pl.DataFrame,
    occupancy_threshold: float | None = 0.0,
    occupancy_threshold_filter_type: str = 'min_ch0'
) -> pl.DataFrame:
    df = apply_occupancy_threshold(
        df,
        occupancy_threshold=occupancy_threshold,
        occupancy_threshold_filter_type=occupancy_threshold_filter_type,
    )

    stats = (
        df.select(
            pl.col("min_occupancy_ratios_ch_0").min().alias("ch0_min"),
            pl.col("min_occupancy_ratios_ch_0").quantile(0.5).alias("ch0_med"),
            pl.col("min_occupancy_ratios_ch_0").max().alias("ch0_max"),
            pl.col("min_occupancy_ratios_ch_1").min().alias("ch1_min"),
            pl.col("min_occupancy_ratios_ch_1").quantile(0.5).alias("ch1_med"),
            pl.col("min_occupancy_ratios_ch_1").max().alias("ch1_max"),
        )
        .to_dicts()
    )
    logger.info(f"Min-occupancy summary: {stats}")
    return df

def load_hypercubes_dataframe(
    hypercubes_dataframe_path: str | Path,
    max_rois: int | None = None,
    max_tiles: int | None = None,
    max_hypercubes: int | None = None,
    hpf_list: list[int] | None = None,
    roi_list: list[int] | None = None,
    tile_list: list[str] | None = None,
    timepoint_list: list[int] | None = None,
    server_folder_path: str | None = None,
    occupancy_threshold: float | None = None,
    occupancy_threshold_filter_type: str = "min_ch0",
) -> tuple[pl.DataFrame, dict]:
    p = Path(hypercubes_dataframe_path)
    if not p.exists():
        raise FileNotFoundError(p)

    t0 = time.perf_counter()
    df = pl.read_csv(p)
    t1 = time.perf_counter()
    logger.info(f"Loaded hypercubes dataframe in {t1 - t0:.2f} s; shape={df.shape}")

    df = filter_hypercubes_dataframe_storage_server(df, server_folder_path)

    t0 = time.perf_counter()
    df = apply_hypercubes_dataframe_selections(
        df,
        max_rois=max_rois,
        max_tiles=max_tiles,
        max_hypercubes=max_hypercubes,
        hpf_list=hpf_list,
        roi_list=roi_list,
        tile_list=tile_list,
        timepoint_list=timepoint_list,
    )
    t1 = time.perf_counter()
    logger.info(f"Applied selections in {t1 - t0:.2f} s; shape={df.shape}")

    # NOTE: as we scale to billions of hypercubes, these filters may become a bottleneck again
    #        so we may need to optimize them further by pre-computation or distributed processing
    t0 = time.perf_counter()
    df = apply_hypercubes_dataframe_filters(
        df,
        occupancy_threshold=occupancy_threshold,
        occupancy_threshold_filter_type=occupancy_threshold_filter_type,
    )
    t1 = time.perf_counter()
    logger.info(f"Applied filters in {t1 - t0:.2f} s; shape={df.shape}")

    try:
        with open(p.with_suffix(".json"), "r") as f:
            configs = ujson.load(f)
    except FileNotFoundError:
        configs = {}

    return df.to_pandas(use_pyarrow_extension_array=True), configs