import functools
import inspect
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import tensorstore as ts
import torch
import ujson
from skimage.io import imread, imsave
from tifffile import TiffFile, imwrite

from cell_observatory_platform.data.data_types import NUMPY_DTYPES, TENSORSTORE_DTYPES, TORCH_DTYPES

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def read_npy(image_path: str, dtype: NUMPY_DTYPES | str = None) -> np.ndarray:
    if isinstance(image_path, torch.Tensor):
        path = Path(str(image_path.numpy(), "utf-8"))
    else:
        path = Path(str(image_path))

    if path.suffix == ".npy":
        with np.load(path) as arr:
            img = arr
    elif path.suffix == ".npz":
        with np.load(path) as data:
            img = data["arr_0"]
    else:
        raise NotImplementedError

    if np.isnan(np.sum(img)):
        logger.error("NaN!")

    if dtype is not None:
        dtype = NUMPY_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        img = img.astype(dtype)
    return img


def read_tiff(image_path: str, dtype: NUMPY_DTYPES | str = None) -> np.ndarray:
    """Read a TIFF file and return the data as a NumPy array"""
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
    cast: bool = False,
) -> np.ndarray:
    """Read a Zarr file and return the data as a NumPy array"""
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
    """Infer the file format of the image based on its extension"""
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
    """Save a NumPy array to a file based on its extension"""
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    image_path = str(image_path)

    if image_path.endswith(".zarr"):
        save_zarr(image_path, data, **kwargs)
    elif image_path.endswith(".tiff") or image_path.endswith(".tif"):
        save_tiff(image_path, data, **kwargs)
    else:
        raise ValueError(f"Unsupported file format for {image_path}")


# NOTE: taken from ml-data-cell_observatory_platform
def create_zarr_spec(
    data_shape, zarr_version, path, input_format, shard_cube_shape, chunk_shape, dtype: str = "uint16"
):
    # NOTE: currently zarr saving format assumes time dimension is present
    #       always, we should consider changing this in the future
    if zarr_version == "zarr3":
        if input_format == "TZYXC":
            num_timepoints_per_image, num_channels = data_shape[0], data_shape[-1]
            shard_shape = [
                num_timepoints_per_image,
                shard_cube_shape[0],
                shard_cube_shape[1],
                shard_cube_shape[2],
                num_channels,
            ]
            chunk_shape = [1, chunk_shape[0], chunk_shape[1], chunk_shape[2], num_channels]

        elif input_format == "TCZYX":
            num_timepoints_per_image, num_channels = data_shape[0], data_shape[1]
            shard_shape = [
                num_timepoints_per_image,
                num_channels,
                shard_cube_shape[0],
                shard_cube_shape[1],
                shard_cube_shape[2],
            ]
            chunk_shape = [1, num_channels, chunk_shape[0], chunk_shape[1], chunk_shape[2]]

        elif input_format == "ZYXC":
            num_timepoints_per_image, num_channels = data_shape[0], data_shape[-1]
            shard_shape = [
                num_timepoints_per_image,
                shard_cube_shape[0],
                shard_cube_shape[1],
                shard_cube_shape[2],
                num_channels,
            ]
            chunk_shape = [1, chunk_shape[0], chunk_shape[1], chunk_shape[2], num_channels]

        elif input_format == "CZYX":
            num_timepoints_per_image, num_channels = data_shape[0], data_shape[1]
            shard_shape = [
                num_timepoints_per_image,
                num_channels,
                shard_cube_shape[0],
                shard_cube_shape[1],
                shard_cube_shape[2],
            ]
            chunk_shape = [1, num_channels, chunk_shape[0], chunk_shape[1], chunk_shape[2]]

        else:
            raise ValueError(f"Unsupported data shape length: {len(data_shape)}")

        zarr_spec = {
            "driver": zarr_version,
            "kvstore": {"driver": "file", "path": path},
            "metadata": {
                "data_type": str(dtype),
                "shape": data_shape,
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shard_shape}},
                "codecs": [
                    {
                        "name": "sharding_indexed",
                        "configuration": {
                            "chunk_shape": chunk_shape,
                            "codecs": [
                                {"name": "bytes", "configuration": {"endian": "little"}},
                                {
                                    "name": "blosc",
                                    "configuration": {
                                        "cname": "zstd",
                                        "clevel": 1,
                                        "blocksize": 0,
                                        "shuffle": "shuffle",
                                    },
                                },
                            ],
                            "index_codecs": [
                                {"name": "bytes", "configuration": {"endian": "little"}},
                                {"name": "crc32c"},
                            ],
                            "index_location": "end",
                        },
                    }
                ],
                "fill_value": 0,
            },
            "create": True,
            "delete_existing": True,
        }
    else:
        zarr_spec = {
            "driver": zarr_version,
            "kvstore": {"driver": "file", "path": path},
            "metadata": {
                "dtype": "<u2",
                "shape": data_shape,
                "chunks": chunk_shape,
                "compressor": {"blocksize": 0, "clevel": 1, "cname": "zstd", "id": "blosc", "shuffle": 1},
                "fill_value": 0,
                "order": "C",
            },
            "create": True,
            "delete_existing": True,
        }
    return zarr_spec


def save_zarr(
    image_path: str,
    data: np.ndarray,
    shard_cube_shape: Tuple[int, int, int],
    chunk_shape: Tuple[int, int, int],
    input_format: str,
    zarr_driver: str = "zarr3",
    dtype: str = "uint16",
) -> None:
    zarr_spec = create_zarr_spec(
        data_shape=data.shape,
        zarr_version=zarr_driver,
        path=image_path,
        input_format=input_format,
        shard_cube_shape=shard_cube_shape,
        chunk_shape=chunk_shape,
        dtype=dtype,
    )

    ds = ts.open(zarr_spec).result()
    ds[:] = data


def save_tiff(image_path: str, data: np.ndarray, axes: str, with_fiji: bool = False) -> None:
    if with_fiji:
        data = np.ascontiguousarray(data)
        image_path = str(image_path).replace(".tif", ".ome.tif")
        imwrite(image_path, data, ome=True, metadata={"axes": axes}, bigtiff=True, photometric="minisblack")
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
        init_args = {name: value for name, value in bound.arguments.items() if name != "self"}
        self._init_args = init_args
        return fn(self, *args, **kwargs)

    return wrapper


# def _coerce_bool_in(df: pl.DataFrame, col: str) -> pl.DataFrame:
#     _INT_TYPES = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
#     dt = df.schema.get(col)

#     if dt == pl.Boolean:
#         expr = pl.col(col).fill_null(False)

#     elif dt in _INT_TYPES:
#         expr = (pl.col(col) != 0).fill_null(False)

#     else:
#         expr = pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_lowercase().is_in(["t", "true", "1"]).fill_null(False)

#     return df.with_columns(expr.alias(col))


# def filter_hypercubes_dataframe_storage_server(df: pl.DataFrame, server_folder_path: str | None = None) -> pl.DataFrame:
#     if server_folder_path is None or str(server_folder_path).startswith("/clusterfs"):
#         flag = "exists"
#         df = _coerce_bool_in(df, flag).filter(pl.col(flag))
#         return df

#     if str(server_folder_path).startswith("/groups"):
#         flag = "exists_prfs"
#     elif str(server_folder_path).startswith("/aws") or str(server_folder_path).startswith(
#         "/workspace/CellObservatoryData"
#     ):
#         flag = "exists_aws"
#     elif str(server_folder_path).startswith("/lustre"):
#         flag = "exists_oak"
#     else:
#         raise ValueError(f"Unknown server_folder_path: {server_folder_path}")

#     df = (
#         _coerce_bool_in(df, flag)
#         .filter(pl.col(flag))
#         .with_columns(pl.lit(str(server_folder_path)).alias("server_folder"))
#     )

#     logger.info(f"Loaded hypercubes on server: {server_folder_path}; shape={df.shape}")
#     return df


# def apply_hypercubes_dataframe_selections(
#     df: pl.DataFrame,
#     max_rois: int | None = None,
#     max_tiles: int | None = None,
#     max_hypercubes: int | None = None,
#     hpf_list: list[int] | None = None,
#     roi_list: list[int] | None = None,
#     tile_list: list[str] | None = None,
#     timepoint_list: list[int] | None = None,
# ) -> pl.DataFrame:
#     logger.info(
#         f"\nApplied selections:\n"
#         f"hpf_list={hpf_list}\n"
#         f"roi_list={roi_list}\n"
#         f"tile_list={tile_list}\n"
#         f"timepoint_list={timepoint_list}\n"
#         f"max_rois={max_rois}\n"
#         f"max_tiles={max_tiles}\n"
#         f"max_hypercubes={max_hypercubes}"
#     )

#     def _to_list_or_none(x):
#         if x is None or len(list(x)) == 0:
#             return None
#         else:
#             return list(x)

#     rois = _to_list_or_none(roi_list)
#     tiles = _to_list_or_none(tile_list)
#     hpfs = _to_list_or_none(hpf_list)
#     tps = _to_list_or_none(timepoint_list)

#     conds = []
#     if rois is not None and "prepared_id" in df.columns:
#         conds.append(pl.col("prepared_id").is_in(rois))
#     if tiles is not None and "tile_name" in df.columns:
#         conds.append(pl.col("tile_name").is_in(tiles))
#     if tps is not None and "time_start" in df.columns:
#         conds.append(pl.col("time_start").is_in(tps))

#     if conds:
#         cond = conds[0]
#         for c in conds[1:]:
#             cond = cond & c
#         df = df.filter(cond)

#     if hpfs is not None and "hpf" in df.columns:
#         df = df.filter(pl.col("hpf").is_in(hpfs))

#     if max_rois is not None and "prepared_id" in df.columns:
#         keep_rois = (
#             df.select(pl.col("prepared_id").unique())
#             .sort("prepared_id")
#             .select(pl.col("prepared_id").head(max_rois))
#             .to_series()
#             .to_list()
#         )
#         df = df.filter(pl.col("prepared_id").is_in(keep_rois))

#     if max_tiles is not None and "tile_name" in df.columns:
#         keep_tiles = (
#             df.select(pl.col("tile_name").unique())
#             .sort("tile_name")
#             .select(pl.col("tile_name").head(max_tiles))
#             .to_series()
#             .to_list()
#         )
#         df = df.filter(pl.col("tile_name").is_in(keep_tiles))

#     if max_hypercubes is not None:
#         df = df.sort(["prepared_id", "tile_name", "z_start", "y_start", "x_start", "time_start"]).head(max_hypercubes)

#     return df


# def add_has_annotations_column(df: pl.DataFrame) -> pl.DataFrame:
#     """
#     look for nested entries for each channel in the
#     pc_metadata_json col:  {'0': {'histogram': {...}}, '1': {'mask_bbox_dict': {...}}}
#     each key is a channel id mapping to a dict of metadata
#     """
#     if "has_annotations" in df.columns:
#         return df
#     if "pc_metadata_json" not in df.columns and "metadata_tile_json" not in df.columns:
#         return df.with_columns(pl.lit(False).alias("has_annotations"))
#     if "pc_metadata_json" in df.columns:
#         has_key = pl.col("pc_metadata_json").str.contains(r'"mask_bbox_dict"', literal=False)
#         empty_obj = pl.col("pc_metadata_json").str.contains(r'"mask_bbox_dict"\s*:\s*\{\s*\}', literal=False)
#         expr = pl.col("pc_metadata_json").is_not_null() & has_key & (~empty_obj)
#         return df.with_columns(expr.alias("has_annotations"))
#     if "metadata_tile_json" in df.columns:
#         has_key = pl.col("metadata_tile_json").str.contains(r'"mask_bbox_dict"', literal=False)
#         empty_obj = pl.col("metadata_tile_json").str.contains(r'"mask_bbox_dict"\s*:\s*\{\s*\}', literal=False)
#         expr = pl.col("metadata_tile_json").is_not_null() & has_key & (~empty_obj)
#         return df.with_columns(expr.alias("has_annotations"))


# # FIXME: current nomenclature for metadata may be improved
# def create_channel_metadata_columns(df: pl.DataFrame, expected_channel_ids=["0", "1"]) -> pl.DataFrame:
#     new_columns = []
#     for ch in expected_channel_ids:
#         if f"histogram_ch_{ch}" not in df.columns:
#             new_columns.append(pl.lit(None).alias(f"histogram_ch_{ch}"))
#         if f"mask_bbox_dict_ch_{ch}" not in df.columns:
#             new_columns.append(pl.lit(None).alias(f"mask_bbox_dict_ch_{ch}"))
#     if new_columns:
#         df = df.with_columns(new_columns)
#     if "mask_bbox_dict" not in df.columns:
#         for ch in expected_channel_ids:
#             if f"mask_bbox_dict_ch_{ch}" in df.columns:
#                 df = df.with_columns(pl.col(f"mask_bbox_dict_ch_{ch}").alias("mask_bbox_dict"))
#     # if "histograms" not in df.columns:
#     #     for ch in expected_channel_ids:
#     #         if f"histogram_ch_{ch}" in df.columns:
#     #             df = df.with_columns(pl.col(f"histogram_ch_{ch}").alias("histograms"))
#     return df


# def load_hypercubes_dataframe(
#     hypercubes_dataframe_path: str | Path,
#     max_rois: int | None = None,
#     max_tiles: int | None = None,
#     max_hypercubes: int | None = None,
#     hpf_list: list[int] | None = None,
#     roi_list: list[int] | None = None,
#     tile_list: list[str] | None = None,
#     timepoint_list: list[int] | None = None,
#     server_folder_path: str | None = None,
#     synthetic_only: bool = False,
#     has_annotations: bool = False,
# ) -> tuple[pl.DataFrame, dict]:
#     p = Path(hypercubes_dataframe_path)
#     if not p.exists():
#         raise FileNotFoundError(p)

#     t0 = time.perf_counter()
#     df = pl.read_csv(p, null_values=["NULL", "null", "NaN", ""])
#     t1 = time.perf_counter()
#     logger.info(f"Loaded hypercubes dataframe in {t1 - t0:.2f} s; shape={df.shape}")

#     # NOTE: database is currently not updated to reflect storage server status
#     #       remove this once the database is updated
#     # df = filter_hypercubes_dataframe_storage_server(df, server_folder_path)
#     df = add_has_annotations_column(df)

#     if synthetic_only:
#         df = _coerce_bool_in(df, "is_synthetic").filter(pl.col("is_synthetic"))

#     if has_annotations:
#         df = _coerce_bool_in(df, "has_annotations").filter(pl.col("has_annotations"))

#     df = create_channel_metadata_columns(df)

#     t0 = time.perf_counter()
#     df = apply_hypercubes_dataframe_selections(
#         df,
#         max_rois=max_rois,
#         max_tiles=max_tiles,
#         max_hypercubes=max_hypercubes,
#         hpf_list=hpf_list,
#         roi_list=roi_list,
#         tile_list=tile_list,
#         timepoint_list=timepoint_list,
#     )
#     t1 = time.perf_counter()
#     logger.info(f"Applied selections in {t1 - t0:.2f} s; shape={df.shape}")

#     try:
#         with open(p.with_suffix(".json"), "r") as f:
#             configs = ujson.load(f)
#     except FileNotFoundError:
#         configs = {}

#     return df.to_pandas(use_pyarrow_extension_array=True), configs


# def load_tiles_dataframe(
#     hypercubes_dataframe_path: str | Path,
#     max_rois: int | None = None,
#     max_tiles: int | None = None,
#     hpf_list: list[int] | None = None,
#     roi_list: list[int] | None = None,
#     tile_list: list[str] | None = None,
#     timepoint_list: list[int] | None = None,
#     server_folder_path: str | None = None,
#     synthetic_only: bool = False,
#     has_annotations: bool = False,
# ) -> tuple[pd.DataFrame, dict]:
#     p = Path(hypercubes_dataframe_path)
#     if not p.exists():
#         raise FileNotFoundError(p)

#     t0 = time.perf_counter()
#     df = pl.read_csv(p)
#     t1 = time.perf_counter()
#     logger.info(f"Loaded tiles dataframe in {t1 - t0:.2f} s; shape={df.shape}")

#     # NOTE: database is currently not updated to reflect storage server status
#     #       remove this once the database is updated
#     # df = filter_hypercubes_dataframe_storage_server(df, server_folder_path)
#     df = add_has_annotations_column(df)

#     if synthetic_only and "is_synthetic" in df.columns:
#         df = _coerce_bool_in(df, "is_synthetic").filter(pl.col("is_synthetic"))

#     if has_annotations and "has_annotations" in df.columns:
#         df = _coerce_bool_in(df, "has_annotations").filter(pl.col("has_annotations"))

#     df = create_channel_metadata_columns(df)

#     t0 = time.perf_counter()
#     df = apply_hypercubes_dataframe_selections(
#         df,
#         max_rois=max_rois,
#         max_tiles=max_tiles,
#         max_hypercubes=None,
#         hpf_list=hpf_list,
#         roi_list=roi_list,
#         tile_list=tile_list,
#         timepoint_list=timepoint_list,
#     )
#     t1 = time.perf_counter()
#     logger.info(f"Applied tile selections in {t1 - t0:.2f} s; shape={df.shape}")

#     try:
#         with open(p.with_suffix(".json"), "r") as f:
#             configs = ujson.load(f)
#     except FileNotFoundError:
#         configs = {}

#     return df.to_pandas(use_pyarrow_extension_array=True), configs
