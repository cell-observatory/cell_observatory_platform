import functools
import inspect
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

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


def save_scores(
    image_path: str,
    model_name: str,
    scores: np.ndarray,
    task: Literal["instance_segmentation", "semantic_segmentation", "detection"],
    input_format: Literal["TN", "N"],
    save_mode: Literal["overwrite", "append"],
    shard_cube_shape: Optional[Tuple[int, int, int]] = None,
    chunk_shape: Optional[Tuple[int, int, int]] = None,
    timepoint_idxs: Optional[List[int]] = None,
) -> None:
    raise NotImplementedError("save_instance_scores is not implemented")

def save_labels(
    image_path: str,
    model_name: str,
    labels: np.ndarray,
    task: Literal["instance_segmentation", "semantic_segmentation", "detection"],
    input_format: Literal["TN", "N"],
    save_mode: Literal["overwrite", "append"],
    shard_cube_shape: Optional[Tuple[int, int, int]] = None,
    chunk_shape: Optional[Tuple[int, int, int]] = None,
    timepoint_idxs: Optional[List[int]] = None,
) -> None:
    raise NotImplementedError("save_instance_labels is not implemented")

def save_masks(
    image_path: str,
    model_name: str,
    masks: np.ndarray,
    input_format: Literal["TZYXC", "ZYXC"],
    task: Literal["instance_segmentation", "semantic_segmentation"],
    save_mode: Literal["overwrite", "append"],
    zarr_driver: str = "zarr3",
    dtype: str = "uint16",
    shard_cube_shape: Optional[Tuple[int, int, int]] = None,
    chunk_shape: Optional[Tuple[int, int, int]] = None,
    timepoint_idxs: Optional[List[int]] = None,
    data_channel_idxs: Optional[List[int]] = None,
    mask_channel_idxs: Optional[List[int]] = None,
) -> None:
    """Save masks to a separate zarr labels group at <image_path>/labels/<label_name>."""
    if save_mode == "create" and (shard_cube_shape is None or chunk_shape is None):
        raise ValueError("shard_cube_shape and chunk_shape are required when creating new labels")
    if task == "instance_segmentation":
        label_name = "instance_masks"
    elif task == "semantic_segmentation":
        label_name = "semantic_masks"
    else:
        raise ValueError(f"Unsupported task for save_masks: {task}")
    # If we are appending masks, that means they don't already exist, so we need to create them.
    # If we are overwriting masks, that means we are either running a new model or re-running the same model.
    # If we are running a new model, we need to create the labels array.
    # If we are re-running the same model, we need to overwrite the old labels array.
    if save_mode == "append":
        label_save_mode = "create"
    elif save_mode == "overwrite":
        exists = label_exists(image_path, model_name, label_name, zarr_driver)
        if exists:
            label_save_mode = "overwrite"
        else:
            label_save_mode = "create"
    else:
        raise ValueError(f"Invalid save_mode: {save_mode}")

    try:
        update_zarr_data(
            image_path=image_path,
            data=masks,
            input_format=input_format,
            zarr_driver=zarr_driver,
            dtype=dtype,
            timepoint_idxs=timepoint_idxs,
            data_channel_idxs=data_channel_idxs,
            mask_channel_idxs=mask_channel_idxs,
            mode=save_mode,
        )
    except Exception as e:
        logger.error(f"Failed to update zarr masks in root of zarr store at {image_path}: {e}")
        raise e
    
    try:
        save_zarr_labels(
            image_path=image_path,
            data=masks,
            source_name=model_name,
            label_name=label_name,
            input_format=input_format,
            shard_cube_shape=shard_cube_shape,
            chunk_shape=chunk_shape,
            save_mode=label_save_mode,
            timepoint_idxs=timepoint_idxs,
            zarr_driver=zarr_driver,
            dtype=dtype,
        )
    except Exception as e:
        logger.error(f"Failed to save zarr labels at {image_path}/{model_name}/{label_name}: {e}")
        raise e

def save_boxes(
    image_path: str,
    model_name: str,
    boxes: np.ndarray,
    input_format: Literal["TN6", "N6"],
    task: Literal["instance_segmentation", "detection"],
    save_mode: Literal["overwrite", "append"],
    shard_cube_shape: Optional[Tuple[int, int, int]] = None,
    chunk_shape: Optional[Tuple[int, int, int]] = None,
    timepoint_idxs: Optional[List[int]] = None,
) -> None:
    raise NotImplementedError("save_instance_boxes is not implemented")

def read_npy(image_path: str, dtype: Optional[NUMPY_DTYPES | str] = None) -> np.ndarray:
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
    context: Optional[ts.Context] = None,
    cast: bool = False,
    subpath: Optional[str] = None,
) -> np.ndarray:
    """Read a Zarr file and return the data as a NumPy array"""
    spec = _make_read_zarr_spec(image_path, subpath=subpath, driver=zarr_driver)
    ds = ts.open(spec, context=context, read=True).result()
    if cast:
        if dtype is None:
            raise ValueError("dtype is required when cast is True")
        dtype = TENSORSTORE_DTYPES[dtype].value if isinstance(dtype, str) else dtype
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
        save_zarr_data(image_path, data, **kwargs)
    elif image_path.endswith(".tiff") or image_path.endswith(".tif"):
        save_tiff(image_path, data, **kwargs)
    else:
        raise ValueError(f"Unsupported file format for {image_path}")


def _make_write_zarr_spec(
    data_shape: Tuple[int, ...],
    zarr_version: str,
    path: str,
    shard_shape: Tuple[int, int, int] | Tuple[int, int, int, int],
    chunk_shape: Tuple[int, int, int] | Tuple[int, int, int, int],
    subpath: Optional[str] = None,
    dtype: NUMPY_DTYPES | str = "uint16",
) -> Dict[str, Any]:
    if zarr_version == "zarr3":
        zarr_spec = {
            "driver": zarr_version,
            "kvstore": {"driver": "file", "path": path},
            "path": subpath if subpath else "",
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
            "path": subpath if subpath else "",
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

def _make_read_zarr_spec(
    image_path: str,
    subpath: Optional[str] = None,
    driver: str = "zarr3",
) -> Dict[str, Any]:
    spec = {
        "driver": driver,
        "kvstore": {"driver": "file", "path": image_path},
        "path": subpath if subpath else "",
    }
    return spec

# NOTE: taken from ml-data-cell_observatory_platform
def create_zarr_spec(
    data_shape: Tuple[int, ...],
    zarr_version: str,
    path: str,
    input_format: str,
    shard_cube_shape: Tuple[int, int, int],
    chunk_shape: Tuple[int, int, int],
    source_name: Optional[str] = None,
    label_name: Optional[str] = None,
    dtype: NUMPY_DTYPES | str = "uint16",
) -> Dict[str, Any]:
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
            num_channels = data_shape[-1]
            shard_shape = [
                shard_cube_shape[0],
                shard_cube_shape[1],
                shard_cube_shape[2],
                num_channels,
            ]
            chunk_shape = [chunk_shape[0], chunk_shape[1], chunk_shape[2], num_channels]

        elif input_format == "CZYX":
            num_channels = data_shape[0]
            shard_shape = [
                num_channels,
                shard_cube_shape[0],
                shard_cube_shape[1],
                shard_cube_shape[2],
            ]
            chunk_shape = [num_channels, chunk_shape[0], chunk_shape[1], chunk_shape[2]]

        else:
            raise ValueError(f"Unsupported data shape length: {len(data_shape)}")
        
        subpath = None
        if source_name is not None and source_name != "" and source_name != "/":
            if label_name is None or label_name == "" or label_name == "/":
                raise ValueError("label_name is required when source_name is provided")
            subpath = f"{source_name}/{label_name}"
        
        zarr_spec = _make_write_zarr_spec(
            data_shape=data_shape,
            zarr_version=zarr_version,
            path=path,
            shard_shape=shard_shape,
            chunk_shape=chunk_shape,
            dtype=dtype,
            subpath=subpath,
        )
    return zarr_spec


def label_exists(image_path: str, source_name: str, label_name: str, zarr_driver: str) -> bool:
    """
    Returns True if the Zarr label array at `label_name` inside the store
    at `image_path` already exists, False otherwise.
    """
    spec = _make_read_zarr_spec(image_path, subpath=f"{source_name}/{label_name}", driver=zarr_driver)
    try:
        ts.open(spec, open=True, create=False).result()
        return True
    except ValueError as e:
        if "NOT_FOUND" in str(e) or "does not exist" in str(e):
            return False
        raise

def normalize_data_shape(data: np.ndarray, input_format: str) -> np.ndarray:
    """Normalize the data shape to the expected format"""
    if data.ndim == len(input_format):
        return data
    if data.ndim == 3:
        new_axes = []
        if "T" in input_format.upper():
            new_axes.append(input_format.upper().index("T"))
        if "C" in input_format.upper():
            new_axes.append(input_format.upper().index("C"))
        data = np.expand_dims(data, axis=new_axes)
    elif data.ndim == 4:
        raise NotImplementedError("4D data is not supported for reformatting yet")
    elif data.ndim == 5:
        return data
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    return data

VALID_SOURCE_NAME = re.compile(r"^[^\/\\]+$")
VALID_LABEL_NAME = re.compile(r"^[a-zA-Z0-9_]+$")

def save_zarr_labels(
    image_path: str,
    data: np.ndarray,
    source_name: str,
    label_name: str,
    input_format: str,
    save_mode: Literal["overwrite", "create"] = "create",
    shard_cube_shape: Optional[Tuple[int, int, int]] = None,
    chunk_shape: Optional[Tuple[int, int, int]] = None,
    timepoint_idxs: Optional[List[int]] = None,
    zarr_driver: str = "zarr3",
    dtype: str = "uint16",
) -> None:
    """Create or overwrite a label array at <image_path>/<source_name>/<label_name>."""
    if not Path(image_path).resolve().exists():
        raise FileNotFoundError(f"Image path {image_path} does not exist")
    if not VALID_SOURCE_NAME.match(source_name):
        raise ValueError(f"Invalid source name: {source_name}. Source name must be a string and not contain any slashes.")
    if not VALID_LABEL_NAME.match(label_name):
        raise ValueError(f"Invalid label name: {label_name}. Label name must be an alphanumeric + underscore string and not contain any slashes.")
    if save_mode == "create" and (shard_cube_shape is None or chunk_shape is None):
        raise ValueError("shard_cube_shape and chunk_shape are required when creating new labels")
    if source_name is None or source_name == "":
        raise ValueError(f"source_name is required but got {source_name}")
    if label_name is None or label_name == "":
        raise ValueError(f"label_name is required but got {label_name}")

    data = normalize_data_shape(data, input_format)
    if timepoint_idxs is not None:
        if len(timepoint_idxs) != data.shape[0]:
            raise ValueError(f"timepoint_idxs must have the same length as the time dimension of the data but got: {len(timepoint_idxs)=}, {data.shape[0]=}")
    
    exists = label_exists(image_path, source_name, label_name, zarr_driver)
    if exists and save_mode == "create":
        raise ValueError(f"Label {label_name} already exists at {image_path}")
    elif not exists and save_mode == "overwrite":
        raise ValueError(f"Label {label_name} does not exist at {image_path}. Use save_mode='create' to create it.")

    if save_mode == "create":
        spec = create_zarr_spec(
            data_shape=data.shape,
            zarr_version=zarr_driver,
            path=image_path,
            input_format=input_format,
            shard_cube_shape=shard_cube_shape,
            chunk_shape=chunk_shape,
            source_name=source_name,
            label_name=label_name,
            dtype=dtype,
        )
    elif save_mode == "overwrite":
        spec = _make_read_zarr_spec(image_path, subpath=f"{source_name}/{label_name}", driver=zarr_driver)
    else:
        raise ValueError(f"Invalid save_mode: {save_mode}")

    ds = ts.open(spec).result()
    with ts.Transaction() as txn:
        if timepoint_idxs is not None:
            ds.with_transaction(txn)[timepoint_idxs, ...] = data.astype(dtype)
        else:
            ds.with_transaction(txn)[:] = data.astype(dtype)


def save_zarr_data(
    image_path: str,
    data: np.ndarray,
    shard_cube_shape: Tuple[int, int, int],
    chunk_shape: Tuple[int, int, int],
    input_format: str,
    zarr_driver: str = "zarr3",
    dtype: str = "uint16",
) -> None:
    """Create a zarr data array at <image_path> in the root of the zarr store."""
    if image_path is None:
        raise ValueError("image_path is required")
    if Path(image_path).resolve().exists():
        raise FileExistsError(f"Image path {image_path} already exists. If you want to update the data, use update_zarr_data instead.")
    zarr_spec = create_zarr_spec(
        data_shape=data.shape,
        zarr_version=zarr_driver,
        path=image_path,
        input_format=input_format,
        shard_cube_shape=shard_cube_shape,
        chunk_shape=chunk_shape,
        dtype=dtype,
    )

    Path(image_path).mkdir(parents=True, exist_ok=True)
    ds = ts.open(zarr_spec).result()
    with ts.Transaction() as txn:
        ds.with_transaction(txn)[:] = data.astype(dtype)

def normalize_idxs(idxs: Iterable[int | float], shape_size: int) -> List[int]:
    """Normalize a list of indices to be within the bounds of the shape size. Converts negative indices to positive indices."""
    new_idxs = []
    for idx in idxs:
        if not (isinstance(idx, int) or (isinstance(idx, float) and idx.is_integer())):
            raise ValueError(f"Index {idx} is not integer-valued.")
        if idx < 0:
            idx = shape_size + idx
        if idx < 0 or idx >= shape_size:
            raise ValueError(f"Index {idx} is out of bounds for shape size {shape_size}.")
        new_idxs.append(int(idx))
    return new_idxs

def update_zarr_data(
    image_path: str,
    data: np.ndarray,
    input_format: Literal["TZYXC", "ZYXC"],
    zarr_driver: str = "zarr3",
    dtype: str = "uint16",
    timepoint_idxs: Optional[List[int]] = None,
    data_channel_idxs: Optional[List[int]] = None,
    mask_channel_idxs: Optional[List[int]] = None,
    mode: Literal["append", "overwrite"] = "append",
) -> None:
    """Append or overwrite data in an existing zarr array.

    Opens the existing zarr read-write, resizes along the channel dimension if appending,
    and writes data into the channel slice. Uses TensorStore resize() if appending.
    """
    if mode not in ["append", "overwrite"]:
        raise ValueError(f"Invalid mode: {mode}. Must be one of ['append', 'overwrite'].")
    if input_format not in ["TZYXC", "ZYXC"]:
        raise ValueError(f"Invalid input_format: {input_format}. Must be one of ['TZYXC', 'ZYXC'].")
    if not Path(image_path).resolve().exists():
        raise FileNotFoundError(f"Image path {image_path} does not exist")
    if not len(data.shape) == len(input_format):
        raise ValueError(f"data.shape and input_format must have the same number of dimensions but got: {len(data.shape)=}, {len(input_format)=}")
    channel_dim = input_format.index("C") if "C" in input_format else None
    if channel_dim is None:
        raise ValueError(f"Channel dimension is not present in the {input_format=}.")
    if channel_dim != len(input_format) - 1:
        raise NotImplementedError("Only channel-last input_formats are currently supported for appending channels.")
    time_dim = input_format.index("T") if "T" in input_format else None
    if time_dim is not None and time_dim != 0:
        raise NotImplementedError("Only time dimension at the first position is currently supported for appending channels.")
    if time_dim is None and timepoint_idxs is not None:
        raise ValueError(f"Got {timepoint_idxs=} but time dimension is not present in the {input_format=}.")

    read_zarr_spec = _make_read_zarr_spec(image_path, subpath=None, driver=zarr_driver)
    ds = ts.open(read_zarr_spec).result()

    if len(ds.shape) != len(data.shape):
        raise ValueError(f"data.shape and store.shape must have the same number of dimensions but got: {len(data.shape)=}, {len(ds.shape)=}")
    for spatial_dim in ["Z", "Y", "X"]:
        dim_idx = input_format.index(spatial_dim)
        if data.shape[dim_idx] != ds.shape[dim_idx]:
            raise ValueError(f"Data and store have different spatial dimensions: got data.shape[{dim_idx}] != store_shape[{dim_idx}] for dimension {spatial_dim} with input_format={input_format}.")
    if time_dim is not None:
        if timepoint_idxs is not None:
            if len(timepoint_idxs) != data.shape[time_dim]:
                raise ValueError(f"timepoint_idxs must have the same length as the time dimension of the data but got: {len(timepoint_idxs)=}, {data.shape[time_dim]=}")
        else:
            if data.shape[time_dim] != ds.shape[time_dim]:
                raise ValueError(f"Time dimension mismatch: got data.shape[{time_dim}] != store_shape[{time_dim}] for dimension T with input_format={input_format}.")


    if mode == "append":
        if mask_channel_idxs is not None:
            raise ValueError(f"Got {mask_channel_idxs=} but mode is 'append'. Specifying custom mask channel indices is not supported for appending.")
        old_channel_count = ds.shape[channel_dim]
        new_shape = list[Any](ds.shape).copy()
        new_shape[channel_dim] = new_shape[channel_dim] + data.shape[channel_dim]
        n_new = data.shape[channel_dim]
        start_c = old_channel_count
        end_c = old_channel_count + n_new
        with ts.Transaction() as txn:
            txn_store = ds.with_transaction(txn)
            txn_store.resize(exclusive_max=tuple(new_shape), expand_only=True).result()
            # Final sanity check: ensure we DO NOT overwrite data, and only allow overwriting masks.
            # Check if any numbers in the zarr array to be overwritten are not whole integers (possible "data" content?).
            # Use positive channel indices — TensorStore does not support NumPy-style negative slices on zarr views.
            if timepoint_idxs is not None:
                timepoint_idxs = normalize_idxs(timepoint_idxs, ds.shape[time_dim])
                values_to_overwrite = txn_store[timepoint_idxs, ..., start_c:end_c]
            else:
                values_to_overwrite = txn_store[..., start_c:end_c]
            if np.asarray(values_to_overwrite.read().result()).any():
                raise RuntimeError(
                    "Attempted to append to zarr channel(s) that are not empty. "
                    "This strongly suggests that tensorstor did not resize the array as expected or that the specification fill value is not zero."
                    f"data.shape={data.shape}, store_shape_before_resize={ds.shape}"
                    f"channel_dim={channel_dim}, n_new_channels={n_new}"
                    f"n_existing_channels={old_channel_count}"
                    f"timepoint_idxs={timepoint_idxs}"
                )
            values_to_overwrite.write(data.astype(dtype)).result()
    
    elif mode == "overwrite":
        if mask_channel_idxs is None or data_channel_idxs is None:
            raise ValueError(f"Got {mask_channel_idxs=} and {data_channel_idxs=} but mode is 'overwrite'. Mask channel indices and data channel indices must be specified for overwriting.")
        data_channel_idxs = normalize_idxs(data_channel_idxs, ds.shape[channel_dim])
        mask_channel_idxs = normalize_idxs(mask_channel_idxs, ds.shape[channel_dim])
        if len(mask_channel_idxs) != data.shape[channel_dim]:
            raise ValueError(f"Number of mask channel indices ({len(mask_channel_idxs)}) must match the channel count of the input data ({data.shape[channel_dim]}).")
        last_data_channel_idx = max(data_channel_idxs)
        first_mask_channel_idx = min(mask_channel_idxs)
        if first_mask_channel_idx <= last_data_channel_idx:
            raise ValueError(f"Attempting to overwrite data channels: {first_mask_channel_idx=} <= {last_data_channel_idx=}.")
        ms = sorted(mask_channel_idxs)
        if ms != list(range(ms[0], ms[-1] + 1)):
            raise NotImplementedError(
                "Non-contiguous mask_channel_idxs are not supported for TensorStore overwrite writes; "
                f"got {mask_channel_idxs=}"
            )
        ch_start, ch_end = ms[0], ms[-1] + 1
        with ts.Transaction() as txn:
            txn_store = ds.with_transaction(txn)
            if timepoint_idxs is not None:
                timepoint_idxs = normalize_idxs(timepoint_idxs, ds.shape[time_dim])
                values_to_overwrite = txn_store[timepoint_idxs, ..., ch_start:ch_end]
            else:
                values_to_overwrite = txn_store[..., ch_start:ch_end]
            values_to_overwrite.write(data.astype(dtype)).result()
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be one of ['append', 'overwrite'].")

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


def _coerce_bool_in(df: pl.DataFrame, col: str) -> pl.DataFrame:
    _INT_TYPES = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
    dt = df.schema.get(col)

    if dt == pl.Boolean:
        expr = pl.col(col).fill_null(False)

    elif dt in _INT_TYPES:
        expr = (pl.col(col) != 0).fill_null(False)

    else:
        expr = pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_lowercase().is_in(["t", "true", "1"]).fill_null(False)

    return df.with_columns(expr.alias(col))


def filter_hypercubes_dataframe_storage_server(df: pl.DataFrame, server_folder_path: str | None = None) -> pl.DataFrame:
    if server_folder_path is None or str(server_folder_path).startswith("/clusterfs"):
        flag = "exists"
        df = _coerce_bool_in(df, flag).filter(pl.col(flag))
        return df

    if str(server_folder_path).startswith("/groups"):
        flag = "exists_prfs"
    elif str(server_folder_path).startswith("/aws") or str(server_folder_path).startswith(
        "/workspace/CellObservatoryData"
    ):
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

    logger.info(f"Loaded hypercubes on server: {server_folder_path}; shape={df.shape}")
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

    rois = _to_list_or_none(roi_list)
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
            .sort("prepared_id")
            .select(pl.col("prepared_id").head(max_rois))
            .to_series()
            .to_list()
        )
        df = df.filter(pl.col("prepared_id").is_in(keep_rois))

    if max_tiles is not None and "tile_name" in df.columns:
        keep_tiles = (
            df.select(pl.col("tile_name").unique())
            .sort("tile_name")
            .select(pl.col("tile_name").head(max_tiles))
            .to_series()
            .to_list()
        )
        df = df.filter(pl.col("tile_name").is_in(keep_tiles))

    if max_hypercubes is not None:
        df = df.sort(["prepared_id", "tile_name", "z_start", "y_start", "x_start", "time_start"]).head(max_hypercubes)

    return df


def add_has_annotations_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    look for nested entries for each channel in the
    pc_metadata_json col:  {'0': {'histogram': {...}}, '1': {'mask_bbox_dict': {...}}}
    each key is a channel id mapping to a dict of metadata
    """
    if "has_annotations" in df.columns:
        return df
    if "pc_metadata_json" not in df.columns and "metadata_tile_json" not in df.columns:
        return df.with_columns(pl.lit(False).alias("has_annotations"))
    if "pc_metadata_json" in df.columns:
        has_key = pl.col("pc_metadata_json").str.contains(r'"mask_bbox_dict"', literal=False)
        empty_obj = pl.col("pc_metadata_json").str.contains(r'"mask_bbox_dict"\s*:\s*\{\s*\}', literal=False)
        expr = pl.col("pc_metadata_json").is_not_null() & has_key & (~empty_obj)
        return df.with_columns(expr.alias("has_annotations"))
    if "metadata_tile_json" in df.columns:
        has_key = pl.col("metadata_tile_json").str.contains(r'"mask_bbox_dict"', literal=False)
        empty_obj = pl.col("metadata_tile_json").str.contains(r'"mask_bbox_dict"\s*:\s*\{\s*\}', literal=False)
        expr = pl.col("metadata_tile_json").is_not_null() & has_key & (~empty_obj)
        return df.with_columns(expr.alias("has_annotations"))


# FIXME: current nomenclature for metadata may be improved
def create_channel_metadata_columns(df: pl.DataFrame, expected_channel_ids=["0", "1"]) -> pl.DataFrame:
    new_columns = []
    for ch in expected_channel_ids:
        if f"histogram_ch_{ch}" not in df.columns:
            new_columns.append(pl.lit(None).alias(f"histogram_ch_{ch}"))
        if f"mask_bbox_dict_ch_{ch}" not in df.columns:
            new_columns.append(pl.lit(None).alias(f"mask_bbox_dict_ch_{ch}"))
    if new_columns:
        df = df.with_columns(new_columns)
    if "mask_bbox_dict" not in df.columns:
        for ch in expected_channel_ids:
            if f"mask_bbox_dict_ch_{ch}" in df.columns:
                df = df.with_columns(pl.col(f"mask_bbox_dict_ch_{ch}").alias("mask_bbox_dict"))
    # if "histograms" not in df.columns:
    #     for ch in expected_channel_ids:
    #         if f"histogram_ch_{ch}" in df.columns:
    #             df = df.with_columns(pl.col(f"histogram_ch_{ch}").alias("histograms"))
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
    synthetic_only: bool = False,
    has_annotations: bool = False,
) -> tuple[pl.DataFrame, dict]:
    p = Path(hypercubes_dataframe_path)
    if not p.exists():
        raise FileNotFoundError(p)

    t0 = time.perf_counter()
    df = pl.read_csv(p, null_values=["NULL", "null", "NaN", ""])
    t1 = time.perf_counter()
    logger.info(f"Loaded hypercubes dataframe in {t1 - t0:.2f} s; shape={df.shape}")

    # NOTE: database is currently not updated to reflect storage server status
    #       remove this once the database is updated
    # df = filter_hypercubes_dataframe_storage_server(df, server_folder_path)
    df = add_has_annotations_column(df)

    if synthetic_only:
        df = _coerce_bool_in(df, "is_synthetic").filter(pl.col("is_synthetic"))

    if has_annotations:
        df = _coerce_bool_in(df, "has_annotations").filter(pl.col("has_annotations"))

    df = create_channel_metadata_columns(df)

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

    try:
        with open(p.with_suffix(".json"), "r") as f:
            configs = ujson.load(f)
    except FileNotFoundError:
        configs = {}

    return df.to_pandas(use_pyarrow_extension_array=True), configs


def load_tiles_dataframe(
    hypercubes_dataframe_path: str | Path,
    max_rois: int | None = None,
    max_tiles: int | None = None,
    hpf_list: list[int] | None = None,
    roi_list: list[int] | None = None,
    tile_list: list[str] | None = None,
    timepoint_list: list[int] | None = None,
    server_folder_path: str | None = None,
    synthetic_only: bool = False,
    has_annotations: bool = False,
) -> tuple[pd.DataFrame, dict]:
    p = Path(hypercubes_dataframe_path)
    if not p.exists():
        raise FileNotFoundError(p)

    t0 = time.perf_counter()
    df = pl.read_csv(p)
    t1 = time.perf_counter()
    logger.info(f"Loaded tiles dataframe in {t1 - t0:.2f} s; shape={df.shape}")

    # NOTE: database is currently not updated to reflect storage server status
    #       remove this once the database is updated
    # df = filter_hypercubes_dataframe_storage_server(df, server_folder_path)
    df = add_has_annotations_column(df)

    if synthetic_only and "is_synthetic" in df.columns:
        df = _coerce_bool_in(df, "is_synthetic").filter(pl.col("is_synthetic"))

    if has_annotations and "has_annotations" in df.columns:
        df = _coerce_bool_in(df, "has_annotations").filter(pl.col("has_annotations"))

    df = create_channel_metadata_columns(df)

    t0 = time.perf_counter()
    df = apply_hypercubes_dataframe_selections(
        df,
        max_rois=max_rois,
        max_tiles=max_tiles,
        max_hypercubes=None,
        hpf_list=hpf_list,
        roi_list=roi_list,
        tile_list=tile_list,
        timepoint_list=timepoint_list,
    )
    t1 = time.perf_counter()
    logger.info(f"Applied tile selections in {t1 - t0:.2f} s; shape={df.shape}")

    try:
        with open(p.with_suffix(".json"), "r") as f:
            configs = ujson.load(f)
    except FileNotFoundError:
        configs = {}

    return df.to_pandas(use_pyarrow_extension_array=True), configs
