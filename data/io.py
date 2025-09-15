import sys
import logging
from pathlib import Path
from typing import Tuple, Literal, Optional, Iterable, Dict

import ujson
import inspect
import functools

import torch

import numpy as np
import tensorstore as ts

from tifffile import TiffFile
from skimage.io import imread, imsave

import pandas as pd

from data.data_types import TENSORSTORE_DTYPES, NUMPY_DTYPES, TORCH_DTYPES

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_npy(image_path: str, dtype: NUMPY_DTYPES | str = NUMPY_DTYPES.fp16) -> np.ndarray:
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

    dtype = NUMPY_DTYPES[dtype].value if isinstance(dtype, str) else dtype
    return img.astype(dtype)


def read_tiff(image_path: str, dtype: NUMPY_DTYPES | str = NUMPY_DTYPES.fp16) -> np.ndarray:
    """ Read a TIFF file and return the data as a NumPy array """
    img = imread(image_path)

    if np.isnan(np.sum(img)):
        logger.error("NaN!")

    dtype = NUMPY_DTYPES[dtype].value if isinstance(dtype, str) else dtype
    return img.astype(dtype)


def read_zarr(
    image_path: str,
    zarr_driver: str = "zarr3",
    dtype: Optional[TENSORSTORE_DTYPES | str] = None,
    context: ts.Context | None = None,
    cast: bool = True
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
    dtype: NUMPY_DTYPES | TENSORSTORE_DTYPES | TORCH_DTYPES | str = NUMPY_DTYPES.fp16,
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

    if image_path.endswith(".zarr"):
        save_zarr(image_path, data, **kwargs)

    elif image_path.endswith(".tiff") or image_path.endswith(".tif"):
        save_tiff(image_path, data)

    else:
        raise ValueError(f"Unsupported file format for {image_path}")


# NOTE: taken from ml-data-cell_observatory_platform
def create_zarr_spec(zarr_version, path, data_shape, shard_cube_shape, chunk_shape, num_timepoints_per_image):

    if zarr_version == 'zarr3':
        if len(data_shape) == 5:
            shard_shape = [num_timepoints_per_image, shard_cube_shape[0], shard_cube_shape[1], shard_cube_shape[2], 2]
            chunk_shape = [1, chunk_shape[0], chunk_shape[1], chunk_shape[2], 2]
        elif len(data_shape) == 4:
            shard_shape = [num_timepoints_per_image, shard_cube_shape[0], shard_cube_shape[1], shard_cube_shape[2]]
            chunk_shape = [1, chunk_shape[0], chunk_shape[1], chunk_shape[2]]
        else:
            raise ValueError(f"Unsupported data shape length: {len(data_shape)}")

        zarr_spec = {
            'driver': zarr_version,
            'kvstore': {
                'driver': 'file',
                'path': path
            },
            'metadata': {
                'data_type': 'uint16',
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
    zarr_driver: str = "zarr3"
) -> None:
    data_shape, num_timepoints_per_image = data.shape, data.shape[0]
    zarr_spec = create_zarr_spec(
        zarr_driver,
        image_path,
        data_shape,
        shard_cube_shape, chunk_shape,
        num_timepoints_per_image
    )

    ds = ts.open(zarr_spec).result()
    ds[:] = data


def save_tiff(image_path: str, data: np.ndarray) -> None:
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
    hypercubes_dataframe: pd.DataFrame,
    server_folder_path: Optional[Path | str] = None
):
    if server_folder_path is None or str(server_folder_path).startswith('/clusterfs'):
        hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists']]
        logger.info(f"Using ABC {server_folder_path=}, {hypercubes_dataframe.shape}")

    elif str(server_folder_path).startswith('/groups'):
        hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists_prfs']]
        hypercubes_dataframe['server_folder'] = server_folder_path
        logger.info(f"Using Janelia {server_folder_path=}, {hypercubes_dataframe.shape}")

    elif str(server_folder_path).startswith('/aws'):
        hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists_aws']]
        hypercubes_dataframe['server_folder'] = server_folder_path
        logger.info(f"Using AWS {server_folder_path=}, {hypercubes_dataframe.shape}")

    elif str(server_folder_path).startswith('/lustre'):
        hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists_oak']]
        hypercubes_dataframe['server_folder'] = server_folder_path
        logger.info(f"Using OakRidge {server_folder_path=}, {hypercubes_dataframe.shape}")

    else:
        raise ValueError(f"Unknown server_folder_path: {server_folder_path}")

    return hypercubes_dataframe


def apply_hypercubes_dataframe_selections(
    hypercubes_dataframe: pd.DataFrame,
    max_rois: Optional[int] = None,
    max_tiles: Optional[int] = None,
    max_hypercubes: Optional[int] = None,
    hpf_list: Optional[Iterable[int]] = None,
    roi_list: Optional[Iterable[int]] = None,
    tile_list: Optional[Iterable[str]] = None,
):
    logger.info(
        f"\nApplied selections:\n"
        f"{hpf_list=}\n"
        f"{roi_list=}\n"
        f"{tile_list=}\n"
        f"{max_rois=}\n"
        f"{max_tiles=}\n"
        f"{max_hypercubes=}"
    )

    if roi_list is not None or tile_list is not None:
        rois = list(roi_list)
        tiles = list(tile_list)

        if rois is not None and tiles is not None:
            hypercubes_dataframe = hypercubes_dataframe[
                (hypercubes_dataframe['prepared_id'].isin(rois)) & (hypercubes_dataframe['tile_name'].isin(tiles))
                ]
        elif rois is not None:
            hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['prepared_id'].isin(rois)]
        elif tiles is not None:
            hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['tile_name'].isin(tiles)]

    if hpf_list is not None:
        hpfs = list(hpf_list)
        hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['hpf'].isin(hpfs)]

    if max_rois is not None:
        unique_rois = hypercubes_dataframe['prepared_id'].unique().tolist()
        hypercubes_dataframe = hypercubes_dataframe[
            hypercubes_dataframe['prepared_id'].isin(unique_rois[:max_rois])
        ]

    if max_tiles is not None:
        unique_tiles = hypercubes_dataframe['tile_name'].unique().tolist()
        hypercubes_dataframe = hypercubes_dataframe[
            hypercubes_dataframe['tile_name'].isin(unique_tiles[:max_tiles])
        ]

    if max_hypercubes is not None:
        hypercubes_dataframe = hypercubes_dataframe.head(max_hypercubes)

    return hypercubes_dataframe


def apply_occupancy_threshold(
    hypercubes_dataframe: pd.DataFrame,
    occupancy_threshold: Optional[float] = 0.
):
    def _string_set_to_list(value):
        if isinstance(value, str):
            clean_str = value.strip('{}')
            if clean_str.startswith('[') and clean_str.endswith(']'):
                clean_str = clean_str.strip('[]')
            if clean_str:
                return [float(x.strip()) for x in clean_str.split(' ') if x.strip()]
            else:
                return []
        elif isinstance(value, list):
            return [float(x) for x in value]
        else:
            return [float(value)] if value is not None else []

    t = 0. if occupancy_threshold is None else occupancy_threshold

    logger.info(f"\nApplied filters:\n{occupancy_threshold=}")

    hypercubes_dataframe['occupancy_ratios_ch_0'] = hypercubes_dataframe['occupancy_ratios_ch_0'].apply(_string_set_to_list)
    hypercubes_dataframe['mean_occupancy_ratios_ch_0'] = hypercubes_dataframe['occupancy_ratios_ch_0'].apply(np.mean)
    hypercubes_dataframe['min_occupancy_ratios_ch_0'] = hypercubes_dataframe['occupancy_ratios_ch_0'].apply(np.min)
    hypercubes_dataframe['med_occupancy_ratios_ch_0'] = hypercubes_dataframe['occupancy_ratios_ch_0'].apply(np.median)

    hypercubes_dataframe['occupancy_ratios_ch_1'] = hypercubes_dataframe['occupancy_ratios_ch_1'].apply(_string_set_to_list)
    hypercubes_dataframe['mean_occupancy_ratios_ch_1'] = hypercubes_dataframe['occupancy_ratios_ch_1'].apply(np.mean)
    hypercubes_dataframe['min_occupancy_ratios_ch_1'] = hypercubes_dataframe['occupancy_ratios_ch_1'].apply(np.min)
    hypercubes_dataframe['med_occupancy_ratios_ch_1'] = hypercubes_dataframe['occupancy_ratios_ch_1'].apply(np.median)

    return hypercubes_dataframe[
        (hypercubes_dataframe['min_occupancy_ratios_ch_0'] >= t) &
        (hypercubes_dataframe['min_occupancy_ratios_ch_1'] >= t)
    ]


def apply_hypercubes_dataframe_filters(
    hypercubes_dataframe: pd.DataFrame,
    occupancy_threshold: Optional[float] = 0.
):
    hypercubes_dataframe = apply_occupancy_threshold(
        hypercubes_dataframe=hypercubes_dataframe,
        occupancy_threshold=occupancy_threshold,
    )

    logger.info(hypercubes_dataframe[['min_occupancy_ratios_ch_0', 'min_occupancy_ratios_ch_1']].describe(
        percentiles=[0, .25, .5, .75, .8, .9, .95, .99, 1])
    )

    return hypercubes_dataframe


def load_hypercubes_dataframe(
    hypercubes_dataframe_path: str | Path,
    max_rois: Optional[int] = None,
    max_tiles: Optional[int] = None,
    max_hypercubes: Optional[int] = None,
    hpf_list: Optional[Iterable[int]] = None,
    roi_list: Optional[Iterable[int]] = None,
    tile_list: Optional[Iterable[str]] = None,
    server_folder_path: Optional[Path | str] = None,
    occupancy_threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, Dict]:

    if not Path(hypercubes_dataframe_path).exists():
        raise FileNotFoundError(f"{hypercubes_dataframe_path} does not exist")

    hypercubes = pd.read_csv(hypercubes_dataframe_path, index_col=0, header=0)
    logger.info(
        f"Setup hypercubes dataframe from {hypercubes_dataframe_path} {hypercubes.shape}"
    )

    hypercubes = filter_hypercubes_dataframe_storage_server(
        hypercubes_dataframe=hypercubes,
        server_folder_path=server_folder_path,
    )

    hypercubes = apply_hypercubes_dataframe_filters(
        hypercubes_dataframe=hypercubes,
        occupancy_threshold=occupancy_threshold,
    )

    hypercubes = apply_hypercubes_dataframe_selections(
        hypercubes_dataframe=hypercubes,
        max_rois=max_rois,
        max_tiles=max_tiles,
        max_hypercubes=max_hypercubes,
        hpf_list=hpf_list,
        roi_list=roi_list,
        tile_list=tile_list,
    )

    logger.info(f"Loaded hypercubes dataframe with {hypercubes.shape}")
    logger.info(f"Columns: {hypercubes.columns}")
    logger.info(hypercubes.head())

    with open(hypercubes_dataframe_path.with_suffix('.json'), 'r') as f:
        configs = ujson.load(f)

    return hypercubes, configs
