import sys
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

from data.data_types import TENSORSTORE_DTYPES, NUMPY_DTYPES, TORCH_DTYPES

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
    hypercubes_dataframe: pd.DataFrame,
    server_folder_path: Optional[Path | str] = None
):
    if server_folder_path is None or str(server_folder_path).startswith('/clusterfs'):
        if hypercubes_dataframe['exists'].dtype == 'str' and hypercubes_dataframe['exists'].str.contains('|'.join(['t', 'f'])).any():
            hypercubes_dataframe['exists'].replace({'t': True, 'f': False}, inplace=True)
            hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists']]
            
        elif hypercubes_dataframe['exists'].dtype == int:
            hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists'] == 1]
            hypercubes_dataframe['exists'] = hypercubes_dataframe['exists'].astype(bool)
            
        else:
            hypercubes_dataframe['exists'] = hypercubes_dataframe['exists'].astype(bool)
            hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists']]
        
        logger.info(f"Using ABC {server_folder_path=}, {hypercubes_dataframe.shape}")

    elif str(server_folder_path).startswith('/groups'):
        if hypercubes_dataframe['exists_prfs'].dtype == 'str' and hypercubes_dataframe['exists_prfs'].str.contains('|'.join(['t', 'f'])).any():
            hypercubes_dataframe['_'].replace({'t': True, 'f': False}, inplace=True)
            hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists_prfs']]
            
        elif hypercubes_dataframe['exists_prfs'].dtype == int:
            hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists_prfs'] == 1]
            hypercubes_dataframe['exists_prfs'] = hypercubes_dataframe['exists_prfs'].astype(bool)
            
        else:
            hypercubes_dataframe['exists_prfs'] = hypercubes_dataframe['exists_prfs'].astype(bool)
            hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists_prfs']]

        hypercubes_dataframe['server_folder'] = server_folder_path
        logger.info(f"Using Janelia {server_folder_path=}, {hypercubes_dataframe.shape}")

    elif str(server_folder_path).startswith('/aws'):
        if hypercubes_dataframe['exists_aws'].dtype == 'str' and hypercubes_dataframe['exists_aws'].str.contains('|'.join(['t', 'f'])).any():
            hypercubes_dataframe['exists_aws'].replace({'t': True, 'f': False}, inplace=True)
            hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists_aws']]
            
        elif hypercubes_dataframe['exists_aws'].dtype == int:
            hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists_aws'] == 1]
            hypercubes_dataframe['exists_aws'] = hypercubes_dataframe['exists_aws'].astype(bool)
            
        else:
            hypercubes_dataframe['exists_aws'] = hypercubes_dataframe['exists_aws'].astype(bool)
            hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists_aws']]

        hypercubes_dataframe['server_folder'] = server_folder_path
        logger.info(f"Using AWS {server_folder_path=}, {hypercubes_dataframe.shape}")

    elif str(server_folder_path).startswith('/lustre'):
        if hypercubes_dataframe['exists_oak'].dtype == 'str' and hypercubes_dataframe['exists_oak'].str.contains('|'.join(['t', 'f'])).any():
            hypercubes_dataframe['exists_oak'].replace({'t': True, 'f': False}, inplace=True)
            hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists_oak']]
            
        elif hypercubes_dataframe['exists_oak'].dtype == int:
            hypercubes_dataframe = hypercubes_dataframe[hypercubes_dataframe['exists_oak'] == 1]
            hypercubes_dataframe['exists_oak'] = hypercubes_dataframe['exists_oak'].astype(bool)
            
        else:
            hypercubes_dataframe['exists_oak'] = hypercubes_dataframe['exists_oak'].astype(bool)
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
    timepoint_list: Optional[Iterable[int]] = None,
):
    df = hypercubes_dataframe

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

    if rois is not None or tiles is not None:
        conds = []
        if rois is not None and 'prepared_id' in df.columns:
            conds.append(df['prepared_id'].isin(rois))
        elif rois is not None:
            logger.warning("Column 'prepared_id' not found; skipping ROI filter.")

        if tiles is not None and 'tile_name' in df.columns:
            conds.append(df['tile_name'].isin(tiles))
        elif tiles is not None:
            logger.warning("Column 'tile_name' not found; skipping tile filter.")

        if timepoint_list is not None and 'time_start' in df.columns:
            conds.append(df['time_start'].isin(timepoint_list))
        elif timepoint_list is not None:
            logger.warning("Column 'time_start' not found; skipping timepoint filter.")

        if conds:
            cond = conds[0]
            for c in conds[1:]:
                cond &= c
            df = df[cond]

    if hpfs is not None:
        if 'hpf' in df.columns:
            df = df[df['hpf'].isin(hpfs)]
        else:
            logger.warning("Column 'hpf' not found; skipping HPF filter.")

    if max_rois is not None and 'prepared_id' in df.columns:
        keep_rois = df['prepared_id'].drop_duplicates().head(max_rois).tolist()
        df = df[df['prepared_id'].isin(keep_rois)]

    if max_tiles is not None and 'tile_name' in df.columns:
        keep_tiles = df['tile_name'].drop_duplicates().head(max_tiles).tolist()
        df = df[df['tile_name'].isin(keep_tiles)]

    if max_hypercubes is not None:
        df = df.head(max_hypercubes)

    return df

def _string_seq_to_float_list(value: Any) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [float(x) for x in np.asarray(value).ravel().tolist()]
    if isinstance(value, (int, float, np.floating, np.integer)):
        return [float(value)]
    if isinstance(value, str):
        s = value.strip()
        if s == "" or s.lower() in {"null", "none", "nan"}:
            raise ValueError("Cannot convert empty or null string to float list")
        s = s.strip("[]{}()").replace("\n", " ")
        s = s.translate(str.maketrans("", "", "\"'"))
        if s == "":
            raise ValueError("Cannot convert empty string to float list")
        s = re.sub(r"[,\s]+", " ", s)
        arr = np.fromstring(s, sep=" ", dtype=float)
        return arr.tolist()
    
    else:
        raise TypeError(f"Cannot convert value of type {type(value)} to float list")


def apply_occupancy_threshold(
    hypercubes_dataframe: pd.DataFrame,
    occupancy_threshold: Optional[float] = 0.,
    occupancy_threshold_filter_type: Literal['min_all', 'min_ch0', 'min_ch1'] = 'min_ch0'
):
    t = 0. if occupancy_threshold is None else occupancy_threshold

    logger.info(f"\nApplied filters:\n{occupancy_threshold=}")

    hypercubes_dataframe['occupancy_ratios_ch_0'] = hypercubes_dataframe['occupancy_ratios_ch_0'].apply(_string_seq_to_float_list)
    hypercubes_dataframe['mean_occupancy_ratios_ch_0'] = hypercubes_dataframe['occupancy_ratios_ch_0'].apply(np.mean)
    hypercubes_dataframe['min_occupancy_ratios_ch_0'] = hypercubes_dataframe['occupancy_ratios_ch_0'].apply(np.min)
    hypercubes_dataframe['med_occupancy_ratios_ch_0'] = hypercubes_dataframe['occupancy_ratios_ch_0'].apply(np.median)

    hypercubes_dataframe['occupancy_ratios_ch_1'] = hypercubes_dataframe['occupancy_ratios_ch_1'].apply(_string_seq_to_float_list)
    hypercubes_dataframe['mean_occupancy_ratios_ch_1'] = hypercubes_dataframe['occupancy_ratios_ch_1'].apply(np.mean)
    hypercubes_dataframe['min_occupancy_ratios_ch_1'] = hypercubes_dataframe['occupancy_ratios_ch_1'].apply(np.min)
    hypercubes_dataframe['med_occupancy_ratios_ch_1'] = hypercubes_dataframe['occupancy_ratios_ch_1'].apply(np.median)

    if occupancy_threshold_filter_type == 'min_all':
        return hypercubes_dataframe[
            (hypercubes_dataframe['min_occupancy_ratios_ch_0'] >= t) &
            (hypercubes_dataframe['min_occupancy_ratios_ch_1'] >= t)
        ]
    elif occupancy_threshold_filter_type == 'min_ch0':
        return hypercubes_dataframe[
            (hypercubes_dataframe['min_occupancy_ratios_ch_0'] >= t)
        ]
    elif occupancy_threshold_filter_type == 'min_ch1':
        return hypercubes_dataframe[
            (hypercubes_dataframe['min_occupancy_ratios_ch_1'] >= t)
        ]
    else:
        raise ValueError(f"Unknown occupancy_threshold_filter_type: {occupancy_threshold_filter_type}")


def apply_hypercubes_dataframe_filters(
    hypercubes_dataframe: pd.DataFrame,
    occupancy_threshold: Optional[float] = 0.,
    occupancy_threshold_filter_type: str = 'min_all'
):
    hypercubes_dataframe = apply_occupancy_threshold(
        hypercubes_dataframe=hypercubes_dataframe,
        occupancy_threshold=occupancy_threshold,
        occupancy_threshold_filter_type=occupancy_threshold_filter_type
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
    timepoint_list: Optional[Iterable[int]] = None,
    server_folder_path: Optional[Path | str] = None,
    occupancy_threshold: Optional[float] = None,
    occupancy_threshold_filter_type: str = 'min_all'
) -> Tuple[pd.DataFrame, Dict]:

    if not Path(hypercubes_dataframe_path).exists():
        raise FileNotFoundError(f"{hypercubes_dataframe_path} does not exist")

    hypercubes = pd.read_csv(hypercubes_dataframe_path, header=0)
    logger.info(
        f"Setup hypercubes dataframe from {hypercubes_dataframe_path} {hypercubes.shape}"
    )

    hypercubes = filter_hypercubes_dataframe_storage_server(
        hypercubes_dataframe=hypercubes,
        server_folder_path=server_folder_path,
    )

    hypercubes = apply_hypercubes_dataframe_selections(
        hypercubes_dataframe=hypercubes,
        max_rois=max_rois,
        max_tiles=max_tiles,
        max_hypercubes=max_hypercubes,
        hpf_list=hpf_list,
        roi_list=roi_list,
        tile_list=tile_list,
        timepoint_list=timepoint_list,
    )

    hypercubes = apply_hypercubes_dataframe_filters(
        hypercubes_dataframe=hypercubes,
        occupancy_threshold=occupancy_threshold,
        occupancy_threshold_filter_type=occupancy_threshold_filter_type
    )

    logger.info(f"Loaded hypercubes dataframe with {hypercubes.shape}")
    logger.info(f"Columns: {hypercubes.columns}")
    logger.info(hypercubes.head())

    try: 
        with open(hypercubes_dataframe_path.with_suffix('.json'), 'r') as f:
            configs = ujson.load(f)
    except FileNotFoundError:
        configs = {}
        
    return hypercubes, configs