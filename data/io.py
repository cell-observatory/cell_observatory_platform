import sys
import logging
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
import tensorstore as ts
from tifffile import TiffFile
from skimage.io import imread, imsave

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_npy(image_path: str) -> np.ndarray:
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

    return img.astype(np.float32)


def read_tiff(image_path: str) -> np.ndarray:
    """ Read a TIFF file and return the data as a NumPy array """
    img = imread(image_path)

    if np.isnan(np.sum(img)):
        logger.error("NaN!")

    return img.astype(np.float32)

def read_zarr(image_path: str, zarr_driver: str = "zarr3") -> np.ndarray:
    """ Read a Zarr file and return the data as a NumPy array """
    spec = {
        "driver": zarr_driver,
        "kvstore": {"driver": "file", "path": image_path},
    }
    ds = ts.open(spec, read=True).result()
    return ds


def read_file(image_path: str | Path, **kwargs) -> str:
    """ Infer the file format of the image based on its extension """
    image_path = str(image_path)

    if image_path.endswith(".zarr"):
        return read_zarr(image_path, **kwargs)

    elif image_path.endswith(".tiff") or image_path.endswith(".tif"):
        return read_tiff(image_path)

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


# NOTE: taken from ml-data-platform
def create_zarr_spec(zarr_version, path, data_shape, shard_cube_shape, chunk_shape, num_timepoints_per_image):

    if zarr_version == 'zarr3':
        if len(data_shape) == 5:
            shard_shape = [num_timepoints_per_image, shard_cube_shape[0], shard_cube_shape[1], shard_cube_shape[2], 1]
            chunk_shape = [num_timepoints_per_image, chunk_shape[0], chunk_shape[1], chunk_shape[2], 1]
        elif len(data_shape) == 4:
            shard_shape = [num_timepoints_per_image, shard_cube_shape[0], shard_cube_shape[1], shard_cube_shape[2]]
            chunk_shape = [num_timepoints_per_image, chunk_shape[0], chunk_shape[1], chunk_shape[2]]
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

