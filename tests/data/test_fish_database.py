import pytest
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import tensorstore as ts
import numpy as np

from data.data_config import DataConfig, ColorMode
from data.fish_database import FishDatabase

@pytest.mark.run(order=1)
def test_construct_fish_database():
    fd = FishDatabase(clean_up_db=True)
    assert fd is not None

@pytest.mark.run(order=2)
def test_indexing_fish_database_single_store(kargs):
    data_config = DataConfig(x=128, y=128, z=128)
    # Create data
    path_to_data = str(kargs['fishdb_dir'])
    spec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': f'{path_to_data}'
        },
        'metadata': {
            'dtype': '<u2',
            'shape': (10,200,512,512,256,4),
            'chunks': (1, data_config.t, data_config.z, data_config.y, data_config.x, data_config.c),
            'compressor': {'blocksize': 0, 'clevel': 1, 'cname': 'zstd', 'id': 'blosc', 'shuffle': 1},
            'fill_value': 0,
            'order': 'C'
        },
        'create': True,
        'delete_existing': True
    }
    dataset = ts.open(spec).result()
    test_arr = np.ones((data_config.z,data_config.y,data_config.x,4), dtype=np.uint16)
    dataset[0,0,0:data_config.z,0:data_config.y,0:data_config.x,:] = test_arr

    # Create metadata
    metadata = {
        "acquisition_id": [1, ],
        "created_at": [datetime(2025, 3, 6, 10, 30),],
        "software_version": ["Petakit 1.0",],
        "output_folder": [path_to_data, ],
        "exists": [True, ]
    }

    fd = FishDatabase(clean_up_db=True, metadata=metadata)
    assert len(fd) == 10*200*(512//128)**2*(256/128)
    test_out = fd[0]
    for i in range(4):
        assert np.array_equal(test_out[0,...,0], test_arr[...,i])

@pytest.mark.run(order=3)
def test_indexing_fish_database_single_store_non_chunk_aligned(kargs):
    # Create data
    path_to_data = str(kargs['fishdb_dir'])
    spec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': f'{path_to_data}'
        },
        'metadata': {
            'dtype': '<u2',
            'shape': (10, 200, 512, 512, 256, 4),
            'chunks': (1, 16, 128, 128, 128, 1),
            'compressor': {'blocksize': 0, 'clevel': 1, 'cname': 'zstd', 'id': 'blosc', 'shuffle': 1},
            'fill_value': 0,
            'order': 'C'
        },
        'create': True,
        'delete_existing': True
    }
    dataset = ts.open(spec).result()
    test_arr = np.ones((16, 256, 256, 256, 1), dtype=np.uint16)
    dataset[0, 0:16, 0:256, 0:256, 0:256, 0:1] = test_arr

    # Create metadata
    metadata = {
        "acquisition_id": [1, ],
        "created_at": [datetime(2025, 3, 6, 10, 30), ],
        "software_version": ["Petakit 1.0", ],
        "output_folder": [path_to_data, ],
        "exists": [True, ]
    }

    # Default data config is implicit
    color_mode = ColorMode.AVG
    data_config = DataConfig(x=128, y=128, z=128, color_mode = color_mode)
    fd = FishDatabase(clean_up_db=True, metadata=metadata, data_config=data_config)
    assert fd is not None
