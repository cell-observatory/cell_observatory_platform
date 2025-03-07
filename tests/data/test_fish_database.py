import pytest
import warnings
from datetime import datetime
import os
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
    assert len(fd) == 10 * 200 * (512 // 128) ** 2 * (256 / 128)
    test_out = fd[0]
    assert np.array_equal(test_out[0,...,0], test_arr[...,0])

@pytest.mark.run(order=3)
def test_indexing_fish_database_single_store_batch_not_equal_chunk(kargs):
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
    test_arr = np.ones((16, 256, 256, 256, 4), dtype=np.uint16)
    dataset[0, 0:16, 0:256, 0:256, 0:256, :] = test_arr

    # Create metadata
    metadata = {
        "acquisition_id": [1, ],
        "created_at": [datetime(2025, 3, 6, 10, 30), ],
        "software_version": ["Petakit 1.0", ],
        "output_folder": [path_to_data, ],
        "exists": [True, ]
    }

    # Default data color mode is implicit
    color_mode = ColorMode.AVG
    data_config = DataConfig(t = 16, x=256, y=256, z=256, color_mode = color_mode)
    fd = FishDatabase(clean_up_db=True, metadata=metadata, data_config=data_config)
    assert len(fd) == 10 * (200 // 16) * int((512 // 256) ** 2)
    test_out = fd[0]
    assert np.array_equal(test_out[..., 0], test_arr[..., 0])

@pytest.mark.run(order=4)
def test_empty_metadata():
    metadata = {
        "acquisition_id": [],
        "created_at": [],
        "software_version": [],
        "output_folder": [],
        "exists": []
    }
    fd = FishDatabase(clean_up_db=True, metadata=metadata)
    assert len(fd) == 0

@pytest.mark.run(order=5)
def test_missing_required_metadata_fields():
    metadata = {
        "acquisition_id": [1],
        "software_version": ["Petakit 1.0"],
        "output_folder": ["some_path"],
    }
    # ValueError since "exists" and "created_at" are missing
    with pytest.raises(ValueError):
        fd = FishDatabase(clean_up_db=True, metadata=metadata)

@pytest.mark.run(order=6)
def test_invalid_data_path():
    metadata = {
        "acquisition_id": [1],
        "created_at": [datetime(2025, 3, 6, 10, 30)],
        "software_version": ["Petakit 1.0"],
        "output_folder": ["invalid_path"],  # Invalid path
        "exists": [True],
    }
    fd = FishDatabase(clean_up_db=True, metadata=metadata)
    assert len(fd) == 0  # Should not load any stores, but should not crash either.

@pytest.mark.run(order=7)
def test_db_cleanup():
    metadata = {
        "acquisition_id": [1],
        "created_at": [datetime(2025, 3, 6, 10, 30)],
        "software_version": ["Petakit 1.0"],
        "output_folder": ["some_path"],
        "exists": [True],
    }
    fd = FishDatabase(clean_up_db=True, metadata=metadata)
    db_name = fd.local_db_name

    # Ensure the database exists
    assert os.path.isfile(db_name)

    # Delete the database during cleanup
    del fd
    assert not os.path.isfile(db_name)  # The database should be deleted

@pytest.mark.run(order=8)
def test_indexing_multiple_stores(kargs):
    data_config = DataConfig(x=128, y=128, z=128, color_mode=ColorMode.AVG)
    # Create data
    paths = []
    shapes = ((10,200,512,512,256,3),
              (3,10,256,256,256,3))
    for i in range(2):
        path_to_data = str(kargs['fishdb_dir']) + "_exp_" + str(i)
        paths.append(path_to_data)
        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': f'{path_to_data}'
            },
            'metadata': {
                'dtype': '<u2',
                'shape': shapes[i],
                'chunks': (1, data_config.t, data_config.z, data_config.y, data_config.x, data_config.c),
                'compressor': {'blocksize': 0, 'clevel': 1, 'cname': 'zstd', 'id': 'blosc', 'shuffle': 1},
                'fill_value': 0,
                'order': 'C'
            },
            'create': True,
            'delete_existing': True
        }
        dataset = ts.open(spec).result()
        test_arr = np.ones((data_config.z,data_config.y,data_config.x,3), dtype=np.uint16)
        dataset[0,0,0:data_config.z,0:data_config.y,0:data_config.x,:] = test_arr

    # Create metadata
    metadata = {
        "acquisition_id": [1, 2],
        "created_at": [datetime(2025, 3, 6, 10, 30),] * 2,
        "software_version": ["Petakit 1.0",] * 2,
        "output_folder": paths,
        "exists": [True, ] * 2
    }

    fd = FishDatabase(clean_up_db=True, metadata=metadata, data_config=data_config)
    first_item = fd[0]
    stride = int(10 * 200 * (512 // 128) ** 2 * (256 // 128))
    second_item = fd[stride]
    assert np.array_equal(first_item, second_item)
    assert np.array_equal(second_item[0,...,0], test_arr[...,0])

@pytest.mark.run(order=9)
def test_non_existent_data_slice():
    metadata = {
        "acquisition_id": [1],
        "created_at": [datetime(2025, 3, 6, 10, 30)],
        "software_version": ["Petakit 1.0"],
        "output_folder": ["some_path"],
        "exists": [True],
    }
    data_config = DataConfig(x=128, y=128, z=128)

    fd = FishDatabase(clean_up_db=True, metadata=metadata, data_config=data_config)

    # Try to access a non-existent slice (index out of range)
    with pytest.raises(IndexError):
        _ = fd[len(fd)]  # Should raise IndexError when accessing out-of-bound index
