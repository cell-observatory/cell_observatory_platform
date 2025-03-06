import pytest
import warnings
warnings.filterwarnings("ignore")

import tensorstore as ts
import numpy as np

from data.data_config import DataConfig
from data.fish_database import FishDatabase

@pytest.mark.run(order=1)
def test_construct_fish_database():
    fd = FishDatabase(clean_up_db=True)
    assert fd is not None

@pytest.mark.run(order=2)
def test_indexing_fish_database():
    data_config = DataConfig(x=128, y=128, z=128)
    path_to_data = '/opt/project/test_data'
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
    test_arr = np.ones((data_config.z,data_config.y,data_config.x), dtype=np.uint16)
    dataset[0,0,0:data_config.z,0:data_config.y,0:data_config.x,0] = test_arr

    fd = FishDatabase(clean_up_db=True)
    assert len(fd) == 10*200*(512//128)**2*(256/128)
    assert np.array_equal(fd[0].squeeze(), test_arr)