import pytest
import warnings
warnings.filterwarnings("ignore")

import os
import logging
import tensorstore as ts

@pytest.mark.run(order=1)
def test_tensorstore():
    cwd = os.getcwd()
    logging.info(f'Working directory = {cwd}')
    path_to_data = os.path.join(cwd, 'test_data')
    logging.info(f'Tensorstore path = {path_to_data}')
    dataset = ts.open({
        'driver': 'zarr3',
        'kvstore': {'driver': 'file', 'path': path_to_data},
        'metadata': {
            'shape': [1000, 1000],
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [100, 100]}},
            "chunk_key_encoding": {"name": "default"},
            "data_type": "int4",
            "codecs": [{"name": "blosc", "configuration": {"cname": "lz4", "clevel": 5}}],
        },
        'create': True,
        'delete_existing': True
    }).result()
    dataset[80, 99:105] = [0, 1, 2, 3, 4, 5]
    result = dataset[80, 99:105].read().result()
    for i in range(6):
        assert result[i] == i