import pytest
import warnings
warnings.filterwarnings("ignore")

from data.data_config import DataConfig
from data.data_utils import index_mapper, middle_out_crop_start_index

@pytest.mark.run(order=1)
def test_index_mapper_single_batch():
    shape = (1, 1, 1, 1, 1, 1)
    out = index_mapper(shape, data_config=DataConfig(x=1,y=1, z=1))
    assert len(out) == 1
    for i in range(6):
        assert out[0][i] == 0

    shape = (1, 1, 128, 128, 128, 1)
    out = index_mapper(shape, data_config=DataConfig(x=128,y=128, z=128))
    assert len(out) == 1
    for i in range(6):
        assert out[0][i] == 0

    shape = (1, 16, 128, 128, 128, 1)
    out = index_mapper(shape, data_config=DataConfig(t=16, x=128, y=128, z=128))
    assert len(out) == 1
    for i in range(6):
        assert out[0][i] == 0

    shape = (1, 16, 128, 128, 128, 2)
    out = index_mapper(shape, data_config=DataConfig(t=16, x=128, y=128, z=128, c=2))
    assert len(out) == 1
    for i in range(6):
        assert out[0][i] == 0

    shape = (1, 16, 128, 128, 128, 3)
    out = index_mapper(shape, data_config=DataConfig(t=16, x=128, y=128, z=128, c=2))
    assert out is None

@pytest.mark.run(order=2)
def test_middle_out_crop_start_index():
    data_config = DataConfig(t=1, x=128, y=128, z=128)

    shape = (1, 1, 1, 128, 128, 1)
    y0, x0 = middle_out_crop_start_index(shape, data_config)
    assert y0 == 0
    assert x0 == 0

    shape = (1, 1, 1, 130, 129, 1)
    y0, x0 = middle_out_crop_start_index(shape, data_config)
    assert y0 == 1
    assert x0 == 0

    shape = (1, 1, 1, 256, 256, 1)
    y0, x0 = middle_out_crop_start_index(shape, data_config)
    assert y0 == 0
    assert x0 == 0

    shape = (1, 1, 1, 258, 258, 1)
    y0, x0 = middle_out_crop_start_index(shape, data_config)
    assert y0 == 1
    assert x0 == 1
