import os
from pathlib import Path
from hydra.utils import instantiate

from tests.conftest import config
from data.io import read_file

def test_access_to_storage_server(config):
    if not Path(config.paths.server_folder_path).exists():
        raise FileNotFoundError(f"{config.paths.server_folder_path} does not exist")

def test_data_loading(config):
    database = instantiate(config.datasets.databases)
    assert database is not None

    dataset = instantiate(config.datasets.dataset)
    sample = dataset._index[0]
    print(sample)

    file = os.path.join(sample["server_folder"], sample["output_folder"], sample["tile_name"])

    if not Path(file).exists():
        raise FileNotFoundError(f"{file} does not exist")
    else:
        print(f"Loading {file}")

    data_tensor = read_file(file)
    hypercube = dataset._slice_hypercube(data_tensor, sample)

    expected_shape = (sample["time_size"], sample["cube_size"], sample["cube_size"], sample["cube_size"], sample["channel_size"])
    assert hypercube.shape == expected_shape, f"Data tensor shape {hypercube.shape} does not match expected shape {expected_shape}"


