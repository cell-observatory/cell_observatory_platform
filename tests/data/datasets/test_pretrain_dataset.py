import pytest
from hydra.utils import instantiate
from omegaconf import open_dict

import torch
from torch.utils.data import DataLoader

from tests.conftest import config


def test_dataloader(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")

    with open_dict(config):
        config.experiment_name = "test_dataloader"

    database = instantiate(config.datasets.databases)
    assert database is not None

    dataset = instantiate(config.datasets.dataset)

    dataloader = DataLoader(
        dataset,
        collate_fn=instantiate(config.datasets.collate_fn),
        batch_size=config.clusters.batch_size_per_gpu,
        shuffle=False,
        pin_memory=True,
        num_workers=config.clusters.cpus_per_worker,
        prefetch_factor=2,
        persistent_workers=False,
        worker_init_fn=dataset.worker_init_fn,
        drop_last=True,
    )

    for idx, data_sample in enumerate(dataloader):
        assert isinstance(data_sample, dict), \
            f"Data sample {idx} is not a dict, got {type(data_sample)}"

        assert "data_tensor" in data_sample and isinstance(data_sample["data_tensor"], torch.Tensor), \
            f"Data sample {idx} does not contain 'data_tensor' key or it is not a tensor, got {type(data_sample['data_tensor'])}"

        assert "metainfo" in data_sample and isinstance(data_sample["metainfo"], dict), \
            f"Data sample {idx} does not contain 'metainfo' key or it is not a dict, got {type(data_sample['metainfo'])}"

        if idx >= 5:
            break
