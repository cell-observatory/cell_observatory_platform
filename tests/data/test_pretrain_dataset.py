import sys
from pathlib import Path

import pytest
import torch
from hydra.utils import get_class, get_method, instantiate
from omegaconf import DictConfig, open_dict
from ray.train import report
from torch.utils.data import DataLoader

from cell_observatory_platform.tests.conftest import config, distributed_test


@pytest.mark.skip("Skipping tests for Torch dataloader, Torch dataloader will be deprecated soon.")
def test_access_to_storage_server(config):
    if not Path(config.paths.server_folder_path).exists():
        raise FileNotFoundError(f"{config.paths.server_folder_path} does not exist")


@pytest.mark.skip("Skipping tests for Torch dataloader, Torch dataloader will be deprecated soon.")
def test_dataloader(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")

    with open_dict(config):
        config.experiment_name = "test_dataloader"

    database = instantiate(config.datasets.databases)
    assert database is not None

    dataset = instantiate(config.datasets.dataset)

    if isinstance(config.datasets.collate_fn, DictConfig):
        collate_fn = instantiate(config.datasets.collate_fn)
    else:
        collate_fn = get_method(config.datasets.collate_fn)

    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=2,
        persistent_workers=False,
        worker_init_fn=dataset.worker_init_fn,
        drop_last=True,
    )

    for idx, data_sample in enumerate(dataloader):
        assert isinstance(data_sample, dict), \
            f"Data sample {idx} is not a dict, got {type(data_sample)}"

        assert "data_tensor" in data_sample and isinstance(data_sample["data_tensor"][0], torch.Tensor), \
            f"Data sample {idx} does not contain 'data_tensor' key or it is not a tensor, got {type(data_sample['data_tensor'][0])}"

        assert "metainfo" in data_sample and isinstance(data_sample["metainfo"], dict), \
            f"Data sample {idx} does not contain 'metainfo' key or it is not a dict, got {type(data_sample['metainfo'])}"
 
        expected_shape = (
            1,
            data_sample['metainfo']["time_size"][0].item(),
            data_sample['metainfo']["cube_size"][0].item(),
            data_sample['metainfo']["cube_size"][0].item(),
            data_sample['metainfo']["cube_size"][0].item(),
            data_sample['metainfo']["channel_size"][0].item()
        )
        assert data_sample['data_tensor'][0].shape == expected_shape, \
            f"Data tensor shape {data_sample['data_tensor'][0].shape} does not match expected shape {expected_shape}"
        
        
        if idx >= 5:
            break


def _test_dataloader_dist(config):
    trainer_cls = get_class(config.trainer)
    trainer = trainer_cls(config)

    for idx, data_sample in enumerate(trainer.train_dataloader):

        assert isinstance(data_sample, dict), f"Data sample {idx} is not a dict, got {type(data_sample)}"

        assert "data_tensor" in data_sample and isinstance(data_sample["data_tensor"][0], torch.Tensor), \
            f"Data sample {idx} does not contain 'data_tensor' key or it is not a tensor, got {type(data_sample['data_tensor'][0])}"

        assert "metainfo" in data_sample and isinstance(data_sample["metainfo"], dict), \
            f"Data sample {idx} does not contain 'metainfo' key or it is not a dict, got {type(data_sample['metainfo'])}"

        expected_shape = (
            1,
            data_sample['metainfo']["time_size"][0].item(),
            data_sample['metainfo']["cube_size"][0].item(),
            data_sample['metainfo']["cube_size"][0].item(),
            data_sample['metainfo']["cube_size"][0].item(),
            data_sample['metainfo']["channel_size"][0].item()
        )

        assert data_sample['data_tensor'][0].shape == expected_shape, \
            f"Data tensor shape {data_sample['data_tensor'][0].shape} does not match expected shape {expected_shape}"

        if idx >= 5:
            break

    return report({"success": True})


@pytest.mark.skip("Skipping distributed test for Torch dataloader, Torch dataloader will be deprecated soon.")
def test_data_pipeline(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")

    with open_dict(config):
        config.experiment_name = "test_data_pipeline"
        config.paths.resume_checkpointdir = None

    metrics = distributed_test(cfg=config, test="tests.data.test_pretrain_dataset._test_dataloader_dist")
    assert metrics.get("success", False), "Distributed dataloader test failed"