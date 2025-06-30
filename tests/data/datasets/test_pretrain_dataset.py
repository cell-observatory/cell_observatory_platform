import pytest

import torch

from omegaconf import open_dict
from hydra.utils import get_class

from tests.conftest import distributed_test, config


def _test_dataloader_dist(config):
    trainer_cls = get_class(config.trainer)
    trainer = trainer_cls(config)

    for idx, data_sample in enumerate(trainer.train_dataloader):
        assert isinstance(data_sample, dict), \
            f"Data sample {idx} is not a dict, got {type(data_sample)}"
        assert "data_tensor" in data_sample and isinstance(data_sample["data_tensor"], torch.Tensor), \
            f"Data sample {idx} does not contain 'data_tensor' key or it is not a tensor, got {type(data_sample['data_tensor'])}"
        assert "metainfo" in data_sample and isinstance(data_sample["metainfo"], dict), \
            f"Data sample {idx} does not contain 'metainfo' key or it is not a dict, got {type(data_sample['metainfo'])}"

        if idx >= 5:
            break

    return {"success": True}


def test_dataloader(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")

    with open_dict(config):
        config.experiment_name = "test_dataloader"
        config.paths.resume_checkpointdir = None
        
        config.clusters.worker_nodes = 1
        config.clusters.gpus_per_worker = torch.cuda.device_count() 
        config.clusters.cpus_per_gpu = 4
        config.clusters.mem_per_cpu = 31000

    metrics = distributed_test(cfg=config, test="tests.data.datasets.test_pretrain_dataset._test_dataloader_dist")
    assert metrics.get("success", True), "Distributed dataloader test failed"