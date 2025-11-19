import sys
from pathlib import Path

import pytest
import torch
from hydra.utils import get_class, get_method, instantiate
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from omegaconf import open_dict
from ray.train import report

from cell_observatory_platform.data.datasets.pretrain_dataset_dali import (
    pretrain_dataset_pipeline,
)
from cell_observatory_platform.tests.conftest import config, distributed_test
from cell_observatory_platform.utils.context import process_rank


@pytest.mark.skip("Skipping distributed test for DALI dataloader, DALI dataloader will be deprecated soon.")
def test_access_to_storage_server(config):
    if not Path(config.paths.server_folder_path).exists():
        raise FileNotFoundError(f"{config.paths.server_folder_path} does not exist")


@pytest.mark.skip("Skipping distributed test for DALI dataloader, DALI dataloader will be deprecated soon.")
def test_dataloader_dali(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")

    with open_dict(config):
        config.datasets.collate_fn = None
        config.datasets.dataset._target_ = "data.datasets.pretrain_dataset_dali.PretrainDatasetDali"
        config.datasets.dali_last_batch_policy = {
            "_target_": "nvidia.dali.plugin.base_iterator.LastBatchPolicy",
            "_args_": [1]
        }

    database = instantiate(config.datasets.databases)
    assert database is not None

    dataset = instantiate(
        config.datasets.dataset,
        batch_size=config.clusters.batch_size_per_gpu
    )

    pipe = pretrain_dataset_pipeline(
        dataset=dataset,
        batch_size=config.clusters.batch_size_per_gpu,
        num_threads=config.datasets.num_workers,
        py_start_method="spawn",
        py_num_workers=config.datasets.num_workers,
        prefetch_queue_depth=config.datasets.prefetch_factor,
        exec_async=False,
        exec_pipelined=True,
        device_id=process_rank()
    )
    pipe.build()
    dataloader = DALIGenericIterator(
        pipelines=pipe,
        output_map=["data_tensor", "get_item_time"] if dataset.time else ["data_tensor"],
        size=dataset.full_iterations * config.clusters.batch_size_per_gpu,
        auto_reset=True,
        last_batch_policy=instantiate(config.datasets.dali_last_batch_policy)
    )

    for idx, data_sample in enumerate(dataloader):
        data_tensor = data_sample[0]["data_tensor"]

        assert isinstance(data_tensor, torch.Tensor), "Data tensor should be a PyTorch tensor"
        assert data_tensor.ndim == (dataset.input_layout.ndim + 1), \
            f"Data tensor should have {dataset.input_layout.ndim + 1} dimensions, got {data_tensor.ndim}"
        assert data_tensor.shape[0] == config.clusters.batch_size_per_gpu, \
            f"Data tensor batch size should be {config.clusters.batch_size_per_gpu}, got {data_tensor.shape[0]}"

        if idx >= 5:
            break


@pytest.mark.skip("Skipping distributed test for DALI dataloader, DALI dataloader will be deprecated soon.")
def _test_dataloader_dali_dist(config):
    trainer_cls = get_class(config.trainer)
    trainer = trainer_cls(config)

    for idx, data_sample in enumerate(trainer.train_dataloader):
        data_tensor = data_sample[0]["data_tensor"]

        assert isinstance(data_tensor, torch.Tensor), "Data tensor should be a PyTorch tensor"
        assert data_tensor.ndim == len(list(config.datasets.input_shape)) + 1, \
            f"Data tensor should have {len(list(config.datasets.input_shape)) + 1} dimensions, got {data_tensor.ndim}"

        if idx >= 5:
            break

    return report({"success": True})


@pytest.mark.skip("Skipping distributed test for DALI dataloader, DALI dataloader will be deprecated soon.")
def test_data_pipeline_dali(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")

    with open_dict(config):
        config.experiment_name = "test_data_pipeline_dali"
        config.paths.resume_checkpointdir = None

        config.datasets.collate_fn = None
        config.datasets.transforms.transforms_list = ["data.transforms.normalize.NormalizeDaliWrapper"]
        config.datasets.dataset._target_ = "data.datasets.pretrain_dataset_dali.PretrainDatasetDali"
        config.datasets.dali_last_batch_policy = {
            "_target_": "nvidia.dali.plugin.base_iterator.LastBatchPolicy",
            "_args_": [1]
        }

    metrics = distributed_test(cfg=config, test="cell_observatory_platform.tests.data.test_pretrain_dataset_dali._test_dataloader_dali_dist")
    assert metrics.get("success", False), "Distributed dataloader test failed"