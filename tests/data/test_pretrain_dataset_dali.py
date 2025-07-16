import pytest
from pathlib import Path
from omegaconf import open_dict
from hydra.utils import get_class

from hydra.utils import instantiate, get_method

import torch

from nvidia.dali.plugin.pytorch import DALIGenericIterator

from ray.train import report

from tests.conftest import distributed_test, config
from data.datasets.pretrain_dataset_dali import pretrain_dataset_pipeline

def test_access_to_storage_server(config):
    if not Path(config.paths.server_folder_path).exists():
        raise FileNotFoundError(f"{config.paths.server_folder_path} does not exist")


# @pytest.mark.skip("Skipping distributed test for DALI dataloader while database is being updated")
def test_dataloader_dali(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")


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
    )
    pipe.build()
    dataloader = DALIGenericIterator(
        pipelines = pipe,
        output_map = ["data_tensor"],
        size = dataset.full_iterations * config.clusters.batch_size_per_gpu,
        auto_reset = True,
        last_batch_policy = instantiate(config.datasets.dali_last_batch_policy)
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

# @pytest.mark.skip("Skipping distributed test for DALI dataloader while database is being updated")
def test_data_pipeline_dali(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")

    with open_dict(config):
        config.experiment_name = "test_data_pipeline_dali"
        config.paths.resume_checkpointdir = None

    metrics = distributed_test(cfg=config, test="tests.data.test_pretrain_dataset._test_dataloader_dali_dist")
    assert metrics.get("success", False), "Distributed dataloader test failed"