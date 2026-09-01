import tempfile

import pytest
import torch
from omegaconf import open_dict
from ray.train import report, Checkpoint

from cell_observatory_platform.tests.conftest import distributed_test
from cell_observatory_platform.utils.cleanup import unlink_shared_memory
from cell_observatory_platform.data.dataloaders import get_dataloader
from cell_observatory_platform.utils.context import is_main_process


def _test_dataloader_ray_dist(config):
    train_dataloader, val_dataloader, dataloader_config, _, _, _ = get_dataloader(config)
    expected_dims = len(list(config.datasets.input_shape)) + 1

    for idx, data_sample in enumerate(train_dataloader):
        data_tensor = data_sample["data_tensor"]

        assert isinstance(data_tensor, torch.Tensor), "data_tensor should be a Torch tensor"
        assert (
            data_tensor.ndim == expected_dims
        ), f"Expected {expected_dims} dims (including batch), got {data_tensor.ndim}"
        assert data_tensor.shape[1:] == tuple(
            config.datasets.input_shape
        ), f"Expected input shape {config.datasets.input_shape}, got {data_tensor.shape[1:]}"

        if idx >= 2:
            break

    metrics = {"success": True}
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Checkpoint.from_directory(tmpdir)
        if is_main_process():
            return report(metrics=metrics, checkpoint=checkpoint)
        else:
            return report(metrics=metrics, checkpoint=None)


@pytest.mark.cuda
@pytest.mark.localdb
def test_data_pipeline_ray_distributed(config):
    """Ray dataloader end to end: every emitted batch has the configured input shape."""
    with open_dict(config):
        config.datasets.split = 0.2
        config.datasets.last_batch_policy = "drop"
        config.datasets.num_workers = "${clusters.cpus_per_worker}"
        config.datasets.use_arrow_tensor_v2 = True
        config.datasets.locality_with_output = True
        config.datasets.rows_per_block = "${clusters.batch_size_per_gpu}"
        config.datasets.buffer_capacity = 4
        config.datasets.pin_numa_node = True
        config.datasets.pin_memory = True
        config.datasets.max_concurrent_calls = 512
        config.datasets.numa_node_affinity_policy = "distance"
        config.datasets.numa_oversub_factor = 2.0
        config.datasets.actor_oversub_factor = 2.0
        config.datasets.debug = True
        config.datasets.with_batched_api = True
        config.datasets.num_actors_min = 1
        config.datasets.num_actors_max = 1
        config.datasets.context = {
            "file_io_concurrency": None,
            "data_copy_concurrency": None,
            "cache_pool": {"total_bytes_limit": 0},
        }
        config.experiment_name = "test_data_pipeline_ray"
        config.paths.resume_checkpointdir = None

    metrics = distributed_test(
        cfg=config,
        test="cell_observatory_platform.tests.data.test_pretrain_dataset_ray._test_dataloader_ray_dist",
    )
    assert metrics.get("success") is True
    unlink_shared_memory()
