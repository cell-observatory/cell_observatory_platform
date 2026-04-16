import sys
from pathlib import Path
import tempfile

import pytest
import torch
from hydra.utils import get_class, instantiate
from omegaconf import open_dict
from ray.train import report, Checkpoint

from cell_observatory_platform.tests.conftest import config, distributed_test
from cell_observatory_platform.utils.cleanup import unlink_shared_memory
from cell_observatory_platform.data.dataloaders import get_dataloader
from cell_observatory_platform.data.datasets.utils import (
    resolve_channel_localization_indices,
)
from cell_observatory_platform.utils.context import is_main_process

def test_access_to_storage_server(config):
    if not Path(config.paths.server_folder_path).exists():
        raise FileNotFoundError(f"{config.paths.server_folder_path} does not exist")


def test_resolve_channel_localization_indices_from_string_mapping():
    mapping = '{"0":"tdmstaygold-membrane","1":"myonghong-histone"}'

    indices = resolve_channel_localization_indices(mapping, ["histone", "membrane"])

    assert indices == [1, 0]


def test_resolve_channel_localization_indices_from_dict_mapping():
    mapping = {"0": "tdmstaygold-membrane", "1": "myonghong-histone"}

    indices = resolve_channel_localization_indices(mapping, ["membrane"])

    assert indices == [0]


def test_resolve_channel_localization_indices_raises_when_missing():
    mapping = {"0": "tdmstaygold-membrane"}

    with pytest.raises(ValueError, match="Unable to resolve requested channel localization"):
        resolve_channel_localization_indices(mapping, ["histone"])


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


def test_data_pipeline_ray_distributed(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for distributed Ray test")

    with open_dict(config):
        config.datasets.split = 0.2
        config.datasets.return_dataloader = True
        config.datasets.distributed_sampler = True
        config.datasets.prefetch_factor = 1
        config.datasets.num_workers = "${clusters.cpus_per_worker}"

        config.datasets.last_batch_policy = "drop"

        config.datasets.collate_fn = {
            "_target_": "cell_observatory_platform.data.datasets.pretrain_dataset_ray.CollatorActor",
            "dtype": "${dataset_dtype}",
            "buffer_dtype": "${storage_dtype}",
            "batch_size": "${clusters.batch_size_per_gpu}",
            "input_shape": "${datasets.input_shape}",
            "device_buffer_capacity": 2,
            "pin_numa_node": "${datasets.pin_numa_node}",
            "pin_pages": "${datasets.pin_memory}",
        }

        config.datasets.dataset = {
            "_target_": "cell_observatory_platform.data.datasets.pretrain_dataset_ray.PretrainDatasourceRay",
            "hypercubes_dataframe_path": "${datasets.hypercubes_dataframe_path}",
            "server_folder_path": "${datasets.server_folder_path}",
            "max_rois": "${datasets.max_rois}",
            "max_tiles": "${datasets.max_tiles}",
            "max_hypercubes": "${datasets.max_hypercubes}",
            "hpf_list": "${datasets.hpf_list}",
            "roi_list": "${datasets.roi_list}",
            "tile_list": "${datasets.tile_list}",
            "synthetic_only": "${datasets.synthetic_only}",
            "has_annotations": "${datasets.has_annotations}",
            "columns": "${datasets.columns}",
            "input_layout": {
                "_target_": "cell_observatory_platform.data.data_shapes.MULTICHANNEL_HYPERCUBE",
                "value": "${dataset_layout_order}",
            },
        }

        config.datasets.channels_subset = None
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

        config.datasets.context = {
            "file_io_concurrency": None,
            "data_copy_concurrency": None,
            "cache_pool": {"total_bytes_limit": 0},
        }

        config.experiment_name = f"test_data_pipeline_ray"
        config.paths.resume_checkpointdir = None

        config.datasets.with_batched_api = True
        config.datasets.num_actors_min = 1
        config.datasets.num_actors_max = 1

        config.datasets.context = {
            "file_io_concurrency": None,
            "data_copy_concurrency": None,
            "cache_pool": {"total_bytes_limit": 0},
        }

    metrics = distributed_test(
        cfg=config,
        test="cell_observatory_platform.tests.data.test_pretrain_dataset_ray._test_dataloader_ray_dist",
    )
    assert metrics.get("success", False), f"Distributed Ray dataloader test failed"

    unlink_shared_memory()
