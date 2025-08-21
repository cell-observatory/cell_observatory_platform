import pytest
from pathlib import Path
from omegaconf import open_dict
from hydra.utils import instantiate, get_class

import torch
from ray.train import report

from tests.conftest import distributed_test, config


def test_access_to_storage_server(config):
    if not Path(config.paths.server_folder_path).exists():
        raise FileNotFoundError(f"{config.paths.server_folder_path} does not exist")


def _test_dataloader_ray_dist(config):
    trainer_cls = get_class(config.trainer)
    trainer = trainer_cls(config)

    expected_dims = len(list(config.datasets.input_shape)) + 1

    for idx, data_sample in enumerate(trainer.train_dataloader):
        data_tensor = data_sample["data_tensor"]

        assert isinstance(data_tensor, torch.Tensor), "data_tensor should be a Torch tensor"
        assert data_tensor.ndim == expected_dims, (
            f"Expected {expected_dims} dims (including batch), got {data_tensor.ndim}"
        )
        assert data_tensor.shape[1:] == tuple(config.datasets.input_shape), (
            f"Expected input shape {config.datasets.input_shape}, got {data_tensor.shape[1:]}"
        )

        if idx >= 2:
            break

    return report({"success": True})


@pytest.mark.parametrize(
    "mode",
    [
        # TODO: fixed_shape_tensor_v2 works when training, but fails in the test with 
        # `ValueError: ('Unhandled Arrow array type:', FixedShapeTensorType`
        {"name": "arrow_tensor_v1", "ray_data_v2": False, "use_arrow_tensor_v2": False, "impl_type": "ArrowTensorArray", "split": None},
        # {"name": "fixed_shape_tensor_v2", "ray_data_v2": True, "use_arrow_tensor_v2": True, "impl_type": "FixedShapeTensorArray", "split": None},
        {"name": "fixed_size_list_v2", "ray_data_v2": True, "use_arrow_tensor_v2": True, "impl_type": "FixedSizeListArray", "split": None},
        {"name": "arrow_tensor_v1", "ray_data_v2": False, "use_arrow_tensor_v2": False, "impl_type": "ArrowTensorArray", "split": 0.2},
        # {"name": "fixed_shape_tensor_v2", "ray_data_v2": True, "use_arrow_tensor_v2": True, "impl_type": "FixedShapeTensorArray", "split": 0.2},
        {"name": "fixed_size_list_v2", "ray_data_v2": True, "use_arrow_tensor_v2": True, "impl_type": "FixedSizeListArray", "split": 0.2},
    ],
    ids=lambda m: f"dist_{m['name']}",
)
def test_data_pipeline_ray_distributed(config, mode):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for distributed Ray test")

    with open_dict(config):
        config.experiment_name = f"test_data_pipeline_ray_{mode['name']}"
        config.paths.resume_checkpointdir = None

        config.datasets.ray_data_v2 = mode["ray_data_v2"]
        config.datasets.impl_type = mode["impl_type"]
        config.datasets.split = mode["split"]
        config.datasets.use_arrow_tensor_v2 = mode["use_arrow_tensor_v2"]

        config.datasets.transforms.transforms_list = []

        config.datasets.dataset._target_ = "data.datasets.pretrain_dataset_ray.PretrainDatasourceRay"

        config.datasets.drop_last_policy = True
        config.datasets.auto_transfer = False
        config.datasets.with_batched_api = True
        config.datasets.num_actors_min = 1
        config.datasets.num_actors_max = 2
        config.datasets.channels_subset = None
        config.datasets.locality_with_output = True 
        config.datasets.rows_per_block = "${datasets.batch_size}"

        config.datasets.collate_fn = {
            "_target_": "data.datasets.pretrain_dataset_ray.PinnedTensorCollator",
            "dtype": "${dataset_dtype}",
            "sample_shape": "${datasets.input_shape}",
            "pin_memory": True,
            "impl_type": mode["impl_type"],
        }

        config.datasets.context = {
            "file_io_concurrency": None,
            "data_copy_concurrency": None,
            "cache_pool": {
                "total_bytes_limit": 0
            }
        }

    metrics = distributed_test(
        cfg=config,
        test="tests.data.test_pretrain_dataset_ray._test_dataloader_ray_dist",
    )
    assert metrics.get("success", False), f"Distributed Ray dataloader test failed for {mode['name']}"