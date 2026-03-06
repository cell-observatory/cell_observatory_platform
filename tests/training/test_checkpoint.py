import os
import sys
from pathlib import Path
import tempfile

import pytest
import torch
from hydra.utils import get_class
from omegaconf import DictConfig, open_dict
from ray.train import report, Checkpoint

from cell_observatory_platform.tests.conftest import config, distributed_test
from cell_observatory_platform.utils.context import is_main_process


# train function to use, this should stay inside
# the setup_ray_cluster function to prevent serialization issues
def _test_ckpt_dist(config: DictConfig):
    from cell_observatory_platform.training.helpers import get_masked_input_data

    trainer_cls = get_class(config.trainer)
    trainer_per_worker = trainer_cls(config)

    # it is tempting to set batches lower
    # but if batches is too low checkpoint saving
    # does not work correctly probably due to the fact that
    # the sharding of the model is not done correctly/completely
    def dummy_loader(model, batches=100):
        inputs = [1] + config.datasets.input_shape
        for _ in range(batches):
            yield get_masked_input_data(model, inputs)

    def _move_to_device(batch, device):
        if isinstance(batch, (list, tuple)):
            return [_move_to_device(b, device=device) for b in batch]
        elif isinstance(batch, dict):
            return {k: _move_to_device(v, device=device) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=True)
        else:
            return batch

    # we need to do a few steps for
    # some state dict fields to be populated
    # such as exp_avg etc.
    model = trainer_per_worker.model
    for batch in dummy_loader(model):
        batch = _move_to_device(batch, device=model.device)
        loss_dict, outputs = model(*batch)
        model.backward(loss_dict["step_loss"])
        model.step()

    # in the save test we save a checkpoint and
    # record start_epoch, start_iter, and best_metric
    # with dummy value of 42 and in load checkpoint test
    # we load the same checkpoints for all combinations of zero stages
    # and check if the client state is restored correctly, i.e. if
    # start_epoch, start_iter, and best_metric are all 42
    # TODO: add check to ensure weights are identical
    if config.save_checkpoint:
        trainer_per_worker.checkpoint_manager.save(
            prefix=config.checkpoint.checkpoint_manager.checkpoint_tag, save_epoch=42, save_best_loss=42, save_step=42
        )
        metrics = {"success": True}
    else:
        metrics = {
            "success": True,
            "start_epoch": trainer_per_worker.start_epoch,
            "start_iter": trainer_per_worker.start_iter,
            "best_metric": trainer_per_worker.best_metric,
        }
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Checkpoint.from_directory(tmpdir)
        if is_main_process():
            return report(metrics=metrics, checkpoint=checkpoint)
        else:
            return report(metrics=metrics, checkpoint=None)


# @pytest.mark.order(1)
# @pytest.mark.parametrize("zero_stage", [1]) # 2, 3
@pytest.mark.skip(reason="This test is temporarily disabled.")
def test_checkpoint_save(config, zero_stage: int):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("At least 2 GPUs are required for this test")

    with open_dict(config):
        # no resume checkpoint for saving test
        config.paths.resume_checkpointdir = None
        config.deepspeed.zero_optimization.stage = zero_stage
        config.experiment_name = "test_checkpoint"
        config.checkpoint.checkpoint_manager.checkpoint_tag = f"stage{zero_stage}"

        config.clusters.worker_nodes = 1
        # need >=2 for testing zero stages > 0
        config.clusters.gpus_per_worker = 2
        config.clusters.cpus_per_gpu = 4

        config.save_checkpoint = True

    result_dict = distributed_test(
        cfg=config, test="cell_observatory_platform.tests.training.test_checkpoint._test_ckpt_dist"
    )
    assert result_dict["success"], "Test did not complete successfully"


# @pytest.mark.order(2)
# @pytest.mark.parametrize("zero_stage_dst", [1]) # , 2, 3
# @pytest.mark.parametrize("zero_stage_src", [1]) # , 2, 3
@pytest.mark.skip(reason="This test is temporarily disabled.")
def test_checkpoint_load(config, zero_stage_src: int, zero_stage_dst: int):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")
    else:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("At least 2 GPUs are required for this test")

    with open_dict(config):
        config.experiment_name = f"test_checkpoint"
        config.deepspeed.zero_optimization.stage = zero_stage_dst
        config.paths.resume_checkpointdir = os.path.join(config.paths.outdir, "checkpoints")
        config.checkpoint.checkpoint_manager.checkpoint_tag = f"stage{zero_stage_src}"

        config.clusters.worker_nodes = 1
        # need >=2 for testing zero stages > 0
        config.clusters.gpus_per_worker = 2
        config.clusters.cpus_per_gpu = 4

        config.deepspeed.checkpoint.load_universal = True

        config.save_checkpoint = False

    result_dict = distributed_test(
        cfg=config, test="cell_observatory_platform.tests.training.test_checkpoint._test_ckpt_dist"
    )

    assert result_dict["start_epoch"] == 42, f"Expected start_epoch to be 42, got {result_dict['start_epoch']}"
    assert result_dict["start_iter"] == 42, f"Expected start_iter to be 42, got {result_dict['start_iter']}"
    assert result_dict["best_metric"] == 42, f"Expected best_metric to be 42, got {result_dict['best_metric']}"
    assert result_dict["success"], "Test did not complete successfully"