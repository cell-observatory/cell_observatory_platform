import os
import sys
from pathlib import Path
import tempfile
from unittest import mock

import pandas as pd
import pytest
import ray
import torch
from hydra.utils import get_class
from omegaconf import DictConfig, open_dict
from ray.train import report
from ray.train import Checkpoint

from cell_observatory_platform.tests.conftest import config, distributed_test
from cell_observatory_platform.training.loggers import EventRecorder, EventWriterList, LocalEventWriter
from cell_observatory_platform.utils.context import get_world_size, process_rank, is_main_process


def _simple_dataloader(config: DictConfig):
    with open_dict(config):
        config.runtime = {
            "train_steps_per_epoch": 1,
            "val_steps_per_epoch": 1,
            "n_train_rows": 2,
            "n_val_rows": 1,
        }

    class _DummyDeviceBuffer:
        def put_free(self, idx):
            pass

    dataloader_config = {
        "cfg": config,
        "batch_size": config.clusters.batch_size_per_gpu,
        "last_batch_policy": "pad",
        "collate_fn": None,
        "database": None,
    }
    return [], [], dataloader_config, None, _DummyDeviceBuffer(), pd.DataFrame()


def test_put_scalar_batch_recorder_and_reduce():
    """Multiple observations per step are stored and reduced like single put_scalar calls."""
    rec = EventRecorder()
    rec._iter, rec._epoch = 3, 0
    rec.put_scalar_batch(
        name="async_metric",
        values=[1.0, 2.0, 3.0],
        scope="step",
        reduce_method=["median", "mean", "max"],
        category="cat",
        prefix="prefix",
        units="units",
    )
    step_name = get_metric_full_name(
        name="async_metric",
        scope="step",
        category="cat",
        prefix="prefix",
        units="units",
    )
    async_metric_rows = rec.get_step_scalars()[step_name]
    assert len(async_metric_rows) == 3
    assert [t[0] for t in async_metric_rows] == [1.0, 2.0, 3.0], "Values not properly recorded"
    assert [t[1] for t in async_metric_rows] == [0, 1, 2], "Step indices not properly recorded"
    assert [t[2] for t in async_metric_rows] == [0, 0, 0], "Epoch indices not properly recorded"

    with tempfile.TemporaryDirectory() as tmpdir:
        lw = LocalEventWriter(
            rec,
            save_dir=tmpdir,
            step_scalars_prefix="step_batch",
            epoch_scalars_prefix="epoch_batch",
        )
        wlist = EventWriterList([lw])
        step_scalars, _ = wlist.reduce_scalars()
        median_step_name = f"{step_name}_median"
        mean_step_name = f"{step_name}_mean"
        max_step_name = f"{step_name}_max"
        assert median_step_name in step_scalars
        assert mean_step_name in step_scalars
        assert max_step_name in step_scalars
        assert pytest.approx(step_scalars[median_step_name][0][0]) == 1.0
        assert pytest.approx(step_scalars[mean_step_name][0][0]) == 2.0
        assert pytest.approx(step_scalars[max_step_name][0][0]) == 1.0


def _test_loggers_dist(cfg: DictConfig):

    rank = process_rank()
    world = get_world_size()

    trainer_cls = get_class(cfg.trainer)
    with mock.patch(
        "cell_observatory_platform.training.loops.get_dataloader",
        side_effect=_simple_dataloader,
    ):
        trainer = trainer_cls(cfg)

    recorder = trainer.event_recorder
    writers_list = trainer.event_writers_list
    writer = writers_list.writers[0]
    assert isinstance(writer, LocalEventWriter), "Expected LocalEventWriter for testing writers"

    step_csv = writer.step_scalars_savepath
    epoch_csv = writer.epoch_scalars_savepath

    assert Path(step_csv).parent.exists(), f"Step scalars directory does not exist: {Path(step_csv).parent}"
    assert Path(epoch_csv).parent.exists(), f"Epoch scalars directory does not exist: {Path(epoch_csv).parent}"

    if Path(step_csv).exists() and Path(step_csv).is_file() and Path(step_csv).match("*.csv"):
        # remove old step scalars CSV if it exists
        Path(step_csv).unlink()
        print("Step scalars CSV removed from previous test runs.")
    if Path(epoch_csv).exists() and Path(epoch_csv).is_file() and Path(epoch_csv).match("*.csv"):
        # remove old epoch scalars CSV if it exists
        Path(epoch_csv).unlink()
        print("Epoch scalars CSV removed from previous test runs.")

    # test putting scalars
    n_steps = 3
    for it in range(n_steps):
        trainer._iter, recorder._iter = it, it
        trainer._epoch, recorder._epoch = 0, 0
        recorder.put_scalar("loss", float(rank + it + 1), scope="step")

    # test all gathers scalars from all workers
    step_scalars, _ = writers_list.reduce_scalars()
    # test write scalars on rank 0
    writer._write_scalar_impl(step_scalars, scope="step")

    # test clearing scalars method
    assert len(recorder.get_step_scalars()["loss"]) == n_steps
    recorder.clear_scalars()
    assert all(len(v) == 0 for v in recorder.get_step_scalars().values())

    # test putting epoch scalars
    trainer._epoch, recorder._epoch = 0, 0
    recorder.put_scalar("val_loss", float(rank + 10), scope="epoch")

    # test all gathers epoch scalars from all workers
    _, epoch_scalars = writers_list.reduce_scalars()
    # test write epoch scalars on rank 0
    writer._write_scalar_impl(epoch_scalars, scope="epoch")

    # no-op for LocalEventWriter
    writers_list.close()

    # test that the scalars were written
    # and reduced correctly
    # TODO: remove old CSVs to prevent counting
    #       old rows
    if rank == 0:
        assert step_csv.exists(), "step CSV missing"
        step_df = pd.read_csv(step_csv)
        expected_means = {it: sum(float(k + it + 1) for k in range(world)) / world for it in range(n_steps)}
        for _, row in step_df.iterrows():
            assert pytest.approx(row["loss_median"]) == expected_means[row["iter"]]

        assert epoch_csv.exists(), "epoch CSV missing"
        epoch_df = pd.read_csv(epoch_csv)
        mean_val_loss = sum(float(k + 10) for k in range(world)) / world
        assert len(epoch_df) == 1
        assert pytest.approx(epoch_df.loc[0, "val_loss_median"]) == mean_val_loss

    # TODO: test appending to existing CSVs

    metrics = {"success": True}
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Checkpoint.from_directory(tmpdir)
        if is_main_process():
            return report(metrics=metrics, checkpoint=checkpoint)
        else:
            return report(metrics=metrics, checkpoint=None)


def test_loggers(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")

    ray.shutdown()

    with open_dict(config):
        config.experiment_name = "test_event_logging"
        config.paths.resume_checkpointdir = None

        config.loggers.event_writers = [
            w for w in config.loggers.event_writers if w._target_.endswith(".LocalEventWriter")
        ]

    metrics = distributed_test(
        cfg=config, test="cell_observatory_platform.tests.training.test_loggers._test_loggers_dist"
    )
    assert metrics.get("success", False), "Distributed event-logging test failed"