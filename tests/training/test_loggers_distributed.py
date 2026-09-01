"""Distributed event logging: every rank records step/epoch scalars into its
own EventRecorder, LocalEventWriter pools them across ranks, reduces once on
rank 0, and writes the CSV logbooks. Runs a Ray TorchTrainer worker group on
the node's GPUs."""

import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import ray
from hydra.utils import get_class
from omegaconf import DictConfig, open_dict
from ray.train import Checkpoint, report

from cell_observatory_platform.tests.conftest import distributed_test
from cell_observatory_platform.training.helpers import get_metric_full_name
from cell_observatory_platform.training.loggers import LocalEventWriter
from cell_observatory_platform.utils.context import (
    get_world_size,
    is_main_process,
    process_rank,
    reduce_values,
)


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

    # remove logbooks left by previous runs so row counts are exact
    if Path(step_csv).is_file() and Path(step_csv).match("*.csv"):
        Path(step_csv).unlink()
    if Path(epoch_csv).is_file() and Path(epoch_csv).match("*.csv"):
        Path(epoch_csv).unlink()

    # step scalars: each rank records a distinct value per iteration
    n_steps = 3
    for it in range(n_steps):
        trainer._iter, recorder._iter = it, it
        trainer._epoch, recorder._epoch = 0, 0
        recorder.put_scalar("loss", float(rank + it + 1), scope="step")

    # gather from all ranks, write on rank 0
    step_scalars, _ = writers_list.reduce_scalars()
    writer._write_scalar_impl(step_scalars, scope="step")

    # the recorder stores under the structured key, and clears on request
    loss_step_key = get_metric_full_name(name="loss", scope="step")
    assert len(recorder.get_step_scalars()[loss_step_key]) == n_steps
    recorder.clear_scalars()
    assert all(len(v) == 0 for v in recorder.get_step_scalars().values())

    # epoch scalars
    trainer._epoch, recorder._epoch = 0, 0
    recorder.put_scalar("val_loss", float(rank + 10), scope="epoch")

    _, epoch_scalars = writers_list.reduce_scalars()
    writer._write_scalar_impl(epoch_scalars, scope="epoch")

    # no-op for LocalEventWriter
    writers_list.close()

    if rank == 0:
        assert step_csv.exists(), "step CSV missing"
        step_df = pd.read_csv(step_csv)
        # the CSV column is the structured key plus the (default median) reduce op
        step_loss_col = get_metric_full_name(name="loss", scope="step") + "_median"
        expected = {it: reduce_values("median", [float(k + it + 1) for k in range(world)])
                    for it in range(n_steps)}
        assert len(step_df) == n_steps
        for _, row in step_df.iterrows():
            assert pytest.approx(row[step_loss_col]) == expected[row["iter"]]

        assert epoch_csv.exists(), "epoch CSV missing"
        epoch_df = pd.read_csv(epoch_csv)
        epoch_val_loss_col = get_metric_full_name(name="val_loss", scope="epoch") + "_median"
        assert len(epoch_df) == 1
        assert pytest.approx(epoch_df.loc[0, epoch_val_loss_col]) == \
            reduce_values("median", [float(k + 10) for k in range(world)])

    metrics = {"success": True}
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Checkpoint.from_directory(tmpdir)
        if is_main_process():
            return report(metrics=metrics, checkpoint=checkpoint)
        else:
            return report(metrics=metrics, checkpoint=None)


@pytest.mark.cuda
def test_local_writer_reduces_across_ranks_and_writes_csv(config):
    """Across the worker group, per-rank step and epoch scalars land in the
    rank-0 CSV logbooks as the cross-rank median the writer is configured to
    compute."""
    ray.shutdown()

    with open_dict(config):
        config.experiment_name = "test_event_logging"
        config.paths.resume_checkpointdir = None

        # Event writers are REGISTRY specs keyed by `name` (see
        # configs/loggers/loggers.yaml); keep only the local CSV writer so the
        # test does not touch W&B.
        local_writers = [w for w in config.loggers.event_writers if w.name == "local"]
        assert local_writers, (
            "no event writer named 'local' in config.loggers.event_writers; "
            "the registered name may have been renamed"
        )
        config.loggers.event_writers = local_writers

    metrics = distributed_test(
        cfg=config,
        test="cell_observatory_platform.tests.training.test_loggers_distributed._test_loggers_dist",
    )
    assert metrics.get("success", False), "Distributed event-logging test failed"
