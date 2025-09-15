import os
import pytest
from pathlib import Path

import torch

from ray.train import report

from omegaconf import open_dict
from omegaconf import DictConfig
from hydra.utils import get_class

from tests.conftest import distributed_test, config


def _test_loggers_dist(cfg: DictConfig):
    import pandas as pd
    from training.loggers import LocalEventWriter
    from utils.context import process_rank, get_world_size
    
    rank = process_rank()
    world = get_world_size()

    trainer_cls = get_class(cfg.trainer)
    trainer = trainer_cls(cfg)

    recorder = trainer.event_recorder
    writers_list = trainer.event_writers_list
    writer = writers_list.writers[0]
    assert isinstance(writer, LocalEventWriter), \
        "Expected LocalEventWriter for testing writers"

    step_csv = writer.step_scalars_savepath
    epoch_csv = writer.epoch_scalars_savepath
    
    assert Path(step_csv).parent.exists(), \
        f"Step scalars directory does not exist: {Path(step_csv).parent}"
    assert Path(epoch_csv).parent.exists(), \
        f"Epoch scalars directory does not exist: {Path(epoch_csv).parent}"

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
        expected_means = {
            it: sum(float(k + it + 1) for k in range(world)) / world
            for it in range(n_steps)
        }
        for _, row in step_df.iterrows():
            assert pytest.approx(row["loss"]) == expected_means[row["iter"]]

        assert epoch_csv.exists(), "epoch CSV missing"
        epoch_df = pd.read_csv(epoch_csv)
        mean_val_loss = sum(float(k + 10) for k in range(world)) / world
        assert len(epoch_df) == 1
        assert pytest.approx(epoch_df.loc[0, "val_loss"]) == mean_val_loss

    # TODO: test appending to existing CSVs

    report({"success": True})


def test_loggers(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")

    with open_dict(config):
        config.experiment_name = "test_event_logging"
        config.paths.resume_checkpointdir = None

        config.loggers.event_writers = [
            w for w in config.loggers.event_writers
            if w._target_.endswith(".LocalEventWriter")
        ]

    metrics = distributed_test(cfg=config, test="tests.training.test_loggers._test_loggers_dist")
    assert metrics.get("success", False), "Distributed event-logging test failed"