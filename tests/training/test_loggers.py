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
from omegaconf import DictConfig, OmegaConf, open_dict
from ray.train import report
from ray.train import Checkpoint

from cell_observatory_platform.tests.conftest import config, distributed_test
from cell_observatory_platform.training.helpers import (
    METRIC_CATEGORY_NAMES,
    get_metric_full_name,
    log_data_sample_metrics,
    log_loss_dict,
    make_timing_metric,
)
from cell_observatory_platform.training.loggers import EventRecorder, EventWriterList, LocalEventWriter, WandBEventWriter
from cell_observatory_platform.utils.context import get_world_size, process_rank, is_main_process


def test_metric_full_name_uses_prefix_and_system_category():
    assert get_metric_full_name(
        name="step_time",
        scope="step",
        category="timing",
        prefix="val",
        units="sec",
    ) == "step_timing/val/step_time_sec"
    assert get_metric_full_name(
        name="reserved_mem",
        scope="step",
        category="system",
        units="GB",
    ) == "step_system/reserved_mem_GB"
    assert "system" in METRIC_CATEGORY_NAMES


def test_log_data_sample_metrics_adds_second_units():
    """Timing records under metainfo['metrics'] resolve to _sec keys with timing reductions."""
    class _Trainer:
        event_recorder = EventRecorder()

    data_sample = {
        "metainfo": {
            "metrics": [
                make_timing_metric("data_time", 0.1),
                make_timing_metric("get_item_time", torch.tensor([0.2, 0.4])),
                make_timing_metric("preprocess_time", 0.3),
                make_timing_metric("masking_time", 0.4),
                make_timing_metric("collate_time", 0.5),
                make_timing_metric("slice_time", torch.tensor([0.6, 0.8])),
                make_timing_metric("transform_time", 0.7),
            ],
        },
    }

    log_data_sample_metrics(_Trainer(), data_sample, default_phase="validation")

    expected = {
        get_metric_full_name(name=name, scope="step", category="timing", prefix="val", units="sec")
        for name in (
            "data_time",
            "get_item_time",
            "preprocess_time",
            "masking_time",
            "collate_time",
            "slice_time",
            "transform_time",
        )
    }
    recorded = _Trainer.event_recorder.get_step_scalars()
    assert expected.issubset(recorded.keys())
    # Reductions on each record should be timing-style (median/max/min)
    for full_name in expected:
        assert _Trainer.event_recorder.get_reduce_op(full_name) == ["median", "max", "min"]
    # Tensor get_item_time was reduced by mean to a scalar before logging
    got_full = get_metric_full_name(
        name="get_item_time", scope="step", category="timing", prefix="val", units="sec",
    )
    assert recorded[got_full][0][0] == pytest.approx(0.3)


def test_log_data_sample_metrics_phase_mapping():
    """phase canonical names map to expected prefixes; training has no prefix."""
    class _Trainer:
        event_recorder = EventRecorder()

    sample = {
        "metainfo": {
            "metrics": [
                {
                    "metric_name": "data_time",
                    "value": 0.1,
                    "category": "timing",
                    "phase": "training",
                    "reduce_method": ["median"],
                },
                {
                    "metric_name": "data_time",
                    "value": 0.2,
                    "category": "timing",
                    "phase": "validation",
                    "reduce_method": ["median"],
                },
                {
                    "metric_name": "data_time",
                    "value": 0.3,
                    "category": "timing",
                    "phase": "testing",
                    "reduce_method": ["median"],
                },
                {
                    "metric_name": "data_time",
                    "value": 0.4,
                    "category": "timing",
                    "phase": "inference",
                    "reduce_method": ["median"],
                },
            ],
        },
    }
    log_data_sample_metrics(_Trainer(), sample)

    keys = set(_Trainer.event_recorder.get_step_scalars().keys())
    assert get_metric_full_name(
        name="data_time", scope="step", category="timing", units="sec",
    ) in keys
    assert get_metric_full_name(
        name="data_time", scope="step", category="timing", prefix="val", units="sec",
    ) in keys
    assert get_metric_full_name(
        name="data_time", scope="step", category="timing", prefix="test", units="sec",
    ) in keys
    assert get_metric_full_name(
        name="data_time", scope="step", category="timing", prefix="inference", units="sec",
    ) in keys


def test_log_data_sample_metrics_value_normalization():
    """Python numbers, scalar tensors, multi-element tensors, and lists all reduce to a float."""
    class _Trainer:
        event_recorder = EventRecorder()

    sample = {
        "metainfo": {
            "metrics": [
                {"metric_name": "py_float", "value": 1.5, "category": "timing",
                 "phase": "training", "reduce_method": ["median"]},
                {"metric_name": "scalar_tensor", "value": torch.tensor(2.0), "category": "timing",
                 "phase": "training", "reduce_method": ["median"]},
                {"metric_name": "vec_tensor", "value": torch.tensor([1.0, 3.0]), "category": "timing",
                 "phase": "training", "reduce_method": ["median"]},
                {"metric_name": "py_list", "value": [2.0, 4.0], "category": "timing",
                 "phase": "training", "reduce_method": ["median"]},
            ],
        },
    }
    log_data_sample_metrics(_Trainer(), sample)
    recorded = _Trainer.event_recorder.get_step_scalars()

    def _val(name):
        full = get_metric_full_name(
            name=name, scope="step", category="timing", units="sec",
        )
        return recorded[full][0][0]

    assert _val("py_float") == pytest.approx(1.5)
    assert _val("scalar_tensor") == pytest.approx(2.0)
    assert _val("vec_tensor") == pytest.approx(2.0)
    assert _val("py_list") == pytest.approx(3.0)


def test_log_data_sample_metrics_skips_invalid_records():
    """Missing required fields or unnormalizable values are skipped, not raised."""
    class _Trainer:
        event_recorder = EventRecorder()

    sample = {
        "metainfo": {
            "metrics": [
                # missing reduce_method
                {"metric_name": "no_reduce", "value": 1.0, "category": "timing"},
                # empty reduce_method
                {"metric_name": "empty_reduce", "value": 1.0, "category": "timing",
                 "reduce_method": []},
                # missing metric_name
                {"value": 1.0, "category": "timing", "reduce_method": ["median"]},
                # unnormalizable value
                {"metric_name": "bad_value", "value": object(), "category": "timing",
                 "reduce_method": ["median"]},
                # valid record passes through
                {"metric_name": "ok", "value": 1.0, "category": "timing",
                 "phase": "training", "reduce_method": ["median"]},
            ],
        },
    }
    log_data_sample_metrics(_Trainer(), sample)
    keys = set(_Trainer.event_recorder.get_step_scalars().keys())
    ok_key = get_metric_full_name(
        name="ok", scope="step", category="timing", units="sec",
    )
    assert ok_key in keys
    assert len(keys) == 1


def test_log_data_sample_metrics_loss_category_records():
    """Records with category='loss' are routed to the loss section without timing units."""
    class _Trainer:
        event_recorder = EventRecorder()

    sample = {
        "metainfo": {
            "metrics": [
                {"metric_name": "aux_loss", "value": 0.42, "category": "loss",
                 "phase": "training", "reduce_method": ["mean"]},
            ],
        },
    }
    log_data_sample_metrics(_Trainer(), sample)
    expected_key = get_metric_full_name(
        name="aux_loss", scope="step", category="loss",
    )
    assert expected_key in _Trainer.event_recorder.get_step_scalars()


def test_log_loss_dict_uses_phase_prefix():
    """log_loss_dict routes losses under the loss category with the resolved prefix."""
    class _Trainer:
        event_recorder = EventRecorder()

    log_loss_dict(_Trainer(), {"step_loss": torch.tensor(1.25), "ce": 0.5}, phase="validation")

    expected_step_loss = get_metric_full_name(
        name="step_loss", scope="step", category="loss", prefix="val",
    )
    expected_ce = get_metric_full_name(
        name="ce", scope="step", category="loss", prefix="val",
    )
    recorded = _Trainer.event_recorder.get_step_scalars()
    assert expected_step_loss in recorded
    assert expected_ce in recorded
    assert recorded[expected_step_loss][0][0] == pytest.approx(1.25)


def test_log_data_sample_metrics_empty_inputs_are_noop():
    """None / empty / missing metrics list should not raise and should record nothing."""
    class _Trainer:
        event_recorder = EventRecorder()

    log_data_sample_metrics(_Trainer(), None)
    log_data_sample_metrics(_Trainer(), {})
    log_data_sample_metrics(_Trainer(), {"metainfo": {}})
    log_data_sample_metrics(_Trainer(), {"metainfo": {"metrics": []}})
    assert _Trainer.event_recorder.get_step_scalars() == {} or all(
        len(v) == 0 for v in _Trainer.event_recorder.get_step_scalars().values()
    )


def test_wandb_writer_preserves_scoped_metric_names():
    """W&B scalar logging should not prepend step/epoch to names that are already scoped."""
    writer = object.__new__(WandBEventWriter)
    writer.run = mock.Mock()

    step_name = get_metric_full_name(name="loss", scope="step")
    writer._write_scalar_impl({step_name: [(1.5, 7, 2)]}, scope="step")
    step_payload = writer.run.log.call_args_list[0].args[0]
    assert step_payload == {
        "iter": 7,
        "epoch": 2,
        f"{step_name}": 1.5,
    }
    assert "step/step/loss" not in step_payload

    epoch_name = get_metric_full_name(name="val_loss", scope="epoch")
    writer._write_scalar_impl({epoch_name: [(2.5, 7, 2)]}, scope="epoch")
    epoch_payload = writer.run.log.call_args_list[1].args[0]
    assert epoch_payload == {
        "epoch": 2,
        "iter": 7,
        f"{epoch_name}": 2.5,
    }
    assert "epoch/epoch/val_loss" not in epoch_payload


def test_wandb_writer_defines_system_metric_category():
    run = mock.Mock()
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("cell_observatory_platform.training.loggers.process_rank", return_value=0), \
             mock.patch("cell_observatory_platform.training.loggers.load_dotenv"), \
             mock.patch("cell_observatory_platform.training.loggers.wandb.login"), \
             mock.patch("cell_observatory_platform.training.loggers.wandb.init", return_value=run):
            WandBEventWriter(EventRecorder(), project="test", dir=tmpdir)

    run.define_metric.assert_any_call(
        get_metric_full_name(name="*", scope="step", category="system"),
        step_metric="iter",
    )
    run.define_metric.assert_any_call(
        get_metric_full_name(name="*", scope="epoch", category="system"),
        step_metric="epoch",
    )


def test_wandb_writer_saves_resolved_config_file():
    writer = object.__new__(WandBEventWriter)
    writer.run = mock.Mock()
    cfg = OmegaConf.create({"paths": {"outdir": "/tmp/run"}, "resolved": "${paths.outdir}"})

    with tempfile.TemporaryDirectory() as tmpdir, \
         mock.patch("cell_observatory_platform.training.loggers.process_rank", return_value=0):
        writer.run.dir = tmpdir
        writer.save_config(cfg)

        config_path = Path(tmpdir) / "resolved_config.yaml"
        assert config_path.exists()
        assert "resolved: /tmp/run" in config_path.read_text()
        writer.run.save.assert_called_once_with(
            str(config_path),
            base_path=tmpdir,
            policy="now",
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
        step_scalars, epoch_scalars = wlist.reduce_scalars()
        median_step_name = f"{step_name}_median"
        mean_step_name = f"{step_name}_mean"
        max_step_name = f"{step_name}_max"
        assert median_step_name in step_scalars
        assert mean_step_name in step_scalars
        assert max_step_name in step_scalars
        assert [row[0] for row in step_scalars[median_step_name]] == [1.0, 2.0, 3.0]
        assert [row[0] for row in step_scalars[mean_step_name]] == [1.0, 2.0, 3.0]
        assert [row[0] for row in step_scalars[max_step_name]] == [1.0, 2.0, 3.0]
        assert pytest.approx(epoch_scalars[median_step_name][0][0]) == 2.0
        assert pytest.approx(epoch_scalars[mean_step_name][0][0]) == 2.0
        assert pytest.approx(epoch_scalars[max_step_name][0][0]) == 3.0


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

    # test clearing scalars method.
    # EventRecorder stores under the structured key produced by
    # get_metric_full_name, not the raw "loss" handle.
    loss_step_key = get_metric_full_name(name="loss", scope="step")
    assert len(recorder.get_step_scalars()[loss_step_key]) == n_steps
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
        # CSV columns use the structured-key form produced by get_metric_full_name
        # plus the reduce-op suffix.
        step_loss_col = get_metric_full_name(name="loss", scope="step") + "_median"
        epoch_val_loss_col = get_metric_full_name(name="val_loss", scope="epoch") + "_median"
        expected_means = {it: sum(float(k + it + 1) for k in range(world)) / world for it in range(n_steps)}
        for _, row in step_df.iterrows():
            assert pytest.approx(row[step_loss_col]) == expected_means[row["iter"]]

        assert epoch_csv.exists(), "epoch CSV missing"
        epoch_df = pd.read_csv(epoch_csv)
        mean_val_loss = sum(float(k + 10) for k in range(world)) / world
        assert len(epoch_df) == 1
        assert pytest.approx(epoch_df.loc[0, epoch_val_loss_col]) == mean_val_loss

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
        cfg=config, test="cell_observatory_platform.tests.training.test_loggers._test_loggers_dist"
    )
    assert metrics.get("success", False), "Distributed event-logging test failed"