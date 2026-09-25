"""Single-process unit tests for training/loggers.py and the metric-logging
helpers: structured metric names, data-sample metric routing, loss-dict
logging, the EventRecorder's reduce bookkeeping, the W&B writer's payloads,
and LocalEventWriter's column-aligned CSV appends. CPU-only."""

import math
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from cell_observatory_platform.training.helpers import (
    METRIC_CATEGORY_NAMES,
    get_metric_full_name,
    log_data_sample_metrics,
    log_loss_dict,
    make_timing_metric,
)
from cell_observatory_platform.training.loggers import EventRecorder, EventWriterList, LocalEventWriter, WandBEventWriter


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


def test_log_loss_dict_materializes_tensors_and_registers_reduce_method():
    """Tensor losses are materialized to floats at the logging boundary and the
    caller's reduce_method is registered for the key (not the recorder default)."""
    class _Trainer:
        event_recorder = EventRecorder()

    trainer = _Trainer()
    loss_dict = {"step_loss": torch.tensor(0.25), "aux": 0.5}

    log_loss_dict(trainer, loss_dict, phase="validation", scope="epoch",
                  reduce_method=["mean"])

    key = get_metric_full_name(name="step_loss", scope="epoch",
                               category="loss", prefix="val")
    records = trainer.event_recorder.get_epoch_scalars()[key]
    assert len(records) == 1
    val = records[0][0]
    assert isinstance(val, float) and val == pytest.approx(0.25)
    assert trainer.event_recorder.get_reduce_op(key) == ["mean"]


def test_log_data_sample_metrics_empty_inputs_are_noop():
    """None / empty / missing metrics list should not raise and should record nothing."""
    class _Trainer:
        event_recorder = EventRecorder()

    log_data_sample_metrics(_Trainer(), None)
    log_data_sample_metrics(_Trainer(), {})
    log_data_sample_metrics(_Trainer(), {"metainfo": {}})
    log_data_sample_metrics(_Trainer(), {"metainfo": {"metrics": []}})
    assert all(len(v) == 0 for v in _Trainer.event_recorder.get_step_scalars().values())
    assert all(len(v) == 0 for v in _Trainer.event_recorder.get_epoch_scalars().values())


def test_reduce_epoch_metric_defaults_to_median():
    """With no reduce_method registered, reduce_epoch_metric reduces the epoch
    buffer by median; an explicit reduce_op overrides it; an unbuffered key
    reduces to None."""
    rec = EventRecorder()
    for v in (1.0, 10.0, 2.0):
        rec.put_scalar("x", v, scope="epoch")
    key = get_metric_full_name(name="x", scope="epoch")
    assert rec.get_reduce_op(key) is None
    assert rec.reduce_epoch_metric(key) == pytest.approx(2.0)
    assert rec.reduce_epoch_metric(key, reduce_op="mean") == pytest.approx(13.0 / 3)
    assert rec.reduce_epoch_metric("epoch/missing") is None


def test_put_scalar_conflicting_reregistration_warns_and_keeps_first():
    """Re-registering a key with a different reduce_method warns, keeps the
    first registration, and still records the value."""
    rec = EventRecorder()
    rec.put_scalar("m", 1.0, reduce_method=["median"])
    with pytest.warns(UserWarning, match="already registered"):
        rec.put_scalar("m", 2.0, reduce_method=["mean"])
    key = get_metric_full_name(name="m", scope="step")
    assert rec.get_reduce_op(key) == ["median"]
    assert [v for v, *_ in rec.get_step_scalars()[key]] == [1.0, 2.0]


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


def test_wandb_writer_empty_flush_logs_nothing():
    """An empty flush logs nothing and never starts the batched log sequence;
    a non-rank-0 writer (run=None) is silent even with scalars."""
    writer = object.__new__(WandBEventWriter)
    writer.run = mock.Mock()
    writer._write_scalar_impl({}, scope="step")
    writer._write_scalar_impl({}, scope="epoch")
    writer.run.log.assert_not_called()
    assert not hasattr(writer, "_log_seq")
    writer.run = None
    writer._write_scalar_impl({get_metric_full_name(name="loss", scope="step"): [(1.0, 0, 0)]})
    assert not hasattr(writer, "_log_seq")


def test_wandb_writer_defines_system_metric_category():
    run = mock.Mock()
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("cell_observatory_platform.training.loggers.process_rank", return_value=0), \
             mock.patch("cell_observatory_platform.training.loggers.load_dotenv"), \
             mock.patch("wandb.login"), \
             mock.patch("wandb.init", return_value=run):
            # wandb is imported lazily inside WandBEventWriter.__init__ (P-P3),
            # so the patch targets the real wandb module, not a loggers attr.
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


def test_csv_append_grows_header_and_aligns_columns(tmp_path):
    """Appending a flush with new/reordered columns rewrites the file with the
    union header and keeps every value under its own column."""
    savepath = tmp_path / "epoch_scalars.csv"
    df1 = pd.DataFrame([{"a": 1.0, "b": 2.0}])
    df2 = pd.DataFrame([{"b": 3.0, "c": 4.0}])
    LocalEventWriter._append_csv_aligned(df1, savepath)
    LocalEventWriter._append_csv_aligned(df2, savepath)
    out = pd.read_csv(savepath)
    assert list(out.columns) == ["a", "b", "c"]
    assert out.loc[0, "b"] == 2.0 and out.loc[1, "b"] == 3.0
    assert math.isnan(out.loc[1, "a"]) and math.isnan(out.loc[0, "c"])


def test_csv_append_subset_reindexes_to_existing_header(tmp_path):
    """Appending a flush whose columns are a subset of the header reindexes to
    the existing column order, leaving absent columns empty."""
    savepath = tmp_path / "s.csv"
    LocalEventWriter._append_csv_aligned(pd.DataFrame([{"a": 1, "b": 2}]), savepath)
    LocalEventWriter._append_csv_aligned(pd.DataFrame([{"b": 5}]), savepath)
    out = pd.read_csv(savepath)
    assert out.loc[1, "b"] == 5 and math.isnan(out.loc[1, "a"])
