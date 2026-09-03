"""Per-hook unit tests for training/hooks.py, driven on CPU through a minimal
trainer stand-in (no Ray, no DeepSpeed). The one GPU test is the
TorchMemoryStats/TorchProfiler smoke, which needs the Ray worker group and the
local database (opt-in via --run-localdb)."""

import math
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
import torch
from hydra.utils import get_class
from omegaconf import OmegaConf, open_dict

from cell_observatory_platform.training.helpers import get_metric_full_name, initial_best_metric
from cell_observatory_platform.training.hooks import (
    AnomalyDetector,
    BestMetricSaver,
    EarlyStopHook,
    EMASchedulerHook,
    InferenceMetricsHook,
    IterationTimer,
    LRScheduler,
    NsysProfilerHook,
    PeriodicWriter,
    TorchProfiler,
)
from cell_observatory_platform.training.loggers import (
    EventRecorder,
    EventWriterList,
    LocalEventWriter,
)
from cell_observatory_platform.training.loops import BaseTrainer
from cell_observatory_platform.training.schedulers import WarmupStableDecaySchedule


class _FakeTrainer:
    """Minimal stand-in exposing the attributes hooks touch."""

    def __init__(self, val_mode: str = "min"):
        self.event_recorder = EventRecorder()
        self._epoch = 0
        self._iter = 0
        self.start_iter = 0
        self.stop_training = False
        self.best_metric = initial_best_metric(
            OmegaConf.create({"evaluation": {"val_mode": val_mode}})
        )
        self._curr_val_metric = self.best_metric

    def state_dict(self):
        return {"iteration": self._iter, "epoch": self._epoch}


def _log_val_metric(trainer, name, value):
    rec = trainer.event_recorder
    rec.clear()
    rec._iter, rec._epoch = trainer._iter, trainer._epoch
    rec.put_scalar(name, value, scope="epoch", category="loss", prefix="val")


def _run_early_stop_sequence(hook, trainer, values):
    for ep, v in enumerate(values):
        trainer._epoch = ep
        _log_val_metric(trainer, hook.metric_name, v)
        hook.after_validation()


# --------------------------------------------------------------------------- #
# AnomalyDetector
# --------------------------------------------------------------------------- #


def test_anomaly_detector_counts_nans_and_raises_after_ten():
    """Each epoch enters (and on exit leaves) the autograd anomaly context;
    finite losses are not counted, NaN losses are, and the eleventh NaN raises."""
    hook = AnomalyDetector()
    hook.trainer = SimpleNamespace(_iter=0, _epoch=0)
    nan_loss = {"step_loss": torch.tensor(float("nan"))}

    assert not torch.is_anomaly_enabled()
    hook.before_epoch()
    assert torch.is_anomaly_enabled()

    hook.after_step(None, None, {"step_loss": torch.tensor(0.5)})
    assert hook.loss_nans == 0
    for i in range(10):
        hook.after_step(None, None, nan_loss)
        assert hook.loss_nans == i + 1
    with pytest.raises(Exception, match="Step loss is"):
        hook.after_step(None, None, nan_loss)

    hook.after_epoch()
    assert not torch.is_anomaly_enabled()
    hook.before_epoch()
    assert torch.is_anomaly_enabled()
    hook.after_epoch()
    assert not torch.is_anomaly_enabled()


# --------------------------------------------------------------------------- #
# LRScheduler
# --------------------------------------------------------------------------- #


def test_lr_scheduler_step_mode_records_lr_then_advances_real_schedule():
    """In step mode the hook records the LR the step just trained at and THEN
    advances the schedule, so the recorded series lags the optimizer by one."""
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = WarmupStableDecaySchedule(opt, warmup_steps=10, anneal_steps=5, T_max=30,
                                      start_lr=0.1, ref_lr=1.0, update_type="step")
    trainer = SimpleNamespace(optimizers=opt, schedulers=sched, _iter=0, _epoch=0,
                              event_recorder=EventRecorder())
    hook = LRScheduler(backend="DEEPSPEED")
    hook.trainer = trainer
    hook.before_train()
    assert hook.update_type == "step" and hook._best_param_group_id == 0

    for it in range(2):
        trainer._iter = it
        hook.after_step(None, None, {})

    key = get_metric_full_name(name="lr", scope="step")
    recorded = [v for v, _it, _ep in trainer.event_recorder.get_step_scalars()[key]]
    assert recorded == pytest.approx([0.1, 0.19])
    assert opt.param_groups[0]["lr"] == pytest.approx(0.28)


def _epoch_lr_hook():
    h = object.__new__(LRScheduler)
    h.backend = "DEEPSPEED"
    h.multi_model_opt = False
    h.update_type = "epoch"
    h.schedulers = MagicMock()
    h.optimizers = SimpleNamespace(param_groups=[{"lr": 0.1}])
    h._best_param_group_id = 0
    h.trainer = SimpleNamespace(
        _epoch=3, _iter=0,
        event_recorder=SimpleNamespace(put_scalar=lambda *a, **k: None),
    )
    return h


def test_lr_scheduler_epoch_mode_does_not_step_per_iteration():
    """Epoch-cadence schedules are not advanced from after_step."""
    h = _epoch_lr_hook()
    h.after_step(None, None, {})
    h.schedulers.step.assert_not_called()


def test_lr_scheduler_epoch_mode_steps_once_with_next_epoch():
    """after_epoch advances an epoch-cadence schedule exactly once, to the
    NEXT epoch index, so the next epoch's first step already trains at its LR."""
    h = _epoch_lr_hook()
    h.after_epoch()
    h.schedulers.step.assert_called_once_with(epoch=4)


def test_lr_scheduler_step_mode_gates_on_accumulation_boundary():
    """In step mode the schedule only advances at optimizer boundaries."""
    h = _epoch_lr_hook()
    h.update_type = "step"
    h.trainer._at_accumulation_boundary = False
    h.after_step(None, None, {})
    h.schedulers.step.assert_not_called()
    h.trainer._at_accumulation_boundary = True
    h.after_step(None, None, {})
    h.schedulers.step.assert_called_once()


# --------------------------------------------------------------------------- #
# IterationTimer
# --------------------------------------------------------------------------- #


def _timer_trainer():
    return SimpleNamespace(_iter=0, start_iter=0, _epoch=0, _max_epochs=3,
                           event_recorder=EventRecorder())


def test_iteration_timer_records_step_time_and_step_losses():
    """after_step records the elapsed step time per iteration and logs the
    training loss dict, un-prefixed under the loss category, as floats."""
    hook = IterationTimer()
    hook.trainer = trainer = _timer_trainer()
    hook.before_train()
    for step in range(2):
        # BaseTrainer.before_step keeps event_recorder._iter == trainer._iter
        trainer._iter = trainer.event_recorder._iter = step
        hook.before_step()
        time.sleep(0.02)
        hook.after_step({"metainfo": {}}, None, {"step_loss": torch.tensor(0.5), "aux": 0.25})

    rec = trainer.event_recorder.get_step_scalars()
    step_key = get_metric_full_name(name="step_time", scope="step", category="timing", units="sec")
    assert [it for _v, it, _ep in rec[step_key]] == [0, 1]
    assert all(v >= 0.015 for v, *_ in rec[step_key])
    loss_key = get_metric_full_name(name="step_loss", scope="step", category="loss")
    assert [v for v, *_ in rec[loss_key]] == [0.5, 0.5]
    assert isinstance(rec[loss_key][0][0], float)
    assert [v for v, *_ in rec[get_metric_full_name(name="aux", scope="step", category="loss")]] == [0.25, 0.25]


def test_iteration_timer_epoch_time_and_eta_use_remaining_epochs():
    """after_epoch records the epoch duration and an ETA in hours computed
    from the number of epochs still to run."""
    hook = IterationTimer()
    hook.trainer = trainer = _timer_trainer()
    hook.before_epoch()
    time.sleep(0.02)
    hook.after_epoch()
    ep = trainer.event_recorder.get_epoch_scalars()
    t_key = get_metric_full_name(name="epoch_time", scope="epoch", category="timing", units="sec")
    eta_key = get_metric_full_name(name="eta", scope="epoch", category="timing", units="hrs")
    (epoch_time, _, _), = ep[t_key]
    (eta, _, _), = ep[eta_key]
    assert epoch_time >= 0.015
    assert eta == pytest.approx(epoch_time * 2 / 3600)


# --------------------------------------------------------------------------- #
# PeriodicWriter
# --------------------------------------------------------------------------- #


def test_periodic_writer_after_epoch_writes_csv_and_clears_recorder(tmp_path):
    """after_epoch flushes step and epoch scalars to the CSV logbooks (with
    iter/epoch columns and the reduce-op suffix), clears the recorder, and a
    later flush appends rather than overwrites."""
    rec = EventRecorder()
    writer = LocalEventWriter(rec, save_dir=tmp_path, step_scalars_prefix="step",
                              epoch_scalars_prefix="epoch")
    hook = PeriodicWriter(writers=EventWriterList([writer]))
    hook.trainer = SimpleNamespace(event_recorder=rec)

    rec.put_scalar("loss", 1.23, scope="step")
    rec.put_scalar("val_metric", 0.90, scope="epoch")
    hook.after_epoch()

    step_df = pd.read_csv(writer.step_scalars_savepath)
    epoch_df = pd.read_csv(writer.epoch_scalars_savepath)
    assert step_df.loc[0, get_metric_full_name(name="loss", scope="step") + "_median"] == pytest.approx(1.23)
    assert step_df.loc[0, "iter"] == 0 and step_df.loc[0, "epoch"] == 0
    assert epoch_df.loc[0, get_metric_full_name(name="val_metric", scope="epoch") + "_median"] == pytest.approx(0.90)
    assert all(len(v) == 0 for v in rec.get_step_scalars().values())
    assert all(len(v) == 0 for v in rec.get_epoch_scalars().values())

    rec.put_scalar("loss", 2.0, scope="step")
    hook.after_epoch()
    assert len(pd.read_csv(writer.step_scalars_savepath)) == 2


# --------------------------------------------------------------------------- #
# EMASchedulerHook
# --------------------------------------------------------------------------- #


def test_ema_scheduler_interpolates_beta_and_clamps_past_horizon():
    """beta moves linearly from ema_start to ema_end over max_epochs *
    steps_per_epoch and is clamped at ema_end when an epoch overruns."""
    model = Mock()
    trainer = SimpleNamespace(model=model, _max_epochs=2, steps_per_epoch=5, _iter=0)
    hook = EMASchedulerHook(ema_start=0.9, ema_end=1.0)
    hook.trainer = trainer
    hook.before_train()

    betas = []
    for it in (0, 5, 10, 50):
        trainer._iter = it
        hook.after_step(None, None, {})
        betas.append(model.ema_update.call_args.kwargs["beta"])
    assert betas == pytest.approx([0.9, 0.95, 1.0, 1.0])
    assert model.ema_update.call_count == 4


# --------------------------------------------------------------------------- #
# BestMetricSaver
# --------------------------------------------------------------------------- #


def test_best_metric_saver_max_mode_records_first_then_only_improvements():
    """In max mode the first validation metric becomes the best, a worse epoch
    leaves it untouched, and a better epoch replaces it."""
    trainer = _FakeTrainer(val_mode="max")
    saver = BestMetricSaver(metric_name="score", compare_fn="max")
    saver.trainer = trainer

    _log_val_metric(trainer, "score", 0.42)
    saver.after_validation()
    assert trainer.best_metric == pytest.approx(0.42)

    trainer._epoch = 1
    _log_val_metric(trainer, "score", 0.10)
    saver.after_validation()
    assert trainer.best_metric == pytest.approx(0.42)

    trainer._epoch = 2
    _log_val_metric(trainer, "score", 0.77)
    saver.after_validation()
    assert trainer.best_metric == pytest.approx(0.77)


@pytest.mark.parametrize("phase", ["after_validation", "after_epoch", "after_test"])
def test_best_metric_saver_skips_when_no_validation_records(phase):
    """When the epoch buffer holds no record for the metric, every phase leaves
    the best metric, the latest metric and the best-epoch bookkeeping untouched."""
    trainer = _FakeTrainer(val_mode="min")
    saver = BestMetricSaver(metric_name="map", eval_after_validation=(phase != "after_epoch"))
    saver.trainer = trainer
    trainer._epoch = saver.period - 1

    getattr(saver, phase)()

    assert trainer.best_metric == math.inf
    assert trainer._curr_val_metric == math.inf
    assert not hasattr(trainer, "best_metric_epoch")
    assert not hasattr(trainer, "best_metric_iter")


# --------------------------------------------------------------------------- #
# EarlyStopHook
# --------------------------------------------------------------------------- #


def test_early_stop_worsening_metric_triggers_stop():
    """A steadily worsening min-mode metric counts as no improvement each epoch
    and stops training once patience is exhausted."""
    trainer = _FakeTrainer()
    hook = EarlyStopHook(patience=2, stopping_threshold=0.01, mode="min", metric_name="loss")
    hook.trainer = trainer

    _run_early_stop_sequence(hook, trainer, [1.0, 2.0, 3.0])
    assert hook.wait_count == 2
    assert trainer.stop_training is True


def test_early_stop_improvement_resets_wait_count():
    """An improvement after a non-improving epoch records the new best and
    resets the wait count."""
    trainer = _FakeTrainer()
    hook = EarlyStopHook(patience=3, stopping_threshold=0.0, mode="min", metric_name="loss")
    hook.trainer = trainer
    _run_early_stop_sequence(hook, trainer, [0.5, 0.6])
    assert hook.wait_count == 1
    _run_early_stop_sequence(hook, trainer, [0.4])
    assert hook.best_metric_val == pytest.approx(0.4)
    assert hook.wait_count == 0 and trainer.stop_training is False


def test_early_stop_state_dict_roundtrip():
    """best_metric_val and wait_count survive a state_dict round trip, and a
    resumed hook continues counting from the restored wait count."""
    hook = EarlyStopHook(patience=3, stopping_threshold=0.0, mode="min", metric_name="loss")
    hook.best_metric_val, hook.wait_count = 0.4, 2
    fresh = EarlyStopHook(patience=3, stopping_threshold=0.0, mode="min", metric_name="loss")
    fresh.load_state_dict(hook.state_dict())
    assert fresh.best_metric_val == pytest.approx(0.4) and fresh.wait_count == 2
    trainer = _FakeTrainer()
    fresh.trainer = trainer
    _run_early_stop_sequence(fresh, trainer, [0.41])
    assert trainer.stop_training is True


def test_early_stop_max_mode_counts_declines():
    """In max mode, epochs below the best count toward patience and stop
    training once exhausted; the best stays at the peak value."""
    trainer = _FakeTrainer(val_mode="max")
    hook = EarlyStopHook(patience=2, stopping_threshold=0.0, mode="max", metric_name="miou")
    hook.trainer = trainer

    _run_early_stop_sequence(hook, trainer, [0.5, 0.6, 0.55, 0.52])
    assert hook.best_metric_val == pytest.approx(0.6)
    assert hook.wait_count == 2
    assert trainer.stop_training is True


def _nan_early_stop_hook(patience):
    h = EarlyStopHook(patience=patience, stopping_threshold=0.0, mode="max", metric_name="map")
    h.trainer = SimpleNamespace(
        stop_training=False,
        event_recorder=SimpleNamespace(reduce_epoch_metric=lambda key: float("nan")),
    )
    return h


def test_early_stop_nan_counts_as_no_improvement():
    """A NaN validation metric increments the wait count instead of raising."""
    h = _nan_early_stop_hook(patience=5)
    h.after_validation()
    assert h.wait_count == 1
    assert h.trainer.stop_training is False


def test_early_stop_patience_of_nans_stops_training():
    """patience consecutive NaN metrics stop training."""
    h = _nan_early_stop_hook(patience=2)
    h.after_validation()
    h.after_validation()
    assert h.wait_count == 2
    assert h.trainer.stop_training is True


# --------------------------------------------------------------------------- #
# Profilers (resume-aware windows)
# --------------------------------------------------------------------------- #


def test_nsys_profiler_window_is_relative_to_start_iter():
    """start_iter/end_iter are offsets from the resumed trainer's start_iter,
    so the window fires on a resumed run too."""
    h = object.__new__(NsysProfilerHook)
    h.closed = False
    h.start_iter, h.end_iter = 40, 90
    h.shutdown_after_profile = False
    h.trainer = SimpleNamespace(start_iter=10_000, _iter=10_040)
    with patch("torch.cuda.cudart") as cudart:
        h.before_step()
        cudart().cudaProfilerStart.assert_called_once()
    h.trainer._iter = 10_090
    with patch("torch.cuda.cudart") as cudart:
        h.after_step()
        cudart().cudaProfilerStop.assert_called_once()
    assert h.closed


def test_torch_profiler_stops_after_profile_times_on_resumed_run():
    """The profiler stops after profile_times steps counted from the resumed
    start_iter, flushing traces exactly once."""
    h = object.__new__(TorchProfiler)
    h._closed = False
    h.profile_times = 50
    h.shutdown_after_profile = False
    h._profiler = MagicMock()
    h._flush_traces = MagicMock()
    h.trainer = SimpleNamespace(start_iter=10_000, _iter=10_049)
    h.after_step()
    h._flush_traces.assert_called_once()
    assert h._closed


# --------------------------------------------------------------------------- #
# Hook ordering
# --------------------------------------------------------------------------- #


def test_inference_metrics_hook_runs_before_periodic_writer():
    """register_hooks orders by priority, so InferenceMetricsHook runs before
    PeriodicWriter regardless of config order (its final after_test record
    must land before the writers flush and close)."""
    t = object.__new__(BaseTrainer)
    t._hooks = []
    writer_hook = PeriodicWriter(writers=Mock())
    metrics_hook = InferenceMetricsHook()
    t.register_hooks([writer_hook, metrics_hook])
    assert t._hooks == [metrics_hook, writer_hook]


# --------------------------------------------------------------------------- #
# GPU smoke: TorchMemoryStats + TorchProfiler on a real trainer
# --------------------------------------------------------------------------- #


def _gpu_hooks_smoke_dist(cfg):
    import torch
    from ray.train import report
    from cell_observatory_platform.training.hooks import TorchMemoryStats, TorchProfiler
    from cell_observatory_platform.training.helpers import get_metric_full_name
    from cell_observatory_platform.utils.context import barrier

    trainer = get_class(cfg.trainer)(cfg)
    mem = next(h for h in trainer._hooks if isinstance(h, TorchMemoryStats))
    prof = next(h for h in trainer._hooks if isinstance(h, TorchProfiler))

    mem._step_period, mem._epoch_period = 1, 1
    trainer._iter, trainer._epoch = 0, 0
    _ = torch.empty((1024, 1024), device="cuda")
    mem.after_step(None, None, {})
    mem.after_epoch()
    want = {get_metric_full_name(name=k, scope="step", category="system", units="GB")
            for k in ("allocated_mem", "reserved_mem", "max_allocated_mem", "max_reserved_mem")}
    mem_keys_ok = want <= set(trainer.event_recorder.get_step_scalars())
    barrier()
    log = mem._logdir / "0.log"
    epoch_log_bytes = log.stat().st_size if log.exists() else 0

    prof._wait, prof._warmup, prof._active, prof._repeat = 0, 0, 2, 1
    prof.profile_times, prof.shutdown_after_profile = 2, False
    prof.before_train()
    for it in range(2):
        trainer._iter = it
        _ = torch.randn(256, 256, device="cuda") @ torch.randn(256, 256, device="cuda")
        prof.after_step()
    trace_files = sum(p.is_file() for p in (Path(prof._output_dir) / "log").rglob("*"))
    barrier()
    report(metrics={"mem_keys_ok": mem_keys_ok, "epoch_log_bytes": epoch_log_bytes,
                    "profiler_closed": prof._closed, "trace_files": trace_files})


@pytest.mark.localdb
@pytest.mark.cuda
def test_gpu_hooks_smoke(config):
    """On a real trainer, TorchMemoryStats records the four CUDA memory gauges
    per step and writes the per-epoch memory summary, and TorchProfiler closes
    after profile_times steps leaving trace files behind."""
    from cell_observatory_platform.tests.conftest import distributed_test

    with open_dict(config):
        config.experiment_name = "test_gpu_hooks_smoke"
        config.paths.resume_checkpointdir = None
    m = distributed_test(cfg=config, test="cell_observatory_platform.tests.training.test_hooks._gpu_hooks_smoke_dist")
    assert m["mem_keys_ok"] is True
    assert m["epoch_log_bytes"] > 0
    assert m["profiler_closed"] is True and m["trace_files"] > 0


class TestFreeDeviceBufferHook:
    """Validation batches may be collated by a collator of their own (step-cadence
    validation); the slot must go back to THAT buffer, never the train one."""

    def _hook(self, trainer):
        from cell_observatory_platform.training.hooks import FreeDeviceBufferHook
        h = FreeDeviceBufferHook()
        h.trainer = trainer
        h.before_train()
        return h

    def test_val_step_frees_into_the_validation_buffer_when_present(self):
        train_buf, val_buf = Mock(), Mock()
        h = self._hook(SimpleNamespace(device_buffer=train_buf, val_device_buffer=val_buf, with_grad_accumulation=False))
        h.after_val_step({"metainfo": {"device_buffer_idx": 3}}, None, None)
        val_buf.put_free.assert_called_once_with(3)
        train_buf.put_free.assert_not_called()
        h.after_step(data_sample={"metainfo": {"device_buffer_idx": 1}}, outputs=None, loss_dict=None)
        train_buf.put_free.assert_called_once_with(1)

    def test_val_step_falls_back_to_the_shared_buffer(self):
        train_buf = Mock()
        h = self._hook(SimpleNamespace(device_buffer=train_buf, with_grad_accumulation=False))
        h.after_val_step({"metainfo": {"device_buffer_idx": 0}}, None, None)
        train_buf.put_free.assert_called_once_with(0)
