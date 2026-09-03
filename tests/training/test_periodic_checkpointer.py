"""Wall-clock checkpoint trigger: parse_duration, WallClockTrigger, the
PeriodicCheckpointer.before_step wiring, and the DeepSpeed-path after_epoch /
after_train save metadata and dedup. Pure-Python fakes, no Ray/GPU."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from cell_observatory_platform.training.checkpoint import WallClockTrigger
from cell_observatory_platform.training.helpers import parse_duration
from cell_observatory_platform.training.hooks import PeriodicCheckpointer
from cell_observatory_platform.training.loggers import EventRecorder


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


# ------------------------------------------------------------ parse_duration
@pytest.mark.parametrize(
    "value, expected",
    [(90, 90.0), (12600.0, 12600.0), ("90", 90.0), ("90s", 90.0), ("45m", 2700.0),
     ("3h30m", 12600.0), ("1h2m3s", 3723.0), ("1:30:00", 5400.0), ("05:00", 300.0)],
)
def test_parse_duration_accepts_common_forms(value, expected):
    assert parse_duration(value) == expected


@pytest.mark.parametrize("value", [0, -1, "0s", "", "abc", "1:2:3:4", True])
def test_parse_duration_rejects_garbage(value):
    with pytest.raises((ValueError, TypeError)):
        parse_duration(value)


# ---------------------------------------------------------- WallClockTrigger
def test_trigger_not_due_before_interval():
    clk = FakeClock()
    trig = WallClockTrigger(100, clock=clk)
    clk.t = 99.9
    assert trig.due() is False


def test_trigger_fires_once_per_interval_and_advances():
    clk = FakeClock()
    trig = WallClockTrigger(100, clock=clk)
    clk.t = 100
    assert trig.due() is True
    assert trig.due() is False          # same instant: already advanced
    clk.t = 199
    assert trig.due() is False
    clk.t = 200
    assert trig.due() is True


def test_trigger_long_stall_yields_single_fire():
    clk = FakeClock()
    trig = WallClockTrigger(100, clock=clk)
    clk.t = 350                         # missed deadlines at 100, 200, 300
    assert trig.due() is True
    assert trig.due() is False          # next deadline is 400, not 200
    assert trig.next_deadline == 400


def test_trigger_rejects_nonpositive_interval():
    with pytest.raises(ValueError):
        WallClockTrigger(0)


# -------------------------------------------------------- PeriodicCheckpointer
def _hook(backend, clock=None, interval=None):
    hook = PeriodicCheckpointer(backend=backend, time_interval=interval)
    if clock is not None:                # swap in the fake clock after construction
        hook._clock = WallClockTrigger(parse_duration(interval), clock=clock)
    hook.trainer = SimpleNamespace(_iter=7, _epoch=2, checkpoint_manager=MagicMock())
    return hook


def test_torch_before_step_probes_manager_every_step_without_timer():
    hook = _hook("TORCHTITAN")
    hook.before_step()
    hook.trainer.checkpoint_manager.save.assert_called_once_with(curr_step=7, force=False)


def test_torch_before_step_forces_when_due():
    clk = FakeClock()
    hook = _hook("TORCHTITAN", clock=clk, interval="10s")
    clk.t = 10
    hook.before_step()
    hook.trainer.checkpoint_manager.save.assert_called_once_with(curr_step=7, force=True)


def test_deepspeed_before_step_saves_only_when_due(monkeypatch):
    clk = FakeClock()
    hook = _hook("DEEPSPEED", clock=clk, interval="10s")
    saved = []
    monkeypatch.setattr(hook, "_save", lambda epoch: saved.append(epoch))
    hook.before_step()                  # t=0: not due
    assert saved == []
    clk.t = 10
    hook.before_step()
    assert saved == [2]                 # in-progress epoch


def test_deepspeed_before_step_dedups_same_iter(monkeypatch):
    clk = FakeClock()
    hook = _hook("DEEPSPEED", clock=clk, interval="10s")
    hook._last_saved_iter = 7           # after_epoch just saved this iter
    saved = []
    monkeypatch.setattr(hook, "_save", lambda epoch: saved.append(epoch))
    clk.t = 10
    hook.before_step()
    assert saved == []


def test_deepspeed_without_timer_never_saves_in_before_step(monkeypatch):
    hook = _hook("DEEPSPEED")
    monkeypatch.setattr(hook, "_save", lambda epoch: pytest.fail("must not save"))
    hook.before_step()


# ------------------------------------------- PeriodicCheckpointer (DeepSpeed)
class _RecordingCheckpointManager:
    save_period = 1

    def __init__(self):
        self.saved = []

    def save(self, prefix, metadata=None):
        self.saved.append({"prefix": prefix, "metadata": metadata})


def _ds_hook(epoch, it, best=0.3, latest=0.9, period=1):
    cfg = OmegaConf.create({"experiment_name": "t", "evaluation": {"val_mode": "min"}})
    trainer = SimpleNamespace(
        event_recorder=EventRecorder(), model=torch.nn.Linear(2, 2), cfg=cfg, pristine_cfg=cfg,
        checkpoint_manager=_RecordingCheckpointManager(), best_metric=best, _curr_val_metric=latest,
        _epoch=epoch, _iter=it, start_iter=0, state_dict=lambda: {"iteration": it, "epoch": epoch})
    hook = PeriodicCheckpointer()
    hook.trainer = trainer
    hook.before_train()
    hook.period = period
    return hook, trainer


def test_after_epoch_metadata_persists_historical_best():
    """The saved sidecar carries the historical best metric (not the worse
    latest value) and the count of completed epochs."""
    hook, trainer = _ds_hook(epoch=4, it=100)
    hook.after_epoch()
    (saved,) = trainer.checkpoint_manager.saved
    assert saved["metadata"]["best_loss"] == pytest.approx(0.3)
    assert saved["metadata"]["epoch"] == 5


def test_after_train_skips_save_when_iter_already_saved():
    """after_train right after the final after_epoch (same iteration) does not
    write a duplicate checkpoint."""
    hook, trainer = _ds_hook(epoch=4, it=100)
    hook.after_epoch()
    trainer._epoch = 5
    hook.after_train()
    assert len(trainer.checkpoint_manager.saved) == 1


def test_after_train_saves_unsaved_state_with_completed_epoch_count():
    """State never saved by after_epoch (period not hit) is saved by
    after_train, stamped with the number of completed epochs."""
    hook, trainer = _ds_hook(epoch=5, it=120, period=10)
    hook.after_train()
    (saved,) = trainer.checkpoint_manager.saved
    assert saved["metadata"]["epoch"] == 5
