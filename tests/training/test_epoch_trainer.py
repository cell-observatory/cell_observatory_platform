"""EpochBasedTrainer (DeepSpeed path) loop mechanics: mid-epoch resume skips
the batches already consumed, run_step dispatches after_backward between
backward and the optimizer step, trainer/hook state round-trips through
state_dict, the Ray worker entry point accepts tune's plain-dict configs, and
__init__ rejects gradient accumulation. CPU-only, object.__new__ + stubs."""

import math
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

import cell_observatory_platform.training.loops as loops
from cell_observatory_platform.training.hooks import EarlyStopHook
from cell_observatory_platform.training.loops import (
    BaseTrainer,
    EpochBasedTrainer,
    train_loop_per_worker,
)


def _make_trainer(steps_per_epoch=10, start_iter=0, start_epoch=0):
    t = object.__new__(EpochBasedTrainer)
    t._hooks = []
    t.event_recorder = SimpleNamespace(_epoch=0)
    t.steps_per_epoch = steps_per_epoch
    t.start_epoch, t.start_iter = start_epoch, start_iter
    t._epoch, t._iter = start_epoch, start_iter
    t._epoch_step_offset = start_iter % steps_per_epoch
    t.dataloader_config = {}
    t.preprocessor = lambda data_sample, data_time, idx: data_sample
    t.val_begin, t.val_interval = 10**9, 1  # never validate
    t.ran = []
    t.run_step = lambda idx, data_sample: (t.ran.append(data_sample), setattr(t, "_iter", t._iter + 1))
    return t


def _capture_loader(monkeypatch, calls, steps_per_epoch=10):
    def fake(*, epoch, skip_batches=0, **kwargs):
        calls.append({"epoch": epoch, "skip_batches": skip_batches})
        return iter(range(skip_batches, steps_per_epoch)), None, None

    monkeypatch.setattr(loops, "get_dataloader_ray", fake)


def test_mid_epoch_resume_skips_consumed_batches(monkeypatch):
    """Resuming 3 steps into epoch 2 asks the loader to skip 3 batches and runs
    only the remaining 7."""
    calls = []
    _capture_loader(monkeypatch, calls)
    t = _make_trainer(steps_per_epoch=10, start_iter=23, start_epoch=2)
    t.run_epoch()
    assert calls == [{"epoch": 2, "skip_batches": 3}]
    assert t.ran == list(range(3, 10))
    assert t._iter == 30 and t._epoch == 3


def test_following_epoch_does_not_skip(monkeypatch):
    """The skip applies only to the interrupted epoch; the next epoch starts
    from its first batch."""
    calls = []
    _capture_loader(monkeypatch, calls)
    t = _make_trainer(steps_per_epoch=10, start_iter=23, start_epoch=2)
    t.run_epoch()
    t.run_epoch()
    assert calls[1] == {"epoch": 3, "skip_batches": 0}
    assert t._iter == 40


def test_boundary_resume_skips_nothing(monkeypatch):
    """Resuming from an epoch-boundary checkpoint skips no batches."""
    calls = []
    _capture_loader(monkeypatch, calls)
    t = _make_trainer(steps_per_epoch=10, start_iter=20, start_epoch=2)
    t.run_epoch()
    assert calls == [{"epoch": 2, "skip_batches": 0}]
    assert t._iter == 30


def test_run_step_dispatches_after_backward_between_backward_and_step():
    """run_step fires hooks in the order before_step, before_backward,
    (backward), after_backward, (step), after_step and records the
    accumulation boundary captured before the engine step."""
    t = object.__new__(EpochBasedTrainer)
    order = []

    class _SpyHook:
        def before_step(self):            order.append("before_step")
        def before_backward(self, **kw):  order.append("before_backward")
        def after_backward(self, **kw):   order.append("after_backward")
        def after_step(self, **kw):       order.append("after_step")

    class _Model:
        def __call__(self, ds):           return {"step_loss": torch.tensor(1.0)}, None
        def backward(self, loss):         order.append("backward")
        def step(self):                   order.append("step")
        def is_gradient_accumulation_boundary(self):  return True

    t._hooks = [_SpyHook()]
    t.model = _Model()
    t.event_recorder = SimpleNamespace(_iter=0)
    t._iter = 0
    for name in ("before_step", "before_backward", "after_backward", "after_step"):
        setattr(t, name, BaseTrainer.__dict__[name].__get__(t))

    t.run_step(0, {"metainfo": {"x": 1}})
    assert order == [
        "before_step", "before_backward", "backward",
        "after_backward", "step", "after_step",
    ]
    assert t._at_accumulation_boundary is True


def test_trainer_state_dict_roundtrip_restores_counters_and_hook_state():
    """BaseTrainer.state_dict carries iteration/epoch/best metric plus each
    stateful hook's state, and load_state_dict restores all of it."""
    hook = object.__new__(EarlyStopHook)
    hook.mode = "min"
    hook.wait_count = 3
    hook.best_metric_val = 0.25

    t = object.__new__(BaseTrainer)
    t._iter, t._epoch = 42, 7
    t.best_metric = 0.25
    t._hooks = [hook]
    state = t.state_dict()

    hook2 = object.__new__(EarlyStopHook)
    hook2.mode = "min"
    hook2.wait_count = 0
    hook2.best_metric_val = math.inf
    t2 = object.__new__(BaseTrainer)
    t2._iter = t2._epoch = 0
    t2._hooks = [hook2]
    t2.load_state_dict(state)
    assert (t2._iter, t2._epoch) == (42, 7)
    assert t2.best_metric == 0.25
    assert hook2.wait_count == 3 and hook2.best_metric_val == 0.25


class _SpyTrainer:
    built = []

    def __init__(self, cfg):
        self.cfg = cfg
        self.calls = []
        _SpyTrainer.built.append(self)

    def run(self):
        self.calls.append("run")

    def test(self):
        self.calls.append("test")

    def predict(self):
        self.calls.append("predict")


@pytest.mark.parametrize("job_type,entry_point", [
    ("train", "run"), ("test", "test"), ("predict", "predict"),
])
def test_train_loop_per_worker_accepts_plain_dict_config(monkeypatch, job_type, entry_point):
    """A plain-dict train_loop_config (what Ray Tune ships) is wrapped into a
    DictConfig before the trainer sees it, and job_type selects the trainer
    entry point (train -> run, test -> test, predict -> predict)."""
    _SpyTrainer.built.clear()
    monkeypatch.setattr(loops, "get_class", lambda target: _SpyTrainer)
    train_loop_per_worker({"trainer": "x.Spy", "job_type": job_type, "clusters": {"batch_size": 8}})
    (t,) = _SpyTrainer.built
    assert isinstance(t.cfg, DictConfig)
    assert t.cfg.clusters.batch_size == 8
    assert t.calls == [entry_point]


def test_train_loop_per_worker_rejects_unknown_job_type(monkeypatch):
    """An unknown job_type raises instead of silently doing nothing."""
    _SpyTrainer.built.clear()
    monkeypatch.setattr(loops, "get_class", lambda target: _SpyTrainer)
    with pytest.raises(ValueError, match="Unknown job type"):
        train_loop_per_worker({"trainer": "x.Spy", "job_type": "finetune"})


def _guard_cfg(tmp_path, batch_size, per_gpu, ds_gas):
    return OmegaConf.create({
        "seed": 0,
        "loggers": {
            "event_recorder": {"_target_": "cell_observatory_platform.training.loggers.EventRecorder"},
            "event_writers_list": {"_target_": "cell_observatory_platform.training.loggers.EventWriterList"},
            "event_writers": [{"name": "local", "save_dir": str(tmp_path),
                               "step_scalars_prefix": "s", "epoch_scalars_prefix": "e"}],
        },
        "hooks": {"hooks_list": []},
        "evaluation": {"val_begin": 0, "val_interval": 1},
        "schedulers": {"epochs": 1},
        "clusters": {"batch_size": batch_size, "batch_size_per_gpu": per_gpu},
        "deepspeed": {"gradient_accumulation_steps": ds_gas},
    })


@pytest.fixture()
def _single_worker(monkeypatch):
    monkeypatch.setattr(loops, "get_context", lambda: None)
    monkeypatch.setattr(loops, "get_world_size", lambda: 1)


def test_init_rejects_gradient_accumulation(tmp_path, _single_worker):
    """A global batch larger than batch_size_per_gpu * world_size implies
    gradient accumulation, which __init__ refuses with a hard raise."""
    with pytest.raises(NotImplementedError, match="Gradient accumulation is not supported"):
        EpochBasedTrainer(_guard_cfg(tmp_path, batch_size=4, per_gpu=2, ds_gas=2))


def test_init_rejects_deepspeed_gas_mismatch(tmp_path, _single_worker):
    """The DeepSpeed config's gradient_accumulation_steps must equal the value
    derived from the batch sizes."""
    with pytest.raises(ValueError, match="gradient_accumulation_steps does not match"):
        EpochBasedTrainer(_guard_cfg(tmp_path, batch_size=4, per_gpu=2, ds_gas=1))


def test_init_rejects_global_batch_not_multiple_of_local(tmp_path, _single_worker):
    """The global batch size must be a multiple of the local batch size times
    the data-parallel degree."""
    with pytest.raises(ValueError, match="multiple of local batch size"):
        EpochBasedTrainer(_guard_cfg(tmp_path, batch_size=3, per_gpu=2, ds_gas=1))
