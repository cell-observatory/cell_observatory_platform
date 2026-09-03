"""Unit tests for TorchNativeTrainer loop mechanics (CPU-only, style-B:
object.__new__ + stub collaborators — no Ray, no distributed, no DeepSpeed
import at collection time)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from cell_observatory_platform.training.hooks import HookBase
from cell_observatory_platform.training.loops import TorchNativeTrainer


def _make_trainer(iteration_mode="epoch", steps_per_epoch=10, max_epochs=3,
                  max_steps=None, _iter=0, _epoch=0, stop_training=False):
    t = object.__new__(TorchNativeTrainer)
    t.iteration_mode = iteration_mode
    t.steps_per_epoch = steps_per_epoch
    t._max_epochs = max_epochs
    t.total_steps = max_steps if max_steps is not None else steps_per_epoch * max_epochs
    t._iter, t._epoch, t.stop_training = _iter, _epoch, stop_training
    return t


class TestStepEpochAccounting:
    def test_epoch_mode_stops_at_max_epochs(self):
        assert _make_trainer("epoch", _epoch=3)._done()
        assert not _make_trainer("epoch", _epoch=2)._done()

    def test_epoch_mode_stops_at_total_steps_too(self):
        # step budget is authoritative in both modes
        assert _make_trainer("epoch", _iter=30)._done()

    def test_step_mode_stops_mid_epoch_at_max_steps(self):
        assert _make_trainer("step", max_steps=25, _iter=25, _epoch=99)._done()
        assert not _make_trainer("step", max_steps=25, _iter=24, _epoch=99)._done()

    def test_step_mode_ignores_epoch_count(self):
        assert not _make_trainer("step", max_steps=100, _epoch=50, _iter=99)._done()

    def test_stop_training_flag_wins(self):
        assert _make_trainer("step", max_steps=100, stop_training=True)._done()


class _RecordingHook(HookBase):
    """Counts dispatches and asserts the per-phase state contract of run_step."""

    def __init__(self):
        super().__init__()
        self.calls = {k: 0 for k in (
            "before_step", "before_backward", "after_backward", "after_step")}

    def before_step(self):
        self.calls["before_step"] += 1

    def before_backward(self, data_sample, loss_dict, outputs=None):
        assert self.trainer.model.lin.weight.grad is None        # forward done, no grads yet
        self.calls["before_backward"] += 1

    def after_backward(self, data_sample, loss_dict, outputs=None):
        assert data_sample["metainfo"] is not None
        assert self.trainer.model.lin.weight.grad is not None    # backward ran
        self.trainer.optimizers.step.assert_not_called()         # optimizer has NOT stepped
        self.calls["after_backward"] += 1

    def after_step(self, data_sample, outputs, loss_dict):
        assert torch.is_tensor(loss_dict["step_loss"]) and not loss_dict["step_loss"].requires_grad
        self.trainer.optimizers.step.assert_called_once()
        self.calls["after_step"] += 1


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 1)

    def forward(self, data_sample):
        loss = self.lin(data_sample["input"]).sum()
        return {"step_loss": loss}, None


def _make_step_trainer():
    t = object.__new__(TorchNativeTrainer)
    model = _TinyModel()
    t.model = model
    t.model_parts = [model]
    # gradient accumulation is rejected in __init__; run_step takes ONE batch
    t.gradient_accumulation_steps = 1
    t.with_grad_accumulation = False
    t.max_norm = 1e9  # effectively no clipping: grad tests compare raw grads
    t.timers = None
    t.metrics_processor = None
    t._iter, t._epoch, t._val_iter = 0, 0, 0
    t._at_accumulation_boundary = True
    t.optimizers = MagicMock()
    t.event_recorder = MagicMock()
    hook = _RecordingHook()
    import weakref

    hook.trainer = weakref.proxy(t)
    t._hooks = [hook]
    return t, model, hook


def _sample(idx=0):
    return {"input": torch.randn(2, 4), "metainfo": {"device_buffer_idx": idx}}


class TestRunStepContract:
    def test_zero_grad_backward_step_ordering_and_counts(self):
        t, model, hook = _make_step_trainer()
        t.run_step(_sample())

        t.optimizers.zero_grad.assert_called_once()
        t.optimizers.step.assert_called_once()
        assert t._iter == 1
        assert t._at_accumulation_boundary is True
        # grads exist on the real module (backward ran)
        assert model.lin.weight.grad is not None
        # hook cadence: exactly one of each per optimizer step -- run_step
        # consumes ONE batch (FreeDeviceBufferHook frees in after_step)
        assert hook.calls == {
            "before_step": 1, "before_backward": 1,
            "after_backward": 1, "after_step": 1,
        }
        # grad_norm recorded without forcing a device sync (tensor, not float)
        (name, value), _ = t.event_recorder.put_scalar.call_args
        assert name == "grad_norm" and torch.is_tensor(value)

    def test_gradients_are_not_scaled(self):
        """No 1/N loss scaling: one batch per optimizer step, so run_step's
        gradients equal a plain backward on the same batch.

        Gradient accumulation (and the 1/N scaling it would need) is rejected
        in __init__.
        """
        torch.manual_seed(0)
        sample = _sample()

        t, model, _ = _make_step_trainer()
        ref = _TinyModel()
        ref.load_state_dict(model.state_dict())

        t.run_step(sample)

        loss, _ = ref(sample)
        loss["step_loss"].backward()
        assert torch.allclose(model.lin.weight.grad, ref.lin.weight.grad, atol=1e-6)

    def test_missing_metainfo_raises_before_forward(self):
        t, _, _ = _make_step_trainer()
        with pytest.raises(RuntimeError, match="metainfo"):
            t.run_step({"input": torch.randn(2, 4), "metainfo": None})

    def test_rejects_a_list_of_microbatches(self):
        """run_step takes ONE dict; the old List[dict] accumulation signature is
        gone, so a list fails at the metainfo lookup (a list has no .get)."""
        t, _, _ = _make_step_trainer()
        with pytest.raises(AttributeError, match="get"):
            t.run_step([_sample(0), _sample(1)])


class TestTrainStateDict:
    """The trainer is the DCP ``train_state`` Stateful: DCP plans a resume from
    the keys ``state_dict()`` emits BEFORE the checkpoint is read, so the
    best-metric lineage keys must be present even while unset."""

    def _bare(self):
        t = object.__new__(TorchNativeTrainer)
        t._iter, t._epoch, t._hooks = 40, 2, []
        return t

    def test_lineage_keys_present_before_best_metric_exists(self):
        sd = self._bare().state_dict()
        assert sd == {"iteration": 40, "epoch": 2, "best_metric": None,
                      "best_metric_epoch": None, "best_metric_iter": None}

    def test_saved_lineage_round_trips_into_a_fresh_trainer(self):
        src = self._bare()
        src.best_metric, src.best_metric_epoch, src.best_metric_iter = 0.25, 1, 30
        dst = self._bare()
        dst.load_state_dict(src.state_dict())
        assert (dst._iter, dst._epoch, dst.best_metric, dst.best_metric_epoch, dst.best_metric_iter) == (40, 2, 0.25, 1, 30)

    def test_none_lineage_does_not_clobber(self):
        dst = self._bare()
        dst.best_metric = 0.5
        dst.load_state_dict({"iteration": 1, "epoch": 0, "best_metric": None})
        assert dst.best_metric == 0.5 and not hasattr(dst, "best_metric_epoch")


def test_val_device_buffer_prefers_the_validation_collator():
    from types import SimpleNamespace
    from cell_observatory_platform.training.loops import _val_device_buffer
    train_buf, val_buf = object(), object()
    assert _val_device_buffer({"val_collate_fn": None}, train_buf) is train_buf
    assert _val_device_buffer({}, train_buf) is train_buf
    assert _val_device_buffer({"val_collate_fn": SimpleNamespace(device_buffer=val_buf)}, train_buf) is val_buf
