"""Inferencer.predict teardown ordering: after_test is dispatched (through the
real hook dispatch) only on the success path and while the save/viz actors are
still alive, and actor/SHM teardown runs on every exit path. CPU-only; Ray is
patched."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import cell_observatory_platform.training.loops as loops_mod
from cell_observatory_platform.training.hooks import HookBase
from cell_observatory_platform.training.loops import Inferencer


@contextmanager
def _null_ctx(_model):
    yield


class _OrderHook(HookBase):
    def __init__(self, order):
        super().__init__()
        self.order = order

    def before_test(self):
        self.order.append("before_test")

    def after_test(self):
        self.order.append("after_test")


def _make_inferencer(order, n_batches=1, finalize_exc=None, step_exc=None):
    inf = object.__new__(Inferencer)
    inf.model = object()
    inf._hooks = [_OrderHook(order)]
    inf.event_recorder = SimpleNamespace(_iter=0)
    inf._iter = 0
    inf.test_dataloader = [{"i": i} for i in range(n_batches)]
    inf.preprocessor = lambda data_sample, data_time, idx: data_sample

    def _step(idx, data_sample):
        if step_exc is not None:
            raise step_exc
        order.append("step")

    inf.run_inference_step = _step

    def _finalize():
        order.append("finalize")
        if finalize_exc is not None:
            raise finalize_exc

    inf.inferencer_worker = SimpleNamespace(
        finalize=_finalize, close=lambda: order.append("close")
    )
    inf.save_worker = object()
    inf.viz_worker = object()
    inf.buffer_manager = SimpleNamespace(shutdown=lambda: order.append("shutdown"))
    return inf


def _run_predict(inf, order):
    with patch.object(loops_mod, "inference_context", _null_ctx), \
         patch.object(loops_mod.ray, "kill", lambda w: order.append("kill")), \
         patch.object(loops_mod.ray, "logger"):
        inf.predict()


def test_predict_runs_after_test_before_killing_workers():
    """On success: finalize drains saves, after_test runs while the actors are
    alive, and only then are the workers killed and the buffers shut down."""
    order = []
    inf = _make_inferencer(order)
    _run_predict(inf, order)
    assert order == [
        "before_test", "step", "finalize", "after_test",
        "kill", "kill", "close", "shutdown",
    ]


def test_predict_loop_error_skips_after_test_but_tears_down():
    """A failing inference step propagates, skips finalize/after_test, and
    still tears down actors and buffers."""
    order = []
    inf = _make_inferencer(order, step_exc=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        _run_predict(inf, order)
    assert "after_test" not in order and "finalize" not in order
    assert order[-4:] == ["kill", "kill", "close", "shutdown"]


def test_predict_finalize_error_skips_after_test_but_tears_down():
    """A finalize failure (dropped/failed saves) propagates, skips after_test,
    and still tears down actors and buffers."""
    order = []
    inf = _make_inferencer(order, finalize_exc=RuntimeError("INCOMPLETE"))
    with pytest.raises(RuntimeError, match="INCOMPLETE"):
        _run_predict(inf, order)
    assert "after_test" not in order
    assert order[-4:] == ["kill", "kill", "close", "shutdown"]


def test_predict_teardown_skips_absent_workers():
    """Without save/viz workers nothing is killed; buffers are still closed and
    shut down."""
    order = []
    inf = _make_inferencer(order)
    inf.save_worker = None
    inf.viz_worker = None
    _run_predict(inf, order)
    assert "kill" not in order
    assert order[-2:] == ["close", "shutdown"]


# --------------------------------------------------------------------------- #
# Inferencer._teardown (moved from the inference area)
# --------------------------------------------------------------------------- #


def _teardown_trainer():
    t = object.__new__(Inferencer)
    t.inferencer_worker = SimpleNamespace(close=MagicMock())
    t.save_worker = object()
    t.viz_worker = object()
    t.buffer_manager = SimpleNamespace(shutdown=MagicMock())
    return t


def test_teardown_kills_both_workers_closes_and_shuts_down():
    """_teardown kills the save and viz workers, closes the inferencer worker
    and shuts the buffer manager down."""
    t = _teardown_trainer()
    killed = []
    with patch.object(loops_mod.ray, "kill", side_effect=killed.append), \
         patch.object(loops_mod.ray, "logger"):
        t._teardown()
    assert killed == [t.save_worker, t.viz_worker]
    t.inferencer_worker.close.assert_called_once()
    t.buffer_manager.shutdown.assert_called_once()


def test_teardown_shutdown_runs_even_if_kill_fails():
    """Kill failures are logged, not raised, and the buffer manager is still
    shut down."""
    t = _teardown_trainer()
    with patch.object(loops_mod.ray, "kill", side_effect=RuntimeError("no ray")), \
         patch.object(loops_mod.ray, "logger"):
        t._teardown()
    t.buffer_manager.shutdown.assert_called_once()
