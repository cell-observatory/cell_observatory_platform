from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
import ray

from cell_observatory_platform.inference import visualizer as visualizer_module
from cell_observatory_platform.inference.visualizer import VizWorker, VizWorkerBase
from cell_observatory_platform.utils.registry import REGISTRY, Entry


# ---------------------------------------------------------------------------
# Fixtures (aligned with test_buffer_manager.py)
# ---------------------------------------------------------------------------


def _make_buffer_manager(ray_node_id, **kwargs):
    from cell_observatory_platform.data.datasets.buffers import BufferManager

    kw = dict(
        local_rank=0,
        global_rank=0,
        node_id=ray_node_id,
        numa_node=0,
        rank_memory_budget_gb=1.0,
        max_concurrent_calls=10,
        safety_margin=0.0,
    )
    kw.update(kwargs)
    return BufferManager(**kw)


def _kill_safe(handle):
    try:
        ray.kill(handle)
    except Exception:
        pass


def _wait_all_slots_free(buffer_actor, *, timeout_s: float = 5.0, poll_s: float = 0.05) -> None:
    """Block until every slot is back in the free queue (``put_free`` is issued
    asynchronously by the worker's ``free_slot``). Fails if any slot leaked."""
    capacity = ray.get(buffer_actor.get_config.remote())["capacity"]
    deadline = time.monotonic() + timeout_s
    while True:
        free = ray.get(buffer_actor.free_count.remote())
        if free == capacity:
            return
        if time.monotonic() > deadline:
            raise AssertionError(f"{free}/{capacity} slots free after {timeout_s}s: slot(s) leaked")
        time.sleep(poll_s)


def _viz_handler_kwargs():
    """Kwargs for ``inference.utils.save_prediction_plots``.

    The helper is plot-only (volume writing lives in the save path), so the only
    artifact is the MIP PDF; ``save_tensors`` is required for the dict branch.
    """
    return {
        "save_tensors": ["pred"],
        "z_step_pdf": 1,
    }


def _viz_handler_kwargs_failing():
    """Handler kwargs that trigger a KeyError inside save_predictions."""
    return {
        "save_tensors": ["nonexistent_key"],
        "z_step_pdf": 1,
    }


def _expected_pdf_name(tile_relative_path: str = "out/folder/tile001") -> str:
    """PDF name produced by save_predictions for a given viz_identifier.

    tile_relative_path already ends in the tile, so the identifier is that path
    alone -- it replaced the old output_folder + tile_name pair.
    """
    return f"pred_{tile_relative_path.replace('/', '_')}_MIP.pdf"


def _metainfo_for_viz():
    """B=1 metainfo. Columns are per-batch-element sequences (sliced by index)."""
    return {
        "tile_relative_path": ["out/folder/tile001"],
        "tile_name": ["tile001"],
        "batch_size_actual": 1,
    }


def _metainfo_for_viz_batch2():
    return {
        "tile_relative_path": np.array(["out/folder/tile001", "out/other/tile002"]),
        "tile_name": np.array(["tile001", "tile002"]),
        "batch_size_actual": 2,
        "roi_id": np.array([1, 2], dtype=np.int64),
    }


def _meta_b1():
    return {"batch_size_actual": 1, "tile_relative_path": ["o/t"], "tile_name": ["t"]}


def _build_inference_outputs_with_slot(
    bm,
    pool_name: str,
    *,
    fill: float = 1.0,
) -> tuple[dict, dict]:
    """Acquire one slot, fill the buffer view, return (inference_outputs, slot_info)."""
    buffer_actor = bm._buffer_actors[pool_name]
    slot_info = ray.get(buffer_actor.get_free.remote())
    view = bm.slot_info_to_view(slot_info)
    view[...] = fill
    inference_outputs = {
        "pred": slot_info,
        "metainfo": _metainfo_for_viz(),
    }
    return inference_outputs, slot_info


class _FakeBufferManager:
    """One ndarray stands in for the SHM slot; free_slot POISONS it (zeros) so a
    handler that reads after free observes 0 instead of the canary."""

    global_rank = 0

    def __init__(self, arr):
        self._arr, self.freed = arr, []

    def slot_info_to_view(self, slot_info):
        return self._arr

    def free_slot(self, slot_info):
        self.freed.append(slot_info)
        self._arr[...] = 0.0


class _ExplodingBufferManager:
    """slot_info_to_view raises on the 2nd call; records free_slot calls."""

    global_rank = 0

    def __init__(self):
        self.freed, self._views = [], 0

    def slot_info_to_view(self, slot_info):
        self._views += 1
        if self._views == 2:
            raise RuntimeError("stale segment")
        return np.zeros((1, 2, 2, 2, 1), dtype=np.float32)

    def free_slot(self, slot_info):
        self.freed.append(slot_info)


def _slot(i):
    # non-ndarray -> slot path
    return {"actor_name": f"host_pinned_shm_buffer_p{i}_numa_0_rank_0", "slot": i}


@pytest.fixture
def recording_viz_handler(monkeypatch):
    """Register a viz handler for ONE test. REGISTRY has no unregister, so the
    entry is inserted into the private ``REGISTRY._entries`` via monkeypatch
    (restored at teardown) instead of leaking into the session at import time."""
    seen = {"indices": [], "values": []}

    def _handler(record, save_dir, *, global_rank, delay_s=0.0, **kwargs):
        time.sleep(delay_s)
        seen["indices"].append(record.index)
        seen["values"].append(float(np.asarray(record.preds["pred"]).ravel()[0]))

    name = "test_recording_handler"
    monkeypatch.setitem(REGISTRY._entries, ("viz_handler", name),
                        Entry(factory=_handler, role="viz_handler", name=name))
    return name, seen


# ---------------------------------------------------------------------------
# Tier 0: VizWorkerBase in-process (no Ray)
# ---------------------------------------------------------------------------


class TestVizWorkerBase:
    def test_init_metrics_start_empty(self, tmp_path):
        w = VizWorkerBase(buffer_manager=SimpleNamespace(global_rank=0), output_dir=str(tmp_path),
                          handler_configs={"save_predictions": _viz_handler_kwargs()})
        m = w.get_metrics()
        assert m == {"visualize_time_ms": [], "visualize_successful": [], "queue_time_ms": [], "visualize_calls": 0.0}
        assert w.get_handler_names() == ["save_predictions"]

    def test_unknown_handler_rejected_at_init(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown viz.handler"):
            VizWorkerBase(buffer_manager=SimpleNamespace(global_rank=0), output_dir=str(tmp_path),
                          handler_configs={"not_a_real_handler": {}})

    def test_metrics_accumulate_and_reset_on_read(self, tmp_path):
        w = VizWorkerBase(buffer_manager=SimpleNamespace(global_rank=0), output_dir=str(tmp_path),
                          handler_configs={"save_predictions": _viz_handler_kwargs()})
        for _ in range(2):
            w.visualize({"pred": np.zeros((1, 2, 2, 2, 1), np.float32), "metainfo": _metainfo_for_viz()})
        assert (tmp_path / _expected_pdf_name()).is_file()
        m = w.get_metrics()
        assert m["visualize_calls"] == 2.0 and m["visualize_successful"] == [True, True]
        assert len(m["visualize_time_ms"]) == 2
        assert w.get_metrics()["visualize_calls"] == 0.0          # read resets

    def test_slots_freed_only_after_handlers_read_them(self, tmp_path, recording_viz_handler):
        """The slot is released only after every in-flight handler finished reading
        it: the slow handler sees the canary, not the post-free poison."""
        name, seen = recording_viz_handler
        arr = np.full((1, 2, 2, 2, 1), 7.0, dtype=np.float32)
        bm = _FakeBufferManager(arr)
        w = VizWorkerBase(buffer_manager=bm, output_dir=str(tmp_path),
                          handler_configs={name: {"delay_s": 0.3}}, max_workers=2)
        slot = {"actor_name": "fake", "slot": 0}
        w.visualize({"pred": slot, "metainfo": _meta_b1()})
        assert bm.freed == [slot]
        assert seen["values"] == [7.0]                             # canary, not the post-free poison
        assert w.get_metrics()["visualize_successful"] == [True]

    def test_structural_failure_reraises_after_inflight_handlers_finish(self, tmp_path, monkeypatch, recording_viz_handler):
        """Structural failures re-raise (finalize must see them) and the slot is
        still freed only after the submitted handler finished reading it."""
        name, seen = recording_viz_handler
        bm = _FakeBufferManager(np.full((1, 2, 2, 2, 1), 7.0, dtype=np.float32))
        w = VizWorkerBase(buffer_manager=bm, output_dir=str(tmp_path),
                          handler_configs={name: {"delay_s": 0.3}}, max_workers=2)
        slot = {"actor_name": "fake", "slot": 0}
        monkeypatch.setattr(visualizer_module, "as_completed",
                            lambda futures: (_ for _ in ()).throw(RuntimeError("structural failure injected")))
        with pytest.raises(RuntimeError, match="structural failure injected"):
            w.visualize({"pred": slot, "metainfo": _meta_b1()})
        assert bm.freed == [slot] and seen["values"] == [7.0]

    def test_all_slots_freed_when_view_resolution_fails(self, tmp_path):
        """Every slot is claimed for freeing BEFORE any view is resolved, so a
        mid-loop resolution failure leaks neither the failing slot nor later ones."""
        w = VizWorkerBase(buffer_manager=_ExplodingBufferManager(), output_dir=str(tmp_path),
                          handler_configs={"save_predictions": _viz_handler_kwargs()})
        outputs = {"a": _slot(0), "b": _slot(1), "c": _slot(2), "metainfo": {}}
        with pytest.raises(RuntimeError, match="stale segment"):
            w.visualize(inference_outputs=outputs)
        assert [s["slot"] for s in w.buffer_manager.freed] == [0, 1, 2]
        assert w.get_metrics()["visualize_calls"] == 1.0

    @pytest.mark.parametrize("viz_idx, expected", [
        (None, [0, 1, 2]),                      # no subset: every actual record rendered
        ([0, 2, 3, 7], [0, 2]),                 # 3 == padded slot (>= batch_size_actual), 7 out of range
        ([1], [1]),
    ], ids=["no_subset", "padded_and_out_of_range_dropped", "single"])
    def test_viz_sample_idx_renders_only_selected_records(self, tmp_path, recording_viz_handler, viz_idx, expected):
        """``metainfo['viz_sample_idx']`` selects which actual records are rendered;
        indices at or past ``batch_size_actual`` are dropped."""
        name, seen = recording_viz_handler
        w = VizWorkerBase(buffer_manager=SimpleNamespace(global_rank=0), output_dir=str(tmp_path),
                          handler_configs={name: {}}, max_workers=2)
        pred = np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1, 1)      # value == batch index
        meta = {"batch_size_actual": 3, "tile_relative_path": ["o"] * 4, "tile_name": ["t"] * 4}
        if viz_idx is not None:
            meta["viz_sample_idx"] = np.asarray(viz_idx)
        w.visualize({"pred": pred, "metainfo": meta})
        assert sorted(seen["indices"]) == expected                          # thread pool: order-free
        assert sorted(seen["values"]) == [float(i) for i in expected]       # right record, right data
        assert w.get_metrics()["visualize_successful"] == [True] * len(expected)


# ---------------------------------------------------------------------------
# Tier 1: VizWorker Ray actor over real SHM pools
# ---------------------------------------------------------------------------


class TestVizWorkerRay:
    def test_viz_worker_visualize_resolves_slot_and_frees(
        self, ray_ctx, ray_node_id, unique_suffix, tmp_path
    ):
        """The slot-backed pred is rendered and the slot returned to the pool."""
        pool = f"vz_slot_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = viz = None
        try:
            actor, _ = bm.set_buffer(pool_name=pool, batch_size=1, input_shape=(2, 2, 2, 1), dtype="float32",
                                     buffer_type="host_memory", buffer_capacity=2, pin_numa_node=False)
            inference_outputs, _ = _build_inference_outputs_with_slot(bm, pool, fill=3.14)
            assert ray.get(actor.free_count.remote()) == 1
            viz = VizWorker.options(name=f"viz_rs_{unique_suffix}").remote(
                buffer_manager=bm, output_dir=str(tmp_path),
                handler_configs={"save_predictions": _viz_handler_kwargs()})
            ray.get(viz.visualize.remote(inference_outputs))
            assert (tmp_path / _expected_pdf_name()).is_file()
            _wait_all_slots_free(actor)
            m = ray.get(viz.get_metrics.remote())
            assert m["visualize_calls"] == 1.0 and m["visualize_successful"] == [True]
        finally:
            if viz is not None:
                _kill_safe(viz)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)

    def test_viz_worker_visualize_handles_raw_ndarray_input(
        self, ray_ctx, ray_node_id, unique_suffix, tmp_path
    ):
        """A plain ndarray rides the plasma path: no SHM pool is needed."""
        bm = _make_buffer_manager(ray_node_id)
        viz = None
        try:
            arr = np.zeros((1, 2, 2, 2, 1), dtype=np.float32)
            inference_outputs = {
                "pred": arr,
                "metainfo": _metainfo_for_viz(),
            }
            viz = VizWorker.options(name=f"viz_nd_{unique_suffix}").remote(
                buffer_manager=bm,
                output_dir=str(tmp_path),
                handler_configs={"save_predictions": _viz_handler_kwargs()},
            )
            ray.get(viz.visualize.remote(inference_outputs))
            assert (tmp_path / _expected_pdf_name()).is_file()
            assert ray.get(viz.get_metrics.remote())["visualize_successful"] == [True]
        finally:
            if viz is not None:
                _kill_safe(viz)
            bm.shutdown()

    def test_viz_worker_visualize_batched_metainfo_writes_two_pdfs(
        self, ray_ctx, ray_node_id, unique_suffix, tmp_path
    ):
        """B=2: per-batch-element names and sliced (TZYXC) arrays -> two MIP PDFs."""
        bm = _make_buffer_manager(ray_node_id)
        viz = None
        try:
            arr = np.zeros((2, 1, 2, 2, 2, 1), dtype=np.float32)
            inference_outputs = {
                "pred": arr,
                "metainfo": _metainfo_for_viz_batch2(),
            }
            viz = VizWorker.options(name=f"viz_b2_{unique_suffix}").remote(
                buffer_manager=bm,
                output_dir=str(tmp_path),
                handler_configs={"save_predictions": _viz_handler_kwargs()},
            )
            ray.get(viz.visualize.remote(inference_outputs))
            assert (tmp_path / _expected_pdf_name("out/folder/tile001")).is_file()
            assert (tmp_path / _expected_pdf_name("out/other/tile002")).is_file()
        finally:
            if viz is not None:
                _kill_safe(viz)
            bm.shutdown()

    def test_viz_worker_handler_failure_records_failure_metric(
        self, ray_ctx, ray_node_id, unique_suffix, tmp_path
    ):
        """A per-record handler failure is tolerated (recorded as False) and the
        slot is still released."""
        pool = f"vz_fail_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        viz = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool,
                batch_size=1,
                input_shape=(2, 2, 2, 1),
                dtype="float32",
                buffer_type="host_memory",
                buffer_capacity=2,
                pin_numa_node=False,
            )
            inference_outputs, _ = _build_inference_outputs_with_slot(bm, pool)
            buf = bm._buffer_actors[pool]
            assert ray.get(buf.free_count.remote()) == 1
            viz = VizWorker.options(name=f"viz_fl_{unique_suffix}").remote(
                buffer_manager=bm,
                output_dir=str(tmp_path),
                handler_configs={"save_predictions": _viz_handler_kwargs_failing()},
            )
            ray.get(viz.visualize.remote(inference_outputs))
            m = ray.get(viz.get_metrics.remote())
            assert m["visualize_calls"] == 1.0
            assert m["visualize_successful"] == [False]
            _wait_all_slots_free(buf)          # failed handler must still release the slot
        finally:
            if viz is not None:
                _kill_safe(viz)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)

    def test_viz_worker_slot_resolution_failure_reraises(
        self, ray_ctx, ray_node_id, unique_suffix, tmp_path
    ):
        """STRUCTURAL failure (slot resolution) must re-raise so finalize() sees it;
        ``visualize_calls`` still increments (per-handler success list stays empty)."""
        pool = f"vz_sr_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        viz = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool,
                batch_size=1,
                input_shape=(2, 2, 2, 1),
                dtype="float32",
                buffer_type="host_memory",
                buffer_capacity=2,
                pin_numa_node=False,
            )
            inference_outputs = {
                "pred": {"invalid": "slot_info"},
                "metainfo": _metainfo_for_viz(),
            }
            viz = VizWorker.options(name=f"viz_sr_{unique_suffix}").remote(
                buffer_manager=bm,
                output_dir=str(tmp_path),
                handler_configs={"save_predictions": _viz_handler_kwargs()},
            )
            with pytest.raises(ray.exceptions.RayTaskError):
                ray.get(viz.visualize.remote(inference_outputs))
            m = ray.get(viz.get_metrics.remote())
            assert m["visualize_calls"] == 1.0
            assert m["visualize_successful"] == []
        finally:
            if viz is not None:
                _kill_safe(viz)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)
