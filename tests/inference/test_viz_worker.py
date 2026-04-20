from __future__ import annotations

import time

import numpy as np
import pytest
import ray

from cell_observatory_platform.inference.visualizer import VizWorker


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


def _wait_buffer_in_use_zero(buffer_actor, *, timeout_s: float = 5.0, poll_s: float = 0.05) -> None:
    """Same semantics as ``test_save_worker._wait_buffer_in_use_zero`` — see there."""
    deadline = time.monotonic() + timeout_s
    last_occ: list = []
    while time.monotonic() < deadline:
        time.sleep(poll_s)
        m = ray.get(buffer_actor.get_metrics.remote())
        occ = m["occupied_slots"]
        last_occ = occ
        if not occ:
            return
        if occ[-1] == 0:
            return
    raise AssertionError(
        f"Expected slot release within {timeout_s}s; last occupied_slots snapshot: {last_occ!r}"
    )


def _viz_handler_kwargs():
    """Kwargs for ``inference.utils.save_predictions`` that write a real TIFF."""
    return {
        "save_tensors": ["pred"],
        "save_as_volume": True,
        "save_as_pdf": False,
        "z_step_pdf": 1,
        "filetype": "tiff",
    }


def _viz_handler_kwargs_failing():
    """Handler kwargs that trigger a KeyError inside save_predictions."""
    return {
        "save_tensors": ["nonexistent_key"],
        "save_as_volume": True,
        "save_as_pdf": False,
        "z_step_pdf": 1,
        "filetype": "tiff",
    }


def _expected_tiff_name() -> str:
    """File name produced by save_predictions with _metainfo_for_viz() defaults."""
    return "pred_out_folder_tile001_pred.tiff"


def _metainfo_for_viz():
    return {
        "output_folder": "out/folder",
        "tile_name": "tile001",
    }


def _metainfo_for_viz_batch2():
    return {
        "output_folder": np.array(["out/folder", "out/other"]),
        "tile_name": np.array(["tile001", "tile002"]),
        "batch_size_actual": 2,
        "prepared_id": np.array([1, 2], dtype=np.int64),
    }


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVizWorkerRay:
    def test_viz_worker_init_and_metrics(self, ray_ctx, ray_node_id, unique_suffix, tmp_path):
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        viz = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=f"vz_{unique_suffix}",
                batch_size=1,
                input_shape=(2, 2, 2, 1),
                dtype="float32",
                buffer_type="host_memory",
                buffer_capacity=2,
                pin_numa_node=False,
            )
            viz = VizWorker.options(name=f"viz_init_{unique_suffix}").remote(
                buffer_manager=bm,
                output_dir=str(tmp_path),
                handler_configs={"save_predictions": _viz_handler_kwargs()},
            )
            m = ray.get(viz.get_metrics.remote())
            assert m["visualize_calls"] == 0.0
            assert m["visualize_time_ms"] == []
        finally:
            if viz is not None:
                _kill_safe(viz)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)

    def test_viz_worker_init_unknown_handler_raises(self, ray_ctx, ray_node_id, unique_suffix, tmp_path):
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        viz = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=f"vz_uh_{unique_suffix}",
                batch_size=1,
                input_shape=(2, 2, 2, 1),
                dtype="float32",
                buffer_type="host_memory",
                buffer_capacity=2,
                pin_numa_node=False,
            )
            viz = VizWorker.options(name=f"viz_bad_{unique_suffix}").remote(
                buffer_manager=bm,
                output_dir=str(tmp_path),
                handler_configs={"not_a_real_handler": {}},
            )
            # Actor __init__ failure surfaces as ActorDiedError; task errors use RayTaskError.
            with pytest.raises(
                (ray.exceptions.ActorDiedError, ray.exceptions.RayTaskError),
                match="Unknown viz.handler",
            ):
                ray.get(viz.get_metrics.remote())
        finally:
            if viz is not None:
                _kill_safe(viz)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)

    def test_viz_worker_visualize_resolves_slot_and_frees(
        self, ray_ctx, ray_node_id, unique_suffix, tmp_path
    ):
        pool = f"vz_slot_{unique_suffix}"
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
            bm.enable_metrics_collection()
            inference_outputs, _ = _build_inference_outputs_with_slot(bm, pool, fill=3.14)
            buf = bm._buffer_actors[pool]
            _slot_held = ray.get(buf.get_metrics.remote())
            assert _slot_held["occupied_slots"][-1] == 1

            viz = VizWorker.options(name=f"viz_rs_{unique_suffix}").remote(
                buffer_manager=bm,
                output_dir=str(tmp_path),
                handler_configs={"save_predictions": _viz_handler_kwargs()},
            )
            ray.get(viz.visualize.remote(inference_outputs))

            assert (tmp_path / _expected_tiff_name()).exists()
            _wait_buffer_in_use_zero(buf)

            m = ray.get(viz.get_metrics.remote())
            assert m["visualize_calls"] == 1.0
        finally:
            if viz is not None:
                _kill_safe(viz)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)

    def test_viz_worker_visualize_handles_raw_ndarray_input(
        self, ray_ctx, ray_node_id, unique_suffix, tmp_path
    ):
        pool = f"vz_nd_{unique_suffix}"
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
            assert (tmp_path / _expected_tiff_name()).exists()
            buf = bm._buffer_actors[pool]
            _wait_buffer_in_use_zero(buf)
        finally:
            if viz is not None:
                _kill_safe(viz)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)

    def test_viz_worker_visualize_batched_metainfo_writes_two_tiffs(
        self, ray_ctx, ray_node_id, unique_suffix, tmp_path
    ):
        """B=2: per-batch-element names and sliced (TZYXC) arrays -> two volume files."""
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
            assert (tmp_path / "pred_out_folder_tile001_pred.tiff").is_file()
            assert (tmp_path / "pred_out_other_tile002_pred.tiff").is_file()
        finally:
            if viz is not None:
                _kill_safe(viz)
            bm.shutdown()

    def test_viz_worker_metrics_accumulate(self, ray_ctx, ray_node_id, unique_suffix, tmp_path):
        pool = f"vz_m_{unique_suffix}"
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
                buffer_capacity=4,
                pin_numa_node=False,
            )
            viz = VizWorker.options(name=f"viz_met_{unique_suffix}").remote(
                buffer_manager=bm,
                output_dir=str(tmp_path),
                handler_configs={"save_predictions": _viz_handler_kwargs()},
            )
            for _ in range(2):
                inference_outputs, _ = _build_inference_outputs_with_slot(bm, pool, fill=1.0)
                ray.get(viz.visualize.remote(inference_outputs))

            m = ray.get(viz.get_metrics.remote())
            assert m["visualize_calls"] == 2.0
            m2 = ray.get(viz.get_metrics.remote())
            assert m2["visualize_calls"] == 0.0
        finally:
            if viz is not None:
                _kill_safe(viz)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)

    def test_viz_worker_handler_failure_records_failure_metric(
        self, ray_ctx, ray_node_id, unique_suffix, tmp_path
    ):
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
            viz = VizWorker.options(name=f"viz_fl_{unique_suffix}").remote(
                buffer_manager=bm,
                output_dir=str(tmp_path),
                handler_configs={"save_predictions": _viz_handler_kwargs_failing()},
            )
            ray.get(viz.visualize.remote(inference_outputs))
            m = ray.get(viz.get_metrics.remote())
            assert m["visualize_calls"] == 1.0
            assert m["visualize_successful"] == [False]
            _wait_buffer_in_use_zero(buf)
        finally:
            if viz is not None:
                _kill_safe(viz)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)

    def test_viz_worker_slot_resolution_failure_records_call_only(
        self, ray_ctx, ray_node_id, unique_suffix, tmp_path
    ):
        """Failure in slot resolution is caught by the outer try/except; only
        ``visualize_calls`` increments (per-handler success list stays empty)."""
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
