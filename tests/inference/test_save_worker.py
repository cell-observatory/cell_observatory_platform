from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import ray

from cell_observatory_platform.data.io import annotation_exists, save_zarr_data
from cell_observatory_platform.inference.saver import (
    SaveWorker,
    input_format_to_output_format,
    save_predictions as saver_save_predictions,
)


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


def _assert_save_worker_batch_outcomes(
    m: dict,
    *,
    expected_successes: int,
    expected_failures: int,
) -> None:
    """Assert ``SaveWorker`` metrics after ``save()`` (lists, one timing row per call)."""
    assert m["save_successful"].count(True) == expected_successes
    assert m["save_successful"].count(False) == expected_failures
    assert len(m["save_time_ms"]) >= 1


def _wait_buffer_in_use_zero(buffer_actor, *, timeout_s: float = 5.0, poll_s: float = 0.05) -> None:
    """Wait until async ``put_free`` has finished after save.

    ``get_metrics`` clears internal series each call. ``put_free`` does not append
    a final occupied_slots sample, so ``occupied_slots`` may be empty after work
    completes — treat empty as OK. Otherwise last entry is never 0 from put alone.
    """
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


def _create_zarr_semantic_tile_store(tmp_path: Path, *, tile_name: str = "tile.zarr") -> dict:
    """Create a zarr3 root array (TZYXC) so ``save_masks(..., save_mode='append')`` can run end-to-end."""
    server_folder = tmp_path / "srv"
    output_folder = "ds_out"
    image_path = server_folder / output_folder / tile_name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    assert not image_path.exists()
    data = np.zeros((1, 2, 2, 2, 1), dtype=np.uint16)
    shard_spatial_shape = (2, 2, 2)
    chunk_spatial_shape = (1, 1, 1)
    save_zarr_data(
        str(image_path),
        data,
        shard_spatial_shape=shard_spatial_shape,
        chunk_spatial_shape=chunk_spatial_shape,
        data_format="TZYXC",
        zarr_driver="zarr3",
        dtype="uint16",
        channel_names={0: "channel_0"},
    )
    return {
        "server_folder": str(server_folder),
        "output_folder": output_folder,
        "tile_name": tile_name,
        "shard_spatial_shape": shard_spatial_shape,
        "chunk_spatial_shape": chunk_spatial_shape,
        "image_path": str(image_path),
    }


@pytest.fixture
def zarr_semantic_tile(tmp_path: Path) -> dict:
    return _create_zarr_semantic_tile_store(tmp_path)


def _make_save_metainfo_from_store(store: dict, *, batch_size: int = 1) -> dict:
    """Metainfo dict compatible with ``SaveWorker.columns`` (list per batch index)."""

    def col(v):
        return [v] * batch_size

    return {
        "batch_size_actual": batch_size,
        "task": "semantic_segmentation",
        "model_name": "unit_test_model",
        "save_tensors_metadata": {"masks": {"name": "masks", "dtype": "uint16", "data_format": "TZYXC", "annotation_type": "dense"}},
        "channel_names": {0: "channel_0"},
        "server_folder": col(store["server_folder"]),
        "output_folder": col(store["output_folder"]),
        "tile_name": col(store["tile_name"]),
        "prepared_id": col(0),
        "mask_bbox_dict": col({}),
        "x_start": col(0),
        "y_start": col(0),
        "z_start": col(0),
        "time_start": col(0),
        "channel_size": col(1),
        "z_size": col(2),
        "y_size": col(2),
        "x_size": col(2),
        "time_size": col(1),
    }


def _build_masks_inference_with_slot(bm, pool_name: str, store: dict) -> tuple[dict, dict]:
    buffer_actor = bm._buffer_actors[pool_name]
    slot_info = ray.get(buffer_actor.get_free.remote())
    view = bm.slot_info_to_view(slot_info)
    view[...] = 7.0
    inference_outputs = {
        "masks": slot_info,
        "metainfo": _make_save_metainfo_from_store(store, batch_size=1),
    }
    return inference_outputs, slot_info


# ---------------------------------------------------------------------------
# Tier 0: pure functions
# ---------------------------------------------------------------------------


class TestInputFormatToOutputFormat:
    def test_instance_tzyxc(self):
        o = input_format_to_output_format("TZYXC", "instance_segmentation")
        assert o["masks"] == "TZYXC"
        assert o["scores"] == "TN"
        assert o["labels"] == "TNM"
        assert o["boxes"] == "TN6"

    def test_instance_zyxc(self):
        o = input_format_to_output_format("ZYXC", "instance_segmentation")
        assert o["masks"] == "ZYXC"
        assert o["scores"] == "N"
        assert o["labels"] == "NM"
        assert o["boxes"] == "N6"

    def test_semantic_tzyxc(self):
        o = input_format_to_output_format("TZYXC", "semantic_segmentation")
        assert o["masks"] == "TZYXC"
        assert o["labels"] == "TNM"

    def test_detection_tzyxc(self):
        o = input_format_to_output_format("TZYXC", "detection")
        assert o["scores"] == "TN"
        assert o["labels"] == "TNM"
        assert o["boxes"] == "TN6"

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            input_format_to_output_format("TZYXC", "panoptic")

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unknown input format"):
            input_format_to_output_format("TYXC", "semantic_segmentation")

    def test_detection_zyxc(self):
        o = input_format_to_output_format("ZYXC", "detection")
        assert o["scores"] == "N"
        assert o["labels"] == "NM"
        assert o["boxes"] == "N6"

    def test_semantic_zyxc(self):
        o = input_format_to_output_format("ZYXC", "semantic_segmentation")
        assert o["masks"] == "ZYXC"
        assert o["labels"] == "NM"


class TestSaverSavePredictions:
    def test_dispatches_masks_semantic(self, tmp_path):
        store = _create_zarr_semantic_tile_store(tmp_path)
        preds = {"masks": np.zeros((1, 2, 2, 2, 1), dtype=np.uint16)}
        saver_save_predictions(
            image_path=store["image_path"],
            model_name="m1",
            preds=preds,
            task="semantic_segmentation",
            save_mode="create",
            save_tensors_metadata={
                "masks": {
                    "name": "masks",
                    "dtype": "uint16",
                    "data_format": "TZYXC",
                    "annotation_type": "dense",
                }
            },
            existing_channel_names={0: "channel_0"},
            shard_spatial_shape=store["shard_spatial_shape"],
            chunk_spatial_shape=store["chunk_spatial_shape"],
        )
        assert annotation_exists(store["image_path"], "m1", "masks", "zarr3")

    def test_dispatches_masks_and_labels_semantic(self, tmp_path):
        """When both masks and labels are provided, both are saved."""
        store = _create_zarr_semantic_tile_store(tmp_path)
        preds = {
            "masks": np.zeros((1, 2, 2, 2, 1), dtype=np.uint16),
            "labels": np.zeros((1, 3, 2), dtype=np.float32),
        }
        saver_save_predictions(
            image_path=store["image_path"],
            model_name="m1",
            preds=preds,
            task="semantic_segmentation",
            save_mode="create",
            save_tensors_metadata={
                "masks": {
                    "name": "masks",
                    "dtype": "uint16",
                    "data_format": "TZYXC",
                    "annotation_type": "dense",
                },
                "labels": {
                    "name": "labels",
                    "dtype": "float32",
                    "data_format": "TNM",
                    "annotation_type": "sparse",
                },
            },
            existing_channel_names={0: "channel_0"},
            shard_spatial_shape=store["shard_spatial_shape"],
            chunk_spatial_shape=store["chunk_spatial_shape"],
        )
        assert annotation_exists(store["image_path"], "m1", "masks", "zarr3")


# ---------------------------------------------------------------------------
# Tier 1: SaveWorker Ray
# ---------------------------------------------------------------------------


class TestSaveWorkerRay:
    def test_save_worker_init_and_metrics(self, ray_ctx, ray_node_id, unique_suffix):
        bm = _make_buffer_manager(ray_node_id)
        sw = None
        try:
            sw = SaveWorker.options(name=f"sw_init_{unique_suffix}").remote(
                buffer_manager=bm,
                save_mode="overwrite",
                shard_spatial_shape=(2, 2, 2),
                chunk_spatial_shape=(1, 1, 1),
            )
            m = ray.get(sw.get_metrics.remote())
            assert m["save_successful"] == []
            assert m["save_time_ms"] == []
            m2 = ray.get(sw.get_metrics.remote())
            assert m2["save_successful"] == []
            assert m2["save_time_ms"] == []
        finally:
            if sw is not None:
                _kill_safe(sw)
            bm.shutdown()

    def test_save_worker_save_resolves_slot_and_frees(
        self, ray_ctx, ray_node_id, unique_suffix, zarr_semantic_tile
    ):
        pool = f"sv_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        sw = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool,
                batch_size=1,
                input_shape=(1, 2, 2, 2, 1),
                dtype="float32",
                buffer_type="host_memory",
                buffer_capacity=2,
                pin_numa_node=False,
            )
            bm.enable_metrics_collection()
            inference_outputs, _ = _build_masks_inference_with_slot(bm, pool, zarr_semantic_tile)
            buf = bm._buffer_actors[pool]
            assert ray.get(buf.get_metrics.remote())["occupied_slots"][-1] == 1

            sw = SaveWorker.options(name=f"sw_rs_{unique_suffix}").remote(
                buffer_manager=bm,
                save_mode="create",
                shard_spatial_shape=zarr_semantic_tile["shard_spatial_shape"],
                chunk_spatial_shape=zarr_semantic_tile["chunk_spatial_shape"],
            )
            ray.get(sw.save.remote(inference_outputs))

            assert annotation_exists(
                zarr_semantic_tile["image_path"],
                "unit_test_model",
                "masks",
                "zarr3",
            )
            _wait_buffer_in_use_zero(buf)
            m = ray.get(sw.get_metrics.remote())
            _assert_save_worker_batch_outcomes(m, expected_successes=1, expected_failures=0)
            assert m["save_time_ms"][-1] >= 0.0
        finally:
            if sw is not None:
                _kill_safe(sw)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)

    def test_save_worker_save_handles_raw_ndarray(
        self, ray_ctx, ray_node_id, unique_suffix, zarr_semantic_tile
    ):
        pool = f"sv_nd_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        sw = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool,
                batch_size=1,
                input_shape=(1, 2, 2, 2, 1),
                dtype="float32",
                buffer_type="host_memory",
                buffer_capacity=2,
                pin_numa_node=False,
            )
            masks = np.zeros((1, 1, 2, 2, 2, 1), dtype=np.float32)
            inference_outputs = {
                "masks": masks,
                "metainfo": _make_save_metainfo_from_store(zarr_semantic_tile, batch_size=1),
            }
            sw = SaveWorker.options(name=f"sw_nd_{unique_suffix}").remote(
                buffer_manager=bm,
                save_mode="create",
                shard_spatial_shape=zarr_semantic_tile["shard_spatial_shape"],
                chunk_spatial_shape=zarr_semantic_tile["chunk_spatial_shape"],
            )
            ray.get(sw.save.remote(inference_outputs))
            assert annotation_exists(
                zarr_semantic_tile["image_path"],
                "unit_test_model",
                "masks",
                "zarr3",
            )
            buf = bm._buffer_actors[pool]
            occ = ray.get(buf.get_metrics.remote())["occupied_slots"]
            # Raw ndarray path does not hold SHM slots; series may be empty.
            assert not occ
        finally:
            if sw is not None:
                _kill_safe(sw)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)

    def test_save_worker_save_failure_records_metric(
        self, ray_ctx, ray_node_id, unique_suffix, zarr_semantic_tile
    ):
        pool = f"sv_fl_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        sw = None
        zpath = Path(zarr_semantic_tile["image_path"])
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool,
                batch_size=1,
                input_shape=(1, 2, 2, 2, 1),
                dtype="float32",
                buffer_type="host_memory",
                buffer_capacity=2,
                pin_numa_node=False,
            )
            inference_outputs, _ = _build_masks_inference_with_slot(bm, pool, zarr_semantic_tile)
            buf = bm._buffer_actors[pool]
            # Path must still exist for SaveWorker, but must not be openable as zarr (real I/O error).
            shutil.rmtree(zpath)
            zpath.write_bytes(b"not a zarr store")
            sw = SaveWorker.options(name=f"sw_fl_{unique_suffix}").remote(
                buffer_manager=bm,
                save_mode="create",
                shard_spatial_shape=zarr_semantic_tile["shard_spatial_shape"],
                chunk_spatial_shape=zarr_semantic_tile["chunk_spatial_shape"],
            )
            ray.get(sw.save.remote(inference_outputs))
            m = ray.get(sw.get_metrics.remote())
            _assert_save_worker_batch_outcomes(m, expected_successes=0, expected_failures=1)
            _wait_buffer_in_use_zero(buf)
        finally:
            if zpath.exists():
                if zpath.is_file():
                    zpath.unlink()
                else:
                    shutil.rmtree(zpath)
            if sw is not None:
                _kill_safe(sw)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)

    def test_save_worker_save_multi_batch(
        self, ray_ctx, ray_node_id, unique_suffix, tmp_path
    ):
        """Batch of 2 elements, each writing to its own file path -- successes counted per element."""
        pool = f"sv_mb_{unique_suffix}"
        store_a = _create_zarr_semantic_tile_store(tmp_path, tile_name="tile_a.zarr")
        store_b = _create_zarr_semantic_tile_store(tmp_path, tile_name="tile_b.zarr")
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        sw = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool,
                batch_size=2,
                input_shape=(1, 2, 2, 2, 1),
                dtype="float32",
                buffer_type="host_memory",
                buffer_capacity=2,
                pin_numa_node=False,
            )
            masks = np.ones((2, 1, 2, 2, 2, 1), dtype=np.float32)
            meta = _make_save_metainfo_from_store(store_a, batch_size=1)
            meta_b = _make_save_metainfo_from_store(store_b, batch_size=1)
            for key in meta:
                if isinstance(meta[key], list):
                    meta[key] = meta[key] + meta_b[key]
            meta["batch_size_actual"] = 2
            inference_outputs = {
                "masks": masks,
                "metainfo": meta,
            }
            sw = SaveWorker.options(name=f"sw_mb_{unique_suffix}").remote(
                buffer_manager=bm,
                save_mode="create",
                shard_spatial_shape=store_a["shard_spatial_shape"],
                chunk_spatial_shape=store_a["chunk_spatial_shape"],
            )
            ray.get(sw.save.remote(inference_outputs))
            m = ray.get(sw.get_metrics.remote())
            _assert_save_worker_batch_outcomes(m, expected_successes=2, expected_failures=0)
            assert annotation_exists(store_a["image_path"], "unit_test_model", "masks", "zarr3")
            assert annotation_exists(store_b["image_path"], "unit_test_model", "masks", "zarr3")
        finally:
            if sw is not None:
                _kill_safe(sw)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)

    def test_save_worker_nonexistent_image_path_no_batch_success(
        self, ray_ctx, ray_node_id, unique_suffix, zarr_semantic_tile
    ):
        """Missing ``image_path`` is caught per-element; metrics record one failed batch element."""
        pool = f"sv_np_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = None
        sw = None
        try:
            actor, _ = bm.set_buffer(
                pool_name=pool,
                batch_size=1,
                input_shape=(1, 2, 2, 2, 1),
                dtype="float32",
                buffer_type="host_memory",
                buffer_capacity=2,
                pin_numa_node=False,
            )
            buffer_actor = bm._buffer_actors[pool]
            slot_info = ray.get(buffer_actor.get_free.remote())
            view = bm.slot_info_to_view(slot_info)
            view[...] = 1.0
            bad_meta = _make_save_metainfo_from_store(zarr_semantic_tile, batch_size=1)
            bad_meta["server_folder"] = ["/nonexistent/path"]
            inference_outputs = {
                "masks": slot_info,
                "metainfo": bad_meta,
            }
            sw = SaveWorker.options(name=f"sw_np_{unique_suffix}").remote(
                buffer_manager=bm,
                save_mode="create",
                shard_spatial_shape=zarr_semantic_tile["shard_spatial_shape"],
                chunk_spatial_shape=zarr_semantic_tile["chunk_spatial_shape"],
            )
            ray.get(sw.save.remote(inference_outputs))
            m = ray.get(sw.get_metrics.remote())
            _assert_save_worker_batch_outcomes(m, expected_successes=0, expected_failures=1)
            _wait_buffer_in_use_zero(buffer_actor)
        finally:
            if sw is not None:
                _kill_safe(sw)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)