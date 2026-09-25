from __future__ import annotations

import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import ray

from cell_observatory_platform.data.io import annotation_exists, read_zarr, save_zarr_data
from cell_observatory_platform.inference import saver as saver_mod
from cell_observatory_platform.inference.saver import (
    SaveWorker,
    SaveWorkerBase,
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


def _read_written(image_path: str, model_name: str, annotation: str) -> np.ndarray:
    # read_zarr returns a TensorStore handle; materialize it.
    return np.asarray(read_zarr(image_path, subpath=f"{model_name}/{annotation}").read().result())


def _create_zarr_semantic_tile_store(tmp_path: Path, *, tile_name: str = "tile.zarr") -> dict:
    """Create a zarr3 root array (TZYXC) so dense annotation writes can run end-to-end."""
    # storage_root + tile_relative_path: the latter already ends in the tile, so
    # the old three-part server_folder / output_folder / tile_name join is gone.
    storage_root = tmp_path / "srv"
    tile_relative_path = f"ds_out/{tile_name}"
    image_path = storage_root / tile_relative_path
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
        "storage_root": str(storage_root),
        "tile_relative_path": tile_relative_path,
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
        "storage_root": col(store["storage_root"]),
        "tile_relative_path": col(store["tile_relative_path"]),
        "tile_name": col(store["tile_name"]),
        "roi_id": col(0),
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


# ---------------------------------------------------------------------------
# Tier 0: pure functions
# ---------------------------------------------------------------------------


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
            shard_spatial_shape=store["shard_spatial_shape"],
            chunk_spatial_shape=store["chunk_spatial_shape"],
        )
        assert annotation_exists(store["image_path"], "m1", "masks", "zarr3")
        assert annotation_exists(store["image_path"], "m1", "labels", "zarr3")
        assert _read_written(store["image_path"], "m1", "labels").shape[-1] == 2   # TNM -> M=2 written

    @staticmethod
    def _save_with_patched_handler(handler_raises: bool):
        """Run save_predictions with a stub save handler; return
        (number of annotations-metadata writes, whether save_predictions raised)."""
        calls = {"metadata": 0}

        def fake_handler(**kw):
            if handler_raises:
                raise RuntimeError("write failed")

        with patch.object(saver_mod.io, "save_annotations_metadata",
                          side_effect=lambda **kw: calls.__setitem__("metadata", calls["metadata"] + 1)), \
             patch.object(saver_mod.REGISTRY, "get",
                          return_value=SimpleNamespace(factory=fake_handler)), \
             patch.object(saver_mod.ray, "logger"):
            try:
                saver_mod.save_predictions(
                    image_path="/tmp/x", model_name="m",
                    preds={"masks": np.zeros((2, 2, 2, 1), dtype=np.uint16)},
                    task="instance_segmentation", save_mode="overwrite",
                    save_tensors_metadata={"masks": {
                        "name": "instance_masks", "annotation_type": "dense",
                        "data_format": "ZYXC", "dtype": "uint16",
                    }},
                )
                raised = False
            except RuntimeError:
                raised = True
        return calls["metadata"], raised

    def test_failed_write_skips_annotations_metadata(self):
        """A failed annotation write must not stamp the annotations metadata that
        advertises the model dir as complete."""
        n_meta, raised = self._save_with_patched_handler(handler_raises=True)
        assert n_meta == 0 and raised

    def test_clean_batch_writes_annotations_metadata(self):
        n_meta, raised = self._save_with_patched_handler(handler_raises=False)
        assert n_meta == 1 and not raised


class TestSaveWorkerBase:
    def test_append_save_mode_rejected(self):
        with pytest.raises(ValueError, match="append"):
            SaveWorkerBase(
                buffer_manager=SimpleNamespace(global_rank=0),
                save_mode="append",
                shard_spatial_shape=(2, 2, 2),
                chunk_spatial_shape=(1, 1, 1),
            )

    def test_overwrite_and_create_accepted(self):
        for mode in ("overwrite", "create"):
            w = SaveWorkerBase(
                buffer_manager=SimpleNamespace(global_rank=0),
                save_mode=mode,
                shard_spatial_shape=(2, 2, 2),
                chunk_spatial_shape=(1, 1, 1),
            )
            assert w.save_mode == mode

    def test_all_slots_freed_when_view_resolution_fails(self):
        """Every slot is claimed for freeing BEFORE any view is resolved, so a
        mid-loop resolution failure leaks neither the failing slot nor later ones."""
        w = SaveWorkerBase(buffer_manager=_ExplodingBufferManager(), save_mode="create",
                           shard_spatial_shape=(2, 2, 2), chunk_spatial_shape=(1, 1, 1))
        outputs = {"a": _slot(0), "b": _slot(1), "c": _slot(2), "metainfo": {}}
        with pytest.raises(RuntimeError, match="stale segment"):
            w.save(inference_outputs=outputs)
        assert [s["slot"] for s in w.buffer_manager.freed] == [0, 1, 2]
        assert w.get_metrics()["save_successful"] == [False]


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
        """The slot's bytes reach disk and the slot is returned to the pool."""
        pool = f"sv_{unique_suffix}"
        bm = _make_buffer_manager(ray_node_id)
        actor = sw = None
        try:
            actor, _ = bm.set_buffer(pool_name=pool, batch_size=1, input_shape=(1, 2, 2, 2, 1), dtype="float32",
                                     buffer_type="host_memory", buffer_capacity=2, pin_numa_node=False)
            inference_outputs, _ = _build_masks_inference_with_slot(bm, pool, zarr_semantic_tile)  # view[...] = 7.0
            assert ray.get(actor.free_count.remote()) == 1                     # slot is checked out

            sw = SaveWorker.options(name=f"sw_rs_{unique_suffix}").remote(
                buffer_manager=bm, save_mode="create",
                shard_spatial_shape=zarr_semantic_tile["shard_spatial_shape"],
                chunk_spatial_shape=zarr_semantic_tile["chunk_spatial_shape"])
            ray.get(sw.save.remote(inference_outputs))

            written = _read_written(zarr_semantic_tile["image_path"], "unit_test_model", "masks")
            assert written.size == 8 and (written == 7).all()                  # slot bytes reached disk
            _wait_all_slots_free(actor)                                         # put_free landed: 2/2
            m = ray.get(sw.get_metrics.remote())
            _assert_save_worker_batch_outcomes(m, expected_successes=1, expected_failures=0)
        finally:
            if sw is not None:
                _kill_safe(sw)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)

    def test_save_worker_save_handles_raw_ndarray(
        self, ray_ctx, ray_node_id, unique_suffix, zarr_semantic_tile
    ):
        """A plain ndarray rides the plasma path: no SHM pool is needed and the
        array is written as-is."""
        bm = _make_buffer_manager(ray_node_id)            # no SHM pool: the ndarray path must not need one
        sw = None
        try:
            masks = np.full((1, 1, 2, 2, 2, 1), 5, dtype=np.float32)
            inference_outputs = {"masks": masks,
                                 "metainfo": _make_save_metainfo_from_store(zarr_semantic_tile, batch_size=1)}
            sw = SaveWorker.options(name=f"sw_nd_{unique_suffix}").remote(
                buffer_manager=bm, save_mode="create",
                shard_spatial_shape=zarr_semantic_tile["shard_spatial_shape"],
                chunk_spatial_shape=zarr_semantic_tile["chunk_spatial_shape"])
            ray.get(sw.save.remote(inference_outputs))
            written = _read_written(zarr_semantic_tile["image_path"], "unit_test_model", "masks")
            assert written.size == 8 and (written == 5).all()
            m = ray.get(sw.get_metrics.remote())
            _assert_save_worker_batch_outcomes(m, expected_successes=1, expected_failures=0)
        finally:
            if sw is not None:
                _kill_safe(sw)
            bm.shutdown()

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
            assert ray.get(buf.free_count.remote()) == 1
            # Path must still exist for SaveWorker, but must not be openable as zarr (real I/O error).
            shutil.rmtree(zpath)
            zpath.write_bytes(b"not a zarr store")
            sw = SaveWorker.options(name=f"sw_fl_{unique_suffix}").remote(
                buffer_manager=bm,
                save_mode="create",
                shard_spatial_shape=zarr_semantic_tile["shard_spatial_shape"],
                chunk_spatial_shape=zarr_semantic_tile["chunk_spatial_shape"],
            )
            with pytest.raises(RuntimeError):
                ray.get(sw.save.remote(inference_outputs))
            m = ray.get(sw.get_metrics.remote())
            _assert_save_worker_batch_outcomes(m, expected_successes=0, expected_failures=1)
            _wait_all_slots_free(buf)          # failed save must still return the slot
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
            assert ray.get(buffer_actor.free_count.remote()) == 1
            bad_meta = _make_save_metainfo_from_store(zarr_semantic_tile, batch_size=1)
            bad_meta["storage_root"] = ["/nonexistent/path"]
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
            with pytest.raises(RuntimeError):
                ray.get(sw.save.remote(inference_outputs))
            m = ray.get(sw.get_metrics.remote())
            _assert_save_worker_batch_outcomes(m, expected_successes=0, expected_failures=1)
            _wait_all_slots_free(buffer_actor)          # failed save must still return the slot
        finally:
            if sw is not None:
                _kill_safe(sw)
            bm.shutdown()
            if actor is not None:
                _kill_safe(actor)
