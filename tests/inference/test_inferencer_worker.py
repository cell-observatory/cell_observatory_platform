"""Unit tests for InferencerWorker.

Tests cover:
- Initialization with each supported task (instance_segmentation, detection)
- Full predict() flow: CUDA tensor in -> D2H transfer -> save/viz worker dispatch
- Output buffer slot lifecycle for ``buffer_tensors`` only (acquire, fill, free); small tensors use inline numpy
- Correct routing through _predict for each task branch
- CPU-only unit tests of the worker's pure helpers (policy gate, D2H guards,
  staging cache, reaping, finalize)

Requirements:
- Ray cluster: session-scoped (tests/inference/conftest.py), no GPU actors
- CUDA device available for the end-to-end classes (skipped otherwise)

Mock models return fixed tensors so we can verify data flows through
the worker correctly without loading real model weights.
"""
from __future__ import annotations

import time
from collections import namedtuple
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import ray
import torch
import torch.nn as nn

import cell_observatory_platform.inference.inferencer as inf_mod
from cell_observatory_platform.inference.inferencer import InferencerWorker

_CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")


# ---------------------------------------------------------------------------
# Mock models
# ---------------------------------------------------------------------------


class _MockInstanceSegModel(nn.Module):
    """Returns fixed masks / boxes / labels like ``MaskDINO.inference_step()``.

    Mirrors the real contract (models/meta_arch/maskdino.py::inference_step):
    ``boxes`` (B, topk, 6), ``labels`` (B, topk) and ``masks`` as a channels-last
    ``(B, Z, Y, X, 1)`` uint16 instance label map.
    """

    def __init__(self, spatial_shape: Tuple[int, ...], topk: int, device: torch.device):
        super().__init__()
        self._spatial = spatial_shape
        self._topk = topk
        self._device = device
        self.output_metadata = {
            "tensor_info": {
                "masks": {"shape": (*spatial_shape, 1), "dtype": "uint16"},
                "boxes": {"shape": (topk, 6), "dtype": "float32"},
                "labels": {"shape": (topk,), "dtype": "float32"},
            },
        }

    def get_output_metadata(self) -> dict:
        return self.output_metadata

    def inference_step(self, data_sample: dict) -> Dict[str, torch.Tensor]:
        B = data_sample["data_tensor"].shape[0]
        masks = torch.ones((B, *self._spatial, 1), dtype=torch.uint16, device=self._device)
        boxes = torch.full((B, self._topk, 6), 0.5, dtype=torch.float32, device=self._device)
        labels = torch.full((B, self._topk), 0.9, dtype=torch.float32, device=self._device)
        return {"masks": masks, "boxes": boxes, "labels": labels}

    def validate_outputs(self, preds: Dict[str, torch.Tensor]) -> None:
        tensor_info = self.output_metadata["tensor_info"]
        for key, info in tensor_info.items():
            assert key in preds, f"Missing key {key}"
            expected = tuple(info["shape"])
            actual = tuple(preds[key].shape[1:])
            assert actual == expected, f"{key}: expected {expected}, got {actual}"


class _MockDetectionModel(nn.Module):
    """Returns fixed boxes / labels like ``PlainDETR.inference_step()``."""

    def __init__(self, topk: int, device: torch.device):
        super().__init__()
        self._topk = topk
        self._device = device
        self.output_metadata = {
            "tensor_info": {
                "boxes": {"shape": (topk, 6), "dtype": "float32"},
                "labels": {"shape": (topk,), "dtype": "float32"},
            },
        }

    def get_output_metadata(self) -> dict:
        return self.output_metadata

    def inference_step(self, data_sample: dict) -> Dict[str, torch.Tensor]:
        B = data_sample["data_tensor"].shape[0]
        boxes = torch.full((B, self._topk, 6), 0.25, dtype=torch.float32, device=self._device)
        labels = torch.full((B, self._topk), 0.8, dtype=torch.float32, device=self._device)
        return {"boxes": boxes, "labels": labels}

    def validate_outputs(self, preds: Dict[str, torch.Tensor]) -> None:
        tensor_info = self.output_metadata["tensor_info"]
        for key, info in tensor_info.items():
            assert key in preds, f"Missing key {key}"
            expected = tuple(info["shape"])
            actual = tuple(preds[key].shape[1:])
            assert actual == expected, f"{key}: expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# Stub save / viz Ray workers
# ---------------------------------------------------------------------------


@ray.remote(namespace="saver", num_cpus=0)
class _StubSaveWorker:
    """Consumes inference outputs without performing I/O."""

    def __init__(self, buffer_manager):
        self._bm = buffer_manager
        self._received: List[dict] = []
        # Mirror ``SaveWorker``: list batches cleared in ``get_metrics`` (no gap vs reset).
        self._metrics: Dict[str, list] = {
            "save_time_ms": [],
            "save_successful": [],
            "queue_time_ms": [],
        }

    def save(self, inference_outputs: dict, queue_t0: Optional[float] = None) -> None:
        if queue_t0 is not None:
            self._metrics["queue_time_ms"].append((time.perf_counter() - queue_t0) * 1000)
        t0 = time.perf_counter()
        ok = False
        try:
            for key, val in inference_outputs.items():
                if key == "metainfo":
                    continue
                if isinstance(val, dict) and "actor_name" in val:
                    view = self._bm.slot_info_to_view(val)
                    self._received.append({key: np.array(view)})
                    self._bm.free_slot(val)
                elif isinstance(val, np.ndarray):
                    self._received.append({key: val.copy()})
            ok = True
        except Exception:
            ok = False
        finally:
            self._metrics["save_time_ms"].append((time.perf_counter() - t0) * 1000)
            self._metrics["save_successful"].append(ok)

    def get_received(self) -> List[dict]:
        return self._received

    def get_metrics(self) -> dict:
        out = {
            "save_time_ms": self._metrics["save_time_ms"].copy(),
            "save_successful": self._metrics["save_successful"].copy(),
            "queue_time_ms": self._metrics["queue_time_ms"].copy(),
        }
        self._metrics = {
            "save_time_ms": [],
            "save_successful": [],
            "queue_time_ms": [],
        }
        return out


@ray.remote(namespace="visualizer", num_cpus=0)
class _StubVizWorker:
    """Consumes inference outputs without producing visualizations."""

    def __init__(self, buffer_manager):
        self._bm = buffer_manager
        self._received: List[dict] = []
        self._metrics: Dict[str, float | list] = {
            "visualize_time_ms": [],
            "visualize_successful": [],
            "queue_time_ms": [],
            "visualize_calls": 0.0,
        }

    def visualize(self, inference_outputs: dict, queue_t0: Optional[float] = None) -> None:
        if queue_t0 is not None:
            self._metrics["queue_time_ms"].append((time.perf_counter() - queue_t0) * 1000)
        self._metrics["visualize_calls"] += 1.0
        t0 = time.perf_counter()
        ok = False
        try:
            for key, val in inference_outputs.items():
                if key == "metainfo":
                    continue
                if isinstance(val, dict) and "actor_name" in val:
                    view = self._bm.slot_info_to_view(val)
                    self._received.append({key: np.array(view)})
                    self._bm.free_slot(val)
                elif isinstance(val, np.ndarray):
                    self._received.append({key: val.copy()})
            ok = True
        except Exception:
            ok = False
        finally:
            self._metrics["visualize_time_ms"].append((time.perf_counter() - t0) * 1000)
            self._metrics["visualize_successful"].append(ok)

    def get_received(self) -> List[dict]:
        return self._received

    def get_metrics(self) -> dict:
        out = {
            "visualize_time_ms": self._metrics["visualize_time_ms"].copy(),
            "visualize_successful": self._metrics["visualize_successful"].copy(),
            "queue_time_ms": self._metrics["queue_time_ms"].copy(),
            "visualize_calls": self._metrics["visualize_calls"],
        }
        self._metrics = {
            "visualize_time_ms": [],
            "visualize_successful": [],
            "queue_time_ms": [],
            "visualize_calls": 0.0,
        }
        return out

# ---------------------------------------------------------------------------
# Fixtures
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


def _bare_worker() -> InferencerWorker:
    """An InferencerWorker with NO __init__ run (no CUDA/cupy stream, no Ray):
    for the pure helper methods that read only the attributes a test sets."""
    return object.__new__(InferencerWorker)


# ---------------------------------------------------------------------------
# InferencerWorker construction helpers
# ---------------------------------------------------------------------------

# Spatial dimensions used for all tests; small to keep memory low.
_SPATIAL = (8, 8, 8)
_INPUT_SHAPE = (*_SPATIAL, 1)  # ZYXC
_PATCH_SHAPE = (4, 4)          # (axial, lateral) for ZYXC
_INPUT_FORMAT = "ZYXC"
_TOPK = 5
_BATCH_SIZE = 1
# Must match default channel_names in ``_build_inferencer_worker`` for ``channel_mapping`` assert.
_CHANNEL_MAPPING_META = {"0": "test_channel_0"}


def _make_outputs_metadata_instance_seg():
    return {
        "save_tensors": {
            "masks": {
                "dtype": "uint16",
                "annotation_type": "dense",
                "data_format": "ZYXC",
            },
            "boxes": {
                "dtype": "float32",
                "annotation_type": "sparse",
                "data_format": "N6",
            },
            "labels": {
                "dtype": "float32",
                "annotation_type": "sparse",
                "data_format": "N",
            },
        },
        "visualize_tensors": ["masks", "boxes", "labels"],
        # Only large spatial arrays use pinned SHM pools; boxes/labels go inline as numpy.
        "buffer_tensors": ["masks"],
        "tensor_info": {
            # masks are channels-last ZYXC, matching MaskDINO.inference_step().
            "masks": {"shape": (*_SPATIAL, 1), "dtype": "uint16"},
            "boxes": {"shape": (_TOPK, 6), "dtype": "float32"},
            "labels": {"shape": (_TOPK,), "dtype": "float32"},
        },
    }


def _make_outputs_metadata_detection():
    return {
        "save_tensors": {
            "boxes": {
                "dtype": "float32",
                "annotation_type": "sparse",
                "data_format": "N6",
            },
            "labels": {
                "dtype": "float32",
                "annotation_type": "sparse",
                "data_format": "N",
            },
        },
        "visualize_tensors": ["boxes", "labels"],
        "buffer_tensors": [],
        "tensor_info": {
            "boxes": {"shape": (_TOPK, 6), "dtype": "float32"},
            "labels": {"shape": (_TOPK,), "dtype": "float32"},
        },
    }


def _register_output_buffers(
    bm,
    outputs_metadata: dict,
    buffer_capacity: int = 4,
) -> List:
    """Create host SHM pools for ``buffer_tensors`` only, mirroring ``init_output_memory_pools``."""
    actors = []
    for tensor_name in outputs_metadata.get("buffer_tensors", []):
        info = outputs_metadata["tensor_info"][tensor_name]
        shape = tuple(info["shape"])
        dtype = info["dtype"]
        if tensor_name in outputs_metadata["save_tensors"]:
            actor, _ = bm.set_buffer(
                pool_name=f"{tensor_name}_save",
                batch_size=_BATCH_SIZE,
                input_shape=shape,
                dtype=dtype,
                buffer_type="host_memory",
                buffer_capacity=buffer_capacity,
                pin_numa_node=False,
            )
            actors.append(actor)
        if tensor_name in outputs_metadata["visualize_tensors"]:
            actor, _ = bm.set_buffer(
                pool_name=f"{tensor_name}_viz",
                batch_size=_BATCH_SIZE,
                input_shape=shape,
                dtype=dtype,
                buffer_type="host_memory",
                buffer_capacity=buffer_capacity,
                pin_numa_node=False,
            )
            actors.append(actor)
    return actors


def _build_inferencer_worker(
    *,
    task: str,
    model: nn.Module,
    buffer_manager,
    save_worker,
    viz_worker,
    outputs_metadata: dict,
    save_outputs: bool = True,
    vizualize_outputs: bool = True,
    block_on_save: bool = True,
    block_on_viz: bool = True,
    viz_sampling_policy: Optional[dict] = None,
    timepoint_idxs_for_save: Optional[list] = None,
    model_name: str = "test_model__run_x__e0_i0",
):
    return InferencerWorker(
        aggregate_mode="none",
        inference_mode="tile",
        task=task,
        outputs_metadata=outputs_metadata,
        input_format=_INPUT_FORMAT,
        input_shape=list(_INPUT_SHAPE),
        patch_shape=list(_PATCH_SHAPE),
        model_name=model_name,
        save_outputs=save_outputs,
        block_on_save=block_on_save,
        vizualize_outputs=vizualize_outputs,
        block_on_viz=block_on_viz,
        model=model,
        buffer_manager=buffer_manager,
        save_worker=save_worker,
        viz_worker=viz_worker,
        viz_sampling_policy=viz_sampling_policy,
        timepoint_idxs_for_save=timepoint_idxs_for_save,
    )


def _make_data_sample(device: torch.device, batch_size: int = 1) -> dict:
    """Build a minimal data_sample dict with tensors on the given CUDA device."""
    data_tensor = torch.randn(batch_size, *_INPUT_SHAPE, device=device, dtype=torch.float32)
    tile_names = [f"tile_{i:03d}.zarr" for i in range(batch_size)]
    return {
        "data_tensor": data_tensor,
        "metainfo": {
            "roi_id": list(range(batch_size)),
            # _should_visualize compares tile_name as a scalar against the
            # policy list, so use a string for single-element batches.
            "tile_name": tile_names[0] if batch_size == 1 else tile_names,
            # Production emits these as batched (B, 3) tensors (see get_image_sizes);
            # postprocess() reads both keys, so both must be present.
            "image_sizes": torch.tensor([_SPATIAL] * batch_size, device=device),
            "orig_image_sizes": torch.tensor([_SPATIAL] * batch_size, device=device),
            "channel_mapping": dict(_CHANNEL_MAPPING_META),
        },
    }


# ---------------------------------------------------------------------------
# Tests: end-to-end over Ray + CUDA
# ---------------------------------------------------------------------------


@requires_cuda
class TestInferencerWorkerInit:
    """Verify that InferencerWorker initializes correctly for each task."""

    @pytest.mark.parametrize("task, make_meta, make_model, expected_outputs", [
        ("instance_segmentation", _make_outputs_metadata_instance_seg,
         lambda d: _MockInstanceSegModel(_SPATIAL, _TOPK, d), {"masks", "boxes", "labels"}),
        ("detection", _make_outputs_metadata_detection,
         lambda d: _MockDetectionModel(_TOPK, d), {"boxes", "labels"}),
    ], ids=["instance_segmentation", "detection"])
    def test_init_smoke(self, ray_ctx, ray_node_id, unique_suffix, task, make_meta, make_model, expected_outputs):
        """Constructor attaches + page-locks every registered pool and accepts each task."""
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        actors, sw, vw = [], None, None
        try:
            outputs_meta = make_meta()
            actors = _register_output_buffers(bm, outputs_meta)
            sw = _StubSaveWorker.options(name=f"sw_init_{unique_suffix}").remote(buffer_manager=bm)
            vw = _StubVizWorker.options(name=f"vw_init_{unique_suffix}").remote(buffer_manager=bm)
            with _patch_context():
                worker = _build_inferencer_worker(task=task, model=make_model(device), buffer_manager=bm,
                                                  save_worker=sw, viz_worker=vw, outputs_metadata=outputs_meta)
            assert worker.task == task
            assert set(worker.outputs_metadata["save_tensors"]) == expected_outputs
            assert set(bm._pinned_ptrs) == set(bm._buffer_actors)   # private: pin_buffers() covered every pool
        finally:
            _kill_safe(sw)
            _kill_safe(vw)
            for a in actors:
                _kill_safe(a)
            bm.shutdown()


@requires_cuda
class TestInferencerWorkerPredict:
    """End-to-end predict(): model -> D2H transfer -> stub save/viz workers."""

    def test_predict_verifies_d2h_transfer_values(
        self, ray_ctx, ray_node_id, unique_suffix
    ):
        """Check that GPU tensor values survive the D2H async copy."""
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        actors = []
        sw, vw = None, None
        try:
            outputs_meta = _make_outputs_metadata_detection()
            actors = _register_output_buffers(bm, outputs_meta)
            sw = _StubSaveWorker.options(
                name=f"sw_d2h_{unique_suffix}"
            ).remote(buffer_manager=bm)
            vw = _StubVizWorker.options(
                name=f"vw_d2h_{unique_suffix}"
            ).remote(buffer_manager=bm)

            model = _MockDetectionModel(_TOPK, device)
            with _patch_context():
                worker = _build_inferencer_worker(
                    task="detection",
                    model=model,
                    buffer_manager=bm,
                    save_worker=sw,
                    viz_worker=vw,
                    outputs_metadata=outputs_meta,
                    block_on_save=True,
                    block_on_viz=True,
                )

            data_sample = _make_data_sample(device)
            worker.predict(data_sample)

            save_received = ray.get(sw.get_received.remote())
            assert len(save_received) > 0, "Expected save worker to receive outputs"
            received_keys = set()
            for item in save_received:
                received_keys.update(item.keys())
                if "boxes" in item:
                    np.testing.assert_allclose(item["boxes"], 0.25, rtol=1e-5)
                if "labels" in item:
                    np.testing.assert_allclose(item["labels"], 0.8, rtol=1e-5)
            assert "boxes" in received_keys
            assert "labels" in received_keys
        finally:
            _kill_safe(sw)
            _kill_safe(vw)
            for a in actors:
                _kill_safe(a)
            bm.shutdown()


@requires_cuda
class TestInferencerWorkerVizPolicy:
    """Tests for viz_sampling_policy gating."""

    def test_viz_skipped_when_tile_not_in_policy(
        self, ray_ctx, ray_node_id, unique_suffix
    ):
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        actors = []
        sw, vw = None, None
        try:
            outputs_meta = _make_outputs_metadata_detection()
            actors = _register_output_buffers(bm, outputs_meta)
            sw = _StubSaveWorker.options(
                name=f"sw_vp_{unique_suffix}"
            ).remote(buffer_manager=bm)
            vw = _StubVizWorker.options(
                name=f"vw_vp_{unique_suffix}"
            ).remote(buffer_manager=bm)

            model = _MockDetectionModel(_TOPK, device)
            policy = {"name": "by_tile", "tile_names": ["other_tile.zarr"]}
            with _patch_context():
                worker = _build_inferencer_worker(
                    task="detection",
                    model=model,
                    buffer_manager=bm,
                    save_worker=sw,
                    viz_worker=vw,
                    outputs_metadata=outputs_meta,
                    block_on_save=True,
                    block_on_viz=True,
                    viz_sampling_policy=policy,
                )

            data_sample = _make_data_sample(device)
            worker.predict(data_sample)

            viz_metrics = ray.get(vw.get_metrics.remote())
            assert viz_metrics["visualize_calls"] == 0.0

            save_metrics = ray.get(sw.get_metrics.remote())
            assert save_metrics["save_successful"] == [True]
        finally:
            _kill_safe(sw)
            _kill_safe(vw)
            for a in actors:
                _kill_safe(a)
            bm.shutdown()

    def test_viz_dispatched_when_tile_in_policy(
        self, ray_ctx, ray_node_id, unique_suffix
    ):
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        actors = []
        sw, vw = None, None
        try:
            outputs_meta = _make_outputs_metadata_detection()
            actors = _register_output_buffers(bm, outputs_meta)
            sw = _StubSaveWorker.options(
                name=f"sw_vpd_{unique_suffix}"
            ).remote(buffer_manager=bm)
            vw = _StubVizWorker.options(
                name=f"vw_vpd_{unique_suffix}"
            ).remote(buffer_manager=bm)

            model = _MockDetectionModel(_TOPK, device)
            policy = {"name": "by_tile", "tile_names": ["tile_000.zarr"]}
            with _patch_context():
                worker = _build_inferencer_worker(
                    task="detection",
                    model=model,
                    buffer_manager=bm,
                    save_worker=sw,
                    viz_worker=vw,
                    outputs_metadata=outputs_meta,
                    block_on_save=True,
                    block_on_viz=True,
                    viz_sampling_policy=policy,
                )

            data_sample = _make_data_sample(device)
            worker.predict(data_sample)

            viz_metrics = ray.get(vw.get_metrics.remote())
            assert viz_metrics["visualize_calls"] == 1.0
        finally:
            _kill_safe(sw)
            _kill_safe(vw)
            for a in actors:
                _kill_safe(a)
            bm.shutdown()


@requires_cuda
class TestInferencerWorkerSaveDisabled:
    """Verify outputs are not saved when save_outputs=False."""

    def test_no_save_when_disabled(
        self, ray_ctx, ray_node_id, unique_suffix
    ):
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        actors = []
        sw, vw = None, None
        try:
            outputs_meta = _make_outputs_metadata_detection()
            actors = _register_output_buffers(bm, outputs_meta)
            sw = _StubSaveWorker.options(
                name=f"sw_nosave_{unique_suffix}"
            ).remote(buffer_manager=bm)
            vw = _StubVizWorker.options(
                name=f"vw_nosave_{unique_suffix}"
            ).remote(buffer_manager=bm)

            model = _MockDetectionModel(_TOPK, device)
            with _patch_context():
                worker = _build_inferencer_worker(
                    task="detection",
                    model=model,
                    buffer_manager=bm,
                    save_worker=sw,
                    viz_worker=vw,
                    outputs_metadata=outputs_meta,
                    save_outputs=False,
                    vizualize_outputs=False,
                    block_on_save=True,
                    block_on_viz=True,
                )

            data_sample = _make_data_sample(device)
            worker.predict(data_sample)

            save_metrics = ray.get(sw.get_metrics.remote())
            assert save_metrics["save_successful"] == []

            viz_metrics = ray.get(vw.get_metrics.remote())
            assert viz_metrics["visualize_calls"] == 0.0
        finally:
            _kill_safe(sw)
            _kill_safe(vw)
            for a in actors:
                _kill_safe(a)
            bm.shutdown()


@requires_cuda
class TestInferencerWorkerMultiplePredictions:
    """Run predict() more times than the pool has slots and verify slot reuse."""

    def test_multiple_predictions_reuse_buffers(self, ray_ctx, ray_node_id, unique_suffix):
        """3 predicts through 2-slot mask pools: every slot is returned (so the 3rd
        batch reuses one) and every batch's masks reach the save worker intact."""
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        actors, sw, vw = [], None, None
        try:
            outputs_meta = _make_outputs_metadata_instance_seg()        # masks in buffer_tensors
            actors = _register_output_buffers(bm, outputs_meta, buffer_capacity=2)
            sw = _StubSaveWorker.options(name=f"sw_multi_{unique_suffix}").remote(buffer_manager=bm)
            vw = _StubVizWorker.options(name=f"vw_multi_{unique_suffix}").remote(buffer_manager=bm)
            model = _MockInstanceSegModel(_SPATIAL, _TOPK, device)
            with _patch_context():
                worker = _build_inferencer_worker(
                    task="instance_segmentation", model=model, buffer_manager=bm,
                    save_worker=sw, viz_worker=vw, outputs_metadata=outputs_meta,
                    block_on_save=True, block_on_viz=True)
            n_iters = 3                                                   # > capacity: forces reuse
            for _ in range(n_iters):
                worker.predict(_make_data_sample(device))
            with _patch_context():
                worker.finalize()              # reaps save/viz tasks; raises on any failed put_free
            for pool in ("masks_save", "masks_viz"):
                _wait_all_slots_free(bm._buffer_actors[pool])             # 2/2 free again
            save_masks = [d["masks"] for d in ray.get(sw.get_received.remote()) if "masks" in d]
            assert len(save_masks) == n_iters
            for m in save_masks:
                assert m.dtype == np.uint16
                np.testing.assert_array_equal(m, 1)
            assert ray.get(sw.get_metrics.remote())["save_successful"] == [True] * n_iters
            assert ray.get(vw.get_metrics.remote())["visualize_successful"] == [True] * n_iters
        finally:
            _kill_safe(sw)
            _kill_safe(vw)
            for a in actors:
                _kill_safe(a)
            bm.shutdown()


@requires_cuda
class TestInferencerWorkerInstanceSegValues:
    """Verify mask / box / label values survive the full pipeline."""

    def test_instance_seg_mask_values(
        self, ray_ctx, ray_node_id, unique_suffix
    ):
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        actors = []
        sw, vw = None, None
        try:
            outputs_meta = _make_outputs_metadata_instance_seg()
            actors = _register_output_buffers(bm, outputs_meta)
            sw = _StubSaveWorker.options(
                name=f"sw_isval_{unique_suffix}"
            ).remote(buffer_manager=bm)
            vw = _StubVizWorker.options(
                name=f"vw_isval_{unique_suffix}"
            ).remote(buffer_manager=bm)

            model = _MockInstanceSegModel(_SPATIAL, _TOPK, device)
            with _patch_context():
                worker = _build_inferencer_worker(
                    task="instance_segmentation",
                    model=model,
                    buffer_manager=bm,
                    save_worker=sw,
                    viz_worker=vw,
                    outputs_metadata=outputs_meta,
                    block_on_save=True,
                    block_on_viz=True,
                )

            data_sample = _make_data_sample(device)
            worker.predict(data_sample)

            save_received = ray.get(sw.get_received.remote())
            assert len(save_received) > 0, "Expected save worker to receive outputs"
            received_keys = set()
            for item in save_received:
                received_keys.update(item.keys())
                if "masks" in item:
                    assert item["masks"].dtype == np.uint16
                    np.testing.assert_array_equal(item["masks"], 1)
                if "boxes" in item:
                    np.testing.assert_allclose(item["boxes"], 0.5, rtol=1e-5)
                if "labels" in item:
                    np.testing.assert_allclose(item["labels"], 0.9, rtol=1e-4)
            assert "masks" in received_keys
            assert "boxes" in received_keys
            assert "labels" in received_keys
        finally:
            _kill_safe(sw)
            _kill_safe(vw)
            for a in actors:
                _kill_safe(a)
            bm.shutdown()


@requires_cuda
class TestInferencerWorkerDroppedSaves:
    def test_non_blocking_save_on_exhausted_pool_drops_tile_and_fails_finalize(self, ray_ctx, ray_node_id, unique_suffix):
        """block_on_save=False on an exhausted save pool drops the tile (nothing is
        dispatched, nothing leaks) and finalize() refuses to declare the output complete."""
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        actors, sw, held, save_pool = [], None, None, None
        try:
            outputs_meta = _make_outputs_metadata_instance_seg()
            outputs_meta["visualize_tensors"] = []                          # save path only
            actors = _register_output_buffers(bm, outputs_meta, buffer_capacity=1)
            sw = _StubSaveWorker.options(name=f"sw_drop_{unique_suffix}").remote(buffer_manager=bm)
            model = _MockInstanceSegModel(_SPATIAL, _TOPK, device)
            with _patch_context():
                worker = _build_inferencer_worker(
                    task="instance_segmentation", model=model, buffer_manager=bm,
                    save_worker=sw, viz_worker=None, outputs_metadata=outputs_meta,
                    vizualize_outputs=False, block_on_save=False)
            save_pool = bm._buffer_actors["masks_save"]
            held = ray.get(save_pool.get_free.remote())                     # exhaust the 1-slot pool
            worker.predict(_make_data_sample(device))                        # must not block
            assert worker._dropped_saves == 1                                # private counter
            assert ray.get(sw.get_received.remote()) == []                   # nothing dispatched
            assert ray.get(save_pool.free_count.remote()) == 0               # our held slot, nothing leaked on top
            with _patch_context(), pytest.raises(RuntimeError, match="INCOMPLETE"):
                worker.finalize()
        finally:
            if held is not None:
                ray.get(save_pool.put_free.remote(held["slot"]))
            _kill_safe(sw)
            for a in actors:
                _kill_safe(a)
            bm.shutdown()


# ---------------------------------------------------------------------------
# Tier-0 pure function tests (no CUDA / Ray required)
# ---------------------------------------------------------------------------


class TestTreeToCpuNumpy:
    """Tests for InferencerWorker._tree_to_cpu_numpy (static method)."""

    def test_cpu_tensor_converted(self):
        meta = {"t": torch.tensor([1.0, 2.0])}
        result = InferencerWorker._tree_to_cpu_numpy(meta)
        assert isinstance(result["t"], np.ndarray)
        np.testing.assert_allclose(result["t"], np.array([1.0, 2.0]))

    def test_list_of_cpu_tensors(self):
        meta = {
            "sizes": [
                torch.tensor([64, 64]),
                torch.tensor([32, 32]),
            ]
        }
        result = InferencerWorker._tree_to_cpu_numpy(meta)
        assert all(isinstance(x, np.ndarray) for x in result["sizes"])

    def test_nested_list_of_tensors(self):
        meta = {"nested": [[torch.tensor(1.0), torch.tensor(2.0)], [torch.tensor(3.0)]]}
        result = InferencerWorker._tree_to_cpu_numpy(meta)
        assert isinstance(result["nested"][0][0], np.ndarray)
        assert isinstance(result["nested"][1][0], np.ndarray)

    def test_nested_dict(self):
        meta = {"inner": {"t": torch.tensor(0.5)}}
        result = InferencerWorker._tree_to_cpu_numpy(meta)
        assert isinstance(result["inner"]["t"], np.ndarray)
        np.testing.assert_allclose(result["inner"]["t"], np.array(0.5))

    def test_non_tensor_values_pass_through(self):
        meta = {"name": "tile_001.zarr", "idx": 42, "flag": True}
        result = InferencerWorker._tree_to_cpu_numpy(meta)
        assert result == {"name": "tile_001.zarr", "idx": 42, "flag": True}

    def test_tuple_preserved(self):
        meta = {"pair": (torch.tensor(1.0), torch.tensor(2.0))}
        result = InferencerWorker._tree_to_cpu_numpy(meta)
        assert isinstance(result["pair"], tuple)
        assert len(result["pair"]) == 2
        for x in result["pair"]:
            assert isinstance(x, np.ndarray)

    def test_namedtuple_type_preserved(self):
        Point = namedtuple("Point", ["x", "y"])
        obj = {"p": Point(torch.tensor([1.0]), torch.tensor([2.0]))}
        out = InferencerWorker._tree_to_cpu_numpy(obj)
        assert isinstance(out["p"], Point)
        np.testing.assert_allclose(out["p"].x, np.array([1.0]))
        np.testing.assert_allclose(out["p"].y, np.array([2.0]))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_tensors_converted(self):
        meta = {"t": torch.tensor([1.0, 2.0], device="cuda")}
        result = InferencerWorker._tree_to_cpu_numpy(meta)
        assert isinstance(result["t"], np.ndarray)
        np.testing.assert_allclose(result["t"], np.array([1.0, 2.0]))


class TestShouldVisualizePolicy:
    """CUDA-free unit tests for the _should_visualize policy logic itself
    (the Ray classes above cover end-to-end gating)."""

    @staticmethod
    def _worker(policy):
        w = _bare_worker()   # policy check reads only this attr
        w.viz_sampling_policy = policy
        return w

    @classmethod
    def _gate(cls, policy, metainfo):
        return cls._worker(policy)._should_visualize({"metainfo": metainfo}, {})

    _META = {
        "tile_name": ["a.zarr", "b.zarr"],
        "roi_id": np.array([3, 17]),
    }

    def test_tile_filter_only(self):
        assert self._gate({"name": "by_tile", "tile_names": ["b.zarr"]}, self._META)
        assert not self._gate({"name": "by_tile", "tile_names": ["c.zarr"]}, self._META)

    def test_roi_filter_only(self):
        assert self._gate({"name": "by_tile", "rois": [17]}, self._META)
        assert self._gate({"name": "by_tile", "rois": ["3"]}, self._META)  # str/int agnostic
        assert not self._gate({"name": "by_tile", "rois": [99]}, self._META)

    def test_roi_and_tile_must_match_same_sample(self):
        # roi 3 pairs with a.zarr, roi 17 with b.zarr. Asking for (3, b.zarr)
        # matches NO single sample even though each value appears in the batch.
        assert not self._gate(
            {"name": "by_tile", "rois": [3], "tile_names": ["b.zarr"]}, self._META
        )
        assert self._gate(
            {"name": "by_tile", "rois": [17], "tile_names": ["b.zarr"]}, self._META
        )

    def test_scalar_columns_b1_fixture(self):
        meta = {"tile_name": "a.zarr", "roi_id": 3}
        assert self._gate({"name": "by_tile", "rois": [3], "tile_names": ["a.zarr"]}, meta)

    def test_scalar_tile_name_matched_whole_not_charwise(self):
        """A bare-string tile_name is one sample, not an iterable of characters."""
        policy = {"name": "by_tile", "tile_names": ["b.zarr"]}
        assert self._gate(policy, {"tile_name": "b.zarr"}) is True
        # single chars of "b.zarr" must NOT match membership semantics
        assert self._gate({"name": "by_tile", "tile_names": ["b"]}, {"tile_name": "b.zarr"}) is False

    def test_no_filters_raises(self):
        with pytest.raises(ValueError, match="tile_names.*rois"):
            self._gate({"name": "by_tile"}, self._META)

    def test_missing_roi_column_raises(self):
        with pytest.raises(KeyError, match="roi_id"):
            self._gate({"name": "by_tile", "rois": [1]}, {"tile_name": ["a.zarr"]})

    def test_unknown_policy_name_raises(self):
        with pytest.raises(ValueError, match="unknown viz_sampling_policy"):
            self._gate({"name": "bogus"}, {"tile_name": ["x"]})

    def test_none_policy_and_random(self):
        assert self._gate(None, self._META)
        assert self._gate({"name": "random_sample", "fraction": 1.0}, self._META)
        assert not self._gate({"name": "random_sample", "fraction": 0.0}, self._META)

    def test_by_tile_subset_match_publishes_sample_indices(self):
        """A by_tile match on a strict subset of the batch records WHICH samples
        matched so the viz worker renders only those."""
        w = self._worker({"name": "by_tile", "tile_names": ["b.zarr"]})
        meta = {"tile_name": ["a.zarr", "b.zarr", "b.zarr"]}
        assert w._should_visualize({"metainfo": meta}, {}) is True
        assert w._viz_sample_idx == [1, 2]

    def test_no_policy_publishes_no_sample_indices(self):
        w = self._worker(None)
        assert w._should_visualize({"metainfo": {}}, {}) is True
        assert w._viz_sample_idx is None


class TestAttachSaveWorkerMetainfo:
    """``_attach_save_worker_metainfo`` fills the keys ``SaveWorker.save`` requires
    without clobbering values the dataset already supplied."""

    @staticmethod
    def _worker(timepoint_idxs):
        w = _bare_worker()
        w.model_name, w.task = "m__run_x__e0_i0", "instance_segmentation"
        w.outputs_metadata = {"save_tensors": {"masks": {"annotation_type": "dense"}}}
        w._save_timepoint_idxs = timepoint_idxs          # private; set by __init__ from timepoint_idxs_for_save
        return w

    def test_attaches_save_keys_and_timepoints(self):
        mi = {"channel_mapping": {"0": "c0"}}
        self._worker([0])._attach_save_worker_metainfo(mi)
        assert mi["model_name"] == "m__run_x__e0_i0" and mi["task"] == "instance_segmentation"
        assert mi["save_tensors_metadata"] == {"masks": {"annotation_type": "dense"}}
        assert mi["timepoint_idxs"] == [0] and mi["channel_mapping"] == {"0": "c0"}

    def test_existing_timepoints_not_overwritten(self):
        mi = {"timepoint_idxs": [9]}
        self._worker([0])._attach_save_worker_metainfo(mi)
        assert mi["timepoint_idxs"] == [9]

    def test_no_timepoints_key_when_unset(self):
        mi = {}
        self._worker(None)._attach_save_worker_metainfo(mi)
        assert "timepoint_idxs" not in mi


class TestStagingBufferCache:
    """_get_staging_buffer: one flat pinned buffer per (role/name, dtype), grown by
    capacity doubling -- variable shapes must NOT mint a new pinned block each."""

    @staticmethod
    def _no_pin():
        # CPU-only envs cannot allocate pinned memory; the caching logic under
        # test is orthogonal to the pinning itself.
        real_empty = torch.empty

        def fake_empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        return patch("torch.empty", new=fake_empty)

    @classmethod
    def _worker(cls):
        w = _bare_worker()
        w._staging_buffers = {}
        return w

    def test_variable_shapes_reuse_one_buffer(self):
        w = self._worker()
        with self._no_pin():
            a = w._get_staging_buffer("save", "boxes", torch.zeros(2, 5, 6))
            b = w._get_staging_buffer("save", "boxes", torch.zeros(2, 3, 6))
        assert len(w._staging_buffers) == 1                       # not per-shape
        assert a.shape == (2, 5, 6) and b.shape == (2, 3, 6)
        assert b.data_ptr() == a.data_ptr()                       # same flat block

    def test_capacity_doubles_not_creeps(self):
        w = self._worker()
        with self._no_pin():
            w._get_staging_buffer("save", "boxes", torch.zeros(10))
            w._get_staging_buffer("save", "boxes", torch.zeros(12))
        (buf,) = w._staging_buffers.values()
        assert buf.numel() == 20                                  # max(12, 2*10)

    def test_roundtrip_contents(self):
        w = self._worker()
        src = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        with self._no_pin():
            view = w._get_staging_buffer("viz", "scores", src)
            view.copy_(src)
        torch.testing.assert_close(view, src)

    def test_keys_split_by_role_name_dtype(self):
        w = self._worker()
        with self._no_pin():
            w._get_staging_buffer("save", "boxes", torch.zeros(4))
            w._get_staging_buffer("viz", "boxes", torch.zeros(4))
            w._get_staging_buffer("save", "labels", torch.zeros(4, dtype=torch.int64))
        assert len(w._staging_buffers) == 3

    def test_zero_size(self):
        w = self._worker()
        with self._no_pin():
            view = w._get_staging_buffer("save", "boxes", torch.zeros(0, 6))
        assert view.shape == (0, 6)

    @pytest.mark.cuda
    @requires_cuda
    def test_staging_block_is_pinned_host_memory(self):
        w = self._worker()
        view = w._get_staging_buffer("save", "boxes", torch.zeros(2, 4, device="cuda"))
        assert view.is_pinned() and view.device.type == "cpu"


class TestBlockingGetFree:
    """_blocking_get_free: patient while the drain worker is alive, fail-fast when
    dead, and the ONE issued get_free request is cancelled before any abort."""

    class _Timeout(Exception):
        pass

    def _run(self, get_script):
        """get_script: list of callables consumed per ray.get call, keyed off the
        awaited ref ('slot' request vs worker 'probe')."""
        w = _bare_worker()
        w._GET_FREE_HEARTBEAT_S = 0.01

        slot_ref, probe_ref = object(), object()
        buffer = type("B", (), {})()
        buffer.get_free = type("M", (), {"remote": staticmethod(lambda: slot_ref)})()

        def _never(*a, **k):
            raise AssertionError("get_metrics RESETS worker metrics; the probe must use ping()")
        worker = type("W", (), {})()
        worker.ping = type("M", (), {"remote": staticmethod(lambda: probe_ref)})()
        worker.get_metrics = type("M", (), {"remote": staticmethod(_never)})()

        calls = {"cancel": []}
        script = list(get_script)

        def fake_get(ref, timeout=None):
            which = "slot" if ref is slot_ref else "probe"
            return script.pop(0)(which)

        result, exc = None, None
        with patch.object(inf_mod.ray, "get", side_effect=fake_get), \
             patch.object(inf_mod.ray, "cancel", side_effect=calls["cancel"].append), \
             patch.object(inf_mod.ray, "exceptions") as exc_mod, \
             patch.object(inf_mod.ray, "logger"):
            exc_mod.GetTimeoutError = self._Timeout
            try:
                result = w._blocking_get_free(buffer, "pool_x", worker)
            except RuntimeError as e:
                exc = e
        assert not script, "unconsumed ray.get script steps"
        return result, exc, calls["cancel"], slot_ref

    def test_returns_slot_immediately(self):
        result, exc, cancelled, _ = self._run([lambda which: "slot0"])
        assert result == "slot0" and exc is None and cancelled == []

    def test_slow_but_alive_drain_keeps_waiting(self):
        def timeout(which):
            assert which == "slot"
            raise self._Timeout()

        result, exc, cancelled, _ = self._run([
            timeout,                         # slot wait times out
            lambda which: True,              # probe: worker alive
            lambda which: "slot0",           # next wait succeeds
        ])
        assert result == "slot0" and exc is None
        assert cancelled == []               # request never abandoned

    def test_dead_drain_fails_fast_and_cancels_the_request(self):
        def timeout(which):
            raise self._Timeout()

        def dead(which):
            assert which == "probe"
            raise RuntimeError("actor died")

        result, exc, cancelled, slot_ref = self._run([timeout, dead])
        assert result is None
        assert exc is not None and "DEAD" in str(exc)
        assert cancelled == [slot_ref]       # abandoned request is cancelled

    def test_busy_probe_timeout_counts_as_alive(self):
        def timeout(which):
            raise self._Timeout()            # slot wait AND probe both time out

        result, exc, cancelled, _ = self._run([
            timeout,                         # slot wait times out
            timeout,                         # probe times out: busy != dead
            lambda which: "slot0",
        ])
        assert result == "slot0" and exc is None and cancelled == []

    def test_liveness_probe_uses_ping_not_get_metrics(self):
        """get_metrics RESETS worker metrics; the heartbeat must probe ping()."""
        calls = []
        worker = SimpleNamespace(
            ping=SimpleNamespace(remote=lambda: calls.append("ping") or "ref"),
            get_metrics=SimpleNamespace(remote=lambda: calls.append("get_metrics") or "ref"),
        )
        with patch.object(inf_mod.ray, "get", return_value=True):
            assert InferencerWorker._drain_worker_alive(worker) is True
        assert calls == ["ping"]


class TestFinalize:
    """finalize(): reaps every save/viz task, drains outstanding slot frees, and
    refuses to declare a run complete when tiles were dropped."""

    @staticmethod
    def _worker(tasks=(), dropped=0, drain=lambda: None):
        w = _bare_worker()
        w._tasks = list(tasks)
        w._dropped_saves = dropped
        w.buffer_manager = SimpleNamespace(drain_free_refs=drain)
        return w

    @staticmethod
    def _quiet():
        return _MultiPatch(
            patch.object(inf_mod.ray, "logger"),
            patch.object(inf_mod.torch.cuda, "synchronize"),
            patch.object(inf_mod, "barrier"),
        )

    def test_raises_when_saves_were_dropped(self):
        with self._quiet():
            with pytest.raises(RuntimeError, match="INCOMPLETE"):
                self._worker(dropped=3).finalize()

    def test_clean_when_nothing_dropped(self):
        with self._quiet():
            self._worker(dropped=0).finalize()  # no raise

    def test_collects_every_failed_task(self):
        """Every failed task is collected (not just the first) and the list is cleared."""
        w = self._worker(tasks=[object(), object(), object()])
        with patch.object(inf_mod.ray, "get", side_effect=RuntimeError("dead")), self._quiet():
            with pytest.raises(RuntimeError, match="3/3 save/viz tasks failed"):
                w.finalize(overall_deadline_s=0.5)
        assert w._tasks == []

    def test_exhausted_deadline_fails_remaining_tasks_without_waiting(self):
        w = self._worker(tasks=[object(), object()])
        with patch.object(inf_mod.ray, "get") as rg, self._quiet():
            with pytest.raises(RuntimeError, match="2/2"):
                w.finalize(overall_deadline_s=0.0)
        rg.assert_not_called()   # past deadline: no per-task blocking waits

    def test_failed_slot_free_surfaces_at_finalize(self):
        """A producer-side double free raises at finalize instead of dying as a
        background log line after teardown."""
        drain = MagicMock(side_effect=RuntimeError("double free"))
        w = self._worker(drain=drain)
        with self._quiet():
            with pytest.raises(RuntimeError, match="failed"):
                w.finalize()
        drain.assert_called_once()


class TestStageOutput:
    """_stage_output: SHM slot when reserved, pinned staging otherwise; CPU
    sources copy host-side (a host src pointer in cudaMemcpyAsync is invalid)."""

    def test_cpu_tensor_copied_host_side_into_slot(self):
        w = _bare_worker()
        w.outputs_metadata = {"tensor_info": {"boxes": {"dtype": "float32"}}}
        dest = np.zeros((2, 3), dtype=np.float32)
        w.buffer_manager = SimpleNamespace(slot_info_to_view=lambda s: dest)
        w._copy_d2h = MagicMock(side_effect=AssertionError("must not memcpyAsync a CPU src"))
        src = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        out = {}
        w._stage_output("save", "boxes", src, {"boxes": {"slot": 0}}, out)
        np.testing.assert_array_equal(dest, src.numpy())     # byte-identical
        assert out["boxes"] == {"slot": 0}
        w._copy_d2h.assert_not_called()

    def test_unbuffered_tensor_goes_to_pinned_staging(self):
        w = _bare_worker()
        w.outputs_metadata = {"tensor_info": {"boxes": {"dtype": "float32"}}}
        w._staging_buffers = {}
        real_empty = torch.empty

        def no_pin(*a, **k):
            k.pop("pin_memory", None)
            return real_empty(*a, **k)

        src = torch.ones(2, 3)
        out = {}
        with patch("torch.empty", new=no_pin):
            w._stage_output("save", "boxes", src, {}, out)
        torch.testing.assert_close(out["boxes"], src)


class TestCopyD2HGuards:
    """_copy_d2h output-contract checks fire before the memcpy -- CPU-testable."""

    def test_per_sample_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="per-sample shape mismatch"):
            InferencerWorker._copy_d2h(_bare_worker(), dst=np.zeros((1, 2, 2), np.float32),
                                       src=torch.zeros(1, 2, 3))

    def test_overrun_raises(self):
        # same per-sample shape, more samples than the slot holds
        with pytest.raises(ValueError, match="overrun"):
            InferencerWorker._copy_d2h(_bare_worker(), dst=np.zeros((1, 2, 2), np.float32),
                                       src=torch.zeros(2, 2, 2))

    def test_same_width_dtype_mismatch_raises(self):
        with pytest.raises(ValueError, match="dtype mismatch"):
            InferencerWorker._copy_d2h(_bare_worker(), dst=np.zeros((1, 2, 2), np.int32),
                                       src=torch.zeros(1, 2, 2, dtype=torch.float32))


class TestVizImageTransportValidation:
    """Startup validation: viz handlers that read ``record.image`` need
    ``data_tensor`` routed through a SHM viz pool."""

    _META_OK = {
        "visualize_tensors": ["masks", "data_tensor"],
        "buffer_tensors": ["masks", "data_tensor"],
    }
    _META_MISSING = {
        "visualize_tensors": ["masks"],
        "buffer_tensors": ["masks"],
    }

    def test_image_consuming_handler_requires_data_tensor(self):
        with pytest.raises(ValueError, match="data_tensor"):
            InferencerWorker._validate_viz_image_transport(
                self._META_MISSING, ["instance_overlay"]
            )

    @pytest.mark.parametrize("handler", sorted(InferencerWorker._IMAGE_CONSUMING_VIZ_HANDLERS))
    def test_every_image_consuming_handler_requires_data_tensor(self, handler):
        with pytest.raises(ValueError, match="data_tensor"):
            InferencerWorker._validate_viz_image_transport(self._META_MISSING, [handler])

    def test_ok_when_data_tensor_buffered(self):
        InferencerWorker._validate_viz_image_transport(
            self._META_OK, ["instance_overlay", "save_predictions"]
        )  # no raise

    def test_no_image_handler_no_requirement(self):
        InferencerWorker._validate_viz_image_transport(
            self._META_MISSING, ["save_predictions"]
        )  # no raise


class _StopAfterReap(Exception):
    pass


class TestPredictReapsFinishedTasks:
    """predict() reaps finished save/viz tasks before the next batch: a finished-but-
    failed task raises, pending tasks are kept."""

    @staticmethod
    def _worker(tasks):
        w = _bare_worker()
        w.aggregate_mode, w._tasks = "none", list(tasks)
        w.model = SimpleNamespace(inference_step=lambda ds: (_ for _ in ()).throw(_StopAfterReap()))
        return w

    def test_failed_finished_task_raises_before_next_batch(self):
        t_ok, t_bad = object(), object()
        w = self._worker([t_ok, t_bad])

        def fake_get(ref, **kw):
            if ref is t_bad:
                raise RuntimeError("save died")

        with patch.object(inf_mod.ray, "wait", return_value=([t_ok, t_bad], [])), \
             patch.object(inf_mod.ray, "get", side_effect=fake_get), patch.object(inf_mod.ray, "logger"):
            with pytest.raises(RuntimeError, match=r"1/2 save/viz tasks failed") as ei:
                w.predict({"data_tensor": None})
        assert isinstance(ei.value.__cause__, RuntimeError) and w._tasks == []

    def test_pending_tasks_are_kept(self):
        t_pending = object()
        w = self._worker([t_pending])
        with patch.object(inf_mod.ray, "wait", return_value=([], [t_pending])), \
             patch.object(inf_mod.ray, "get") as rg, patch.object(inf_mod.ray, "logger"):
            with pytest.raises(_StopAfterReap):          # reaping done; model call is the next step
                w.predict({"data_tensor": None})
        rg.assert_not_called()
        assert w._tasks == [t_pending]


# ---------------------------------------------------------------------------
# Context-manager helpers
# ---------------------------------------------------------------------------


def _patch_context():
    """Patch process_rank / get_world_size / barrier for non-distributed tests."""
    return _MultiPatch(
        patch("cell_observatory_platform.inference.inferencer.process_rank", return_value=0),
        patch("cell_observatory_platform.inference.inferencer.get_world_size", return_value=1),
        patch("cell_observatory_platform.inference.inferencer.barrier"),
    )


class _MultiPatch:
    """Context manager that enters multiple patches."""

    def __init__(self, *patches):
        self._patches = patches

    def __enter__(self):
        for p in self._patches:
            p.__enter__()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.__exit__(*exc)
