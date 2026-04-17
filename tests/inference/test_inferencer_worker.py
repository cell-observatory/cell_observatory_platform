"""Unit tests for InferencerWorker.

Tests cover:
- Initialization with each supported task (instance_segmentation, detection)
- Full predict() flow: CUDA tensor in -> D2H transfer -> save/viz worker dispatch
- Output buffer slot lifecycle for ``buffer_tensors`` only (acquire, fill, free); small tensors use inline numpy
- Correct routing through _predict for each task branch

Requirements:
- Ray cluster (module-scoped, no GPU actors)
- CUDA device available (tests skipped otherwise)

Mock models return fixed tensors so we can verify data flows through
the worker correctly without loading real model weights.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pytest
import ray
import torch
import torch.nn as nn

_CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")


# ---------------------------------------------------------------------------
# Mock models
# ---------------------------------------------------------------------------


class _MockInstanceSegModel(nn.Module):
    """Returns fixed masks / boxes / labels like MaskDINO.predict()."""

    def __init__(self, spatial_shape: Tuple[int, ...], topk: int, device: torch.device):
        super().__init__()
        self._spatial = spatial_shape
        self._topk = topk
        self._device = device
        self.output_metadata = {
            "tensor_info": {
                "masks": {"shape": spatial_shape, "dtype": "uint16"},
                "boxes": {"shape": (topk, 6), "dtype": "float32"},
                "labels": {"shape": (topk,), "dtype": "float32"},
            },
        }

    def predict(self, data_sample: dict) -> Dict[str, torch.Tensor]:
        B = data_sample["data_tensor"].shape[0]
        masks = torch.ones((B, *self._spatial), dtype=torch.uint16, device=self._device)
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
    """Returns fixed boxes / labels like PlainDETR.predict()."""

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

    def predict(self, data_sample: dict) -> Dict[str, torch.Tensor]:
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


def _wait_buffer_in_use_zero(buffer_actor, *, timeout_s: float = 5.0, poll_s: float = 0.05) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        m = ray.get(buffer_actor.get_metrics.remote())
        if m["occupied_slots"][-1] == 0:
            return
        time.sleep(poll_s)
    m = ray.get(buffer_actor.get_metrics.remote())
    assert m["occupied_slots"][-1] == 0, f"Expected occupied_slots==0 within {timeout_s}s, got {m!r}"


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
            "masks": {"shape": _SPATIAL, "dtype": "uint16"},
            "boxes": {"shape": (_TOPK, 6), "dtype": "float32"},
            "labels": {"shape": (_TOPK,), "dtype": "float32"},
        },
        "save_tensors_dtypes": {
            "masks": "uint16",
            "boxes": "float32",
            "labels": "float32",
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
        "save_tensors_dtypes": {
            "boxes": "float32",
            "labels": "float32",
        },
    }


def _register_output_buffers(
    bm,
    outputs_metadata: dict,
    suffix: str,
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
    decoder_head_type: str,
    save_outputs: bool = True,
    vizualize_outputs: bool = True,
    block_on_save: bool = True,
    block_on_viz: bool = True,
    viz_sampling_policy: Optional[dict] = None,
    channel_names: Optional[dict] = None,
    timepoint_idxs_for_save: Optional[list] = None,
    model_name: str = "test_model__run_x__e0_i0",
):
    from cell_observatory_platform.inference.inferencer import InferencerWorker

    if save_outputs and channel_names is None:
        channel_names = {0: "test_channel_0"}

    return InferencerWorker(
        aggregate_mode="none",
        inference_mode="tile",
        task=task,
        outputs_metadata=outputs_metadata,
        input_format=_INPUT_FORMAT,
        input_shape=list(_INPUT_SHAPE),
        patch_shape=list(_PATCH_SHAPE),
        decoder_head_type=decoder_head_type,
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
        channel_names=channel_names,
        timepoint_idxs_for_save=timepoint_idxs_for_save,
    )


def _make_data_sample(device: torch.device, batch_size: int = 1) -> dict:
    """Build a minimal data_sample dict with tensors on the given CUDA device."""
    data_tensor = torch.randn(batch_size, *_INPUT_SHAPE, device=device, dtype=torch.float32)
    tile_names = [f"tile_{i:03d}.zarr" for i in range(batch_size)]
    return {
        "data_tensor": data_tensor,
        "metainfo": {
            "prepared_id": list(range(batch_size)),
            # _should_visualize compares tile_name as a scalar against the
            # policy list, so use a string for single-element batches.
            "tile_name": tile_names[0] if batch_size == 1 else tile_names,
            "orig_image_sizes": [torch.tensor(_SPATIAL, device=device)] * batch_size,
            "channel_mapping": dict(_CHANNEL_MAPPING_META),
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_cuda
class TestInferencerWorkerInit:
    """Verify that InferencerWorker initializes correctly for each task."""

    def test_init_instance_segmentation(self, ray_ctx, ray_node_id, unique_suffix):
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        actors = []
        sw, vw = None, None
        try:
            outputs_meta = _make_outputs_metadata_instance_seg()
            actors = _register_output_buffers(bm, outputs_meta, unique_suffix)
            sw = _StubSaveWorker.options(name=f"sw_init_is_{unique_suffix}").remote(buffer_manager=bm)
            vw = _StubVizWorker.options(name=f"vw_init_is_{unique_suffix}").remote(buffer_manager=bm)

            model = _MockInstanceSegModel(_SPATIAL, _TOPK, device)
            with _patch_context():
                worker = _build_inferencer_worker(
                    task="instance_segmentation",
                    model=model,
                    buffer_manager=bm,
                    save_worker=sw,
                    viz_worker=vw,
                    outputs_metadata=outputs_meta,
                    decoder_head_type="maskdino",
                )
            assert worker.task == "instance_segmentation"
            assert worker.main_output_name == "masks"
            assert worker.input_format == _INPUT_FORMAT
        finally:
            _kill_safe(sw)
            _kill_safe(vw)
            for a in actors:
                _kill_safe(a)
            bm.shutdown()

    def test_init_detection(self, ray_ctx, ray_node_id, unique_suffix):
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        actors = []
        sw, vw = None, None
        try:
            outputs_meta = _make_outputs_metadata_detection()
            actors = _register_output_buffers(bm, outputs_meta, unique_suffix)
            sw = _StubSaveWorker.options(name=f"sw_init_det_{unique_suffix}").remote(buffer_manager=bm)
            vw = _StubVizWorker.options(name=f"vw_init_det_{unique_suffix}").remote(buffer_manager=bm)

            model = _MockDetectionModel(_TOPK, device)
            with _patch_context():
                worker = _build_inferencer_worker(
                    task="detection",
                    model=model,
                    buffer_manager=bm,
                    save_worker=sw,
                    viz_worker=vw,
                    outputs_metadata=outputs_meta,
                    decoder_head_type="plaindetr",
                )
            assert worker.task == "detection"
            assert worker.main_output_name == "boxes"
        finally:
            _kill_safe(sw)
            _kill_safe(vw)
            for a in actors:
                _kill_safe(a)
            bm.shutdown()

    def test_timepoint_idxs_for_save_attached_to_save_metainfo(
        self, ray_ctx, ray_node_id, unique_suffix
    ):
        """datasets.timepoint_list → worker → metainfo timepoint_idxs for N-format IO."""
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        actors = []
        sw, vw = None, None
        try:
            outputs_meta = _make_outputs_metadata_instance_seg()
            actors = _register_output_buffers(bm, outputs_meta, unique_suffix)
            sw = _StubSaveWorker.options(name=f"sw_tpidx_{unique_suffix}").remote(buffer_manager=bm)
            vw = _StubVizWorker.options(name=f"vw_tpidx_{unique_suffix}").remote(buffer_manager=bm)
            model = _MockInstanceSegModel(_SPATIAL, _TOPK, device)
            with _patch_context():
                worker = _build_inferencer_worker(
                    task="instance_segmentation",
                    model=model,
                    buffer_manager=bm,
                    save_worker=sw,
                    viz_worker=vw,
                    outputs_metadata=outputs_meta,
                    decoder_head_type="maskdino",
                    timepoint_idxs_for_save=[0],
                )
            mi = {"channel_mapping": dict(_CHANNEL_MAPPING_META)}
            worker._attach_save_worker_metainfo(mi)
            assert mi["timepoint_idxs"] == [0]
            mi_existing = {"timepoint_idxs": [9], "channel_mapping": dict(_CHANNEL_MAPPING_META)}
            worker._attach_save_worker_metainfo(mi_existing)
            assert mi_existing["timepoint_idxs"] == [9]
        finally:
            _kill_safe(sw)
            _kill_safe(vw)
            for a in actors:
                _kill_safe(a)
            bm.shutdown()

    @pytest.mark.parametrize("bad_channels", [None, {}])
    def test_save_outputs_requires_channel_names(self, ray_ctx, ray_node_id, unique_suffix, bad_channels):
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        sw = _StubSaveWorker.options(name=f"sw_chreq_{unique_suffix}").remote(buffer_manager=bm)
        vw = _StubVizWorker.options(name=f"vw_chreq_{unique_suffix}").remote(buffer_manager=bm)
        try:
            outputs_meta = _make_outputs_metadata_instance_seg()
            model = _MockInstanceSegModel(_SPATIAL, _TOPK, device)
            from cell_observatory_platform.inference.inferencer import InferencerWorker

            kwargs = dict(
                aggregate_mode="none",
                inference_mode="tile",
                task="instance_segmentation",
                outputs_metadata=outputs_meta,
                input_format=_INPUT_FORMAT,
                input_shape=list(_INPUT_SHAPE),
                patch_shape=list(_PATCH_SHAPE),
                decoder_head_type="maskdino",
                model_name="need_channels",
                save_outputs=True,
                block_on_save=True,
                vizualize_outputs=False,
                block_on_viz=False,
                model=model,
                buffer_manager=bm,
                save_worker=sw,
                viz_worker=vw,
            )
            if bad_channels is not None:
                kwargs["channel_names"] = bad_channels
            with _patch_context(), pytest.raises(ValueError, match="channel_names"):
                InferencerWorker(**kwargs)
        finally:
            _kill_safe(sw)
            _kill_safe(vw)
            bm.shutdown()


@requires_cuda
class TestInferencerWorkerPredict:
    """End-to-end predict(): model -> D2H transfer -> stub save/viz workers."""

    def test_predict_instance_segmentation_dispatches_outputs(
        self, ray_ctx, ray_node_id, unique_suffix
    ):
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        actors = []
        sw, vw = None, None
        try:
            outputs_meta = _make_outputs_metadata_instance_seg()
            actors = _register_output_buffers(bm, outputs_meta, unique_suffix)
            sw = _StubSaveWorker.options(
                name=f"sw_pred_is_{unique_suffix}"
            ).remote(buffer_manager=bm)
            vw = _StubVizWorker.options(
                name=f"vw_pred_is_{unique_suffix}"
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
                    decoder_head_type="maskdino",
                    block_on_save=True,
                    block_on_viz=True,
                )

            data_sample = _make_data_sample(device)
            worker.predict(data_sample)

            save_metrics = ray.get(sw.get_metrics.remote())
            assert save_metrics["save_successful"] == [True]
            assert save_metrics["save_successful"].count(False) == 0

            viz_metrics = ray.get(vw.get_metrics.remote())
            assert viz_metrics["visualize_calls"] == 1.0

            save_received = ray.get(sw.get_received.remote())
            assert len(save_received) > 0
            received_keys = set()
            for d in save_received:
                received_keys.update(d.keys())
            assert "masks" in received_keys
            assert "boxes" in received_keys
            assert "labels" in received_keys
        finally:
            _kill_safe(sw)
            _kill_safe(vw)
            for a in actors:
                _kill_safe(a)
            bm.shutdown()

    def test_predict_detection_dispatches_outputs(
        self, ray_ctx, ray_node_id, unique_suffix
    ):
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        actors = []
        sw, vw = None, None
        try:
            outputs_meta = _make_outputs_metadata_detection()
            actors = _register_output_buffers(bm, outputs_meta, unique_suffix)
            sw = _StubSaveWorker.options(
                name=f"sw_pred_det_{unique_suffix}"
            ).remote(buffer_manager=bm)
            vw = _StubVizWorker.options(
                name=f"vw_pred_det_{unique_suffix}"
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
                    decoder_head_type="plaindetr",
                    block_on_save=True,
                    block_on_viz=True,
                )

            data_sample = _make_data_sample(device)
            worker.predict(data_sample)

            save_metrics = ray.get(sw.get_metrics.remote())
            assert save_metrics["save_successful"] == [True]

            save_received = ray.get(sw.get_received.remote())
            received_keys = set()
            for d in save_received:
                received_keys.update(d.keys())
            assert "boxes" in received_keys
            assert "labels" in received_keys
        finally:
            _kill_safe(sw)
            _kill_safe(vw)
            for a in actors:
                _kill_safe(a)
            bm.shutdown()

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
            actors = _register_output_buffers(bm, outputs_meta, unique_suffix)
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
                    decoder_head_type="plaindetr",
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
            actors = _register_output_buffers(bm, outputs_meta, unique_suffix)
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
                    decoder_head_type="plaindetr",
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
            actors = _register_output_buffers(bm, outputs_meta, unique_suffix)
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
                    decoder_head_type="plaindetr",
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
            actors = _register_output_buffers(bm, outputs_meta, unique_suffix)
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
                    decoder_head_type="plaindetr",
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
    """Run predict() multiple times and verify buffer reuse."""

    def test_multiple_predictions_reuse_buffers(
        self, ray_ctx, ray_node_id, unique_suffix
    ):
        device = torch.device("cuda:0")
        bm = _make_buffer_manager(ray_node_id)
        actors = []
        sw, vw = None, None
        try:
            outputs_meta = _make_outputs_metadata_detection()
            actors = _register_output_buffers(bm, outputs_meta, unique_suffix, buffer_capacity=4)
            sw = _StubSaveWorker.options(
                name=f"sw_multi_{unique_suffix}"
            ).remote(buffer_manager=bm)
            vw = _StubVizWorker.options(
                name=f"vw_multi_{unique_suffix}"
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
                    decoder_head_type="plaindetr",
                    block_on_save=True,
                    block_on_viz=True,
                )

            n_iters = 3
            for _ in range(n_iters):
                data_sample = _make_data_sample(device)
                worker.predict(data_sample)

            save_metrics = ray.get(sw.get_metrics.remote())
            assert len(save_metrics["save_successful"]) == n_iters
            assert save_metrics["save_successful"].count(True) == n_iters
            assert save_metrics["save_successful"].count(False) == 0

            for tensor_name in outputs_meta["buffer_tensors"]:
                if tensor_name in outputs_meta["save_tensors"]:
                    pool_name = f"{tensor_name}_save"
                    buf = bm._buffer_actors[pool_name]
                    _wait_buffer_in_use_zero(buf)
                if tensor_name in outputs_meta["visualize_tensors"]:
                    pool_name = f"{tensor_name}_viz"
                    buf = bm._buffer_actors[pool_name]
                    _wait_buffer_in_use_zero(buf)
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
            actors = _register_output_buffers(bm, outputs_meta, unique_suffix)
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
                    decoder_head_type="maskdino",
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


# ---------------------------------------------------------------------------
# Tier-0 pure function tests (no CUDA / Ray required)
# ---------------------------------------------------------------------------


class TestResolvePath:
    """Tests for InferencerWorker.resolve_path (static method)."""

    def test_simple_key(self):
        from cell_observatory_platform.inference.inferencer import InferencerWorker

        root = {"foo": 42}
        assert InferencerWorker.resolve_path(root, "foo") == 42

    def test_nested_key(self):
        from cell_observatory_platform.inference.inferencer import InferencerWorker

        root = {"a": {"b": {"c": 99}}}
        assert InferencerWorker.resolve_path(root, "a.b.c") == 99

    def test_list_index(self):
        from cell_observatory_platform.inference.inferencer import InferencerWorker

        root = {"items": [10, 20, 30]}
        assert InferencerWorker.resolve_path(root, "items[1]") == 20

    def test_bare_numeric_index(self):
        from cell_observatory_platform.inference.inferencer import InferencerWorker

        root = {"items": [10, 20, 30]}
        assert InferencerWorker.resolve_path(root, "items.2") == 30


class TestTreeToCpuNumpy:
    """Tests for InferencerWorker._tree_to_cpu_numpy (static method)."""

    def test_cpu_tensor_converted(self):
        from cell_observatory_platform.inference.inferencer import InferencerWorker

        meta = {"t": torch.tensor([1.0, 2.0])}
        result = InferencerWorker._tree_to_cpu_numpy(meta)
        assert isinstance(result["t"], np.ndarray)
        np.testing.assert_allclose(result["t"], np.array([1.0, 2.0]))

    def test_list_of_cpu_tensors(self):
        from cell_observatory_platform.inference.inferencer import InferencerWorker

        meta = {
            "sizes": [
                torch.tensor([64, 64]),
                torch.tensor([32, 32]),
            ]
        }
        result = InferencerWorker._tree_to_cpu_numpy(meta)
        assert all(isinstance(x, np.ndarray) for x in result["sizes"])

    def test_nested_list_of_tensors(self):
        from cell_observatory_platform.inference.inferencer import InferencerWorker

        meta = {"nested": [[torch.tensor(1.0), torch.tensor(2.0)], [torch.tensor(3.0)]]}
        result = InferencerWorker._tree_to_cpu_numpy(meta)
        assert isinstance(result["nested"][0][0], np.ndarray)
        assert isinstance(result["nested"][1][0], np.ndarray)

    def test_nested_dict(self):
        from cell_observatory_platform.inference.inferencer import InferencerWorker

        meta = {"inner": {"t": torch.tensor(0.5)}}
        result = InferencerWorker._tree_to_cpu_numpy(meta)
        assert isinstance(result["inner"]["t"], np.ndarray)
        np.testing.assert_allclose(result["inner"]["t"], np.array(0.5))

    def test_non_tensor_values_pass_through(self):
        from cell_observatory_platform.inference.inferencer import InferencerWorker

        meta = {"name": "tile_001.zarr", "idx": 42, "flag": True}
        result = InferencerWorker._tree_to_cpu_numpy(meta)
        assert result == {"name": "tile_001.zarr", "idx": 42, "flag": True}

    def test_tuple_preserved(self):
        from cell_observatory_platform.inference.inferencer import InferencerWorker

        meta = {"pair": (torch.tensor(1.0), torch.tensor(2.0))}
        result = InferencerWorker._tree_to_cpu_numpy(meta)
        assert isinstance(result["pair"], tuple)
        assert len(result["pair"]) == 2
        for x in result["pair"]:
            assert isinstance(x, np.ndarray)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_tensors_converted(self):
        from cell_observatory_platform.inference.inferencer import InferencerWorker

        meta = {"t": torch.tensor([1.0, 2.0], device="cuda")}
        result = InferencerWorker._tree_to_cpu_numpy(meta)
        assert isinstance(result["t"], np.ndarray)
        np.testing.assert_allclose(result["t"], np.array([1.0, 2.0]))


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
