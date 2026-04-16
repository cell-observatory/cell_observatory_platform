"""Optional benchmarks: SHM-buffer vs Ray-serialized mask transfer via InferencerWorker.

Run with::

    pytest tests/inference/test_transfer_benchmark.py -v -s

These tests require CUDA and are gated behind ``@pytest.mark.benchmark``.
They exercise the real ``InferencerWorker.predict()`` path, comparing:

  * **buffer path** – masks listed in ``buffer_tensors`` → async ``cudaMemcpyAsync``
    into a pinned SHM slot → lightweight ``slot_info`` dict sent via Ray.
  * **ray path** – masks *not* in ``buffer_tensors`` → ``tensor.cpu().numpy()`` →
    full numpy array sent through Ray object store.

As spatial size grows the buffer path should increasingly dominate because
it avoids pageable memcpy, Ray serialisation, and plasma object-store copies.
"""
from __future__ import annotations

import statistics
import time
import uuid
from typing import Dict, List, Tuple
from unittest.mock import patch

import numpy as np
import pytest
import ray
import torch
import torch.nn as nn

from tests.ray_init_helpers import init_ray_like_training

_CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")

SPATIAL_SIZES: List[Tuple[int, int, int]] = [
    (128, 128, 128),
    (128, 256, 512),
    (256, 1024, 2048),
]

_TOPK = 5
_BATCH_SIZE = 1
_INPUT_FORMAT = "ZYXC"
_PATCH_SHAPE = (4, 4)
WARMUP_ITERS = 3
BENCH_ITERS = 10
BUFFER_CAPACITY = 4


# ---------------------------------------------------------------------------
# Mock model (spatial-size parameterised)
# ---------------------------------------------------------------------------

class _BenchInstanceSegModel(nn.Module):
    def __init__(self, spatial: Tuple[int, ...], topk: int, device: torch.device):
        super().__init__()
        self._spatial = spatial
        self._topk = topk
        self._device = device
        self.output_metadata = {
            "tensor_info": {
                "masks": {"shape": spatial, "dtype": "uint16"},
                "boxes": {"shape": (topk, 6), "dtype": "float32"},
                "labels": {"shape": (topk,), "dtype": "float32"},
            },
        }

    def predict(self, data_sample: dict) -> Dict[str, torch.Tensor]:
        B = data_sample["data_tensor"].shape[0]
        return {
            "masks": torch.ones((B, *self._spatial), dtype=torch.uint16, device=self._device),
            "boxes": torch.full((B, self._topk, 6), 0.5, dtype=torch.float32, device=self._device),
            "labels": torch.full((B, self._topk), 0.9, dtype=torch.float32, device=self._device),
        }

    def validate_outputs(self, preds: Dict[str, torch.Tensor]) -> None:
        pass


# ---------------------------------------------------------------------------
# Stub save / viz workers (same as unit tests)
# ---------------------------------------------------------------------------

@ray.remote(namespace="saver", num_cpus=0)
class _StubSaveWorker:
    """Mirror ``SaveWorker`` metrics: list batches, snapshot + clear in ``get_metrics``."""

    def __init__(self, buffer_manager):
        self._bm = buffer_manager
        self._metrics: Dict[str, List] = {
            "save_time_ms": [],
            "save_successful": [],
        }

    def save(self, inference_outputs: dict) -> None:
        t0 = time.perf_counter()
        ok = False
        try:
            for key, val in inference_outputs.items():
                if key == "metainfo":
                    continue
                if isinstance(val, dict) and "actor_name" in val:
                    self._bm.slot_info_to_view(val)
                    self._bm.free_slot(val)
            ok = True
        except Exception:
            ok = False
        finally:
            self._metrics["save_time_ms"].append((time.perf_counter() - t0) * 1000)
            self._metrics["save_successful"].append(ok)

    def get_metrics(self) -> dict:
        out = {
            "save_time_ms": self._metrics["save_time_ms"].copy(),
            "save_successful": self._metrics["save_successful"].copy(),
        }
        self._metrics = {"save_time_ms": [], "save_successful": []}
        return out


@ray.remote(namespace="visualizer", num_cpus=0)
class _StubVizWorker:
    def __init__(self, buffer_manager):
        self._bm = buffer_manager
        self._metrics: Dict[str, float | List] = {
            "visualize_time_ms": [],
            "visualize_successful": [],
            "visualize_calls": 0.0,
        }

    def visualize(self, inference_outputs: dict) -> None:
        self._metrics["visualize_calls"] += 1.0
        t0 = time.perf_counter()
        ok = False
        try:
            for key, val in inference_outputs.items():
                if key == "metainfo":
                    continue
                if isinstance(val, dict) and "actor_name" in val:
                    self._bm.slot_info_to_view(val)
                    self._bm.free_slot(val)
            ok = True
        except Exception:
            ok = False
        finally:
            self._metrics["visualize_time_ms"].append((time.perf_counter() - t0) * 1000)
            self._metrics["visualize_successful"].append(ok)

    def get_metrics(self) -> dict:
        out = {
            "visualize_time_ms": self._metrics["visualize_time_ms"].copy(),
            "visualize_successful": self._metrics["visualize_successful"].copy(),
            "visualize_calls": self._metrics["visualize_calls"],
        }
        self._metrics = {
            "visualize_time_ms": [],
            "visualize_successful": [],
            "visualize_calls": 0.0,
        }
        return out

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ray_ctx():
    init_ray_like_training(num_cpus=4, num_gpus=0)
    yield
    ray.shutdown()


@pytest.fixture
def ray_node_id(ray_ctx):
    return ray.nodes()[0]["NodeID"]


@pytest.fixture
def unique_suffix():
    return uuid.uuid4().hex[:8]


def _make_buffer_manager(ray_node_id, **kwargs):
    from cell_observatory_platform.data.datasets.buffers import BufferManager
    kw = dict(
        local_rank=0,
        global_rank=0,
        node_id=ray_node_id,
        numa_node=0,
        rank_memory_budget_gb=16.0,
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


def _patch_context():
    class _MultiPatch:
        def __init__(self, *patches):
            self._patches = patches
        def __enter__(self):
            for p in self._patches:
                p.__enter__()
            return self
        def __exit__(self, *exc):
            for p in reversed(self._patches):
                p.__exit__(*exc)

    return _MultiPatch(
        patch("cell_observatory_platform.inference.inferencer.process_rank", return_value=0),
        patch("cell_observatory_platform.inference.inferencer.get_world_size", return_value=1),
        patch("cell_observatory_platform.inference.inferencer.barrier"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_outputs_metadata(spatial: Tuple[int, ...], *, use_buffer: bool):
    """Build outputs_metadata; ``use_buffer`` controls whether masks use SHM."""
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
        "visualize_tensors": [],
        "buffer_tensors": ["masks"] if use_buffer else [],
        "tensor_info": {
            "masks": {"shape": spatial, "dtype": "uint16"},
            "boxes": {"shape": (_TOPK, 6), "dtype": "float32"},
            "labels": {"shape": (_TOPK,), "dtype": "float32"},
        },
        "save_tensors_dtypes": {
            "masks": "uint16",
            "boxes": "float32",
            "labels": "float32",
        },
    }


def _register_buffers(bm, outputs_metadata: dict, batch_size: int = _BATCH_SIZE) -> List:
    actors = []
    for name in outputs_metadata.get("buffer_tensors", []):
        info = outputs_metadata["tensor_info"][name]
        if name in outputs_metadata["save_tensors"]:
            actor, _ = bm.set_buffer(
                pool_name=f"{name}_save",
                batch_size=batch_size,
                input_shape=tuple(info["shape"]),
                dtype=info["dtype"],
                buffer_type="host_memory",
                buffer_capacity=BUFFER_CAPACITY,
                pin_numa_node=False,
            )
            actors.append(actor)
    return actors


def _make_data_sample(spatial: Tuple[int, ...], device: torch.device) -> dict:
    input_shape = (*spatial, 1)
    data_tensor = torch.randn(_BATCH_SIZE, *input_shape, device=device, dtype=torch.float32)
    return {
        "data_tensor": data_tensor,
        "metainfo": {
            "prepared_id": [0],
            "tile_name": "bench_tile.zarr",
            "orig_image_sizes": [torch.tensor(spatial, device=device)],
        },
    }


def _build_worker(
    *,
    spatial: Tuple[int, ...],
    model: nn.Module,
    buffer_manager,
    save_worker,
    viz_worker,
    outputs_metadata: dict,
):
    from cell_observatory_platform.inference.inferencer import InferencerWorker
    input_shape = [*spatial, 1]
    return InferencerWorker(
        aggregate_mode="none",
        inference_mode="tile",
        task="instance_segmentation",
        outputs_metadata=outputs_metadata,
        input_format=_INPUT_FORMAT,
        input_shape=input_shape,
        patch_shape=list(_PATCH_SHAPE),
        decoder_head_type="maskdino",
        model_name="bench__run_x__e0_i0",
        save_outputs=True,
        block_on_save=True,
        vizualize_outputs=False,
        block_on_viz=False,
        model=model,
        buffer_manager=buffer_manager,
        save_worker=save_worker,
        viz_worker=viz_worker,
        channel_names={0: "benchmark_ch0"},
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _run_predict_n(worker, spatial, device, n):
    """Run predict() n times, return list of wall-clock ms per call."""
    times = []
    for _ in range(n):
        ds = _make_data_sample(spatial, device)
        t0 = time.perf_counter()
        worker.predict(ds)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


@requires_cuda
@pytest.mark.benchmark
class TestTransferBenchmark:
    """Compare InferencerWorker.predict() with and without SHM buffers for masks."""

    @pytest.mark.parametrize(
        "spatial",
        SPATIAL_SIZES,
        ids=[f"{'x'.join(str(d) for d in s)}" for s in SPATIAL_SIZES],
    )
    def test_buffer_vs_ray_transfer(
        self, ray_ctx, ray_node_id, unique_suffix, spatial
    ):
        device = torch.device("cuda:0")
        nbytes = int(np.prod(spatial)) * _BATCH_SIZE * np.dtype(np.uint16).itemsize
        mbytes = nbytes / (1024 * 1024)

        results: Dict[str, dict] = {}

        for label, use_buffer in [("buffer", True), ("ray", False)]:
            tag = f"{label}_{unique_suffix}"
            bm = _make_buffer_manager(
                ray_node_id,
                global_rank=hash(tag) % 100_000,
            )
            actors: list = []
            sw, vw = None, None
            try:
                meta = _make_outputs_metadata(spatial, use_buffer=use_buffer)
                actors = _register_buffers(bm, meta)

                sw = _StubSaveWorker.options(name=f"sw_{tag}").remote(buffer_manager=bm)
                vw = _StubVizWorker.options(name=f"vw_{tag}").remote(buffer_manager=bm)

                model = _BenchInstanceSegModel(spatial, _TOPK, device)
                with _patch_context():
                    worker = _build_worker(
                        spatial=spatial,
                        model=model,
                        buffer_manager=bm,
                        save_worker=sw,
                        viz_worker=vw,
                        outputs_metadata=meta,
                    )

                # warmup
                _run_predict_n(worker, spatial, device, WARMUP_ITERS)
                ray.get(sw.get_metrics.remote())
                ray.get(vw.get_metrics.remote())

                # timed
                times = _run_predict_n(worker, spatial, device, BENCH_ITERS)

                save_metrics = ray.get(sw.get_metrics.remote())
                assert len(save_metrics["save_successful"]) == BENCH_ITERS, (
                    f"{label}: expected {BENCH_ITERS} successful saves, "
                    f"got {save_metrics}"
                )
                assert save_metrics["save_successful"].count(True) == BENCH_ITERS

                results[label] = {
                    "median_ms": statistics.median(times),
                    "p90_ms": sorted(times)[int(0.9 * len(times))],
                    "min_ms": min(times),
                }
            finally:
                _kill_safe(sw)
                _kill_safe(vw)
                for a in actors:
                    _kill_safe(a)
                bm.shutdown()

        buf = results["buffer"]
        ray_r = results["ray"]
        speedup = ray_r["median_ms"] / buf["median_ms"] if buf["median_ms"] > 0 else float("inf")

        print(
            f"\n  spatial={spatial}  mask_size={mbytes:.1f} MB"
            f"\n    buffer : median={buf['median_ms']:.2f} ms  "
            f"p90={buf['p90_ms']:.2f} ms  min={buf['min_ms']:.2f} ms"
            f"\n    ray    : median={ray_r['median_ms']:.2f} ms  "
            f"p90={ray_r['p90_ms']:.2f} ms  min={ray_r['min_ms']:.2f} ms"
            f"\n    speedup: {speedup:.2f}x  (buffer median / ray median)"
        )
