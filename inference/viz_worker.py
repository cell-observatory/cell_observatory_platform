"""
VizWorker: per-rank thread that consumes viz_output slots and invokes InferenceVisualizer.

Best-effort: if no slot available, InferencerWorker skips viz (no backpressure).
"""

from __future__ import annotations

import io
import logging
import time
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, Optional

import numpy as np

from cell_observatory_platform.inference.buffer_manager import BufferManager, SlotHandle
from cell_observatory_platform.inference.visualizer import InferenceVisualizer

logger = logging.getLogger(__name__)


class VizWorker:
    """
    Worker that pops (slot_handle, metadata) from viz queue, deserializes artifact,
    calls InferenceVisualizer, then buffer_manager.free("viz_output", handle).
    """

    def __init__(
        self,
        queue: Queue,
        buffer_manager: BufferManager,
        output_type_configs: Dict[str, Any],
        run_id: str,
        metrics_queue: Optional[Queue] = None,
        total_rendered_ref: Optional[list] = None,
        total_rendered_lock: Optional[Lock] = None,
    ):
        self.queue = queue
        self.buffer_manager = buffer_manager
        self.output_type_configs = output_type_configs
        self.run_id = run_id
        self.metrics_queue = metrics_queue or Queue()
        self._total_rendered_ref = total_rendered_ref or [0]
        self.total_rendered_lock = total_rendered_lock or Lock()
        self._visualizer = InferenceVisualizer()
        self._thread: Optional[Thread] = None
        self._stop = False

    def _run_loop(self) -> None:
        from multiprocessing import shared_memory

        while not self._stop:
            try:
                item = self.queue.get(timeout=0.5)
            except Empty:
                continue
            if item is None:
                break
            slot_handle, metadata = item
            t0 = time.perf_counter()
            try:
                # Read artifact from slot
                shm = shared_memory.SharedMemory(name=slot_handle.shm_name)
                try:
                    raw = bytes(
                        shm.buf[
                            slot_handle.slot_idx
                            * slot_handle.slot_bytes : (slot_handle.slot_idx + 1)
                            * slot_handle.slot_bytes
                        ]
                    )
                finally:
                    shm.close()

                # Deserialize: metadata has shape, dtype
                shape = metadata.get("artifact_shape")
                dtype = metadata.get("artifact_dtype", "float32")
                if shape is not None:
                    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
                    arr = np.frombuffer(raw[:n], dtype=dtype).reshape(shape).copy()
                else:
                    arr = np.load(io.BytesIO(raw), allow_pickle=False)

                output_name = metadata.get("output_name")
                output_type_cfg = metadata.get("output_type_cfg") or self.output_type_configs.get(
                    metadata.get("output_type", ""), {}
                )
                context = metadata.get("context", {})
                context.setdefault("save_dir", metadata.get("save_dir", "."))
                context.setdefault("identifier", metadata.get("identifier", output_name))

                self._visualizer.visualize(
                    output_name=output_name,
                    output_type_cfg=output_type_cfg,
                    data=arr,
                    context=context,
                )

                with self.total_rendered_lock:
                    self._total_rendered_ref[0] += 1
            except Exception as e:
                logger.warning("VizWorker failed for %s: %s", metadata.get("output_name"), e)
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self.metrics_queue.put(
                    {"viz_time_ms": elapsed_ms, "slots_rendered": 1}
                )
                self.buffer_manager.free("viz_output", slot_handle)

    def start(self) -> None:
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread:
            self.queue.put(None)
            self._thread.join(timeout=5.0)


class VizWorkerManager:
    """Manages VizWorker threads and lifecycle."""

    def __init__(
        self,
        num_workers: int,
        queue: Queue,
        buffer_manager: BufferManager,
        output_type_configs: Dict[str, Any],
        run_id: str,
    ):
        self.num_workers = num_workers
        self.queue = queue
        self.buffer_manager = buffer_manager
        self.output_type_configs = output_type_configs
        self.run_id = run_id
        self._total_rendered: list = [0]
        self.total_rendered_lock = Lock()
        self.metrics_queue: Queue = Queue()

        self._workers = [
            VizWorker(
                queue=queue,
                buffer_manager=buffer_manager,
                output_type_configs=output_type_configs,
                run_id=run_id,
                metrics_queue=self.metrics_queue,
                total_rendered_ref=self._total_rendered,
                total_rendered_lock=self.total_rendered_lock,
            )
            for _ in range(num_workers)
        ]

    def start(self) -> None:
        for w in self._workers:
            w.start()

    def stop(self) -> None:
        for _ in self._workers:
            self.queue.put(None)
        for w in self._workers:
            w.stop()

    def get_summary(self) -> Dict[str, Any]:
        with self.total_rendered_lock:
            tiles_visualized = self._total_rendered[0]
        metrics = self.buffer_manager.get_metrics()
        viz_metrics = metrics.get("viz_output", {})
        return {
            "tiles_visualized": tiles_visualized,
            "slots_dropped": viz_metrics.get("drops", 0),
        }
