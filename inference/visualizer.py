"""
VizWorker: turns model outputs into human-consumable visual artifacts.

Dispatch is driven by ``handler_configs`` (``viz.handler`` names). The worker materializes
the batched SHM views, builds uniform per-sample :class:`InferenceRecord`s once
(batch-first, cropped to each sample's original tile size like the save path), then
hands each ``(handler, record)`` to the thread pool. Instance-mask normalization is
per-handler: ``instance_overlay`` normalizes per-object stacks; instance label maps
pass through untouched and are rendered natively (never exploded per object).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

from cell_observatory_platform.data.data_types import OutputKind
from cell_observatory_platform.data.datasets.buffers import BufferManager
from cell_observatory_platform.inference.inference_postprocess import (
    InferenceRecord,
    build_records,
    to_instance_stack,
    viz_identifier,
)
from cell_observatory_platform.inference.utils import (
    save_bbox_overlay,
    save_semantic_predictions,
    save_feature_visualizations,
    save_instance_predictions,
    save_prediction_plots,
)

from cell_observatory_platform.utils.registry import REGISTRY


# --- viz handlers: free functions (registry-dispatched). Read the record's declared
# --- fields + the explicit `global_rank`, then call a plotter. ---

@REGISTRY.register("viz_handler", "semantic_map")
def semantic_map_handler(record: InferenceRecord, save_dir: str, *, global_rank: int, **kwargs: Any) -> None:
    save_semantic_predictions(
        name=viz_identifier(record, global_rank),
        preds=record.preds, image=record.image, targets=record.targets,
        save_dir=save_dir, **kwargs,
    )


@REGISTRY.register("viz_handler", "instance_overlay")
def instance_overlay_handler(
    record: InferenceRecord, save_dir: str, *, global_rank: int, **kwargs: Any,
) -> None:
    # INSTANCE_STACK: normalize to the (N,1,Z,Y,X) stack plotting expects.
    # INSTANCE_LABEL_MAP: pass the integer volume through untouched -- the
    # renderer slices it directly (O(volume) memory, independent of instance
    # count), dispatched off ``kinds`` inside save_instance_predictions.
    preds = {
        name: (
            to_instance_stack(arr)
            if record.kinds.get(name) == OutputKind.INSTANCE_STACK.value
            else arr
        )
        for name, arr in record.preds.items()
    }
    save_instance_predictions(
        save_dir=save_dir, identifier=viz_identifier(record, global_rank),
        image=record.image, preds=preds, targets=record.targets,
        region=record.region, kinds=record.kinds, **kwargs,
    )


@REGISTRY.register("viz_handler", "save_predictions")
def save_predictions_handler(record: InferenceRecord, save_dir: str, *, global_rank: int, **kwargs: Any) -> None:
    # Handler name is config-facing and stays "save_predictions".
    save_prediction_plots(
        name=viz_identifier(record, global_rank),
        predictions=record.preds, save_dir=save_dir, **kwargs,
    )


@REGISTRY.register("viz_handler", "feature_viz")
def feature_viz_handler(record: InferenceRecord, save_dir: str, *, global_rank: int, **kwargs: Any) -> None:
    if kwargs.get("feat_key") is None:
        raise ValueError("feat_key must be specified for feature visualization")
    # feature viz reads the GT image (gt_key, default "data_tensor") from predictions.
    predictions = {**record.preds, "data_tensor": record.image}
    save_feature_visualizations(
        name=viz_identifier(record, global_rank),
        predictions=predictions, save_dir=save_dir, **kwargs,
    )


@REGISTRY.register("viz_handler", "bbox_overlay")
def bbox_overlay_handler(
    record: InferenceRecord, save_dir: str, *, global_rank: int,
    pred_boxes_key: str, background_channel: int = 0, z_step: int = 10,
    pmin: float = 1.0, pmax: float = 99.0, **kwargs: Any,
) -> None:
    save_bbox_overlay(
        pred_boxes_xyzxyz=record.preds[pred_boxes_key], image=record.image,
        save_dir=save_dir, identifier=viz_identifier(record, global_rank),
        z_step=z_step, pmin=pmin, pmax=pmax, background_channel=background_channel,
    )


class VizWorkerBase:
    """Dispatches visualization handlers over uniform per-sample records.

    Plain class holding all logic; ``VizWorker`` below is its Ray-actor wrapper
    (kept separate so unit tests can exercise the logic in-process).
    """

    def __init__(
        self,
        buffer_manager: BufferManager,
        output_dir: str | Path,
        handler_configs: Dict[str, Dict[str, Any]],
        max_workers: int = 4,
    ) -> None:
        self.buffer_manager = buffer_manager
        self.global_rank = buffer_manager.global_rank
        self.handler_configs = handler_configs
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"visualizer_worker_rank_{buffer_manager.global_rank}",
        )
        for handler_name in self.handler_configs:
            if not REGISTRY.has("viz_handler", handler_name):
                raise ValueError(
                    f"Unknown viz.handler: {handler_name}. "
                    f"Registered: {sorted(REGISTRY.names('viz_handler'))}"
                )

        self._metrics: Dict[str, float | int | List[float | bool]] = {
            "visualize_time_ms": [],
            "visualize_successful": [],
            "queue_time_ms": [],
            "visualize_calls": 0.0,
        }

    def get_handler_names(self) -> List[str]:
        """Configured viz handler names (used by InferencerWorker's startup
        validation of the data_tensor transport)."""
        return list(self.handler_configs)

    def ping(self) -> bool:
        """Side-effect-free liveness probe (see saver.SaveWorkerBase.ping)."""
        return True

    def get_metrics(self) -> Dict[str, List[float | bool]]:
        metrics = self._metrics.copy()
        self._metrics = {
            "visualize_time_ms": [],
            "visualize_successful": [],
            "queue_time_ms": [],
            "visualize_calls": 0.0,
        }
        return metrics

    def visualize(
        self,
        inference_outputs: Dict[str, Any],
        queue_t0: Optional[float] = None,
    ) -> None:
        """Materialize SHM views, build per-sample records once, dispatch each
        ``(handler, record)`` to the thread pool.

        Per-record handler failures are tolerated (logged + recorded as a failed
        metric); STRUCTURAL failures (slot resolution, record building, handler
        lookup) re-raise so ``InferencerWorker.finalize()`` sees them.
        """
        if queue_t0 is not None:
            self._metrics["queue_time_ms"].append((time.perf_counter() - queue_t0) * 1000)
        self._metrics["visualize_calls"] += 1.0
        slots_to_free = []
        start_time = time.perf_counter()
        futures = []
        try:
            output_arrays: Dict[str, Any] = {}
            # Claim EVERY slot for freeing BEFORE resolving any view (see
            # saver.py: a mid-loop resolution failure must not leak the
            # failing slot or the ones after it).
            for name, slot_info in inference_outputs.items():
                if name in ("metainfo", "targets") or isinstance(slot_info, np.ndarray):
                    continue
                slots_to_free.append(slot_info)
            for name, slot_info in inference_outputs.items():
                if name in ("metainfo", "targets"):
                    continue
                if isinstance(slot_info, np.ndarray):
                    ray.logger.warning(f"Inference output being passed through the ray plasma store: {name}")
                    output_arrays[name] = slot_info
                    continue
                output_arrays[name] = self.buffer_manager.slot_info_to_view(slot_info)

            metainfo = inference_outputs["metainfo"]
            # Viz tensors are pre-restore, uniform at the model's working
            # resolution -- nothing to crop (the processed frame).
            records = build_records(
                output_arrays,
                metainfo,
                columns=("tile_relative_path", "tile_name"),
                image_key="data_tensor",
                targets=inference_outputs.get("targets"),
            )

            # by_tile policy matched only a subset of the batch: render just
            # those records (the gate is batch-granular; this is per-sample).
            # Out-of-range indices (padded batch slots) are dropped -- the
            # producer clamps to batch_size_actual, this is defense in depth.
            viz_idx = metainfo.get("viz_sample_idx")
            if viz_idx is not None:
                records = [
                    records[int(i)] for i in np.asarray(viz_idx).ravel()
                    if int(i) < len(records)
                ]

            for handler_name, kwargs in self.handler_configs.items():
                handler = REGISTRY.get("viz_handler", handler_name).factory
                for record in records:
                    futures.append(
                        self.thread_pool.submit(handler, record=record, save_dir=self.output_dir, global_rank=self.global_rank, **kwargs)
                    )

            for future in as_completed(futures):
                try:
                    future.result()
                    self._metrics["visualize_successful"].append(True)
                except Exception as e:
                    ray.logger.error(f"Failed to execute viz handler: {e}", exc_info=True)
                    self._metrics["visualize_successful"].append(False)
                    continue
        except Exception as e:
            ray.logger.error(f"Failed to visualize: {e}", exc_info=True)
            raise  # structural failure: finalize() must see it
        finally:
            self._metrics["visualize_time_ms"].append((time.perf_counter() - start_time) * 1000)
            # records hold numpy VIEWS into the SHM slots: never free a slot
            # while a submitted handler may still be reading it.
            wait(futures)
            for slot_info in slots_to_free:
                self.buffer_manager.free_slot(slot_info)


@ray.remote(namespace="visualizer", lifetime="detached", num_cpus=1)
class VizWorker(VizWorkerBase):
    """Ray-actor wrapper around :class:`VizWorkerBase` (see its docstring)."""
