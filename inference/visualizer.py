"""
InferenceVisualizer: pure component that turns model outputs into human-consumable visual artifacts.

Handler dispatch is driven by output_type.viz.handler. Used by VizWorkers (Phase 3+) and post-hoc tools.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, List, Tuple

import numpy as np
from cell_observatory_platform.data.datasets.buffers import BufferManager
from cell_observatory_platform.inference.utils import (
    save_semantic_predictions,
    save_feature_visualizations,
    save_instance_predictions,
    save_bbox_overlay,
    save_predictions,
    unpack_batched_tensors,
    )
import ray
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


@ray.remote(namespace="visualizer", lifetime="detached", num_cpus=0)
class VizWorker:
    """
    Pure component that dispatches to visualization handlers based on output_type.viz.handler.
    """

    def __init__(
        self, 
        buffer_manager: BufferManager,
        output_dir: str | Path,
        handler_configs: Dict[str, Any],
        max_workers: int = 4,
    ) -> None:
        self.buffer_manager = buffer_manager
        self.global_rank = buffer_manager.global_rank
        self.handler_configs = handler_configs
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._handlers: Dict[str, Callable[..., None]] = {}
        self._register_default_handlers()
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"visualizer_worker_rank_{buffer_manager.global_rank}"
        )
        for handler_name, kwargs in self.handler_configs.items():
            if handler_name not in self._handlers:
                raise ValueError(
                    f"Unknown viz.handler: {handler_name}. Registered: {list(self._handlers.keys())}"
                )

        self._metrics: Dict[str, float | int] = {
            "visualize_time_ms": 0.0,
            "visualize_calls": 0.0,
            "visualize_successes": 0.0,
            "visualize_failures": 0.0,
        }

    def _register_default_handlers(self) -> None:
        """Register built-in handlers for viz.handler names."""
        self._handlers["semantic_map"] = self._handle_semantic_map
        self._handlers["instance_overlay"] = self._handle_instance_overlay
        self._handlers["feature_viz"] = self._handle_feature_viz
        self._handlers["save_predictions"] = self._handle_save_predictions
        self._handlers["bbox_overlay"] = self._handle_bbox_overlay

    def _handle_semantic_map(
        self,
        inference_outputs: dict[str, Any],
        save_dir: str,
        **kwargs: Any,
    ) -> None:
        base_sample_name = self._get_base_sample_name(inference_outputs)
        targets = inference_outputs.pop("targets")
        targets_unpacked = unpack_batched_tensors(targets)
        data_tensor = inference_outputs.pop("data_tensor")
        data_tensor_unpacked = list(data_tensor.unbind(0))
        unpacked_inference_outputs = unpack_batched_tensors(inference_outputs, skip_keys={"metainfo"})

        save_semantic_predictions(
            name=base_sample_name,
            preds=unpacked_inference_outputs,
            images=data_tensor_unpacked,
            save_dir=save_dir,
            **kwargs,
        )

    def _handle_instance_overlay(
        self,
        inference_outputs: dict[str, Any],
        save_dir: str,
        **kwargs: Any,
    ) -> None:

        regions, identifiers = self._prepare_regions_and_identifiers(inference_outputs["metainfo"])

        # Per-batch unpacking: _unpack_batch(inference_outputs, skip_keys={"metainfo"})
        # -> List[Dict[str, Tensor ZYX]] (len = B) for per-sample handling
        targets = inference_outputs.pop("targets")
        targets_unpacked = unpack_batched_tensors(targets)
        data_tensor = inference_outputs.pop("data_tensor")
        data_tensor_unpacked = list(data_tensor.unbind(0))
        unpacked_inference_outputs = unpack_batched_tensors(inference_outputs, skip_keys={"metainfo"})
        save_instance_predictions(
            save_dir=save_dir,
            images=data_tensor_unpacked,
            targets=targets_unpacked,
            preds=unpacked_inference_outputs,
            identifiers=identifiers,
            regions=regions,
            **kwargs,
        )
    
    def _handle_save_predictions(
        self,
        inference_outputs: dict[str, Any],
        save_dir: str,
        **kwargs: Any,
    ) -> None:
        base_sample_name = self._get_base_sample_name(inference_outputs)
        save_predictions(
            name=base_sample_name,
            predictions=inference_outputs,
            save_dir=save_dir,
            **kwargs,
        )

    def _handle_feature_viz(
        self,
        inference_outputs: dict[str, Any],
        save_dir: str,
        **kwargs: Dict[str, Any],
    ) -> None:

        base_sample_name = self._get_base_sample_name(inference_outputs)
        save_feature_visualizations(
            name=base_sample_name,
            predictions=inference_outputs,
            save_dir=save_dir,
            **kwargs,
        )

    def _handle_bbox_overlay(
        self,
        output_name: str,
        output_type_cfg: Dict[str, Any],
        data: np.ndarray,
        context: Dict[str, Any],
        **kwargs: Any,
    ) -> None:

        image = context.get("image")
        save_dir = context.get("save_dir", kwargs.get("save_dir", "."))
        identifier = context.get("identifier") or output_name
        if isinstance(identifier, (list, tuple)):
            identifier = identifier[0] if identifier else output_name
        if image is None:
            raise ValueError("bbox_overlay handler requires context['image']")
        save_bbox_overlay(
            pred_boxes_xyzxyz=data,
            image=image,
            save_dir=save_dir,
            identifier=str(identifier),
            **{k: v for k, v in context.items() if k in ("z_step", "pmin", "pmax")},
        )


    def _get_base_sample_name(self, inference_outputs: dict[str, Any]) -> str:
        try:
            base = str(inference_outputs["metainfo"]["output_folder"])
        except KeyError:
            base = f"inference_roi{inference_outputs.get('metainfo', {}).get('id', 'unknown')}"
        base_sample_name = base.replace("/", "_") + "_" + inference_outputs["metainfo"]["tile_name"]
        base_sample_name = base_sample_name.replace(".zarr", "").replace(".tiff", "")
        return base_sample_name

    def _prepare_regions_and_identifiers(self, metadata: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        B = len(metadata["prepared_id"])

        # rank_dir = Path(self.inference_save_dir) / f"rank{self.rank:03d}"
        # os.makedirs(rank_dir, exist_ok=True)

        regions: List[Dict[str, Any]] = []
        identifiers: List[str] = []

        for b in range(B):
            roi = int(metadata["prepared_id"][b])
            tile_nm = str(metadata["tile_name"][b])

            t0 = int(metadata["time_start"][b])
            T = int(metadata["time_size"][b])
            t1 = t0 + T

            z0 = int(metadata["z_start"][b])
            sz = int(metadata["z_size"][b])
            z1 = z0 + sz
            y0 = int(metadata["y_start"][b])
            sy = int(metadata["y_size"][b])
            y1 = y0 + sy
            x0 = int(metadata["x_start"][b])
            sx = int(metadata["x_size"][b])
            x1 = x0 + sx

            region = dict(
                roi=roi,
                tile_name=tile_nm,
                coords=(t0, t1, z0, z1, y0, y1, x0, x1),
                coord_frame="voxel",
            )
            ident = (
                f"rank{self.global_rank:03d}_roi{roi}_{tile_nm}"
                f"_t{t0}-{t1}_z{z0}-{z1}_y{y0}-{y1}_x{x0}-{x1}"
            )

            regions.append(region)
            identifiers.append(ident)
        
        return regions, identifiers

    def get_metrics(self) -> Dict[str, float | int]:
        return self._metrics.copy()

    def clear_metrics(self) -> None:
        self._metrics = {
            "visualize_time_ms": 0.0,
            "visualize_calls": 0.0,
            "visualize_successes": 0.0,
            "visualize_failures": 0.0,
        }

    def visualize(
        self,
        inference_outputs: Dict[str, Any],
    ) -> None:
        """
        Dispatch to the appropriate handler based on output_type.viz.handler.
        """
        slots_to_free = []
        start_time = time.perf_counter()
        try:
            for name, slot_info in inference_outputs.items():
                if name == "metainfo":
                    continue
                if isinstance(slot_info, np.ndarray):
                    continue
                output_array = self.buffer_manager.slot_info_to_view(slot_info)
                inference_outputs[name] = output_array
                slots_to_free.append(slot_info)
            futures = []
            for handler_name, kwargs in self.handler_configs.items():
                futures.append(
                    self.thread_pool.submit(
                        self._handlers[handler_name],
                        inference_outputs=inference_outputs,
                        save_dir=self.output_dir,
                        **kwargs,
                    )
                )

            for future in as_completed(futures):
                try:
                    future.result()
                    self._metrics["visualize_successes"] += 1
                except Exception as e:
                    ray.logger.error(f"Failed to execute viz handler: {e}", exc_info=True)
                    self._metrics["visualize_failures"] += 1
                    continue
        except Exception as e:
                ray.logger.error(f"Failed to visualize: {e}", exc_info=True)
        finally:
            self._metrics["visualize_time_ms"] += (time.perf_counter() - start_time) * 1000
            self._metrics["visualize_calls"] += 1
            for slot_info in slots_to_free:
                self.buffer_manager.free_slot(slot_info)