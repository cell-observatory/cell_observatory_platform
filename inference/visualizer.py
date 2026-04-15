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
    save_predictions,
    unpack_batched_tensors,
    )
import ray
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def _infer_batch_size(metainfo: Dict[str, Any]) -> int:
    """Batch size from metainfo; scalar string fields imply B=1."""
    if "batch_size_actual" in metainfo:
        return int(metainfo["batch_size_actual"])
    if "prepared_id" in metainfo:
        pid = metainfo["prepared_id"]
        if isinstance(pid, np.ndarray):
            return int(pid.shape[0])
        return len(pid)
    tn = metainfo.get("tile_name")
    if isinstance(tn, str):
        return 1
    if isinstance(tn, (np.ndarray, list, tuple)):
        return len(tn)
    return 1


def _metainfo_scalar_at(metainfo: Dict[str, Any], key: str, b: int) -> Any:
    """Value for batch index ``b``, or the scalar if not batched."""
    v = metainfo.get(key)
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return v.item()
        return v[b]
    if isinstance(v, (list, tuple)):
        return v[b]
    return v


def _get_base_sample_name_for_index(metainfo: Dict[str, Any], b: int) -> str:
    try:
        folder = _metainfo_scalar_at(metainfo, "output_folder", b)
        if folder is None:
            raise KeyError
        base = str(folder).replace("/", "_")
    except (KeyError, IndexError):
        id_raw = metainfo.get("id", "unknown")
        if isinstance(id_raw, (np.ndarray, list, tuple)) and not isinstance(id_raw, (str, bytes)):
            try:
                idv = id_raw[b]
            except (IndexError, TypeError):
                idv = "unknown"
        else:
            idv = id_raw
        base = f"inference_roi{idv}"
    tn = _metainfo_scalar_at(metainfo, "tile_name", b)
    if tn is None:
        tn = "unknown"
    tile_name = str(tn)
    base_sample_name = base + "_" + tile_name
    return base_sample_name.replace(".zarr", "").replace(".tiff", "")


@ray.remote(namespace="visualizer", lifetime="detached", num_cpus=0)
class VizWorker:
    """
    Pure component that dispatches to visualization handlers based on output_type.viz.handler.
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

        self._metrics: Dict[str, float | int | List[float | bool]] = {
            "visualize_time_ms": [],
            "visualize_successful": [],
            "visualize_calls": 0.0,
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
        metainfo = inference_outputs["metainfo"]
        if "targets" in inference_outputs:
            targets = inference_outputs.pop("targets")
            targets_unpacked = unpack_batched_tensors(targets)
        else:
            targets_unpacked = None
        data_tensor = inference_outputs.pop("data_tensor")
        data_tensor_unpacked = [np.squeeze(chunk, axis=0) for chunk in np.split(data_tensor, data_tensor.shape[0], axis=0)]
        unpacked_inference_outputs = unpack_batched_tensors(inference_outputs, skip_keys={"metainfo"})
        bsz = len(unpacked_inference_outputs)
        if bsz == 0:
            return
        inferred = _infer_batch_size(metainfo)
        if inferred != bsz:
            raise ValueError(
                f"metainfo batch ({inferred}) != unpacked preds batch ({bsz})"
            )
        names = [_get_base_sample_name_for_index(metainfo, b) for b in range(bsz)]

        save_semantic_predictions(
            name=names[0],
            preds=unpacked_inference_outputs,
            targets=targets_unpacked,
            images=data_tensor_unpacked,
            save_dir=save_dir,
            names=names,
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
        bu = len(data_tensor_unpacked)
        if bu != len(regions) or bu != len(identifiers):
            raise ValueError(
                f"Batch mismatch: tensors={bu}, regions={len(regions)}, identifiers={len(identifiers)}"
            )
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
        metainfo = inference_outputs["metainfo"]
        save_tensors: List[str] = kwargs["save_tensors"]
        tensor_dict = {k: inference_outputs[k] for k in save_tensors}
        unpacked = unpack_batched_tensors(tensor_dict)
        bsz = _infer_batch_size(metainfo)
        if len(unpacked) != bsz:
            raise ValueError(
                f"metainfo batch ({bsz}) != unpacked tensor batch ({len(unpacked)})"
            )
        for b, preds_b in enumerate(unpacked):
            name_b = _get_base_sample_name_for_index(metainfo, b)
            save_predictions(
                name=name_b,
                predictions=preds_b,
                save_dir=save_dir,
                **kwargs,
            )

    def _handle_feature_viz(
        self,
        inference_outputs: dict[str, Any],
        save_dir: str,
        **kwargs: Dict[str, Any],
    ) -> None:
        metainfo = inference_outputs["metainfo"]
        feat_key = kwargs.get("feat_key")
        if feat_key is None:
            raise ValueError("feat_key must be specified for feature visualization")
        gt_key = kwargs.get("gt_key", "data_tensor")
        tensor_dict = {
            gt_key: inference_outputs[gt_key],
            feat_key: inference_outputs[feat_key],
        }
        unpacked = unpack_batched_tensors(tensor_dict)
        bsz = _infer_batch_size(metainfo)
        if len(unpacked) != bsz:
            raise ValueError(
                f"metainfo batch ({bsz}) != unpacked tensor batch ({len(unpacked)})"
            )
        for b, preds_b in enumerate(unpacked):
            name_b = _get_base_sample_name_for_index(metainfo, b)
            save_feature_visualizations(
                name=name_b,
                predictions=preds_b,
                save_dir=save_dir,
                **kwargs,
            )

    def _handle_bbox_overlay(
        self,
        inference_outputs: dict[str, Any],
        save_dir: str,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            "bbox_overlay is not supported via VizWorker.visualize(); "
            "use inference.utils.save_bbox_overlay with pred_boxes_xyzxyz, image, and save_dir."
        )

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

    def get_metrics(self) -> Dict[str, List[float | bool]]:
        return self._metrics.copy()

    def clear_metrics(self) -> None:
        self._metrics = {
            "visualize_time_ms": [],
            "visualize_successful": [],
            "visualize_calls": 0.0,
        }

    def visualize(
        self,
        inference_outputs: Dict[str, Any],
    ) -> None:
        """
        Dispatch to the appropriate handler based on output_type.viz.handler.
        """
        self._metrics["visualize_calls"] += 1.0
        slots_to_free = []
        start_time = time.perf_counter()
        futures = []
        try:
            for name, slot_info in inference_outputs.items():
                if name == "metainfo":
                    continue
                if isinstance(slot_info, np.ndarray):
                    ray.logger.warning(f"Inference output being passed through the ray plasma store: {name}")
                    continue
                output_array = self.buffer_manager.slot_info_to_view(slot_info)
                inference_outputs[name] = output_array
                slots_to_free.append(slot_info)
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
                    self._metrics["visualize_successful"].append(True)
                except Exception as e:
                    ray.logger.error(f"Failed to execute viz handler: {e}", exc_info=True)
                    self._metrics["visualize_successful"].append(False)
                    continue
        except Exception as e:
            ray.logger.error(f"Failed to visualize: {e}", exc_info=True)
        finally:
            self._metrics["visualize_time_ms"].append((time.perf_counter() - start_time) * 1000)
            for slot_info in slots_to_free:
                self.buffer_manager.free_slot(slot_info)