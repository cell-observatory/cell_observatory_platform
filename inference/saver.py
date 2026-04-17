from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import ray

from cell_observatory_platform.data.datasets.buffers import BufferManager
from cell_observatory_platform.data.io import (
    save_annotations_metadata,
    save_dense_annotations,
    save_sparse_annotations,
)

def input_format_to_output_format(
    input_format: Literal["TZYXC", "ZYXC"],
    task: Literal["instance_segmentation", "semantic_segmentation", "detection"],
) -> Dict[str, Literal["TZYXC", "ZYXC", "TN6", "N6", "TN", "N"]]:
    """
    Convert input format to output format.

    Here, semantically:
    - T is time
    - Z is depth
    - Y is height
    - X is width
    - C is channels
    - N is number of objects / queries
    - M is number of classes

    """

    output_format = {}
    if task == "instance_segmentation":
        if input_format == "TZYXC":
            output_format["masks"] = "TZYXC"
            output_format["scores"] = "TN"
            output_format["labels"] = "TNM"
            output_format["boxes"] = "TN6"
        elif input_format == "ZYXC":
            output_format["masks"] = "ZYXC"
            output_format["scores"] = "N"
            output_format["labels"] = "NM"
            output_format["boxes"] = "N6"
        else:
            raise ValueError(f"Unknown input format: {input_format}")
    elif task == "semantic_segmentation":
        if input_format == "TZYXC":
            output_format["masks"] = "TZYXC"
            output_format["labels"] = "TNM"
        elif input_format == "ZYXC":
            output_format["masks"] = "ZYXC"
            output_format["labels"] = "NM"
        else:
            raise ValueError(f"Unknown input format: {input_format}")
    elif task == "detection":
        if input_format == "TZYXC":
            output_format["scores"] = "TN"
            output_format["labels"] = "TNM"
            output_format["boxes"] = "TN6"
        elif input_format == "ZYXC":
            output_format["scores"] = "N"
            output_format["labels"] = "NM"
            output_format["boxes"] = "N6"
        else:
            raise ValueError(f"Unknown input format: {input_format}")
    else:
        raise ValueError(f"Unknown task: {task}")
    return output_format


def _expand_channel_names_for_batch(
    channel_names: Union[Dict[int, str], List[Dict[int, str]]],
    batch_size: int,
) -> List[Dict[int, str]]:
    if isinstance(channel_names, dict):
        return [channel_names] * batch_size
    if len(channel_names) != batch_size:
        raise ValueError(
            f"channel_names list length {len(channel_names)} must equal batch_size {batch_size}"
        )
    return list(channel_names)


def _timepoint_idxs_for_batch_idx(
    timepoint_idxs: Optional[Any],
    batch_index: int,
    batch_size: int,
) -> Optional[List[int]]:
    if timepoint_idxs is None:
        return None
    if (
        isinstance(timepoint_idxs, list)
        and len(timepoint_idxs) == batch_size
        and timepoint_idxs
        and isinstance(timepoint_idxs[0], list)
    ):
        return timepoint_idxs[batch_index]
    return timepoint_idxs


def save_predictions(
    image_path: str,
    model_name: str,
    preds: Dict[str, Any],
    task: Literal["instance_segmentation", "semantic_segmentation", "detection"],
    save_mode: Literal["overwrite", "create"],
    save_tensors_metadata: Dict[str, Dict[str, Any]],
    existing_channel_names: Dict[int, str],
    zarr_driver: str = "zarr3",
    timepoint_idxs: Optional[List[int]] = None,
    shard_spatial_shape: Optional[Tuple[int, int, int]] = None,
    chunk_spatial_shape: Optional[Tuple[int, int, int]] = None,
) -> None:
    exceptions = {}
    metadata = {"task": task}
    for output_name, output_metadata in save_tensors_metadata.items():
        try:
            data = preds[output_name]
        except KeyError as e:
            ray.logger.error(f"Save tensor {output_name} not found in preds: {e}")
            continue
        try:
            save_name = output_metadata["name"]
        except KeyError as e:
            ray.logger.error(f"Name for {output_name} not found in save_tensors_metadata: {e}")
            continue
        try:
            dtype = output_metadata["dtype"]
        except KeyError as e:
            ray.logger.error(f"Dtype for {output_name} not found in save_tensors_metadata: {e}")
            continue
        try:
            data_format = output_metadata["data_format"]
        except KeyError as e:
            ray.logger.error(f"Data format for {output_name} not found in save_tensors_metadata: {e}")
            continue
        try:
            annotation_type = output_metadata["annotation_type"]
        except KeyError as e:
            ray.logger.error(f"Annotation type for {output_name} not found in save_tensors_metadata: {e}")
            continue
        try:
            if annotation_type == "dense":
                assert output_metadata["data_format"] in ["TZYXC", "ZYXC"], f"Invalid data format: {output_metadata['data_format']}"
                save_dense_annotations(
                    image_path=image_path,
                    model_name=model_name,
                    data=data,
                    annotation_name=save_name,
                    channel_names=existing_channel_names,
                    data_format=cast(Literal["TZYXC", "ZYXC"], data_format),
                    save_mode=save_mode,
                    chunk_spatial_shape=chunk_spatial_shape,
                    shard_spatial_shape=shard_spatial_shape,
                    timepoint_idxs=timepoint_idxs,
                    zarr_driver=zarr_driver,
                    dtype=dtype,
                )
            elif annotation_type == "sparse":
                save_sparse_annotations(
                    image_path=image_path,
                    model_name=model_name,
                    data=data,
                    annotation_name=cast(Literal["scores", "labels", "boxes"], save_name),
                    data_format=cast(
                        Literal["TNM", "TN6", "TN", "NM", "N6", "N"],
                        data_format,
                    ),
                    save_mode=save_mode,
                    timepoint_idxs=timepoint_idxs,
                    zarr_driver=zarr_driver,
                    dtype=dtype,   
                )
            else:
                raise ValueError(f"Unknown annotation type: {annotation_type}")
        except Exception as e:
            ray.logger.error(f"Failed to save {annotation_type} annotation {output_name} with data format {data_format}: {e}", exc_info=True)
            exceptions[output_name] = e
    try:
        save_annotations_metadata(
            image_path=image_path,
            model_name=model_name,
            timepoint_idxs=timepoint_idxs,
            metadata=metadata,
        )
    except Exception as e:
        ray.logger.error(f"Failed to save metadata for {model_name} at {image_path}: {e}", exc_info=True)
        exceptions["metadata"] = e
    if len(exceptions) > 0:
        raise Exception(f"Failed to save {exceptions.keys()}", list(exceptions.values()))

@ray.remote(namespace="saver", lifetime="detached", num_cpus=0)
class SaveWorker:
    """
    Worker that pops (slot_handle, metadata) from queue, unpacks byte slots via layout,
    routes each output by output_type config, then buffer_manager.free(slot_handle).
    """

    def __init__(
        self,
        buffer_manager: BufferManager,
        save_mode: Literal["overwrite", "create"],
        max_workers: int = 4,
        columns: List[str] = [
            "server_folder",
            "output_folder",
            "tile_name",
        ],
        shard_spatial_shape: Optional[Tuple[int, int, int]] = None,
        chunk_spatial_shape: Optional[Tuple[int, int, int]] = None,
    ):
        self.buffer_manager = buffer_manager
        self.save_mode = save_mode
        self.columns = columns
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"save_worker_rank_{buffer_manager.global_rank}"
        )
        self._metrics = {
            "save_time_ms": [],
            "save_successful": [],
            "queue_time_ms": [],
        }
        if shard_spatial_shape is None or chunk_spatial_shape is None:
            raise ValueError(
                "shard_spatial_shape and chunk_spatial_shape must be provided because dense annotation arrays (masks) may need to be created"
            )
        
        self.shard_spatial_shape = shard_spatial_shape
        self.chunk_spatial_shape = chunk_spatial_shape

    def get_metrics(self) -> Dict[str, List[float | bool]]:
        metrics = self._metrics.copy()
        self._metrics = {
            "save_time_ms": [],
            "save_successful": [],
            "queue_time_ms": [],
        }
        return metrics

    def save(
        self, 
        inference_outputs: Dict[str, Any], 
        queue_t0: Optional[float] = None,
    ) -> None:
        if queue_t0 is not None:
            self._metrics["queue_time_ms"].append((time.perf_counter() - queue_t0) * 1000)
        output_arrays = {}
        slots_to_free = []
        t0 = time.perf_counter()
        try:
            # Do this first so that we can ensure slots get freed even
            # if inference_outputs has incomplete metadata
            for name, slot_info in inference_outputs.items():
                if name == "metainfo":
                    continue
                if isinstance(slot_info, np.ndarray):
                    output_arrays[name] = slot_info
                    ray.logger.warning(f"Inference output being passed through the ray plasma store: {name}")
                    continue
                output_array = self.buffer_manager.slot_info_to_view(slot_info)
                output_arrays[name] = output_array
                slots_to_free.append(slot_info)
            sample_metainfo = inference_outputs["metainfo"]
            required = (
                "task",
                "batch_size_actual",
                "save_tensors_metadata",
                "model_name",
                "channel_names",
            )
            for key in required:
                if key not in sample_metainfo:
                    raise KeyError(f"metainfo missing required key {key!r}")
            task = sample_metainfo["task"]
            batch_size = sample_metainfo["batch_size_actual"]
            timepoint_idxs_raw = sample_metainfo.get("timepoint_idxs", None)
            channel_names_list = _expand_channel_names_for_batch(
                sample_metainfo["channel_names"], batch_size
            )
            save_tensors_metadata = sample_metainfo["save_tensors_metadata"]

            batch_futures: List[Future] = []

            def _save_batch_element(b: int) -> None:
                metadata_element = {col: sample_metainfo[col][b] for col in self.columns}
                image_path = os.path.join(
                    metadata_element["server_folder"],
                    metadata_element["output_folder"],
                    metadata_element["tile_name"],
                )
                if not os.path.exists(image_path):
                    raise ValueError(
                        f"Save mode is {self.save_mode} but image path {image_path} does not exist"
                    )
                preds_element = {name: output_arrays[name][b] for name in output_arrays}
                save_predictions(
                    image_path=str(image_path),
                    model_name=sample_metainfo["model_name"],
                    preds=preds_element,
                    task=cast(
                        Literal[
                            "instance_segmentation",
                            "semantic_segmentation",
                            "detection",
                        ],
                        task,
                    ),
                    save_mode=cast(Literal["overwrite", "create"], self.save_mode),
                    save_tensors_metadata=save_tensors_metadata,
                    existing_channel_names=channel_names_list[b],
                    timepoint_idxs=_timepoint_idxs_for_batch_idx(
                        timepoint_idxs_raw, b, batch_size
                    ),
                    chunk_spatial_shape=self.chunk_spatial_shape,
                    shard_spatial_shape=self.shard_spatial_shape,
                )

            for b in range(batch_size):
                batch_futures.append(self.thread_pool.submit(_save_batch_element, b))

            for future in as_completed(batch_futures):
                try:
                    future.result()
                    self._metrics["save_successful"].append(True)
                except Exception as e:
                    ray.logger.error(f"Failed to save batch element: {e}", exc_info=True)
                    self._metrics["save_successful"].append(False)
                    continue
        except Exception as e:
            self._metrics["save_successful"].append(False)
            ray.logger.error(f"Failed to save: {e}", exc_info=True)
        finally:
            for slot_info in slots_to_free:
                self.buffer_manager.free_slot(slot_info)
            self._metrics["save_time_ms"].append((time.perf_counter() - t0) * 1000)

# TODO: Consider using this retry logic
# def submit_with_state(executor: ThreadPoolExecutor, fn: Callable, arg: Any, attempt: int, future_state: Dict[Future, Dict[str, Any]]):
#     fut = executor.submit(fn, arg)
#     future_state[fut] = {
#         "fn": fn,
#         "arg": arg,
#         "attempt": attempt,
#     }
#     return fut

# def run_with_retries(fn: Callable, args: List[Any], thread_pool: ThreadPoolExecutor, max_retries: int = 3, backoff_base: float = 0.1):
#     results = {}
#     errors = {}

#     with thread_pool as executor:
#         future_state = {}
#         for arg in args:
#             submit_with_state(executor, fn, arg, attempt=0, future_state=future_state)

#         while future_state:
#             for fut in as_completed(list(future_state.keys())):
#                 state = future_state.pop(fut)
#                 arg = state["arg"]
#                 attempt = state["attempt"]

#                 exc = fut.exception()
#                 if exc is None:
#                     results[arg] = fut.result()
#                     continue

#                 if attempt < max_retries:
#                     delay = backoff_base * (2 ** attempt)
#                     time.sleep(delay)
#                     submit_with_state(
#                         executor,
#                         state["fn"],
#                         arg,
#                         attempt=attempt + 1,
#                         future_state=future_state,
#                     )
#                 else:
#                     errors[arg] = exc

#     return results, errors

