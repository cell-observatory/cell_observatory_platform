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
    save_masks,
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
    input_format: Literal["TZYXC", "ZYXC"],
    save_mode: Literal["overwrite", "append"],
    save_tensors_dtypes: Dict[str, str],
    existing_channel_names: Dict[int, str],
    zarr_driver: str = "zarr3",
    timepoint_idxs: Optional[List[int]] = None,
    shard_spatial_shape: Optional[Tuple[int, int, int]] = None,
    chunk_spatial_shape: Optional[Tuple[int, int, int]] = None,
) -> None:
    preds_formats = input_format_to_output_format(input_format, task)
    exceptions = {}
    metadata = {"task": task}
    for output_name, output_format in preds_formats.items():
        try:
            data = preds[output_name]
        except KeyError as e:
            ray.logger.error(f"Prediction {output_name} not found in preds: {e}")
            continue
        try:
            dtype = save_tensors_dtypes[output_name]
        except KeyError as e:
            ray.logger.error(f"Dtype for {output_name} not found in save_tensors_dtypes: {e}")
            continue
        try:

            if output_name == "masks":
                assert output_format in ["TZYXC", "ZYXC"], f"Invalid output format: {output_format}"
                save_masks(
                    image_path=image_path,
                    model_name=model_name,
                    masks=data,
                    task=task,
                    existing_channel_names=existing_channel_names,
                    input_format=cast(Literal["TZYXC", "ZYXC"], output_format),
                    save_mode=save_mode,
                    chunk_spatial_shape=chunk_spatial_shape,
                    shard_spatial_shape=shard_spatial_shape,
                    timepoint_idxs=timepoint_idxs,
                    zarr_driver=zarr_driver,
                    dtype=dtype,
                )
            else:
                save_sparse_annotations(
                    image_path=image_path,
                    model_name=model_name,
                    data=data,
                    annotation_name=cast(Literal["scores", "labels", "boxes"], output_name),
                    input_format=cast(
                        Literal["TNM", "TN6", "TN", "NM", "N6", "N"],
                        output_format,
                    ),
                    save_mode=save_mode,
                    timepoint_idxs=timepoint_idxs,
                    zarr_driver=zarr_driver,
                    dtype=dtype,
                )
        except Exception as e:
            ray.logger.error(f"Failed to save {output_name}: {e}", exc_info=True)
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
        save_mode: Literal["overwrite", "append"],
        max_workers: int = 4,
        columns: List[str] = [
            "x_start",
            "y_start",
            "z_start",
            "time_start",
            "channel_size",
            "z_size",
            "y_size",
            "x_size",
            "time_size",
            "server_folder",
            "output_folder",
            "tile_name",
            "prepared_id",
            "mask_bbox_dict",
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
            "save_time_ms": 0.0,
            "save_calls": 0,
            "save_successes": 0,
            "save_failures": 0,
        }
        self.shard_spatial_shape = shard_spatial_shape
        self.chunk_spatial_shape = chunk_spatial_shape

    def get_metrics(self) -> Dict[str, Any]:
        return self._metrics.copy()

    def clear_metrics(self) -> None:
        self._metrics = {
            "save_time_ms": 0.0,
            "save_calls": 0,
            "save_successes": 0,
            "save_failures": 0,
        }

    def save(
        self, 
        inference_outputs: Dict[str, Any], 
    ) -> None:
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
                "input_format",
                "model_name",
                "save_tensors_dtypes",
                "channel_names",
            )
            for key in required:
                if key not in sample_metainfo:
                    raise KeyError(f"metainfo missing required key {key!r}")
            task = sample_metainfo["task"]
            batch_size = sample_metainfo["batch_size_actual"]
            save_tensors_dtypes = sample_metainfo["save_tensors_dtypes"]
            timepoint_idxs_raw = sample_metainfo.get("timepoint_idxs", None)
            channel_names_list = _expand_channel_names_for_batch(
                sample_metainfo["channel_names"], batch_size
            )

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
                    input_format=cast(Literal["TZYXC", "ZYXC"], sample_metainfo["input_format"]),
                    save_mode=cast(Literal["overwrite", "append"], self.save_mode),
                    save_tensors_dtypes=save_tensors_dtypes,
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
                    self._metrics["save_successes"] += 1
                except Exception as e:
                    ray.logger.error(f"Failed to save batch element: {e}", exc_info=True)
                    self._metrics["save_failures"] += 1
                    continue
        except Exception as e:
            self._metrics["save_failures"] += 1
            ray.logger.error(f"Failed to save: {e}", exc_info=True)
        finally:
            for slot_info in slots_to_free:
                self.buffer_manager.free_slot(slot_info)
            self._metrics["save_time_ms"] += (time.perf_counter() - t0) * 1000
            self._metrics["save_calls"] += 1

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

