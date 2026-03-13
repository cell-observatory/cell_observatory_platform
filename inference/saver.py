from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Literal, Optional, Tuple, Dict, Any, List, Callable

import ray

from cell_observatory_platform.data.io import save_masks, save_scores, save_labels, save_boxes
from cell_observatory_platform.data.datasets.buffers import BufferManager, slot_info_to_view
from pathlib import Path
import os

def input_format_to_output_format(
    input_format: Literal["TZYXC", "ZYXC"],
    task: Literal["instance_segmentation", "semantic_segmentation", "detection"],
) -> Dict[str, Literal["TZYXC", "ZYXC", "TN6", "N6", "TN", "N"]]:
    output_format = {}
    if task == "instance_segmentation":
        if input_format == "TZYXC":
            output_format["instance_masks"] = "TZYX"
            output_format["instance_scores"] = "TN"
            output_format["instance_labels"] = "TN"
            output_format["instance_boxes"] = "TN6"
        elif input_format == "ZYXC":
            output_format["instance_masks"] = "ZYXC"
            output_format["instance_scores"] = "N"
            output_format["instance_labels"] = "N"
            output_format["instance_boxes"] = "N6"
        else:
            raise ValueError(f"Unknown input format: {input_format}")
    elif task == "semantic_segmentation":
        if input_format == "TZYXC":
            output_format["semantic_masks"] = "TZYXC"
            output_format["semantic_labels"] = "TN"
        else:
            raise ValueError(f"Unknown input format: {input_format}")
    elif task == "detection":
        if input_format == "TZYXC":
            output_format["detection_scores"] = "TN"
            output_format["detection_labels"] = "TN"
            output_format["detection_boxes"] = "TN6"
        elif input_format == "ZYXC":
            output_format["detection_scores"] = "N"
            output_format["detection_labels"] = "N"
            output_format["detection_boxes"] = "N6"
        else:
            raise ValueError(f"Unknown input format: {input_format}")
    else:
        raise ValueError(f"Unknown task: {task}")
    return output_format


def save_instance_predictions(
    image_path: str,
    preds: Dict[str, Any],
    input_format: Literal["TZYXC", "ZYXC"],
    save_mode: Literal["overwrite", "append", "new_image"],
    shard_cube_shape: Optional[Tuple[int, int, int]] = None,
    chunk_shape: Optional[Tuple[int, int, int]] = None,
) -> None:
    instance_masks = preds.get("instance_masks", None)
    instance_scores = preds.get("instance_scores", None)
    instance_labels = preds.get("instance_labels", None)
    instance_boxes = preds.get("instance_boxes", None)
    preds_formats = input_format_to_output_format(input_format, "instance_segmentation")
    if instance_masks is not None:
        try:
            masks_format = preds_formats["instance_masks"]
            save_masks(
                image_path=image_path,
                masks=instance_masks,
                task="instance_segmentation",
                input_format=masks_format,
                save_mode=save_mode,
                chunk_shape=chunk_shape,
                shard_cube_shape=shard_cube_shape,
            )
        except Exception as e:
            ray.logger.error(f"Failed to save instance masks: {e}")
    if instance_scores is not None:
        try:
            scores_format = preds_formats["instance_scores"]
            save_scores(
                image_path=image_path,
                scores=instance_scores,
                task="instance_segmentation",
                input_format=scores_format,
                save_mode=save_mode,
            )
        except Exception as e:
            ray.logger.error(f"Failed to save instance scores: {e}")
    if instance_labels is not None:
        try:
            labels_format = preds_formats["instance_labels"]
            label_names = preds["label_names"]
            save_labels(
                image_path=image_path,
                labels=instance_labels,
                label_names=label_names,
                task="instance_segmentation",
                input_format=labels_format,
                save_mode=save_mode,
            )
        except Exception as e:
            ray.logger.error(f"Failed to save instance labels: {e}")
    if instance_boxes is not None:
        try:
            boxes_format = preds_formats["instance_boxes"]
            save_boxes(
                image_path=image_path,
                boxes=instance_boxes,
                task="instance_segmentation",
                input_format=boxes_format,
                save_mode=save_mode,
            )
        except Exception as e:
            ray.logger.error(f"Failed to save instance boxes: {e}")


def save_semantic_predictions(
    image_path: str,
    preds: Dict[str, Any],
    input_format: Literal["TZYXC", "ZYXC"],
    save_mode: Literal["overwrite", "append", "new_image"],
    shard_cube_shape: Optional[Tuple[int, int, int]] = None,
    chunk_shape: Optional[Tuple[int, int, int]] = None,
) -> None:
    semantic_masks = preds.get("semantic_masks", None)
    semantic_labels = preds.get("semantic_labels", None)
    preds_formats = input_format_to_output_format(input_format, "semantic_segmentation")
    if semantic_masks is not None:
        try:
            masks_format = preds_formats["semantic_masks"]
            save_masks(
                image_path=image_path,
                masks=semantic_masks,
                task="semantic_segmentation",
                input_format=masks_format,
                save_mode=save_mode,
                chunk_shape=chunk_shape,
                shard_cube_shape=shard_cube_shape,
            )
        except Exception as e:
            ray.logger.error(f"Failed to save semantic masks: {e}")
    if semantic_labels is not None:
        try:
            labels_format = preds_formats["semantic_labels"]
            label_names = preds["label_names"]
            save_labels(
                image_path=image_path,
                labels=semantic_labels,
                label_names=label_names,
                task="semantic_segmentation",
                input_format=labels_format,
                save_mode=save_mode,
            )
        except Exception as e:
            ray.logger.error(f"Failed to save semantic labels: {e}")


def save_detection_predictions(
    image_path: str,
    preds: Dict[str, Any],
    input_format: Literal["TZYXC", "ZYXC"],
    save_mode: Literal["overwrite", "append", "new_image"],
) -> None:
    detection_scores = preds.get("detection_scores", None)
    detection_labels = preds.get("detection_labels", None)
    detection_boxes = preds.get("detection_boxes", None)
    preds_formats = input_format_to_output_format(input_format, "detection")
    if detection_scores is not None:
        try:
            scores_format = preds_formats["detection_scores"]
            save_scores(
                image_path=image_path,
                scores=detection_scores,
                task="detection",
                input_format=scores_format,
                save_mode=save_mode,
            )
        except Exception as e:
            ray.logger.error(f"Failed to save detection scores: {e}")
    if detection_labels is not None:
        try:
            labels_format = preds_formats["detection_labels"]
            label_names = preds["label_names"]
            save_labels(
                image_path=image_path,
                labels=detection_labels,
                label_names=label_names,
                task="detection",
                input_format=labels_format,
                save_mode=save_mode,
            )
        except Exception as e:
            ray.logger.error(f"Failed to save detection labels: {e}")
    if detection_boxes is not None:
        try:
            boxes_format = preds_formats["detection_boxes"]
            save_boxes(
                image_path=image_path,
                boxes=detection_boxes,
                task="detection",
                input_format=boxes_format,
                save_mode=save_mode,
            )
        except Exception as e:
            ray.logger.error(f"Failed to save detection boxes: {e}")

@ray.remote(namespace="saver", lifetime="detached", num_cpus=0)
class SaveWorker:
    """
    Worker that pops (slot_handle, metadata) from queue, unpacks byte slots via layout,
    routes each output by output_type config, then buffer_manager.free(slot_handle).
    """

    def __init__(
        self,
        buffer_manager: BufferManager,
        max_retries: int = 3,
        retry_backoff_s: float = 0.5,
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
    ):
        self.buffer_manager = buffer_manager
        self.max_retries = max_retries
        self.columns = columns
        # TODO: Support threading with retries for each save operation
        if max_retries > 0:
            ray.logger.warning("Saving with retries is not supported yet")
        self.retry_backoff_s = retry_backoff_s
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers, 
            thread_name_prefix=f"save_worker_rank_{buffer_manager.global_rank}"
        )
        self._save_metrics = {
            "save_time_ms": 0.0,
            "save_successes": 0,
            "save_failures": 0,
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        return self._save_metrics.copy()

    def save(
        self, 
        inference_outputs: Dict[str, Any], 
        save_mode: Literal["overwrite", "append", "new_image"],
        shard_cube_shape: Optional[Tuple[int, int, int]] = None,
        chunk_shape: Optional[Tuple[int, int, int]] = None,
        save_dir: Optional[Path | str] = None,
    ) -> None:
        output_arrays = {}
        slots_to_free = []
        t0 = time.perf_counter()
        try:
            sample_metainfo = inference_outputs["metainfo"]
            task = sample_metainfo["task"]
            batch_size = sample_metainfo["batch_size_actual"]
            for name, slot_info in inference_outputs.items():
                if name == "metainfo":
                    continue
                output_array = slot_info_to_view(slot_info)
                output_arrays[name] = output_array
                slots_to_free.append(slot_info)

            batch_futures: List[Future] = []
            for b in range(batch_size):
                preds_element = {}
                metadata_element = {col: sample_metainfo[col][b] for col in self.columns}
                if save_mode == "overwrite" or save_mode == "append":
                    image_path = os.path.join(
                        metadata_element["server_folder"],
                        metadata_element["output_folder"],
                        metadata_element["tile_name"],
                    )
                elif save_mode == "new_image":
                    if save_dir is None:
                        raise ValueError("save_dir is required for new_image mode")
                    image_path: Path = Path(save_dir) / metadata_element["output_folder"] / metadata_element["tile_name"]
                    image_path.resolve()
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                for name, output_array in output_arrays.items():
                    preds_element[name] = output_array[b]
                if task == "instance_segmentation":
                    batch_futures.append(self.thread_pool.submit(
                        save_instance_predictions,
                        image_path=str(image_path),
                        preds=preds_element,
                        input_format=sample_metainfo["input_format"],
                        save_mode=save_mode,
                        chunk_shape=chunk_shape,
                        shard_cube_shape=shard_cube_shape,
                    ))
                elif task == "semantic_segmentation":
                    batch_futures.append(self.thread_pool.submit(
                        save_semantic_predictions,
                        image_path=str(image_path),
                        preds=preds_element,
                        input_format=sample_metainfo["input_format"],
                        save_mode=save_mode,
                        chunk_shape=chunk_shape,
                        shard_cube_shape=shard_cube_shape,
                    ))
                elif task == "detection":
                    batch_futures.append(self.thread_pool.submit(
                        save_detection_predictions,
                        image_path=str(image_path),
                        preds=preds_element,
                        input_format=sample_metainfo["input_format"],
                        save_mode=save_mode,
                    ))
                else:
                    raise ValueError(f"Unknown task: {task}")

            for future in as_completed(batch_futures):
                try:
                    future.result()
                    self._save_metrics["save_successes"] += 1
                except Exception as e:
                    ray.logger.error(f"Failed to save batch element: {e}", exc_info=True)
                    self._save_metrics["save_failures"] += 1
                    continue
            self._save_metrics["save_time_ms"] += (time.perf_counter() - t0) * 1000
        except Exception as e:
            ray.logger.error(f"Failed to save: {e}", exc_info=True)
        finally:
            for slot_info in slots_to_free:
                self.buffer_manager.free_slot(slot_info)


def submit_with_state(executor: ThreadPoolExecutor, fn: Callable, arg: Any, attempt: int, future_state: Dict[Future, Dict[str, Any]]):
    fut = executor.submit(fn, arg)
    future_state[fut] = {
        "fn": fn,
        "arg": arg,
        "attempt": attempt,
    }
    return fut

def run_with_retries(fn: Callable, args: List[Any], thread_pool: ThreadPoolExecutor, max_retries: int = 3, backoff_base: float = 0.1):
    results = {}
    errors = {}

    with thread_pool as executor:
        future_state = {}
        for arg in args:
            submit_with_state(executor, fn, arg, attempt=0, future_state=future_state)

        while future_state:
            for fut in as_completed(list(future_state.keys())):
                state = future_state.pop(fut)
                arg = state["arg"]
                attempt = state["attempt"]

                exc = fut.exception()
                if exc is None:
                    results[arg] = fut.result()
                    continue

                if attempt < max_retries:
                    delay = backoff_base * (2 ** attempt)
                    time.sleep(delay)
                    submit_with_state(
                        executor,
                        state["fn"],
                        arg,
                        attempt=attempt + 1,
                        future_state=future_state,
                    )
                else:
                    errors[arg] = exc

    return results, errors

