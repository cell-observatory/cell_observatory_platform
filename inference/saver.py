"""SaveWorker: persists inference predictions to disk as nested Zarr annotations.

A Ray actor that consumes the inferencer's ``dict[str, Tensor]`` outputs (plus
``metainfo``) and writes them under ``<pred_path>/<model_name>/<annotation_name>``
via :mod:`cell_observatory_platform.data.io`.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, Future
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np
import ray

from cell_observatory_platform.data import io
from cell_observatory_platform.data.datasets.buffers import BufferManager
from cell_observatory_platform.inference.inference_postprocess import (
    crop_sample_spatial,
    InferenceRecord,
    build_records,
)


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


from cell_observatory_platform.utils.registry import REGISTRY


@REGISTRY.register("save_handler", "save_labelmap")
def save_labelmap(
    *,
    image_path: str,
    model_name: str,
    data: np.ndarray,
    annotation_name: str,
    data_format: str,
    save_mode: Literal["overwrite", "create"],
    dtype: str = "uint16",
    chunk_spatial_shape: Optional[Tuple[int, int, int]] = None,
    shard_spatial_shape: Optional[Tuple[int, int, int]] = None,
    timepoint_idxs: Optional[List[int]] = None,
    zarr_driver: str = "zarr3",
    **_: Any,
) -> None:
    """Save an integer label-map (instance or semantic masks) via save_dense_annotations.

    ``data_format`` must be ``ZYXC`` or ``TZYXC``.
    ``annotation_name`` must contain ``"mask"`` (the name contract lives in
    save_masks / save_masks_channel_names; the underlying io call does NOT
    validate the name).
    """
    if data_format.upper() not in ("ZYXC", "TZYXC"):  # survives python -O
        raise ValueError(
            f"save_labelmap expects ZYXC or TZYXC data_format; got {data_format!r}"
        )
    io.save_dense_annotations(
        image_path=image_path,
        model_name=model_name,
        data=data,
        annotation_name=annotation_name,
        data_format=cast(Literal["TZYXC", "ZYXC"], data_format.upper()),
        save_mode=save_mode,
        chunk_spatial_shape=chunk_spatial_shape,
        shard_spatial_shape=shard_spatial_shape,
        timepoint_idxs=timepoint_idxs,
        zarr_driver=zarr_driver,
        dtype=dtype,
    )


@REGISTRY.register("save_handler", "save_dense_image")
def save_dense_image(
    *,
    image_path: str,
    model_name: str,
    data: np.ndarray,
    annotation_name: str,
    data_format: str,
    save_mode: Literal["overwrite", "create"],
    dtype: str = "float32",
    chunk_spatial_shape: Optional[Tuple[int, int, int]] = None,
    shard_spatial_shape: Optional[Tuple[int, int, int]] = None,
    timepoint_idxs: Optional[List[int]] = None,
    zarr_driver: str = "zarr3",
    **_: Any,
) -> None:
    """Save a float dense reconstruction volume via io.save_dense_image.

    ``dtype`` must be a float dtype (``float32``, ``float64``, etc.).
    ``annotation_name`` must NOT contain ``"mask"`` — use ``save_labelmap`` for masks.
    """
    if data_format.upper() not in ("ZYXC", "TZYXC"):  # survives python -O
        raise ValueError(
            f"save_dense_image expects ZYXC or TZYXC data_format; got {data_format!r}"
        )
    io.save_dense_image(
        image_path=image_path,
        model_name=model_name,
        data=data,
        annotation_name=annotation_name,
        data_format=cast(Literal["TZYXC", "ZYXC"], data_format.upper()),
        save_mode=save_mode,
        chunk_spatial_shape=chunk_spatial_shape,
        shard_spatial_shape=shard_spatial_shape,
        timepoint_idxs=timepoint_idxs,
        zarr_driver=zarr_driver,
        dtype=dtype,
    )


@REGISTRY.register("save_handler", "save_sparse")
def save_sparse(
    *,
    image_path: str,
    model_name: str,
    data: np.ndarray,
    annotation_name: str,
    data_format: str,
    save_mode: Literal["overwrite", "create"],
    dtype: str = "float32",
    timepoint_idxs: Optional[List[int]] = None,
    zarr_driver: str = "zarr3",
    **_: Any,
) -> None:
    """Save sparse per-object arrays (boxes, scores, labels) via save_sparse_annotations."""
    io.save_sparse_annotations(
        image_path=image_path,
        model_name=model_name,
        data=data,
        annotation_name=cast(Literal["scores", "labels", "boxes"], annotation_name),
        data_format=cast(
            Literal["TNM", "TN6", "TN", "NM", "N6", "N"],
            data_format,
        ),
        save_mode=save_mode,
        timepoint_idxs=timepoint_idxs,
        zarr_driver=zarr_driver,
        dtype=dtype,
    )


_DENSE_FORMATS = ("TZYXC", "ZYXC")


def _derive_save_handler(output_metadata: Dict[str, Any]) -> str:
    """Map a ``save_tensors`` config entry onto a registered ``save_handler`` name.

    Configs declare intent with ``annotation_type`` (``dense`` | ``sparse``) -- the same
    key the pre-registry dispatch used. The dense branch splits further because
    ``io.save_dense_image`` refuses mask-named or non-float arrays (see data/io.py):

      dense  + name contains "mask"  -> save_labelmap    (integer label map)
      dense  + name has no "mask"    -> save_dense_image (float volume)
      sparse                         -> save_sparse

    Raises rather than defaulting: a typo'd annotation_type must fail at the first batch,
    not produce an empty output directory.
    """
    try:
        annotation_type = output_metadata["annotation_type"]
    except KeyError:
        raise KeyError(
            "save_tensors entry is missing required key 'annotation_type' "
            f"(expected 'dense' or 'sparse'); got keys {sorted(output_metadata)}"
        ) from None

    if annotation_type == "sparse":
        return "save_sparse"

    if annotation_type == "dense":
        name = str(output_metadata["name"])
        dtype = str(output_metadata.get("dtype", ""))
        if "mask" in name.lower():
            return "save_labelmap"
        if dtype and np.dtype(dtype).kind != "f":
            raise ValueError(
                f"dense non-mask output {name!r} must declare a float dtype for "
                f"save_dense_image; got dtype={dtype!r}. Rename it to contain 'mask' "
                "if it is a label map."
            )
        return "save_dense_image"

    raise ValueError(
        f"Unknown annotation_type {annotation_type!r}; expected 'dense' or 'sparse'."
    )


def save_predictions(
    image_path: str,
    model_name: str,
    preds: Dict[str, Any],
    task: Literal["instance_segmentation", "semantic_segmentation", "object_detection"],
    save_mode: Literal["overwrite", "create"],
    save_tensors_metadata: Dict[str, Dict[str, Any]],
    zarr_driver: str = "zarr3",
    timepoint_idxs: Optional[List[int]] = None,
    shard_spatial_shape: Optional[Tuple[int, int, int]] = None,
    chunk_spatial_shape: Optional[Tuple[int, int, int]] = None,
    orig_spatial: Optional[Tuple[int, int, int]] = None,
) -> None:
    exceptions = {}
    metadata = {"task": task}
    for output_name, output_metadata in save_tensors_metadata.items():
        # Every failure below is RECORDED, not skipped. A bare `continue` here is what
        # made a missing dispatch key produce a "successful" run that wrote nothing.
        save_handler_key = "<underived>"
        data_format = "<unknown>"
        try:
            data = preds[output_name]
            save_name = output_metadata["name"]
            dtype = output_metadata["dtype"]
            data_format = output_metadata["data_format"]
            save_handler_key = _derive_save_handler(output_metadata)

            handler = REGISTRY.get("save_handler", save_handler_key).factory
            # Dense handlers require spatial-crop before writing.  Tile-mode restore
            # places each prediction top-left in a full-tile buffer with trailing
            # zero-pad; cropping to orig_spatial drops that pad so only the original
            # tile is written.
            if save_handler_key in ("save_labelmap", "save_dense_image"):
                if data_format.upper() not in _DENSE_FORMATS:  # survives python -O
                    raise ValueError(
                        f"Invalid data format for dense handler {save_handler_key!r}: {data_format!r}"
                    )
                if orig_spatial is not None:
                    data = crop_sample_spatial(
                        data,
                        tuple(int(s) for s in orig_spatial),
                        data_format.upper().startswith("T"),
                    )
            handler(
                image_path=image_path,
                model_name=model_name,
                data=data,
                annotation_name=save_name,
                data_format=data_format,
                save_mode=save_mode,
                dtype=dtype,
                chunk_spatial_shape=chunk_spatial_shape,
                shard_spatial_shape=shard_spatial_shape,
                timepoint_idxs=timepoint_idxs,
                zarr_driver=zarr_driver,
            )
        except Exception as e:
            ray.logger.error(f"Failed to save {save_handler_key!r} annotation {output_name} with data format {data_format}: {e}", exc_info=True)
            exceptions[output_name] = e
    if not exceptions:
        try:
            io.save_annotations_metadata(
                image_path=image_path,
                model_name=model_name,
                timepoint_idxs=timepoint_idxs,
                metadata=metadata,
            )
        except Exception as e:
            ray.logger.error(f"Failed to save metadata for {model_name} at {image_path}: {e}", exc_info=True)
            exceptions["metadata"] = e
    else:
        # Do NOT advertise annotations that failed to write: stamping the
        # metadata on a failed batch leaves an on-disk stub that makes the
        # model dir look complete to any reader that trusts it.
        ray.logger.error(
            f"Skipping annotations metadata for {image_path}: "
            f"{len(exceptions)} annotation writes failed."
        )
    if exceptions:
        detail = "\n".join(f"{k}: {v}" for k, v in exceptions.items())
        raise RuntimeError(
            f"{len(exceptions)}/{len(save_tensors_metadata)} failed to save.\n{detail}"
        )

class SaveWorkerBase:
    """
    Worker that pops (slot_handle, metadata) from queue, unpacks byte slots via layout,
    routes each output by output_type config, then buffer_manager.free(slot_handle).

    Plain class holding all logic; ``SaveWorker`` below is its Ray-actor wrapper
    (kept separate so unit tests can exercise the logic in-process).
    """

    def __init__(
        self,
        buffer_manager: BufferManager,
        save_mode: Literal["overwrite", "create"],
        max_workers: int = 4,
        # tuple default: a mutable list default is shared across instances
        columns: Sequence[str] = (
            "server_folder",
            "output_folder",
            "tile_name",
        ),
        shard_spatial_shape: Optional[Tuple[int, int, int]] = None,
        chunk_spatial_shape: Optional[Tuple[int, int, int]] = None,
    ):
        if save_mode not in ["overwrite", "create"]:
            raise ValueError(
                f"Invalid save_mode {save_mode!r}. Must be 'overwrite' or 'create' "
                "('append' is not implemented -- it was previously coerced silently "
                "to 'overwrite')."
            )

        self.buffer_manager = buffer_manager
        self.save_mode = save_mode
        self.columns = list(columns)
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

    def ping(self) -> bool:
        """Side-effect-free liveness probe (used by the inferencer's
        backpressure heartbeat; get_metrics RESETS metrics and must not be
        used as a probe)."""
        return True

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
        # Bound before the try: an early failure (bad metainfo) must not raise
        # NameError/UnboundLocalError in the final error report / finally block.
        batch_size = -1
        errors: List[Exception] = []
        records: List[InferenceRecord] = []
        batch_futures: List[Future] = []
        t0 = time.perf_counter()
        try:
            # Claim EVERY slot for freeing BEFORE resolving any view: a failed
            # resolution (dead buffer actor, stale segment) aborts the loop, and
            # the finally must still free the failing slot AND the ones after it
            # or the detached pool leaks permanently. Two passes, claim first.
            for name, slot_info in inference_outputs.items():
                if name == "metainfo" or isinstance(slot_info, np.ndarray):
                    continue
                slots_to_free.append(slot_info)
            for name, slot_info in inference_outputs.items():
                if name == "metainfo":
                    continue
                if isinstance(slot_info, np.ndarray):
                    output_arrays[name] = slot_info
                    ray.logger.warning(f"Inference output being passed through the ray plasma store: {name}")
                    continue
                output_arrays[name] = self.buffer_manager.slot_info_to_view(slot_info)
            sample_metainfo = inference_outputs["metainfo"]
            required = (
                "task",
                "batch_size_actual",
                "save_tensors_metadata",
                "model_name",
            )
            for key in required:
                if key not in sample_metainfo:
                    raise KeyError(f"metainfo missing required key {key!r}")
            task = sample_metainfo["task"]
            batch_size = sample_metainfo["batch_size_actual"]
            timepoint_idxs_raw = sample_metainfo.get("timepoint_idxs", None)
            save_tensors_metadata = sample_metainfo["save_tensors_metadata"]

            orig_image_sizes = sample_metainfo.get("orig_image_sizes", None)

            # Uniform per-sample unpack (batch-first). image_key=None keeps every output
            # in record.preds; save writes the raw labelmap (no viz mask normalization).
            records = build_records(
                output_arrays, sample_metainfo, columns=tuple(self.columns), image_key=None,
            )

            def _save_batch_element(record: InferenceRecord) -> None:
                image_path = os.path.join(
                    record.metadata["server_folder"],
                    record.metadata["output_folder"],
                    record.metadata["tile_name"],
                )
                if not os.path.exists(image_path):
                    raise ValueError(
                        f"Save mode is {self.save_mode} but image path {image_path} does not exist"
                    )
                # Original tile spatial size (Z, Y, X) this element was restored to;
                # dense outputs are cropped to it (drops the full-tile buffer zero-pad).
                orig_spatial = None
                if orig_image_sizes is not None:
                    row = orig_image_sizes[record.index]
                    orig_spatial = tuple(int(x) for x in tuple(np.asarray(row).ravel())[-3:])
                save_predictions(
                    image_path=str(image_path),
                    model_name=sample_metainfo["model_name"],
                    preds=record.preds,
                    orig_spatial=orig_spatial,
                    task=cast(
                        Literal[
                            "instance_segmentation",
                            "semantic_segmentation",
                            "object_detection",
                        ],
                        task,
                    ),
                    save_mode=cast(Literal["overwrite", "create"], self.save_mode),
                    save_tensors_metadata=save_tensors_metadata,
                    timepoint_idxs=_timepoint_idxs_for_batch_idx(
                        timepoint_idxs_raw, record.index, batch_size
                    ),
                    chunk_spatial_shape=self.chunk_spatial_shape,
                    shard_spatial_shape=self.shard_spatial_shape,
                )

            for record in records:
                batch_futures.append(self.thread_pool.submit(_save_batch_element, record))

            for future in as_completed(batch_futures):
                try:
                    future.result()
                    self._metrics["save_successful"].append(True)
                except Exception as e:
                    ray.logger.error(f"Failed to save batch element: {e}", exc_info=True)
                    self._metrics["save_successful"].append(False)
                    errors.append(e)
        except Exception as e:
            self._metrics["save_successful"].append(False)
            ray.logger.error(f"Failed to save: {e}", exc_info=True)
            errors.append(e)
        finally:
            # record.preds are numpy VIEWS into the SHM slots: never free a slot
            # while a submitted save may still be reading it.
            wait(batch_futures)
            for slot_info in slots_to_free:
                self.buffer_manager.free_slot(slot_info)
            self._metrics["save_time_ms"].append((time.perf_counter() - t0) * 1000)
        if errors:
            raise RuntimeError(
                f"{errors}\n{len(errors)}/{max(batch_size, len(records))} failed."
            )


@ray.remote(namespace="saver", lifetime="detached", num_cpus=1)
class SaveWorker(SaveWorkerBase):
    """Ray-actor wrapper around :class:`SaveWorkerBase` (see its docstring)."""

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

