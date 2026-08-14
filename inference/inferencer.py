import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import ray
import torch
from torch.nn import functional as F
from ray.actor import ActorHandle

import cupy as cp
from cupy.cuda import runtime as cudart
import numpy as np

from omegaconf import DictConfig, OmegaConf

from cell_observatory_platform.training.helpers import get_patch_sizes
from cell_observatory_platform.utils.context import barrier, get_world_size, process_rank
from cell_observatory_platform.models.layers.patch_embeddings import calc_num_patches
from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_platform.data.datasets.buffers import BufferManager
from cell_observatory_platform.inference.saver import SaveWorker
from cell_observatory_platform.inference.visualizer import VizWorker
from cell_observatory_platform.inference.inference_postprocess import postprocess

logger = logging.getLogger(__name__)

from cell_observatory_platform.utils.registry import REGISTRY
from cell_observatory_platform.utils.config import registers_as


@registers_as("inferencer", "inferencer_worker")
class InferencerWorker:
    def __init__(
        self,
        aggregate_mode: Literal["stitch_volume", "none"],
        inference_mode: Literal["tile", "cube"],
        task: Optional[Literal[
            "object_detection",
            "instance_segmentation",
            "semantic_segmentation",
            "denoising",
            "channel_split",
            "upsample_space",
            "upsample_time",
            "upsample_spacetime",
        ]],
        outputs_metadata: dict,
        input_format: str,
        input_shape: List[int],
        patch_shape: List[Optional[int]],
        model_name: str,
        save_outputs: bool,
        block_on_save: bool,
        vizualize_outputs: bool,
        block_on_viz: bool,
        model: torch.nn.Module,
        buffer_manager: BufferManager,
        save_worker: ActorHandle[SaveWorker],
        viz_worker: ActorHandle[VizWorker],
        viz_sampling_policy: Optional[Dict[str, Any]] = None,
        timepoint_idxs_for_save: Optional[List[int]] = None,
    ):

        self.input_shape = input_shape
        self.patch_shape = patch_shape
        self.input_format = input_format

        temporal_patch_size, self.axial_patch_size, self.lateral_patch_size = get_patch_sizes(
            input_format=self.input_format, patch_shape=list(patch_shape)
        )
        _, token_shape = calc_num_patches(
            input_fmt=self.input_format,
            input_shape=self.input_shape,
            patch_shape=tuple(patch_shape),
        )
        # NOTE: if input format does not contain 'T' we set patch size to 1
        #       since buffers assume that the T-axis exists
        self.temporal_patch_size = temporal_patch_size if temporal_patch_size else 1
        self.token_shape = self._get_token_shape(token_shape, self.input_format)

        self.model = model

        if inference_mode == "cube" and save_outputs:
            raise NotImplementedError("Saving outputs in cube mode not implemented")

        self.inference_mode = inference_mode

        self.task = task

        if aggregate_mode != "none":
            raise NotImplementedError("Aggregate mode not implemented for inference")
        self.aggregate_mode = aggregate_mode

        self.vizualize_outputs = vizualize_outputs
        self.save_outputs = save_outputs
        self.block_on_save = block_on_save
        self.block_on_viz = block_on_viz
        self.save_worker = save_worker
        self.viz_worker = viz_worker
        self.viz_sampling_policy = viz_sampling_policy
        # Visualizations render the PROCESSED frame: the pre-restore tensors at
        # the model's working resolution (uniform across the batch; viz slots keep
        # the processed shape). The save path is unaffected (always restored to the
        # original tile on disk).
        self.model_name = model_name

        self._save_timepoint_idxs = (
            None
            if timepoint_idxs_for_save is None
            else [int(x) for x in timepoint_idxs_for_save]
        )

        self.buffer_manager = buffer_manager

        device_idx = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{device_idx}")
        with cp.cuda.Device(device_idx):
            self._cp_d2h_stream = cp.cuda.Stream(non_blocking=True)
        self._d2h_stream = torch.cuda.ExternalStream(
            int(self._cp_d2h_stream.ptr), device=self.device
        )
        self.buffer_manager.pin_buffers()

        assert outputs_metadata is not None, "outputs_metadata must be provided"
        self.outputs_metadata = OmegaConf.to_container(outputs_metadata, resolve=True) \
            if isinstance(outputs_metadata, DictConfig) else dict(outputs_metadata)

        self.rank, self.world_size = process_rank(), get_world_size()

        self._metrics: Dict[str, float] = {
        }

        self._tasks = []
        # Tiles dropped from the SAVE path (block_on_save=False + pool exhaustion).
        # finalize() raises when non-zero: the persisted output is incomplete.
        self._dropped_saves = 0
        # Pinned host staging buffers for NON-buffer tensors, one FLAT buffer per
        # (role/name, dtype) grown by capacity doubling. cudaHostAlloc synchronizes
        # the device, so allocating pinned memory per sample stalls the pipeline;
        # and keying by shape would mint a new never-freed pinned block for every
        # distinct data-dependent shape (e.g. per-batch detection counts).
        self._staging_buffers: Dict[Tuple[str, torch.dtype], torch.Tensor] = {}

        if self.vizualize_outputs and self.viz_worker is not None:
            try:
                handler_names = ray.get(
                    self.viz_worker.get_handler_names.remote(), timeout=60
                )
            except AttributeError:
                # Non-standard viz worker (e.g. a test stub) without introspection;
                # the transport validation below cannot run.
                ray.logger.warning(
                    "viz_worker does not expose get_handler_names(); skipping "
                    "data_tensor viz-transport validation."
                )
            else:
                self._validate_viz_image_transport(self.outputs_metadata, handler_names)

        ray.logger.info(f"Main output metadata: {self.outputs_metadata}")
        ray.logger.info(f"Aggregate mode: {self.aggregate_mode}")

    # Viz handlers that consume ``record.image`` (the input volume). For these,
    # data_tensor must ride a SHM viz pool like every other viz tensor -- the old
    # silent plasma fallback (a full pageable D2H + plasma serialization per
    # visualized tile) was removed; see ``_save_inference_outputs``.
    _IMAGE_CONSUMING_VIZ_HANDLERS = frozenset(
        {"semantic_map", "instance_overlay", "feature_viz", "bbox_overlay"}
    )

    @classmethod
    def _validate_viz_image_transport(
        cls, outputs_metadata: Dict[str, Any], handler_names
    ) -> None:
        """Fail at startup if a configured viz handler needs ``record.image`` but
        ``data_tensor`` is not routed through a SHM viz pool."""
        image_handlers = sorted(set(handler_names) & cls._IMAGE_CONSUMING_VIZ_HANDLERS)
        if not image_handlers:
            return
        visualize_tensors = list(outputs_metadata.get("visualize_tensors") or ())
        buffer_tensors = list(outputs_metadata.get("buffer_tensors") or ())
        missing = [
            key
            for key, listed in (
                ("visualize_tensors", visualize_tensors),
                ("buffer_tensors", buffer_tensors),
            )
            if "data_tensor" not in listed
        ]
        if missing:
            raise ValueError(
                f"viz handlers {image_handlers} consume record.image (the input "
                "volume), which must be shipped through a shared-memory viz pool. "
                "Add 'data_tensor' to "
                f"{' and '.join(missing)} (and declare its shape/dtype under "
                "outputs_metadata.tensor_info) in the inference config "
                "(configs/inference/*.yaml -> inferencer_worker.outputs_metadata). "
                "The plasma fallback for record.image was removed."
            )

    def _get_token_shape(self, token_shape: Tuple[int, ...], input_format: str) -> Tuple[int, ...]:
        if input_format == "ZYXC":
            return token_shape[1:-1]  # drop T and C dimensions
        elif input_format == "TZYXC":
            return token_shape[:-1] # drop C dimension
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

    def _predict(self, batch_tensor: torch.Tensor, data_sample: dict) -> Dict[str, torch.Tensor]:
        """Run the model's ``inference_step()`` and return a ``dict[str, torch.Tensor]``.

        Output naming is owned by the model: ``inference_step()`` must return a
        dict keyed by the names declared in the model's ``output_metadata``. 

        Each model owns its full output contract (mask rank normalisation, fuse,
        any layout permutes). The inferencer is model-agnostic: it calls
        ``inference_step(data_sample)`` for all models including SAM2.
        """
        preds = self.model.inference_step(data_sample)

        if not isinstance(preds, dict):
            raise ValueError(
                "model.inference_step() must return a dict[str, torch.Tensor]; "
                f"got {type(preds)}. Each model owns its output naming via output_metadata."
            )

        return preds

    def predict(self, data_sample: dict):
        """
        Run model prediction on a full tile and dispatch outputs to
        save / viz workers.  Each data_sample is an entire tile (zarr image).
        """
        if self.aggregate_mode != "none":
            ray.logger.warning("Full-tile inference does not support aggregation.")

        # Incrementally reap finished save/viz tasks so an actor death / failed
        # batch surfaces within a few batches instead of only at finalize().
        if self._tasks:
            done, self._tasks = ray.wait(
                self._tasks, num_returns=len(self._tasks), timeout=0
            )
            # Collect errors across the WHOLE done set (like finalize does):
            # raising on the first would drop sibling failures untracked and
            # undercount finalize()'s "N/M failed" report.
            errors = []
            for task in done:
                try:
                    ray.get(task)
                except Exception as e:
                    errors.append(e)
            if errors:
                raise RuntimeError(
                    f"{len(errors)}/{len(done)} save/viz tasks failed; "
                    f"first error: {errors[0]!r}"
                ) from errors[0]

        X = data_sample["data_tensor"]
        t0_predict = time.perf_counter()
        preds = self._predict(X, data_sample)
        self._metrics["predict_time_ms"] = (time.perf_counter() - t0_predict) * 1000

        # Viz renders the PROCESSED frame: shallow-copy the pre-restore dict now --
        # postprocess() replaces entries with restored tensors but never mutates the
        # originals, so these refs stay at processed res.
        preds_viz = dict(preds)

        # Restore dense predictions to the original tile resolution (tile-mode
        # resize undo) and rescale boxes to the original coordinate frame.
        t0_restore = time.perf_counter()
        preds = postprocess(preds, data_sample, self.outputs_metadata)
        self._metrics["restore_outputs_time_ms"] = (time.perf_counter() - t0_restore) * 1000

        t0_save = time.perf_counter()
        self._save_inference_outputs(
            data_sample=data_sample,
            preds=preds,
            preds_viz=preds_viz,
        )
        self._metrics["save_inference_outputs_time_ms"] = (time.perf_counter() - t0_save) * 1000

    def _should_visualize(self, data_sample: dict, preds: dict) -> bool:
        """
        Check if the data sample should be visualized
        """
        # TODO: Should this be moved to the viz worker?
        # I'm leaving here because this way we don't have to flood the viz worker
        # with data samples if we are using a sampling policy.
        # Per-sample match indices for the by_tile policy: the gate is
        # batch-granular, so without this the viz worker renders every sample
        # of a batch where ONE tile matched. Consumed by _save_inference_outputs.
        self._viz_sample_idx = None
        if self.viz_sampling_policy is None:
            return True
        if self.viz_sampling_policy["name"] == "by_tile":
            # Allowlist filter over the per-sample region columns. `tile_names`
            # and `rois` (prepared_id values) are each optional; a sample matches
            # when it passes every filter that IS set, so the three modes are:
            #   tile_names only          -> tile filter
            #   rois only                -> ROI filter
            #   tile_names + rois        -> that tile within those ROIs
            tile_allow = self.viz_sampling_policy.get("tile_names")
            roi_allow = self.viz_sampling_policy.get("rois")
            if tile_allow is None and roi_allow is None:
                raise ValueError(
                    "by_tile viz_sampling_policy needs 'tile_names' and/or 'rois'"
                )

            def _column(key: str) -> list:
                # Batched per-sample column (list/array), or a bare scalar for
                # B=1 fixtures; per-element membership, not column == scalar.
                if key not in data_sample["metainfo"]:
                    raise KeyError(
                        f"viz_sampling_policy filters on {key!r} but metainfo has "
                        f"no such column"
                    )
                col = data_sample["metainfo"][key]
                if isinstance(col, (str, bytes)) or np.ndim(col) == 0:
                    col = [col]
                return np.asarray(col).ravel().tolist()

            tiles = _column("tile_name")
            tile_set = None if tile_allow is None else {str(t) for t in tile_allow}
            roi_set = None if roi_allow is None else {str(r) for r in roi_allow}
            rois = _column("prepared_id") if roi_set is not None else [None] * len(tiles)
            # Both filters must hit on the SAME sample (zip), not one sample each.
            match_idx = [
                b for b, (r, t) in enumerate(zip(rois, tiles))
                if (roi_set is None or str(r) in roi_set)
                and (tile_set is None or str(t) in tile_set)
            ]
            if not match_idx:
                return False
            self._viz_sample_idx = match_idx
            return True
        if self.viz_sampling_policy["name"] == "random_sample":
            return np.random.rand() < self.viz_sampling_policy["fraction"]
        raise ValueError(
            f"unknown viz_sampling_policy {self.viz_sampling_policy['name']!r}; "
            "expected 'by_tile' or 'random_sample'"
        )

    def _copy_d2h(self, dst: np.ndarray, src: torch.Tensor) -> None:
        """Async device-to-host memcpy via CuPy, mirroring CollatorActor.copy_h2d.

        Guards against host-memory corruption: the destination SHM slot is sized
        from the declared ``tensor_info`` (DB maxima x declared C/dtype), so a
        model output whose channel/time count or dtype disagrees with that
        declaration must fail loudly here rather than overrun the slot. This
        fail-hard guard is the sole runtime check on the output contract.
        """
        if not src.is_contiguous():
            # The raw memcpy serializes storage order: a non-contiguous view
            # (permute/slice) would land scrambled bytes in the slot.
            src = src.contiguous()
        if tuple(src.shape[1:]) != tuple(dst.shape[1:]):
            raise ValueError(
                "d2h copy per-sample shape mismatch: src "
                f"{tuple(src.shape)} vs slot {tuple(dst.shape)} -- a smaller "
                "total byte count could silently interleave samples in the "
                "batched slot. Model output shape disagrees with the declared "
                "output_metadata tensor_info shape."
            )
        src_nbytes = src.nelement() * src.element_size()
        dst_nbytes = int(dst.nbytes)
        if src_nbytes > dst_nbytes:
            raise ValueError(
                "d2h copy would overrun host slot: src "
                f"{tuple(src.shape)} ({src.dtype}, {src_nbytes} B) does not fit "
                f"dst {tuple(dst.shape)} ({dst.dtype}, {dst_nbytes} B). Model "
                "output shape/dtype disagrees with declared output_metadata "
                "tensor_info; fix the declaration or the model output."
            )
        # Same-WIDTH dtype mismatches pass the byte check but reinterpret bits
        # (float32 src into an int32 slot). The raw memcpy has no cast step, so
        # dtype identity is part of the output contract. Dtypes without a numpy
        # equivalent (bfloat16 slots ride as their byte-width view) are skipped
        # -- the nbytes check above still bounds those.
        try:
            src_np_dtype = np.dtype(str(src.dtype).replace("torch.", ""))
        except TypeError:
            src_np_dtype = None
        if src_np_dtype is not None and src_np_dtype != dst.dtype:
            raise ValueError(
                f"d2h copy dtype mismatch: src {src.dtype} vs slot {dst.dtype} -- "
                "the raw memcpy would reinterpret bits, not cast. Align the model "
                "output dtype with the declared tensor_info dtype."
            )
        dst_ptr = dst.__array_interface__["data"][0]
        src_ptr = src.data_ptr()
        cudart.memcpyAsync(
            dst_ptr, src_ptr, src_nbytes,
            cudart.memcpyDeviceToHost, int(self._cp_d2h_stream.ptr),
        )

    @staticmethod
    def _tree_to_cpu_numpy(obj: Any) -> Any:
        """Recursively detach every ``torch.Tensor`` to CPU NumPy for Ray IPC.

        Ray cannot pickle CUDA tensors. Nested dicts, lists, tuples, and object
        ndarrays are walked; buffer ``slot_info`` dicts only contain primitives
        and pass through unchanged.
        """
        if torch.is_tensor(obj):
            # Numpy doesn't support bFloat16, convert to float32
            if obj.dtype == torch.bfloat16:
                obj = obj.float()
            return obj.detach().cpu().contiguous().numpy()
        if isinstance(obj, dict):
            return {k: InferencerWorker._tree_to_cpu_numpy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [InferencerWorker._tree_to_cpu_numpy(x) for x in obj]
            # namedtuples take positional fields.
            if hasattr(obj, "_fields"):
                return type(obj)(*converted)
            return type(obj)(converted)
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                flat = obj.ravel()
                out = np.empty(flat.shape[0], dtype=object)
                for i, x in enumerate(flat):
                    out[i] = InferencerWorker._tree_to_cpu_numpy(x)
                return out.reshape(obj.shape)
            return obj
        return obj

    def _attach_save_worker_metainfo(self, metainfo_numpy: Dict[str, Any]) -> None:
        """Populate keys required by ``SaveWorker.save`` (see ``saver.save_predictions``).

        Output naming comes from the config's ``save_tensors`` metadata; channel
        provenance rides along in ``metainfo['channel_mapping']`` (carried from the
        dataset/DB).
        """
        metainfo_numpy.setdefault("model_name", self.model_name)
        metainfo_numpy.setdefault("task", self.task)
        metainfo_numpy.setdefault("save_tensors_metadata", self.outputs_metadata["save_tensors"])

        if self._save_timepoint_idxs is not None:
            metainfo_numpy.setdefault("timepoint_idxs", self._save_timepoint_idxs)

    def _attach_viz_worker_metainfo(self, metainfo_numpy: Dict[str, Any]) -> None:
        """Populate keys required by viz handlers.

        Carries the merged, model-sourced output tensor metadata (``tensor_info``,
        which includes each output's declared ``kind``) so viz handlers dispatch
        on the producer's declared semantics (e.g. ``instance_label_map`` vs
        ``instance_stack``) instead of sniffing tensor rank/dtype. Mirrors
        ``_attach_save_worker_metainfo``; the same transport pattern as
        ``save_tensors_metadata``.
        """
        metainfo_numpy.setdefault("tensor_metadata", self.outputs_metadata["tensor_info"])
        # Viz mirrors the saver's crop-to-original-tile (frame symmetry): the viz
        # worker needs the dense/sparse declarations to know WHICH tensors were
        # restored to the full-tile buffer and must be cropped per sample.
        metainfo_numpy.setdefault("save_tensors_metadata", self.outputs_metadata["save_tensors"])

    def _stage_output(self, role, name, tensor, slots, outputs_dict) -> None:
        """Cast + copy one output to host: SHM slot when reserved, pinned
        staging otherwise; CPU tensors copy host-side (a host src pointer in
        cudaMemcpyAsync is cudaErrorInvalidValue)."""
        dtype_str = self.outputs_metadata["tensor_info"][name]["dtype"]
        dtype = TORCH_DTYPES[dtype_str].value if isinstance(dtype_str, str) else dtype_str
        tensor = tensor.to(dtype=dtype)
        if name in slots:
            slot_info = slots[name]
            dest_array = self.buffer_manager.slot_info_to_view(slot_info)
            if tensor.is_cuda:
                self._copy_d2h(dst=dest_array, src=tensor)
            else:
                # AMG/postprocess outputs can be CPU tensors already.
                np.copyto(dest_array, tensor.numpy())
            outputs_dict[name] = slot_info
        else:
            host_array = self._get_staging_buffer(role, name, tensor)
            host_array.copy_(tensor, non_blocking=True)
            outputs_dict[name] = host_array

    def _get_staging_buffer(
        self, role: str, name: str, src: torch.Tensor
    ) -> torch.Tensor:
        """Cached pinned host staging buffer for a non-buffer tensor.

        One FLAT buffer per (role/name, dtype), grown by capacity doubling and
        viewed to the shape each batch needs -- shapes with a data-dependent
        axis (per-batch detection counts, AMG mask counts) therefore reuse one
        pinned block instead of minting a new one per distinct shape (pinned
        blocks are never reclaimed and each cudaHostAlloc device-syncs).
        Reuse across batches is safe because the subsequent ``.remote()`` call
        serializes the contents into plasma before the next batch overwrites.
        """
        key = (f"{role}/{name}", src.dtype)
        need = src.numel()
        buf = self._staging_buffers.get(key)
        if buf is None or buf.numel() < need:
            capacity = need if buf is None else max(need, 2 * buf.numel())
            buf = torch.empty(
                capacity, dtype=src.dtype, device="cpu", pin_memory=True
            )
            self._staging_buffers[key] = buf
        return buf[:need].view(src.shape)

    def close(self) -> None:
        """Release the pinned staging buffers while the D2H stream still exists.

        torch's caching host allocator records an event on every stream a pinned
        block was used on AT FREE TIME; if the (cupy-owned) d2h stream has
        already been destroyed -- GC teardown order between the buffers and the
        stream is arbitrary -- that records against a dead stream and segfaults.
        Idempotent; called from ``__del__`` (while ``self`` still holds the
        stream, so it is guaranteed alive here).
        """
        buffers = getattr(self, "_staging_buffers", None)
        if buffers:
            try:
                self._cp_d2h_stream.synchronize()
            except Exception:
                pass
            buffers.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # Heartbeat between free-slot wait checks; class attr so tests can shrink it.
    _GET_FREE_HEARTBEAT_S = 120.0

    @staticmethod
    def _drain_worker_alive(worker: Optional[ActorHandle]) -> bool:
        """Probe the actor draining a slot pool. A dead actor raises promptly on
        any method call; a busy-but-alive one times out (its queue is backed up
        behind a long write) -- busy counts as alive.

        Probes ``ping()`` (side-effect free), NOT ``get_metrics()`` -- the
        workers' get_metrics RESETS their accumulated metrics, so probing it
        wiped the save/viz timings on every backpressure heartbeat.
        """
        if worker is None:
            return True  # nothing to probe; keep waiting rather than guess dead
        probe = getattr(worker, "ping", None) or worker.get_metrics
        try:
            ray.get(probe.remote(), timeout=30)
            return True
        except ray.exceptions.GetTimeoutError:
            return True
        except Exception:
            return False

    def _blocking_get_free(
        self, buffer: ActorHandle, pool_name: str, drain_worker: Optional[ActorHandle]
    ):
        """Blocking slot acquisition: patient while the draining worker is alive
        (slow writes to a slow filesystem are backpressure, not failure -- warn
        loudly and keep waiting), fail fast once it is dead (the slot would
        never free; the old fixed 600s timeout killed hours-long runs on
        transient write stalls). ONE get_free request is issued and polled;
        it is ray.cancel'ed before any abort so an abandoned request cannot
        later consume -- and permanently leak -- a slot.
        """
        ref = buffer.get_free.remote()
        waited = 0.0
        while True:
            try:
                return ray.get(ref, timeout=self._GET_FREE_HEARTBEAT_S)
            except ray.exceptions.GetTimeoutError:
                waited += self._GET_FREE_HEARTBEAT_S
                if not self._drain_worker_alive(drain_worker):
                    ray.cancel(ref)
                    raise RuntimeError(
                        f"Waited {waited:.0f}s for a free SHM slot in pool "
                        f"{pool_name!r} and the draining save/viz worker is DEAD "
                        "-- the slot will never free. Check the worker actor logs."
                    )
                ray.logger.warning(
                    f"[Inferencer] still waiting for a free SHM slot in pool "
                    f"{pool_name!r} ({waited:.0f}s); drain worker is alive -- "
                    "write throughput is behind inference (backpressure)."
                )

    def _save_inference_outputs(
        self, data_sample: dict, preds: dict, preds_viz: dict,
    ) -> None:
        """
        Prepare data sample for saving.

        ``preds`` feeds the SAVE path (restored to the original tile frame by
        ``postprocess``); ``preds_viz`` feeds the VIZ path -- the pre-restore
        (processed-resolution) tensors.
        """
        try:
            sample_metainfo: Dict[str, Any] = data_sample["metainfo"]
        except KeyError:
            raise ValueError("data_sample must contain a 'metainfo' key")
        # TODO: Move targets to outer key rather than in metadata
        # That way the dict has all heavy elements at the outer level
        # and we can freely pass metadata via IPC without worrying 
        # about serialization of huge tensors
        targets = sample_metainfo.pop("targets", None)
        # Strict contract validation, NOT a converter (see data/data_types.py):
        # targets must be Form S (List[Dict], length B) or Form D (dict[role ->
        # tensor]). Legacy shapes (the old [targets] wrap, an external
        # collate_fn's dict-of-lists) are rejected loudly, never adapted.
        if targets is not None and not (
            (isinstance(targets, list) and all(isinstance(t, dict) for t in targets))
            or (isinstance(targets, dict) and all(torch.is_tensor(v) for v in targets.values()))
        ):
            raise ValueError(
                "metainfo['targets'] must be contract-shaped -- Form S List[Dict] or "
                "Form D dict[role -> tensor]. Custom "
                f"collate_fns must emit one of these; got {type(targets).__name__}."
            )
        metainfo_numpy = self._tree_to_cpu_numpy(sample_metainfo)
        if self.save_outputs:
            self._attach_save_worker_metainfo(metainfo_numpy)
        if self.vizualize_outputs:
            self._attach_viz_worker_metainfo(metainfo_numpy)
        save_outputs = {
            "metainfo": metainfo_numpy,
        }
        viz_outputs = {
            "metainfo": metainfo_numpy,
        }
        #FIXME: Decide on where targets should be stored
        if targets is not None:
            sample_metainfo["targets"] = targets
            data_sample["targets"] = targets

        should_visualize = self._should_visualize(data_sample, preds)
        if should_visualize and getattr(self, "_viz_sample_idx", None) is not None:
            # by_tile matched a strict subset of the batch: tell the viz worker
            # which records to render (batch-granular gate, per-sample render).
            # Clamp to batch_size_actual: metainfo columns are padded to the
            # full batch, but the worker's records list has only the ACTUAL
            # samples -- a matched padded slot would IndexError downstream.
            bsa = metainfo_numpy.get("batch_size_actual")
            idx = [
                int(i) for i in self._viz_sample_idx
                if bsa is None or int(i) < int(bsa)
            ]
            if idx:
                metainfo_numpy["viz_sample_idx"] = idx
            else:   
                should_visualize = False   # only padded slots matched
        should_save = self.save_outputs

        t_total0 = time.perf_counter()
        save_buffer_slots = {}
        viz_buffer_slots = {}
        
        """
        Optional save/viz skipping based on user config and buffer pool back-pressure:
        
            Reserve a shared-memory slot for each buffer, from separate `_save` and `_viz` pools.
            block_on_(save/viz):
                - True: blocks until a slot frees up
                - False: if the pool is full, skip sample's save/viz
                    
            D2H stream which waits on the compute stream and copies each tensor:
                - buffered: async cudaMemcpyAsync into its SHM slot and store its handle
                - unbuffered: pinned host copy and sent later via Ray's plasma store
                
            Hand the slots to the Save/Viz actors by reference (arrays stay in SHM)
            Slots are freed exactly once by the actor on success, or by finally on failure
        """
        try:
            if self.save_outputs:
                for output_tensor_name in self.outputs_metadata["save_tensors"]:
                    if output_tensor_name in (self.outputs_metadata.get("buffer_tensors") or ()):
                        t0_buffer = time.perf_counter()
                        save_buffer = self.buffer_manager.get_buffer(f"{output_tensor_name}_save")
                        if self.block_on_save:
                            slot_info = self._blocking_get_free(
                                save_buffer, f"{output_tensor_name}_save", self.save_worker
                            )
                            if slot_info is None:
                                raise RuntimeError(f"No free slot found for {output_tensor_name}")
                        else:
                            slot_info = ray.get(save_buffer.try_get_free.remote())

                        if slot_info is None:
                            self._dropped_saves += 1
                            ray.logger.error(
                                f"Save pool exhausted for {output_tensor_name}; DROPPING this "
                                f"tile's predictions ({self._dropped_saves} dropped so far). "
                                "finalize() will fail because the output is incomplete."
                            )
                            should_save = False
                            for allocated_slot in save_buffer_slots.values():
                                self.buffer_manager.free_slot(allocated_slot)
                            save_buffer_slots.clear()
                            break
                        save_buffer_slots[output_tensor_name] = slot_info
                        self._metrics[f"buffer_get_time_ms/{output_tensor_name}_save"] = (time.perf_counter() - t0_buffer) * 1000

            if should_visualize and self.vizualize_outputs:
                for output_tensor_name in self.outputs_metadata["visualize_tensors"]:
                    if output_tensor_name in (self.outputs_metadata.get("buffer_tensors") or ()):
                        t0_buffer = time.perf_counter()
                        viz_buffer = self.buffer_manager.get_buffer(f"{output_tensor_name}_viz")
                        if self.block_on_viz:
                            slot_info = self._blocking_get_free(
                                viz_buffer, f"{output_tensor_name}_viz", self.viz_worker
                            )
                            if slot_info is None:
                                raise RuntimeError(f"No free slot found for {output_tensor_name}")
                        else:
                            slot_info = ray.get(viz_buffer.try_get_free.remote())

                        if slot_info is None:
                            ray.logger.warning(f"No free slot found for {output_tensor_name} in viz buffer. Skipping visualization.")
                            should_visualize = False
                            for allocated_slot in viz_buffer_slots.values():
                                self.buffer_manager.free_slot(allocated_slot)
                            viz_buffer_slots.clear()
                            break
                        viz_buffer_slots[output_tensor_name] = slot_info
                        self._metrics[f"buffer_get_time_ms/{output_tensor_name}_viz"] = (time.perf_counter() - t0_buffer) * 1000
            self._metrics["buffer_get_time_ms_total"] = (time.perf_counter() - t_total0) * 1000
            t0_transfer = time.perf_counter()

            # d2h stream must wait for the compute stream that produced `preds` before any changes to those tensors
            compute_stream = torch.cuda.current_stream(device=self.device)
            with torch.cuda.stream(self._d2h_stream):
                self._d2h_stream.wait_stream(compute_stream)
                if should_save:
                    for output_tensor_name in self.outputs_metadata["save_tensors"]:
                        if output_tensor_name in preds.keys():
                            output_tensor = preds[output_tensor_name]
                        elif output_tensor_name in data_sample.keys():
                            output_tensor = data_sample[output_tensor_name]
                        else:
                            raise ValueError(f"Tensor {output_tensor_name} not found in preds or data_sample")
                        if output_tensor is None:
                            continue
                        self._stage_output("save", output_tensor_name, output_tensor,
                                           save_buffer_slots, save_outputs)

                if should_visualize and self.vizualize_outputs:
                    # Viz renders the PROCESSED frame: preds_viz holds the pre-restore
                    # tensors at working resolution, staged straight into the viz pool.
                    for output_tensor_name in self.outputs_metadata["visualize_tensors"]:
                        if output_tensor_name in preds_viz.keys():
                            output_tensor = preds_viz[output_tensor_name]
                        elif output_tensor_name in data_sample.keys():
                            output_tensor = data_sample[output_tensor_name]
                        else:
                            raise ValueError(f"Output {output_tensor_name} not found in preds or data_sample")
                        if output_tensor is None:
                            continue
                        self._stage_output("viz", output_tensor_name, output_tensor,
                                           viz_buffer_slots, viz_outputs)

            self._cp_d2h_stream.synchronize()
            self._metrics["buffer_transfer_time_ms"] = (time.perf_counter() - t0_transfer) * 1000
            # TODO: Ideally we do this async with buffer tensors because if we block on a buffer
            # get slot call we can still transfer these data in the meantime. 
            # Final pass: any remaining nested CUDA tensors (e.g. in metainfo) and
            # non-buffer prediction tensors must be CPU NumPy before Ray serialization.
            t0_cpu = time.perf_counter()
            save_outputs = self._tree_to_cpu_numpy(save_outputs)
            viz_outputs = self._tree_to_cpu_numpy(viz_outputs)
            self._metrics["tree_to_cpu_transfer_time_ms"] = (time.perf_counter() - t0_cpu) * 1000

            if self.save_outputs and should_save:
                if self.save_worker is None:
                    raise RuntimeError("Attempting to save outputs but save_worker is None")
                save_task = self.save_worker.save.remote(
                    inference_outputs=save_outputs,
                    queue_t0=time.perf_counter(),
                )
                self._tasks.append(save_task)
                save_buffer_slots = {}

            if should_visualize and self.vizualize_outputs:
                if self.viz_worker is None:
                    raise RuntimeError("Attempting to visualize outputs but viz_worker is None")
                # data_tensor must come through a SHM viz pool like every other
                # viz tensor (list it under visualize_tensors + buffer_tensors);
                # validated at __init__ -- no silent plasma fallback here.
                # The targets guard is defensive (targets is not buffered today).
                if targets is not None and "targets" not in viz_outputs:
                    viz_outputs["targets"] = self._tree_to_cpu_numpy(targets)
                vis_task = self.viz_worker.visualize.remote(
                    inference_outputs=viz_outputs,
                    queue_t0=time.perf_counter(),
                )
                self._tasks.append(vis_task)
                viz_buffer_slots = {}
        finally:
            if save_buffer_slots or viz_buffer_slots:
                # Exception path: async D2H copies issued for earlier tensors may
                # still be in flight into slots we are about to free. Safe today
                # via single-stream serialization, but sync defensively so a
                # freed-and-reused slot can never race an in-flight memcpy.
                try:
                    self._cp_d2h_stream.synchronize()
                except Exception:
                    pass
            for slot_info in save_buffer_slots.values():
                self.buffer_manager.free_slot(slot_info)
            for slot_info in viz_buffer_slots.values():
                self.buffer_manager.free_slot(slot_info)

    def get_metrics(self) -> Dict[str, float]:
        metrics = self._metrics.copy()
        self._metrics = {
        }
        return metrics

    def finalize(self, overall_deadline_s: float = 1800.0):
        n_tasks = len(self._tasks)
        errors = []
        # Per-task 300s AND an overall deadline: N stalled tasks serially
        # waiting 300s each would block teardown for N*300s; past the overall
        # deadline the remaining tasks are recorded as failed instead.
        deadline = time.perf_counter() + overall_deadline_s
        for i, task in enumerate(self._tasks):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                errors.append((i, TimeoutError(
                    f"finalize overall deadline ({overall_deadline_s:.0f}s) exhausted"
                )))
                continue
            try:
                ray.get(task, timeout=min(300.0, remaining))
            except Exception as e:
                errors.append((i, e))
                ray.logger.error(f"Task {i}/{n_tasks} failed: {e}")
        self._tasks.clear()
        # Reap outstanding put_free refs: a producer-side double-free raises
        # HERE (visible in the run's failure) instead of dying as a background
        # log line after teardown.
        try:
            self.buffer_manager.drain_free_refs()
        except Exception as e:
            errors.append(("drain_free_refs", e))
            ray.logger.error(f"Outstanding slot frees failed at finalize: {e}")
        torch.cuda.synchronize()
        barrier()
        if errors:
            raise RuntimeError(f"{errors}\n{len(errors)}/{n_tasks} save/viz tasks failed.")
        if self._dropped_saves:
            raise RuntimeError(
                f"{self._dropped_saves} tiles were dropped from the save path "
                "(block_on_save=false + save pool exhaustion) -- the persisted "
                "output is INCOMPLETE. Increase the save pool capacity or set "
                "block_on_save=true."
            )
