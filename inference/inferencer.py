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
logging.basicConfig(level=logging.INFO)

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
        ray.logger.info(f"Main output metadata: {self.outputs_metadata}")
        ray.logger.info(f"Aggregate mode: {self.aggregate_mode}")

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

        X = data_sample["data_tensor"]
        metadata = data_sample["metainfo"]
        t0_predict = time.perf_counter()
        preds = self._predict(X, data_sample)
        self._metrics["predict_time_ms"] = (time.perf_counter() - t0_predict) * 1000

        # Restore dense predictions to the original tile resolution (tile-mode
        # resize undo) and rescale boxes to the original coordinate frame.
        t0_restore = time.perf_counter()
        preds = postprocess(preds, data_sample, self.outputs_metadata)
        self._metrics["restore_outputs_time_ms"] = (time.perf_counter() - t0_restore) * 1000

        B = len(metadata["prepared_id"])
        targets = metadata.get("targets", None)
        if targets is None:
            targets = [{} for _ in range(B)]
        elif isinstance(targets, dict):
            targets = [targets for _ in range(B)]
        elif isinstance(targets, (list, tuple)) and len(targets) == 1 and isinstance(targets[0], (list, tuple)):
            targets = targets[0]
        t0_save = time.perf_counter()
        self._save_inference_outputs(
            data_sample=data_sample,
            preds=preds,
        )
        self._metrics["save_inference_outputs_time_ms"] = (time.perf_counter() - t0_save) * 1000

    def _should_visualize(self, data_sample: dict, preds: dict) -> bool:
        """
        Check if the data sample should be visualized
        """
        # TODO: Should this be moved to the viz worker?
        # I'm leaving here because this way we don't have to flood the viz worker 
        # with data samples if we are using a sampling policy.
        if self.viz_sampling_policy is None:
            return True
        if self.viz_sampling_policy["name"] == "by_tile":
            return data_sample["metainfo"]["tile_name"] in self.viz_sampling_policy["tile_names"]
        if self.viz_sampling_policy["name"] == "random_sample":
            return np.random.rand() < self.viz_sampling_policy["fraction"]
        return False

    def _copy_d2h(self, dst: np.ndarray, src: torch.Tensor) -> None:
        """Async device-to-host memcpy via CuPy, mirroring CollatorActor.copy_h2d.

        Guards against host-memory corruption: the destination SHM slot is sized
        from the declared ``tensor_info`` (DB maxima x declared C/dtype), so a
        model output whose channel/time count or dtype disagrees with that
        declaration must fail loudly here rather than overrun the slot. This
        fail-hard guard is the sole runtime check on the output contract.
        """
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

    def _save_inference_outputs(self, data_sample: dict, preds: dict) -> None:
        """
        Prepare data sample for saving
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
        should_save = self.save_outputs

        t0_buffer = time.perf_counter()
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
                            slot_info = ray.get(save_buffer.get_free.remote())
                            if slot_info is None:
                                raise RuntimeError(f"No free slot found for {output_tensor_name}")
                        else:
                            slot_info = ray.get(save_buffer.try_get_free.remote())

                        if slot_info is None:
                            ray.logger.warning(f"No free slot found for {output_tensor_name} in save buffer. Skipping save for this sample.")
                            should_save = False
                            for allocated_slot in save_buffer_slots.values():
                                self.buffer_manager.free_slot(allocated_slot)
                            save_buffer_slots.clear()
                            break
                        save_buffer_slots[output_tensor_name] = slot_info
                        self._metrics[f"buffer_get_time_ms/{output_tensor_name}_save"] = (time.perf_counter() - t0_buffer) * 1000

            if should_visualize and self.vizualize_outputs:
                t0_buffer = time.perf_counter()
                for output_tensor_name in self.outputs_metadata["visualize_tensors"]:
                    if output_tensor_name in (self.outputs_metadata.get("buffer_tensors") or ()):
                        viz_buffer = self.buffer_manager.get_buffer(f"{output_tensor_name}_viz")
                        if self.block_on_viz:
                            slot_info = ray.get(viz_buffer.get_free.remote())
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
            self._metrics["buffer_get_time_ms_total"] = (time.perf_counter() - t0_buffer) * 1000
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

                        output_tensor_dtype_str = self.outputs_metadata["tensor_info"][output_tensor_name]["dtype"]
                        output_tensor_dtype = TORCH_DTYPES[output_tensor_dtype_str].value if isinstance(output_tensor_dtype_str, str) else output_tensor_dtype_str
                        output_tensor = output_tensor.to(dtype=output_tensor_dtype)

                        if output_tensor_name in save_buffer_slots.keys():
                            slot_info = save_buffer_slots[output_tensor_name]
                            dest_array = self.buffer_manager.slot_info_to_view(slot_info)
                            self._copy_d2h(dst=dest_array, src=output_tensor)
                            save_outputs[output_tensor_name] = slot_info
                        else:
                            host_array = torch.empty_like(output_tensor, device="cpu", pin_memory=True)
                            host_array.copy_(output_tensor, non_blocking=True)
                            save_outputs[output_tensor_name] = host_array

                if should_visualize and self.vizualize_outputs:
                    for output_tensor_name in self.outputs_metadata["visualize_tensors"]:
                        if output_tensor_name in preds.keys():
                            output_tensor = preds[output_tensor_name]
                        elif output_tensor_name in data_sample.keys():
                            output_tensor = data_sample[output_tensor_name]
                        else:
                            raise ValueError(f"Output {output_tensor_name} not found in preds or data_sample")

                        if output_tensor is None:
                            continue

                        output_tensor_dtype_str = self.outputs_metadata["tensor_info"][output_tensor_name]["dtype"]
                        output_tensor_dtype = TORCH_DTYPES[output_tensor_dtype_str].value if isinstance(output_tensor_dtype_str, str) else output_tensor_dtype_str
                        output_tensor = output_tensor.to(dtype=output_tensor_dtype)

                        if output_tensor_name in viz_buffer_slots.keys():
                            slot_info = viz_buffer_slots[output_tensor_name]
                            dest_array = self.buffer_manager.slot_info_to_view(slot_info)
                            self._copy_d2h(dst=dest_array, src=output_tensor)
                            viz_outputs[output_tensor_name] = slot_info
                        else:
                            host_array = torch.empty_like(output_tensor, device="cpu", pin_memory=True)
                            host_array.copy_(output_tensor, non_blocking=True)
                            viz_outputs[output_tensor_name] = host_array

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
                # Invariant: only attach a host copy if the key was NOT already
                # buffered into viz_outputs above. When data_tensor/targets are
                # listed in visualize_tensors+buffer_tensors they already hold a
                # SHM slot_info here; overwriting it with a host array orphans the
                # slot (viz_buffer_slots is cleared at dispatch, so finally never
                # frees it) -> pool exhaustion/stall under block_on_viz. The
                # targets guard is defensive (targets is not buffered today).
                if "data_tensor" in data_sample and "data_tensor" not in viz_outputs:
                    viz_outputs["data_tensor"] = self._tree_to_cpu_numpy(data_sample["data_tensor"])
                if targets is not None and "targets" not in viz_outputs:
                    viz_outputs["targets"] = self._tree_to_cpu_numpy(targets)
                vis_task = self.viz_worker.visualize.remote(
                    inference_outputs=viz_outputs,
                    queue_t0=time.perf_counter(),
                )
                self._tasks.append(vis_task)
                viz_buffer_slots = {}
        finally:
            for slot_info in save_buffer_slots.values():
                self.buffer_manager.free_slot(slot_info)
            for slot_info in viz_buffer_slots.values():
                self.buffer_manager.free_slot(slot_info)

    def get_metrics(self) -> Dict[str, float]:
        metrics = self._metrics.copy()
        self._metrics = {
        }
        return metrics

    def finalize(self):
        n_tasks = len(self._tasks)
        errors = []
        for i, task in enumerate(self._tasks):
            try:
                ray.get(task, timeout=300)
            except Exception as e:
                errors.append((i, e))
                ray.logger.error(f"Task {i}/{n_tasks} failed: {e}")
        self._tasks.clear()
        torch.cuda.synchronize()
        barrier()
        if errors:
            raise RuntimeError(f"{errors}\n{len(errors)}/{n_tasks} save/viz tasks failed.")
