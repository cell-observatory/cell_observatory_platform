import os
import time
import math
import logging
import functools
from typing import Any, Dict, Optional, Tuple, List, Callable, Mapping, Union
from dataclasses import dataclass, asdict, field

import numpy as np

import torch

from omegaconf import DictConfig
from hydra.utils import get_method, instantiate

from cell_observatory_platform.data.data_types import (
    TORCH_DTYPES,
    DataKind,
    get_role,
    parse_annotations_metadata,
)
from cell_observatory_platform.data.io import read_file
from cell_observatory_platform.data.structures import convert_bbox_format, mask_ids_to_masks
from cell_observatory_platform.data.utils import create_na_masks, downsample, resize_mask
from cell_observatory_platform.data.datasets.utils import _parse_channel_mapping
from cell_observatory_platform.models.layers.patch_embeddings import PatchEmbedding, calc_num_patches
from cell_observatory_platform.training.helpers import get_patch_sizes, make_timing_metric
from cell_observatory_platform.utils.shape_format import get_spatial_shape
from cell_observatory_platform.utils.registry import REGISTRY
from cell_observatory_platform.utils.config import registers_as
from cell_observatory_platform.data.databases.local_metadata_store import is_object_role

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Channel role partition (transitional; the DB will own per-channel roles)
# --------------------------------------------------------------------------- #

def _object_type_names_from_meta(meta: Any) -> dict[int, str]:
    """``object_type_id -> nk``, as written into metainfo by the collator.
    """
    if not isinstance(meta, Mapping):
        return {}
    raw = meta.get("object_type_names") or {}
    if isinstance(raw, (list, tuple)):  # per-sample column convention
        raw = raw[0] if raw else {}
    return {int(k): str(v) for k, v in dict(raw).items()}


def _first_annotations_metadata(meta: Any) -> Any:
    """The per-batch annotations_metadata payload, as one row.
    """
    if not isinstance(meta, Mapping):
        return None
    raw = meta.get("annotations_metadata")
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)):
        return raw[0] if raw else None
    return raw


def _channel_mapping_from_meta(meta: Any) -> Optional[dict]:
    """Pull the ``{channel_index -> role/token}`` mapping out of metainfo.

    Tolerates the platform ``List[...]``/ndarray per-sample convention (one entry
    per batch element) and missing keys. When a per-sample sequence is present,
    all entries MUST be identical: the partition derived from it is applied to
    the whole batch, so a heterogeneous batch would silently mis-split every
    sample but the first. JSON-string payloads (the DB column format) are parsed.
    """
    if not isinstance(meta, Mapping):
        return None
    cm = meta.get("channel_mapping")
    if cm is None:
        return None
    if isinstance(cm, np.ndarray):
        cm = cm.tolist()
    if isinstance(cm, (list, tuple)):
        if not cm:
            return None
        first = cm[0]
        for i, entry in enumerate(cm[1:], start=1):
            if entry != first:
                raise ValueError(
                    "channel_mapping is not homogeneous across the batch: "
                    f"row 0 = {first!r} but row {i} = {entry!r}. The channel "
                    "partition is computed once per batch, so mixed mappings "
                    "would silently mis-split every sample but the first."
                )
        cm = first
    if isinstance(cm, (str, bytes)) or hasattr(cm, "as_py"):
        parsed = _parse_channel_mapping(cm)
        return dict(parsed) if parsed else None
    if isinstance(cm, Mapping):
        return dict(cm)
    return None


# --------------------------------------------------------------------------- #
# General role-driven channel partition
# --------------------------------------------------------------------------- #

@dataclass
class ChannelPartition:
    input_idxs: list[int] = field(default_factory=list)
    targets_by_role: dict[str, list[int]] = field(default_factory=dict)
    dropped_idxs: list[int] = field(default_factory=list)


def _role_matches_target(role: str, target_roles) -> bool:
    return any(role == t or role.startswith(t + "_") for t in target_roles)


def partition_channels(channel_mapping, num_channels, target_roles):
    """Partition channels into INPUT / TARGET(by role) / DROPPED, driven solely by
    the channel_mapping role table.

      role matches a target role/family           -> TARGET (kept as GT, keyed by concrete role)
      role is an object role but not consumed     -> DROPPED (discarded)
      otherwise                                   -> INPUT (signal -> model)

    With no channel_mapping every channel is INPUT (pre-DB data / recon).
    """
    p = ChannelPartition()
    cm = channel_mapping or {}
    # normalize {idx:role}; tolerate the platform List[...] wrapper handled by callers
    role_by_idx = {int(i): r for i, r in cm.items()}
    for idx in range(num_channels):
        role = role_by_idx.get(idx)
        if role is not None and is_object_role(role):
            if _role_matches_target(role, target_roles):
                p.targets_by_role.setdefault(role, []).append(idx)
            else:
                p.dropped_idxs.append(idx)
        else:
            # A role that matches a target family but is NOT an object role is
            # always a config/DB bug -- it would silently become model input.
            if role is not None and _role_matches_target(role, target_roles):
                raise ValueError(
                    f"channel {idx} role {role!r} matches a target family but is not "
                    f"classified as an object role (is_object_role() == False); "
                    f"refusing to treat GT as model input. Fix the DB channel role "
                    f"or the object-family table in local_metadata_store."
                )
            p.input_idxs.append(idx)
    return p


# --------------------------------------------------------------------------- #
# Pretraining preprocessor
# --------------------------------------------------------------------------- #


def _reads_raw_counts(transforms) -> bool:
    """True if any configured transform needs exact integer counts.

    Transforms opt in with a ``reads_raw_counts`` class attribute rather than
    being named in a list here, so a new one declares its own requirement.
    Recurses into wrapper transforms (ProbabilisticChoice holds its children in
    ``.transforms``): a nested sensor-model transform needs the exact counts just
    as much as a top-level one.
    """
    for transform in transforms or ():
        if getattr(transform, "reads_raw_counts", False):
            return True
        nested = getattr(transform, "transforms", None)
        if nested and _reads_raw_counts(nested):
            return True
    return False


@registers_as("preprocessor", "ray")
class RayPreprocessor(torch.nn.Module):
    def __init__(
        self,
        dtype: torch.dtype,
        with_masking: bool,
        input_format: str,
        input_shape: tuple[int, ...],
        patch_shape: tuple[int, int, int],
        mask_generator,
        transforms_list=None,
    ):
        super().__init__()
        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype
        self.with_masking = with_masking
        self.mask_generator = mask_generator
        self.transforms = []
        for t in transforms_list or []:
            if isinstance(t, DictConfig):
                # not yet instantiated
                self.transforms.append(instantiate(t))
            elif isinstance(t, str):
                # a dotted‑path string
                self.transforms.append(get_method(t))
            else:
                # already an instantiated callable object
                self.transforms.append(t)

        # uint16 counts are not exactly representable below fp32 (bfloat16 is
        # exact to 256, float16 to 2048 and overflows at 65504), so an fp32
        # intermediate is kept ONLY when a transform reads the counts. Otherwise
        # the data is cast once, straight to the model dtype, halving resident
        # activation memory. A non-floating self.dtype falls back to fp32, since
        # casting to it would leave integer tensors flowing into the transforms.
        self._count_dtype = (
            torch.float32
            if _reads_raw_counts(self.transforms) or not self.dtype.is_floating_point
            else self.dtype
        )

        self.input_shape = input_shape
        self.patch_shape = patch_shape
        self.input_format = input_format
        self.num_patches, self._token_shape = calc_num_patches(
            input_fmt=self.input_format,
            input_shape=self.input_shape,
            patch_shape=self.patch_shape,
        )

        assert self.input_format in ["TZYXC", "ZYXC"], f"Input format {self.input_format} not supported yet."
        spatial_token_shape = [s for s in self._token_shape[:-1] if s is not None]
        self.spatial_token_shape = spatial_token_shape

        # The dense ``data_tensor`` entry shared by every task's ``_data_types``.
        self.base_dense_data_type = {
            "kind": "dense",
            "layout": self.input_format,
            "role": "input",
            "has_time": "T" in self.input_format,
        }

    def _build_spatial_kwargs(self, batch_size: int, device: torch.device) -> dict:
        spatial_shapes = torch.tensor(
            [self.spatial_token_shape], dtype=torch.long, device=device,
        )
        level_start_index = torch.zeros(1, dtype=torch.long, device=device)
        valid_ratios = torch.ones(
            batch_size, 1, len(self.spatial_token_shape), device=device,
        )
        tokens_per_level = [int(math.prod(self.spatial_token_shape))]
        return {
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
            "valid_ratios": valid_ratios,
            "tokens_per_level": tokens_per_level,
        }

    def _data_types(self) -> Dict[str, Dict[str, Any]]:
        """Single declaration of this stream's fields, keyed by field name.

        Each entry is ``{"kind", "layout", "role", ("channel_role")}``; it is the
        one source of truth the transforms inspect (``metainfo["data_types"]``) to
        decide 3D/4D dispatch and per-field ops. The base case declares only the
        dense ``data_tensor`` (``self.base_dense_data_type``) -- all the
        reconstruction tasks (MAE/JEPA/denoising/channel-split/upsample) need,
        since their targets are derived from the resized image. Task preprocessors
        that carry spatial GT through the transforms (seg/det/SAM) override this to
        add their target fields.
        """
        return {"data_tensor": self.base_dense_data_type}

    def _assert_input_shape_spatial(self, tensor: torch.Tensor) -> None:
        """Fail fast if the (post-transform) spatial shape != ``input_shape``.

        Patch/token grids and the mask generator are sized from ``input_shape``,
        so a Resize target (or any spatial transform) MUST land on it.
        """
        expected = get_spatial_shape(tuple(self.input_shape), self.input_format)
        actual = get_spatial_shape(tuple(tensor.shape[1:]), self.input_format)
        if tuple(actual) != tuple(expected):
            raise ValueError(
                f"{type(self).__name__}: post-transform spatial shape {actual} "
                f"!= input_shape spatial {expected}. Patch/token grids and "
                f"masking are sized from input_shape; a Resize target must "
                f"match it."
            )

    def forward(self, data_sample: dict, data_time: float, idx: int) -> dict:
        """
        Preprocess the input data sample to maintain uniform data
        layout accross data loaders.
        """
        preprocess_time = time.time()

        inputs = data_sample["data_tensor"]
        meta = data_sample["metainfo"]

        if inputs.dtype == torch.uint16:
            # fp32 when a transform reads raw counts, else straight to the model
            # dtype (see self._count_dtype). The fp32 path narrows to self.dtype
            # at the END below; the gated path makes that narrowing a no-op.
            inputs = inputs.to(self._count_dtype)
        elif inputs.dtype != self.dtype:
            # ray.logger.warning(f"Casting inputs to {self.dtype}")
            inputs = inputs.to(self.dtype)

        # skipping checks for NaN/Inf values
        # if torch.isnan(inputs).all() or torch.isinf(inputs).all():
        #     raise ValueError(f"Invalid training data")

        transform_time = -1.0
        if self.transforms:
            transform_t0 = time.time()
            # Route a dict (not a bare tensor) so transforms like Resize can read
            # image_sizes (crop_to_valid) and the declared data_types.
            meta = dict(meta)
            meta["data_types"] = self._data_types()
            sample: Any = {"data_tensor": inputs, "metainfo": meta}
            for transform in self.transforms:
                sample = transform(sample)
            inputs, meta = sample["data_tensor"], sample["metainfo"]
            transform_time = time.time() - transform_t0

        self._assert_input_shape_spatial(inputs)

        # Narrow the float32 intermediate to the model dtype at the boundary
        # (the transforms above ran on exact counts).
        if inputs.dtype != self.dtype:
            inputs = inputs.to(self.dtype)

        masking_time = -1.0
        mask_lists: dict[str, Any] = {}
        if self.with_masking:
            mt0 = time.time()
            mask_data = self.mask_generator(inputs.shape[0])
            masking_time = time.time() - mt0
            mask_lists = {k: [v] for k, v in mask_data.items() if v is not None}

        metrics: list[dict[str, Any]] = [
            make_timing_metric("data_time", data_time),
            make_timing_metric("preprocess_time", time.time() - preprocess_time),
            make_timing_metric("transform_time", transform_time),
            make_timing_metric("masking_time", masking_time),
        ]
        existing_metrics = meta.get("metrics") if isinstance(meta, Mapping) else None
        if existing_metrics:
            metrics = list(existing_metrics) + metrics

        metainfo = {
            **mask_lists,
            **meta,
            "idx": idx,
            "metrics": metrics,
        }
        metainfo["spatial_kwargs"] = self._build_spatial_kwargs(
            batch_size=inputs.shape[0], device=inputs.device,
        )
        return {"data_tensor": inputs, "metainfo": metainfo}


@dataclass(frozen=True)
class InputDataStreamSpec:
    """
    Per-dataset stream metadata needed to compute patch/token info and dtype casting.
    This is used to normalize the input data streams and ensure they are compatible with the preprocessor.
    """
    input_format: str
    input_shape: Tuple[int, ...]
    patch_shape: Tuple[int, ...]
    dtype: Union[str, torch.dtype] = "bfloat16"
    with_masking: bool = False


def _instantiate_transform_list(transforms_list: Optional[List[Any]]) -> List[Callable]:
    out: List[Callable] = []
    for t in transforms_list or []:
        if isinstance(t, DictConfig):
            out.append(instantiate(t))
        elif isinstance(t, str):
            out.append(get_method(t))
        else:
            out.append(t)
    return out


# @REGISTRY.register("preprocessor", "multi_sequence")
# class MultiSequenceRayPreprocessor(torch.nn.Module):
#     """
#     Batch preprocessor for multiple dataset streams 
#     (e.g., DINO global/local crops, or multiple datasets with different shapes).

#     Output:
#       {
#         "data_tensors": {name: tensor, ...},
#         "metainfo": {
#             ... original meta ...,
#             "dataset_stream_metainfo": {
#                 name: {
#                     "input_format": ...,
#                     "input_shape": ...,
#                     "patch_shape": ...,
#                     "transform_time": float,
#                     "masking_time": float,
#                     # masking outputs if enabled:
#                     "masks": [Tensor] (optional),
#                     "context_masks": [Tensor] (optional),
#                     "target_masks": [Tensor] (optional),
#                     "original_patch_indices": [Tensor] (optional),
#                     "channels_to_mask": [Any] (optional),
#                     "patches_used": [Tensor] (optional),
#                 },
#             },
#             "preprocess_time": float,
#             "data_time": float,
#         }
#       }
#     """

#     def __init__(
#         self,
#         local_batch_size: int,
#         global_batch_size: int,
#         input_metadict: Mapping[str, Union[InputDataStreamSpec, Mapping[str, Any]]],
#         transforms_metadict: Optional[Mapping[str, Optional[List[Any]]]] = None,
#         mask_generators: Optional[Mapping[str, Any]] = None,
#         dtype: torch.dtype | str = "bfloat16",
#     ):
#         super().__init__()

#         # TODO: decide if this should be property of Preprocessor or DataLoader
#         self.local_batch_size = local_batch_size
#         self.global_batch_size = global_batch_size

#         self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype

#         data_streams: Dict[str, InputDataStreamSpec] = {}
#         for name, spec in input_metadict.items():
#             if isinstance(spec, InputDataStreamSpec):
#                 data_streams[name] = spec
#             else:
#                 data_streams[name] = InputDataStreamSpec(**dict(spec))
#         self.data_streams = data_streams

#         # Per-dataset stream transforms
#         self.transforms: Dict[str, List[Callable]] = {}
#         transforms_metadict = transforms_metadict or {}
#         for name in self.data_streams.keys():
#             self.transforms[name] = _instantiate_transform_list(transforms_metadict.get(name, None))
#         # NOTE: for single dataset stream, we use the BASE transforms pipeline to create the dataset stream tensors
#         #       for example, see multicrop augmentation for dino
#         self.transforms["BASE"] = _instantiate_transform_list(transforms_metadict.get("BASE", None))

#         # Per-dataset stream mask generators
#         self.mask_generators = dict(mask_generators or {})

#     def _apply_dataset_stream_transforms(
#         self,
#         name: str,
#         x: torch.Tensor,
#         meta: Dict[str, Any],
#     ) -> Tuple[torch.Tensor, Dict[str, Any], float]:
#         """
#         Apply transforms for a given dataset stream.
#         """
#         t0 = time.time()
#         data_dict: Dict[str, Any] = {"data_tensor": x, "metainfo": meta}

#         for tr in self.transforms.get(name, []):
#             out = tr(data_dict)
#             if isinstance(out, torch.Tensor):
#                 data_dict["data_tensor"] = out
#             elif isinstance(out, dict):
#                 if "data_tensor" not in out:
#                     raise KeyError(f"Transform for dataset stream={name!r} returned dict without 'data_tensor'.")
#                 data_dict = out
#                 data_dict.setdefault("metainfo", {})
#             else:
#                 raise TypeError(
#                     f"Transform for dataset stream={name!r} must return Tensor or dict; got {type(out)}."
#                 )

#         return data_dict["data_tensor"], data_dict.get("metainfo", {}), (time.time() - t0)

#     def _apply_dataset_stream_masking(
#         self,
#         name: str,
#         x: torch.Tensor,
#     ) -> Tuple[Dict[str, Any], float]:
#         """
#         Calls the dataset stream's mask generator if enabled.

#         Expects mask generator to return a dict with optional keys (unused as None).
#         Values are wrapped in single-element lists and merged into the stream's metainfo.
#         """
#         spec = self.data_streams[name]
#         if not spec.with_masking:
#             return {}, -1.0

#         mask_gen = self.mask_generators.get(name, None)
#         if mask_gen is None:
#             raise ValueError(f"with_masking=True for stream={name!r} but no mask generator was provided.")

#         B = x.shape[0]
#         mt0 = time.time()
#         mask_data = mask_gen(B)
#         masking_time = time.time() - mt0

#         if not isinstance(mask_data, dict):
#             raise TypeError(
#                 f"Mask generator for stream={name!r} returned {type(mask_data)}, expected dict."
#             )
#         mask_kwargs = {k: [v] for k, v in mask_data.items() if v is not None}
#         return mask_kwargs, masking_time

#     def forward(self, data_sample: Any, data_time: float, idx: int):
#         preprocess_t0 = time.time()

#         if isinstance(data_sample, dict):
#             # TODO: dataset, dataloader, and database stack currently does not support this branch properly
#             if "data_tensors" in data_sample and isinstance(data_sample["data_tensors"], dict):
#                 input_mode = "multi_dataset_streams"
#                 # NOTE: we match the tensor and metadict names to the outputs of the base transforms
#                 # in the single_dataset_stream mode
#                 dataset_streams_tensors = data_sample["data_tensors"]
#                 dataset_streams_meta = data_sample.get("metainfo", {})
#             elif "data_tensor" in data_sample and torch.is_tensor(data_sample["data_tensor"]):
#                 input_mode = "single_dataset_stream"
#                 in_tensors = data_sample["data_tensor"]
#                 input_dataset_stream_metainfo = data_sample.get("metainfo", {})
#             else:
#                 raise KeyError(
#                     "MultiSequenceRayPreprocessor expects either:\n"
#                     "  - {'data_tensors': {name: tensor, ...}, 'metainfo': {...}}  OR\n"
#                     "  - {'data_tensor': tensor, 'metainfo': {...}}  OR\n"
#                 )
#         else:
#             raise TypeError(f"data_sample must be a dict; got {type(data_sample)}")

#            # Preserve collator/dataloader metainfo (e.g. device_buffer_idx, host_buffer_idx) for hooks
#         if input_mode == "single_dataset_stream":
#             original_metainfo = dict(input_dataset_stream_metainfo)
#         else:
#             original_metainfo = dict(data_sample.get("metainfo", {}))

#         # If we have 1 dataset stream, apply base transforms to generate a dict-of-tensors
#         if input_mode == "single_dataset_stream":
#             if not torch.is_tensor(in_tensors):
#                 raise TypeError(f"data_tensor must be a torch.Tensor; got {type(in_tensors)}")

#             base_dtype = TORCH_DTYPES[self.dtype].value if isinstance(self.dtype, str) else self.dtype
#             if in_tensors.dtype != base_dtype:
#                 in_tensors = in_tensors.to(base_dtype)

#             base_transforms = self.transforms.get("BASE", [])
#             assert base_transforms, "single_dataset_stream mode requires a BASE transforms pipeline."

#             data_dict: Dict[str, Any] = {"data_tensor": in_tensors, "metainfo": dict(input_dataset_stream_metainfo)}
#             t_base0 = time.time()
#             for tr in base_transforms:
#                 data_dict = tr(data_dict)

#             base_transform_time = time.time() - t_base0

#             # Interpret output as dataset streams
#             if "data_tensor" in data_dict and isinstance(data_dict["data_tensor"], dict):
#                 dataset_streams_tensors = data_dict["data_tensor"]
#                 dataset_streams_meta = data_dict.get("metainfo", {})
#             else:
#                 raise KeyError(
#                     "Base transforms pipeline must create streams by returning:\n"
#                     "  - {'data_tensors': {name: tensor, ...}, 'metainfo': ...} "
#                     f"Got keys={list(data_dict.keys())} and data_tensor type={type(data_dict.get('data_tensor', None))}."
#                 )
#         else:
#             base_transform_time = 0.0

#         if not isinstance(dataset_streams_tensors, dict):
#             raise TypeError(f"Expected dict-of-tensors, got {type(dataset_streams_tensors)}")

#         expected_dataset_streams = set(self.data_streams.keys())
#         dataset_streams_tensors_keys = set(dataset_streams_tensors.keys())
#         assert dataset_streams_tensors_keys == expected_dataset_streams, (
#             "Dataset streams produced by data ingestion pipeline don't match data_streams.\n"
#             f"produced={sorted(dataset_streams_tensors_keys)}\n"
#             f"expected={sorted(expected_dataset_streams)}"
#         )

#         # Apply per-dataset stream transforms
#         transform_time_by_dataset_stream: Dict[str, float] = {}

#         for name in dataset_streams_tensors.keys():
#             x = dataset_streams_tensors[name]
#             if not torch.is_tensor(x):
#                 raise TypeError(f"Produced dataset stream tensor {name!r} is not a tensor; got {type(x)}")

#             dataset_stream_meta = dataset_streams_meta[name]

#             # apply dataset stream-specific transforms (if any)
#             x, dataset_stream_meta, t_dataset_stream = self._apply_dataset_stream_transforms(name=name, x=x, meta=dataset_stream_meta)

#             dataset_streams_tensors[name] = x
#             dataset_streams_meta[name] = dataset_stream_meta
#             transform_time_by_dataset_stream[name] = float(base_transform_time + t_dataset_stream)

#         # Apply per-dataset stream dtype conversion + (optional) masking
#         dataset_stream_metainfo: Dict[str, Dict[str, Any]] = {}
#         per_stream_metrics: list[dict[str, Any]] = []

#         for name, x in dataset_streams_tensors.items():
#             if not torch.is_tensor(x):
#                 raise TypeError(f"Produced stream {name!r} is not a tensor; got {type(x)}")

#             spec = self.data_streams[name]

#             # Cast to per-dataset stream dtype if we have a spec
#             target_dtype = TORCH_DTYPES[spec.dtype].value if isinstance(spec.dtype, str) else spec.dtype
#             if x.dtype != target_dtype:
#                 x = x.to(target_dtype)

#             if spec.with_masking:
#                 mask_meta, masking_time = self._apply_dataset_stream_masking(name=name, x=x)
#             else:
#                 mask_meta = {}
#                 masking_time = -1.0

#             stream_transform_time = float(transform_time_by_dataset_stream[name])
#             stream_masking_time = float(masking_time)

#             dataset_stream_metainfo[name] = {
#                 "input_format": spec.input_format,
#                 "input_shape": spec.input_shape,
#                 "patch_shape": spec.patch_shape,
#                 **mask_meta,
#                 **dataset_streams_meta[name],
#             }

#             # Per-stream timing records use a namespaced metric name so they
#             # land in dedicated W&B panels (e.g. step_timing/stream/<name>/...).
#             per_stream_metrics.append(make_timing_metric(
#                 f"stream/{name}/transform_time", stream_transform_time,
#             ))
#             per_stream_metrics.append(make_timing_metric(
#                 f"stream/{name}/masking_time", stream_masking_time,
#             ))

#         out_meta = original_metainfo

#         out_meta["dataset_stream_metainfo"] = dataset_stream_metainfo
#         out_meta["idx"] = idx
#         out_meta["local_batch_size"] = self.local_batch_size
#         out_meta["global_batch_size"] = self.global_batch_size

#         aggregate_metrics: list[dict[str, Any]] = [
#             make_timing_metric("data_time", float(data_time)),
#             make_timing_metric("preprocess_time", float(time.time() - preprocess_t0)),
#         ]
#         existing_metrics = out_meta.get("metrics")
#         out_meta["metrics"] = (
#             (list(existing_metrics) if existing_metrics else [])
#             + aggregate_metrics
#             + per_stream_metrics
#         )

#         return {"data_tensors": dataset_streams_tensors, "metainfo": out_meta}


# --------------------------------------------------------------------------- #
# Base Finetune preprocessor
# --------------------------------------------------------------------------- #


class BaseFinetunePreprocessor(RayPreprocessor):
    # Reconstruction tasks (denoising / channel-split / upsample) patchify their
    # targets and therefore need a known signal-channel count. Detection and
    # segmentation legitimately have none, so the check is opt-in per subclass.
    REQUIRES_CHANNEL_COUNT: bool = False

    @property
    def TARGET_ROLES(self) -> "frozenset[str]":
        """Channel roles this task consumes as targets, DERIVED from the single
        ``_data_types()`` declaration (the channel-backed target entries). Empty
        for reconstruction tasks (no channel-backed targets).

        Cached after the first evaluation: every attribute ``_data_types()`` reads
        is set at ``__init__`` time and never mutated, so recomputing per batch is
        pure waste (no invalidation needed). The cache lives in ``self.__dict__``
        so ``__new__``-based test fixtures still work.
        """
        cached = self.__dict__.get("_target_roles_cache")
        if cached is not None:
            return cached
        try:
            roles = frozenset(
                e["channel_role"] for e in self._data_types().values() if "channel_role" in e
            )
            self.__dict__["_target_roles_cache"] = roles
            return roles
        except AttributeError as exc:
            # nn.Module.__getattr__ is the fallback for ANY AttributeError escaping
            # __getattribute__ -- including one raised from INSIDE this property body.
            # Left alone, a missing self.input_format (e.g. a fixture that bypasses
            # __init__ via __new__) surfaces as the misleading "object has no attribute
            # 'TARGET_ROLES'". Re-raise as a non-AttributeError so the real cause
            # survives, chained for the traceback.
            raise RuntimeError(
                f"{type(self).__name__}.TARGET_ROLES failed while evaluating "
                f"_data_types(); an attribute normally set in __init__ is missing: {exc}"
            ) from exc

    def __init__(
        self,
        *,
        transforms_list: list | None,
        with_masking: bool,
        mask_generator,
        patch_shape: tuple,
        dtype: torch.dtype | str,
        input_format: str,
        input_shape: tuple[int, ...],
        seed: int | None = None,
        min_channel_count: int | None = None,
        max_channel_count: int | None = None,
    ):
        super().__init__(
            dtype=dtype,
            transforms_list=transforms_list,
            with_masking=with_masking,
            mask_generator=mask_generator,
            patch_shape=patch_shape,
            input_format=input_format,
            input_shape=input_shape,
        )

        self.input_format = input_format
        assert input_format[-1] == "C", "Input format must end with 'C' (channels)"
        self.input_shape = input_shape

        # increment axis indices for batch dim
        self.axis_index = {ax: i + 1 for i, ax in enumerate(input_format)}
        self.channel_idx = self.axis_index.get("C", None)
        self.time_idx = self.axis_index.get("T", None)
        self.z_idx = self.axis_index.get("Z", None)
        self.y_idx = self.axis_index.get("Y", None)
        self.x_idx = self.axis_index.get("X", None)

        # spatial dims for downsample task
        self.spatial_dims = tuple(i for ax, i in self.axis_index.items() if ax in ("Z", "Y", "X"))

        axis_to_size = dict(zip(input_format, input_shape))
        self.axial_shape = axis_to_size.get("Z", None)
        self.timepoints = axis_to_size.get("T", None)
        if "Y" not in axis_to_size or "X" not in axis_to_size:
            raise ValueError("Input must include Y and X axes.")
        self.lateral_shape = (axis_to_size["Y"], axis_to_size["X"])

        # DB-sourced channel count.
        # self.channels is the signal-channel count fed to the model.
        self.min_channel_count = min_channel_count
        self.max_channel_count = max_channel_count
        if (
            min_channel_count is not None
            and max_channel_count is not None
            and min_channel_count != max_channel_count
        ):
            # TODO: Handle variable channel counts across runs.
            raise NotImplementedError(
                "dynamic/variable channel count not implemented yet; "
                f"min_channel_count ({min_channel_count}) != max_channel_count ({max_channel_count})"
            )
        self.channels = max_channel_count
        if self.REQUIRES_CHANNEL_COUNT and self.channels is None:
            # Reconstruction tasks patchify their targets, which needs a known
            # signal-channel count. Below, channels=None means "no patchification"
            # (pe_patchify = None) -- correct for detection/segmentation, silently
            # wrong here: forward() would hand the loss un-patchified targets and the
            # shape mismatch would surface far downstream. Fail at construction.
            raise ValueError(
                f"{type(self).__name__} requires a known signal-channel count, but "
                "max_channel_count was not provided (self.channels is None). This is "
                "normally sourced from the dataset DB; patchification would be "
                "silently disabled. Pass max_channel_count explicitly."
            )

        self.spatial_shape = (
            (self.axial_shape,) + self.lateral_shape if self.axial_shape is not None else self.lateral_shape
        )

        # dtype normalization
        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        # RNG
        self.rng = torch.Generator()
        if seed is None:
            self.rng.manual_seed(torch.initial_seed())
        else:
            self.rng.manual_seed(int(seed))

        self.patch_shape = patch_shape
        self.temporal_patch_size, self.axial_patch_size, self.lateral_patch_size = get_patch_sizes(
            input_format=input_format, patch_shape=patch_shape
        )
        self.num_patches, self.token_shape = calc_num_patches(
            input_fmt=self.input_format,
            input_shape=self.input_shape,
            patch_shape=patch_shape,
        )
        # Currently only tasks with a known signal-channel count (recon: denoising/
        # channel_split/upsample, which set max_channel_count) patchify their
        # targets.
        if self.channels is not None:
            self.pixels_per_patch = PatchEmbedding.compute_num_pixels_per_patch(
                channels=self.channels,
                temporal_patch_size=self.temporal_patch_size,
                axial_patch_size=self.axial_patch_size,
                lateral_patch_size=self.lateral_patch_size,
                input_format=self.input_format,
            )
            self.pe_patchify = functools.partial(
                PatchEmbedding.patchify,
                temporal_patch_size=self.temporal_patch_size,
                axial_patch_size=self.axial_patch_size,
                lateral_patch_size=self.lateral_patch_size,
                input_format=self.input_format,
                num_patches=self.num_patches,
                token_shape=self.token_shape,
                pixels_per_patch=self.pixels_per_patch,
            )
        else:
            self.pixels_per_patch = None
            self.pe_patchify = None

    def _common_pre(
        self,
        data_sample: dict,
        data_time: float,
    ) -> tuple[torch.Tensor, dict, float, float]:
        """Shared beginning of forward()."""
        preprocess_t0 = time.time()

        inputs = data_sample["data_tensor"]
        meta = data_sample["metainfo"]

        if inputs.dtype != self.dtype:
            inputs = inputs.to(self.dtype)

        data_time_value = data_time

        return inputs, meta, preprocess_t0, data_time_value

    def _split_channels(
        self,
        inputs: torch.Tensor,
        meta: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fast contiguous-prefix channel split.

        Enforces the layout contract: signal (input) channels are a contiguous
        prefix ``[0..n_in)`` and every object-role channel (consumed targets AND
        dropped object channels) occupies the tail. That lets us take the signal
        channels as a **basic-slice view** (``inputs[..., :n_in]``) -- no CUDA
        gather kernel, so uint16 needs no int32 cast -- and cast only the small
        object-channel tail to int32 for the labelmap snapshot.

        Returns ``(images, targets_by_role)``:
          - ``images``: signal prefix, cast to the model dtype.
          - ``targets_by_role``: ``{channel_role -> int32 (B, *spatial)}`` from the
            tail; empty for reconstruction tasks (no target roles). uint16 has no
            CUDA sort/compare kernel and a bf16 cast would alias ids > 256, so the
            snapshot is int32.
        """
        C = inputs.shape[-1]
        cm = _channel_mapping_from_meta(meta)
        partition = partition_channels(cm, C, self.TARGET_ROLES)

        n_in = len(partition.input_idxs)
        if partition.input_idxs != list(range(n_in)):
            raise ValueError(
                f"{type(self).__name__}: signal channels must be a contiguous prefix "
                f"[0..{n_in}); all object/target channels must occupy the tail. "
                f"Got input_idxs={partition.input_idxs} (C={C})."
            )

        images = inputs[..., :n_in]                # uint16 basic-slice view; no gather
        if images.dtype == torch.uint16:
            # fp32 when a transform reads raw counts, else straight to the model
            # dtype (see self._count_dtype). No /65535 scaling either way: the
            # sensor model works in actual counts.
            images = images.to(self._count_dtype)

        targets_by_role: dict[str, torch.Tensor] = {}
        if n_in < C:
            tail = inputs[..., n_in:].to(torch.int32)   # cheap: only the object channels
            for role, idxs in partition.targets_by_role.items():
                if len(idxs) > 1:
                    raise NotImplementedError(
                        f"multi-target-channel split not wired yet; role {role!r} "
                        f"has {len(idxs)} channels"
                    )
                targets_by_role[role] = tail[..., idxs[0] - n_in]
        return images, targets_by_role

    def _apply_transforms(self, data: dict) -> tuple[dict, float]:
        """Apply the configured transforms to a ``{"data_tensor", "metainfo"}`` dict.

        Single source of truth for ``data_types``: the per-task declarative spec
        (``_data_types``) is injected into ``metainfo`` here so every task drives
        the transforms the same way. dict in -> dict out; a malformed sample
        raises rather than silently skipping.
        """
        data["metainfo"]["data_types"] = self._data_types()
        if self.transforms:
            t0 = time.time()
            for transform in self.transforms:
                data = transform(data)
            transform_time = time.time() - t0
        else:
            transform_time = -1.0
        # Every task gets the post-transform spatial guard for free: patch/token
        # grids and masking are sized from input_shape, so the (possibly
        # transformed) tensor MUST land on it -- previously only the four
        # reconstruction forwards checked this.
        self._assert_input_shape_spatial(data["data_tensor"])
        return data, transform_time

    def _finalize(
        self,
        *,
        inputs: torch.Tensor,
        meta: dict,
        targets: Any,
        data_time: float,
        preprocess_t0: float,
        transform_time: float,
        idx: int,
    ) -> dict:
        """Attach masking info and timing, returning the standard dict.

        ``targets`` must already be contract-shaped (see data/data_types.py):
        Form D (``dict[role -> (B, ...) tensor]``, dense/reconstruction) or
        Form S (``List[Dict]`` of length B, structured/semantic-packaged) — or
        None when the task has no target. It is published AS-IS: no wrapping,
        and consumers never index ``targets[0]`` to unwrap.
        """

        # END-of-forward dtype boundary: the pipeline runs on an exact float32
        # intermediate (see _split_channels); the model input narrows to
        # self.dtype here. Targets are intentionally NOT narrowed -- keeping
        # them float32 preserves exact counts for the loss (bf16 - f32 promotes
        # to f32 in the loss path).
        if inputs.dtype != self.dtype:
            inputs = inputs.to(self.dtype)

        masking_time = -1.0
        mask_lists: dict[str, Any] = {}
        if self.with_masking:
            mt0 = time.time()
            B = inputs.shape[0]
            mask_data = self.mask_generator(B)
            masking_time = time.time() - mt0
            mask_lists = {k: [v] for k, v in mask_data.items() if v is not None}

        metrics: list[dict[str, Any]] = [
            make_timing_metric("data_time", data_time),
            make_timing_metric("preprocess_time", time.time() - preprocess_t0),
            make_timing_metric("transform_time", transform_time),
            make_timing_metric("masking_time", masking_time),
        ]
        existing_metrics = meta.get("metrics") if isinstance(meta, Mapping) else None
        if existing_metrics:
            metrics = list(existing_metrics) + metrics

        return {
            "data_tensor": inputs,
            "metainfo": {
                **meta,
                **mask_lists,
                "targets": targets,
                "idx": idx,
                "metrics": metrics,
            },
        }


# --------------------------------------------------------------------------- #
# Denoising task
# --------------------------------------------------------------------------- #


@registers_as("preprocessor", "denoising")
class DenoisingPreprocessor(BaseFinetunePreprocessor):
    REQUIRES_CHANNEL_COUNT = True

    """
    Task: denoising
    - inputs: noisy image (in counts, uint16 range)
    - targets: clean image (in counts, uint16 range)

    args:
    - denoising_type: str representing the type of denoising task to perform
    - transforms_list: list of transforms to apply to the input data
    
    Current denoising tasks:
    - microscopy: adds realistic sensor noise to the input data
    
    NOTE: Noise is added as a transform to the input data.
    If there is added noise (e.g. sensor noise), it should be added to the transforms_list.

    """
    
    def __init__(
        self,
        denoising_type: str,
        *,
        transforms_list: list | None,
        with_masking: bool,
        mask_generator,
        patch_shape: tuple,
        dtype: torch.dtype | str,
        input_format: str,
        input_shape: tuple[int, ...],
        seed: int | None = None,
        min_channel_count: int | None = None,
        max_channel_count: int | None = None,
        recon_role: str = "denoising",
    ):
        if denoising_type not in ("microscopy",):
            raise ValueError(f"Unknown denoising type: {denoising_type}")
        if (transforms_list is None or len(transforms_list) == 0) and denoising_type == "microscopy":
            raise ValueError("transforms_list must be provided with at least one transform for microscopy denoising")

        super().__init__(
            transforms_list=transforms_list,
            with_masking=with_masking,
            mask_generator=mask_generator,
            patch_shape=patch_shape,
            dtype=dtype,
            input_format=input_format,
            input_shape=input_shape,
            seed=seed,
            min_channel_count=min_channel_count,
            max_channel_count=max_channel_count,
        )

        self.denoising_type = denoising_type
        # Form-D role the clean clone is published under by
        # DeepCopyInputsAsTargets(role=...); both default to "denoising".
        self.recon_role = recon_role

    def forward(self, data_sample: dict, data_time: float, idx: int) -> dict:
        preprocess_t0 = time.time()
        raw_inputs = data_sample["data_tensor"]
        meta = data_sample["metainfo"]
        data_time_value = data_time

        # Strip any stray object channels (TARGET_ROLES=empty -> no target channel;
        # object channels go to dropped_idxs and are excluded from `images`).
        # Snapshot BEFORE dtype cast so int32 ids are preserved exactly.
        images, _targets = self._split_channels(raw_inputs, meta)

        # NOTE: no dtype narrowing here -- the float32 count intermediate from
        # _split_channels must reach the PSF/noise transforms intact; _finalize
        # narrows the model input to self.dtype at the end.

        # Parity guard: after stripping object channels, signal channel count
        # must match the DB-configured fixed count (when set).
        # TODO: eventuallly relax this to channels <= max_channel_count.
        if self.channels is not None and images.shape[-1] != self.channels:
            raise ValueError(
                f"DenoisingPreprocessor channel parity failure: "
                f"got {images.shape[-1]} signal channels but self.channels={self.channels} "
                f"(from max_channel_count). Check the DB channel config."
            )

        sample = {
            "data_tensor": images,
            "metainfo": meta,
        }

        sample, transform_time = self._apply_transforms(sample)

        # TODO: Consider refactoring this to support non-transformer-based decoders
        # Patchify targets for transformer-based decoders. The clean clone is a
        # Form-D role published by DeepCopyInputsAsTargets (see data/data_types.py);
        # the patchified target is re-published under the same role.
        targets = self.pe_patchify(
            get_role(sample["metainfo"]["targets"], self.recon_role),
            channels=self.channels,
        )

        # FIXME: Streamline this so that we either consistently pass data_sample
        # or its components (e.g. data_tensor, metainfo, targets, etc.)
        return self._finalize(
            inputs=sample["data_tensor"],
            meta=sample["metainfo"],
            targets={self.recon_role: targets},
            data_time=data_time_value,
            preprocess_t0=preprocess_t0,
            transform_time=transform_time,
            idx=idx,
        )


# --------------------------------------------------------------------------- #
# Channel-splitting task
# --------------------------------------------------------------------------- #


@registers_as("preprocessor", "channel_split")
class ChannelSplitPreprocessor(BaseFinetunePreprocessor):
    REQUIRES_CHANNEL_COUNT = True

    """
    Task: "channel_split"
    - inputs: original multi-channel image
    - targets: patchified original (per-channel)
    - model input: channel-averaged single-channel image
    """

    def __init__(
        self,
        *,
        patch_shape: tuple,
        transforms_list: list | None,
        with_masking: bool,
        mask_generator,
        dtype: torch.dtype | str,
        input_format: str,
        input_shape: tuple[int, ...],
        seed: int | None = None,
        min_channel_count: int | None = None,
        max_channel_count: int | None = None,
    ):
        super().__init__(
            transforms_list=transforms_list,
            with_masking=with_masking,
            mask_generator=mask_generator,
            patch_shape=patch_shape,
            dtype=dtype,
            input_format=input_format,
            input_shape=input_shape,
            seed=seed,
            min_channel_count=min_channel_count,
            max_channel_count=max_channel_count,
        )

    def forward(self, data_sample: dict, data_time: float, idx: int) -> dict:
        preprocess_t0 = time.time()
        raw_inputs = data_sample["data_tensor"]
        meta = data_sample["metainfo"]
        data_time_value = data_time

        if self.channel_idx is None:
            raise ValueError("Channel axis 'C' not present in input_format; cannot channel_split.")

        # Strip any stray object channels (TARGET_ROLES=empty -> no target channel;
        # object channels go to dropped_idxs and are excluded from `images`).
        # Snapshot BEFORE dtype cast so int32 ids are preserved exactly.
        images, _targets = self._split_channels(raw_inputs, meta)

        # NOTE: no dtype narrowing here -- the float32 count intermediate from
        # _split_channels must reach the transforms intact; _finalize narrows
        # the model input to self.dtype at the end.

        # Parity guard: after stripping object channels, signal channel count
        # must match the DB-configured fixed count (when set).
        # TODO: eventuallly relax this to channels <= max_channel_count.
        if self.channels is not None and images.shape[-1] != self.channels:
            raise ValueError(
                f"ChannelSplitPreprocessor channel parity failure: "
                f"got {images.shape[-1]} signal channels but self.channels={self.channels} "
                f"(from max_channel_count). Check the DB channel config."
            )

        # Route a dict so transforms (e.g. Resize) can crop_to_valid + read the
        # data_fields (injected in _apply_transforms); the target is derived from
        # the SAME transformed tensor.
        sample = {"data_tensor": images, "metainfo": meta}
        sample, transform_time = self._apply_transforms(data=sample)
        inputs_wo_mask = sample["data_tensor"]
        meta = sample["metainfo"]

        # targets are per-channel patches from original (transformed) input
        # input_shape must describe data after mask strip; self.channels matches inputs_wo_mask
        targets = self.pe_patchify(inputs_wo_mask, channels=self.channels)

        # model input: average over channels -> [B, ..., 1]
        inputs = inputs_wo_mask.mean(dim=self.channel_idx, keepdim=True)

        return self._finalize(
            inputs=inputs,
            meta=meta,
            targets={"channel_split": targets},   # Form D: role = task name
            data_time=data_time_value,
            preprocess_t0=preprocess_t0,
            transform_time=transform_time,
            idx=idx,
        )


# --------------------------------------------------------------------------- #
# Upsampling tasks (space / spacetime / time)
# --------------------------------------------------------------------------- #


@registers_as("preprocessor", "upsample")
class UpsamplePreprocessor(BaseFinetunePreprocessor):
    REQUIRES_CHANNEL_COUNT = True

    """
    Task: upsample
      mode in {"upsample_space", "upsample_spacetime", "upsample_time"}

    - "upsample_space" / "upsample_spacetime":
        targets = patchified HR inputs
        inputs = downsampled via NA mask
    """

    def __init__(
        self,
        *,
        transforms_list: list | None,
        with_masking: bool,
        mask_generator,
        patch_shape: tuple[int, int, int],
        dtype: torch.dtype | str,
        input_format: str,
        input_shape: tuple[int, ...],
        seed: int | None = None,
        ideal_psf_path: str | None = None,
        na_mask_thresholds: list[float] | None = None,
        resize_na_masks: bool = True,
        mode: str = "upsample_space",
        min_channel_count: int | None = None,
        max_channel_count: int | None = None,
    ):
        super().__init__(
            transforms_list=transforms_list,
            with_masking=with_masking,
            mask_generator=mask_generator,
            patch_shape=patch_shape,
            dtype=dtype,
            input_format=input_format,
            input_shape=input_shape,
            seed=seed,
            min_channel_count=min_channel_count,
            max_channel_count=max_channel_count,
        )

        if mode not in ("upsample_space", "upsample_spacetime", "upsample_time"):
            raise ValueError(f"Unknown upsample mode: {mode}")
        self.mode = mode

        self.resize_na_masks = resize_na_masks

        # Only required for space/spacetime upsampling
        if self.mode in ("upsample_space", "upsample_spacetime"):
            if ideal_psf_path is None:
                raise ValueError("ideal_psf_path must be provided for upsample_space/spacetime")
            if na_mask_thresholds is None:
                raise ValueError("na_mask_thresholds must be provided for upsample_space/spacetime")

            self.ideal_psf = torch.from_numpy(read_file(ideal_psf_path))
            self.na_masks = create_na_masks(
                self.ideal_psf,
                thresholds=na_mask_thresholds,
                target_shape=self.spatial_shape,
                resize=self.resize_na_masks,
            )
        else:
            self.ideal_psf = None
            self.na_masks = None

    def forward(self, data_sample: dict, data_time: float, idx: int) -> dict:
        preprocess_t0 = time.time()
        raw_inputs = data_sample["data_tensor"]
        meta = data_sample["metainfo"]
        data_time_value = data_time

        # Strip any stray object channels (TARGET_ROLES=empty -> no target channel;
        # object channels go to dropped_idxs and are excluded from `images`).
        # Snapshot BEFORE dtype cast so int32 ids are preserved exactly.
        images, _targets = self._split_channels(raw_inputs, meta)

        # NOTE: no dtype narrowing here -- the float32 count intermediate from
        # _split_channels must reach the transforms intact; _finalize narrows
        # the model input to self.dtype at the end.

        # Parity guard: after stripping object channels, signal channel count
        # must match the DB-configured fixed count (when set).
        # TODO: Eventually relax this to channels <= max_channel_count.
        if self.channels is not None and images.shape[-1] != self.channels:
            raise ValueError(
                f"UpsamplePreprocessor channel parity failure: "
                f"got {images.shape[-1]} signal channels but self.channels={self.channels} "
                f"(from max_channel_count). Check the DB channel config."
            )

        # Route a dict so transforms (e.g. Resize) can crop_to_valid + read the
        # data_fields (injected in _apply_transforms); both the HR target and the
        # downsampled input are derived from the SAME transformed tensor (no
        # original-vs-transformed mismatch).
        sample = {"data_tensor": images, "metainfo": meta}
        sample, transform_time = self._apply_transforms(data=sample)
        inputs = sample["data_tensor"]
        meta = sample["metainfo"]

        if self.mode in ("upsample_space", "upsample_spacetime"):
            # targets are HR patches
            targets = self.pe_patchify(inputs, channels=self.channels)

            # pick one NA mask and downsample
            na_mask_i = torch.randint(
                low=0,
                high=self.na_masks.shape[0],
                size=(1,),
                generator=self.rng,
            ).item()
            na_mask = resize_mask(
                self.na_masks[na_mask_i],
                input_format=self.input_format,
                channels=self.channels,
                timepoints=self.timepoints,
                axial_shape=self.axial_shape,
                lateral_shape=self.lateral_shape,
                dtype=inputs.real.dtype if inputs.is_complex() else inputs.dtype,
                device=inputs.device,
            )
            inputs = downsample(
                na_mask=na_mask,
                inputs=inputs,
                spatial_dims=self.spatial_dims,
            )
        elif self.mode == "upsample_time":
            targets = None
        else:
            raise RuntimeError(f"Unexpected mode: {self.mode}")

        return self._finalize(
            inputs=inputs,
            meta=meta,
            # Form D: role = task name (the mode); upsample_time has no target at all.
            targets={self.mode: targets} if targets is not None else None,
            data_time=data_time_value,
            preprocess_t0=preprocess_t0,
            transform_time=transform_time,
            idx=idx,
        )


# --------------------------------------------------------------------------- #
# Instance Segmentation task
# --------------------------------------------------------------------------- #


@registers_as("preprocessor", "instance_segmentation")
class InstanceSegmentationPreprocessor(BaseFinetunePreprocessor):
    """
    Task: instance segmentation

    Assumes upstream FinetuneCollatorActor has already:
      - kept the dense instance-id labelmap in the last data channel,
      - built per-instance targets from `annotations_metadata`,
      - used `local_segmentation_id` to align targets with the dense last-channel labelmap,
      - populated metainfo["targets"] with per-element dicts:
          {
            "masks": (N_inst, Z, Y, X),
            "boxes": (N_inst, 6),
            "mask_ids": (N_inst,),
            "labels": (N_inst,)
          },
      - computed image_sizes / orig_image_sizes / padding_mask,
      - (optionally) applied Resize() to image + masks + boxes + padding_mask.

    Here we only:
      - run any remaining transforms (if configured) on the
        {"data_tensor", "metainfo"} dict, and
      - package everything into the final standard output format.
    """

    def __init__(
        self,
        *,
        transforms_list: list | None,
        with_masking: bool,
        mask_generator,
        patch_shape: tuple[int, int, int],
        dtype: torch.dtype | str,
        input_format: str,
        input_shape: tuple[int, ...],
        seed: int | None = None,
        bbox_data_format: Optional[str] = None,
        bbox_output_format: Optional[str] = None,
        debug_savepath: str = None,
        require_targets: bool = True,
        materialize_binary_masks: bool = False,
        min_channel_count: int | None = None,
        max_channel_count: int | None = None,
    ):
        super().__init__(
            transforms_list=transforms_list,
            with_masking=with_masking,
            mask_generator=mask_generator,
            patch_shape=patch_shape,
            dtype=dtype,
            input_format=input_format,
            input_shape=input_shape,
            seed=seed,
            min_channel_count=min_channel_count,
            max_channel_count=max_channel_count,
        )

        if bbox_data_format is None or bbox_output_format is None:
            raise ValueError("bbox_data_format and bbox_output_format must be specified for instance_segmentation.")
        self.bbox_data_format = bbox_data_format
        self.bbox_output_format = bbox_output_format

        self.debug_savepath = debug_savepath
        self.require_targets = require_targets
        # When True, materialize per-instance binary masks on-device from the
        # integer labelmap (last channel of data_tensor) and attach them as
        # targets[b]["masks"]. Dense-mask heads (Mask2Former / PlainDETR /
        # multilabel) opt in here; labelmap-native heads (MaskDINO) leave this
        # off and read "label_map" directly.
        self.materialize_binary_masks = materialize_binary_masks

    def _data_types(self) -> Dict[str, Dict[str, Any]]:
        """Instance seg: image + per-target integer ``label_map`` (nearest) and
        ``boxes`` (coordinate-scaled). Binary ``masks`` are materialized AFTER
        transforms from the resized ``label_map``, so they are not listed here.
        """
        return {
            "data_tensor": self.base_dense_data_type,
            "label_map": {"kind": "instance_masks", "layout": self.input_format,
                          "role": "target", "channel_role": "instance_masks"},
            "boxes": {"kind": "boxes", "layout": self.bbox_output_format, "role": "target"},
        }

    def forward(self, data_sample: dict, data_time: float, idx: int) -> dict:
        """
        Now expects `data_sample` coming from FinetuneCollatorActor, i.e.:

          data_sample = {
            "data_tensor": (B, Z, Y, X, C_full)   # dense labelmap remains in the last channel
            "metainfo": {
                ...,
                "image_sizes": (B, 3),
                "orig_image_sizes": (B, 3),
                "padding_mask": (B, Z, Y, X),
                "targets": List[Dict[str, Tensor]],  # boxes/mask_ids/labels and masks or label_map
            }
          }

        We only:
          - split the int32 labelmap off the channel (pre-cast) and attach it
            per-target so transforms warp it coherently with the image,
          - ensure dtype,
          - run any remaining transforms on the full dict (if configured),
          - unpack targets, materialize masks if requested, and finalize.
        """
        preprocess_t0 = time.time()
        raw_inputs = data_sample["data_tensor"]
        meta = data_sample["metainfo"]
        data_time_value = data_time

        # Fast split: signal channels off the front (view + dtype cast), object
        # channels off the int32 tail. `label_map` rides the targets so Resize
        # warps it in lockstep with the image and boxes.
        images, targets_by_role = self._split_channels(raw_inputs, meta)
        labelmap = targets_by_role.get(DataKind.INSTANCE_MASKS.value)

        # Attach the per-target labelmap BEFORE transforms so Crop / Resize
        # warp `label_map` in lockstep with the image (and boxes). Labelmap-
        # native heads (MaskDINO) read this directly; dense heads turn it into
        # `masks` below.
        pre_targets = meta.get("targets", None)
        if labelmap is not None and pre_targets is not None:
            for b, t in enumerate(pre_targets):
                if b < labelmap.shape[0]:
                    t["label_map"] = labelmap[b]

        sample = {
            "data_tensor": images,
            "metainfo": meta,
        }
        sample, transform_time = self._apply_transforms(sample)

        if self.debug_savepath is not None:
            self._debug_visualize_batch(sample)

        inputs = sample["data_tensor"]
        meta = sample["metainfo"]
        targets = meta.pop("targets", None)
        if targets is None:
            # Keep downstream contract: Form S, list[dict] length B (see data/data_types.py).
            B = inputs.shape[0]
            targets = [
                {
                    "boxes": torch.zeros((0, 6), device=inputs.device, dtype=torch.float32),
                    "mask_ids": torch.zeros((0,), device=inputs.device, dtype=torch.long),
                    "labels": torch.zeros((0,), device=inputs.device, dtype=torch.long),
                }
                for _ in range(B)
            ]

        # Dense-mask heads (Mask2Former / PlainDETR / multilabel) consume
        # targets[b]["masks"]; build them here on-device from the per-target
        # integer "label_map" so the CPU collator never materializes them.
        if self.materialize_binary_masks:
            self._materialize_target_masks(targets, inputs.device)

        return self._finalize(
            inputs=inputs,
            meta=meta,
            targets=targets,
            data_time=data_time_value,
            preprocess_t0=preprocess_t0,
            transform_time=transform_time,
            idx=idx,
        )

    def _materialize_target_masks(self, targets: list[dict], device: torch.device) -> None:
        """Attach `t["masks"]` per target, built on `device` from the integer
        labelmap. Skips silently if any target lacks a `label_map` (e.g. ad-hoc
        inference views). The labelmap is cast to int32 (uint16 has no CUDA
        sort/compare kernel for downstream ops).
        """
        if not all("label_map" in t for t in targets):
            return
        labelmap = torch.stack(
            [t["label_map"] for t in targets]
        ).to(device=device, dtype=torch.int32)  # (B, *spatial)
        spatial = tuple(labelmap.shape[1:])
        binary_masks_batch = mask_ids_to_masks(
            batch_size=len(targets),
            spatiotemporal_shape=spatial,
            mask_ids_batch=[t["mask_ids"] for t in targets],
            masks=labelmap,
            device=device,
        )
        for t, bm in zip(targets, binary_masks_batch):
            t["masks"] = bm

    def _debug_visualize_batch(self, sample: dict) -> None:
        """
        Debug helper:
        - plots middle Z slice of the first sample's image
        - plots corresponding mask slice
        - overlays all bboxes on the image slice
        - prints full metainfo
        - raises an error to stop training
        """
        import matplotlib.pyplot as plt

        inputs = sample["data_tensor"]
        meta = sample["metainfo"]
        targets = meta["targets"]

        vol = inputs[0]
        if self.input_format == "ZYXC":
            # vol: (Z, Y, X, C)
            Z, Y, X, C = vol.shape
            z_mid = Z // 2
            img_slice = vol[z_mid, :, :, 0].float().detach().cpu().numpy()
        else:
            raise RuntimeError(f"Debug visualize only supports ZYXC/TZYXC, got {self.input_format}")

        lo = float(np.percentile(img_slice, 1))
        hi = float(np.percentile(img_slice, 99))

        tgt0 = targets[0]
        masks = tgt0["masks"]
        boxes = tgt0["boxes"]

        masks = masks.float().detach().cpu()
        boxes = boxes.float().detach().cpu()

        if masks.ndim == 4:
            N_inst, Zm, Ym, Xm = masks.shape
            z_mid_mask = min(z_mid, Zm - 1)
            label_slice = torch.zeros((Ym, Xm), dtype=torch.int64)
            for idx in range(N_inst):
                label_slice[masks[idx, z_mid_mask] > 0.5] = idx + 1
        else:
            label_slice = None

        # label_slice = None  # skipping mask slice for now

        boxes_zyx = convert_bbox_format(boxes, self.bbox_output_format, "zyxzyx")

        print("=== DEBUG metainfo ===")
        print(meta)
        print("[DEBUG] inputs min/max:", float(inputs.min()), float(inputs.max()))
        # print("[DEBUG] masks sum:", float(masks.sum()))

        # Plot image + boxes and mask slice
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # 1) image with bboxes
        ax_img = axs[0]
        ax_img.imshow(img_slice, cmap="gray", vmin=lo, vmax=hi)
        for b in boxes_zyx:
            z1, y1, x1, z2, y2, x2 = b.tolist()
            z1 = int(round(z1))
            z2 = int(round(z2))
            if z1 <= z_mid <= z2:
                rect = plt.Rectangle(
                    (x1, y1),
                    (x2 - x1),
                    (y2 - y1),
                    fill=False,
                    edgecolor="r",
                    linewidth=1,
                )
                ax_img.add_patch(rect)
        ax_img.set_title("Image + bboxes")
        ax_img.set_axis_off()

        # 2) mask slice (labelmap)
        ax_mask = axs[1]
        if label_slice is not None:
            ax_mask.imshow(label_slice.numpy(), interpolation="nearest")
            ax_mask.set_title("Instance mask slice")
        else:
            ax_mask.imshow(img_slice, cmap="gray")
            ax_mask.set_title("Mask slice (none)")
        ax_mask.set_axis_off()

        plt.tight_layout()
        plt.savefig(self.debug_savepath)

        raise RuntimeError("Debug visualization — stopping after first batch.")

# --------------------------------------------------------------------------- #
# Semantic Segmentation task
# --------------------------------------------------------------------------- #


def build_semantic_targets(
    targets: dict,
    semantic_classes,
    *,
    semantic_map: Optional[torch.Tensor],
    semantic_legend: list[dict],
    object_type_names: dict[int, str],
    batch_size: int,
    spatial: tuple[int, ...],
    device,
) -> tuple[list[dict], list[str]]:
    """Package semantic GT into Form-S set-loss targets.

    All semantic classes are squashed onto ONE integer channel whose voxel
    values are ``local_segmentation_id``, and the annotation bucket's ``semantic``
    list is the legend (``local_segmentation_id -> object_type_id``). A class is
    resolved from whichever source owns it:

      * a legend class -> ``semantic_map == local_segmentation_id``
      * a derived role -> ``targets[name] > 0``, published by a transform
        (``InstanceToBoundaryMask``, ``ForegroundMasks``)

    Both still exist, so ``semantic_classes: [boundary, foreground]`` keeps
    working unchanged.

    Background needs no special case -- it is simply absent from the legend, so
    it cannot become a class.

    Returns ``(per_sample_targets, resolved_classes)`` where the first is Form S:
    ``[{"masks": (N, *spatial) bool, "labels": (N,)}, ...]`` of length B.
    """
    # name -> the integer to compare the labelmap against
    name_to_value: dict[str, int] = {}
    for leaf in semantic_legend or ():
        type_id = leaf.get("object_type_id")
        if type_id is None:
            continue
        name = object_type_names.get(int(type_id))
        if name is None:
            raise KeyError(
                f"object_type_id={type_id} has no name; the api.object_types "
                f"cache is stale -- refetch it at startup"
            )
        name_to_value[name] = int(leaf["local_segmentation_id"])

    derived = set(targets.keys())

    if semantic_classes == "all":
        # Legend order, which is the DB's own ordering of the bucket.
        resolved = list(name_to_value)
    else:
        resolved = [str(c) for c in semantic_classes]

    unknown = [c for c in resolved if c not in name_to_value and c not in derived]
    if unknown:
        raise KeyError(
            f"semantic_classes {unknown} are neither legend classes "
            f"{sorted(name_to_value)} nor derived roles {sorted(derived)}. "
            f"List only classes something actually produced, or use 'all'."
        )
    if any(c in name_to_value for c in resolved) and semantic_map is None:
        raise ValueError(
            f"semantic_classes {resolved} reference legend classes but no semantic "
            f"channel was present (channel_type='mask', annotation_type='semantic')"
        )

    per_class: list[torch.Tensor] = []
    for c in resolved:
        if c in derived:
            # transform-published binary map
            per_class.append(targets[c] > 0)
        else:
            # legend class: one-hot straight out of the integer labelmap
            per_class.append(semantic_map == name_to_value[c])

    if per_class:
        masks = torch.stack(per_class, dim=1)          # (B, N, *spatial) bool
    else:
        masks = torch.zeros((batch_size, 0, *spatial), dtype=torch.bool, device=device)
    labels = torch.arange(masks.shape[1], dtype=torch.int64, device=masks.device)

    return [{"masks": masks[b], "labels": labels} for b in range(batch_size)], resolved


@registers_as("preprocessor", "semantic_segmentation")
class SemanticSegmentationPreprocessor(BaseFinetunePreprocessor):
    """
    Task: semantic segmentation

    Assumes upstream FinetuneCollatorActor has already:
      - kept the dense instance-id labelmap in the last data channel,
      - built per-instance targets from `annotations_metadata`,
      - used `local_segmentation_id` to align targets with the dense last-channel labelmap,
      - populated metainfo["targets"] with per-element dicts:
          {
            "masks": (N_inst, Z, Y, X),
            "boxes": (N_inst, 6),
            "mask_ids": (N_inst,),
            "labels": (N_inst,)
          },
      - computed image_sizes / orig_image_sizes / padding_mask,
      - (optionally) applied Resize() to image + masks + boxes + padding_mask.

    Here we only:
      - run any remaining transforms (if configured) on the
        {"data_tensor", "metainfo"} dict, and
      - package everything into the final standard output format.
    """

    def __init__(
        self,
        *,
        transforms_list: list | None,
        with_masking: bool,
        mask_generator,
        patch_shape: tuple[int, int, int],
        dtype: torch.dtype | str,
        input_format: str,
        input_shape: tuple[int, ...],
        seed: int | None = None,
        bbox_data_format: Optional[str] = None,
        bbox_output_format: Optional[str] = None,
        debug_savepath: str = None,
        min_channel_count: int | None = None,
        max_channel_count: int | None = None,
        semantic_classes="all",
    ):
        super().__init__(
            transforms_list=transforms_list,
            with_masking=with_masking,
            mask_generator=mask_generator,
            patch_shape=patch_shape,
            dtype=dtype,
            input_format=input_format,
            input_shape=input_shape,
            seed=seed,
            min_channel_count=min_channel_count,
            max_channel_count=max_channel_count,
        )

        self.debug_savepath = debug_savepath
        # The literal "all" (every matched channel role) or an explicit ordered role
        # list. See build_semantic_targets.
        self.semantic_classes = semantic_classes

    def _data_types(self) -> Dict[str, Dict[str, Any]]:
        """Semantic seg: image + semantic map targets.

        Every ``semantic_masks*`` channel is grabbed (role family) and
        published as a Form-D target role -- a batched ``(B, Z, Y, X)`` integer
        labelmap per role (see data/data_types.py). Optional boundary/foreground
        transforms publish derived roles. NOTE: the ``semantic_maps`` field
        declared here feeds the (currently parked) in-preprocessor Resize target
        adjustment, which predates Form D -- revive only together with Form-D
        support in resize.py.
        """
        return {
            "data_tensor": self.base_dense_data_type,
            "semantic_maps": {"kind": "semantic_masks", "layout": self.input_format,
                              "role": "target", "channel_role": "semantic_masks"},
        }

    def forward(self, data_sample: dict, data_time: float, idx: int) -> dict:
        """
        Now expects `data_sample` coming from FinetuneCollatorActor, i.e.:

          data_sample = {
            "data_tensor": (B, Z, Y, X, C_full)   # dense labelmap remains in the last channel
            "metainfo": {
                ...,
                "image_sizes": (B, 3),
                "orig_image_sizes": (B, 3),
                "padding_mask": (B, Z, Y, X),
                "targets": List[Dict[str, Tensor]],  # boxes/mask_ids/labels and masks or label_map
            }
          }

        We only:
          - ensure dtype,
          - run any remaining transforms on the full dict (if configured),
          - unpack targets and finalize.
        """
        t0 = time.time()
        raw_inputs = data_sample["data_tensor"]
        meta = data_sample["metainfo"]
        data_time_value = data_time

        # Fast split: signal channels off the front (view + dtype cast); every
        # semantic_segmentation_* channel off the int32 tail, keyed by concrete role.
        images, targets_by_role = self._split_channels(raw_inputs, meta)

        if self.channels is not None and images.shape[-1] != self.channels:
            raise ValueError(
                f"SemanticSegmentationPreprocessor channel parity failure: "
                f"got {images.shape[-1]} signal channels but self.channels={self.channels} "
                f"(from max_channel_count). Check the DB channel config."
            )

        B = images.shape[0]

        # Publish the matched semantic channels as Form-D targets: one batched
        # (B, Z, Y, X) map per role (see data/data_types.py). Deterministic role
        # order; empty dict at inference with no GT channels. `channel_roles`
        # records the DB-matched roles BEFORE any transform publishes a derived
        # role (boundary/foreground) -- this is what semantic_classes="all"
        # resolves to at packaging.
        # Semantic GT is ONE integer channel now, keyed by class id, so pull it
        # out of the role dict: what remains is purely DERIVED roles (boundary,
        # foreground) published by transforms. It goes back in under its kind for
        # the transform pass so geometric ops warp it in lockstep with the image,
        # then comes out again for packaging.
        semantic_map = targets_by_role.pop(DataKind.SEMANTIC_MASKS.value, None)
        _instances, semantic_legend = parse_annotations_metadata(
            _first_annotations_metadata(meta), window_offset=0
        )

        channel_roles = sorted(targets_by_role.keys())
        meta["targets"] = {r: targets_by_role[r] for r in channel_roles}
        if semantic_map is not None:
            meta["targets"][DataKind.SEMANTIC_MASKS.value] = semantic_map

        sample = {"data_tensor": images, "metainfo": meta}
        sample, transform_time = self._apply_transforms(sample)
        meta = sample["metainfo"]

        # Package into the semantic target contract: Form-D maps in -> Form-S
        # set-loss GT out (the ONE form conversion in the pipeline). The taxonomy
        # comes from config (see build_semantic_targets): "all" = every matched
        # channel role, or an explicit ordered list.
        post_transform = dict(meta["targets"])
        semantic_map = post_transform.pop(DataKind.SEMANTIC_MASKS.value, None)

        targets, semantic_classes = build_semantic_targets(
            post_transform,
            self.semantic_classes,
            semantic_map=semantic_map,
            semantic_legend=semantic_legend,
            object_type_names=_object_type_names_from_meta(meta),
            batch_size=B,
            spatial=tuple(sample["data_tensor"].shape[1:-1]),
            device=images.device,
        )
        meta["targets"] = targets
        meta["semantic_classes"] = semantic_classes

        # Log the resolved class taxonomy once: class index == position in this
        # list, so a silently different resolution order would relabel every
        # voxel. (Config is explicit; this is the runtime record of it.)
        if not self.__dict__.get("_taxonomy_logged", False):
            self.__dict__["_taxonomy_logged"] = True
            logger.info(
                "[SemanticSegmentationPreprocessor] resolved semantic class list "
                "(index -> role): %s",
                {i: c for i, c in enumerate(semantic_classes)},
            )

        if self.debug_savepath is not None:
            self._debug_visualize_batch(sample)

        inputs = sample["data_tensor"]
        meta = sample["metainfo"]

        return self._finalize(
            inputs=inputs,
            meta=meta,
            targets=targets,
            data_time=data_time_value,
            preprocess_t0=t0,
            transform_time=transform_time,
            idx=idx,
        )

    def _debug_visualize_batch(self, sample: dict) -> None:
        """
        Debug helper:
        - plots middle Z slice of the first sample's image
        - plots corresponding semantic mask slice
        - prints full metainfo
        - raises an error to stop training
        """
        import matplotlib.pyplot as plt

        inputs = sample["data_tensor"]
        meta = sample["metainfo"]
        targets = meta["targets"]

        vol = inputs[0]
        if self.input_format == "ZYXC":
            # vol: (Z, Y, X, C)
            Z, Y, X, C = vol.shape
            z_mid = Z // 2
            img_slice = vol[z_mid, :, :, 0].float().detach().cpu().numpy()
        else:
            raise RuntimeError(f"Debug visualize only supports ZYXC/TZYXC, got {self.input_format}")

        lo = float(np.percentile(img_slice, 1))
        hi = float(np.percentile(img_slice, 99))

        tgt0 = targets[0]
        masks = tgt0["masks"].float().detach().cpu()

        if masks.ndim == 4:
            N_inst, Zm, Ym, Xm = masks.shape
            z_mid_mask = min(z_mid, Zm - 1)
            label_slice = torch.zeros((Ym, Xm), dtype=torch.int64)
            for idx in range(N_inst):
                label_slice[masks[idx, z_mid_mask] > 0.5] = idx + 1
        else:
            label_slice = None

        # Masks labelmap (raw instance IDs from input); (B, Z, Y, X)
        if "masks_labelmap" in sample:
            ml = sample["masks_labelmap"]
            Zm_ml = ml.shape[1]
            z_mid_ml = min(z_mid, Zm_ml - 1)
            masks_labelmap_slice = ml[0, z_mid_ml].float().detach().cpu().numpy()
        else:
            masks_labelmap_slice = None

        print("=== DEBUG metainfo ===")
        print(meta)
        print("[DEBUG] inputs min/max:", float(inputs.min()), float(inputs.max()))

        # Plot image, masks labelmap, and semantic mask slice
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        ax_img = axs[0]
        ax_img.imshow(img_slice, cmap="gray", vmin=lo, vmax=hi)
        ax_img.set_title("Image")
        ax_img.set_axis_off()

        ax_labelmap = axs[1]
        if masks_labelmap_slice is not None:
            ax_labelmap.imshow(masks_labelmap_slice, interpolation="nearest")
            ax_labelmap.set_title("Masks labelmap")
        else:
            ax_labelmap.imshow(np.zeros_like(label_slice, dtype=np.int64), cmap="gray")
            ax_labelmap.set_title("Masks labelmap (none)")
        ax_labelmap.set_axis_off()

        ax_mask = axs[2]
        if label_slice is not None:
            ax_mask.imshow(label_slice.numpy(), interpolation="nearest")
            ax_mask.set_title("Semantic mask slice")
        else:
            ax_mask.imshow(img_slice, cmap="gray")
            ax_mask.set_title("Semantic mask slice (none)")
        ax_mask.set_axis_off()

        plt.tight_layout()
        plt.savefig(self.debug_savepath)

        raise RuntimeError("Debug visualization — stopping after first batch.")

# --------------------------------------------------------------------------- #
# Object Detection task
# --------------------------------------------------------------------------- #

@registers_as("preprocessor", "object_detection")
class ObjectDetectionPreprocessor(BaseFinetunePreprocessor):
    """
    Task: object detection

    Assumes upstream FinetuneCollatorActor has already:
      - built 3D bboxes,
      - populated metainfo["targets"] with per-element dicts:
          {
            "boxes": (N_obj, 6),
            "labels": (N_obj,)
          },
      - computed image_sizes / orig_image_sizes / padding_mask,
      - (optionally) applied Resize() to image + boxes + padding_mask.

    Here we only:
      - run any remaining transforms (if configured) on the
        {"data_tensor", "metainfo"} dict, and
      - package everything into the final standard output format.
    """

    def __init__(
        self,
        *,
        transforms_list: list | None,
        with_masking: bool,
        mask_generator,
        patch_shape: tuple[int, int, int],
        dtype: torch.dtype | str,
        input_format: str,
        input_shape: tuple[int, ...],
        seed: int | None = None,
        bbox_data_format: Optional[str] = None,
        bbox_output_format: Optional[str] = None,
        debug_savepath: str = None,
        min_channel_count: int | None = None,
        max_channel_count: int | None = None,
    ):
        super().__init__(
            transforms_list=transforms_list,
            with_masking=with_masking,
            mask_generator=mask_generator,
            patch_shape=patch_shape,
            dtype=dtype,
            input_format=input_format,
            input_shape=input_shape,
            seed=seed,
            min_channel_count=min_channel_count,
            max_channel_count=max_channel_count,
        )

        if bbox_data_format is None or bbox_output_format is None:
            raise ValueError("bbox_data_format and bbox_output_format must be specified for instance_segmentation.")
        self.bbox_data_format = bbox_data_format
        self.bbox_output_format = bbox_output_format

        self.debug_savepath = debug_savepath

    def _data_types(self) -> Dict[str, Dict[str, Any]]:
        """Object detection: image + ``boxes`` (coordinate-scaled). No dense GT."""
        return {
            "data_tensor": self.base_dense_data_type,
            "boxes": {"kind": "boxes", "layout": self.bbox_output_format, "role": "target"},
        }

    def forward(self, data_sample: dict, data_time: float, idx: int) -> dict:
        """
        Now expects `data_sample` coming from FinetuneCollatorActor, i.e.:

          data_sample = {
            "data_tensor": (B, Z, Y, X, C_no_mask)   # already resized if Resize was used
            "metainfo": {
                ...,
                "image_sizes": (B, 3),
                "orig_image_sizes": (B, 3),
                "padding_mask": (B, Z, Y, X),
                "targets": List[Dict[str, Tensor]],  # masks/boxes/mask_ids/labels
            }
          }

        We only:
          - ensure dtype,
          - run any remaining transforms on the full dict (if configured),
          - unpack targets and finalize.
        """
        t0 = time.time()
        raw_inputs = data_sample["data_tensor"]
        meta = data_sample["metainfo"]
        data_time_value = data_time

        # Fast split: object detection targets come from meta["targets"] (boxes),
        # so the object channels are just stripped off the model image (the tail
        # snapshot is discarded).
        inputs_wo_mask, _targets = self._split_channels(raw_inputs, meta)

        if self.channels is not None and inputs_wo_mask.shape[-1] != self.channels:
            raise ValueError(
                f"ObjectDetectionPreprocessor channel parity failure: "
                f"got {inputs_wo_mask.shape[-1]} signal channels but self.channels={self.channels} "
                f"(from max_channel_count). Check the DB channel config."
            )

        sample = {
            "data_tensor": inputs_wo_mask,
            "metainfo": meta,
        }
        sample, transform_time = self._apply_transforms(sample)

        if self.debug_savepath is not None:
            self._debug_visualize_batch(sample)

        inputs = sample["data_tensor"]
        meta = sample["metainfo"]
        targets = meta.pop("targets")

        return self._finalize(
            inputs=inputs,
            meta=meta,
            targets=targets,
            data_time=data_time_value,
            preprocess_t0=t0,
            transform_time=transform_time,
            idx=idx,
        )

    def _debug_visualize_batch(self, sample: dict) -> None:
        """
        Debug helper:
        - plots middle Z slice of the first sample's image
        - overlays all bboxes on the image slice
        - prints full metainfo
        - raises an error to stop training
        """
        import matplotlib.pyplot as plt

        inputs = sample["data_tensor"]
        meta = sample["metainfo"]
        targets = meta["targets"]

        vol = inputs[0]
        if self.input_format == "ZYXC":
            # vol: (Z, Y, X, C)
            Z, Y, X, C = vol.shape
            z_mid = Z // 2
            img_slice = vol[z_mid, :, :, 0].float().detach().cpu().numpy()
        else:
            raise RuntimeError(f"Debug visualize only supports ZYXC/TZYXC, got {self.input_format}")

        lo = float(np.percentile(img_slice, 1))
        hi = float(np.percentile(img_slice, 99))

        tgt0 = targets[0]
        boxes = tgt0["boxes"]

        boxes = boxes.float().detach().cpu()

        boxes_zyx = convert_bbox_format(boxes, self.bbox_output_format, "zyxzyx")

        print("=== DEBUG metainfo ===")
        print(meta)
        print("[DEBUG] inputs min/max:", float(inputs.min()), float(inputs.max()))

        # Plot image + boxes
        fig, ax_img = plt.subplots(1, 1, figsize=(10, 5))
        ax_img.imshow(img_slice, cmap="gray", vmin=lo, vmax=hi)
        for b in boxes_zyx:
            z1, y1, x1, z2, y2, x2 = b.tolist()
            z1 = int(round(z1))
            z2 = int(round(z2))
            if z1 <= z_mid <= z2:
                rect = plt.Rectangle(
                    (x1, y1),
                    (x2 - x1),
                    (y2 - y1),
                    fill=False,
                    edgecolor="r",
                    linewidth=1,
                )
                ax_img.add_patch(rect)
        ax_img.set_title("Image + bboxes")
        ax_img.set_axis_off()

        plt.tight_layout()
        plt.savefig(self.debug_savepath)

        raise RuntimeError("Debug visualization — stopping after first batch.")



# --------------------------------------------------------------------------- #
# Video Segmentation / Tracking task
# --------------------------------------------------------------------------- #


class _LazyFrameMasks:
    """Lazily materialized per-frame ``(B*K_full, Z, Y, X)`` bool mask blocks.

    Sequence facade over the SAM2 target view's ``masks`` field: ``masks[t]``
    builds frame ``t``'s block on demand from the labelmap and the sampled id
    subset instead of materializing all ``T`` frames up front (~B*K*Z*Y*X bool
    per frame, ~268 MB/frame at production shapes). Consumers that never read
    ``masks`` (the point-loss criterion path, empty inference views) pay
    nothing; the dense-loss / correction-sampler path materializes exactly the
    frames it touches. A materialized frame is cached: the SAM2 forward holds
    every frame it reads for the whole step (correction-point sampling), so
    the criterion's later read of the same frame must not rebuild it -- that
    was a second ``B*K*Z*Y*X`` bool per step for nothing. ``release()`` drops
    the cache when the step's data views are discarded.
    """

    def __init__(
        self,
        mask_labelmap: torch.Tensor,          # (B, T, Z, Y, X) int
        sampled_ids_per_b: "list[torch.Tensor]",
        max_masks: int,
        device: torch.device,
    ):
        self._lm = mask_labelmap
        self._ids = sampled_ids_per_b
        self._K_full = int(max_masks)
        self._device = device
        self._B = int(mask_labelmap.shape[0])
        self._T = int(mask_labelmap.shape[1])
        self._spatial = tuple(mask_labelmap.shape[2:])
        self._cache: dict[int, torch.Tensor] = {}

    def release(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return self._T

    def __iter__(self):
        for t in range(self._T):
            yield self[t]

    def __getitem__(self, t) -> torch.Tensor:
        t = int(t)
        if t < 0:
            t += self._T
        if not (0 <= t < self._T):
            raise IndexError(f"frame index {t} out of range [0, {self._T})")
        cached = self._cache.get(t)
        if cached is not None:
            return cached
        frame = torch.zeros(
            (self._B * self._K_full, *self._spatial), dtype=torch.bool, device=self._device
        )
        for b in range(self._B):
            sampled = self._ids[b]
            K = int(sampled.numel())
            if K == 0:
                continue
            lm_bt = self._lm[b, t]
            lo = b * self._K_full
            frame[lo:lo + K] = (
                lm_bt.unsqueeze(0) == sampled.to(lm_bt.dtype).view(K, 1, 1, 1)
            )
        self._cache[t] = frame
        return frame


@registers_as("preprocessor", "sam2_video")
class SAM2VideoPreprocessor(BaseFinetunePreprocessor):
    """
    Preprocessor for prompt-based video segmentation/tracking.
    Transforms collator output from BTZYXC format into expected views:
      - Flat image batch: (B*T, C_img, Z, Y, X) channels-first, mask channel stripped
      - Per-frame binary masks indexed by frame (stage_id)
      - Flat object-to-image index maps for tracking loop
    """

    def __init__(
        self,
        transforms_list: list | None,
        with_masking: bool,
        mask_generator,
        patch_shape: tuple[int, int, int],
        dtype: torch.dtype | str,
        input_format: str,
        input_shape: tuple[int, ...],
        seed: int | None = None,
        # FIXME: review and redo this preprocessor once infrence/eval lands!!!
        expect_mask_channel: bool = True,
        max_masks: int | None = None,
        require_targets: bool = True,
        bbox_format: str = "zyxzyx",
        boxes_normalized: bool = False,
        min_channel_count: int | None = None,
        max_channel_count: int | None = None,
    ):
        super().__init__(
            transforms_list=transforms_list,
            with_masking=with_masking,
            mask_generator=mask_generator,
            patch_shape=patch_shape,
            dtype=dtype,
            input_format=input_format,
            input_shape=input_shape,
            seed=seed,
            min_channel_count=min_channel_count,
            max_channel_count=max_channel_count,
        )
        if "T" not in self.input_format:
            raise ValueError(
                f"SAM2VideoPreprocessor requires temporal dimension 'T' in input_format, "
                f"got {self.input_format!r}"
            )
        self.expect_mask_channel = expect_mask_channel
        self.max_masks = max_masks
        # When a mask channel is present (training/eval-with-GT), per-instance
        # binary masks are built on-device from the integer labelmap (last
        # channel of data_tensor); the collator never materializes masks on
        # CPU and only K=max_masks slices get built. When there is no mask
        # channel (inference), the preprocessor skips mask generation entirely
        # and emits an empty target view. max_masks is only needed for the
        # mask-building path.
        if self.expect_mask_channel and self.max_masks is None:
            raise ValueError(
                "SAM2VideoPreprocessor requires max_masks when expect_mask_channel=True"
            )
        # Box format emitted by the collator; recorded in the target view so
        # downstream consumers (SAM2 box-prompt sampler) convert at the boundary
        # without inferring silently.
        self.bbox_format = bbox_format
        # True when the collator normalized boxes to [0, 1] against the buffer
        # spatial shape; the box-prompt sampler works in voxels, so the view
        # denormalizes once when it copies the sampled boxes.
        self.boxes_normalized = bool(boxes_normalized)

    def _data_types(self) -> Dict[str, Dict[str, Any]]:
        """SAM2 video: image + per-target integer ``label_map`` (nearest) and
        ``boxes`` (coordinate-scaled). Per-frame mask/index views are built AFTER
        transforms from the resized labelmap, so they are not listed here.
        """
        return {
            "data_tensor": self.base_dense_data_type,
            "label_map": {"kind": "instance_masks", "layout": self.input_format,
                          "role": "target", "channel_role": "instance_masks"},
            "boxes": {"kind": "boxes", "layout": self.bbox_format, "role": "target"},
        }

    # ------------------------------------------------------------------ #
    # Mask & index transformations
    # ------------------------------------------------------------------ #

    def _empty_data_views(
        self,
        num_frames: int,
        num_videos: int,
        spatial: tuple[int, ...],
        device: torch.device,
    ) -> dict:
        """Target view for the no-mask-channel (inference) path.

        Same key contract as `_build_data_views` but every per-frame tensor is
        empty (0 rows): no labelmap, no masks, no prompts are derived from GT.
        Downstream prompt sampling supplies its own clicks at inference time.
        """
        T = num_frames
        B = num_videos
        empty_mask = [torch.zeros((0, *spatial), dtype=torch.bool, device=device) for _ in range(T)]
        empty_i32 = [torch.zeros(0, dtype=torch.int32, device=device) for _ in range(T)]
        empty_i64 = [torch.zeros(0, dtype=torch.int64, device=device) for _ in range(T)]
        empty_bool = [torch.zeros(0, dtype=torch.bool, device=device) for _ in range(T)]
        empty_boxes = [torch.zeros((0, 6), dtype=torch.float32, device=device) for _ in range(T)]
        return {
            "num_frames": T,
            "num_videos": B,
            "masks": empty_mask,
            "img_ids": empty_i32,
            "labelmaps": torch.zeros((B * T, *spatial), dtype=torch.int32, device=device),
            "instance_ids": empty_i64,
            "valid": empty_bool,
            "presence_t": empty_bool,
            "boxes": empty_boxes,
            "box_format": self.bbox_format,
        }

    def _build_data_views(
        self,
        targets: list[dict],
        num_frames: int,
        num_videos: int,
        device: torch.device,
        mask_labelmap: torch.Tensor,  # (B, T, Z, Y, X) int32
    ) -> dict:
        """
        Build the SAM2 per-frame target view from the integer labelmap, using
        only the K=min(N_inst, max_masks) IDs we keep after the per-frame cap.
        The dense `masks` field (consumed by the dense-loss criterion path and
        the SAM2 prompt/correction sampler) is a LAZY per-frame sequence
        (`_LazyFrameMasks`): each (B*K_full, Z, Y, X) bool block is built on
        first access, so point-loss consumers never materialize it. The
        labelmap-native fields are built eagerly (they are small).

        Sampling is per-VIDEO, not per-(video, frame): the same K IDs are
        reused across all T frames so that a given object retains a stable
        flat index across the tracking loop. The K IDs are drawn uniformly
        at random (without replacement) from each video's `mask_ids` via
        `self.rng`, so we materialize a random subset rather than the first K.

        Extra fields (per-frame lists, each of length T, each tensor `[B*K_full, ...]`):
        - `labelmaps`: flat `[B*T, Z, Y, X]` int32 view of mask_labelmap, indexed
            by `img_ids[t]` (flat_id = b*T + t).
        - `instance_ids[t]`: int64, real labelmap id per row, `-1` for padded
            rows. Pad sentinel avoids collision with background `0`.
        - `valid[t]`: bool, True iff the row is a real selected object.
        - `presence_t[t]`: bool, True iff the object's id appears in frame `t`.
            Computed via `torch.isin` against `torch.unique(labelmap_b_t)`,
            avoiding the `[K, Z, Y, X]` materialization that `masks` uses.
        - `boxes[t]`: float32 `[B*K_full, 6]`, padded zeros, aligned with
            `instance_ids[t]`. Format declared in `box_format`.
        - `box_format`: string from preprocessor config (e.g. "zyxzyx").
        """
        B = num_videos
        T = num_frames
        # Rows per video: the largest instance count sampled in this batch,
        # capped by max_masks. max_masks is the memory/supervision cap; the
        # decoder and point sampler run over every row, so padding to it when
        # each sample has fewer instances is pure overhead.
        counts = [
            min(int(t["mask_ids"].numel()), self.max_masks)
            for t in targets[:B]
            if t.get("mask_ids", None) is not None
        ]
        K_full = max(1, max(counts)) if counts else 1

        if mask_labelmap.shape[0] != B or mask_labelmap.shape[1] != T:
            raise ValueError(
                f"mask_labelmap shape {tuple(mask_labelmap.shape)} does not match "
                f"(B={B}, T={T}, Z, Y, X)"
            )
        spatial = mask_labelmap.shape[2:]  # (Z, Y, X)

        # Per-video sampled ID subset. Default to empty so missing/short
        # targets (e.g. inference, dropped samples) just produce all-pad rows.
        empty_ids = torch.zeros(0, dtype=torch.long, device=device)
        sampled_ids_per_b: list[torch.Tensor] = [empty_ids] * B
        sampled_boxes_per_b: list[torch.Tensor | None] = [None] * B
        for b, tgt in enumerate(targets[:B]):
            ids_b = tgt.get("mask_ids", None)
            if ids_b is None or ids_b.numel() == 0:
                continue
            N = int(ids_b.numel())
            K = min(N, K_full)
            # Randomly pick which K of the N instances we materialize (without
            # replacement). torch.randperm runs on CPU with self.rng; the index
            # selection moves to ids_b's device for the gather. The same perm
            # selects the aligned boxes so rows stay consistent.
            perm = torch.randperm(N, generator=self.rng)[:K].to(ids_b.device)
            sampled_ids_per_b[b] = ids_b[perm]
            tgt_boxes = tgt.get("boxes", None)
            if tgt_boxes is not None and tgt_boxes.numel() > 0:
                # Boxes are 1:1 with mask_ids by construction (the collator
                # appends a box and an id in lockstep behind the same guard).
                # Gather with the SAME perm so row i of the boxes stays paired
                # with row i of sampled_ids. A count mismatch means the target
                # view is malformed; fail fast rather than guess the box->id
                # assignment (silent padding would misalign downstream prompts).
                if tgt_boxes.shape[0] != N:
                    raise ValueError(
                        f"boxes/mask_ids count mismatch for video {b}: "
                        f"{tgt_boxes.shape[0]} boxes vs {N} mask_ids. The collator "
                        f"emits exactly one box per mask id; this view is malformed."
                    )
                sampled_boxes_per_b[b] = tgt_boxes[perm]

        out_img_ids: list[torch.Tensor] = []
        out_instance_ids: list[torch.Tensor] = []
        out_valid: list[torch.Tensor] = []
        out_presence: list[torch.Tensor] = []
        out_boxes: list[torch.Tensor] = []

        sentinel_pad_id = -1

        for t in range(T):
            # Preallocate the per-frame block once and write each video's rows into
            # its slice.
            frame_ids = torch.zeros((B * K_full,), dtype=torch.int32, device=device)
            frame_instance = torch.full(
                (B * K_full,), sentinel_pad_id, dtype=torch.int64, device=device
            )
            frame_valid = torch.zeros((B * K_full,), dtype=torch.bool, device=device)
            frame_presence = torch.zeros((B * K_full,), dtype=torch.bool, device=device)
            frame_boxes = torch.zeros((B * K_full, 6), dtype=torch.float32, device=device)

            for b in range(B):
                lo = b * K_full
                sampled = sampled_ids_per_b[b]
                K = sampled.numel()
                flat_id = b * T + t

                # Every row of this video's block carries the frame's flat img id,
                # pad rows included -- they index the same frame.
                frame_ids[lo:lo + K_full] = flat_id
                if K == 0:
                    continue

                lm_bt = mask_labelmap[b, t]  # (Z, Y, X) int32

                frame_instance[lo:lo + K] = sampled.to(torch.int64)
                frame_valid[lo:lo + K] = True
                # cheap per-object presence: ids actually present in lm_bt
                unique_ids = torch.unique(lm_bt)
                frame_presence[lo:lo + K] = torch.isin(
                    sampled.to(unique_ids.dtype), unique_ids
                )

                boxes_src = sampled_boxes_per_b[b]
                if boxes_src is not None:
                    boxes_b_t = boxes_src.to(device=device, dtype=torch.float32)
                    if self.boxes_normalized:
                        Z_, Y_, X_ = (float(v) for v in spatial)
                        fmt = self.bbox_format.lower()
                        if fmt in ("cxcyczwhd", "xyzxyz"):
                            scale = boxes_b_t.new_tensor([X_, Y_, Z_, X_, Y_, Z_])
                        elif fmt == "zyxzyx":
                            scale = boxes_b_t.new_tensor([Z_, Y_, X_, Z_, Y_, X_])
                        else:
                            raise ValueError(f"unsupported bbox_format {self.bbox_format!r}")
                        boxes_b_t = boxes_b_t * scale
                    # sampled_boxes is gathered with the same perm as sampled_ids, so
                    # it must already have exactly K rows.
                    if boxes_b_t.shape[0] != K:
                        raise ValueError(
                            f"sampled boxes row count {boxes_b_t.shape[0]} != "
                            f"K={K} for video {b}; box/id alignment is broken."
                        )
                    frame_boxes[lo:lo + K] = boxes_b_t

            out_img_ids.append(frame_ids)
            out_instance_ids.append(frame_instance)
            out_valid.append(frame_valid)
            out_presence.append(frame_presence)
            out_boxes.append(frame_boxes)

        # Dense per-frame (B*K_full, Z, Y, X) bool blocks are built LAZILY:
        # only consumers that actually index `masks[t]` (dense-loss criterion,
        # SAM2 correction-point sampler) materialize them; the point-loss path
        # never pays the ~B*K*Z*Y*X-per-frame allocation.
        out_masks = _LazyFrameMasks(
            mask_labelmap=mask_labelmap,
            sampled_ids_per_b=sampled_ids_per_b,
            max_masks=K_full,
            device=device,
        )

        # Flatten (B, T, Z, Y, X) -> (B*T, Z, Y, X). With img_ids[t][i] = b*T + t,
        # mask_labelmap.reshape(B*T, ...) is indexed by flat_id without permutation.
        # mask_labelmap is already int32 (cast at snapshot in forward).
        flat_labelmaps = mask_labelmap.reshape(B * T, *spatial)

        return {
            "num_frames": T,
            "num_videos": B,
            "masks": out_masks,
            "img_ids": out_img_ids,
            "labelmaps": flat_labelmaps,
            "instance_ids": out_instance_ids,
            "valid": out_valid,
            "presence_t": out_presence,
            "boxes": out_boxes,
            "box_format": self.bbox_format,
        }

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def forward(self, data_sample: dict, data_time: float, idx: int) -> dict:
        preprocess_t0 = time.time()

        inputs = data_sample["data_tensor"]
        meta = data_sample.get("metainfo", {})

        # Fast split: signal channels off the front (view + dtype cast), object
        # channels off the int32 tail. `mask_labelmap` is (B, T, Z, Y, X) or None
        # at inference (no mask channel -> no labelmap, no masks built).
        images, targets_by_role = self._split_channels(inputs, meta)
        mask_labelmap = targets_by_role.get(DataKind.INSTANCE_MASKS.value)

        # Attach the per-target labelmap BEFORE transforms so geometric ops
        # (Crop / Resize / flip) warp it in lockstep with the image and
        # boxes; _build_data_views then reassembles it from the per-target
        # source. This makes the preprocessor the sole owner of the labelmap.
        pre_targets = meta.get("targets", [])
        if mask_labelmap is not None:
            for b, t in enumerate(pre_targets):
                if b < mask_labelmap.shape[0]:
                    t["label_map"] = mask_labelmap[b]

        # --- apply any configured transforms ------------------------------------
        sample = {"data_tensor": images, "metainfo": meta}
        sample, transform_time = self._apply_transforms(sample)
        images = sample["data_tensor"]
        meta = sample["metainfo"]

        B, T = images.shape[0], images.shape[1]

        # END-of-forward dtype boundary (SAM2 does not use _finalize): narrow the
        # exact float32 intermediate from _split_channels to the model dtype.
        if images.dtype != self.dtype:
            images = images.to(self.dtype)

        # data_tensor stays in the platform layout (B, T, Z, Y, X, C). SAM2 flattens
        # and permutes to its own (B*T, C, Z, Y, X) at the model boundary -- see
        # SAM2._to_model_layout. img_ids/labelmaps below are plain b*T + t arithmetic,
        # so they are layout-independent and stay valid.
        flat_img_batch = images

        # --- build per-frame masks & index maps ----------------------------
        # No mask channel (inference) -> emit an empty target view and skip
        # all mask generation.
        targets = meta.pop("targets", [])
        if mask_labelmap is None:
            spatial = tuple(images.shape[2:-1])  # (Z, Y, X)
            data_views = self._empty_data_views(
                num_frames=T,
                num_videos=B,
                spatial=spatial,
                device=flat_img_batch.device,
            )
        else:
            # Reassemble the (possibly transformed) labelmap from its per-target
            # source. Transforms keep image and labelmap aligned; assert that
            # invariant cheaply rather than re-deriving from the raw channel.
            if not targets or any("label_map" not in t for t in targets):
                # torch.stack([]) would RuntimeError with an opaque message;
                # name the broken contract instead.
                raise ValueError(
                    "SAM2VideoPreprocessor: expect_mask_channel=True and a "
                    "labelmap channel was split off, but metainfo['targets'] is "
                    f"empty or missing 'label_map' entries ({len(targets)} targets). "
                    "The collator must emit one target dict per video (the "
                    "preprocessor attaches label_map to each before transforms); "
                    "for GT-free inference set expect_mask_channel=False."
                )
            mask_labelmap = torch.stack([t["label_map"] for t in targets])
            if images.shape[1:-1] != mask_labelmap.shape[1:]:
                raise ValueError(
                    f"image/labelmap spatial mismatch after transforms: "
                    f"{tuple(images.shape[1:-1])} != {tuple(mask_labelmap.shape[1:])}"
                )
            data_views = self._build_data_views(
                targets=targets,
                num_frames=T,
                num_videos=B,
                device=flat_img_batch.device,
                mask_labelmap=mask_labelmap,
            )

        # Publish GT under the platform contract: metainfo["targets"] is Form S --
        # a per-image list of dicts (labels/boxes/mask_ids/label_map; see
        # data/data_types.py) -- which is what evaluate_postprocess.extract_targets
        # and every evaluator expect. The SAM2-specific per-frame view the model
        # needs goes under its own key (a DERIVED layout cache, not a second GT).
        # Shallow-copy each target (tensors shared by reference) and squeeze the
        # single-timepoint label_map (T,Z,Y,X) -> (Z,Y,X) as a VIEW -- no voxel copy.
        # Empty on the no-GT inference path (targets == []).
        gt_list = []
        for t in targets:
            g = dict(t)  # shallow: shares every tensor by reference
            lm = g.get("label_map")
            # Squeeze the single-timepoint case to the (Z, Y, X) the 3D instance
            # evaluator expects -- a VIEW, no voxel copy. Multi-frame label_maps pass
            # through as (T, Z, Y, X). Evaluation is single-timepoint only, and 
            # extract_targets(squeeze_label_map=True)
            # raises its own clear error if a multi-frame sample reaches an evaluator.
            if lm is not None and torch.is_tensor(lm) and lm.dim() == 4 and lm.shape[0] == 1:
                g["label_map"] = lm[0]  # view -> (Z, Y, X)
            gt_list.append(g)

        sam2_metrics: list[dict[str, Any]] = [
            make_timing_metric("data_time", data_time),
            make_timing_metric("preprocess_time", time.time() - preprocess_t0),
            make_timing_metric("transform_time", transform_time),
        ]
        existing_metrics = meta.get("metrics") if isinstance(meta, Mapping) else None
        if existing_metrics:
            sam2_metrics = list(existing_metrics) + sam2_metrics

        return {
            "data_tensor": flat_img_batch,
            "metainfo": {
                **meta,
                "targets": gt_list,        # platform contract: List[per-image dict]
                "sam2_views": data_views,  # model-private per-frame view
                "idx": idx,
                "metrics": sam2_metrics,
            },
        }
