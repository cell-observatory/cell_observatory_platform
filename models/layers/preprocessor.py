import os
import time
import math
import functools
from typing import Any, Dict, Optional, Tuple, List, Callable, Mapping, Union
from dataclasses import dataclass, asdict, field

import numpy as np

import torch

from omegaconf import DictConfig
from dataclasses import dataclass
from hydra.utils import get_method, instantiate

from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_platform.data.io import read_file
from cell_observatory_platform.data.structures import convert_bbox_format
from cell_observatory_platform.data.utils import create_na_masks, downsample, resize_mask
from cell_observatory_platform.models.layers.patch_embeddings import PatchEmbedding, calc_num_patches
from cell_observatory_platform.training.helpers import get_patch_sizes

# --------------------------------------------------------------------------- #
# Pretraining preprocessor
# --------------------------------------------------------------------------- #


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

    def forward(self, data_sample: dict, data_time: float, idx: int) -> dict:
        """
        Preprocess the input data sample to maintain uniform data
        layout accross data loaders.
        """
        preprocess_time = time.time()

        inputs = data_sample["data_tensor"]
        meta = data_sample["metainfo"]

        if inputs.dtype != self.dtype:
            # ray.logger.warning(f"Casting inputs to {self.dtype}")
            inputs = inputs.to(self.dtype)

        # skipping checks for NaN/Inf values
        # if torch.isnan(inputs).all() or torch.isinf(inputs).all():
        #     raise ValueError(f"Invalid training data")

        if self.transforms is not None:
            transform_t0 = time.time()
            for transform in self.transforms:
                inputs = transform(inputs)
            transform_time = time.time() - transform_t0

        assert inputs.dtype == self.dtype, f"{inputs.dtype} != {self.dtype}"

        if self.with_masking:
            masking_time = time.time()
            mask_data = self.mask_generator(inputs.shape[0])
            masking_time = time.time() - masking_time

            mask_lists = {k: [v] for k, v in mask_data.items() if v is not None}
            metainfo = {
                **mask_lists,
                "preprocess_time": time.time() - preprocess_time,
                "data_time": data_time,
                "masking_time": masking_time,
                "transform_time": transform_time if self.transforms is not None else -1,
                **meta,
                "idx": idx,
            }
            metainfo["spatial_kwargs"] = self._build_spatial_kwargs(
                batch_size=inputs.shape[0], device=inputs.device,
            )
            return {"data_tensor": inputs, "metainfo": metainfo}
        else:
            metainfo = {
                "preprocess_time": time.time() - preprocess_time,
                "data_time": data_time,
                "masking_time": -1.0,
                "transform_time": transform_time if self.transforms is not None else -1,
                **meta,
                "idx": idx,
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


class MultiSequenceRayPreprocessor(torch.nn.Module):
    """
    Batch preprocessor for multiple dataset streams 
    (e.g., DINO global/local crops, or multiple datasets with different shapes).

    Output:
      {
        "data_tensors": {name: tensor, ...},
        "metainfo": {
            ... original meta ...,
            "dataset_stream_metainfo": {
                name: {
                    "input_format": ...,
                    "input_shape": ...,
                    "patch_shape": ...,
                    "transform_time": float,
                    "masking_time": float,
                    # masking outputs if enabled:
                    "masks": [Tensor] (optional),
                    "context_masks": [Tensor] (optional),
                    "target_masks": [Tensor] (optional),
                    "original_patch_indices": [Tensor] (optional),
                    "channels_to_mask": [Any] (optional),
                    "patches_used": [Tensor] (optional),
                },
            },
            "preprocess_time": float,
            "data_time": float,
        }
      }
    """

    def __init__(
        self,
        local_batch_size: int,
        global_batch_size: int,
        input_metadict: Mapping[str, Union[InputDataStreamSpec, Mapping[str, Any]]],
        transforms_metadict: Optional[Mapping[str, Optional[List[Any]]]] = None,
        mask_generators: Optional[Mapping[str, Any]] = None,
        dtype: torch.dtype | str = "bfloat16",
    ):
        super().__init__()

        # TODO: decide if this should be property of Preprocessor or DataLoader
        self.local_batch_size = local_batch_size
        self.global_batch_size = global_batch_size

        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        data_streams: Dict[str, InputDataStreamSpec] = {}
        for name, spec in input_metadict.items():
            if isinstance(spec, InputDataStreamSpec):
                data_streams[name] = spec
            else:
                data_streams[name] = InputDataStreamSpec(**dict(spec))
        self.data_streams = data_streams

        # Per-dataset stream transforms
        self.transforms: Dict[str, List[Callable]] = {}
        transforms_metadict = transforms_metadict or {}
        for name in self.data_streams.keys():
            self.transforms[name] = _instantiate_transform_list(transforms_metadict.get(name, None))
        # NOTE: for single dataset stream, we use the BASE transforms pipeline to create the dataset stream tensors
        #       for example, see multicrop augmentation for dino
        self.transforms["BASE"] = _instantiate_transform_list(transforms_metadict.get("BASE", None))

        # Per-dataset stream mask generators
        self.mask_generators = dict(mask_generators or {})

    def _apply_dataset_stream_transforms(
        self,
        name: str,
        x: torch.Tensor,
        meta: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any], float]:
        """
        Apply transforms for a given dataset stream.
        """
        t0 = time.time()
        data_dict: Dict[str, Any] = {"data_tensor": x, "metainfo": meta}

        for tr in self.transforms.get(name, []):
            out = tr(data_dict)
            if isinstance(out, torch.Tensor):
                data_dict["data_tensor"] = out
            elif isinstance(out, dict):
                if "data_tensor" not in out:
                    raise KeyError(f"Transform for dataset stream={name!r} returned dict without 'data_tensor'.")
                data_dict = out
                data_dict.setdefault("metainfo", {})
            else:
                raise TypeError(
                    f"Transform for dataset stream={name!r} must return Tensor or dict; got {type(out)}."
                )

        return data_dict["data_tensor"], data_dict.get("metainfo", {}), (time.time() - t0)

    def _apply_dataset_stream_masking(
        self,
        name: str,
        x: torch.Tensor,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Calls the dataset stream's mask generator if enabled.

        Expects mask generator to return a dict with optional keys (unused as None).
        Values are wrapped in single-element lists and merged into the stream's metainfo.
        """
        spec = self.data_streams[name]
        if not spec.with_masking:
            return {}, -1.0

        mask_gen = self.mask_generators.get(name, None)
        if mask_gen is None:
            raise ValueError(f"with_masking=True for stream={name!r} but no mask generator was provided.")

        B = x.shape[0]
        mt0 = time.time()
        mask_data = mask_gen(B)
        masking_time = time.time() - mt0

        if not isinstance(mask_data, dict):
            raise TypeError(
                f"Mask generator for stream={name!r} returned {type(mask_data)}, expected dict."
            )
        mask_kwargs = {k: [v] for k, v in mask_data.items() if v is not None}
        return mask_kwargs, masking_time

    def forward(self, data_sample: Any, data_time: float, idx: int):
        preprocess_t0 = time.time()

        if isinstance(data_sample, dict):
            # TODO: dataset, dataloader, and database stack currently does not support this branch properly
            if "data_tensors" in data_sample and isinstance(data_sample["data_tensors"], dict):
                input_mode = "multi_dataset_streams"
                # NOTE: we match the tensor and metadict names to the outputs of the base transforms
                # in the single_dataset_stream mode
                dataset_streams_tensors = data_sample["data_tensors"]
                dataset_streams_meta = data_sample.get("metainfo", {})
            elif "data_tensor" in data_sample and torch.is_tensor(data_sample["data_tensor"]):
                input_mode = "single_dataset_stream"
                in_tensors = data_sample["data_tensor"]
                input_dataset_stream_metainfo = data_sample.get("metainfo", {})
            else:
                raise KeyError(
                    "MultiSequenceRayPreprocessor expects either:\n"
                    "  - {'data_tensors': {name: tensor, ...}, 'metainfo': {...}}  OR\n"
                    "  - {'data_tensor': tensor, 'metainfo': {...}}  OR\n"
                )
        else:
            raise TypeError(f"data_sample must be a dict; got {type(data_sample)}")

           # Preserve collator/dataloader metainfo (e.g. device_buffer_idx, host_buffer_idx) for hooks
        if input_mode == "single_dataset_stream":
            original_metainfo = dict(input_dataset_stream_metainfo)
        else:
            original_metainfo = dict(data_sample.get("metainfo", {}))

        # If we have 1 dataset stream, apply base transforms to generate a dict-of-tensors
        if input_mode == "single_dataset_stream":
            if not torch.is_tensor(in_tensors):
                raise TypeError(f"data_tensor must be a torch.Tensor; got {type(in_tensors)}")

            base_dtype = TORCH_DTYPES[self.dtype].value if isinstance(self.dtype, str) else self.dtype
            if in_tensors.dtype != base_dtype:
                in_tensors = in_tensors.to(base_dtype)

            base_transforms = self.transforms.get("BASE", [])
            assert base_transforms, "single_dataset_stream mode requires a BASE transforms pipeline."

            data_dict: Dict[str, Any] = {"data_tensor": in_tensors, "metainfo": dict(input_dataset_stream_metainfo)}
            t_base0 = time.time()
            for tr in base_transforms:
                data_dict = tr(data_dict)

            base_transform_time = time.time() - t_base0

            # Interpret output as dataset streams
            if "data_tensor" in data_dict and isinstance(data_dict["data_tensor"], dict):
                dataset_streams_tensors = data_dict["data_tensor"]
                dataset_streams_meta = data_dict.get("metainfo", {})
            else:
                raise KeyError(
                    "Base transforms pipeline must create streams by returning:\n"
                    "  - {'data_tensors': {name: tensor, ...}, 'metainfo': ...} "
                    f"Got keys={list(data_dict.keys())} and data_tensor type={type(data_dict.get('data_tensor', None))}."
                )
        else:
            base_transform_time = 0.0

        if not isinstance(dataset_streams_tensors, dict):
            raise TypeError(f"Expected dict-of-tensors, got {type(dataset_streams_tensors)}")

        expected_dataset_streams = set(self.data_streams.keys())
        dataset_streams_tensors_keys = set(dataset_streams_tensors.keys())
        assert dataset_streams_tensors_keys == expected_dataset_streams, (
            "Dataset streams produced by data ingestion pipeline don't match data_streams.\n"
            f"produced={sorted(dataset_streams_tensors_keys)}\n"
            f"expected={sorted(expected_dataset_streams)}"
        )

        # Apply per-dataset stream transforms
        transform_time_by_dataset_stream: Dict[str, float] = {}

        for name in dataset_streams_tensors.keys():
            x = dataset_streams_tensors[name]
            if not torch.is_tensor(x):
                raise TypeError(f"Produced dataset stream tensor {name!r} is not a tensor; got {type(x)}")

            dataset_stream_meta = dataset_streams_meta[name]

            # apply dataset stream-specific transforms (if any)
            x, dataset_stream_meta, t_dataset_stream = self._apply_dataset_stream_transforms(name=name, x=x, meta=dataset_stream_meta)

            dataset_streams_tensors[name] = x
            dataset_streams_meta[name] = dataset_stream_meta
            transform_time_by_dataset_stream[name] = float(base_transform_time + t_dataset_stream)

        # Apply per-dataset stream dtype conversion + (optional) masking
        dataset_stream_metainfo: Dict[str, Dict[str, Any]] = {}

        for name, x in dataset_streams_tensors.items():
            if not torch.is_tensor(x):
                raise TypeError(f"Produced stream {name!r} is not a tensor; got {type(x)}")

            spec = self.data_streams[name]

            # Cast to per-dataset stream dtype if we have a spec
            target_dtype = TORCH_DTYPES[spec.dtype].value if isinstance(spec.dtype, str) else spec.dtype
            if x.dtype != target_dtype:
                x = x.to(target_dtype)

            if spec.with_masking:
                mask_meta, masking_time = self._apply_dataset_stream_masking(name=name, x=x)
            else:
                mask_meta = {}
                masking_time = -1.0
            
            dataset_stream_metainfo[name] = {
                "input_format": spec.input_format,
                "input_shape": spec.input_shape,
                "patch_shape": spec.patch_shape,
                "transform_time": float(transform_time_by_dataset_stream[name]),
                "masking_time": float(masking_time),
                **mask_meta,
                **dataset_streams_meta[name],
            }

        out_meta = original_metainfo

        out_meta["dataset_stream_metainfo"] = dataset_stream_metainfo
        out_meta["preprocess_time"] = float(time.time() - preprocess_t0)
        out_meta["data_time"] = float(data_time)

        out_meta["idx"] = idx
        out_meta["local_batch_size"] = self.local_batch_size
        out_meta["global_batch_size"] = self.global_batch_size

        return {"data_tensors": dataset_streams_tensors, "metainfo": out_meta}


# --------------------------------------------------------------------------- #
# Base Finetune preprocessor
# --------------------------------------------------------------------------- #


class BaseFinetunePreprocessor(RayPreprocessor):
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
        mask_channel_idx: int | None,
        seed: int | None = None,
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
        self.channels = axis_to_size.get("C", None)
        self.mask_channel_idx = mask_channel_idx
        if self.channels is not None and self.mask_channel_idx is not None:
            self.channels -= 1
            assert self.channels > 0, "Expected at least 1 channel after mask channel removal"
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

    def _apply_transforms(self, data: torch.Tensor | dict) -> tuple[torch.Tensor | dict, float]:
        """
        Apply transforms to either:
          - a torch.Tensor (image only), or
          - a dict with keys {"data_tensor", "metainfo"}.
        Each transform is responsible for returning the same type it was given.
        """
        if self.transforms is not None:
            t0 = time.time()
            for transform in self.transforms:
                data = transform(data)
            transform_time = time.time() - t0
        else:
            transform_time = -1.0
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
        """Attach masking info and timing, returning the standard dict."""

        if self.with_masking:
            mt0 = time.time()
            B = inputs.shape[0]
            mask_data = self.mask_generator(B)
            masking_time = time.time() - mt0

            mask_lists = {k: [v] for k, v in mask_data.items() if v is not None}
            return {
                "data_tensor": inputs,
                "metainfo": {
                    **meta,
                    **mask_lists,
                    "targets": [targets],
                    "preprocess_time": time.time() - preprocess_t0,
                    "data_time": data_time,
                    "masking_time": masking_time,
                    "transform_time": transform_time,
                    "idx": idx,
                },
            }
        else:
            return {
                "data_tensor": inputs,
                "metainfo": {
                    **meta,
                    "targets": [targets],
                    "preprocess_time": time.time() - preprocess_t0,
                    "data_time": data_time,
                    "transform_time": transform_time,
                    "masking_time": -1.0,
                    "idx": idx,
                },
            }


# --------------------------------------------------------------------------- #
# Denoising task
# --------------------------------------------------------------------------- #


class DenoisingPreprocessor(BaseFinetunePreprocessor):
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
        mask_channel_idx: int | None,
        seed: int | None = None,
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
            mask_channel_idx=mask_channel_idx,
            seed=seed,
        )

        self.denoising_type = denoising_type

    def _split_inputs_and_mask(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        inputs: (B, Z, Y, X, C_full)
        returns:
          inputs_wo_mask: (B, Z, Y, X, C_full-1)
          masks_labelmap: (B, Z, Y, X)
        """
        assert inputs.ndim == 5, f"Expected (B, Z, Y, X, C), got {inputs.shape}"
        B, Z, Y, X, C = inputs.shape
        
        if self.mask_channel_idx is None:
            return inputs, None

        if C < 2:
            raise ValueError(f"Expected at least 2 channels (image + mask), got C={C}")

        # For zero-copy we *require* the mask to be the last channel
        if self.mask_channel_idx not in (-1, C - 1):
            raise ValueError(
                f"For zero-copy split, mask_channel_idx must be -1 or C-1; " f"got mask_channel_idx={self.mask_channel_idx}, C={C}."
            )

        masks = inputs[..., -1].clone()  # (B, Z, Y, X), view
        inputs_wo_mask = inputs[..., :-1]  # (B, Z, Y, X, C-1), view
        
        return inputs_wo_mask, masks

    def forward(self, data_sample: dict, data_time: float, idx: int) -> dict:
        inputs, meta, preprocess_t0, data_time_value = self._common_pre(
            data_sample=data_sample,
            data_time=data_time,
        )
        inputs_wo_mask, _ = self._split_inputs_and_mask(inputs)
        sample = {
            "data_tensor": inputs_wo_mask,
            "metainfo": meta,
        }

        sample, transform_time = self._apply_transforms(sample)

        # TODO: Consider refactoring this to support non-transformer-based decoders
        # Patchify targets for transformer-based decoders
        targets = self.pe_patchify(
            sample["metainfo"]["targets"][0], 
            channels=self.channels,
        )

        # FIXME: Streamline this so that we either consistently pass data_sample 
        # or its components (e.g. data_tensor, metainfo, targets, etc.)
        return self._finalize(
            inputs=sample["data_tensor"],
            meta=sample["metainfo"],
            targets=targets,
            data_time=data_time_value,
            preprocess_t0=preprocess_t0,
            transform_time=transform_time,
            idx=idx,
        )


# --------------------------------------------------------------------------- #
# Channel-splitting task
# --------------------------------------------------------------------------- #


class ChannelSplitPreprocessor(BaseFinetunePreprocessor):
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
        mask_channel_idx: int | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            transforms_list=transforms_list,
            with_masking=with_masking,
            mask_generator=mask_generator,
            patch_shape=patch_shape,
            dtype=dtype,
            input_format=input_format,
            input_shape=input_shape,
            mask_channel_idx=mask_channel_idx,
            seed=seed,
        )

    def _split_inputs_and_mask(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Remove mask channel if present; return inputs (with or without mask stripped)."""
        assert inputs.ndim == 5, f"Expected (B, Z, Y, X, C), got {inputs.shape}"
        B, Z, Y, X, C = inputs.shape

        if self.mask_channel_idx is None:
            return inputs, None

        if C < 2:
            raise ValueError(
                f"Expected at least 2 channels (image + mask), got C={C}"
            )
        if self.mask_channel_idx not in (-1, C - 1):
            raise ValueError(
                f"For zero-copy split, mask_channel_idx must be -1 or C-1; "
                f"got mask_channel_idx={self.mask_channel_idx}, C={C}."
            )

        masks = inputs[..., -1].clone()
        inputs_wo_mask = inputs[..., :-1]
        return inputs_wo_mask, masks

    def forward(self, data_sample: dict, data_time: float, idx: int) -> dict:
        inputs, meta, preprocess_t0, data_time_value = self._common_pre(
            data_sample=data_sample,
            data_time=data_time,
        )

        if self.channel_idx is None:
            raise ValueError("Channel axis 'C' not present in input_format; cannot channel_split.")

        inputs_wo_mask, _ = self._split_inputs_and_mask(inputs)

        # FIXME: consider if this is the correct order of operations
        inputs_wo_mask, transform_time = self._apply_transforms(data=inputs_wo_mask)

        # targets are per-channel patches from original (transformed) input
        # input_shape must describe data after mask strip; self.channels matches inputs_wo_mask
        targets = self.pe_patchify(inputs_wo_mask, channels=self.channels)

        # model input: average over channels -> [B, ..., 1]
        inputs = inputs_wo_mask.mean(dim=self.channel_idx, keepdim=True)

        return self._finalize(
            inputs=inputs,
            meta=meta,
            targets=targets,
            data_time=data_time_value,
            preprocess_t0=preprocess_t0,
            transform_time=transform_time,
            idx=idx,
        )


# --------------------------------------------------------------------------- #
# Upsampling tasks (space / spacetime / time)
# --------------------------------------------------------------------------- #


class UpsamplePreprocessor(BaseFinetunePreprocessor):
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
        mask_channel_idx: int = -1,
        mode: str = "upsample_space",
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
            mask_channel_idx=mask_channel_idx,
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
        inputs, meta, preprocess_t0, data_time_value = self._common_pre(
            data_sample=data_sample,
            data_time=data_time,
        )

        inputs, transform_time = self._apply_transforms(data=inputs)

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
                inputs=data_sample["data_tensor"],
                spatial_dims=self.spatial_dims,
            )
        elif self.mode == "upsample_time":
            targets = None
        else:
            raise RuntimeError(f"Unexpected mode: {self.mode}")

        return self._finalize(
            inputs=inputs,
            meta=meta,
            targets=targets,
            data_time=data_time_value,
            preprocess_t0=preprocess_t0,
            transform_time=transform_time,
            idx=idx,
        )


# --------------------------------------------------------------------------- #
# Instance Segmentation task
# --------------------------------------------------------------------------- #


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
        mask_channel_idx: int = -1,
        bbox_data_format: Optional[str] = None,
        bbox_output_format: Optional[str] = None,
        debug_savepath: str = None,
        require_targets: bool = True,
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
            mask_channel_idx=mask_channel_idx,
        )

        if bbox_data_format is None or bbox_output_format is None:
            raise ValueError("bbox_data_format and bbox_output_format must be specified for instance_segmentation.")
        self.bbox_data_format = bbox_data_format
        self.bbox_output_format = bbox_output_format

        self.debug_savepath = debug_savepath
        self.require_targets = require_targets

    def _split_inputs_and_mask(self, inputs: torch.Tensor):
        """
        inputs: (B, Z, Y, X, C_full)
        returns:
          inputs_wo_mask: (B, Z, Y, X, C_full-1)
          masks_labelmap: (B, Z, Y, X)
        """
        assert inputs.ndim == 5, f"Expected (B, Z, Y, X, C), got {inputs.shape}"
        B, Z, Y, X, C = inputs.shape

        if C < 2:
            raise ValueError(f"Expected at least 2 channels (image + mask), got C={C}")

        # For zero-copy we *require* the mask to be the last channel
        if self.mask_channel_idx not in (-1, C - 1):
            raise ValueError(
                f"For zero-copy split, mask_channel_idx must be -1 or C-1; " f"got mask_channel_idx={self.mask_channel_idx}, C={C}."
            )

        masks = inputs[..., -1]  # (B, Z, Y, X), view
        inputs_wo_mask = inputs[..., :-1]  # (B, Z, Y, X, C-1), view

        return inputs_wo_mask, masks

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
        inputs, meta, t0, data_time_value = self._common_pre(data_sample, data_time)

        if self.mask_channel_idx is not None:
            inputs_wo_mask, _ = self._split_inputs_and_mask(inputs)
        else:
            inputs_wo_mask = inputs

        sample = {
            "data_tensor": inputs_wo_mask,
            "metainfo": meta,
        }
        sample, transform_time = self._apply_transforms(sample)

        if self.debug_savepath is not None:
            self._debug_visualize_batch(sample)

        inputs = sample["data_tensor"]
        meta = sample["metainfo"]
        targets = meta.pop("targets", None)
        if targets is None:
            # Keep downstream contract: list[dict] length B (then _finalize wraps as [targets]).
            B = inputs.shape[0]
            targets = [
                {
                    "boxes": torch.zeros((0, 6), device=inputs.device, dtype=torch.float32),
                    "mask_ids": torch.zeros((0,), device=inputs.device, dtype=torch.long),
                    "labels": torch.zeros((0,), device=inputs.device, dtype=torch.long),
                }
                for _ in range(B)
            ]

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
        mask_channel_idx: int = 1,
        bbox_data_format: Optional[str] = None,
        bbox_output_format: Optional[str] = None,
        debug_savepath: str = None,
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
            mask_channel_idx=mask_channel_idx,
        )

        self.debug_savepath = debug_savepath

    def _split_inputs_and_mask(self, inputs: torch.Tensor):
        """
        inputs: (B, Z, Y, X, C_full)
        returns:
          inputs_wo_mask: (B, Z, Y, X, C_full-1)
          masks_labelmap: (B, Z, Y, X)
        """
        assert inputs.ndim == 5, f"Expected (B, Z, Y, X, C), got {inputs.shape}"
        B, Z, Y, X, C = inputs.shape

        if self.mask_channel_idx is None:
            masks = torch.zeros(
                (B, Z, Y, X),
                dtype=torch.bool, 
                device=inputs.device
                )
            return inputs, masks

        if C < 2:
            raise ValueError(f"Expected at least 2 channels (image + mask), got C={C}")

        # For zero-copy we *require* the mask to be the last channel
        if self.mask_channel_idx not in (-1, C - 1):
            raise ValueError(
                f"For zero-copy split, mask_channel_idx must be -1 or C-1; " f"got mask_channel_idx={self.mask_channel_idx}, C={C}."
            )

        masks = inputs[..., -1]  # (B, Z, Y, X), view
        inputs_wo_mask = inputs[..., :-1]  # (B, Z, Y, X, C-1), view

        return inputs_wo_mask, masks

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
        inputs, meta, t0, data_time_value = self._common_pre(data_sample, data_time)

        inputs_wo_mask, masks_labelmap = self._split_inputs_and_mask(inputs)

        sample = {
            "data_tensor": inputs_wo_mask,
            "metainfo": meta,
            "masks_labelmap": masks_labelmap, # This is used to generate the boundary masks
        }
        sample, transform_time = self._apply_transforms(sample)
        
        semantic_masks = []
        if "boundary_masks" in sample:
            semantic_masks.append(sample["boundary_masks"])
        if "foreground_masks" in sample:
            semantic_masks.append(sample["foreground_masks"])
        semantic_masks = torch.stack(semantic_masks, dim=1)  # [B, N_masks, D, H, W]
        labels = torch.arange(
            semantic_masks.shape[1], dtype=torch.int64, device=semantic_masks.device
        )  # [N_masks]
        targets = []
        for batch_idx in range(semantic_masks.shape[0]):
            targets.append({
                "masks": semantic_masks[batch_idx],  # [N_masks, D, H, W]
                "labels": labels.clone(),  # [N_masks]
            })
        meta["targets"] = targets # List[Dict[str, Tensor | Tuple]]
        
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
        mask_channel_idx: int = -1,
        bbox_data_format: Optional[str] = None,
        bbox_output_format: Optional[str] = None,
        debug_savepath: str = None,
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
            mask_channel_idx=mask_channel_idx,
        )

        if bbox_data_format is None or bbox_output_format is None:
            raise ValueError("bbox_data_format and bbox_output_format must be specified for instance_segmentation.")
        self.bbox_data_format = bbox_data_format
        self.bbox_output_format = bbox_output_format

        self.debug_savepath = debug_savepath

    def _split_inputs_and_mask(self, inputs: torch.Tensor):
        """
        inputs: (B, Z, Y, X, C_full)
        returns:
          inputs_wo_mask: (B, Z, Y, X, C_full-1)
          masks_labelmap: (B, Z, Y, X)
        """
        assert inputs.ndim == 5, f"Expected (B, Z, Y, X, C), got {inputs.shape}"
        B, Z, Y, X, C = inputs.shape

        if C < 2:
            raise ValueError(f"Expected at least 2 channels (image + mask), got C={C}")

        # For zero-copy we *require* the mask to be the last channel
        if self.mask_channel_idx not in (-1, C - 1):
            raise ValueError(
                f"For zero-copy split, mask_channel_idx must be -1 or C-1; " f"got mask_channel_idx={self.mask_channel_idx}, C={C}."
            )

        masks = inputs[..., -1]  # (B, Z, Y, X), view
        inputs_wo_mask = inputs[..., :-1]  # (B, Z, Y, X, C-1), view

        return inputs_wo_mask, masks

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
        inputs, meta, t0, data_time_value = self._common_pre(data_sample, data_time)

        inputs_wo_mask, masks_labelmap = self._split_inputs_and_mask(inputs)

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
        mask_channel_idx: int = -1,
        expect_mask_channel: bool = True,
        max_masks: int | None = None,
        require_targets: bool = True,
        defer_binary_masks: bool = False,
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
            mask_channel_idx=mask_channel_idx,
        )
        if "T" not in self.input_format:
            raise ValueError(
                f"SAM2VideoPreprocessor requires temporal dimension 'T' in input_format, "
                f"got {self.input_format!r}"
            )
        self.expect_mask_channel = expect_mask_channel
        self.max_masks = max_masks
        # When True, build per-instance binary masks on-device from the
        # labelmap (last channel of data_tensor) instead of consuming
        # pre-built masks from targets[b]["masks"]. Pairs with the matching
        # FinetuneCollatorActor flag of the same name. Only the K=max_masks
        # slices that downstream actually uses get materialized.
        self.defer_binary_masks = defer_binary_masks
        if self.defer_binary_masks and not self.expect_mask_channel:
            raise ValueError(
                "defer_binary_masks=True requires expect_mask_channel=True "
                "(the labelmap rides on the last channel of data_tensor)."
            )
        if self.defer_binary_masks and self.max_masks is None:
            raise ValueError("defer_binary_masks=True requires max_masks to be set")

    # ------------------------------------------------------------------ #
    # Image transformations
    # ------------------------------------------------------------------ #

    def _strip_mask_channel(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        inputs:  (B, T, Z, Y, X, C)
        returns: images (B, T, Z, Y, X, C-1), mask_labelmap (B, T, Z, Y, X) or None
        """
        # TODO: generalize
        if not self.expect_mask_channel:
            return inputs, None
        images = inputs[..., :-1]             # (B, T, Z, Y, X, C-1)
        mask_labelmap = inputs[..., -1]       # (B, T, Z, Y, X)
        return images, mask_labelmap

    @staticmethod
    def _build_flat_img_batch(images: torch.Tensor) -> torch.Tensor:
        """(B, T, Z, Y, X, C) -> (T, B, C, Z, Y, X)"""
        B, T, Z, Y, X, C = images.shape
        images = images.permute(1, 0, 5, 2, 3, 4).contiguous()
        return images.reshape(T * B, C, Z, Y, X)

    # ------------------------------------------------------------------ #
    # Mask & index transformations
    # ------------------------------------------------------------------ #

    def _build_data_views(
        self,
        targets: list[dict],
        num_frames: int,
        num_videos: int,
        device: torch.device,
        mask_labelmap: torch.Tensor | None = None,
    ) -> dict:
        """
        Build per-frame masks + flat object-to-image indices from per-batch targets.
        Returns dict ready to stash in data_sample["metainfo"]["data_views"].

        Two modes:
          - mask_labelmap is None (default, eager):
              consume pre-built binary masks from targets[b]["masks"].
          - mask_labelmap is not None (lazy, paired with defer_binary_masks=True):
              materialize K=min(N_inst, max_masks) binary masks per (b,t)
              on-device by comparing the labelmap to a sampled subset of
              targets[b]["mask_ids"]. Avoids ever building (N_inst − max_masks)
              masks that the head loop would only throw away.
        """
        if mask_labelmap is not None:
            return self._build_data_views_lazy(
                targets=targets,
                num_frames=num_frames,
                num_videos=num_videos,
                device=device,
                mask_labelmap=mask_labelmap,
            )

        B = num_videos
        T = num_frames

        # Accumulate per (frame t, video b) so we can cap/pad per flat index b*T+t
        masks_per_frame: list[list[torch.Tensor]] = [
            [torch.zeros(0, *self.spatial_shape, dtype=torch.bool, device=device) for _ in range(B)]
            for _ in range(T)
        ]
        flat_idx_per_frame: list[list[torch.Tensor]] = [
            [torch.zeros(0, dtype=torch.int32, device=device) for _ in range(B)]
            for _ in range(T)
        ]

        for b, tgt in enumerate(targets):
            inst_masks = tgt.get("masks", None)
            if inst_masks is None or inst_masks.numel() == 0:
                continue

            assert inst_masks.ndim == 5, (
                f"Expected masks shape (N,T,Z,Y,X) got {inst_masks.shape}"
            )
            N_inst = inst_masks.shape[0]

            for t in range(T):
                masks_per_frame[t][b] = inst_masks[:, t].bool()  # (N_inst, Z, Y, X)
                flat_idx_per_frame[t][b] = torch.full(
                    (N_inst,), b * T + t,
                    dtype=torch.int32, device=device,
                )

        # Build out_masks / out_img_ids: per-frame
        out_masks: list[torch.Tensor] = []
        out_img_ids: list[torch.Tensor] = []

        for t in range(T):
            if self.max_masks is not None:
                # Each unique (b*T+t) gets exactly max_masks slots
                seg_masks: list[torch.Tensor] = []
                seg_ids: list[torch.Tensor] = []
                for b in range(B):
                    m = masks_per_frame[t][b]
                    ids = flat_idx_per_frame[t][b]
                    N_bt = m.shape[0]
                    flat_id = b * T + t
                    if N_bt > self.max_masks:
                        seg_masks.append(m[:self.max_masks])
                        seg_ids.append(ids[:self.max_masks])
                    elif N_bt < self.max_masks:
                        pad_n = self.max_masks - N_bt
                        seg_masks.append(
                            torch.cat(
                                [m, m.new_zeros(pad_n, *m.shape[1:], dtype=torch.bool)],
                                dim=0,
                            )
                        )
                        seg_ids.append(
                            torch.cat(
                                [ids, ids.new_full((pad_n,), flat_id)],
                                dim=0,
                            )
                        )
                    else:
                        seg_masks.append(m)
                        seg_ids.append(ids)
                out_masks.append(torch.cat(seg_masks, dim=0))
                out_img_ids.append(torch.cat(seg_ids, dim=0))
            else: 
                raise ValueError("No Max masks setting is currently not supported")

        return {
            "num_frames": T,
            "num_videos": B,
            "masks": out_masks,         # list[T] of (N_obj, Z, Y, X)
            "img_ids": out_img_ids,     # list[T] of (N_obj,) — flat index into data_tensor
        }

    def _build_data_views_lazy(
        self,
        targets: list[dict],
        num_frames: int,
        num_videos: int,
        device: torch.device,
        mask_labelmap: torch.Tensor,  # (B, T, Z, Y, X) int32
    ) -> dict:
        """
        Lazy path: materialize per-instance binary masks on-device from the
        labelmap, using only the K=min(N_inst, max_masks) IDs we'll actually
        keep after the per-frame cap. Same output contract as the eager path.

        Sampling is per-VIDEO, not per-(video, frame): the same K IDs are
        reused across all T frames so that a given object retains a stable
        flat index across the tracking loop. Step 1 uses a deterministic
        head-slice; random sampling will be added in a follow-up.
        """
        B = num_videos
        T = num_frames
        K_full = self.max_masks

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
        for b, tgt in enumerate(targets[:B]):
            ids_b = tgt.get("mask_ids", None)
            if ids_b is None or ids_b.numel() == 0:
                continue
            K = min(int(ids_b.numel()), K_full)
            sampled_ids_per_b[b] = ids_b[:K]

        out_masks: list[torch.Tensor] = []
        out_img_ids: list[torch.Tensor] = []

        for t in range(T):
            seg_masks: list[torch.Tensor] = []
            seg_ids: list[torch.Tensor] = []
            for b in range(B):
                sampled = sampled_ids_per_b[b]
                K = sampled.numel()
                flat_id = b * T + t

                if K > 0:
                    lm_bt = mask_labelmap[b, t]  # (Z, Y, X) int32
                    # (K, Z, Y, X) bool; only K ≤ max_masks materialized.
                    m = lm_bt.unsqueeze(0) == sampled.to(lm_bt.dtype).view(K, 1, 1, 1)
                else:
                    m = torch.zeros((0, *spatial), dtype=torch.bool, device=device)

                ids = torch.full((K,), flat_id, dtype=torch.int32, device=device)

                pad_n = K_full - K
                if pad_n > 0:
                    m = torch.cat(
                        [m, torch.zeros((pad_n, *spatial), dtype=torch.bool, device=device)],
                        dim=0,
                    )
                    ids = torch.cat(
                        [ids, ids.new_full((pad_n,), flat_id)],
                        dim=0,
                    )

                seg_masks.append(m)
                seg_ids.append(ids)

            out_masks.append(torch.cat(seg_masks, dim=0))
            out_img_ids.append(torch.cat(seg_ids, dim=0))

        return {
            "num_frames": T,
            "num_videos": B,
            "masks": out_masks,
            "img_ids": out_img_ids,
        }

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #

    def forward(self, data_sample: dict, data_time: float, idx: int) -> dict:
        preprocess_t0 = time.time()

        inputs = data_sample["data_tensor"]
        meta = data_sample.get("metainfo", {})

        # Snapshot the integer labelmap BEFORE the bf16 cast below. Segment
        # IDs are stored as uint16 in data_tensor's last channel, so the
        # source view is already a clean integer tensor — no copy needed.
        # (Casting through bf16 would alias any ID > 256 on the downstream
        # `==` comparison since bf16 has only an 8-bit mantissa.)
        if self.defer_binary_masks:
            mask_labelmap = inputs[..., -1]
        else:
            mask_labelmap = None

        if inputs.dtype != self.dtype:
            inputs = inputs.to(self.dtype)

        # --- strip mask channel -------------------------------------------------
        images, _mask_labelmap = self._strip_mask_channel(inputs)

        # --- apply any configured transforms ------------------------------------
        sample = {"data_tensor": images, "metainfo": meta}
        sample, transform_time = self._apply_transforms(sample)
        images = sample["data_tensor"]
        meta = sample["metainfo"]

        # The lazy-mask path consumes `mask_labelmap` AFTER transforms run on
        # `images`. Geometric transforms (resize, crop, flip) currently only
        # touch `images`, so they would silently misalign the labelmap and
        # the materialized binary masks. Detect via spatial-shape mismatch.
        if mask_labelmap is not None and images.shape[1:-1] != mask_labelmap.shape[1:]:
            raise RuntimeError(
                f"defer_binary_masks=True is incompatible with geometric transforms: "
                f"images spatial shape {tuple(images.shape[1:-1])} != labelmap spatial shape "
                f"{tuple(mask_labelmap.shape[1:])}. Either disable geometric transforms or "
                f"apply them to the labelmap as well."
            )

        B, T = images.shape[0], images.shape[1]

        # --- flatten to (T*B, C, Z, Y, X) ---------------------
        flat_img_batch = self._build_flat_img_batch(images)

        # --- build per-frame masks & index maps ----------------------------
        targets = meta.pop("targets", [])
        data_views = self._build_data_views(
            targets=targets,
            num_frames=T,
            num_videos=B,
            device=flat_img_batch.device,
            mask_labelmap=mask_labelmap,
        )

        return {
            "data_tensor": flat_img_batch,
            "metainfo": {
                **meta,
                "targets": data_views,
                "preprocess_time": time.time() - preprocess_t0,
                "data_time": data_time,
                "transform_time": transform_time,
                "idx": idx,
            },
        }