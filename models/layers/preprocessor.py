import functools
import os
import time
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
from hydra.utils import get_method, instantiate
from omegaconf import DictConfig

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
        self.num_patches, _ = calc_num_patches(
            input_fmt=self.input_format,
            input_shape=self.input_shape,
            patch_shape=self.patch_shape,
        )

        # TODO: once we start supporting variable input shapes,
        #       update this helper to use input_shape from data samples
        #       and call calc_num_patches() from PatchEmbedding to get num_patches
        if (
            self.with_masking
            and self.mask_generator is not None
            and hasattr(self.mask_generator, "random_masking_ratio")
        ):
            self.masking_ratio = self.mask_generator.random_masking_ratio
        else:
            self.masking_ratio = 0.0
        self.seq_len = self._calculate_seq_len()

    def _calculate_seq_len(self):
        masking_ratio = self.masking_ratio if self.with_masking else 0.0
        seq_len = int(self.num_patches * (1 - masking_ratio))
        return seq_len

    def forward(self, data_sample: dict, data_time: float) -> dict:
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

        tokens_per_batch = inputs.shape[0] * self.seq_len

        if self.with_masking:
            masking_time = time.time()
            masks, context_masks, target_masks, original_patch_indices, channels_to_mask, patches_used = (
                self.mask_generator(inputs.shape[0])
            )
            masking_time = time.time() - masking_time

            return {
                "data_tensor": inputs,
                "metainfo": {
                    "masks": [masks] if self.with_masking else None,
                    "context_masks": [context_masks] if self.with_masking else None,
                    "target_masks": [target_masks] if self.with_masking else None,
                    "original_patch_indices": [original_patch_indices] if self.with_masking else None,
                    "channels_to_mask": [channels_to_mask] if self.with_masking else None,
                    "patches_used": [patches_used] if self.with_masking else None,
                    "preprocess_time": time.time() - preprocess_time,
                    "data_time": data_time,
                    "masking_time": masking_time,
                    "transform_time": transform_time if self.transforms is not None else -1,
                    "tokens_per_batch": tokens_per_batch,
                    **meta,
                },
            }
        else:
            return {
                "data_tensor": inputs,
                "metainfo": {
                    "preprocess_time": time.time() - preprocess_time,
                    "data_time": data_time,
                    "masking_time": -1.0,
                    "transform_time": transform_time if self.transforms is not None else -1,
                    "tokens_per_batch": tokens_per_batch,
                    **meta,
                },
            }


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
    ) -> dict:
        """Attach masking info and timing, returning the standard dict."""
        tokens_per_batch = inputs.shape[0] * self.seq_len

        if self.with_masking:
            mt0 = time.time()
            B = inputs.shape[0]
            (
                masks,
                context_masks,
                target_masks,
                original_patch_indices,
                channels_to_mask,
                patches_used,
            ) = self.mask_generator(B)
            masking_time = time.time() - mt0

            mask_lists = {}
            for name, mask in zip(
                [
                    "masks",
                    "context_masks",
                    "target_masks",
                    "original_patch_indices",
                    "channels_to_mask",
                ],
                [
                    masks,
                    context_masks,
                    target_masks,
                    original_patch_indices,
                    channels_to_mask,
                ],
            ):
                if mask is not None:
                    mask_lists[name] = [mask]

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
                    "tokens_per_batch": tokens_per_batch,
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
                    "tokens_per_batch": tokens_per_batch,
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

    def forward(self, data_sample: dict, data_time: float) -> dict:
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

    def forward(self, data_sample: dict, data_time: float) -> dict:
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

    def forward(self, data_sample: dict, data_time: float) -> dict:
        inputs, meta, preprocess_t0, data_time_value = self._common_pre(
            data_sample=data_sample,
            data_time=data_time,
        )

        inputs, transform_time = self._apply_transforms(data=inputs)

        if self.mode in ("upsample_space", "upsample_spacetime"):
            # targets are HR patches
            targets = self.pe_patchify(inputs, channels=self.channels)

            # pick one NA mask and downsample
            idx = torch.randint(
                low=0,
                high=self.na_masks.shape[0],
                size=(1,),
                generator=self.rng,
            ).item()
            na_mask = resize_mask(
                self.na_masks[idx],
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
        )


# --------------------------------------------------------------------------- #
# Instance Segmentation task
# --------------------------------------------------------------------------- #


class InstanceSegmentationPreprocessor(BaseFinetunePreprocessor):
    """
    Task: instance segmentation

    Assumes upstream FinetuneCollatorActor has already:
      - split off the mask channel (instance IDs),
      - built binary masks and 3D bboxes from mask_bbox_dict,
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

    def forward(self, data_sample: dict, data_time: float) -> dict:
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
      - split off the mask channel (instance IDs),
      - built binary masks and 3D bboxes from mask_bbox_dict,
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

    def forward(self, data_sample: dict, data_time: float) -> dict:
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

    def forward(self, data_sample: dict, data_time: float) -> dict:
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
