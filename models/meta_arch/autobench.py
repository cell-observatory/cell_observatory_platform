import sys
import logging
from abc import ABC, abstractmethod
from typing import Any, Literal, Mapping, Optional

import torch
import torch.nn as nn
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

from cell_observatory_platform.data.masking.mask_generator import apply_masks
from cell_observatory_platform.models.layers.attention import RopeAttention
from cell_observatory_platform.models.layers.patch_embeddings import PatchEmbedding, calc_num_patches
from cell_observatory_platform.training.helpers import (
    get_input_data,
    get_nparams_and_flops,
    get_patch_sizes,
)
from cell_observatory_platform.training.losses import get_loss_fn

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Base class: only shared config/utility, no module instantiation
# -------------------------------------------------------------------


class AutoBench(nn.Module, ABC):
    """
    Base class for AutoBench-style finetuning models.

    Responsibilities:
      - store shared meta (input_fmt, shapes, loss_fn, etc.)
      - define interface and common utilities (get_num_patches, default predict)
    """

    def __init__(
        self,
        backbone_args: Any,
        decoder_args: Any,
        task: Literal[
            "denoising",
            "channel_split",
            "upsample_time",
            "upsample_space",
            "upsample_spacetime",
        ],
        input_fmt: str = "TZYXC",
        input_shape=(16, 128, 128, 128, 2),
        patch_shape=(4, 16, 16, 16),
        loss_fn: str = "l2_masked",
        abs_sincos_enc: bool = False,
        weight_init_type: str = "mae",
        with_auxiliary_loss: bool = False,
        freeze_backbone: bool = False,
        buffer_device: str = "cuda",
    ):
        super().__init__()
        self.backbone_args = backbone_args
        self.decoder_args = decoder_args

        self.task = task

        self.input_fmt = input_fmt
        self.input_shape = tuple(input_shape)
        self.patch_shape = tuple(patch_shape)
        self.abs_sincos_enc = abs_sincos_enc

        self.loss_fn = get_loss_fn(loss_fn)
        self.with_auxiliary_loss = with_auxiliary_loss
        self.weight_init_type = weight_init_type

        # Will be set in subclasses
        self.backbone: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None

        self.freeze_backbone = freeze_backbone

    def _freeze_backbone(self):
        """
        Freeze the backbone parameters.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    @abstractmethod
    def forward(self, data_sample: dict):
        """
        Task-specific forward that returns (loss_dict, predictions).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data_sample: dict):
        """
        Task-specific prediction (usually unpatchified outputs).
        """
        raise NotImplementedError

    def _init_model_weights(self, buffer_device: str | None = None):
        """No-op: AutoBench delegates weight initialization to each backbone
        and decoder module. Each module built via its Hydra BUILD function
        handles its own weight initialization in __init__.

        If you add a new backbone or decoder for use with AutoBench,
        ensure it calls its own weight init at the end of __init__.
        See, for example, the following self-initializing modules:
          - MaskedEncoder, MaskedHieraEncoder (backbones)
          - MaskedPredictor, MaskedHieraPredictor (decoders)
          - LinearHead, LinearProbe (decoders)
        """
        pass

    def get_param_groups(self, weight_decay: float, **kwargs) -> list[dict]:
        """
        Backbone/decoder split with decay/no-decay within each.
        """
        NO_WD_KEYWORDS = ("bias", "pos_embedding", "cls_token", "token_param", "level_embed")

        def _split(module: nn.Module) -> list[dict]:
            decay, no_decay = [], []
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim == 1 or any(kw in name for kw in NO_WD_KEYWORDS):
                    no_decay.append(p)
                else:
                    decay.append(p)
            groups = []
            if decay:
                groups.append({"params": decay, "weight_decay": weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "weight_decay": 0.0})
            return groups

        groups = []
        if self.backbone is not None:
            groups.extend(_split(self.backbone))
        if self.decoder is not None:
            groups.extend(_split(self.decoder))
        return groups

    # TODO: implement for each meta_arch 
    # @torch.jit.ignore
    # def _get_nparams_and_flops(
    #     self, batch_size: int, device: Literal["cuda", "meta"] = "cuda", masking_ratio: float = 0.0
    # ):
    #     # FIXME: this may be inaccurate when we start working on
    #     #        temporal masking related tasks
    #     if device == "cuda":
    #         # TODO: test this path more thoroughly
    #         with torch.cuda.device(device):
    #             input_shape = (batch_size, *self.input_shape)
    #             data_sample = get_input_data(
    #                 inputs=input_shape,
    #                 device="cuda",
    #             )
    #             seq_len = int(self.get_num_patches()) * (1 - masking_ratio)
    #             model_summary = get_nparams_and_flops(self, data_sample, seq_len)
    #             model_param_count, num_flops_per_token = (
    #                 model_summary["total_params"],
    #                 model_summary["training_flops"],
    #             )
    #     elif device == "meta":
    #         print(f"Warning: using 'meta' device for flops/nparams calculation is not yet supported.")
    #         return -1, -1
    #     else:
    #         # TODO: add support for meta device calculation for other backends
    #         raise ValueError(f"Unsupported device for flops/nparams calculation: {device}")

    #     return model_param_count, num_flops_per_token

    @torch.jit.ignore
    def get_num_patches(self) -> int:
        """
        Get number of patches based on input/patch shapes:
          - if abs_sincos_enc, use precomputed num_patches on backbone.pos_embedding
          - otherwise, recompute via calc_num_patches.
        """
        if self.abs_sincos_enc and hasattr(self.backbone, "pos_embedding"):
            return self.backbone.pos_embedding.num_patches

        num_patches, _ = calc_num_patches(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            patch_shape=self.patch_shape,
        )
        return num_patches

    @torch.jit.ignore
    def forward_features(self, data_tensor: torch.Tensor):
        """
        Convenience feature extractor. Assumes backbone returns (features, patches).
        """
        if self.backbone is None:
            raise RuntimeError("Backbone is not initialized.")
        x, patches = self.backbone(data_tensor)
        return x


# -------------------------------------------------------------------
# Task-specific subclasses
# -------------------------------------------------------------------

class DenoisingAutoBench(AutoBench):
    """
    Task: denoising
    - backbone: BUILD-backbone(backbone_args) -> (x, patches)
    - decoder:  BUILD-decoder(decoder_args) -> x
    - loss: uses targets, with num_patches from encoder
    """

    def __init__(self, *, backbone_args: Any, decoder_args: Any, **kwargs):
        super().__init__(
            backbone_args=backbone_args,
            decoder_args=decoder_args,
            task="denoising",
            **kwargs,
        )
        build_backbone = get_method(backbone_args.BUILD)
        build_decoder = get_method(decoder_args.BUILD)
        self.backbone = build_backbone(backbone_args)
        self.decoder = build_decoder(decoder_args)

    def forward(self, data_sample: dict):
        inputs, meta = data_sample["data_tensor"], data_sample["metainfo"]
        targets = meta.get("targets", [None])[0]

        x, patches = self.backbone(inputs)
        x = self.decoder(x)

        predictions = x
        
        if self.with_auxiliary_loss:
            aux_loss_meta = {
                "targets": targets,
                "predictions": predictions,
            }
        else:
            aux_loss_meta = None

        loss, aux_losses = self.loss_fn(
            predictions=predictions,
            targets=targets,
            num_patches=self.get_num_patches(),
            aux_loss_meta=aux_loss_meta,
        )
        loss_dict = {"step_loss": loss, **(aux_losses or {})}
        return loss_dict, predictions

    def predict(self, data_sample: dict):
        inputs = data_sample["data_tensor"]
        x, patches = self.backbone(inputs)
        x = self.decoder(x)

        # TODO: make this more general to support models which don't use patch_embedding._unpatchify
        # Assume backbone exposes patch_embedding._unpatchify 
        return self.backbone.patch_embedding._unpatchify(x, out_channels=None)

class ChannelSplitAutoBench(AutoBench):
    """
    Task: channel_split
    - backbone: BUILD-backbone(backbone_args) -> (x, patches)
    - decoder:  BUILD-decoder(decoder_args) -> x
    - loss: uses targets, with num_patches from encoder
    """

    def __init__(self, *, backbone_args: Any, decoder_args: Any, **kwargs):
        super().__init__(
            backbone_args=backbone_args,
            decoder_args=decoder_args,
            task="channel_split",
            **kwargs,
        )
        build_backbone = get_method(backbone_args.BUILD)
        build_decoder = get_method(decoder_args.BUILD)
        self.backbone = build_backbone(backbone_args)
        self.decoder = build_decoder(decoder_args)

        if self.input_fmt[-1] != "C":
            raise ValueError(f"ChannelSplitAutoBench expects input_fmt to end with 'C', got {self.input_fmt}")
        self.output_channels = self.input_shape[-1]

        if self.freeze_backbone:
            self._freeze_backbone()

    def forward(self, data_sample: dict):
        inputs, meta = data_sample["data_tensor"], data_sample["metainfo"]
        targets = meta.get("targets", [None])[0]
        target_masks = meta.get("target_masks", [None])[0]
        patches_used = meta.get("patches_used", [None])[0]

        x, patches = self.backbone(inputs)
        x = self.decoder(x)

        predictions = x

        if self.with_auxiliary_loss:
            aux_loss_meta = {
                "targets": targets,
                "predictions": predictions,
            }
        else:
            aux_loss_meta = None

        loss, aux_losses = self.loss_fn(predictions, targets, num_patches=self.get_num_patches(), aux_loss_meta=aux_loss_meta)

        loss_dict = {"step_loss": loss, **(aux_losses or {})}
        return loss_dict, predictions

    def predict(self, data_sample: dict):
        inputs, meta = data_sample["data_tensor"], data_sample["metainfo"]

        x, patches = self.backbone(inputs)
        x = self.decoder(x)

        # Assume backbone exposes patch_embedding.unpatchify (MaskedEncoder-style)
        return self.backbone.patch_embedding._unpatchify(
            x,
            out_channels=self.output_channels if self.output_channels is not None else None,
        )


class UpsampleTimeAutoBench(AutoBench):
    """
    Task: upsample_time
    - uses masks/context_masks/target_masks/original_patch_indices
    - loss only over masked timepoints
    """

    def __init__(self, *, backbone_args: Any, decoder_args: Any, **kwargs):
        super().__init__(
            backbone_args=backbone_args,
            decoder_args=decoder_args,
            task="upsample_time",
            **kwargs,
        )

        build_backbone = get_method(backbone_args.BUILD)
        build_decoder = get_method(decoder_args.BUILD)
        self.backbone = build_backbone(backbone_args)
        self.decoder = build_decoder(decoder_args)

        if self.freeze_backbone:
            self._freeze_backbone()

    def forward(self, data_sample: dict):
        inputs, meta = data_sample["data_tensor"], data_sample["metainfo"]
        masks = meta.get("masks", [None])[0]
        context_masks = meta.get("context_masks", [None])[0]
        target_masks = meta.get("target_masks", [None])[0]
        original_patch_indices = meta.get("original_patch_indices", [None])[0]

        x, patches = self.backbone(inputs, masks=context_masks)

        x = self.decoder(
            x,
            original_patch_indices=original_patch_indices,
            target_masks=target_masks,
        )

        # only supervise the masked timepoints
        targets = apply_masks(patches, masks=target_masks)
        predictions = apply_masks(x, masks=target_masks)

        loss, aux_losses = self.loss_fn(predictions, targets, num_patches=self.get_num_patches())
        loss_dict = {"step_loss": loss, **(aux_losses or {})}

        return loss_dict, predictions

    def predict(self, data_sample: dict):
        inputs, meta = data_sample["data_tensor"], data_sample["metainfo"]
        context_masks = meta.get("context_masks", [None])[0]
        target_masks = meta.get("target_masks", [None])[0]
        original_patch_indices = meta.get("original_patch_indices", [None])[0]

        x, patches = self.backbone(inputs, masks=context_masks)

        x = self.decoder(
            x,
            original_patch_indices=original_patch_indices,
            target_masks=target_masks,
        )

        return self.backbone.patch_embedding._unpatchify(x, out_channels=None)


class UpsampleSpaceAutoBench(AutoBench):
    """
    Task: upsample_space
    - no temporal masking, just reconstructs spatially upsampled targets
    """

    def __init__(self, *, backbone_args: Any, decoder_args: Any, **kwargs):
        super().__init__(
            backbone_args=backbone_args,
            decoder_args=decoder_args,
            task="upsample_space",
            **kwargs,
        )

        build_backbone = get_method(backbone_args.BUILD)
        build_decoder = get_method(decoder_args.BUILD)
        self.backbone = build_backbone(backbone_args)
        self.decoder = build_decoder(decoder_args)

        if self.freeze_backbone:
            self._freeze_backbone()

    def forward(self, data_sample: dict):
        inputs, meta = data_sample["data_tensor"], data_sample["metainfo"]
        targets = meta.get("targets", [None])[0]
        target_masks = meta.get("target_masks", [None])[0]
        patches_used = meta.get("patches_used", [None])[0]

        x, patches = self.backbone(inputs)
        x = self.decoder(x)

        predictions = x
        if self.with_auxiliary_loss:
            aux_loss_meta = {
                "targets": targets,
                "predictions": predictions,
            }
        else:
            aux_loss_meta = None

        loss, aux_losses = self.loss_fn(predictions, targets, num_patches=self.get_num_patches(), aux_loss_meta=aux_loss_meta)

        loss_dict = {"step_loss": loss, **(aux_losses or {})}
        return loss_dict, predictions

    def predict(self, data_sample: dict):
        inputs, meta = data_sample["data_tensor"], data_sample["metainfo"]

        x, patches = self.backbone(inputs)
        x = self.decoder(x)

        return self.backbone.patch_embedding._unpatchify(x, out_channels=None)


class UpsampleSpaceTimeAutoBench(AutoBench):
    """
    Task: upsample_spacetime
    - uses context masks + indices like upsample_time
    - but loss is computed on full targets
    """

    def __init__(self, *, backbone_args: Any, decoder_args: Any, **kwargs):
        super().__init__(
            backbone_args=backbone_args,
            decoder_args=decoder_args,
            task="upsample_spacetime",
            **kwargs,
        )

        build_backbone = get_method(backbone_args.BUILD)
        build_decoder = get_method(decoder_args.BUILD)
        self.backbone = build_backbone(backbone_args)
        self.decoder = build_decoder(decoder_args)

        if self.freeze_backbone:
            self._freeze_backbone()

    def forward(self, data_sample: dict):
        inputs, meta = data_sample["data_tensor"], data_sample["metainfo"]
        targets = meta.get("targets", [None])[0]
        context_masks = meta.get("context_masks", [None])[0]
        target_masks = meta.get("target_masks", [None])[0]
        original_patch_indices = meta.get("original_patch_indices", [None])[0]

        # encoder sees masks/context
        x, patches = self.backbone(inputs, masks=context_masks)
        x = self.decoder(
            x,
            original_patch_indices=original_patch_indices,
            target_masks=target_masks,
        )

        predictions = x

        loss, aux_losses = self.loss_fn(x, targets, num_patches=self.get_num_patches())
        loss_dict = {"step_loss": loss, **(aux_losses or {})}

        return loss_dict, predictions

    def predict(self, data_sample: dict):
        inputs, meta = data_sample["data_tensor"], data_sample["metainfo"]
        context_masks = meta.get("context_masks", [None])[0]
        target_masks = meta.get("target_masks", [None])[0]
        original_patch_indices = meta.get("original_patch_indices", [None])[0]

        x, patches = self.backbone(inputs, masks=context_masks)
        x = self.decoder(
            x,
            original_patch_indices=original_patch_indices,
            target_masks=target_masks,
        )

        return self.backbone.patch_embedding._unpatchify(x, out_channels=None)


# -------------------------------------------------------------------
# BUILD entrypoint for Hydra / training script
# -------------------------------------------------------------------


def BUILD(cfg: Mapping[str, Any]) -> AutoBench:
    """
    Dispatcher that picks the appropriate AutoBench subclass
    based on cfg.task and wires backbone/decoder BUILD functions.
    """
    task = cfg["tasks"]["task"]
    if task == "denoising":
        model_cfg = cfg.models.meta_arch.autobench.DenoisingAutoBench
    elif task == "channel_split":
        model_cfg = cfg.models.meta_arch.autobench.ChannelSplitAutoBench
    elif task == "upsample_time":
        model_cfg = cfg.models.meta_arch.autobench.UpsampleTimeAutoBench
    elif task == "upsample_space":
        model_cfg = cfg.models.meta_arch.autobench.UpsampleSpaceAutoBench
    elif task == "upsample_spacetime":
        model_cfg = cfg.models.meta_arch.autobench.UpsampleSpaceTimeAutoBench
    else:
        raise ValueError(f"Unknown AutoBench task: {task}")

    backbone_args = model_cfg["backbone_args"]
    decoder_args = model_cfg["decoder_args"]

    embed_dim = model_cfg.get("embed_dim", backbone_args.get("embed_dim", None))

    if model_cfg["input_fmt"] == "ZYXC":
        temporal_patch_size, axial_patch_size, lateral_patch_size = get_patch_sizes(
            input_format=model_cfg["input_fmt"],
            patch_shape=model_cfg["patch_shape"],
        )
        output_dim = PatchEmbedding.compute_num_pixels_per_patch(
            channels=model_cfg["train_shape"][-1],
            temporal_patch_size=temporal_patch_size,
            axial_patch_size=axial_patch_size,
            lateral_patch_size=lateral_patch_size,
            input_format=model_cfg["input_fmt"],
        )
    else:
        raise ValueError(f"AutoBench currently only supports 'ZYXC' input_fmt, got {model_cfg['input_fmt']}")

    if embed_dim is None:
        raise ValueError(
            "Either model_cfg.embed_dim or backbone_args.embed_dim must be set " "to derive decoder_args.input_dim"
        )

    # Single simple contract: decoders get `input_dim`
    # and decide themselves how to map it to their ctor args.
    if isinstance(decoder_args, DictConfig):
        prev_struct = OmegaConf.is_struct(decoder_args)
        if prev_struct:
            OmegaConf.set_struct(decoder_args, False)

        decoder_args["input_dim"] = embed_dim
        decoder_args["output_dim"] = output_dim

        if prev_struct:
            OmegaConf.set_struct(decoder_args, True)
    else:
        decoder_args["input_dim"] = embed_dim

    common_kwargs = dict(
        backbone_args=backbone_args,
        decoder_args=decoder_args,
        input_fmt=model_cfg.get("input_fmt"),
        input_shape=tuple(model_cfg.get("input_shape")),
        patch_shape=tuple(model_cfg.get("patch_shape")),
        loss_fn=model_cfg.get("loss_fn"),
        abs_sincos_enc=model_cfg.get("abs_sincos_enc"),
        weight_init_type=model_cfg.get("weight_init_type"),
        with_auxiliary_loss=model_cfg.get("with_auxiliary_loss", False),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    )

    if task == "denoising":
        return DenoisingAutoBench(**common_kwargs)
    elif task == "channel_split":
        return ChannelSplitAutoBench(**common_kwargs)
    elif task == "upsample_time":
        return UpsampleTimeAutoBench(**common_kwargs)
    elif task == "upsample_space":
        return UpsampleSpaceAutoBench(**common_kwargs)
    elif task == "upsample_spacetime":
        return UpsampleSpaceTimeAutoBench(**common_kwargs)
    else:
        raise ValueError(f"Unknown AutoBench task: {task}")
