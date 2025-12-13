import inspect
import logging
import sys
from typing import Any, Dict, Literal, Mapping, Tuple, Union

import torch
import torch.nn as nn

from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_platform.data.masking.mask_generator import apply_masks
from cell_observatory_platform.models.backbones.maskedencoder import MaskedEncoder
from cell_observatory_platform.models.heads.maskedpredictor import MaskedPredictor
from cell_observatory_platform.models.layers.activation import get_activation
from cell_observatory_platform.models.layers.attention import RopeAttention
from cell_observatory_platform.models.layers.mlp import get_mlp
from cell_observatory_platform.models.layers.norm import get_norm
from cell_observatory_platform.models.layers.patch_embeddings import calc_num_patches
from cell_observatory_platform.training.helpers import get_masked_input_data, get_nparams_and_flops, init_weights
from cell_observatory_platform.training.losses import get_loss_fn

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


CONFIGS = {
    "mae-tiny": {
        "embed_dim": 192,
        "decoder_embed_dim": 96,
        "depth": 12,
        "decoder_depth": 3,
        "num_heads": 3,
        "decoder_num_heads": 3,
        "mlp_ratio": 4,
    },
    "mae-small": {
        "embed_dim": 384,
        "decoder_embed_dim": 192,
        "depth": 12,
        "decoder_depth": 6,
        "num_heads": 6,
        "decoder_num_heads": 6,
        "mlp_ratio": 4,
    },
    "mae-base": {
        "embed_dim": 768,
        "decoder_embed_dim": 256,
        "depth": 12,
        "decoder_depth": 8,
        "num_heads": 12,
        "decoder_num_heads": 8,
        "mlp_ratio": 4,
    },
    "mae-large": {
        "embed_dim": 1024,
        "decoder_embed_dim": 512,
        "depth": 24,
        "decoder_depth": 8,
        "num_heads": 16,
        "decoder_num_heads": 8,
        "mlp_ratio": 4,
    },
    "mae-huge": {
        "embed_dim": 1280,
        "decoder_embed_dim": 512,
        "depth": 32,
        "decoder_depth": 8,
        "num_heads": 16,
        "decoder_num_heads": 8,
        "mlp_ratio": 4,
    },
    "mae-2billion": {
        "embed_dim": 2560,
        "decoder_embed_dim": 512,
        "depth": 24,
        "decoder_depth": 8,
        "num_heads": 32,
        "decoder_num_heads": 8,
        "mlp_ratio": 4,
    },
    "mae-6billion": {
        "embed_dim": 4096,
        "decoder_embed_dim": 512,
        "depth": 32,
        "decoder_depth": 8,
        "num_heads": 32,
        "decoder_num_heads": 8,
        "mlp_ratio": 4,
    },
    "mae-giant": {
        "embed_dim": 1408,
        "decoder_embed_dim": 512,
        "depth": 40,
        "decoder_depth": 8,
        "num_heads": 16,
        "decoder_num_heads": 8,
        "mlp_ratio": 48 / 11,
    },
    "mae-gigantic": {
        "embed_dim": 1664,
        "decoder_embed_dim": 1024,
        "depth": 48,
        "decoder_depth": 16,
        "num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 64 / 13,
    },
    "mae-enormous": {
        "embed_dim": 1792,
        "decoder_embed_dim": 1024,
        "depth": 56,
        "decoder_depth": 16,
        "num_heads": 16,
        "decoder_num_heads": 16,
        "mlp_ratio": 8.5714285714,
    },
}


class MaskedAutoEncoder(nn.Module):
    def __init__(
        self,
        model_template: Literal[
            "mae",  # custom use `embed_dim`, `decoder_embed_dim`, `depth`, `num_heads` and `mlp_ratio` to config model
            "mae-tiny",
            "mae-small",
            "mae-base",
            "mae-large",
            "mae-huge",
            "mae-giant",
            "mae-gigantic",
        ] = "mae",
        input_fmt="TZYXC",
        input_shape: tuple = (16, 128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        embed_dim=768,
        decoder_embed_dim=256,
        depth=12,
        decoder_depth=8,
        num_heads=12,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        proj_drop_rate=0.0,
        att_drop_rate=0.0,
        drop_path_rate=0.1,
        init_std=0.02,
        fixed_dropout_depth=False,
        norm_layer: Union[nn.Module, Literal["RmsNorm", "LayerNorm", "SyncBatchNorm", "GroupNorm"]] = "RmsNorm",
        act_layer: Union[nn.Module, Literal["GELU", "SiLU", "LeakyReLU", "GLU", "Sigmoid", "Tanh"]] = "SiLU",
        mlp_layer: Union[nn.Module, Literal["Mlp", "SwiGLU"]] = "SwiGLU",
        abs_sincos_enc: bool = False,
        rope_pos_enc: bool = True,
        rope_random_rotation_per_head: bool = True,
        rope_mixed: bool = True,
        rope_theta: float = 10.0,
        weight_init_type: str = "mae",
        mlp_wide_silu: bool = False,
        loss_fn: str = "l2_masked",
        with_auxiliary_loss: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super().__init__()

        if model_template in CONFIGS.keys():
            config = CONFIGS[model_template]
            self.depth = config["depth"]
            self.decoder_depth = config["decoder_depth"]
            self.embed_dim = config["embed_dim"]
            self.decoder_embed_dim = config["decoder_embed_dim"]
            self.num_heads = config["num_heads"]
            self.decoder_num_heads = config["decoder_num_heads"]
            self.mlp_ratio = config["mlp_ratio"]
        else:
            self.depth = depth
            self.decoder_depth = decoder_depth
            self.embed_dim = embed_dim
            self.decoder_embed_dim = decoder_embed_dim
            self.num_heads = num_heads
            self.decoder_num_heads = decoder_num_heads
            self.mlp_ratio = mlp_ratio

        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        self.input_fmt = input_fmt
        self.input_shape = input_shape

        axis_to_value = dict(zip(input_fmt, input_shape))
        self.in_chans = axis_to_value["C"]
        self.num_frames = axis_to_value.get("T", None)

        self.patch_shape = patch_shape

        self.proj_drop_rate = proj_drop_rate
        self.att_drop_rate = att_drop_rate
        self.drop_path_rate = drop_path_rate
        self.fixed_dropout_depth = fixed_dropout_depth

        self.init_std = init_std

        self.norm_layer = get_norm(norm_layer)
        self.act_layer = get_activation(act_layer)
        self.mlp_layer = get_mlp(mlp_layer)

        # positional encoding parameters
        self.abs_sincos_enc = abs_sincos_enc
        self.rope_pos_enc = rope_pos_enc
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        self.wide_silu = mlp_wide_silu
        self.rope_random_rotation_per_head = rope_random_rotation_per_head

        self.masked_encoder = MaskedEncoder(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            patch_shape=self.patch_shape,
            channels=self.in_chans,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            proj_drop_rate=self.proj_drop_rate,
            att_drop_rate=self.att_drop_rate,
            drop_path_rate=self.drop_path_rate,
            fixed_dropout_depth=self.fixed_dropout_depth,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
            mlp_layer=self.mlp_layer,
            init_std=self.init_std,
            abs_sincos_enc=self.abs_sincos_enc,
            rope_pos_enc=self.rope_pos_enc,
            rope_random_rotation_per_head=self.rope_random_rotation_per_head,
            rope_mixed=self.rope_mixed,
            rope_theta=self.rope_theta,
            mlp_wide_silu=mlp_wide_silu,
            dtype=self.dtype,
        )

        self.masked_decoder = MaskedPredictor(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            patch_shape=self.patch_shape,
            channels=self.in_chans,
            input_embed_dim=self.embed_dim,
            output_embed_dim=self.masked_encoder.patch_embedding.pixels_per_patch,
            embed_dim=self.decoder_embed_dim,
            depth=self.decoder_depth,
            num_heads=self.decoder_num_heads,
            mlp_ratio=self.mlp_ratio,
            proj_drop_rate=self.proj_drop_rate,
            att_drop_rate=self.att_drop_rate,
            drop_path_rate=self.drop_path_rate,
            fixed_dropout_depth=self.fixed_dropout_depth,
            norm_layer=self.norm_layer,
            act_layer=self.act_layer,
            mlp_layer=self.mlp_layer,
            init_std=self.init_std,
            abs_sincos_enc=self.abs_sincos_enc,
            rope_pos_enc=self.rope_pos_enc,
            rope_random_rotation_per_head=self.rope_random_rotation_per_head,
            rope_mixed=self.rope_mixed,
            rope_theta=self.rope_theta,
            mlp_wide_silu=mlp_wide_silu,
            dtype=self.dtype,
        )

        self.weight_init_type = weight_init_type

        self.loss_fn = get_loss_fn(loss_fn)
        self.with_auxiliary_loss = with_auxiliary_loss

    def init_model_weights(self, buffer_device: str | None = None):
        # TODO: move model inits back into each model class
        init_weights(self, weight_init_type=self.weight_init_type)
        for mod in self.modules():
            if isinstance(mod, RopeAttention):
                mod.init_rope_parameters(device=buffer_device)

    @torch.jit.ignore
    def _get_nparams_and_flops(
        self, batch_size: int, device: Literal["cuda", "meta"] = "cuda", masking_ratio: float = 0.0
    ):
        if device == "cuda":
            # TODO: test this path more thoroughly
            with torch.cuda.device(device):
                input_shape = (batch_size, *self.input_shape)
                data_sample = get_masked_input_data(
                    self,
                    inputs=input_shape,
                    device="cuda",
                    mask_ratio=masking_ratio,
                )
                seq_len = int(self.get_num_patches()) * (1 - masking_ratio)
                model_summary = get_nparams_and_flops(self, data_sample, seq_len)
                model_param_count, num_flops_per_token = (
                    model_summary["total_params"],
                    model_summary["training_flops"],
                )
        elif device == "meta":
            print(f"Warning: using 'meta' device for flops/nparams calculation is not yet supported.")
            return -1, -1
        else:
            # TODO: add support for meta device calculation for other backends
            raise ValueError(f"Unsupported device for flops/nparams calculation: {device}")
        return model_param_count, num_flops_per_token

    @torch.jit.ignore
    def get_patch_embedding(self):
        return self.masked_encoder.patch_embedding

    @torch.jit.ignore
    def get_encoder(self):
        return self.masked_encoder

    @torch.jit.ignore
    def get_decoder(self):
        return self.masked_decoder

    @torch.jit.ignore
    def get_num_patches(self):
        if self.abs_sincos_enc:
            return self.masked_encoder.pos_embedding.num_patches
        else:
            num_patches, _ = calc_num_patches(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                patch_shape=self.patch_shape,
            )
            return num_patches

    def forward(self, data_sample: dict):
        inputs, meta = data_sample["data_tensor"], data_sample["metainfo"]
        masks, context_masks, patches_used = meta["masks"][0], meta["context_masks"][0], meta["patches_used"][0]
        target_masks, original_patch_indices = meta["target_masks"][0], meta["original_patch_indices"][0]

        x, patches = self.masked_encoder(inputs, masks=context_masks)
        x = self.masked_decoder(
            x, original_patch_indices=original_patch_indices, target_masks=target_masks, patches_used=patches_used
        )

        # compute loss over masked patches (re-index if blocked masking removed some patches)
        if patches_used is not None:
            target_idx_in_patches_used = torch.searchsorted(patches_used, target_masks)
        else:
            target_idx_in_patches_used = target_masks
        targets = apply_masks(patches, masks=target_masks)
        predictions = apply_masks(x, masks=target_idx_in_patches_used)

        if self.with_auxiliary_loss:
            aux_loss_meta = {
                "targets": patches,
                "predictions": x,
                "patches_used": patches_used,
                "target_masks": target_masks,
                "prediction_masks": target_idx_in_patches_used,
            }
        else:
            aux_loss_meta = None

        loss, aux_losses = self.loss_fn(targets, predictions, masks.sum(), aux_loss_meta)

        loss_dict = {"step_loss": loss, **(aux_losses or {})}
        return loss_dict, predictions

    def predict(self, data_sample: dict):
        inputs, meta = data_sample["data_tensor"], data_sample["metainfo"]
        masks, context_masks, patches_used = meta["masks"][0], meta["context_masks"][0], meta["patches_used"][0]
        target_masks, original_patch_indices = meta["target_masks"][0], meta["original_patch_indices"][0]

        x, patches = self.masked_encoder(inputs, masks=context_masks)
        x = self.masked_decoder(
            x, original_patch_indices=original_patch_indices, target_masks=target_masks, patches_used=patches_used
        )

        predictions = self.masked_encoder.patch_embedding._unpatchify(x, out_channels=None)
        return predictions


def _extract_model_kwargs(cfg: Mapping[str, Any]) -> dict:
    sig = inspect.signature(MaskedAutoEncoder.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    ignore = {"_target_", "BUILD"}
    kwargs = {}
    for k in cfg.keys():
        if k in ignore or k not in allowed:
            continue
        kwargs[k] = cfg[k]
    return kwargs


def BUILD(cfg: Mapping[str, Any]) -> MaskedAutoEncoder:
    model_cfg = cfg.models.meta_arch.mae
    return MaskedAutoEncoder(**_extract_model_kwargs(model_cfg))
