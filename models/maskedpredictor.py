import inspect
import logging
import sys
from typing import Any, Literal, Mapping, Union

import torch
import torch.nn as nn

from cell_observatory_platform.models.activation import get_activation
from cell_observatory_platform.models.encoder import Encoder
from cell_observatory_platform.models.mlp import get_mlp
from cell_observatory_platform.models.norm import get_norm
from cell_observatory_platform.models.patch_embeddings import calc_num_patches
from cell_observatory_platform.models.positional_encoding import PosEmbedding

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


CONFIGS = {
    "mp-tiny": {
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_ratio": 4,
    },
    "mp-small": {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4,
    },
    "mp-base": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
    },
    "mp-large": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4,
    },
    "mp-huge": {
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "mlp_ratio": 4,
    },
    "mp-2billion": {
        "embed_dim": 2560,
        "depth": 24,
        "num_heads": 32,
        "mlp_ratio": 4,
    },
    "mp-6billion": {
        "embed_dim": 4096,
        "depth": 32,
        "num_heads": 32,
        "mlp_ratio": 4,
    },
    "mp-giant": {
        "embed_dim": 1408,
        "depth": 40,
        "num_heads": 16,
        "mlp_ratio": 48 / 11,
    },
    "mp-gigantic": {
        "embed_dim": 1664,
        "depth": 48,
        "num_heads": 16,
        "mlp_ratio": 64 / 13,
    },
    "mp-enormous": {
        "embed_dim": 1792,
        "depth": 56,
        "num_heads": 16,
        "mlp_ratio": 8.5714285714,
    },
}


class MaskedPredictor(nn.Module):
    def __init__(
        self,
        model_template: Literal[
            "mp",  # custom use `embed_dim`, `depth`, `num_heads` and `mlp_ratio` to config model
            "mp-tiny",
            "mp-small",
            "mp-base",
            "mp-large",
            "mp-huge",
            "mp-giant",
            "mp-gigantic",
        ] = "mp",
        input_fmt="TZYXC",
        input_shape: tuple = (16, 128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        input_embed_dim=768,
        output_embed_dim=768,
        embed_dim=384,
        depth=12,
        num_heads=12,
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
        mlp_wide_silu: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super().__init__()

        if model_template in CONFIGS.keys():
            config = CONFIGS[model_template]
            self.depth = config["depth"]
            self.embed_dim = config["embed_dim"]
            self.num_heads = config["num_heads"]
            self.mlp_ratio = config["mlp_ratio"]
        else:
            self.depth = depth
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.mlp_ratio = mlp_ratio

        self.input_embed_dim = input_embed_dim
        self.output_embed_dim = output_embed_dim

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

        self.norm = self.norm_layer(self.embed_dim) if norm_layer is not None else nn.Identity()

        self.token_param = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.patch_projection = nn.Linear(self.input_embed_dim, self.embed_dim, bias=True)

        self.output_projection = nn.Linear(self.embed_dim, self.output_embed_dim, bias=True)

        # positional encoding parameters
        self.abs_sincos_enc = abs_sincos_enc
        self.rope_pos_enc = rope_pos_enc
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        self.wide_silu = mlp_wide_silu
        self.rope_random_rotation_per_head = rope_random_rotation_per_head

        if self.abs_sincos_enc:
            self.pos_embedding = PosEmbedding(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                patch_shape=self.patch_shape,
                embed_dim=self.embed_dim,
            )

        self.encoder = Encoder(
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
            rope_pos_enc=rope_pos_enc,
            rope_random_rotation_per_head=rope_random_rotation_per_head,
            rope_mixed=rope_mixed,
            rope_theta=rope_theta,
            input_fmt=input_fmt,
            input_shape=input_shape,
            patch_shape=self.patch_shape,
            mlp_wide_silu=mlp_wide_silu,
            dtype=dtype,
        )

    @torch.jit.ignore
    def get_num_layers(self):
        return self.encoder.get_num_layers()

    @torch.jit.ignore
    def get_encoder(self):
        return self.encoder

    @torch.jit.ignore
    def get_num_patches(self):
        if self.abs_sincos_enc:
            return self.pos_embedding.num_patches
        else:
            num_patches, _ = calc_num_patches(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                patch_shape=self.patch_shape,
            )
            return num_patches

    def forward(self, inputs, original_patch_indices=None, target_masks=None, patches_used=None):
        batch_size = inputs.shape[0]

        tokens = self.patch_projection(inputs)
        if target_masks is not None:
            mask_tokens = self.token_param.repeat(batch_size, target_masks.shape[1], 1)
            patches = torch.cat([tokens, mask_tokens], dim=1)
            patches = torch.gather(
                patches, dim=1, index=original_patch_indices.unsqueeze(-1).repeat(1, 1, self.embed_dim)
            )  # reorder patches to original order
        else:
            patches = tokens

        if self.abs_sincos_enc:
            x = patches + self.pos_embedding(patches, patches_used=patches_used)
        else:
            x = patches

        x = self.encoder(x, masks=patches_used)
        x = self.norm(x)
        x = self.output_projection(x)
        return x


def _extract_model_kwargs(cfg: Mapping[str, Any]) -> dict:
    cfg = dict(cfg)

    # Mandatory: AutoBench must set input_dim
    in_dim = cfg.get("input_dim", None)
    out_dim = cfg.get("output_dim", None)
    if in_dim is None or out_dim is None:
        raise ValueError("input_dim must be specified in the config for MaskedPredictor")

    # Map generic `input_dim` to the actual args
    cfg["input_embed_dim"] = in_dim
    cfg["output_embed_dim"] = out_dim

    sig = inspect.signature(MaskedPredictor.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    ignore = {"_target_", "BUILD"}

    kwargs = {}
    for k, v in cfg.items():
        if k in ignore or k not in allowed:
            continue
        kwargs[k] = v
    return kwargs


def BUILD(cfg: Mapping[str, Any]) -> MaskedPredictor:
    """
    Hydra entrypoint for MaskedPredictor.

    Contract:
      - `input_dim` (from AutoBench) → `input_embed_dim` (and `output_embed_dim` if not set)
      - `embed_dim`, `depth`, `num_heads`, etc. can be set directly in cfg.
    """
    return MaskedPredictor(**_extract_model_kwargs(cfg))
