import logging
import sys
from typing import Literal, Union

import torch
import torch.nn as nn
from timm.layers import AttentionPoolLatent
from timm.models.vision_transformer import global_pool_nlc

from cell_observatory_platform.models.layers.mlp import get_mlp
from cell_observatory_platform.models.layers.norm import get_norm
from cell_observatory_platform.models.backbones.encoder import Encoder
from cell_observatory_platform.models.layers.activation import get_activation
from cell_observatory_platform.models.layers.attention import RopeAttention
from cell_observatory_platform.models.layers.patch_embeddings import PatchEmbedding, calc_num_patches
from cell_observatory_platform.models.layers.positional_encoding import PosEmbedding, make_axial_rope_freqs

logger = logging.getLogger(__name__)


CONFIGS = {
    "vit-tiny": {
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_ratio": 4,
    },
    "vit-small": {
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4,
    },
    "vit-base": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
    },
    "vit-large": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4,
    },
    "vit-huge": {
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "mlp_ratio": 4,
    },
    "vit-2billion": {
        "embed_dim": 2560,
        "depth": 24,
        "num_heads": 32,
        "mlp_ratio": 4,
    },
    "vit-6billion": {
        "embed_dim": 4096,
        "depth": 32,
        "num_heads": 32,
        "mlp_ratio": 4,
    },
    "vit-giant": {
        "embed_dim": 1408,
        "depth": 40,
        "num_heads": 16,
        "mlp_ratio": 48 / 11,
    },
    "vit-gigantic": {
        "embed_dim": 1664,
        "depth": 48,
        "num_heads": 16,
        "mlp_ratio": 64 / 13,
    },
    "vit-enormous": {
        "embed_dim": 1792,
        "depth": 56,
        "num_heads": 16,
        "mlp_ratio": 8.5714285714,
    },
}


class ViT(nn.Module):
    def __init__(
        self,
        model_template: Literal[
            "vit",  # custom use `embed_dim`, `depth`, `num_heads` and `mlp_ratio` to config model
            "vit-tiny",
            "vit-small",
            "vit-base",
            "vit-large",
            "vit-huge",
            "vit-giant",
            "vit-gigantic",
        ] = "vit",
        input_fmt="TZYXC",
        input_shape: tuple = (16, 128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        modes=15,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        proj_drop_rate=0.0,
        att_drop_rate=0.0,
        drop_path_rate=0.1,
        init_std=0.02,
        fixed_dropout_depth=False,
        global_pool: Literal["", "avg", "avgmax", "max", "token", "map"] = "avgmax",
        norm_layer: Union[nn.Module, Literal["RmsNorm", "LayerNorm", "SyncBatchNorm", "GroupNorm"]] = "LayerNorm",
        act_layer: Union[nn.Module, Literal["GELU", "SiLU", "LeakyReLU", "GLU", "Sigmoid", "Tanh"]] = "GELU",
        mlp_layer: Union[nn.Module, Literal["Mlp", "SwiGLU"]] = "Mlp",
        abs_sincos_enc: bool = False,
        rope_pos_enc: bool = True,
        rope_random_rotation_per_head: bool = True,
        rope_type: Literal["mixed", "axial", "custom"] = "axial",
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

        self.input_fmt = input_fmt
        self.input_shape = input_shape

        axis_to_value = dict(zip(input_fmt, input_shape))
        self.in_chans = axis_to_value["C"]
        self.num_frames = axis_to_value["T"]

        self.patch_shape = patch_shape

        self.proj_drop_rate = proj_drop_rate
        self.att_drop_rate = att_drop_rate
        self.drop_path_rate = drop_path_rate
        self.fixed_dropout_depth = fixed_dropout_depth

        self.init_std = init_std
        self.global_pool = global_pool

        self.norm_layer = get_norm(norm_layer)
        self.act_layer = get_activation(act_layer)
        self.mlp_layer = get_mlp(mlp_layer)

        self.norm = self.norm_layer(self.embed_dim) if norm_layer is not None else nn.Identity()

        self.patch_embedding = PatchEmbedding(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            patch_shape=self.patch_shape,
            embed_dim=self.embed_dim,
            channels=self.in_chans,
        )

        # positional encoding parameters
        self.abs_sincos_enc = abs_sincos_enc
        self.rope_pos_enc = rope_pos_enc
        self.rope_type = rope_type
        self.rope_theta = rope_theta
        self.wide_silu = mlp_wide_silu
        self.rope_random_rotation_per_head = rope_random_rotation_per_head

        if self.abs_sincos_enc:
            self.pos_embedding = PosEmbedding(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                patch_shape=self.patch_shape,
                embed_dim=self.embed_dim,
                channels=self.in_chans,
                cls_token=False,
            )
        # precompute axial RoPE frequencies once and store as buffer
        if self.rope_pos_enc and self.rope_type == "axial":
            freqs_cis = make_axial_rope_freqs(
                input_fmt=self.input_fmt,
                input_shape=self.input_shape,
                patch_shape=self.patch_shape,
                dim=self.embed_dim // self.num_heads,
                theta=self.rope_theta,
            )
            self.register_buffer("freqs_cis", freqs_cis)
        else:
            self.freqs_cis = None

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
            rope_type=rope_type,
            rope_theta=rope_theta,
            input_fmt=input_fmt,
            input_shape=input_shape,
            patch_shape=self.patch_shape,
            # Encoder's parameter is `wide_silu`; the old `mlp_wide_silu=` kwarg
            # was silently swallowed, so wide-SiLU configs never took effect.
            wide_silu=mlp_wide_silu,
            dtype=dtype,
        )

        self.global_pool = global_pool

        if global_pool == "map":
            self.att_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=self.norm_layer,
            )
        else:
            self.att_pool = None

        self.head = nn.Linear(self.embed_dim, modes) if modes > 0 else nn.Identity()
        self.head_drop = nn.Dropout(self.proj_drop_rate)

        self._init_model_weights()

    def _init_model_weights(self, buffer_device: str | None = None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)
        w = self.patch_embedding.proj.weight
        torch.nn.init.xavier_uniform_(w.view(w.shape[0], -1))

        for mod in self.modules():
            if isinstance(mod, RopeAttention):
                mod.init_rope_parameters(device=buffer_device)

    @torch.jit.ignore
    def get_num_layers(self):
        return self.encoder.get_num_layers()

    @torch.jit.ignore
    def get_encoder(self):
        return self.encoder

    @torch.jit.ignore
    def get_head(self):
        return self.head

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

    def pool(self, x, pool_type=None, num_prefix_tokens=0):
        # This ViT has NO cls/prefix token (pure PatchEmbedding): the previous
        # default of 1 made timm's global_pool_nlc average x[:, 1:], silently
        # dropping the first patch token from every pooled output.
        if self.att_pool is not None:
            x = self.att_pool(x)
            return x

        pool_type = self.global_pool if pool_type is None else pool_type
        x = global_pool_nlc(x, pool_type=pool_type, num_prefix_tokens=num_prefix_tokens)
        return x

    def forward_head(self, x):
        x = self.pool(x)
        x = self.norm(x)
        x = self.head_drop(x)
        return self.head(x)

    def forward(self, data_sample: dict):
        inputs, meta = data_sample["data_tensor"], data_sample["metainfo"]

        x = self.patch_embedding(inputs)
        if self.abs_sincos_enc:
            x += self.pos_embedding(inputs)

        x = self.encoder(x, pos_enc=self.freqs_cis)
        x = self.forward_head(x)
        return x
