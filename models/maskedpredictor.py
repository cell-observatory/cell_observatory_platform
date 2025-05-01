import logging
import sys
from typing import Literal, Union

import torch
import torch.nn as nn

from models.norm import get_norm
from models.activation import get_activation
from models.mlp import get_mlp
from models.encoder import Encoder
from models.patch_embeddings import PosEmbedding

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


CONFIGS = {
    'mp-tiny': {
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'mlp_ratio': 4,
    },
    'mp-small': {
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4,
    },
    'mp-base': {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
    },
    'mp-large': {
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4,
    },
    'mp-huge': {
        'embed_dim': 1280,
        'depth': 32,
        'num_heads': 16,
        'mlp_ratio': 4,
    },
    'mp-giant': {
        'embed_dim': 1408,
        'depth': 40,
        'num_heads': 16,
        'mlp_ratio': 48/11,
    },
    'mp-gigantic': {
        'embed_dim': 1664,
        'depth': 48,
        'num_heads': 16,
        'mlp_ratio': 64/13,
    }
}


class MaskedPredictor(nn.Module):
    def __init__(
        self,
        model_template: Literal[
            'mp', # custom use `embed_dim`, `depth`, `num_heads` and `mlp_ratio` to config model
            'mp-tiny',
            'mp-small',
            'mp-base',
            'mp-large',
            'mp-huge',
            'mp-giant',
            'mp-gigantic'
        ] = 'mp',
        input_fmt='BZYXC',
        input_shape=(1, 6, 64, 64, 1),
        lateral_patch_size=16,
        axial_patch_size=1,
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
        norm_layer: Union[nn.Module, Literal['RmsNorm', 'LayerNorm', 'SyncBatchNorm', 'GroupNorm']] = 'RmsNorm',
        act_layer: Union[nn.Module, Literal['GELU', 'SiLU', 'LeakyReLU', 'GLU', 'Sigmoid', 'Tanh']] = 'SiLU',
        mlp_layer: Union[nn.Module, Literal['Mlp', 'SwiGLU']] = 'SwiGLU',
        **kwargs,
    ):
        super().__init__()

        if model_template in CONFIGS.keys():
            config = CONFIGS[model_template]
            self.depth = config['depth']
            self.embed_dim = config['embed_dim']
            self.num_heads = config['num_heads']
            self.mlp_ratio = config['mlp_ratio']
        else:
            self.depth = depth
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.mlp_ratio = mlp_ratio

        self.input_embed_dim = input_embed_dim
        self.output_embed_dim = output_embed_dim

        self.input_fmt = input_fmt
        self.input_shape = input_shape
        self.img_size = input_shape[-2]
        self.in_chans = input_shape[-1]
        self.num_frames = input_shape[1]

        self.axial_patch_size = axial_patch_size
        self.lateral_patch_size = lateral_patch_size

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

        self.patch_projection = nn.Linear(
            self.input_embed_dim,
            self.embed_dim,
            bias=True
        )

        self.output_projection = nn.Linear(
            self.embed_dim,
            self.output_embed_dim,
            bias=True
        )

        self.pos_embedding = PosEmbedding(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            lateral_patch_size=self.lateral_patch_size,
            axial_patch_size=self.axial_patch_size,
            embed_dim=self.embed_dim
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
            init_std=self.init_std
        )

    @torch.jit.ignore
    def get_num_layers(self):
        return self.encoder.get_num_layers()

    @torch.jit.ignore
    def get_encoder(self):
        return self.encoder

    @torch.jit.ignore
    def get_num_patches(self):
        return self.pos_embedding.num_patches

    def forward(self, inputs, original_patch_indices=None, target_masks=None):
        batch_size = inputs.shape[0]

        tokens = self.patch_projection(inputs)
        mask_tokens = self.token_param.repeat(batch_size, target_masks.shape[1], 1)

        patches = torch.cat([tokens, mask_tokens], dim=1)
        patches = torch.gather(
            patches,
            dim=1,
            index=original_patch_indices.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        ) # reorder patches to original order

        x = patches + self.pos_embedding(patches)

        x = self.encoder(x)
        x = self.norm(x)
        x = self.output_projection(x)
        return x
