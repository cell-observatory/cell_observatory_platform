import logging
import sys
from typing import Literal, Union

import torch
import torch.nn as nn

from models.norm import get_norm
from models.activation import get_activation
from models.mlp import get_mlp
from models.maskedencoder import MaskedEncoder
from models.maskedpredictor import MaskedPredictor
from training.masking import mask_random_patches, apply_masks

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


CONFIGS = {
    'mae-tiny': {
        'embed_dim': 192,
        'decoder_embed_dim': 96,
        'depth': 12,
        'decoder_depth': 3,
        'num_heads': 3,
        'decoder_num_heads': 3,
        'mlp_ratio': 4,
    },
    'mae-small': {
        'embed_dim': 384,
        'decoder_embed_dim': 192,
        'depth': 12,
        'decoder_depth': 6,
        'num_heads': 6,
        'decoder_num_heads': 6,
        'mlp_ratio': 4,
    },
    'mae-base': {
        'embed_dim': 768,
        'decoder_embed_dim': 256,
        'depth': 12,
        'decoder_depth': 8,
        'num_heads': 12,
        'decoder_num_heads': 8,
        'mlp_ratio': 4,
    },
    'mae-large': {
        'embed_dim': 1024,
        'decoder_embed_dim': 512,
        'depth': 24,
        'decoder_depth': 8,
        'num_heads': 16,
        'decoder_num_heads': 8,
        'mlp_ratio': 4,
    },
    'mae-huge': {
        'embed_dim': 1280,
        'decoder_embed_dim': 512,
        'depth': 32,
        'decoder_depth': 8,
        'num_heads': 16,
        'decoder_num_heads': 8,
        'mlp_ratio': 4,
    },
    'mae-giant': {
        'embed_dim': 1408,
        'decoder_embed_dim': 512,
        'depth': 40,
        'decoder_depth': 8,
        'num_heads': 16,
        'decoder_num_heads': 8,
        'mlp_ratio': 48/11,
    },
    'mae-gigantic': {
        'embed_dim': 1664,
        'decoder_embed_dim': 1024,
        'depth': 48,
        'decoder_depth': 16,
        'num_heads': 16,
        'decoder_num_heads': 16,
        'mlp_ratio': 64/13,
    }
}


class MaskedAutoEncoder(nn.Module):
    def __init__(
        self,
        model_template: Literal[
            'mae', # custom use `embed_dim`, `decoder_embed_dim`, `depth`, `num_heads` and `mlp_ratio` to config model
            'mae-tiny',
            'mae-small',
            'mae-base',
            'mae-large',
            'mae-huge',
            'mae-giant',
            'mae-gigantic'
        ] = 'mae',
        input_fmt='BZYXC',
        input_shape=(1, 6, 64, 64, 1),
        lateral_patch_size=16,
        axial_patch_size=1,
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
        norm_layer: Union[nn.Module, Literal['RmsNorm', 'LayerNorm', 'SyncBatchNorm', 'GroupNorm']] = 'RmsNorm',
        act_layer: Union[nn.Module, Literal['GELU', 'SiLU', 'LeakyReLU', 'GLU', 'Sigmoid', 'Tanh']] = 'SiLU',
        mlp_layer: Union[nn.Module, Literal['Mlp', 'SwiGLU']] = 'SwiGLU',
        use_conv_proj=False,
        mask_ratio=.9,
        window_mask_shape=None,
        **kwargs,
    ):
        super().__init__()

        if model_template in CONFIGS.keys():
            config = CONFIGS[model_template]
            self.depth = config['depth']
            self.decoder_depth = config['decoder_depth']
            self.embed_dim = config['embed_dim']
            self.decoder_embed_dim = config['decoder_embed_dim']
            self.num_heads = config['num_heads']
            self.decoder_num_heads = config['decoder_num_heads']
            self.mlp_ratio = config['mlp_ratio']
        else:
            self.depth = depth
            self.decoder_depth = decoder_depth
            self.embed_dim = embed_dim
            self.decoder_embed_dim = decoder_embed_dim
            self.num_heads = num_heads
            self.decoder_num_heads = decoder_num_heads
            self.mlp_ratio = mlp_ratio

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
        self.mask_ratio = mask_ratio
        self.window_mask_shape = window_mask_shape

        self.init_std = init_std

        self.norm_layer = get_norm(norm_layer)
        self.act_layer = get_activation(act_layer)
        self.mlp_layer = get_mlp(mlp_layer)

        self.masked_encoder = MaskedEncoder(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            lateral_patch_size=self.lateral_patch_size,
            axial_patch_size=self.axial_patch_size,
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
            use_conv_proj=use_conv_proj,
            cls_token=False,
        )

        self.masked_decoder = MaskedPredictor(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            lateral_patch_size=self.lateral_patch_size,
            axial_patch_size=self.axial_patch_size,
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
            cls_token=False,
        )

    @torch.jit.ignore
    def get_encoder(self):
        return self.masked_encoder

    @torch.jit.ignore
    def get_decoder(self):
        return self.masked_decoder

    @torch.jit.ignore
    def get_num_patches(self):
        return self.masked_encoder.pos_embedding.num_patches

    def forward(self, inputs):
        masks, context_masks, target_masks, original_patch_indices = mask_random_patches(
            inputs=inputs,
            num_patches=self.get_num_patches(),
            ratio=self.mask_ratio,
            window_mask_shape=self.window_mask_shape
        )

        x, patches = self.masked_encoder(inputs, masks=context_masks)
        x = self.masked_decoder(x, original_patch_indices=original_patch_indices, target_masks=target_masks)

        # compute loss over masked patches
        targets = apply_masks(patches, masks=target_masks)
        predictions = apply_masks(x, masks=target_masks)
        loss = (targets - predictions) ** 2
        loss = loss.mean(dim=-1)  # mean loss per patch
        loss = loss.sum() / masks.sum()
        return loss
