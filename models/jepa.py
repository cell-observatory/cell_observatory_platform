import logging
import sys
from copy import deepcopy
from typing import Literal, Union

import torch
import torch.nn as nn
from deepspeed.runtime.zero import GatheredParameters
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

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
    'jepa-tiny': {
        'embed_dim': 192,
        'predictor_embed_dim': 96,
        'depth': 12,
        'predictor_depth': 3,
        'num_heads': 3,
        'predictor_num_heads': 3,
        'mlp_ratio': 4,
    },
    'jepa-small': {
        'embed_dim': 384,
        'predictor_embed_dim': 192,
        'depth': 12,
        'predictor_depth': 6,
        'num_heads': 6,
        'predictor_num_heads': 6,
        'mlp_ratio': 4,
    },
    'jepa-base': {
        'embed_dim': 768,
        'predictor_embed_dim': 384,
        'depth': 12,
        'predictor_depth': 12,
        'num_heads': 12,
        'predictor_num_heads': 12,
        'mlp_ratio': 4,
    },
    'jepa-large': {
        'embed_dim': 1024,
        'predictor_embed_dim': 384,
        'depth': 24,
        'predictor_depth': 12,
        'num_heads': 16,
        'predictor_num_heads': 12,
        'mlp_ratio': 4,
    },
    'jepa-huge': {
        'embed_dim': 1280,
        'predictor_embed_dim': 384,
        'depth': 32,
        'predictor_depth': 12,
        'num_heads': 16,
        'predictor_num_heads': 12,
        'mlp_ratio': 4,
    },
    'jepa-2billion': {
        'embed_dim': 2560,
        'predictor_embed_dim': 512,
        'depth': 24,
        'predictor_depth': 8,
        'num_heads': 32,
        'predictor_num_heads': 8,
        'mlp_ratio': 4,
    },
    'jepa-6billion': {
        'embed_dim': 4096,
        'predictor_embed_dim': 512,
        'depth': 32,
        'predictor_depth': 8,
        'num_heads': 32,
        'predictor_num_heads': 8,
        'mlp_ratio': 4,
    },
    'jepa-giant': {
        'embed_dim': 1408,
        'predictor_embed_dim': 512,
        'depth': 40,
        'predictor_depth': 12,
        'num_heads': 16,
        'predictor_num_heads': 12,
        'mlp_ratio': 48/11,
    },
    'jepa-gigantic': {
        'embed_dim': 1664,
        'predictor_embed_dim': 1024,
        'depth': 48,
        'predictor_depth': 16,
        'num_heads': 16,
        'predictor_num_heads': 16,
        'mlp_ratio': 64/13,
    },
    'jepa-enormous': {
        'embed_dim': 1792,
        'predictor_embed_dim': 1024,
        'depth': 56,
        'predictor_depth': 16,
        'num_heads': 16,
        'predictor_num_heads': 16,
        'mlp_ratio': 8.5714285714,
    }
}


class JEPA(nn.Module):
    def __init__(
        self,
        model_template: Literal[
            'jepa', # custom use `embed_dim`, `predictor_embed_dim`, `depth`, `num_heads` and `mlp_ratio` to config model
            'jepa-tiny',
            'jepa-small',
            'jepa-base',
            'jepa-large',
            'jepa-huge',
            'jepa-giant',
            'jepa-gigantic'
        ] = 'jepa',
        input_fmt='BZYXC',
        input_shape=(1, 6, 64, 64, 1),
        lateral_patch_size=16,
        axial_patch_size=1,
        embed_dim=768,
        predictor_embed_dim=256,
        depth=12,
        predictor_depth=8,
        num_heads=12,
        predictor_num_heads=8,
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
            self.predictor_depth = config['predictor_depth']
            self.embed_dim = config['embed_dim']
            self.predictor_embed_dim = config['predictor_embed_dim']
            self.num_heads = config['num_heads']
            self.predictor_num_heads = config['predictor_num_heads']
            self.mlp_ratio = config['mlp_ratio']
        else:
            self.depth = depth
            self.predictor_depth = predictor_depth
            self.embed_dim = embed_dim
            self.predictor_embed_dim = predictor_embed_dim
            self.num_heads = num_heads
            self.predictor_num_heads = predictor_num_heads
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

        self.input_encoder = MaskedEncoder(
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

        self.target_encoder = deepcopy(self.input_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.target_predictor = MaskedPredictor(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            lateral_patch_size=self.lateral_patch_size,
            axial_patch_size=self.axial_patch_size,
            channels=self.in_chans,
            input_embed_dim=self.embed_dim,
            output_embed_dim=self.embed_dim,
            embed_dim=self.predictor_embed_dim,
            depth=self.predictor_depth,
            num_heads=self.predictor_num_heads,
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

    def ema_update(self, beta=0.99):
        def collect_params(params):
            return [
                p for p in params
                if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
            ]

        with torch.no_grad():
            for iparam, tparam in zip(self.input_encoder.parameters(), self.target_encoder.parameters()):
                fetch = collect_params([iparam, tparam])
                with GatheredParameters(fetch, enabled=len(fetch) > 0):
                    tparam.data.copy_(torch.lerp(iparam.data, tparam.data, beta))

    @torch.jit.ignore
    def get_input_encoder(self):
        return self.input_encoder

    @torch.jit.ignore
    def get_target_encoder(self):
        return self.target_encoder

    @torch.jit.ignore
    def get_predictor(self):
        return self.target_predictor

    @torch.jit.ignore
    def get_num_patches(self):
        return self.input_encoder.pos_embedding.num_patches

    def forward(self, inputs):
        masks, context_masks, target_masks, original_patch_indices = mask_random_patches(
            inputs=inputs,
            num_patches=self.get_num_patches(),
            ratio=self.mask_ratio,
            window_mask_shape=self.window_mask_shape
        )

        embedding, patches = self.input_encoder(inputs, masks=context_masks)
        predictions = self.target_predictor(
            embedding,
            original_patch_indices=original_patch_indices,
            target_masks=target_masks
        )

        with torch.no_grad():
            targets, _ = self.target_encoder(inputs)

        targets = apply_masks(targets, masks=target_masks)
        predictions = apply_masks(predictions, masks=target_masks)

        # compute loss over masked patches
        loss = torch.abs(targets - predictions)
        loss = loss.mean(dim=-1)  # mean loss per patch
        loss = loss.sum() / masks.sum()
        loss = loss.to(targets.dtype)
        return loss

