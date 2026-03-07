"""
Adapted from:
https://github.com/IDEA-Research/MaskDINO/main/maskdino/modeling/pixel_decoder/maskdino_encoder.py
"""

import copy
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_platform.models.layers.activation import get_activation
from cell_observatory_platform.models.layers.conv3d import Conv3d
from cell_observatory_platform.models.layers.norm import get_norm
from cell_observatory_platform.models.layers.positional_encoding import PositionalEmbeddingSinCos
from cell_observatory_platform.models.layers.utils import c2_xavier_fill, compute_unmasked_ratio, get_reference_points
from cell_observatory_platform.models.layers.attention import FlashDeformAttn3D, CrossAttention

try:
    from ops3d import _C
    OPS3D_AVAILABLE = True
except ImportError:
    OPS3D_AVAILABLE = False


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(
        self, 
        embed_dim=256, 
        feedforward_dim=1024, 
        dropout=0.1, 
        activation="RELU", 
        n_levels=4, 
        n_heads=8, 
        n_points=4,
        use_deform_attention=False,
    ):
        super().__init__()
        
        if use_deform_attention and not OPS3D_AVAILABLE:
            raise ImportError("Please install the deformable attention module.")
        
        if use_deform_attention:
            self.with_deform_attention = True
            self.attn = FlashDeformAttn3D(
                d_model=embed_dim,
                n_levels=n_levels,
                n_heads=n_heads,
                n_points=n_points,
            )
        else:
            self.with_deform_attention = False
            self.attn = CrossAttention(dim=embed_dim, num_heads=n_heads)
                
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.activation = get_activation(activation)()
        self.dropout2 = nn.Dropout(dropout)

        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, x):
        x = x + self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(x)))))
        return self.norm2(x)

    def forward(self, x, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        q = self.with_pos_embed(x, pos)
        if self.with_deform_attention:
            x = x + self.dropout1(
                self.attn(
                    q,
                    reference_points,
                    x,
                    spatial_shapes,
                    level_start_index,
                    padding_mask,
                )
            )
        else:
            x = x + self.dropout1(
                # cross attention: query, key, value
                self.attn(
                    q, 
                    q, 
                    x,
                    # TODO: support cross_attention_mask
                    # padding_mask,
                )
            )
        x = self.norm1(x)
        x = self.forward_ffn(x)
        return x


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        feedforward_dim=1024,
        num_heads=8,
        num_encoder_layers=6,
        dropout=0.1,
        activation="relu",
        num_feature_levels=4,
        enc_num_points=4,
        use_deform_attention: bool = True
    ):
        super().__init__()

        self.n_head = num_heads
        self.embed_dim = embed_dim
        self.num_layers = num_encoder_layers

        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            embed_dim, feedforward_dim, dropout, activation, 
            num_feature_levels, num_heads, enc_num_points, use_deform_attention
        )
        self.encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_encoder_layers)])

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, embed_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, FlashDeformAttn3D):
                m._reset_parameters()
        normal_(self.level_embed)

    def forward_features(self, x, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        reference_points = get_reference_points(spatial_shapes, valid_ratios, device=x.device)
        for layer in self.encoder_layers:
            x = layer(x, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
        return x

    def get_padding_mask(self, masks, features):
        # if any feature needs padding (dim not divisible by 32) we keep the user-passed masks,
        # otherwise we return all-false masks of the right shape
        enable_mask = any(
            (f.size(2) % 32 != 0) or (f.size(3) % 32 != 0) or (f.size(4) % 32 != 0)
            for f in features
        )
        if masks is None or not enable_mask:
            return [
                torch.zeros((f.size(0), f.size(2), f.size(3), f.size(4)), device=f.device, dtype=torch.bool)
                for f in features
            ]
        return masks

    def forward(self, features, masks, pos_embeddings):
        # we pad if input features don't divide the required map size
        masks = self.get_padding_mask(masks, features)

        # feature: [bs, c, d, h, w]
        feature_shapes = [feature.shape[2:] for feature in features]
        # feature_shapes: [num_levels, 3], with each row = (D, H, W)
        # NOTE: forces GPU-to-CPU sync and is hence very bad for performance, should fix
        feature_shapes = torch.as_tensor(feature_shapes, dtype=torch.long, device=features[0].device)
        # [D1*H1*W1, ..., Dn*Hn*Wn] -> [0, D1*H1*W1, D1*H1*W1 + D2*H2*W2, ...]
        level_start_index = torch.cat((feature_shapes.new_zeros((1,)), feature_shapes.prod(1).cumsum(0)[:-1]))

        # [bs, c, d, h, w] -> [bs, c, d*h*w] -> [bs, d*h*w, c]
        features_flattened = torch.cat([feature.flatten(2).transpose(1, 2) for feature in features], dim=1)

        # [bs, c, d, h, w] -> [bs, c, d*h*w] -> [bs, d*h*w, c]
        positional_embeddings = [pos_embed.flatten(2).transpose(1, 2) for pos_embed in pos_embeddings]
        # level_embed: [embed_dim] -> [1, 1, embed_dim]
        # for given level add a level-specific embedding broadcasted to all positions
        positional_embeddings = [
            pos_embed + self.level_embed[lvl].view(1, 1, -1) for lvl, pos_embed in enumerate(positional_embeddings)
        ]
        positional_embeddings = torch.cat(positional_embeddings, dim=1)  # [bs, d*h*w, embed_dim]

        # list: [bs, d, h, w] -> [bs, d*h*w] for each level list
        masks_flattened = [mask.flatten(1) for mask in masks]
        # [bs, num_levels, d*h*w]
        masks_flattened = torch.cat(masks_flattened, dim=1)

        # [bs, num_levels, 3] (valid ratio for each level), compute_unmasked_ratio gives (W, H, D)
        # which is what get_reference_points expects
        valid_ratios = torch.stack([compute_unmasked_ratio(m) for m in masks], 1)

        # call deformable attention layer on features with masks
        # to ensure only working over valid pixels
        memory = self.forward_features(
            features_flattened, feature_shapes, level_start_index, 
            valid_ratios, positional_embeddings, masks_flattened
        )

        return memory, feature_shapes, level_start_index


class MaskDINOEncoder(nn.Module):
    def __init__(
        self,
        input_shape_metadata: Dict,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        target_min_stride: int,
        total_num_feature_levels: int,
        transformer_encoder_dropout: float,
        transformer_encoder_num_heads: int,
        transformer_encoder_dim_feedforward: int,
        num_transformer_encoder_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Callable = None,
        dtype: str = "bfloat16",
        use_deform_attention: bool = True,
        buffer_device: str = "cuda",
    ):
        super().__init__()

        self.use_deform_attention = use_deform_attention

        if use_deform_attention and not OPS3D_AVAILABLE:
            raise ImportError("Please install the deformable attention module.")

        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        # MaskDINOEncoder:
        # 1. Backbone inputs : len(transformer_in_features)
        # 2. Add Extra encoder levels (downsampled 2x using Conv3D and Groupnorm) N1 = total_num_feature_levels - N0
        # 3. Pass maps (all at conv_dim channels) to Transformer encoder -> outputs same dim as inputs
        # 4. FPN lateral adapters (from backbone) for M0 = num_fpn_levels we do:
        # 5. FPN outputs (top‐down fusion) 2x upsamples transformer output until target_min_stride
        # 6. Final multi‐scale outputs M0 + 1 total maps from coarsest transformer down to target_min_stride
        # 7. Mask head : single 1×1 conv on coarsest FPN map

        # determine shapes of input features
        input_shapes = {k: v for k, v in input_shape_metadata.items() if k in transformer_in_features}
        # sort feature shapes from high to low resolution (take values and sort by stride in ascending order)
        input_shapes_sorted = sorted(input_shapes.items(), key=lambda x: -x[1]["stride"])

        # define feature maps and determine number of feature levels (turn into tuples)
        data_items = [(feature, map["stride"], map["channels"]) for feature, map in input_shapes_sorted]
        self.feature_maps, self.feature_maps_strides, feature_maps_in_channels = zip(*data_items)
        self.num_feature_levels = len(self.feature_maps)

        # note that this is not sorted in high resolution -> low resolution order
        # this will be important for order in which we iterate for FPN lateral fusion
        input_shape_metadata = sorted(input_shape_metadata.items(), key=lambda x: x[1]["stride"])
        self.full_feature_map_set, _, self.full_feature_set_channels = zip(
            *[(k, v["stride"], v["channels"]) for k, v in input_shape_metadata]
        )
        self.total_num_feature_levels = total_num_feature_levels

        # define modules:
        # 1. channel alignment projection blocks to align all feature maps to have the same channel dim
        #    also includes downsampling layers for extra feature levels if needed
        # 2. transformer encoder (uses deformable attention)
        # 3. position embedding (sine positional encoding)
        # 4. mask feature conv layer (1x1 conv to reduce channels for mask prediction)
        # 5. FPN layers (lateral and output convs for top-down fusion)

        assert conv_dim % 32 == 0 and conv_dim % 3 == 0, "conv_dim must be divisible by 32 and 3!"

        assert (
            conv_dim / transformer_encoder_num_heads
        ) % 8 == 0, "conv_dim / transformer_encoder_num_heads must be divisible by 8!"

        if self.num_feature_levels > 1:
            channel_align_blocks = []
            # align all feature maps to have the same channel dim
            for in_channels in feature_maps_in_channels[::-1]:
                channel_align_blocks.append(
                    nn.Sequential(
                        nn.Conv3d(in_channels, conv_dim, kernel_size=1),
                        nn.GroupNorm(32, conv_dim),
                    )
                )

            # we optionally add extra feature levels (assumes coarsest backbone level has max channels)
            extra_in_channels = [max(feature_maps_in_channels)] + [conv_dim] * (
                self.total_num_feature_levels - self.num_feature_levels - 1
            )
            extra_downsample_layers = [
                nn.Sequential(
                    nn.Conv3d(in_channels, conv_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, conv_dim),
                )
                for in_channels in extra_in_channels
            ]
            channel_align_blocks.extend(extra_downsample_layers)
            self.channel_align_projection = nn.ModuleList(channel_align_blocks)
        else:
            self.channel_align_projection = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv3d(feature_maps_in_channels[-1], conv_dim, kernel_size=1),
                        nn.GroupNorm(32, conv_dim),
                    )
                ]
            )

        self.transformer_encoder = MSDeformAttnTransformerEncoder(
            embed_dim=conv_dim,
            feedforward_dim=transformer_encoder_dim_feedforward,
            dropout=transformer_encoder_dropout,
            num_heads=transformer_encoder_num_heads,
            num_encoder_layers=num_transformer_encoder_layers,
            num_feature_levels=self.total_num_feature_levels,
            use_deform_attention=self.use_deform_attention,
        )
        self.pos_embedding = PositionalEmbeddingSinCos(num_pos_feats=conv_dim // 3, normalize=True, scale=None)
        self.mask_features = Conv3d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self._init_model_weights(buffer_device=buffer_device)

    def _init_model_weights(self, buffer_device: str):
        weight_init.c2_xavier_fill(self.mask_features)
        for proj in self.channel_align_projection:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # NOTE: the following FPN logic is in practice not used in MaskDINO, but is left here for reference
        # for lateral_conv, output_conv in zip(self.lateral_convs, self.output_convs):
        #     weight_init.c2_xavier_fill(lateral_conv)
        #     weight_init.c2_xavier_fill(output_conv)

    def forward_features(self, features, masks):
        extra_features_list, extra_pos_embeddings_list = [], []
        # if add extra feature maps, apply input projection to coarsest feature map iteratively
        if self.total_num_feature_levels > self.num_feature_levels:
            # we must have at least one extra feature level
            feature_smallest = self.channel_align_projection[self.num_feature_levels](features[self.feature_maps[0]])
            extra_features_list.append(feature_smallest)
            extra_pos_embeddings_list.append(self.pos_embedding(feature_smallest))
            for l in range(self.num_feature_levels + 1, self.total_num_feature_levels):
                # feature_smallest: [B, C, D1, H1, W1] -> feature: [B, C, D1/2, H1/2, W1/2] etc.
                feature = self.channel_align_projection[l](extra_features_list[-1])
                extra_features_list.append(feature)
                extra_pos_embeddings_list.append(self.pos_embedding(feature))

        features_list, pos_embeddings_list = [], []
        for idx, feature_map in enumerate(self.feature_maps[::-1]):
            # x: [B, C, D, H, W] append to feature list and get pos. encodings
            # reverse order to go from finest to coarsest resolution and then append extra levels
            # which are already in finest to coarsest order (relative to coarsest backbone level)
            x = features[feature_map]
            features_list.append(self.channel_align_projection[idx](x))
            pos_embeddings_list.append(self.pos_embedding(x))

        # NOTE: goes finest -> coarsest order
        features_list.extend(extra_features_list)
        pos_embeddings_list.extend(extra_pos_embeddings_list)

        # encode feature maps with deformable attention transformer encoder
        # features_list: List[Tensor[B, C, D, H, W]] ordered
        # masks: List[Tensor[B, D, H, W]] ordered
        # returns:
        # transformer_features: Tensor[B, sum(D*H*W), C], shapes: List[(D, H, W)], level_start_index: Tensor[num_levels]
        transformer_features, shapes, level_start_index = self.transformer_encoder(
            features_list, masks, pos_embeddings_list
        )

        # output from transformer is flattened, so we need to split per level
        tokens_per_level = [None] * self.total_num_feature_levels
        for i in range(self.total_num_feature_levels - 1):
            # get number of tokens per level, i.e. D*H*W
            tokens_per_level[i] = level_start_index[i + 1] - level_start_index[i]
        tokens_per_level[self.total_num_feature_levels - 1] = (
            transformer_features.shape[1] - level_start_index[self.total_num_feature_levels - 1]
        )

        # [bs, num_levels * tokens_per_level, embed_dim] -> List[Tensor[bs, tokens_in_level_i, C]]
        transformer_features = torch.split(transformer_features, tokens_per_level, dim=1)

        output_feature_maps = []
        for transformer_feature, shape in zip(transformer_features, shapes):
            # unflatten sequence: [bs, tokens_in_level_i, C] -> [bs, C, tokens_in_level_i] -> [bs, C, D, H, W]
            output_map = transformer_feature.transpose(1, 2).view(
                transformer_feature.shape[0], -1, shape[0], shape[1], shape[2]
            )
            output_feature_maps.append(output_map)

        return self.mask_features(output_feature_maps[0]), output_feature_maps[0], output_feature_maps


class Mask2FormerPixelDecoder(nn.Module):
    def __init__(
        self,
        input_shape: Dict[str, Tuple[int]],
        transformer_in_features: List[str],
        total_num_feature_levels: int,
        target_min_stride: int,
        transformer_encoder_dropout: float,
        transformer_encoder_num_heads: int,
        transformer_encoder_dim_feedforward: int,
        transformer_encoder_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        use_deform_attention: bool = True,
    ):
        super().__init__()

        self.use_deform_attention = use_deform_attention
        if use_deform_attention and not OPS3D_AVAILABLE:
            raise ImportError("Please install the deformable attention module.")

        # determine shapes of input features
        input_shapes = {k: v for k, v in input_shape.items() if k in transformer_in_features}
        # sort feature shapes from low to high resolution
        input_shapes_sorted = sorted(input_shapes.items(), key=lambda x: -x[1]["stride"])

        # define feature maps and determine number of feature levels
        data_items = [(feature, map["stride"], map["channels"]) for feature, map in input_shapes_sorted]
        self.feature_maps, self.feature_maps_strides, feature_maps_in_channels = zip(*data_items)
        self.num_feature_levels = len(self.feature_maps)

        # note that this is sorted high resolution -> low resolution order. important for FPN upsampling
        input_shape = sorted(input_shape.items(), key=lambda x: x[1]["stride"])
        self.full_feature_map_set, _, self.full_feature_set_channels = zip(
            *[(k, v["stride"], v["channels"]) for k, v in input_shape]
        )

        self.conv_dim = conv_dim

        self.transformer_num_feature_levels = len(transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            channel_align_blocks = []
            for in_channels in feature_maps_in_channels[::-1]:
                channel_align_blocks.append(
                    nn.Sequential(
                        nn.Conv3d(in_channels, conv_dim, kernel_size=1),
                        nn.GroupNorm(32, conv_dim),
                    )
                )
            self.channel_align_projection = nn.ModuleList(channel_align_blocks)
        else:
            self.channel_align_projection = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv3d(feature_maps_in_channels[-1], conv_dim, kernel_size=1),
                        nn.GroupNorm(32, conv_dim),
                    )
                ]
            )

        self.encoder = MSDeformAttnTransformerEncoder(
            embed_dim=conv_dim,
            dropout=transformer_encoder_dropout,
            num_heads=transformer_encoder_num_heads,
            feedforward_dim=transformer_encoder_dim_feedforward,
            num_encoder_layers=transformer_encoder_layers,
            num_feature_levels=self.transformer_num_feature_levels,
            use_deform_attention=use_deform_attention
        )
        self.pe_layer = PositionalEmbeddingSinCos(math.ceil(conv_dim / 3), normalize=True)
        self.mask_features = Conv3d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # NOTE: we always use 3 scales
        self.total_num_feature_levels = total_num_feature_levels

        self.target_min_stride = target_min_stride

        # extra fpn levels
        stride = min(self.feature_maps_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.target_min_stride))

        lateral_convs, output_convs = [], []
        use_bias = norm == ""
        for in_channels in self.full_feature_set_channels[: self.num_fpn_levels]:
            lateral_norm = get_norm(norm)(conv_dim)
            output_norm = get_norm(norm)(conv_dim)
            lateral_conv = Conv3d(in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm)
            output_conv = Conv3d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )

            # self.add_module("lateral_convs".format(idx + 1), lateral_conv)  # TODO replace "adapter_{}"
            # self.add_module("output_convs".format(idx + 1), output_conv)  # TODO replace layer_{}""

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        # place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer
        self.lateral_convs = nn.ModuleList(lateral_convs[::-1])
        self.output_convs = nn.ModuleList(output_convs[::-1])

        self.weight_init()

    def weight_init(self):
        weight_init.c2_xavier_fill(self.mask_features)
        for proj in self.channel_align_projection:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        for lateral_conv, output_conv in zip(self.lateral_convs, self.output_convs):
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)

    def _pad_pos_embedding(self, pe, conv_dim):
        Cpos = pe.shape[1]
        if Cpos < conv_dim:
            pe = F.pad(pe, (0, 0, 0, 0, 0, 0, 0, conv_dim - Cpos))
        elif Cpos > conv_dim:
            pe = pe[:, :conv_dim, ...]
        return pe

    def forward_features(self, features):
        features_list, pos_embeddings_list = [], []
        # reverse feature maps into top-down order (from high to low resolution)
        for idx, f in enumerate(self.feature_maps[::-1]):
            x = features[f]
            features_list.append(self.channel_align_projection[idx](x))
            pos_embeddings_list.append(self._pad_pos_embedding(self.pe_layer(x), self.conv_dim))

        # encode feature maps with deformable attention transformer encoder
        # features_list: List[Tensor[B, C, D, H, W]] ordered
        # returns:
        # transformer_features: Tensor[B, sum(D*H*W), C], shapes: List[(D, H, W)], level_start_index: Tensor[num_levels]
        masks = [
            torch.zeros((x.size(0), x.size(2), x.size(3), x.size(4)), device=x.device, dtype=torch.bool)
            for x in features_list
        ]
        transformer_features, spatial_shapes, level_start_index = self.encoder(
            features_list, masks, pos_embeddings_list
        )

        tokens_per_level = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                tokens_per_level[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                tokens_per_level[i] = transformer_features.shape[1] - level_start_index[i]

        # [bs, num_levels * tokens_per_level, embed_dim] -> List[Tensor[bs, tokens_in_level_i, C]]
        transformer_features = torch.split(transformer_features, tokens_per_level, dim=1)

        output_map = []
        for transformer_feature, shape in zip(transformer_features, spatial_shapes):
            output_map.append(
                transformer_feature.transpose(1, 2).view(transformer_feature.shape[0], -1, shape[0], shape[1], shape[2])
            )

        # append `output_map` with extra FPN levels
        # reverse feature maps into top-down order (from low to high resolution from finest K backbone maps)
        for idx, f in enumerate(self.full_feature_map_set[: self.num_fpn_levels][::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)

            # following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(output_map[-1], size=cur_fpn.shape[-3:], mode="trilinear", align_corners=False)
            y = output_conv(y)
            output_map.append(y)

        multi_scale_features = list(output_map[: self.total_num_feature_levels])
        return self.mask_features(output_map[-1]), output_map[0], multi_scale_features
