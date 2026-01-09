"""
Adapted from:
https://github.com/IDEA-Research/MaskDINO/maskdino/modeling/transformer_decoder/dino_decoder.py
"""

import copy
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cell_observatory_platform.models.layers.mlp import MLP
from cell_observatory_platform.models.layers.utils import inverse_sigmoid
from cell_observatory_platform.models.layers.activation import get_activation
from cell_observatory_platform.models.layers.positional_encoding import PositionalEmbeddingSinCos
from cell_observatory_platform.models.layers.attention import CrossAttention, FlashDeformAttn3D

try:
    from ops3d import _C
    OPS3D_AVAILABLE = True
except ImportError:
    OPS3D_AVAILABLE = False


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        feedforward_dim=1024,
        dropout=0.1,
        activation="Relu",
        num_levels=4,
        num_heads=8,
        num_points=4,
        use_deformable_box_attention=False,
        summarize_memory_method=None,
        use_deform_attention=False,
    ):
        super().__init__()

        if use_deform_attention and not OPS3D_AVAILABLE:
            raise ImportError("Please install the deformable attention module.")

        assert embed_dim // num_heads % 8 == 0, "embed_dim//num_heads must be divisible by 8 ..."

        # cross attention
        if use_deformable_box_attention:
            raise NotImplementedError("Deformable box attention is not implemented yet")
        else:
            if use_deform_attention:
                self.with_deform_attention = True
                self.cross_attention = FlashDeformAttn3D(
                    d_model=embed_dim,
                    n_levels=num_levels,
                    n_heads=num_heads,
                    n_points=num_points,
                )
            else:
                self.with_deform_attention = False
                self.cross_attention = CrossAttention(dim=embed_dim, num_heads=num_heads)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)    

        # self attention
        # self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.self_attention = True
        assert embed_dim % num_heads == 0, "d_model must be divisible by nhead"
        self.d_model = embed_dim
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads
        self.attn_drop = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.activation = get_activation(activation)()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.summarize_memory_method = summarize_memory_method
        self.memory_projection = None

    def remove_self_attn_modules(self):
        self.self_attention = False
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embeddings(tensor, pos):
        return tensor if pos is None else tensor + pos

    # (B, L, C) -> (B, H, L, d_head)
    def split_heads(self, x: Tensor) -> Tensor:
        B, L, C = x.shape
        x = x.view(B, L, self.num_heads, self.d_head)
        return x.permute(0, 2, 1, 3)  # (B, H, L, d_head)

    # (B, H, L, d_head) -> (B, L, C)
    def combine_heads(self, x: Tensor) -> Tensor:
        B, H, L, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, L, H * d)

    def _build_attn_mask(
        self,
        self_attention_mask: Optional[Tensor],
        device: torch.device,
    ) -> Optional[Tensor]:
        attn_mask = None
        if self_attention_mask is not None:
            m = self_attention_mask.to(device)
            if m.dtype != torch.bool and not torch.is_floating_point(m):
                m = m.bool()
            # (L, L) -> (1, 1, L, L) so it broadcasts over batch & heads
            attn_mask = m.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        return attn_mask

    def forward_ffn(self, x):
        res = x
        x = self.dropout3(self.activation(self.linear1(x)))
        x = res + self.dropout4(self.linear2(x))
        return self.norm3(x)

    def forward(
        self,
        # for tgt
        target: Optional[Tensor],
        target_query_pos_embeddings: Optional[Tensor] = None,
        target_query_sine_embed: Optional[Tensor] = None,
        target_key_padding_mask: Optional[Tensor] = None,
        target_reference_points: Optional[Tensor] = None,
        # for memory
        memory: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,
        memory_shapes: Optional[Tensor] = None,
        memory_pos_embeddings: Optional[Tensor] = None,  # pos_embeddings for memory
        # for attention masking
        self_attention_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attention_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        # self attention: q,k pos encoding -> self_attn(q,k,v) -> dropout + res -> norm
        if self.self_attention:
            # q = k = self.with_pos_embeddings(target, target_query_pos_embeddings) # optionally add positional encodings
            # attn_out = self.self_attention(q, k, target, attn_mask=self_attention_mask)[0]

            tgt = target.transpose(0, 1)

            device = target.device

            if target_query_pos_embeddings is not None:
                pos = target_query_pos_embeddings.transpose(0, 1)  # (B, L, C)
            else:
                pos = None

            q_in = self.with_pos_embeddings(tgt, pos)
            k_in = q_in
            v_in = tgt

            q = self.split_heads(self.q_proj(q_in))  # (B, H, L, d)
            k = self.split_heads(self.k_proj(k_in))  # (B, H, L, d)
            v = self.split_heads(self.v_proj(v_in))  # (B, H, L, d)

            attn_mask = self._build_attn_mask(
                self_attention_mask=self_attention_mask,
                device=device,
            )

            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                attn_out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop if self.training else 0.0,
                    attn_mask=attn_mask,
                    is_causal=False,
                )  # (B, H, L, d)

            attn_out = self.combine_heads(attn_out).transpose(0, 1)  # (L, B, C)
            target = target + self.dropout2(attn_out)
            target = self.norm2(target)

        # inject a global summary of the memory into the queries
        if self.summarize_memory_method is not None:
            if self.summarize_memory_method == "mean":
                target = target + memory.mean(0, keepdim=True)
            elif self.summarize_memory_method == "projection_mean":
                target = target + self.memory_projection(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown summarize_memory_method: {}".format(self.summarize_memory_method))

        # pos encoding q -> deformable cross attention -> res + dropout -> norm
        if self.with_deform_attention:
            target_cross_attn = self.cross_attention(
                self.with_pos_embeddings(target, target_query_pos_embeddings).transpose(
                    0, 1
                ),  # (bs, num_queries, embed_dim)
                target_reference_points.transpose(0, 1).contiguous(),  # (bs, num_queries, 3/6)
                memory.transpose(0, 1),  # (bs, num_tokens, embed_dim)
                memory_shapes,  # (bs, num_levels, 3)
                memory_level_start_index,  # (num_levels)
                memory_key_padding_mask,
            ).transpose(0, 1)
        else:
            target_cross_attn = self.cross_attention(
                self.with_pos_embeddings(target, target_query_pos_embeddings).transpose(0, 1),  # (bs, num_queries, embed_dim)
                self.with_pos_embeddings(memory, memory_pos_embeddings).transpose(0, 1),  # (bs, num_tokens, embed_dim)
                # TODO: support memory_key_padding_mask
                # memory_key_padding_mask,
            ).transpose(0, 1)
        
        target = target + self.dropout1(target_cross_attn)
        target = self.norm1(target)

        # ffn
        target = self.forward_ffn(target)
        return target  # (num_queries, batch_size, embed_dim)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediates=True,
        embed_dim=256,
        query_dim=4,
        modulate_dhw_attn=True,
        num_feature_levels=1,
        deformable_decoder=True,
        decoder_query_perturber=None,
        num_decoder_layers=None,  # number of queries for each layer in decoder
        remove_decoder_query_scale=True,
        share_decoder_layers=False,
        decoder_layer_dropout_prob=None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_feature_levels = num_feature_levels
        self.num_decoder_layers = num_decoder_layers
        if num_decoder_layers is not None:
            assert isinstance(num_decoder_layers, list)
            assert len(num_decoder_layers) == num_layers

        self.embed_dim = embed_dim  # transformer decoder hidden dim
        self.modulate_dhw_attn = modulate_dhw_attn

        # query dimension (3D (x, y, z) or 6D (x, y, z, w, h, d))
        self.query_dim = query_dim
        assert query_dim in [3, 6], "query_dim should be 3 or 6 but got {}".format(query_dim)

        # return decoder outputs from all layers
        self.return_intermediates = return_intermediates
        assert return_intermediates, "Currently return_intermediate=True is only supported"

        # removes query scaling logic
        if remove_decoder_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError("decoder_query_scale is not implemented yet")
            self.query_scale = MLP(embed_dim, embed_dim, embed_dim, 2)

        if num_layers > 0:
            if share_decoder_layers:
                # share deformable transformer decoder layer (i.e. no deep copy)
                self.decoder_layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
            else:
                self.decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        else:
            self.decoder_layers = []

        self.norm = norm
        # HACK: bbox regressor is defined in the MaskDINODecoder class
        self.bbox_regressor = None
        self.class_predictor = None
        self.deformable_decoder = deformable_decoder

        # TODO: is this necessary?
        if not deformable_decoder and modulate_dhw_attn:
            self.ref_anchor_head = MLP(embed_dim, embed_dim, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        # reference point head: maps initial query positions into embed_dim space
        # used for generating reference points for attention sampling
        # we embed_dim-dim positional encodings per axis
        # recall: MLP args = input_dim, hidden_dim, output_dim, num_layers
        self.num_pos_feats = embed_dim // 3
        self.ref_point_head = MLP(self.query_dim * self.num_pos_feats, embed_dim, embed_dim, 2)
        self.pos_embedding = PositionalEmbeddingSinCos(self.num_pos_feats, normalize=True)

        #  learn a scale to modulate sine positional encodings per query
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(embed_dim, embed_dim, embed_dim, 2)
        else:
            self.query_pos_sine_scale = None

        # probability of dropout in decoder layers
        self.dec_layer_dropout_prob = decoder_layer_dropout_prob
        if decoder_layer_dropout_prob is not None:
            assert isinstance(decoder_layer_dropout_prob, list)
            assert len(decoder_layer_dropout_prob) == num_layers
            for i in decoder_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, FlashDeformAttn3D):
                m._reset_parameters()

    def forward(
        self,
        target,
        memory,
        target_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        target_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos_embeddings: Optional[Tensor] = None,
        reference_points: Optional[Tensor] = None,  # num_queries, bs, 3
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        shapes: Optional[Tensor] = None,  # bs, num_levels, 3
        # (B, L, 3) each element is (w_ratio, h_ratio, d_ratio) for each level
        valid_ratios: Optional[Tensor] = None,
    ):
        intermediates = []
        reference_points = reference_points.sigmoid().to(target.device)  # (num_queries, bs, 3/6) scaled to [0, 1]
        reference_points_list = [reference_points]

        # refine object queries and reference points per iteration & save intermediates
        for layer_id, decoder_layer in enumerate(self.decoder_layers):
            # perturb reference points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)  # (num_queries, bs, 3/6)

            # scale the reference points per feature level using valid_ratios
            # this way attention doesn’t sample into padded regions
            # (num_queries, bs, 1, ref_dim) * (1, bs, nlevel, ref_dim) broadcast-multiply
            # returns: (num_queries, bs, n_levels, query_dim)
            # result: for each query and each level, get scaled reference points based on valid ratio
            reference_points_per_level = reference_points[:, :, None] * torch.cat([valid_ratios] * 2, dim=-1)[None, :]
            # (num_queries, bs, 256*query_dim)
            query_sine_embeddings = self.pos_embedding(reference_points_per_level[:, :, 0, :].to(target.dtype))
            # MLP(query_dim//3*embed_dim,embed_dim,embed_dim,2) returns: (num_queries, bs, embed_dim)
            query_pos_embeddings = self.ref_point_head(query_sine_embeddings)

            decoder_output = decoder_layer(
                # for target
                target=target,
                target_query_pos_embeddings=query_pos_embeddings,
                target_query_sine_embed=query_sine_embeddings,
                target_key_padding_mask=target_key_padding_mask,
                target_reference_points=reference_points_per_level,
                # for memory
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_shapes=shapes,
                memory_pos_embeddings=pos_embeddings,
                # for attention masking
                self_attention_mask=target_mask,
                cross_attention_mask=memory_mask,
            )

            if self.bbox_embed is not None:
                reference_points_pre_sigmoid = inverse_sigmoid(reference_points)
                deltas = self.bbox_embed[layer_id](decoder_output).to(target.device)
                reference_points_updated = (deltas + reference_points_pre_sigmoid).sigmoid()
                reference_points = reference_points_updated.detach()
                reference_points_list.append(reference_points_updated)

            intermediates.append(self.norm(decoder_output))

        return [
            [intermediate.transpose(0, 1) for intermediate in intermediates],
            [reference_point_element.transpose(0, 1) for reference_point_element in reference_points_list],
        ]
