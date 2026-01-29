"""
Adapted from:
https://github.com/facebookresearch/dinov3/main/dinov3/eval/segmentation/models/heads/mask2former_transformer_decoder.py
"""

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from cell_observatory_platform.models.layers.activation import get_activation
from cell_observatory_platform.models.layers.conv3d import Conv3d
from cell_observatory_platform.models.layers.mlp import MLP
from cell_observatory_platform.models.layers.positional_encoding import PositionalEmbeddingSinCos
from cell_observatory_platform.models.layers.utils import c2_xavier_fill


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.num_heads = nhead
        self.d_head = d_model // nhead
        self.attn_drop = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation(activation)()
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
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
        tgt_mask: Optional[Tensor],
        device: torch.device,
    ) -> Optional[Tensor]:
        attn_mask = None
        if tgt_mask is not None:
            m = tgt_mask.to(device)
            if m.dtype != torch.bool and not torch.is_floating_point(m):
                m = m.bool()
            # (L, L) -> (1, 1, L, L) so it broadcasts over batch & heads
            attn_mask = m.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        return attn_mask

    def forward_post(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        device = tgt.device

        # (L, B, C) -> (B, L, C)
        target = tgt.transpose(0, 1)
        query_pos = query_pos.transpose(0, 1) if query_pos is not None else None

        q_in = self.with_pos_embed(target, query_pos)
        k_in = q_in
        v_in = target

        q = self.split_heads(self.q_proj(q_in))  # (B, H, L, d)
        k = self.split_heads(self.k_proj(k_in))  # (B, H, L, d)
        v = self.split_heads(self.v_proj(v_in))  # (B, H, L, d)

        attn_mask = self._build_attn_mask(
            tgt_mask=tgt_mask,
            device=device,
        )

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop if self.training else 0.0,
                is_causal=False,
            )  # (B, H, L, d)

        attn_out = self.combine_heads(attn_out)  # (B, L, C)
        tgt = target + self.dropout(attn_out)
        tgt = self.norm(tgt)
        return tgt.transpose(0, 1)

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        device = tgt.device
        target = tgt.transpose(0, 1)
        query_pos = query_pos.transpose(0, 1) if query_pos is not None else None
        tgt2 = self.norm(target)

        q_in = self.with_pos_embed(tgt2, query_pos)
        k_in = q_in
        v_in = tgt2

        q = self.split_heads(self.q_proj(q_in))  # (B, H, L, d)
        k = self.split_heads(self.k_proj(k_in))  # (B, H, L, d)
        v = self.split_heads(self.v_proj(v_in))  # (B, H, L, d)

        attn_mask = self._build_attn_mask(
            tgt_mask=tgt_mask,
            device=device,
        )

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop if self.training else 0.0,
                is_causal=False,
            )  # (B, H, L, d)

        attn_out = self.combine_heads(attn_out)
        tgt = target + self.dropout(attn_out)
        return tgt.transpose(0, 1)

    # def forward_post(
    #     self,
    #     tgt,
    #     tgt_mask: Optional[Tensor] = None,
    #     tgt_key_padding_mask: Optional[Tensor] = None,
    #     query_pos: Optional[Tensor] = None,
    # ):
    #     q = k = self.with_pos_embed(tgt, query_pos)
    #     tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
    #                           key_padding_mask=tgt_key_padding_mask)[0]
    #     tgt = tgt + self.dropout(tgt2)
    #     tgt = self.norm(tgt)
    #     return tgt

    # def forward_pre(
    #     self,
    #     tgt,
    #     tgt_mask: Optional[Tensor] = None,
    #     tgt_key_padding_mask: Optional[Tensor] = None,
    #     query_pos: Optional[Tensor] = None,
    # ):
    #     tgt2 = self.norm(tgt)
    #     q = k = self.with_pos_embed(tgt2, query_pos)
    #     tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
    #                           key_padding_mask=tgt_key_padding_mask)[0]
    #     tgt = tgt + self.dropout(tgt2)
    #     return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation(activation)()
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = get_activation(activation)()
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        in_channels,
        mask_classification: bool,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        decoder_nheads: int,
        dim_feedforward: int,
        decoder_num_layers: int,
        decoder_pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        num_feature_levels: int,
    ):
        super().__init__()
        
        assert input_dim == 3, "We only support 3D input currently."
        assert mask_classification, "We currently only support mask classification."
        self.mask_classification = mask_classification

        # positional encoding
        self.input_proj_hidden_dim = hidden_dim
        self.pe_layer = PositionalEmbeddingSinCos(math.ceil(hidden_dim // 3), normalize=True)

        # transformer decoder
        self.num_heads = decoder_nheads
        self.num_layers = decoder_num_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=decoder_nheads,
                    dropout=0.0,
                    normalize_before=decoder_pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=decoder_nheads,
                    dropout=0.0,
                    normalize_before=decoder_pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=decoder_pre_norm,
                )
            )

        self.post_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query positional encodings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embeddings
        self.num_feature_levels = num_feature_levels
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv3d(in_channels, hidden_dim, kernel_size=1))
                c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def _pad_pos_embedding(self, pe, conv_dim):
        Cpos = pe.shape[1]
        if Cpos < conv_dim:
            pe = F.pad(pe, (0, 0, 0, 0, 0, 0, 0, conv_dim - Cpos))
        elif Cpos > conv_dim:
            pe = pe[:, :conv_dim, ...]
        return pe

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels, f"Expect {self.num_feature_levels} feature levels, got {len(x)}."
        src, pos, size_list = [], [], []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-3:])
            # pos: [B, Cpos, D, H, W] -> [B, Cpos, DHW] -> [DHW, B, Cpos]
            p = self.pe_layer(x[i], None)
            p = self._pad_pos_embedding(p, self.input_proj_hidden_dim).flatten(2).permute(2, 0, 1)
            pos.append(p)
            # src: [B, C, D, H, W] -> [B, C, DHW] -> [DHW, B, C]
            s = self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]
            src.append(s.permute(2, 0, 1))

        _, B, _ = src[0].shape
        # query_embed/output: [num_queries, C] -> [num_queries, B, C]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, B, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        # outputs_class: [B, num_queries, num_classes]
        # outputs_mask: [B, num_queries, Z, Y, X]
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index],
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.post_norm(output)
        # decoder_output: [num_queries, B, C] -> [B, num_queries, C]
        decoder_output = decoder_output.transpose(0, 1)
        # outputs_class: [B, num_queries, num_classes]
        outputs_class = self.class_embed(decoder_output)
        # outputs_mask: [B, num_queries, mask_dim]
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bcdhw->bqdhw", mask_embed, mask_features)

        # prediction is of higher-resolution
        # [B, Q, D, H, W] -> [B, Q, D*H*W] -> [B, h, Q, D*H*W] -> [B*h, Q, DHW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="trilinear", align_corners=False)
        # must use bool type
        # if a BoolTensor is provided, positions with ``True`` are not allowed
        # to attend while ``False`` values will be unchanged
        attn_mask = (
            attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5
        ).bool()
        attn_mask = attn_mask.detach()
        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list
        if self.mask_classification:
            return [{"pred_logits": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
