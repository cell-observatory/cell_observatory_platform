"""
Adapted from:
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, normal_, xavier_uniform_

from cell_observatory_platform.data.structures import box_xyzxyz_to_cxcyczwhd, delta2bbox

# from cell_observatory_platform.models.heads.global_ape_decoder import build_global_ape_decoder
from cell_observatory_platform.models.heads.global_rpe_decomp_decoder import build_global_rpe_decomp_decoder
from cell_observatory_platform.models.heads.plain_detr_transformer_encoder import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from cell_observatory_platform.models.layers.norm import LayerNorm3D
from cell_observatory_platform.models.layers.utils import compute_unmasked_ratio


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_feature_levels=4,
        two_stage=False,
        two_stage_num_proposals=300,
        mixed_selection=False,
        norm_type="post_norm",
        decoder_type="deform",
        proposal_feature_levels=1,
        proposal_in_stride=16,
        proposal_tgt_strides=[8, 16, 32, 64],
        proposal_min_size=50,
        global_decoder_args=None,
        # transformer_encoder
        add_transformer_encoder=False,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        num_encoder_layers=6,
    ):
        super().__init__()

        self.nhead = nhead
        self.d_model = d_model
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        assert norm_type in [
            "pre_norm",
            "post_norm",
        ], f"expected norm type either pre_norm or post_norm, got {norm_type}"

        if decoder_type == "global_ape":
            raise NotImplementedError("global_ape decoder not yet implemented!")
            # self.decoder = build_global_ape_decoder(global_decoder_args)
        elif decoder_type == "global_rpe_decomp":
            self.decoder = build_global_rpe_decomp_decoder(global_decoder_args)
            assert two_stage == True, "global_rpe_decomp decoder only supports two_stage=True"
        else:
            raise NotImplementedError

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            in_dim = 3 * self.d_model
            self.pos_trans = nn.Linear(in_dim, 2 * self.d_model)
            self.pos_trans_norm = nn.LayerNorm(2 * self.d_model)
        else:
            # FIXME: not currently excercised and not compatible with rpe_decomp decoder
            #        which is the only decoder that is currently implemented
            #        so perhaps we should remove this branch altogether
            self.reference_points = nn.Linear(d_model, 3)

        self.mixed_selection = mixed_selection

        self.proposal_feature_levels = proposal_feature_levels
        self.proposal_tgt_strides = proposal_tgt_strides
        self.proposal_min_size = proposal_min_size

        if two_stage and proposal_feature_levels > 1:
            assert (
                len(proposal_tgt_strides) == proposal_feature_levels
            ), "proposal_tgt_strides should have same length as proposal_feature_levels"

            self.proposal_in_stride = proposal_in_stride
            self.enc_output_proj = nn.ModuleList([])

            for stride in proposal_tgt_strides:
                if stride == proposal_in_stride:
                    self.enc_output_proj.append(nn.Identity())

                elif stride > proposal_in_stride:
                    scale = int(math.log2(stride / proposal_in_stride))
                    layers = []
                    for _ in range(scale - 1):
                        layers += [
                            nn.Conv3d(d_model, d_model, kernel_size=2, stride=2),
                            LayerNorm3D(d_model),
                            nn.GELU(),
                        ]
                    layers.append(nn.Conv3d(d_model, d_model, kernel_size=2, stride=2))
                    self.enc_output_proj.append(nn.Sequential(*layers))

                else:
                    scale = int(math.log2(proposal_in_stride / stride))
                    layers = []
                    for _ in range(scale - 1):
                        layers += [
                            nn.ConvTranspose3d(d_model, d_model, kernel_size=2, stride=2),
                            LayerNorm3D(d_model),
                            nn.GELU(),
                        ]
                    layers.append(nn.ConvTranspose3d(d_model, d_model, kernel_size=2, stride=2))
                    self.enc_output_proj.append(nn.Sequential(*layers))

        self.encoder = None
        if add_transformer_encoder:
            encoder_layer = TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                normalize_before,
            )
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.0)
        normal_(self.level_embed)

        if hasattr(self.decoder, "_reset_parameters"):
            self.decoder._reset_parameters()

    def get_proposal_pos_embed(self, proposals, temperature=10000):
        """
        proposals: [N, L, K] where K = 6 (3D).
        Returns:   [N, L, K * (d_model//2)]
        """
        num_pos_feats = self.d_model // 2

        scale = 2 * math.pi
        device = proposals.device
        dtype = proposals.dtype

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        proposals = proposals * scale  # [N, L, K]
        pos = proposals[..., None] / dim_t  # [N, L, K, num_pos_feats]

        pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(2)  # [N, L, K * num_pos_feats]

        # 3D: K=6  → 6 * (C//2) = 3C
        return pos.to(dtype)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        if self.proposal_feature_levels > 1:
            memory, memory_padding_mask, spatial_shapes = self.expand_encoder_output(
                memory, memory_padding_mask, spatial_shapes
            )
        N_, S_, C_ = memory.shape

        # memory: [N_, S_, C_]
        # memory_padding_mask: [N_, S_]
        # spatial_shapes: List[(D_l, H_l, W_l)] for each level l

        proposals, _cur = [], 0
        for lvl, (D_, H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + D_ * H_ * W_)].view(N_, D_, H_, W_, 1)

            # valid_i: [B] — any valid pixel in each i-slice
            # Example:
            # (~mask_flatten_).any(dim=(2,3)): [N_, D_, 1]
            # sum(dim=1): [N_, 1]  (# of depth slices with any valid pixel)
            valid_D = (~mask_flatten_).any(dim=(2, 3)).sum(dim=1)  # [B] — any valid pixel in each D-slice
            valid_H = (~mask_flatten_).any(dim=(1, 3)).sum(dim=1)  # [B] — any valid pixel in each H-slice
            valid_W = (~mask_flatten_).any(dim=(1, 2)).sum(dim=1)  # [B] — any valid pixel in each W-slice

            # grid_z: [D_, H_, W_], grid_y: [D_, H_, W_], grid_x: [D_, H_, W_]
            grid_z, grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, D_ - 1, D_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
            )
            # grid: [D_, H_, W_, 3]
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), grid_z.unsqueeze(-1)], -1)

            # valid_*: [N_, 1]
            # valid_*.unsqueeze(-1): [N_, 1, 1]
            # cat along dim=1: [N_, 3, 1]
            # view -> [N_, 1, 1, 3]
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1), valid_D.unsqueeze(-1)], 1).view(
                N_, 1, 1, 1, 3
            )
            # grid.unsqueeze(0): [1, D_, H_, W_, 3]
            # expand -> [N_, D_, H_, W_, 3]
            # scale -> [N_, 1, 1, 3], broadcast over (D_, H_, W_)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1, -1) + 0.5) / scale
            whd = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            # proposal: [N_, D_*H_*W_, 6] (x,y,z,w,h,d)
            proposal = torch.cat((grid, whd), -1).view(N_, -1, 6)
            proposals.append(proposal)
            _cur += H_ * W_ * D_

        # output_proposals: [N_, \sum{D*H*W}, 6]
        output_proposals = torch.cat(proposals, 1)
        # output_proposals_valid: [N_, \sum{D*H*W}, 1]
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        max_shape = None
        return output_memory, output_proposals.to(memory.dtype), max_shape

    def expand_encoder_output(self, memory, memory_padding_mask, spatial_shapes):
        assert len(spatial_shapes) == 1, f"Recieved encoder output of shape: {spatial_shapes}!"

        bs, _, c = memory.shape
        d, h, w = spatial_shapes[0]

        # out_memory: [bs, c, d, h, w], out_memory_padding_mask: [bs, d, h, w]
        _out_memory = memory.view(bs, d, h, w, c).permute(0, 4, 1, 2, 3)
        _out_memory_padding_mask = memory_padding_mask.view(bs, d, h, w).to(_out_memory.device)

        out_memory, out_memory_padding_mask, out_spatial_shapes = [], [], []
        for i in range(self.proposal_feature_levels):
            # project the output memory to the desired feature dimension
            mem = self.enc_output_proj[i](_out_memory)
            mask = F.interpolate(
                _out_memory_padding_mask[None].float(),
                size=mem.shape[-3:],
                mode="nearest",
                align_corners=None,
            ).to(torch.bool)
            out_memory.append(mem)
            out_memory_padding_mask.append(mask.squeeze(0))
            out_spatial_shapes.append(mem.shape[-3:])

        # out_memory: [bs, \sum{d*h*w}, c]
        # out_memory_padding_mask: [bs, \sum{d*h*w}]
        out_memory = torch.cat([mem.flatten(2).transpose(1, 2) for mem in out_memory], dim=1)
        out_memory_padding_mask = torch.cat([mask.flatten(1) for mask in out_memory_padding_mask], dim=1)
        return out_memory, out_memory_padding_mask, out_spatial_shapes

    def get_reference_points(self, memory, mask_flatten, spatial_shapes):
        output_memory, output_proposals, max_shape = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )

        # HACK: two-stage Deformable DETR
        assert hasattr(self.decoder, "class_embed"), "Ensure that plainDETR has initial decoder class_embed!"
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_delta = None
        # NOTE: use final layer bbox_embed for enc_outputs_coord_unact
        enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

        topk_proposals = torch.topk(enc_outputs_class[..., 0], self.two_stage_num_proposals, dim=1)[1]
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 6))
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()
        return (
            reference_points,
            max_shape,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
        )

    def forward(self, srcs, masks, pos_embeds, query_embed=None, self_attn_mask=None):
        # TODO: we may remove this loop as we only have one feature level
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, d, h, w = src.shape
            spatial_shape = (d, h, w)
            spatial_shapes.append(spatial_shape)
            # src: [bs, c, d, h, w] -> [bs, d*h*w, c]
            src = src.flatten(2).transpose(1, 2)
            # mask: [bs, d, h, w] -> [bs, d*h*w]
            mask = mask.flatten(1)
            # pos_embed: [bs, c, d, h, w] -> [bs, d*h*w, c]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # lvl_pos_embed: add level embedding to pos_embed
            # lvl_pos_embed: [1, 1, d_model=c]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        # src_flatten: [bs, \sum{d*h*w}, c]
        # mask_flatten: [bs, \sum{d*h*w}]
        # lvl_pos_embed_flatten: [bs, \sum{d*h*w}, c]
        # spatial_shapes: [(d1, h1, w1), (d2, h2, w2), ...]
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        level_start_index = None
        valid_ratios = torch.stack([compute_unmasked_ratio(m) for m in masks], 1)

        if self.encoder is not None:
            memory = self.encoder(src_flatten, src_key_padding_mask=mask_flatten, pos=lvl_pos_embed_flatten)
        else:
            memory = src_flatten

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            (
                reference_points,  # (B, Q, Dr)
                max_shape,  # (D_valid, H_valid, W_valid)
                enc_outputs_class,  # (B, S, C)
                enc_outputs_coord_unact,
                enc_outputs_delta,  # (B, Q, Dr)
                output_proposals,  # (B, Q, Dr)
            ) = self.get_reference_points(memory, mask_flatten, spatial_shapes)

            # NOTE: reference_points and enc_outputs_class main outputs
            init_reference_out = reference_points
            # pos_trans_out: (B, Q, 2C) from proposal_pos_embed -> MLP -> LayerNorm
            # pos_trans_out = torch.zeros((bs, self.two_stage_num_proposals, 2 * c), device=init_reference_out.device)
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(reference_points)))

            # NOTE: either target queries and learned position embeddings are
            #       both obtained from reference points OR target queries
            #       are learned embeddings and query embeddings are obtained
            #       from reference points
            if not self.mixed_selection:
                # split query_embed into content and position embeddings
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
            else:
                # query_embed here is the content embed for deformable DETR
                # query_embed: (1, Q, C) -> (B, Q, C)
                tgt = query_embed.unsqueeze(0).expand(bs, -1, -1)
                # pos_trans_out: (B, Q, 2C) -> (B, Q, C) content, (B, Q, C) position
                query_embed, _ = torch.split(pos_trans_out, c, dim=2)
        else:
            # query_embed: (Q,  2C) -> query_embed, tgt: (Q, C) -> (B, Q, C)
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            # query_embed: (B, Q, C) -> reference_points: (B, Q, 3) in [0, 1]
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points
            max_shape = None

        # decoder
        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            lvl_pos_embed_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embed,
            mask_flatten,
            self_attn_mask,
            max_shape,
        )

        inter_references_out = inter_references
        if self.two_stage:
            return (
                hs,
                init_reference_out,
                inter_references_out,
                enc_outputs_class,
                enc_outputs_coord_unact,
                enc_outputs_delta,
                output_proposals,
                max_shape,
            )
        return hs, init_reference_out, inter_references_out, None, None, None, None, None


class TransformerReParam(Transformer):
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        if self.proposal_feature_levels > 1:
            memory, memory_padding_mask, spatial_shapes = self.expand_encoder_output(
                memory, memory_padding_mask, spatial_shapes
            )
        N_, S_, C_ = memory.shape

        proposals, _cur = [], 0
        for lvl, (D_, H_, W_) in enumerate(spatial_shapes):
            stride = self.proposal_tgt_strides[lvl]

            # grid_z: [D_, H_, W_], grid_y: [D_, H_, W_], grid_x: [D_, H_, W_]
            grid_z, grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, D_ - 1, D_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
            )
            # grid: [D_, H_, W_, 3] -> [N_, D_, H_, W_, 3]
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), grid_z.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1, -1) + 0.5) * stride
            # whd: [N_, D_, H_, W_, 3]
            whd = torch.ones_like(grid) * self.proposal_min_size * (2.0**lvl)
            # proposal: [N_, D_*H_*W_, 6] (x,y,z,w,h,d)
            proposal = torch.cat((grid, whd), -1).view(N_, -1, 6)
            proposals.append(proposal)
            _cur += H_ * W_ * D_

        # NOTE: all proposals are in absolute coord space given by largest feature map size
        #       imp. see how this relates to global decoder layer pos encodings etc.
        # output_proposals: [N_, \sum{D*H*W}, 6]
        output_proposals = torch.cat(proposals, 1)

        D_, H_, W_ = spatial_shapes[0]
        stride = self.proposal_tgt_strides[0]
        # mask_flatten: [N_, D_*H_*W_] -> [N_, D_, H_, W_, 1]
        mask_flatten_ = memory_padding_mask[:, : D_ * H_ * W_].view(N_, D_, H_, W_, 1)

        # valid_*: [N_] — any valid pixel in each i-slice
        valid_D = (~mask_flatten_).any(dim=(2, 3)).sum(dim=1) * stride  # [N_, 1] — any valid pixel in each D-slice
        valid_H = (~mask_flatten_).any(dim=(1, 3)).sum(dim=1) * stride  # [N_, 1] — any valid pixel in each H-slice
        valid_W = (~mask_flatten_).any(dim=(1, 2)).sum(dim=1) * stride  # [N_, 1] — any valid pixel in each W-slice

        # img_size: [N_, 1, 6] (W, H, D, W, H, D)
        img_size = torch.cat([valid_W, valid_H, valid_D, valid_W, valid_H, valid_D], dim=-1).unsqueeze(1)

        # output_proposals_valid: [N_, \sum{D*H*W}, 1]
        output_proposals_valid = ((output_proposals > 0.01 * img_size) & (output_proposals < 0.99 * img_size)).all(
            -1, keepdim=True
        )
        # output_proposals: [N_, \sum{D*H*W}, 6] broadcast against
        # memory_padding_mask: [N_, \sum{D*H*W}, 1]
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), max(D_, H_, W_) * stride)
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, max(D_, H_, W_) * stride)

        # output_memory: [N_, \sum{D*H*W}, C] broadcast against
        # memory_padding_mask: [N_, \sum{D*H*W}, 1] OR output_proposals_valid: [N_, \sum{D*H*W}, 1]
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        max_shape = (valid_D[:, None, :], valid_H[:, None, :], valid_W[:, None, :])
        return output_memory, output_proposals.to(memory.dtype), max_shape

    def get_reference_points(self, memory, mask_flatten, spatial_shapes):
        output_memory, output_proposals, max_shape = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )

        # HACK: two-stage Deformable DETR
        assert hasattr(self.decoder, "class_embed"), "Ensure that plainDETR has initial decoder class_embed!"
        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        enc_outputs_delta = self.decoder.bbox_embed[self.decoder.num_layers](output_memory)
        enc_outputs_coord_unact = box_xyzxyz_to_cxcyczwhd(delta2bbox(output_proposals, enc_outputs_delta, max_shape))

        topk_proposals = torch.topk(enc_outputs_class[..., 0], self.two_stage_num_proposals, dim=1)[1]
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 6))
        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact
        return (
            reference_points,
            max_shape,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
        )


def BUILD(cfg: Mapping[str, Any]) -> Transformer:
    """
    Factory for Transformer / TransformerReParam.
    Expects:
      d_model: int
      nheads: int
      num_feature_levels: int
      two_stage: bool
      two_stage_num_proposals: int (optional; if missing we use num_queries_* sum)
      norm_type: str
      decoder_type: str
      proposal_feature_levels: int
      proposal_in_stride: int
      proposal_tgt_strides: list[int]
      proposal_min_size: int
      global_decoder_args: Mapping
      add_transformer_encoder: bool
      dim_feedforward: int
      dropout: float
      activation: str
      normalize_before: bool
      num_encoder_layers: int

      # injected by PlainDETR.BUILD
      reparam: bool
      num_queries_one2one: int
      num_queries_one2many: int
      mixed_selection: bool
    """
    reparam = cfg.get("reparam")
    num_q1 = cfg.get("num_queries_one2one")
    num_qm = cfg.get("num_queries_one2many")

    two_stage_num_proposals = cfg.get(
        "two_stage_num_proposals",
        num_q1 + num_qm,
    )

    model_class = TransformerReParam if reparam else Transformer

    return model_class(
        d_model=cfg.get("d_model"),
        nhead=cfg.get("nheads"),
        num_feature_levels=cfg.get("num_feature_levels"),
        two_stage=cfg.get("two_stage"),
        two_stage_num_proposals=two_stage_num_proposals,
        mixed_selection=cfg.get("mixed_selection"),
        norm_type=cfg.get("norm_type"),
        decoder_type=cfg.get("decoder_type"),
        proposal_feature_levels=cfg.get("proposal_feature_levels"),
        proposal_in_stride=cfg.get("proposal_in_stride"),
        proposal_tgt_strides=cfg.get("proposal_tgt_strides"),
        proposal_min_size=cfg.get("proposal_min_size"),
        global_decoder_args=cfg.get("global_decoder_args"),
        # transformer encoder bits
        add_transformer_encoder=cfg.get("add_transformer_encoder"),
        dim_feedforward=cfg.get("dim_feedforward"),
        dropout=cfg.get("dropout"),
        activation=cfg.get("activation"),
        normalize_before=cfg.get("normalize_before"),
        num_encoder_layers=cfg.get("num_encoder_layers"),
    )
