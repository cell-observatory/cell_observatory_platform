import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from cell_observatory_platform.data.structures import box_xyzxyz_to_cxcyczwhd, delta2bbox
from cell_observatory_platform.models.layers.activation import get_activation
from cell_observatory_platform.models.layers.utils import inverse_sigmoid
from cell_observatory_platform.training.helpers import get_clones


class GlobalCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        rpe_hidden_dim=512,
        rpe_type="linear",
        feature_stride=16,
        reparam=False,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim**-0.5

        self.rpe_type = rpe_type
        self.feature_stride = feature_stride

        self.reparam = reparam

        self.cpb_mlp1 = self.build_cpb_mlp(2, rpe_hidden_dim, num_heads)
        self.cpb_mlp2 = self.build_cpb_mlp(2, rpe_hidden_dim, num_heads)
        self.cpb_mlp3 = self.build_cpb_mlp(2, rpe_hidden_dim, num_heads)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def build_cpb_mlp(self, in_dim, hidden_dim, out_dim):
        cpb_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True), nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim, bias=False)
        )
        return cpb_mlp

    def forward(
        self,
        query,
        reference_points,
        k_input_flatten,
        v_input_flatten,
        input_spatial_shapes,
        input_padding_mask=None,
    ):
        assert len(input_spatial_shapes) == 1, "GlobalCrossAttention only supports single-scale decoders."

        d, h, w = input_spatial_shapes[0]
        stride = self.feature_stride

        # reference points are in cxcyczwhd format, we convert to xyzxyz for rpe calculation
        # ref_pts: [B, nQ, 1, 6] -> [B, nQ, 1, 6]
        ref_pts = torch.cat(
            [
                reference_points[:, :, :, :3] - reference_points[:, :, :, 3:] / 2,
                reference_points[:, :, :, :3] + reference_points[:, :, :, 3:] / 2,
            ],
            dim=-1,
        )

        if not self.reparam:
            ref_pts[..., 0::3] *= w * stride
            ref_pts[..., 1::3] *= h * stride
            ref_pts[..., 2::3] *= d * stride

        pos_x = (
            torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=query.device)[None, None, :, None] * stride
        )  # 1, 1, w, 1
        pos_y = (
            torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=query.device)[None, None, :, None] * stride
        )  # 1, 1, h, 1
        pos_z = (
            torch.linspace(0.5, d - 0.5, d, dtype=torch.float32, device=query.device)[None, None, :, None] * stride
        )  # 1, 1, d, 1

        if self.rpe_type == "abs_log8":
            delta_x = ref_pts[..., 0::3] - pos_x  # B, nQ, w, 3
            delta_y = ref_pts[..., 1::3] - pos_y  # B, nQ, h, 3
            delta_z = ref_pts[..., 2::3] - pos_z  # B, nQ, d, 3
            delta_x = torch.sign(delta_x) * torch.log2(torch.abs(delta_x) + 1.0) / np.log2(8)
            delta_y = torch.sign(delta_y) * torch.log2(torch.abs(delta_y) + 1.0) / np.log2(8)
            delta_z = torch.sign(delta_z) * torch.log2(torch.abs(delta_z) + 1.0) / np.log2(8)
        elif self.rpe_type == "linear":
            delta_x = ref_pts[..., 0::3] - pos_x  # B, nQ, w, 3
            delta_y = ref_pts[..., 1::3] - pos_y  # B, nQ, h, 3
            delta_z = ref_pts[..., 2::3] - pos_z  # B, nQ, d, 3
        else:
            raise NotImplementedError

        rpe_x, rpe_y, rpe_z = (
            self.cpb_mlp1(delta_x.to(query.dtype)),
            self.cpb_mlp2(delta_y.to(query.dtype)),
            self.cpb_mlp3(delta_z.to(query.dtype)),
        )  # B, nQ, w/h/d, nheads

        # Lift to 3D and sum: (B, nQ, d, h, w, nheads)
        rpe = (
            rpe_x.unsqueeze(2).unsqueeze(2)  # (B, nQ, 1, 1, w, nheads)
            + rpe_y.unsqueeze(2).unsqueeze(4)  # (B, nQ, 1, h, 1, nheads)
            + rpe_z.unsqueeze(3).unsqueeze(4)  # (B, nQ, d, 1, 1, nheads)
        )

        B_, nQ, D, H_, W_, nheads = rpe.shape
        assert D == d and H_ == h and W_ == w and nheads == self.num_heads

        # Flatten spatial dims to match k_input_flatten's N = d*h*w
        rpe = rpe.view(B_, nQ, d * h * w, nheads)  # (B, nQ, N, nheads)
        rpe = rpe.permute(0, 3, 1, 2)  # (B, nheads, nQ, N)

        B_, N, C = k_input_flatten.shape
        assert N == d * h * w
        k = self.k(k_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(v_input_flatten).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B_, N, C = query.shape
        q = self.q(query).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn_mask = rpe
        if input_padding_mask is not None:
            input_padding_mask = input_padding_mask.to(attn_mask.device)
            attn_mask += input_padding_mask[:, None, None] * -100
        attn_mask = attn_mask.contiguous()  # to enable efficient attention

        x = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0,
            scale=self.scale,
        )

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_heads=8,
        norm_type="post_norm",
        rpe_hidden_dim=512,
        rpe_type="linear",
        feature_stride=16,
        reparam=False,
    ):
        super().__init__()

        self.norm_type = norm_type

        # global cross attention
        self.cross_attn = GlobalCrossAttention(
            d_model,
            n_heads,
            rpe_hidden_dim=rpe_hidden_dim,
            rpe_type=rpe_type,
            feature_stride=feature_stride,
            reparam=reparam,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation(activation)()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_pre(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_pos_embed,
        src_spatial_shapes,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        # self attention
        tgt2 = self.norm2(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt2.transpose(0, 1), attn_mask=self_attn_mask, need_weights=False
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)

        # global cross attention
        tgt2 = self.norm1(tgt)
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt2, query_pos),
            reference_points,
            self.with_pos_embed(src, src_pos_embed),
            src,
            src_spatial_shapes,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout4(tgt2)

        return tgt

    def forward_post(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_pos_embed,
        src_spatial_shapes,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), attn_mask=self_attn_mask, need_weights=False
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            self.with_pos_embed(src, src_pos_embed),
            src,
            src_spatial_shapes,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_pos_embed,
        src_spatial_shapes,
        src_padding_mask=None,
        self_attn_mask=None,
    ):
        if self.norm_type == "pre_norm":
            return self.forward_pre(
                tgt,
                query_pos,
                reference_points,
                src,
                src_pos_embed,
                src_spatial_shapes,
                src_padding_mask,
                self_attn_mask,
            )
        if self.norm_type == "post_norm":
            return self.forward_post(
                tgt,
                query_pos,
                reference_points,
                src,
                src_pos_embed,
                src_spatial_shapes,
                src_padding_mask,
                self_attn_mask,
            )


class GlobalDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        return_intermediate=False,
        look_forward_twice=False,
        d_model=256,
        norm_type="post_norm",
        reparam=False,
    ):
        super().__init__()

        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.look_forward_twice = look_forward_twice

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None
        self.reparam = reparam

        self.norm_type = norm_type
        if self.norm_type == "pre_norm":
            self.final_layer_norm = nn.LayerNorm(d_model)
        else:
            self.final_layer_norm = None

    def _reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_pos_embed,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        self_attn_mask=None,
        max_shape=None,
    ):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if self.reparam:
                # add level dim
                # reference_points_input: [B,Q,Dr] -> [B,Q,1,Dr]
                reference_points_input = reference_points[:, :, None]
            else:
                if reference_points.shape[-1] == 6:
                    # reference_points_input: (B, Q, 1, 6) * (B, 1, L, 6) -> (B, Q, L, 6)
                    reference_points_input = (
                        reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                    )
                else:
                    assert reference_points.shape[-1] == 3, "Last dim of reference_points should be 3 or 6."
                    # reference_points_input: (B, Q, 1, 3) * (B, 1, L, 3) -> (B, Q, L, 3)
                    reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                src_pos_embed,
                src_spatial_shapes,
                src_padding_mask,
                self_attn_mask,
            )

            if self.final_layer_norm is not None:
                output_after_norm = self.final_layer_norm(output)
            else:
                output_after_norm = output

            # HACK: iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output_after_norm)
                if reference_points.shape[-1] == 6:
                    if self.reparam:
                        new_reference_points = box_xyzxyz_to_cxcyczwhd(delta2bbox(reference_points, tmp, max_shape))
                    else:
                        new_reference_points = tmp + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()

                else:
                    # TODO: remove this branch?
                    if self.reparam:
                        raise NotImplementedError("reparam iterative refinement not implemented for this case!")

                    assert reference_points.shape[-1] == 3, "Last dim of reference_points should be 3 or 6."

                    new_reference_points = tmp
                    new_reference_points[..., :3] = tmp[..., :3] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output_after_norm)
                # NOTE: with look_forward_twice, we do not use reference_points (deattached)
                #       and hence output_after_norm contributes to new_reference_points
                #       and to tmp in final prediction which impacts gradients in both cases
                intermediate_reference_points.append(
                    new_reference_points if self.look_forward_twice else reference_points
                )

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output_after_norm, reference_points


def build_global_rpe_decomp_decoder(args):
    decoder_layer = GlobalDecoderLayer(
        d_model=args.hidden_dim,
        d_ffn=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        n_heads=args.nheads,
        norm_type=args.norm_type,
        rpe_hidden_dim=args.hidden_dim,
        # rpe_type=args.rpe_type,
        feature_stride=args.proposal_in_stride,
        reparam=args.reparam,
    )
    decoder = GlobalDecoder(
        decoder_layer,
        num_layers=args.dec_layers,
        return_intermediate=True,
        look_forward_twice=args.look_forward_twice,
        d_model=args.hidden_dim,
        norm_type=args.norm_type,
        reparam=args.reparam,
    )
    return decoder
