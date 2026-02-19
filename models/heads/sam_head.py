"""
Adapted from:
https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam/mask_decoder.py#
"""

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from cell_observatory_platform.models.layers.mlp import MLP
from cell_observatory_platform.models.layers.norm import LayerNorm3D
from cell_observatory_platform.models.heads.sam_decoder import TwoWayTransformer


class MaskDecoder(nn.Module):
    def __init__(
        self,
        input_fmt: str,
        mask_downsample_factor: int,
        transformer_dim: int,
        transformer_depth: int,
        transformer_num_heads: int,
        transformer_mlp_dim: int,
        transformer_activation: Type[nn.Module] = nn.ReLU,
        transformer_attention_downsample_rate: int = 2,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.
        """
        super().__init__()

        self.input_fmt = input_fmt
        self.transformer_dim = transformer_dim

        self.transformer = TwoWayTransformer(
            depth=transformer_depth,
            embedding_dim=transformer_dim,
            num_heads=transformer_num_heads,
            mlp_dim=transformer_mlp_dim,
            activation=transformer_activation,
            attention_downsample_rate=transformer_attention_downsample_rate,
            input_fmt=input_fmt,
        )

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.mask_downsample_factor = mask_downsample_factor
        if self.input_fmt == "TZYXC":
            assert mask_downsample_factor == 4, "Mask downsample factor must be 4."
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose3d(
                    transformer_dim, transformer_dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)
                ),
                LayerNorm3D(transformer_dim // 4),
                activation(),
                nn.ConvTranspose3d(
                    transformer_dim // 4, transformer_dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)
                ),
                activation(),
            )
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            if self.input_fmt == "TZYXC":
                self.conv_s0 = nn.Conv3d(
                    transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
                )
                self.conv_s1 = nn.Conv3d(
                    transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
                )
            else:
                raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # NOTE: When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            # multiple masks output
            if self.input_fmt == "TZYXC":
                masks = masks[:, 1:, :, :, :]
            else:
                raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            # based on stability score
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            # single mask output
            if self.input_fmt == "TZYXC":
                masks = masks[:, 0:1, :, :, :]
            else:
                raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            # idx for iou token
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        # output tokens: [b, N_output_tokens, c]
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0], "image_embeddings and tokens must have the same batch size"
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert (
            image_pe.size(0) == 1
        ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        
        if self.input_fmt == "TZYXC":
            b, c, z, y, x = src.shape
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        pos_src = pos_src.to(src.dtype)
        tokens = tokens.to(src.dtype)

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        if self.input_fmt == "TZYXC":
            # NOTE: src: [B, N, C] -> [B, C, Z, Y, X]
            src = src.transpose(1, 2).view(b, c, z, y, x)
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        if self.input_fmt == "TZYXC":
            b, c, z, y, x = upscaled_embedding.shape
            masks = (hyper_in @ upscaled_embedding.view(b, c, z * y * x)).view(b, -1, z, y, x)
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # NOTE: Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        # mask_logits: [B, N, ...] -> [B, N, total_num_pixels]
        if self.input_fmt == "TZYXC":
            mask_logits = mask_logits.flatten(-3)
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        # best_scores_inds: [B, N] -> [B]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        # best_multimask_logits: [B, N, ...] -> [B, 1, ...]
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        # best_multimask_iou_scores: [B, N] -> [B, 1]
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        # is_stable: [B,1]
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        if self.input_fmt == "TZYXC":
            # NOTE: mask_logits_out: [B, 1, Z, Y, X]
            mask_logits_out = torch.where(
                is_stable[..., None, None, None].expand_as(singlemask_logits),
                singlemask_logits,
                best_multimask_logits,
            )
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out