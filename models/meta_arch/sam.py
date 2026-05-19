"""
Adapted from:
https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/sam2_base.py
https://github.com/facebookresearch/sam2/blob/main/training/model/sam2.py
"""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Literal, Tuple

import numpy as np

import torch
import torch.distributed
import torch.nn.functional as F

from hydra.utils import get_method
from torch.nn.init import trunc_normal_

from cell_observatory_platform.models.layers.mlp import MLP
from cell_observatory_platform.models.heads.sam_head import MaskDecoder
from cell_observatory_platform.models.layers.activation import get_activation
from cell_observatory_platform.models.layers.memory_encoders import MemoryEncoder
from cell_observatory_platform.models.layers.prompt_encoders import PromptEncoder
from cell_observatory_platform.models.layers.patch_embeddings import calc_num_patches
from cell_observatory_platform.models.layers.attention import MemoryAttention, RopeAttention
from cell_observatory_platform.models.layers.positional_encoding import PositionalEmbeddingSinCos
from cell_observatory_platform.models.layers.utils import (
    select_closest_cond_frames,
    sample_random_points_from_errors,
    sample_one_point_from_error_center,
    get_next_point,
    sample_box_points,
    concat_points
)
from cell_observatory_platform.models.ops.point_sampling import (
    gt_masks_from_labelmap,
    sample_box_points_from_boxes,
    sample_prompt_point_from_labelmap,
)
from cell_observatory_platform.data.structures import (
    box_volume,
    is_box_near_crop_edge_3d,
    masks_to_boxes_v2,
    nms_3d,
    uncrop_boxes_3d,
    uncrop_masks_3d,
    uncrop_points_3d,
)
from cell_observatory_platform.inference.amg import (
    MaskData,
    batch_iterator,
    build_all_layer_point_grids_3d,
    calculate_stability_score_3d,
    generate_crop_boxes_3d,
    remove_small_regions_3d,
)

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM2Base(torch.nn.Module):
    def __init__(
        self,
        input_fmt: str,
        input_shape: tuple[int, int, int],
        patch_shape: tuple[int, int, int],
        criterion,
        image_encoder,
        memory_attention,
        memory_encoder,
        sam_prompt_encoder,
        sam_mask_decoder,
        mask_downsample_factor: int = 4,
        num_maskmem=7,  # default 1 input frame + 6 previous frames
        sigmoid_scale_for_mem_enc=1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc=0.0,  # bias factor for mask sigmoid prob
        # During evaluation, whether to binarize the sigmoid mask logits on interacted frames with clicks
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,  # on frames with mask input, whether to directly output the input mask without using a SAM prompt encoder + mask decoder
        # The maximum number of conditioning frames to participate in the memory attention (-1 means no limit; if there are more conditioning frames than this limit,
        # we only cross-attend to the temporally closest `max_cond_frames_in_attn` conditioning frames in the encoder when tracking each frame). This gives the model
        # a temporal locality when handling a large number of annotated frames (since closer frames should be more important) and also avoids GPU OOM.
        max_cond_frames_in_attn=-1,
        # on the first frame, whether to directly add the no-memory embedding to the image feature
        # (instead of using the transformer encoder)
        directly_add_no_mem_embed=False,
        # whether to use high-resolution feature maps in the SAM mask decoder
        use_high_res_features_in_sam=False,
        # whether to output multiple (3) masks for the first click on initial conditioning frames
        multimask_output_in_sam=False,
        # the minimum and maximum number of clicks to use multimask_output_in_sam (only relevant when `multimask_output_in_sam=True`;
        # default is 1 for both, meaning that only the first click gives multimask output; also note that a box counts as two points)
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        # whether to also use multimask output for tracking (not just for the first click on initial conditioning frames; only relevant when `multimask_output_in_sam=True`)
        multimask_output_for_tracking=False,
        # Whether to use multimask tokens for obj ptr; Only relevant when both
        # use_obj_ptrs_in_encoder=True and multimask_output_for_tracking=True
        use_multimask_token_for_obj_ptr: bool = False,
        # whether to use sigmoid to restrict ious prediction to [0-1]
        iou_prediction_use_sigmoid=False,
        # The memory bank's temporal stride during evaluation (i.e. the `r` parameter in XMem and Cutie; XMem and Cutie use r=5).
        # For r>1, the (self.num_maskmem - 1) non-conditioning memory frames consist of
        # (self.num_maskmem - 2) nearest frames from every r-th frames, plus the last frame.
        memory_temporal_stride_for_eval=1,
        # whether to apply non-overlapping constraints on the object masks in the memory encoder during evaluation (to avoid/alleviate superposing masks)
        non_overlap_masks_for_mem_enc=False,
        # whether to cross-attend to object pointers from other frames (based on SAM output tokens) in the encoder
        use_obj_ptrs_in_encoder=False,
        # the maximum number of object pointers from other frames in encoder cross attention (only relevant when `use_obj_ptrs_in_encoder=True`)
        max_obj_ptrs_in_encoder=16,
        # whether to add temporal positional encoding to the object pointers in the encoder (only relevant when `use_obj_ptrs_in_encoder=True`)
        add_tpos_enc_to_obj_ptrs=True,
        # whether to add an extra linear projection layer for the temporal positional encoding in the object pointers to avoid potential interference
        # with spatial positional encoding (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        proj_tpos_enc_in_obj_ptrs=False,
        # whether to use signed distance (instead of unsigned absolute distance) in the temporal positional encoding in the object pointers
        # (only relevant when both `use_obj_ptrs_in_encoder=True` and `add_tpos_enc_to_obj_ptrs=True`)
        use_signed_tpos_enc_to_obj_ptrs=False,
        # whether to only attend to object pointers in the past (before the current frame) in the encoder during evaluation
        # (only relevant when `use_obj_ptrs_in_encoder=True`; this might avoid pointer information too far in the future to distract the initial tracking)
        only_obj_ptrs_in_the_past_for_eval=False,
        # Whether to predict if there is an object in the frame
        pred_obj_scores: bool = False,
        # Whether to use an MLP to predict object scores
        pred_obj_scores_mlp: bool = False,
        # Only relevant if pred_obj_scores=True and use_obj_ptrs_in_encoder=True;
        # Whether to have a fixed no obj pointer when there is no object present
        # or to use it as an additive embedding with obj_ptr produced by decoder
        fixed_no_obj_ptr: bool = False,
        # Soft no object, i.e. mix in no_obj_ptr softly,
        # hope to make recovery easier if there is a mistake and mitigate accumulation of errors
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        # add no obj embedding to spatial frames
        no_obj_embed_spatial: bool = False,
        # disable memory encoder
        disable_memory_encoder: bool = False,
        # Skip the trilinear upsample from low-res (stride 16) to full-res masks
        # inside `_forward_sam_heads`. Saves ~16^3 = 4096x the multimask
        # tensor memory at the cost of low-resolution masks flowing into:
        #   - memory encoder (downsamples by 4x; coarser inputs reduce quality)
        #   - correction-prompt FP/FN region calculation
        #   - eval output (`pred_masks_high_res`)
        # Safe only when:
        #   - the labelmap loss path is active (criterion already consumes
        #     `multistep_pred_multimasks` not `_high_res`),
        #   - and the memory encoder + correction sampling either also run on
        #     low-res inputs or are disabled (e.g. single-frame static-image
        #     training with `num_correction_pt_per_frame=0` and
        #     `disable_memory_encoder=True` + `num_frames=1`).
        # Defaults to False so existing configs retain the high-res upsample.
        skip_high_res_upsample: bool = False,
    ):
        super().__init__()

        self.input_fmt = input_fmt
        self.input_shape = input_shape
        self.patch_shape = patch_shape
        _, token_shape = calc_num_patches(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            patch_shape=patch_shape,
        )
        if self.input_fmt == "TZYXC":
            t, z, y, x, c = token_shape
            self.token_shape = [z, y, x]
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        # Part 0: criterion
        self.criterion = criterion

        # Part 1: the image backbone
        self.image_encoder = image_encoder
        # Use level 0, 1, 2 for high-res setting, or just level 2 for the default setting
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            # A conv layer to downsample the mask prompt to stride 4 (the same stride as
            # low-res SAM mask logits) and to change its scales from 0~1 to SAM logit scale,
            # so that it can be fed into the SAM mask decoder to generate a pointer.
            if self.input_fmt == "TZYXC":
                self.mask_downsample = torch.nn.Conv3d(1, 1, kernel_size=4, stride=4)
            else:
                raise ValueError(f"Input format {self.input_fmt} not supported yet.")
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs  # these options need to be used together
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval

        # Part 2: memory attention to condition current frame's visual features
        # with memories (and obj ptrs) from past frames
        self.memory_attention = memory_attention
        # NOTE: see sam_backbone.py for the definition of backbone_embed_dims
        self.hidden_dim = image_encoder.backbone_embed_dims[-1]

        # Part 3: memory encoder for the previous frame's outputs
        self.memory_encoder = memory_encoder
        self.mem_dim = self.hidden_dim
        if hasattr(self.memory_encoder, "out_proj") and hasattr(
            self.memory_encoder.out_proj, "weight"
        ):
            # if there is compression of memories along channel dim
            self.mem_dim = self.memory_encoder.out_proj.weight.shape[0]
        self.num_maskmem = num_maskmem  # Number of memories accessible
        # Temporal encoding of the memories
        self.maskmem_tpos_enc = torch.nn.Parameter(
            torch.zeros(num_maskmem, 1, 1, self.mem_dim)
        )
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.no_mem_embed, std=0.02)
        trunc_normal_(self.no_mem_pos_enc, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        # Apply sigmoid to the output raw mask logits (to turn them from
        # range (-inf, +inf) to range (0, 1)) before feeding them into the memory encoder
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        # On frames with mask input, whether to directly output the input mask without
        # using a SAM prompt encoder + mask decoder
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid

        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.soft_no_obj_ptr = soft_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores, "pred_obj_scores must be True when fixed_no_obj_ptr is True"
            assert self.use_obj_ptrs_in_encoder, "use_obj_ptrs_in_encoder must be True when fixed_no_obj_ptr is True"
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, self.mem_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)

        # Part 4: SAM-style prompt encoder and mask decoder (built externally via BUILD)
        self.mask_downsample_factor = mask_downsample_factor
        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_prompt_encoder = sam_prompt_encoder
        self.sam_mask_decoder = sam_mask_decoder
        # Projection layers for object pointers
        if self.use_obj_ptrs_in_encoder:
            # a linear projection on SAM output tokens to turn them into object pointers
            self.obj_ptr_proj = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, 3
                )
        else:
            self.obj_ptr_proj = torch.nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            # a linear projection on temporal positional encoding in object pointers to
            # avoid potential interference with spatial positional encoding
            self.obj_ptr_tpos_proj = torch.nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = torch.nn.Identity()

        self.max_cond_frames_in_attn = max_cond_frames_in_attn

        # disable memory encoder
        self.disable_memory_encoder = disable_memory_encoder
        self.skip_high_res_upsample = skip_high_res_upsample

    @property
    def device(self):
        return next(self.parameters()).device

    @abstractmethod
    def forward(self, data_sample: dict):
        raise NotImplementedError

    @abstractmethod
    def _init_model_weights(self, buffer_device: str | None = None):
        raise NotImplementedError

    def _forward_sam_heads(
        self,
        backbone_features,
        point_inputs=None,
        mask_inputs=None,
        high_res_features=None,
        multimask_output=False,
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, D, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 3] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y, z) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, D*scale_factor, H*scale_factor, W*scale_factor] 
          shape, float or bool, with the same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*D, 4*H, 4*W] and [B, C, 2*D, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, D*4, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, D*16, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, D*4, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, D*16, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.size(0)
        device = backbone_features.device
        if self.input_fmt == "TZYXC":
            D, H, W = self.token_shape
            assert backbone_features.size(1) == self.sam_prompt_embed_dim
            assert backbone_features.size(2) == D and backbone_features.size(3) == H and backbone_features.size(4) == W
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        # a) Handle point prompts
        if point_inputs is not None:
            sam_point_coords = point_inputs["point_coords"]
            sam_point_labels = point_inputs["point_labels"]
            assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B, \
                f"sam_point_coords.size(0) = {sam_point_coords.size(0)}, sam_point_labels.size(0) = {sam_point_labels.size(0)}"
        else:
            # If no points are provide, pad with an empty point (with label -1)
            sam_point_coords = torch.zeros(B, 1, 3, device=device)
            sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

        # b) Handle mask prompts
        if mask_inputs is not None:
            # If mask_inputs is provided, downsize it into low-res mask input if needed
            # and feed it as a dense mask prompt into the SAM mask encoder
            if self.input_fmt == "TZYXC":
                # mask_inputs: [B, 1, D, H, W]
                assert len(mask_inputs.shape) == 5 and mask_inputs.shape[:2] == (B, 1)
                if mask_inputs.shape[-3:] != self.sam_prompt_encoder.mask_input_size:
                    sam_mask_prompt = F.interpolate(
                        mask_inputs.float(),
                        size=self.sam_prompt_encoder.mask_input_size,
                        align_corners=False,
                        mode="trilinear",
                        antialias=True,  # use antialias for downsampling
                    )
                else:
                    sam_mask_prompt = mask_inputs
            else:
                raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
        else:
            # Otherwise, simply feed None (and SAM's prompt encoder will add
            # a learned `no_mask_embed` to indicate no mask input in this case).
            sam_mask_prompt = None

        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=sam_mask_prompt,
        )
        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.sam_mask_decoder(
            image_embeddings=backbone_features,
            image_pe=self.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )
        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            if self.input_fmt == "TZYXC":
                # Mask used for spatial memories is always a *hard* choice between obj and no obj,
                # consistent with the actual mask prediction
                low_res_multimasks = torch.where(
                    is_obj_appearing[:, None, None, None],
                    low_res_multimasks,
                    NO_OBJ_SCORE,
                )
            else:
                raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        if self.input_fmt == "TZYXC":
            if self.skip_high_res_upsample:
                # Alias to low-res; downstream consumers (criterion via the
                # labelmap path use normalized coords; memory encoder /
                # correction sampling must tolerate the coarser resolution).
                high_res_multimasks = low_res_multimasks
            else:
                high_res_multimasks = F.interpolate(
                    low_res_multimasks,
                    # TODO: less restrictive to upsample based on real image size
                    size=tuple(self.input_shape[1:4]),
                    mode="trilinear",
                    align_corners=False,
                )
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.obj_ptr_proj(sam_output_token)
        if self.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def _use_mask_as_output(self, backbone_features, high_res_features, mask_inputs, downscale_factor: int = 4):
        """
        Directly turn binary `mask_inputs` into a output mask logits without using SAM.
        (same input and output shapes as in _forward_sam_heads above).
        """
        # Use -10/+10 as logits for neg/pos pixels (very close to 0/1 in prob after sigmoid).
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.float()
        high_res_masks = mask_inputs_float * out_scale + out_bias
        if self.input_fmt == "TZYXC":
            low_res_masks = F.interpolate(
                high_res_masks,
                size=(
                    high_res_masks.size(-3) // downscale_factor, 
                    high_res_masks.size(-2) // downscale_factor, 
                    high_res_masks.size(-1) // downscale_factor
                ),
                align_corners=False,
                mode="trilinear",
                antialias=True,  # use antialias for downsampling
            )
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
        # a dummy IoU prediction of all 1's under mask input
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1).float()
        if not self.use_obj_ptrs_in_encoder:
            # all zeros as a dummy object pointer (of shape [B, C])
            obj_ptr = torch.zeros(
                mask_inputs.size(0), self.hidden_dim, device=mask_inputs.device
            )
        else:
            # produce an object pointer using the SAM decoder from the mask input
            _, _, _, _, _, obj_ptr, _ = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.mask_downsample(mask_inputs_float),
                high_res_features=high_res_features,
            )
        # In this method, we are treating mask_input as output, e.g. using it directly to create spatial mem;
        # Below, we follow the same design axiom to use mask_input to decide if obj appears or not instead of relying
        # on the object_scores from the SAM decoder.
        is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)
        is_obj_appearing = is_obj_appearing[..., None]
        lambda_is_obj_appearing = is_obj_appearing.float()
        object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
        if self.pred_obj_scores:
            if self.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.no_obj_ptr

        return (
            low_res_masks,
            high_res_masks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )

    def forward_image(self, data_sample):
        """Get the image feature on the input batch."""
        backbone_out = self.image_encoder(data_sample)
        if self.use_high_res_features_in_sam:
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"]), "backbone_fpn and vision_pos_enc must have the same length"
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels, "backbone_fpn must have at least num_feature_levels levels"

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        if self.input_fmt == "TZYXC":
            feat_sizes = [(x.shape[-3], x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
        # flatten NxCxDxHxW to DHWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _get_1d_sine_pe(self, pos_inds, dim, temperature=10000, out_dtype=None):
        """
        Get 1D sine positional embedding as in the original Transformer paper.
        """
        pe_dim = dim // 2
        dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
        dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

        pos_embed = pos_inds.unsqueeze(-1) / dim_t
        pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
        return pos_embed.to(out_dtype)

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        if self.input_fmt == "TZYXC":
            D, H, W = feat_sizes[-1]
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
        device = current_vision_feats[-1].device
        
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            if self.input_fmt == "TZYXC":
                pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, D, H, W)
            else:
                raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0, "cond_frame_outputs must not be empty"
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // stride) * stride
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * stride
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                # feats: [B, C, D, H, W] -> [B, C, D*H*W] -> [D*H*W, B, C]
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"],
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list).to(
                            device=device, non_blocking=True
                        )
                        obj_pos = self._get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim, out_dtype=current_vision_feats[-1].dtype)
                        base_dtype = current_vision_feats[-1].dtype
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos.to(base_dtype))
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                if self.input_fmt == "TZYXC":
                    pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, D, H, W)
                else:
                    raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        if self.input_fmt == "TZYXC":
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, D, H, W)
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
        return pix_feat_with_mem

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        if self.input_fmt == "TZYXC":
            D, H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
            # top-level feature, (D*H*W)BC => BCDHW
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, D, H, W)
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        if self.non_overlap_masks_for_mem_enc and not self.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )

        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)

        # apply scale and bias terms to the sigmoid probabilities
        if self.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
        if self.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        maskmem_out = self.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            if self.input_fmt == "TZYXC":
                maskmem_features += (
                    1 - is_obj_appearing[..., None, None, None]
                ) * self.no_obj_embed_spatial[..., None, None, None].expand(
                    *maskmem_features.shape
                )
            else:
                raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        return maskmem_features, maskmem_pos_enc

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (DHW)BC => BCDHW
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            # fused the visual feature with previous memory features in the memory bank
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
            )
            # apply SAM-style segmentation head
            # here we might feed previously predicted low-res SAM mask logits into the SAM mask decoder,
            # e.g. in demo where such logits come from earlier interaction instead of correction sampling
            # (in this case, any `mask_inputs` shouldn't reach here as they are sent to _use_mask_as_output instead)
            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )

        return current_out, sam_outputs, high_res_features, pix_feat

    def _encode_memory_in_output(
        self,
        current_vision_feats,
        feat_sizes,
        point_inputs,
        run_mem_encoder,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        if run_mem_encoder and self.num_maskmem > 0:
            high_res_masks_for_mem_enc = high_res_masks
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks_for_mem_enc,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        if not self.training:
            # Only add this in inference (to avoid unused param in activation checkpointing;
            # it's mainly used in the demo to encode spatial memories w/ consolidated masks)
            current_out["object_score_logits"] = object_score_logits

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )

        return current_out

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether to use multimask output in the SAM head."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        multimask_output = (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
        )
        return multimask_output

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`]
        if self.input_fmt == "TZYXC":
            batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None, None]
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks


class SAM2(SAM2Base):
    def __init__(
        self,
        criterion,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        sam_prompt_encoder=None,
        sam_mask_decoder=None,
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        # if it is greater than 1, we interactive point sampling in the 1st frame and other randomly selected frames
        num_frames_to_correct_for_train=1,  # default: only iteratively sample on first frame
        num_frames_to_correct_for_eval=1,  # default: only iteratively sample on first frame
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        # how many frames to use as initial conditioning frames (for both point input and mask input; the first frame is always used as an initial conditioning frame)
        # - if `rand_init_cond_frames` below is True, we randomly sample 1~num_init_cond_frames initial conditioning frames
        # - otherwise we sample a fixed number of num_init_cond_frames initial conditioning frames
        # note: for point input, we sample correction points on all such initial conditioning frames, and we require that `num_frames_to_correct` >= `num_init_cond_frames`;
        # these are initial conditioning frames because as we track the video, more conditioning frames might be added
        # when a frame receives correction clicks under point input if `add_all_frames_to_correct_as_cond=True`
        num_init_cond_frames_for_train=1,  # default: only use the first frame as initial conditioning frame
        num_init_cond_frames_for_eval=1,  # default: only use the first frame as initial conditioning frame
        rand_init_cond_frames_for_train=True,  # default: random 1~num_init_cond_frames_for_train cond frames (to be constent w/ previous TA data loader)
        rand_init_cond_frames_for_eval=False,
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=False,
        # how many additional correction points to sample (on each frame selected to be corrected)
        # note that the first frame receives an initial input click (in addition to any correction clicks)
        num_correction_pt_per_frame=7,
        # method for point sampling during evaluation
        # "uniform" (sample uniformly from error region) or "center" (use the point with the largest distance to error region boundary)
        # default to "center" to be consistent with evaluation in the SAM paper
        pt_sampling_for_eval="center",
        # During training, we optionally allow sampling the correction points from GT regions
        # instead of the prediction error regions with a small probability. This might allow the
        # model to overfit less to the error regions in training datasets
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        # whether to forward image features per frame (as it's being tracked) during evaluation, instead of forwarding image features
        # of all frames at once. This avoids backbone OOM errors on very long videos in evaluation, but could be slightly slower.
        forward_backbone_per_frame_for_eval=False,
        freeze_image_encoder=False,
        # --- Automatic mask generation args (inference only) ---
        points_per_side: int = 16,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.92,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        use_m2m: bool = False,
        multimask_output_for_predict: bool = True,
        min_mask_region_area: int = 0,
        debug: bool = False,
        buffer_device: str = "cuda",
        **kwargs,
    ):
        super().__init__(
            criterion=criterion,
            image_encoder=image_encoder,
            memory_attention=memory_attention,
            memory_encoder=memory_encoder,
            sam_prompt_encoder=sam_prompt_encoder,
            sam_mask_decoder=sam_mask_decoder,
            **kwargs,
        )

        self.debug = debug

        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval

        # Point sampler and conditioning frames
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        if prob_to_use_pt_input_for_train > 0 or prob_to_use_pt_input_for_eval > 0:
            logging.info(
                f"Training with points (sampled from masks) as inputs with p={prob_to_use_pt_input_for_train}"
            )
            assert num_frames_to_correct_for_train >= num_init_cond_frames_for_train
            assert num_frames_to_correct_for_eval >= num_init_cond_frames_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        
        # Initial multi-conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        
        # A random number generator with a fixed initial seed across GPUs
        self.rng = np.random.default_rng(seed=42)

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        # # Inference-time config
        self._amg_points_per_side = points_per_side
        self._amg_points_per_batch = points_per_batch
        self._amg_pred_iou_thresh = pred_iou_thresh
        self._amg_stability_score_thresh = stability_score_thresh
        self._amg_stability_score_offset = stability_score_offset
        self._amg_mask_threshold = mask_threshold
        self._amg_box_nms_thresh = box_nms_thresh
        self._amg_crop_n_layers = crop_n_layers
        self._amg_crop_nms_thresh = crop_nms_thresh
        self._amg_crop_overlap_ratio = crop_overlap_ratio
        self._amg_crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self._amg_use_m2m = use_m2m
        self._amg_multimask_output = multimask_output_for_predict
        self._amg_min_mask_region_area = min_mask_region_area

        self._amg_point_grids = build_all_layer_point_grids_3d(
            points_per_side,
            crop_n_layers,
            crop_n_points_downscale_factor,
        )

        self._init_model_weights(buffer_device=buffer_device)

    def _init_model_weights(self, buffer_device: str | None = None):
        # Weight init for SAM2 submodules happens inside their respective __init__ methods (if at all),
        # following the reference implementation:
        #   - MemoryAttention
        #   - SAMBackbone (implicitly via backbone build)
        #   - MaskDecoder
        #   - PromptEncoder
        #   - MemoryEncoder
        #   - SAM2Base
        #   - SAM2
        for mod in self.modules():
            if isinstance(mod, RopeAttention):
                mod.init_rope_parameters(device=buffer_device)

    def get_param_groups(self, weight_decay: float, **kwargs) -> list[dict]:
        """Standard decay/no-decay split for SAM2.
        TODO: consider more options such as layer-wise decay, etc.
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or "bias" in name:
                no_decay.append(p)
            else:
                decay.append(p)

        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def forward(self, data_sample: dict):
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(data_sample)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, data_sample)
        previous_stages_out = self.forward_tracking(backbone_out, data_sample)
        # Pass the structured target view (labelmaps, instance_ids, valid,
        # presence_t, boxes, masks, ...) directly to the criterion. The
        # criterion picks the labelmap-native or legacy dense-mask path based
        # on which fields the view exposes.
        target_view = data_sample["metainfo"]["targets"]
        loss = self.criterion(previous_stages_out, target_view)
        return loss, previous_stages_out

    def _prepare_backbone_features_per_frame(self, data_sample, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        # image = img_batch[unique_img_ids]
        image = data_sample["data_tensor"][unique_img_ids]
        backbone_out = self.forward_image({"data_tensor": image})
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    def prepare_prompt_inputs(self, backbone_out, data_sample, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts. Optionally, we allow tracking from
        a custom `start_frame_idx` to the end of the video (for evaluation purposes).
        """
        # Load the ground-truth masks on all frames (so that we can later
        # sample correction points from them)
        # gt_masks_per_frame = {
        #     stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
        #     for stage_id, targets in enumerate(input.find_targets)
        # }
        data_views = data_sample["metainfo"]["targets"]

        # Per-frame GT masks. Preferred path: derive from the labelmap target
        # view so the preprocessor can eventually stop emitting dense per-row
        # masks entirely. Falls back to the eager `masks` field for configs
        # that have not migrated.
        has_labelmap = (
            "labelmaps" in data_views
            and "instance_ids" in data_views
            and "img_ids" in data_views
        )
        if has_labelmap:
            labelmaps = data_views["labelmaps"]
            gt_masks_per_frame = {
                t: gt_masks_from_labelmap(
                    labelmap=labelmaps,
                    img_ids=data_views["img_ids"][t],
                    instance_ids=data_views["instance_ids"][t],
                )
                for t in range(int(data_views["num_frames"]))
            }
        else:
            gt_masks_per_frame = {
                t: m.unsqueeze(1)          # -> (N_obj, 1, Z, Y, X)
                for t, m in enumerate(data_views["masks"])
            }
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = data_views["num_frames"]
        backbone_out["num_frames"] = num_frames

        # Randomly decide whether to use point inputs or mask inputs
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        if num_frames == 1:
            # here we handle a special case for mixing video + SAM on image training,
            # where we force using point input for the SAM task on static images
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        assert num_init_cond_frames >= 1, "num_init_cond_frames must be >= 1"
        # (here `self.rng.random()` returns value in range 0.0 <= X < 1.0)
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        if rand_init_cond_frames and num_init_cond_frames > 1:
            # randomly select 1 to `num_init_cond_frames` frames as initial conditioning frames
            num_init_cond_frames = self.rng.integers(
                1, num_init_cond_frames, endpoint=True
            )
        if (
            use_pt_input
            and rand_frames_to_correct
            and num_frames_to_correct > num_init_cond_frames
        ):
            # randomly select `num_init_cond_frames` to `num_frames_to_correct` frames to sample
            # correction clicks (only for the case of point input)
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames, num_frames_to_correct, endpoint=True
            )
        backbone_out["use_pt_input"] = use_pt_input

        # Sample initial conditioning frames
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]  # starting frame
        else:
            # starting frame + randomly selected remaining frames (without replacement)
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        # Prepare mask or point inputs on initial conditioning frames
        backbone_out["mask_inputs_per_frame"] = {}  # {frame_idx: <input_masks>}
        backbone_out["point_inputs_per_frame"] = {}  # {frame_idx: <input_points>}
        for t in init_cond_frames:
            if not use_pt_input:
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
            else:
                # During training # P(box) = prob_to_use_pt_input * prob_to_use_box_input
                use_box_input = self.rng.random() < prob_to_use_box_input
                if use_box_input:
                    target_boxes_t = data_views.get("boxes")
                    box_format = data_views.get("box_format")
                    if target_boxes_t is not None and box_format is not None:
                        # Labelmap target view supplies per-row boxes already;
                        # avoid recomputing them from dense masks. Image shape
                        # comes from the flat labelmaps tensor (B*T, Z, Y, X).
                        Z, Y, X = data_views["labelmaps"].shape[-3:]
                        points, labels = sample_box_points_from_boxes(
                            boxes=target_boxes_t[t],
                            box_format=box_format,
                            image_shape=(Z, Y, X),
                            valid=data_views["valid"][t],
                        )
                    else:
                        points, labels = sample_box_points(
                            input_fmt=self.input_fmt,
                            time_separable=True,
                            masks=gt_masks_per_frame[t],
                        )
                else:
                    # Initial-prompt site: sample one click from the GT mask.
                    # When the labelmap target view is available, route through
                    # the labelmap-first helper so we do not depend on
                    # `data_views["masks"]`. Training: GPU-friendly uniform
                    # sampling. Eval: opt into exact scipy EDT when method=center.
                    method = "uniform" if self.training else self.pt_sampling_for_eval
                    if has_labelmap:
                        points, labels = sample_prompt_point_from_labelmap(
                            labelmap=data_views["labelmaps"],
                            img_ids=data_views["img_ids"][t],
                            instance_ids=data_views["instance_ids"][t],
                            pred_masks=None,
                            input_fmt=self.input_fmt,
                            time_separable=True,
                            method=method,
                            exact_edt_for_eval=not self.training,
                        )
                    else:
                        points, labels = get_next_point(
                            input_fmt=self.input_fmt,
                            time_separable=True,
                            gt_masks=gt_masks_per_frame[t],
                            pred_masks=None,
                            method=method,
                        )

                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][t] = point_inputs

        # Sample frames where we will add correction clicks on the fly
        # based on the error between prediction and ground-truth masks
        if not use_pt_input:
            # no correction points will be sampled when using mask inputs
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            assert num_frames_to_correct > num_init_cond_frames
            # initial cond frame + randomly selected remaining frames (without replacement)
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    backbone_out["frames_not_in_init_cond"], extra_num, replace=False
                ).tolist()
            )
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt

        return backbone_out

    def forward_tracking(self, backbone_out, data_sample: dict, return_dict=False):
        """Forward video tracking on each frame (and sample correction clicks)."""
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (DHW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        # Starting the stage loop
        data_views = data_sample["metainfo"]["targets"]
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]

        run_mem_encoder = not (self.disable_memory_encoder and num_frames == 1)
        
        # first process all the initial conditioning frames to encode them as memory,
        # and then conditioning on them to track the remaining frames
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        for stage_id in processing_order:
            # Get the image features for the current frames
            # img_ids = input.find_inputs[stage_id].img_ids
            # img_ids = input.flat_obj_to_img_idx[stage_id]
            img_ids = data_views["img_ids"][stage_id]
            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if they are already computed).
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Otherwise, compute the image features on the fly for the given img_ids
                # (this might be used for evaluation on long videos to avoid backbone OOM).
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    data_sample, img_ids
                )

            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
                output_dict=output_dict,
                num_frames=num_frames,
                run_mem_encoder=run_mem_encoder,
            )
            # Append the output, depending on whether it's a conditioning frame
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict
        
        # turn `output_dict` into a list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        # Make DDP happy with activation checkpointing by removing unused keys
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]

        return all_frame_outputs

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        frames_to_add_correction_pt=None,
        gt_masks=None,
    ):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        # Optionally, sample correction points iteratively to correct the mask
        if frame_idx in frames_to_add_correction_pt:
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame,
                point_inputs,
                gt_masks,
                high_res_features,
                pix_feat,
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                object_score_logits,
                current_out,
            )
            (
                _,
                _,
                _,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = final_sam_outputs

        # Use the final prediction (after all correction steps for output and eval)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out

    def _iter_correct_pt_sampling(
        self,
        is_init_cond_frame,
        point_inputs,
        gt_masks,
        high_res_features,
        pix_feat_with_mem,
        low_res_multimasks,
        high_res_multimasks,
        ious,
        low_res_masks,
        high_res_masks,
        object_score_logits,
        current_out,
    ):
        assert gt_masks is not None, "gt_masks must not be None"
        all_pred_masks = [low_res_masks]
        all_pred_high_res_masks = [high_res_masks]
        all_pred_multimasks = [low_res_multimasks]
        all_pred_high_res_multimasks = [high_res_multimasks]
        all_pred_ious = [ious]
        all_point_inputs = [point_inputs]
        all_object_score_logits = [object_score_logits]
        
        for _ in range(self.num_correction_pt_per_frame):
            # sample a new point from the error between prediction and ground-truth
            # (with a small probability, directly sample from GT masks instead of errors)
            if self.training and self.prob_to_sample_from_gt_for_train > 0:
                sample_from_gt = (
                    self.rng.random() < self.prob_to_sample_from_gt_for_train
                )
            else:
                sample_from_gt = False

            # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
            pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
            new_points, new_labels = get_next_point(
                input_fmt=self.input_fmt,
                time_separable=True,
                gt_masks=gt_masks,
                pred_masks=pred_for_new_pt,
                method="uniform" if self.training else self.pt_sampling_for_eval,
            )
            point_inputs = concat_points(point_inputs, new_points, new_labels)

            # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
            # For tracking, this means that when the user adds a correction click, we also feed
            # the tracking output mask logits along with the click as input to the SAM decoder.
            mask_inputs = low_res_masks
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            if self.use_act_ckpt_iterative_pt_sampling and not multimask_output:
                sam_outputs = torch.utils.checkpoint.checkpoint(
                    self._forward_sam_heads,
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                    use_reentrant=False,
                )
            else:
                sam_outputs = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    high_res_features=high_res_features,
                    multimask_output=multimask_output,
                )
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                _,
                object_score_logits,
            ) = sam_outputs
            all_pred_masks.append(low_res_masks)
            all_pred_high_res_masks.append(high_res_masks)
            all_pred_multimasks.append(low_res_multimasks)
            all_pred_high_res_multimasks.append(high_res_multimasks)
            all_pred_ious.append(ious)
            all_point_inputs.append(point_inputs)
            all_object_score_logits.append(object_score_logits)

        # Concatenate the masks along channel (to compute losses on all of them,
        # using `MultiStepIteractiveMasks`)
        current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
        current_out["multistep_pred_masks_high_res"] = torch.cat(
            all_pred_high_res_masks, dim=1
        )
        current_out["multistep_pred_multimasks"] = all_pred_multimasks
        current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
        current_out["multistep_pred_ious"] = all_pred_ious
        current_out["multistep_point_inputs"] = all_point_inputs
        current_out["multistep_object_score_logits"] = all_object_score_logits

        return point_inputs, sam_outputs

    @torch.no_grad()
    def predict(self, data_sample: dict, type: Literal["volume", "video"] = "volume") -> dict:
        """
        Automatic mask generation for a single volume.
        """
        if type == "volume":
            vol = data_sample["data_tensor"]
            assert vol.shape[0] == 1, "predict() expects batch_size=1"

            mask_data = self._predict_generate_masks(vol)
            mask_data.to_numpy()

            return {
                "masks": mask_data["masks"],
                "boxes": mask_data["boxes"],
                "iou_preds": mask_data["iou_preds"],
                "stability_score": mask_data["stability_score"],
                "points": mask_data["points"],
            }
        else:
            # TODO: implement video prediction
            raise NotImplementedError(f"type {type} not supported yet")

    def _predict_generate_masks(self, vol: torch.Tensor) -> "MaskData":
        """
        Generate masks for the full volume, possibly with multi-scale crops.
        """
        if self.input_fmt == "TZYXC":
            _, C, vol_z, vol_y, vol_x = vol.shape
            orig_size = (vol_z, vol_y, vol_x)

            crop_boxes, layer_idxs = generate_crop_boxes_3d(
                orig_size,
                self._amg_crop_n_layers,
                self._amg_crop_overlap_ratio,
            )

            data = MaskData()
            for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
                crop_data = self._predict_process_crop(
                    vol, crop_box, layer_idx, orig_size
                )
                data.cat(crop_data)

            # Cross-crop NMS: prefer masks from smaller crops
            if len(crop_boxes) > 1:
                volumes = box_volume(data["crop_boxes"])
                scores = 1.0 / volumes.clamp(min=1)
                keep = nms_3d(
                    data["boxes"].float(),
                    scores,
                    iou_threshold=self._amg_crop_nms_thresh,
                )
                data.filter(keep)

            # Optional: remove small disconnected regions and holes
            if self._amg_min_mask_region_area > 0:
                data = self._predict_postprocess_small_regions(
                    data,
                    self._amg_min_mask_region_area,
                    self._amg_crop_nms_thresh,
                )
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        return data

    # Per-crop: encode, generate points, iterate batches, within-crop NMS
    def _predict_process_crop(
        self,
        vol: torch.Tensor,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, int, int],
    ) -> "MaskData":
        """
        Process a single crop: encode image, run batched point prediction,
        NMS within crop, then map results back to global coordinates.
        """
        if self.input_fmt == "TZYXC":
            x0, y0, z0, x1, y1, z1 = crop_box
            cropped = vol[:, :, z0:z1, y0:y1, x0:x1]  # (1, C, Zc, Yc, Xc)
            crop_size = (z1 - z0, y1 - y0, x1 - x0)

            # Encode the crop
            features = self._predict_encode_crop(cropped)

            # Build point grid scaled to crop pixel coords (x, y, z)
            # point_grids are in [0,1]^3; scale to pixel coords
            scale = np.array([crop_size[2], crop_size[1], crop_size[0]])[None, :]  # (1, 3) as x, y, z
            points_for_crop = self._amg_point_grids[crop_layer_idx] * scale  # (N_pts, 3)

            # Process in batches
            data = MaskData()
            for (points_batch,) in batch_iterator(
                self._amg_points_per_batch, points_for_crop
            ):
                batch_data = self._predict_process_batch(
                    points_batch, features, crop_size, crop_box, orig_size
                )
                batch_data.to_cpu()  # offload before accumulation
                data.cat(batch_data)
                del batch_data

            if len(data) == 0:
                return data

            # Within-crop NMS
            keep = nms_3d(
                data["boxes"].float(),
                data["iou_preds"],
                iou_threshold=self._amg_box_nms_thresh,
            )
            data.filter(keep)

            # Map back to global coordinates
            data["boxes"] = uncrop_boxes_3d(data["boxes"], crop_box)
            data["points"] = uncrop_points_3d(data["points"], crop_box)
            n = len(data["iou_preds"])
            data["crop_boxes"] = torch.tensor(
                [crop_box] * n, device=data["boxes"].device
            )

        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        return data

    # Image encoding for a crop
    def _predict_encode_crop(self, crop_vol: torch.Tensor) -> dict:
        """
        Run image encoder + prepare backbone features for a single crop.
        """
        backbone_out = self.forward_image({"data_tensor": crop_vol})
        _, vision_feats, vision_pos_embeds, feat_sizes = (
            self._prepare_backbone_features(backbone_out)
        )

        # Add no_mem_embed for single-image SAM mode
        if self.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

        if self.input_fmt == "TZYXC":
            # Reshape from (DHW, B, C) to (B, C, D, H, W)
            B = crop_vol.shape[0]
            feats_5d = []
            for feat, size in zip(vision_feats[::-1], feat_sizes[::-1]):
                feats_5d.append(feat.permute(1, 2, 0).view(B, -1, *size))
            feats_5d = feats_5d[::-1]  # back to fine->coarse order
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        result = {
            "image_embed": feats_5d[-1],           # coarsest level
            "high_res_feats": feats_5d[:-1],        # finer levels
            "feat_sizes": feat_sizes,
            "vision_feats": vision_feats,
            "vision_pos_embeds": vision_pos_embeds,
        }
        return result

    # Per-batch: run model, filter by IoU / stability / crop edge
    def _predict_process_batch(
        self,
        points: np.ndarray,
        features: dict,
        crop_size: Tuple[int, int, int],
        crop_box: List[int],
        orig_size: Tuple[int, int, int],
    ) -> "MaskData":
        """
        Run inference on a batch of point prompts.
        """
        device = self.device
        if self.input_fmt == "TZYXC":
            orig_z, orig_y, orig_x = orig_size
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        # Prepare point inputs
        points_t = torch.as_tensor(points, dtype=torch.float32, device=device)
        # Each point is one prompt => (P, 1, 3) coords, (P, 1) labels=1 (foreground)
        point_coords = points_t[:, None, :]  # (P, 1, 3)
        point_labels = torch.ones(
            point_coords.shape[0], 1, dtype=torch.int32, device=device
        )

        point_inputs = {
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

        # Get backbone features for single image (idx 0)
        backbone_feats = features["image_embed"]
        high_res = features["high_res_feats"]      # list of (1, C, ...)

        # Run SAM heads
        P = point_coords.shape[0]
        if self.input_fmt == "TZYXC":
            backbone_expanded = backbone_feats.expand(P, -1, -1, -1, -1)
            high_res_expanded = [f.expand(P, -1, -1, -1, -1) for f in high_res] if high_res else None
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        (
            low_res_multimasks,  # (P, M, D4, H4, W4)
            high_res_multimasks, # (P, M, Z, Y, X)
            ious,                # (P, M)
            low_res_masks,       # (P, 1, D4, H4, W4)
            high_res_masks,      # (P, 1, Z, Y, X)
            _obj_ptr,
            _obj_score,
        ) = self._forward_sam_heads(
            backbone_features=backbone_expanded,
            point_inputs=point_inputs,
            mask_inputs=None,
            high_res_features=high_res_expanded,
            multimask_output=self._amg_multimask_output,
        )

        # Flatten multi-mask dim: (P, M, ...) -> (P*M, ...)
        masks = high_res_multimasks.flatten(0, 1)      # (P*M, Z, Y, X)
        iou_preds = ious.flatten(0, 1)                  # (P*M,)
        low_res = low_res_multimasks.flatten(0, 1)      # (P*M, D4, H4, W4)
        M = high_res_multimasks.shape[1]
        pts_repeated = points_t.repeat_interleave(M, dim=0)  # (P*M, 3)

        data = MaskData(
            masks=masks,
            iou_preds=iou_preds,
            points=pts_repeated,
            low_res_masks=low_res,
        )

        if self.debug:
            print(f"data['masks'].shape: {data['masks'].shape}")
            print(f"data['iou_preds'].shape: {data['iou_preds'].shape}")
            print(f"IOU PREDS: {data['iou_preds']}")
            print(f"data['low_res_masks'].shape: {data['low_res_masks'].shape}")

        # Optionally do mask-to-mask refinement
        if self._amg_use_m2m:
            refined_masks, refined_ious = self._predict_refine_with_m2m(
                data["points"], data["low_res_masks"], features
            )
            data["masks"] = refined_masks.squeeze(1)
            data["iou_preds"] = refined_ious.squeeze(1)

        # Filter by predicted IoU
        if self._amg_pred_iou_thresh > 0.0:
            keep = data["iou_preds"] > self._amg_pred_iou_thresh
            data.filter(keep)

            if self.debug:
                print(f"keep (after iou filter): {keep}")

        # Calculate and filter by stability score
        if self.input_fmt == "TZYXC":
            data["stability_score"] = calculate_stability_score_3d(
                data["masks"],
                self._amg_mask_threshold,
                self._amg_stability_score_offset,
            )                
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
        
        if self._amg_stability_score_thresh > 0.0:
            keep = data["stability_score"] >= self._amg_stability_score_thresh
            data.filter(keep)

            if self.debug:
                print(f"stability_score: {data['stability_score']}")
                print(f"keep (after stability score filter): {keep}")

        # Threshold masks and compute boxes
        data["masks"] = data["masks"] > self._amg_mask_threshold
        if self.input_fmt == "TZYXC":
            data["boxes"] = masks_to_boxes_v2(data["masks"])
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        # Filter boxes that touch crop boundaries but not volume boundaries
        if self.input_fmt == "TZYXC":
            keep = ~is_box_near_crop_edge_3d(
                data["boxes"],
                crop_box,
                [0, 0, 0, orig_x, orig_y, orig_z],
            )
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
        
        if not torch.all(keep):
            data.filter(keep)

        if self.input_fmt == "TZYXC":
            # Uncrop masks back to full volume for cross-crop NMS later
            data["masks"] = uncrop_masks_3d(
                data["masks"], crop_box, orig_z, orig_y, orig_x
            )
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        # Drop low_res_masks to save memory
        del data["low_res_masks"]

        if self.debug:
            print(f"data['masks'].shape: {data['masks'].shape}")
            print(f"data['iou_preds'].shape: {data['iou_preds'].shape}")
            print(f"IOU PREDS: {data['iou_preds']}")
            print(f"data['boxes'].shape: {data['boxes'].shape}")

        return data

    def _predict_refine_with_m2m(
        self,
        points: torch.Tensor,
        low_res_masks: torch.Tensor,
        features: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One-step refinement: feed previous low-res mask + original point
        back through the decoder with multimask_output=False.
        """
        if self.input_fmt == "TZYXC":
            all_masks, all_ious = [], []
            backbone_feats = features["image_embed"]
            high_res = features["high_res_feats"]

            for (pts_batch, lr_batch,) in batch_iterator(
                self._amg_points_per_batch, points, low_res_masks
            ):
                B = pts_batch.shape[0]
                point_inputs = {
                    "point_coords": pts_batch[:, None, :],
                    "point_labels": torch.ones(B, 1, dtype=torch.int32, device=self.device),
                }
                backbone_expanded = backbone_feats.expand(B, -1, -1, -1, -1)
                high_res_expanded = (
                    [f.expand(B, -1, -1, -1, -1) for f in high_res]
                    if high_res else None
                )

                (
                    _, _,
                    ious,            # (B, 1)
                    _,
                    high_res_masks,  # (B, 1, Z, Y, X)
                    _, _,
                ) = self._forward_sam_heads(
                    backbone_features=backbone_expanded,
                    point_inputs=point_inputs,
                    mask_inputs=lr_batch[:, None, :, :, :],
                    high_res_features=high_res_expanded,
                    multimask_output=False,
                )
                all_masks.append(high_res_masks)
                all_ious.append(ious)
        
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        return torch.cat(all_masks, dim=0), torch.cat(all_ious, dim=0)

    def _predict_postprocess_small_regions(
        self,
        mask_data: "MaskData",
        min_mask_region_area: int,
        nms_thresh: float,
    ) -> "MaskData":
        """
        Remove small disconnected regions and holes from masks,
        then rerun box NMS to remove any new duplicates.
        Edits mask_data in place.
        """
        if len(mask_data) == 0:
            return mask_data

        if self.input_fmt == "TZYXC":
            new_masks = []
            scores = []
            for i in range(len(mask_data["masks"])):
                mask_np = mask_data["masks"][i].cpu().numpy().astype(bool)

                mask_np, changed_holes = remove_small_regions_3d(mask_np, min_mask_region_area, mode="holes")
                unchanged = not changed_holes
                mask_np, changed_islands = remove_small_regions_3d(mask_np, min_mask_region_area, mode="islands")
                unchanged = unchanged and (not changed_islands)

                new_masks.append(torch.as_tensor(mask_np, device=mask_data["masks"].device).unsqueeze(0))
                scores.append(float(unchanged))

            masks = torch.cat(new_masks, dim=0)
            boxes = masks_to_boxes_v2(masks)

            keep = nms_3d(
                boxes.float(),
                torch.as_tensor(scores, device=boxes.device),
                iou_threshold=nms_thresh,
            )

            mask_data["masks"] = masks
            mask_data["boxes"] = boxes
            mask_data.filter(keep)

        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        return mask_data


# ---------------------------------------------------------------------------
# Factory / BUILD
# ---------------------------------------------------------------------------

_COMPONENT_CFG_KEYS = frozenset({
    "backbone_wrapper_args",
    "adapter_args",
    "memory_attention_args",
    "memory_encoder_args",
    "prompt_encoder_args",
    "mask_decoder_args",
    "criterion_args",
    "sam_embed_dim",
    "BUILD",
    "_target_",
})


def _extract_kwargs(
    cfg: Mapping[str, Any],
    extra_ignores: Optional[set] = None,
) -> Dict[str, Any]:
    """Drop Hydra/meta keys like ``_target_``, ``BUILD`` and any explicitly
    ignored keys, returning plain kwargs suitable for a constructor call."""
    ignore = {"_target_", "BUILD"}
    if extra_ignores:
        ignore.update(extra_ignores)
    return {k: v for k, v in cfg.items() if k not in ignore}


def BUILD(cfg: Mapping[str, Any]) -> SAM2:
    """
    Factory that builds a complete :class:`SAM2` model from a nested
    Hydra/OmegaConf config.
    """
    model_cfg = cfg.models.meta_arch.sam

    # ------------------------------------------------------------------
    # 0) Criterion
    # ------------------------------------------------------------------
    criterion_cfg = model_cfg["criterion_args"]
    criterion = get_method(criterion_cfg.BUILD)
    criterion = criterion(**_extract_kwargs(criterion_cfg))

    # ------------------------------------------------------------------
    # 1) Image encoder
    # ------------------------------------------------------------------
    bw_cfg = model_cfg["backbone_wrapper_args"]
    build_backbone = get_method(bw_cfg.BUILD)
    adapter_cfg = model_cfg.get("adapter_args", None)
    image_encoder = build_backbone(bw_cfg, adapter_cfg)

    # Derive hidden_dim for downstream component defaults
    hidden_dim = image_encoder.backbone_embed_dims[-1]

    # ------------------------------------------------------------------
    # 2) Memory attention (MemoryAttention)
    # ------------------------------------------------------------------
    mem_attn_cfg = model_cfg["memory_attention_args"]
    memory_attention = MemoryAttention(**_extract_kwargs(mem_attn_cfg))

    # ------------------------------------------------------------------
    # 3) Memory encoder (MemoryEncoder)
    # ------------------------------------------------------------------
    mem_enc_cfg = model_cfg["memory_encoder_args"]
    mem_enc_kwargs = _extract_kwargs(mem_enc_cfg)
    out_dim = mem_enc_kwargs.get("out_dim")
    assert out_dim % 3 == 0, (
        f"memory_encoder out_dim={out_dim} must be divisible by 3 "
        "for 3D sincos positional encoding (Z/Y/X split)."
    )
    # Resolve activation string to class for MaskDownSampler
    mask_act_str = mem_enc_kwargs.pop("mask_activation", "GELU")
    mem_enc_kwargs["mask_activation"] = get_activation(mask_act_str)
    # Build the positional encoding required by MemoryEncoder
    mem_pos_enc = PositionalEmbeddingSinCos(
        # NOTE: split the embedding dim into 3 for each spatial dimension
        num_pos_feats=mem_enc_kwargs.get("out_dim") // 3,
    )
    memory_encoder = MemoryEncoder(
        position_encoding=mem_pos_enc,
        **mem_enc_kwargs,
    )

    # ------------------------------------------------------------------
    # 4) SAM prompt encoder (PromptEncoder)
    # ------------------------------------------------------------------
    prompt_enc_cfg = dict(model_cfg.get("prompt_encoder_args"))
    # Resolve activation string to class
    act_str = prompt_enc_cfg.pop("activation", "GELU")
    prompt_enc_cfg["activation"] = get_activation(act_str)
    sam_prompt_encoder = PromptEncoder(
        embed_dim=prompt_enc_cfg.pop("embed_dim"),
        mask_in_chans=prompt_enc_cfg.pop("mask_in_chans"),
        mask_downsample_factor=model_cfg.get("mask_downsample_factor"),
        input_shape=model_cfg["input_shape"],
        patch_shape=model_cfg["patch_shape"],
        input_format=model_cfg["input_fmt"],
        **prompt_enc_cfg,
    )

    # ------------------------------------------------------------------
    # 5) SAM mask decoder (MaskDecoder)
    # ------------------------------------------------------------------
    mask_dec_cfg = dict(model_cfg.get("mask_decoder_args"))
    # Resolve activation strings to classes
    transformer_act_str = mask_dec_cfg.pop("transformer_activation", "relu")
    act_str = mask_dec_cfg.pop("activation", "gelu")
    sam_mask_decoder = MaskDecoder(
        input_fmt=model_cfg["input_fmt"],
        mask_downsample_factor=model_cfg.get("mask_downsample_factor"),
        transformer_dim=mask_dec_cfg.pop("transformer_dim"),
        transformer_depth=mask_dec_cfg.pop("transformer_depth"),
        transformer_num_heads=mask_dec_cfg.pop("transformer_num_heads"),
        transformer_mlp_dim=mask_dec_cfg.pop("transformer_mlp_dim"),
        num_multimask_outputs=mask_dec_cfg.pop("num_multimask_outputs"),
        iou_head_depth=mask_dec_cfg.pop("iou_head_depth"),
        iou_head_hidden_dim=mask_dec_cfg.pop("iou_head_hidden_dim"),
        transformer_activation=get_activation(transformer_act_str),
        activation=get_activation(act_str),
        use_high_res_features=model_cfg.get("use_high_res_features_in_sam"),
        iou_prediction_use_sigmoid=model_cfg.get("iou_prediction_use_sigmoid"),
        pred_obj_scores=model_cfg.get("pred_obj_scores"),
        pred_obj_scores_mlp=model_cfg.get("pred_obj_scores_mlp"),
        use_multimask_token_for_obj_ptr=model_cfg.get(
            "use_multimask_token_for_obj_ptr"
        ),
        **mask_dec_cfg,
    )

    # ------------------------------------------------------------------
    # 6) Collect remaining scalar / flag params and build SAM2
    # ------------------------------------------------------------------
    scalar_kwargs = {
        k: v for k, v in model_cfg.items() if k not in _COMPONENT_CFG_KEYS
    }

    return SAM2(
        criterion=criterion,
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        sam_prompt_encoder=sam_prompt_encoder,
        sam_mask_decoder=sam_mask_decoder,
        **scalar_kwargs,
    )