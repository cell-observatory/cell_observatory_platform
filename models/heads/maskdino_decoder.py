"""
Adapted from:
https://github.com/IDEA-Research/MaskDINO/maskdino/modeling/transformer_decoder/maskdino_decoder.py
"""

from typing import Literal, Optional

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn

from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_platform.data.structures import bitmask_to_boxes, box_xyzxyz_to_cxcyczwhd, masks_to_boxes_v2
from cell_observatory_platform.models.heads.dino_decoder import DeformableTransformerDecoderLayer, TransformerDecoder
from cell_observatory_platform.models.layers.activation import get_activation
from cell_observatory_platform.models.layers.conv3d import Conv3d
from cell_observatory_platform.models.layers.mlp import MLP
from cell_observatory_platform.models.layers.utils import compute_unmasked_ratio, inverse_sigmoid


class MaskDINODecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        feedforward_dim: int,
        decoder_num_layers: int,
        mask_dim: int,
        enforce_input_projection: bool,
        two_stage_flag: bool,
        denoise_queries_flag: bool,
        noise_scale: float,
        total_denosing_queries: int,
        # TODO: make this Enumeral
        initialize_box_type: Optional[Literal["bitmask", "mask2box"]],
        with_initial_prediction: bool,
        learn_query_embeddings: bool,
        total_num_feature_levels: int = 4,
        dropout: float = 0.0,
        activation: str = "RELU",
        num_heads: int = 8,
        decoder_num_points: int = 4,
        return_intermediates_decoder: bool = True,
        query_dim: int = 6,
        share_decoder_layers: bool = False,
        dtype: str = "bfloat16",
        use_deform_attention: bool = False,
    ):
        """
        Args:
            in_channels: channels of the input features
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            n_heads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_projection: add input project 1x1 conv even wehen input
                channels and hidden dim are identical
            d_model: transformer dimension
            dropout: dropout rate
            activation: activation function
            num_heads: num heads in multi-head attention
            decoder_num_points: number of sampling points in decoder
            return_intermediate_dec: return the intermediate results of decoder
            query_dim: 6 -> (x, y, z, w, h, d)
            share_decoder_layers: whether to share each decoder layer
        """
        super().__init__()

        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        # flag to support prediction from initial matching and denoising outputs
        self.with_initial_prediction = with_initial_prediction
        self.initialize_box_type = initialize_box_type
        self.init_boxes = initialize_box_type is not None

        # flag to support two-stage pipeline
        self.two_stage_flag = two_stage_flag
        # number of decoder layers
        self.num_layers = decoder_num_layers
        self.num_feature_levels = total_num_feature_levels
        # flag to generate learnable query embeddings, otherwise use
        # encoder outputs (two-stage mode)
        self.learn_query_embeddings = learn_query_embeddings
        # number of object queries transformer decoder will use
        self.num_queries = num_queries

        # flag to support denoising queries
        self.denoise_queries_flag = denoise_queries_flag
        # noise_scale for bbox: max = 1.0 => noise is up to full d/h/w range
        # noise_scale for labels: corrupt labels with p = noise_scale * 0.5
        self.noise_scale = noise_scale
        # total number of noisy copies of ground-truth targets
        # injected into the decoder
        self.total_denosing_queries = total_denosing_queries

        # define modules:
        # 1. define learnable query features (in two-stage queries are generated from encoder output)
        #    includes bbox initialization if not two-stage
        # 2. define proposal/query input projection layer (if two-stage pipeline): Linear -> LayerNorm
        # 3. define input query projection layers for each feature level: Conv3d with 1x1 kernel
        # 4. define decoder transformer layers (deformable attention transformer layer)
        # 5. define transformer decoder
        # 6. define class predictor FFN
        # 7. define label embeddings and mask FFN
        # 8. define bbox regressor FFN
        # 9. define bbox regressor module (multiple layers of bbox regressor FFN)

        # define learnable query features in two-stage pipeline
        # queries are generated from encoder output
        if not two_stage_flag or self.learn_query_embeddings:
            self.query_features = nn.Embedding(num_queries, hidden_dim)

        # NOTE: decide if we want to support all possible options
        #       regarding initialization for query_embeddings and
        #       query_features
        if not two_stage_flag or not self.init_boxes:
            self.query_embeddings = nn.Embedding(num_queries, 6)

        # encoder generates class logits and proposals
        # project and normalize proposals before feeding into the decoder
        if two_stage_flag:
            self.encoder_output_mlp = nn.Linear(hidden_dim, hidden_dim)
            self.encoder_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_projection:
                self.input_proj.append(Conv3d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim,
            feedforward_dim,
            dropout,
            get_activation(activation),
            self.num_feature_levels,
            num_heads,
            decoder_num_points,
            use_deform_attention=use_deform_attention,
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            self.num_layers,
            decoder_norm,
            return_intermediates=return_intermediates_decoder,
            embed_dim=hidden_dim,
            query_dim=query_dim,
            num_feature_levels=self.num_feature_levels,
            share_decoder_layers=share_decoder_layers,
        )

        self.num_classes = num_classes
        self.class_predictor = nn.Linear(hidden_dim, num_classes)

        self.label_embeddings = nn.Embedding(num_classes, hidden_dim)
        self.mask_embeddings = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.hidden_dim = hidden_dim
        self._bbox_regressor = _bbox_regressor = MLP(hidden_dim, hidden_dim, 6, 3)

        # set the last layer of the box prediction FFN to 0
        # implies decoder's first prediction will be exactly
        # the initial reference point
        nn.init.constant_(_bbox_regressor.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_regressor.layers[-1].bias.data, 0)

        # share box prediction each layer
        _bbox_regressor_layerlist = [_bbox_regressor for i in range(self.num_layers)]
        self.bbox_regressor = nn.ModuleList(_bbox_regressor_layerlist)
        # IMPORTANT: share bbox regressor with decoder
        self.decoder.bbox_embed = self.bbox_regressor

    @staticmethod
    def gen_encoder_output_proposals(memory, memory_padding_mask, shapes):
        N, S, C = memory.shape

        level_start_index, proposals = 0, []
        for lvl, (D, H, W) in enumerate(shapes):
            # (bs, SUM{dxhxw}) -> (bs, d_lvl*h_lvl*w_lvl) -> (bs, d_lvl, h_lvl, w_lvl, 1)
            level_padding_mask = memory_padding_mask[:, level_start_index : (level_start_index + D * H * W)].view(
                N, D, H, W, 1
            )

            # level_padding_mask: (N, D, H, W), True where padded
            # (N, D, H, W) -> (N, D) bool check any voxel in D slice is valid ->
            # (N,) number of valid D slices in mask
            valid_D = (~level_padding_mask).any(dim=(2, 3)).sum(dim=1)  # (N,)
            valid_H = (~level_padding_mask).any(dim=(1, 3)).sum(dim=1)  # (N,)
            valid_W = (~level_padding_mask).any(dim=(1, 2)).sum(dim=1)  # (N,)

            grid_z, grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, D - 1, D, dtype=torch.float32, device=memory.device),
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
                indexing="ij",
            )
            # grid: 3x(W, H, D, 1) -> (W, H, D, 3)
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), grid_z.unsqueeze(-1)], -1)
            # scale: 3 x (N,1) -> (N,3) -> (N, 1, 1, 1, 3)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1), valid_D.unsqueeze(-1)], 1).view(
                N, 1, 1, 1, 3
            )
            # grid: (W, H, D, 3) -> (1, W, H, D, 3) -> (N, W, H, D, 3) -> +0.5 (move corner to centre) & scale (normalize [0,1])
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1, -1) + 0.5) / scale

            # all anchors at this level get same d,h,w in normalized units
            # with different anchor locations given by grid centre positions
            whd = torch.ones_like(grid) * 0.05 * (2.0**lvl)  # scale size by 0.05 × 2^level
            # (N, W, H, D, 3) -> (N, W, H, D, 6) -> (N, D*H*W, 6)
            proposal = torch.cat((grid, whd), -1).view(N, -1, 6)
            proposals.append(proposal)

            level_start_index += D * H * W

        # (N, D*H*W, 6) -> (N, SUM{dxhxw}, 6)
        output_proposals = torch.cat(proposals, 1)
        # check which proposals have (x,y,z,w,h,d) in range [0,1] (normalized by padding mask)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        # standard logit transform log(p / (1 - p)), we do additive bbox prediction and then sigmoid to revert to [0,1]
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        # set to inf where padding mask is True (invalid proposals) or outside valid range (any softmax will ignore)
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        # zero out corresponding memory features for invalid positions
        output_memory = memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        return output_memory, output_proposals

    def generate_denoising_queries(self, targets, target_query_embeddings, reference_point_embeddings, batch_size):
        if self.training:
            labels_per_image = [
                (torch.ones_like(target["labels"])).cuda() for target in targets
            ]  # (bs, num_known_per_image)
            label_indices_per_image = [
                torch.nonzero(target) for target in labels_per_image
            ]  # (bs, num_known_per_image)
            num_labels_per_image = [sum(gt_labels) for gt_labels in labels_per_image]  # (bs, )

            if max(num_labels_per_image) == 0:
                return None, None, None, None

            # fix number of denoising queries such that each label
            # gets the same number of queries with total = total query num
            # FIXME: may error if num_labels_per_image is very large
            denoise_queries_per_label = self.total_denosing_queries // (int(max(num_labels_per_image)))

            # binary mask indicating which GT boxes/labels should be used for denoising (currently all 1s)
            # can be modified to selectively denosie some labels or boxes (hence the overcomplicated logic)
            bboxes_denoise_index_mask = labels_denoise_index_mask = torch.cat(
                labels_per_image
            )  # (total_num_labels, ) all 1s
            # retrieve labels and boxes
            labels = torch.cat([target["labels"] for target in targets])  # (total num labels)
            bboxes = torch.cat([target["boxes"] for target in targets])  # (total num bboxes)

            # allows for assigning each DN query to the correct image
            # in the batch: [0] * num_labels_image1 + [1] * num_labels_image2 + ... => (num_targets, total_num_labels)
            batch_idx = torch.cat([torch.full_like(target["labels"], idx) for idx, target in enumerate(targets)])

            # roundabout way to get the num of labels and bboxes to denoise (legacy code)
            # shape: (total_num_objects_denoise,) = 2 x (total_num_labels)
            denoise_target_indices = torch.nonzero(bboxes_denoise_index_mask + labels_denoise_index_mask).view(-1)

            # generate denoise_queries_per_label num of queries for each denoise_item (copies to be populated)
            # shape: (total_num_objects_denoise * denoise_queries_per_label,)
            denoise_target_indices = denoise_target_indices.repeat(denoise_queries_per_label, 1).view(-1)
            # same logic for lables, batch_idx, bboxes i.e. (total_num_objects_denoise * denoise_queries_per_label,)
            total_denoise_labels = labels.repeat(denoise_queries_per_label, 1).view(-1)
            denoise_query_batch_id = batch_idx.repeat(denoise_queries_per_label, 1).view(-1)
            total_denoise_bboxes = bboxes.repeat(denoise_queries_per_label, 1)

            denoise_target_indices_copy = denoise_target_indices.clone()
            total_denoise_bboxes_copy = total_denoise_bboxes.clone()

            # add noise to labels and bboxes
            # label ids are randomly assigned with prob p = noise_scale * 0.5
            # bbox coordinates are randomly shifted
            if self.noise_scale > 0:
                # uniform sampling [0,1]
                label_flip_probs = torch.rand_like(denoise_target_indices_copy.float())
                # 50% chance to flip with max noise_scale = 1
                flipped_indices = torch.nonzero(label_flip_probs < (self.noise_scale * 0.5)).view(-1)
                # what new class to assign flipped indices
                new_label_ids = torch.randint_like(flipped_indices, 0, self.num_classes)
                # scatter new noisy labels to total_denoise_labels
                total_denoise_labels.scatter_(0, flipped_indices, new_label_ids)

                denoise_bbox_deltas = torch.zeros_like(total_denoise_bboxes_copy)
                denoise_bbox_deltas[:, :3] = total_denoise_bboxes_copy[:, 3:] / 2  # shift amount of dd, dh, dw
                denoise_bbox_deltas[:, 3:] = total_denoise_bboxes_copy[:, 3:]  # starting dd, dw, dh

                # randomly shift bbox coordinates ([-1,1] * bbox_deltas * noise_scale)
                total_denoise_bboxes_copy += (
                    torch.mul((torch.rand_like(total_denoise_bboxes_copy) * 2 - 1.0), denoise_bbox_deltas).cuda()
                    * self.noise_scale
                )
                total_denoise_bboxes_copy = total_denoise_bboxes_copy.clamp(
                    min=0.0, max=1.0
                )  # clamp new bbox coordinates to [0,1] range

            # embed/encode noised labels and bboxes
            total_denoise_label_embeddings = self.label_embeddings(total_denoise_labels.long().cuda()).to(self.dtype)
            # encode bboxes into sigmoid space
            total_denoise_bboxes_encoded = inverse_sigmoid(total_denoise_bboxes_copy)

            # pad all denoising queries to the same size
            max_labels_per_image = int(max(num_labels_per_image))
            max_query_pad_size = int(max_labels_per_image * denoise_queries_per_label)

            # pad the number of denoise queries (labels per image * num queries per label)
            # per image to the same size
            denoise_labels_padded = torch.zeros(max_query_pad_size, self.hidden_dim, dtype=self.dtype).cuda()
            denoise_bboxes_padded = torch.zeros(max_query_pad_size, 6).cuda()

            # combine the denoised labels and bboxes with target queries/bbox embeddings if they exist
            # TODO: check if branch based on both reference_point_embeddings and target_query_embeddings is necessary
            if reference_point_embeddings is not None and target_query_embeddings is not None:
                label_queries = torch.cat([denoise_labels_padded, target_query_embeddings], dim=0).repeat(
                    batch_size, 1, 1
                )  # (batch_size, num_dn + num_queries, d_model)
                bbox_queries = torch.cat([denoise_bboxes_padded, reference_point_embeddings], dim=0).repeat(
                    batch_size, 1, 1
                )  # (batch_size, num_dn + num_queries, 6)
            else:
                label_queries = denoise_labels_padded.repeat(batch_size, 1, 1)  # (batch_size, num_dn, d_model)
                bbox_queries = denoise_bboxes_padded.repeat(batch_size, 1, 1)  # (batch_size, num_dn, 6)

            # we have batch_size number of images, each with 0 to max_labels_per_image labels
            # each image has denoise_queries_per_label number of queries per label
            # hence we create a mapping from our denoise_label_embeddings/denoise_bboxes_encodings
            # to locations in our label_queries/bbox_queries tensors with padding logic below
            denoise_target_indices_map = torch.tensor([]).cuda()

            if len(num_labels_per_image) > 0:
                # for each image, create range [0, ..., num_labels_image_i-1] and cat them together
                # denoise_target_indices_map: (1, total_num_labels_denoise)
                # in pattern [0,...,num_labels_image1-1, 0,...,num_labels_image2-1, ...]
                denoise_target_indices_map = torch.cat(
                    [torch.tensor(range(num_labels)) for num_labels in num_labels_per_image]
                )
                # each group (for replication i of denoise queries) is assigned a slot of size max_labels_per_image
                # hence shift the indices (element values) by i × max_labels_per_image
                # thus to write to the correct row in the padded tensor we need to shift the indices by i * max_labels_per_image
                # denoise_target_indices_map: (1, max_labels_per_image * queries_per_label)
                # in pattern: [0,...,num_labels_image1-1, ..., max_labels_per_image,...,max_labels_per_image+num_labels_image2-1, ..., max_labels_per_image*2,...]
                denoise_target_indices_map = torch.cat(
                    [
                        denoise_target_indices_map + max_labels_per_image * query_num
                        for query_num in range(denoise_queries_per_label)
                    ]
                ).long()

            if len(denoise_query_batch_id) > 0:
                # batch ids are of form [0, 0, 1, 1, 1, ...] x N copies for N denoising queries per label where element
                # in batch_ids is of the form: [0] * num_labels_image1 + [1] * num_labels_image2 + ...
                # denoise_target_indices_map is of form: [0,...,num_labels_image1-1, ..., max_labels_per_image,...] (see above)
                # this way we assign the embeddings of the noisy labels and bboxes to position in our query matrices where
                # previously we had vectors of zeros from denoise_labels_padded/denoise_bboxes_padded (its split into num_dn & num_queries)
                # indexing logic example:  0 * [0, ..., num_labels_image_0-1] matched with range([0, ..., num_labels_image_0-1])
                # this way we index exactly as many elements as there exists for each batch element and insert noisy labels/bboxes
                label_queries[(denoise_query_batch_id.long(), denoise_target_indices_map)] = (
                    total_denoise_label_embeddings
                )
                bbox_queries[(denoise_query_batch_id.long(), denoise_target_indices_map)] = total_denoise_bboxes_encoded

            # num_queries regular queries and max_query_pad_size denoising queries for each image
            # attention mask will determine which queries can attend to which
            # rules (False = can attend, True = cannot attend -> think of it as True = To Mask is True):
            # regular queries cannot attend to denoising queries
            # denoising queries cannot attend to each other
            # denoising queries can only attend within their group (different gt query)
            # NOTE: attention mask is defined her for one batch element
            attn_mask_size = max_query_pad_size + self.num_queries
            # first set all False => everyone can see everyone initially
            attn_mask = torch.ones(attn_mask_size, attn_mask_size).cuda() < 0

            # the attention mask matrix is organized as follows:
            # [denoising queries Group 0 | denoising queries Group 1 | .... | regular queries]
            # [..............................................................................]
            # [denoising queries Group 0 | denoising queries Group 1 | .... | regular queries]

            # regular queries cannot see the denoising queries
            # attn_mask[max_query_pad_size:] is all regular queries
            # attn_mask[:, :max_query_pad_size] is all denoising queries
            attn_mask[max_query_pad_size:, :max_query_pad_size] = True

            # denoising queries cannot see each other
            # we iterate over denoising query copies each group
            # has one per label (possible padded)
            for i in range(denoise_queries_per_label):
                if i == 0:
                    # for denoising group 0, mask all queries that do not live in [0, max_labels_per_image],
                    # i.e. everything not in top left corner of our attention mask matrix
                    # we don't have any blocks to mask before this one since it's the first
                    attn_mask[:max_labels_per_image, max_labels_per_image:max_query_pad_size] = True
                if i == denoise_queries_per_label - 1:
                    # for the last denoinsing group, mask all queries that do not live in
                    # [max_labels_per_image * (denoise_queries_per_label-1), max_labels_per_image * denoise_queries_per_label]
                    # we do not have to mask the regular queries as they are already masked out
                    # hence we save on one masking operation at the expense of more branching logic
                    attn_mask[max_labels_per_image * i : max_labels_per_image * (i + 1), : max_labels_per_image * i] = (
                        True
                    )
                else:
                    # if not at the ends of the matrix, mask all queries that do not live in the correct block
                    # we do this in two steps: first all blocks after current block, then all blocks before current block
                    attn_mask[
                        max_labels_per_image * i : max_labels_per_image * (i + 1),
                        max_labels_per_image * (i + 1) : max_query_pad_size,
                    ] = True
                    attn_mask[max_labels_per_image * i : max_labels_per_image * (i + 1), : max_labels_per_image * i] = (
                        True
                    )

            denoise_data = {
                "denoise_target_indices": torch.as_tensor(denoise_target_indices).long(),
                "batch_idx": torch.as_tensor(batch_idx).long(),
                "denoise_target_indices_map": torch.as_tensor(denoise_target_indices_map).long(),
                "known_lbs_bboxes": (total_denoise_labels, total_denoise_bboxes),
                "label_indices_per_image": label_indices_per_image,
                "max_query_pad_size": max_query_pad_size,
                "denoise_queries_per_label": denoise_queries_per_label,
            }

        else:
            if reference_point_embeddings is not None and target_query_embeddings is not None:
                # (batch_size, num_queries, d_model)
                label_queries = target_query_embeddings.repeat(batch_size, 1, 1)
                # (batch_size, num_queries, 6)
                bbox_queries = reference_point_embeddings.repeat(batch_size, 1, 1)
            else:
                label_queries = None
                bbox_queries = None

            attn_mask = None
            denoise_data = None

        return label_queries, bbox_queries, attn_mask, denoise_data

    def denoise_post_process(self, outputs_labels, outputs_bboxes, outputs_mask, denoise_data):
        max_query_pad_size = denoise_data["max_query_pad_size"]
        assert max_query_pad_size > 0, "max_query_pad_size must be greater than 0"

        # first max_query_pad_size queries are the denoising queries
        # we return the non-denoising queries and store the denoising
        # queries in the mask_dict
        outputs_labels_denoise = outputs_labels[:, :, :max_query_pad_size, :]
        outputs_labels = outputs_labels[:, :, max_query_pad_size:, :]

        outputs_bboxes_denoise = outputs_bboxes[:, :, :max_query_pad_size, :]
        outputs_bboxes = outputs_bboxes[:, :, max_query_pad_size:, :]

        if outputs_mask is not None:
            outputs_mask_denoise = outputs_mask[:, :, :max_query_pad_size, :]
            outputs_mask = outputs_mask[:, :, max_query_pad_size:, :]

        out = {
            "pred_logits": outputs_labels_denoise[-1],
            "pred_boxes": outputs_bboxes_denoise[-1],
            "pred_masks": outputs_mask_denoise[-1],
        }
        out["auxiliary_outputs"] = self._set_aux_loss(
            outputs_labels_denoise, outputs_mask_denoise, outputs_bboxes_denoise
        )
        denoise_data["predicted_denoise_bboxes"] = out
        return outputs_labels, outputs_bboxes, outputs_mask

    def predict_bboxes(self, reference_pts_list, intermediates, initial_reference_pts=None):
        device = reference_pts_list[0].device

        if initial_reference_pts is None:
            outputs_bbox_list = []
        else:
            outputs_bbox_list = [initial_reference_pts.to(device)]

        # iterate over all decoder layers and update the reference points for each layer
        for layer_id, (reference_pts, layer_bbox_regressor, intermediate) in enumerate(
            zip(reference_pts_list[:-1], self.bbox_regressor, intermediates)
        ):
            layer_deltas = layer_bbox_regressor(intermediate).to(device)
            reference_pts_output = layer_deltas + inverse_sigmoid(reference_pts).to(device)
            outputs_bbox_list.append(reference_pts_output.sigmoid())

        outputs_bboxes = torch.stack(outputs_bbox_list)
        return outputs_bboxes

    def forward_prediction_heads(self, output_queries, pixel_decoder_output, predict_masks=True):
        decoder_output = self.decoder_norm(output_queries)
        decoder_output = decoder_output.transpose(0, 1)  # (bs, num_queries, C)
        outputs_class = self.class_predictor(decoder_output)  # (bs, num_queries, num_classes)

        outputs_mask = None
        if predict_masks:
            mask_embeddings = self.mask_embeddings(decoder_output)
            # dot product between mask embeddings and pixel decoder output
            outputs_mask = torch.einsum(
                "bqc, bcdhw->bqdhw", mask_embeddings, pixel_decoder_output
            )  # (bs, num_queries, d, h, w)
        return outputs_class, outputs_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_masks, out_bboxes=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if out_bboxes is None:
            return [{"pred_logits": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_masks[:-1])]
        else:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes": c}
                for a, b, c in zip(outputs_class[:-1], outputs_masks[:-1], out_bboxes[:-1])
            ]

    def forward(self, x, pixel_decoder_output, masks, targets=None):
        assert len(x) == self.num_feature_levels, "The number of feature maps much match self.num_feature_levels"
        device = x[0].device

        # allegedly disabling masking does not affect performance unless feature map
        # is not divisible by 32
        if masks is None or not any(
            feature_map.size(2) % 32 or feature_map.size(3) % 32 or feature_map.size(4) % 32 for feature_map in x
        ):
            # masks: (num_feature_levels, bs, d, h, w)
            masks = [
                torch.zeros(
                    (feature_map.size(0), feature_map.size(2), feature_map.size(3), feature_map.size(4)),
                    device=feature_map.device,
                    dtype=torch.bool,
                )
                for feature_map in x
            ]

        x_flatten, mask_flatten, shapes = [], [], []
        # iterate over feature levels in reverse order
        for idx in range(self.num_feature_levels - 1, -1, -1):
            bs, c, d, h, w = x[idx].shape
            # (bs, c, d, h, w) -> (bs, d_model, d, h, w) -> (bs, d_model, dxhxw) -> (bs, dxhxw, d_model)
            shapes.append(torch.as_tensor([d, h, w], dtype=torch.long, device=device))
            # (bs, c, d, h, w) -> (bs, d_model, d, h, w) → (bs, d_model, dxhxw) -> (bs, dxhxw, d_model)
            x_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            # (bs, d, h, w) -> (bs, dxhxw)
            mask_flatten.append(masks[idx].flatten(1))

        # concatenate all feature levels
        x_flatten = torch.cat(x_flatten, 1)  # bs, SUM{dxhxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, SUM{dxhxw}
        shapes = torch.stack(shapes, dim=0).to(x_flatten.device)  # (num_feature_levels, 3)

        # get the start index of each feature level in token sequence [0, d1*h1*w1, d1*h1*w1 + d2*h2*w2, ...]
        # prod(1) => (num_feature_levels, ) = [d1*h1*w1, d2*h2*w2, ...], .cumsum(0) => CUMSUM([d1*h1*w1, d2*h2*w2, ...])
        level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))  # (num_feature_levels, )
        # get ratio mask vs unmasked volume for each feature level (padded, not real data vs real data)
        # reverse axis dims of valid_ratios to HDW since this is what is expected downstream
        masks_rev = [masks[i] for i in range(self.num_feature_levels - 1, -1, -1)]
        valid_ratios = torch.stack([compute_unmasked_ratio(m)[..., [2, 1, 0]] for m in masks_rev], 1)
        
        # queries_learned_topk is populated if learned query_features is used
        predictions_class, predictions_mask, queries_learned_topk = [], [], None
        if self.two_stage_flag:
            # generate queries for topk query selection which will be used to initialize the decoder
            # note that output_memory and queries_learned are terminology used largely interchangeably
            output_memory, output_proposals = self.gen_encoder_output_proposals(x_flatten, mask_flatten, shapes)
            output_memory = self.encoder_output_norm(self.encoder_output_mlp(output_memory))

            # predict class logits and proposals from encoder memory
            class_predictions = self.class_predictor(output_memory)
            # TODO: Check that is equivalent to the original implementation
            # shape: (bs, \sum{dxhxw}, 6)
            bbox_predictions = self._bbox_regressor(output_memory) + output_proposals

            # select topk proposals based on class prediction confidence (regardless of class)
            # class_predictions: (bs, num_queries, num_classes) -> max(-1)[0] is most confident class
            #                    for each query (bs, num_queries)
            # returns: indices for topk queries based on highest query class confidence
            # shape: (bs, k)
            topk_proposals = torch.topk(class_predictions.max(-1)[0], self.num_queries, dim=1)[1]

            bbox_preds_topk = torch.gather(
                bbox_predictions,
                # query index dimension for gather op.
                1,
                # select topk queries from bbox_preds
                # (bs, k) -> (bs, k, 1) -> (bs, k, 6)
                topk_proposals.unsqueeze(-1).repeat(1, 1, 6),
            )

            # select topk queries
            # gather op: output[b, k, c] = input[b, index[b, k, c], c]
            output_memory_topk = torch.gather(
                output_memory,  # (bs, num_queries, d_model)
                1,
                # select (bs, k, d_model)
                topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim),
            )

            outputs_labels, outputs_masks = self.forward_prediction_heads(
                output_memory_topk.transpose(0, 1), pixel_decoder_output
            )

            # the predicted bboxes / queries from the encoder are used as initial proposals
            # and hence should not be backpropagated through thus we detach them
            # from the computation graph here
            bbox_preds_topk = bbox_preds_topk.detach()
            output_memory_topk = output_memory_topk.detach()

            #  store initial predictions from the encoder
            intermediate_outputs = dict()
            intermediate_outputs["pred_logits"] = outputs_labels
            intermediate_outputs["pred_boxes"] = bbox_preds_topk.sigmoid()
            intermediate_outputs["pred_masks"] = outputs_masks

            # optionally: initialize decoder box queries using predicted masks
            if self.init_boxes:
                # convert masks into boxes to better initialize boxes in the decoder
                assert (
                    self.with_initial_prediction
                ), "Initial prediction flag must be set to True when using box initialization!"
                # we flatten masks since the mask_to_boxes functions expect (N, D, H, W) mask tensor
                flatten_mask = outputs_masks.detach().flatten(0, 1)  # (B * num_queries, D, H, W)
                d, h, w = outputs_masks.shape[-3:]

                if self.initialize_box_type == "bitmask":  # slower, but more accurate
                    # TODO: implement same safety check as in masks_to_boxes_v2
                    # refpoint_embeddings = BitMasks(flatten_mask > 0).get_bounding_boxes().tensor.to(device)
                    refpoint_embeddings = bitmask_to_boxes(flatten_mask > 0).to(device)
                elif self.initialize_box_type == "mask2box":  # faster
                    # returns: (N, 6)
                    refpoint_embeddings = masks_to_boxes_v2(flatten_mask > 0).to(device)
                else:
                    assert NotImplementedError, "Unknown box initialization type: {}".format(self.initialize_box_type)

                # box_ops returns: (cx, cy, cz, w, h, d), we divide by the feature map size to normalize to [0, 1] range
                refpoint_embeddings = box_xyzxyz_to_cxcyczwhd(refpoint_embeddings) / torch.as_tensor(
                    [w, h, d, w, h, d], dtype=torch.float
                ).to(device)
                # (B * num_queries, 6) -> (B, num_queries, 6)
                refpoint_embeddings = refpoint_embeddings.reshape(outputs_masks.shape[0], outputs_masks.shape[1], 6)
                refpoint_embeddings = inverse_sigmoid(refpoint_embeddings)

            else:
                refpoint_embeddings = self.query_embeddings.weight[None].repeat(bs, 1, 1)

            # optionally override topk queries with learned embeddings
            if self.learn_query_embeddings:
                # shape: (1, num_queries, d_model) -> (bs, num_queries, d_model)
                queries_learned_topk = self.query_features.weight[None].repeat(bs, 1, 1)

        else:
            # (1, num_queries, d_model) -> (bs, num_queries, d_model)
            queries_learned_topk = self.query_features.weight[None].repeat(bs, 1, 1)
            # (1, num_queries, 6) -> (bs, num_queries, 6)
            refpoint_embeddings = self.query_embeddings.weight[None].repeat(bs, 1, 1)

        # if two_stage flag is False or if learned query embeddings are used, we use learned
        # query embeddings
        queries = queries_learned_topk if queries_learned_topk is not None else output_memory_topk

        # generate denoising queries if training and denoising queries flag is set
        attn_mask, denoise_metadata = None, None
        if self.denoise_queries_flag and self.training:
            assert targets is not None, "If denoising queries are used, targets must be provided"
            # generates noisy copies of the ground-truth labels (flipped) and bboxes (shifted)
            # TODO: consider including target_query_embeddings and reference_point_embeddings
            #       here or removing that logic from generate_denoising_queries
            label_queries, bbox_queries, attn_mask, denoise_metadata = self.generate_denoising_queries(
                targets, None, None, x[0].shape[0]
            )

            # denoise_metadata will be None if there are no labels to denoise
            # OR if we are not training, henceforth its a used as a flag for if
            # denoising queries are being used or not
            if denoise_metadata is not None:
                queries = torch.cat([label_queries, queries], dim=1)  # (bs, num_dn + num_queries, d_model)

        if self.with_initial_prediction:
            # predict class logits and masks from queries and pixel decoder output
            # NOTE: consider reworking transpose logic
            outputs_class, outputs_mask = self.forward_prediction_heads(
                queries.transpose(0, 1), pixel_decoder_output, self.training
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        # TODO: three flag logic possibly redundant
        if self.denoise_queries_flag and self.training and denoise_metadata is not None:
            refpoint_embeddings = torch.cat([bbox_queries, refpoint_embeddings], dim=1)

        # NOTE: target = queries in this case (nn.Transformer vs DETR nomenclature)
        intermediates, reference_points_list = self.decoder(
            # (bs,d_model,num_queries+num_dn_queries)
            target=queries.transpose(0, 1),
            # (bs,c,SUM{dxhxw})
            memory=x_flatten.transpose(0, 1),
            # (bs,SUM{dxhxw})
            memory_key_padding_mask=mask_flatten,
            pos_embeddings=None,
            # (bs,6,num_queries)
            reference_points=refpoint_embeddings.transpose(0, 1),
            level_start_index=level_start_index,
            shapes=shapes,
            valid_ratios=valid_ratios,
            target_mask=attn_mask,
        )

        num_intermediates = len(intermediates)
        for idx, output in enumerate(intermediates):
            outputs_class, outputs_mask = self.forward_prediction_heads(
                output.transpose(0, 1), pixel_decoder_output, self.training or (idx == num_intermediates - 1)
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        if self.with_initial_prediction:
            # include refpoint embeddings in pred_box predictions
            # recall: refpoint_embeddings are given by encoded mask preds of encoder proposals
            # (+ optional denosing bboxes) or learned query embeddings
            # shape: # (num_layers (+1 if init), B, num_queries, 4 or 6)
            predictions_boxes = self.predict_bboxes(reference_points_list, intermediates, refpoint_embeddings.sigmoid())
            assert (
                len(predictions_class) == self.num_layers + 1
            ), "predictions_class should be of size self.num_layers + 1"
        else:
            predictions_boxes = self.predict_bboxes(reference_points_list, intermediates)

        if denoise_metadata is not None:
            predictions_mask = torch.stack(
                predictions_mask
            )  # (decode_layers, bs, num_queries + num_dn_queries, D, H, W)
            predictions_class = torch.stack(
                predictions_class
            )  # (decode_layers, bs, num_queries + num_dn_queries, class_num)
            predictions_class, predictions_boxes, predictions_mask = self.denoise_post_process(
                predictions_class, predictions_boxes, predictions_mask, denoise_metadata
            )
            predictions_class, predictions_mask = list(predictions_class), list(predictions_mask)

        # ensures self.label_embeddings is marked as used for computation graph
        elif self.training:
            predictions_class[-1] += 0.0 * self.label_embeddings.weight.sum()

        outputs = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "pred_boxes": predictions_boxes[-1],
            "auxiliary_outputs": self._set_aux_loss(predictions_class, predictions_mask, predictions_boxes),
        }

        if self.two_stage_flag:
            outputs["intermediates"] = intermediate_outputs

        return outputs, denoise_metadata
