import copy
import math
import contextlib
import functools
from collections import defaultdict
from typing import List, Dict

from omegaconf import DictConfig, OmegaConf
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.functional import interpolate
from torch.utils.checkpoint import checkpoint

from cell_observatory_platform.data.masking.mask_generator import apply_masks
from cell_observatory_platform.data.structures import (
    bbox2delta,
    box_cxcyczwhd_to_xyzxyz,
    box_xyzxyz_to_cxcyczwhd,
    generalized_box_iou,
    generalized_box_iou_diag,
)
from cell_observatory_platform.models.layers.matchers import build_plain_detr_matcher
from cell_observatory_platform.models.layers.patch_embeddings import PatchEmbedding, calc_num_patches
from cell_observatory_platform.models.layers.utils import (
    batch_tensors,
    get_uncertain_point_coords_with_randomness,
    point_sample_labelmap_batched,
    point_sample,
)
from cell_observatory_platform.models.ops.losses import (
    batch_dice_loss,
    batch_sigmoid_ce_loss,
    calculate_uncertainty,
    dice_loss,
    sigmoid_ce_loss,
    sigmoid_focal_loss,
    iou_loss
)
from cell_observatory_platform.training.helpers import get_patch_sizes
from cell_observatory_platform.utils.context import get_world_size, is_torch_dist_initialized, process_rank


# adapted from: https://github.com/pytorch/torchtitan/torchtitan/components/loss.py
class RescaleAccumulatedLoss:
    def __init__(self, unwrapped_loss_fn, accumulation_steps):
        self.skip_rescale = False
        self.unwrapped_loss_fn = unwrapped_loss_fn
        self.accumulation_steps = accumulation_steps

        functools.update_wrapper(self, unwrapped_loss_fn, updated=tuple())

    def __call__(self, *args, **kwargs):
        result = self.unwrapped_loss_fn(*args, **kwargs)
        if self.skip_rescale:
            return result

        if isinstance(result, tuple):
            loss, *rest = result
            loss = loss / self.accumulation_steps
            return (loss, *rest)
        else:
            return result / self.accumulation_steps

    @contextlib.contextmanager
    def no_rescale(self):
        """Context manager for disabling rescaling"""
        previous = self.skip_rescale
        self.skip_rescale = True
        try:
            yield
        finally:
            self.skip_rescale = previous


def get_loss_fn(loss):
    if isinstance(loss, str):
        mapping = {
            "l2_masked": L2_masked_loss,
            "l1_masked": L1_masked_loss,
            "smooth_l1_masked": smooth_L1_masked_loss,
            "sigmoid_focal_loss": sigmoid_focal_loss,
            "sigmoid_ce_loss": sigmoid_ce_loss,
            "dice_loss": dice_loss,
        }
        if loss in mapping:
            return mapping[loss]
        raise ValueError(f"Unknown loss type: {loss}")

    rescale = False
    if isinstance(loss, DictConfig):
        rescale = bool(loss.get("rescale", False))
        loss = OmegaConf.to_container(loss, resolve=True)

    if isinstance(loss, dict) and loss.get("loss_type") == "fourier_loss":
        fourier_loss = FourierLoss(
            alpha=loss.get("alpha", 0.001),
            fft_loss=loss.get("fft_loss", "l1_masked"),
            spatial_loss=loss.get("spatial_loss", "l2_masked"),
            input_fmt=loss["input_fmt"],
            input_shape=loss["input_shape"],
            patch_shape=loss["patch_shape"],
            embed_dim=loss["embed_dim"],
        )

        if rescale:
            accumulation_steps = loss.get("accumulation_steps", None)
            if accumulation_steps is None:
                raise ValueError(
                    "Loss config has rescale=True but accumulation_steps was not provided " "to get_loss_fn()."
                )
            return RescaleAccumulatedLoss(fourier_loss, accumulation_steps)

        return fourier_loss

    raise ValueError(f"Unknown loss configuration: {loss}")


def L2_masked_loss(targets, predictions, num_patches, aux_loss_meta=None):
    if isinstance(targets, (list, tuple)) and isinstance(predictions, (list, tuple)):
        total_loss = 0.0
        for t, p in zip(targets, predictions):
            total_loss = total_loss + ((t - p) ** 2).mean(dim=-1).sum()
        return total_loss / num_patches, None
    elif isinstance(targets, torch.Tensor) and isinstance(predictions, torch.Tensor):
        loss = (targets - predictions) ** 2
        loss = loss.mean(dim=-1) # mean loss per patch
        loss = loss.sum() / num_patches
        return loss, None
    else:
        raise TypeError(
            f"targets and predictions must both be tensors or both be lists of tensors; "
            f"got {type(targets)}, {type(predictions)}"
        )


def L1_masked_loss(targets, predictions, num_patches, aux_loss_meta=None):
    if isinstance(targets, (list, tuple)) and isinstance(predictions, (list, tuple)):
        total_loss = 0.0
        for t, p in zip(targets, predictions):
            total_loss = total_loss + torch.abs(t - p).mean(dim=-1).sum()
        return total_loss / num_patches, None
    elif isinstance(targets, torch.Tensor) and isinstance(predictions, torch.Tensor):
        loss = torch.abs(targets - predictions)
        loss = loss.mean(dim=-1)
        loss = loss.sum() / num_patches
        return loss, None
    else:
        raise TypeError(
            f"targets and predictions must both be tensors or both be lists of tensors; "
            f"got {type(targets)}, {type(predictions)}"
        )


def smooth_L1_masked_loss(targets, predictions, num_patches, aux_loss_meta=None):
    if isinstance(targets, (list, tuple)) and isinstance(predictions, (list, tuple)):
        total_loss = 0.0
        for t, p in zip(targets, predictions):
            total_loss = total_loss + F.smooth_l1_loss(t, p, reduction="sum")
        return total_loss / num_patches, None
    elif isinstance(targets, torch.Tensor) and isinstance(predictions, torch.Tensor):
        return F.smooth_l1_loss(targets, predictions), None
    else:
        raise TypeError(
            f"targets and predictions must both be tensors or both be lists of tensors; "
            f"got {type(targets)}, {type(predictions)}"
        )


class FourierLoss(torch.nn.Module):
    def __init__(self, alpha, fft_loss, spatial_loss, input_fmt, input_shape, patch_shape, embed_dim):
        super(FourierLoss, self).__init__()
        self.loss_type = "fourier_loss"

        self.input_fmt = input_fmt
        self.input_shape = input_shape
        self.patch_shape = patch_shape

        self.embed_dim = embed_dim

        axis_to_value = dict(zip(input_fmt, input_shape))
        self.in_chans = axis_to_value["C"]
        self.num_frames = axis_to_value.get("T", None)

        self.temporal_patch_size, self.axial_patch_size, self.lateral_patch_size = get_patch_sizes(
            input_format=input_fmt, patch_shape=patch_shape
        )
        _, self.token_shape = calc_num_patches(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            patch_shape=patch_shape,
        )
        self.pe_unpatchify = functools.partial(
            PatchEmbedding.unpatchify,
            temporal_patch_size=self.temporal_patch_size,
            axial_patch_size=self.axial_patch_size,
            lateral_patch_size=self.lateral_patch_size,
            token_shape=self.token_shape,
            input_format=self.input_fmt,
            out_channels=None,
        )

        self.alpha = alpha
        self.fft_loss = fft_loss
        if spatial_loss == "l1_masked":
            self.spatial_loss = get_loss_fn("l1_masked")
        elif spatial_loss == "l2_masked":
            self.spatial_loss = get_loss_fn("l2_masked")
        else:
            raise ValueError(f"Unknown spatial loss type: {spatial_loss}")
        
        if self.fft_loss not in ["l1_masked", "l2_masked"]:
            raise ValueError(f"Unknown fft loss type: {self.fft_loss}")

    def forward(self, targets, predictions, num_patches, aux_loss_meta):
        full_targets, full_predictions = aux_loss_meta["targets"], aux_loss_meta["predictions"]

        full_targets = self.pe_unpatchify(full_targets)
        full_predictions = self.pe_unpatchify(full_predictions)

        # NOTE: works for ZYXC and TZYXC formats
        full_targets_fft = torch.fft.fftn(full_targets.to(torch.float32), dim=(-4, -3, -2))
        full_predictions_fft = torch.fft.fftn(full_predictions.to(torch.float32), dim=(-4, -3, -2))

        full_targets_fft = torch.abs(full_targets_fft)
        full_predictions_fft = torch.abs(full_predictions_fft)

        if self.fft_loss == "l1_masked":
            fft_loss = (full_targets_fft - full_predictions_fft).abs().mean()
        elif self.fft_loss == "l2_masked":
            fft_loss = ((full_targets_fft - full_predictions_fft) ** 2).mean()
        else:
            raise ValueError(f"Unknown fft loss type: {self.fft_loss}, {type(self.fft_loss)}")

        spatial_loss, _ = self.spatial_loss(targets, predictions, num_patches, aux_loss_meta=None)

        fft_loss = self.alpha * fft_loss
        spatial_loss = (1 - self.alpha) * spatial_loss

        aux_losses = {"fft_loss": fft_loss, "spatial_loss": spatial_loss}

        loss = fft_loss + spatial_loss
        return loss, aux_losses


class DETR_Set_Loss(nn.Module):
    """
    This class computes the loss for DETR.

    The process happens in two steps:
        1) Compute hungarian assignment between ground truth boxes and the outputs of the model
        2) Supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        loss_weight_dict,
        no_object_loss_weight,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
        denoise: bool = False,
        with_segmentation: bool = True,
        denoise_losses=[],
        semantic_ce_loss=True,
        focal_alpha: float = 0.25,
    ):
        super().__init__()

        self.matcher = matcher
        self.num_classes = num_classes

        self.with_segmentation = with_segmentation

        if self.with_segmentation:
            self.costs = ["cls", "box", "mask"]
        else:
            self.costs = ["cls", "box"]

        self.losses = losses
        self.loss_weight_dict = loss_weight_dict
        self.no_object_loss_weight = no_object_loss_weight

        self.denoise = denoise
        self.denoise_losses = denoise_losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.no_object_loss_weight
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.focal_alpha = focal_alpha
        self.semantic_ce_loss = semantic_ce_loss

    def loss_labels_ce(self, outputs, targets, indices, num_boxes):
        """
        Classification Loss: Cross Entropy Loss
        """
        # model predictions: (B, num_queries, num_classes)
        source_logits = outputs["pred_logits"].float()

        # idx is a tuple (batch_idx, src_idx), batch_idx is the index of the batch
        # for a given set of matched source and target indices
        # hence idx = (B, num_queries) where each element is batch idx, source query
        query_indices = self._get_query_indices(indices)

        # get the labels for all targets that were matched to the source indices
        # indices is a list of tuples (src_idx, tgt_idx) where src_idx are the indices of
        # the source boxes that were matched to the target boxes, and similar for tgt_idx
        target_labels = torch.cat(
            [target["labels"][matched_target_idx] for target, (_, matched_target_idx) in zip(targets, indices)]
        )
        # make tensor (B, num_queries) with values equal to num_classes for all elements
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        # for each (batch_idx, source_idx) tuple we set the corresponding target label
        # in target_classes to the value of target_labels, thus target_classes
        # is now of the form (batch_idx, num_queries) = matched target label
        target_classes[query_indices] = target_labels

        # compute cross entropy loss between source logits (B, num_queries, num_classes) and target_classes (B, num_queries)
        loss_ce = F.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss: Binary Focal Loss
        """
        source_logits = outputs["pred_logits"]
        query_indices = self._get_query_indices(indices)

        target_labels = torch.cat(
            [target["labels"][matched_target_idx] for target, (_, matched_target_idx) in zip(targets, indices)]
        )
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[query_indices] = target_labels

        # create one-hot encoding of target classes, add 1 extra dimension for the no-object class
        target_classes_onehot = torch.zeros(
            [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        # scatter_ to write 1s in the correct class slot for each query (we write to channel dim, i.e. dim=2)
        # we need index tensor of shape (B, Q, 1) to tell scatter_ where to put the 1
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        # drop the last channel since focal loss is computed over the real object classes
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = (
            sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * source_logits.shape[1]
        )
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute L1 regression loss and the GIoU loss over bounding box coordinates.
        Target boxes are expected in format (center_x, center_y, center_z, w, h, d), normalized by the image size.
        """
        query_indices = self._get_query_indices(indices)
        source_boxes = outputs["pred_boxes"][query_indices]
        target_boxes = torch.cat(
            [target["boxes"][matched_target_idx] for target, (_, matched_target_idx) in zip(targets, indices)], dim=0
        )

        losses = {}
        # L1 loss over bounding box corordinates, normalized by nr. of boxes
        loss_bbox = F.l1_loss(source_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # generalized IoU loss over bounding box corordinates, normalized by nr. of boxes
        # generalized box IOU (https://giou.stanford.edu/):
        # 1. get intersection volume for boxes A,B
        # 2. get union volume for boxes A,B
        # 3. find the smallest box that encloses both A and B (convex hull)
        # 4. compute GIOU = IoU−|C∖(A∪B)||C| where C is the convex hull
        # generalized_box_iou call returns an MxM matrix comparing every source against every target
        # we hence take the diagonal to get the IoU for each matched pair
        loss_giou = 1 - generalized_box_iou_diag(
            box_cxcyczwhd_to_xyzxyz(source_boxes),
            box_cxcyczwhd_to_xyzxyz(target_boxes),
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses
        
    # NOTE: Legacy binary mask-based sampling 
    # def loss_masks(self, outputs, targets, indices, num_masks):
    #     """
    #     Compute mask loss: Focal Loss and Dice Loss.
    #     """
    #     query_indices = self._get_query_indices(indices)
    #     target_class_indices = self._get_target_class_indices(indices)

    #     source_masks = outputs["pred_masks"][query_indices]
    #     masks = [target["masks"] for target in targets]

    #     # TODO: use valid to mask invalid areas due to padding in loss
    #     target_masks, valid = batch_tensors(masks)
    #     target_masks = target_masks.to(source_masks)
    #     target_masks = target_masks[target_class_indices]

    #     # no need to upsample predictions as we are using normalized coordinates
    #     # source/target masks: (N, 1, D, H, W)
    #     source_masks = source_masks[:, None]
    #     target_masks = target_masks[:, None]

    #     # Motivated by PointRend & Implicit PointRend
    #     # train with mask loss calculated on K randomly
    #     # sampled points instead of whole mask
    #     with torch.no_grad():
    #         # sample point_coordinates
    #         point_coords = get_uncertain_point_coords_with_randomness(
    #             source_masks,
    #             lambda logits: calculate_uncertainty(logits),
    #             self.num_points,  # K
    #             self.oversample_ratio,
    #             # ratio of points that are sampled via importance sampling
    #             self.importance_sample_ratio,
    #         )
    #         # samples from target mask at point_coords
    #         point_labels = point_sample(
    #             target_masks,
    #             point_coords,
    #             align_corners=False,
    #         ).squeeze(1)

    #     # samples from source mask at point_coords
    #     point_logits = point_sample(
    #         source_masks,
    #         point_coords,
    #         align_corners=False,
    #     ).squeeze(1)

    #     # compute losses: cross entropy classifcation loss and dice mask loss
    #     losses = {
    #         "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
    #         "loss_dice": dice_loss(point_logits, point_labels, num_masks),
    #     }

    #     del source_masks, target_masks
    #     return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """
        Compute mask loss: Focal Loss and Dice Loss.
        """
        query_indices = self._get_query_indices(indices)
        source_masks = outputs["pred_masks"][query_indices]
        
        # source masks: (N, 1, D, H, W)
        source_masks = source_masks[:, None]

        # NOTE: Efficient labelmap-based sampling (avoids materializing binary masks)
        # Gather matched instance IDs (actual labelmap values, not indices)
        batch_indices_list = []
        instance_ids_list = []
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if tgt_idx.numel() > 0:
                # nr. of batch i elements in tgt_idx
                batch_indices_list.append(torch.full_like(tgt_idx, i))
                # labelmap values for each target in batch i
                instance_ids_list.append(targets[i]["mask_ids"][tgt_idx])
        
        if not instance_ids_list:
            # No matches - return zero loss
            return {
                "loss_mask": source_masks.new_tensor(0.0), 
                "loss_dice": source_masks.new_tensor(0.0),
            }
        
        # batch_indices: [N_total_targets], instance_ids: [N_total_targets]
        batch_indices = torch.cat(batch_indices_list)
        instance_ids = torch.cat(instance_ids_list)
        # stack labelmaps into a single tensor of shape (B, Z, Y, X)
        labelmap = torch.stack([target["label_map"] for target in targets])

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                coarse_logits=source_masks,
                uncertainty_func=calculate_uncertainty,
                num_points=self.num_points,
                oversample_ratio=self.oversample_ratio,
                importance_sample_ratio=self.importance_sample_ratio,
            )
            point_labels = point_sample_labelmap_batched(
                labelmap=labelmap,
                point_coords=point_coords,
                batch_indices=batch_indices,
                instance_ids=instance_ids,
            )


        point_logits = point_sample(
            source_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        
        losses = {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }
        
        del source_masks
        return losses

    def preprocess_masks(self, mask_dict):
        predicted_denoise_outputs, denoise_target_indices = (
            mask_dict["predicted_denoise_outputs"],
            mask_dict["denoise_target_indices"],
        )
        max_query_pad_size, denoise_queries_per_label = (
            mask_dict["max_query_pad_size"],
            mask_dict["denoise_queries_per_label"],
        )
        max_num_labels = max_query_pad_size // denoise_queries_per_label
        num_targets = denoise_target_indices.numel()
        return predicted_denoise_outputs, num_targets, max_num_labels, denoise_queries_per_label

    def _get_query_indices(self, indices):
        batch_indices = torch.cat(
            [torch.full_like(source_idx, i) for i, (source_idx, target_idx) in enumerate(indices)]
        )
        source_indices = torch.cat([source_idx for (source_idx, target_idx) in indices])
        return batch_indices, source_indices

    def _get_target_class_indices(self, indices):
        batch_indices = torch.cat(
            [torch.full_like(target_idx, i) for i, (source_idx, target_idx) in enumerate(indices)]
        )
        target_indices = torch.cat([target_idx for (source_idx, target_idx) in indices])
        return batch_indices, target_indices

    def compute_loss(self, loss, outputs, targets, indices, num_masks):
        loss_dict = {
            "labels": self.loss_labels_ce if self.semantic_ce_loss else self.loss_labels,
            "masks": self.loss_masks,
            "boxes": self.loss_boxes,
        }
        return loss_dict[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, denoise_predictions=None):
        outputs_without_aux_data = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # compute loss for denoising and mask predictions
        if self.denoise and denoise_predictions is not None:
            predicted_denoise_outputs, num_targets, max_num_labels, denoise_queries_per_label = (
                self.preprocess_masks(denoise_predictions)
            )
            denoise_query_target_indices = []
            for target in targets:
                # we have L target labels and for each label we create denoise_queries_per_label queries
                # hence we have L * denoise_queries_per_label queries in total, here we get their indices
                # however, in decoder each batch element has a different number of target labels, so we need to pad
                target_labels = target["labels"]
                if len(target_labels) > 0:
                    # (num_target_labels, ) = [0,1,...,num_target_labels-1]
                    target_label_indices = torch.arange(0, len(target_labels)).long().cuda()
                    # (1, num_target_labels) -> (denoise_queries_per_label, num_target_labels)
                    target_label_indices = target_label_indices.unsqueeze(0).repeat(denoise_queries_per_label, 1)
                    # (denoise_queries_per_label, num_target_labels) -> (denoise_queries_per_label * num_target_labels)
                    denoise_query_target_index = target_label_indices.flatten()
                    # shifts the target_label_indices by multiples of max_num_labels to account for padding to max_num_labels
                    padded_denoise_query_target_index = (
                        (torch.tensor(range(denoise_queries_per_label)) * max_num_labels)
                        .long()
                        .cuda()
                        .unsqueeze(1)
                    )
                    # pad target label indices
                    padded_denoise_query_target_index = padded_denoise_query_target_index + target_label_indices
                    padded_denoise_query_target_index = padded_denoise_query_target_index.flatten()
                else:
                    padded_denoise_query_target_index = denoise_query_target_index = torch.tensor([]).long().cuda()
                denoise_query_target_indices.append((padded_denoise_query_target_index, denoise_query_target_index))

        # use Hungarian matcher to compute the indices of the matched predictions and targets
        matched_target_indices = self.matcher(outputs_without_aux_data, targets, costs=self.costs)

        # compute number of target boxes accross all nodes for normalization
        total_num_masks = sum(len(target["labels"]) for target in targets)
        total_num_masks = torch.as_tensor(
            [total_num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )

        if is_torch_dist_initialized():
            dist.all_reduce(total_num_masks)
        average_num_masks_per_device = torch.clamp(total_num_masks / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.compute_loss(loss, outputs, targets, matched_target_indices, average_num_masks_per_device))

        # compute denosing losses if denoise is enabled
        if self.denoise and denoise_predictions is not None:
            extra_losses = {}
            for loss in self.denoise_losses:
                extra_losses.update(
                    self.compute_loss(
                        loss,
                        predicted_denoise_outputs,
                        targets,
                        denoise_query_target_indices,
                        average_num_masks_per_device * denoise_queries_per_label,
                    )
                )
            extra_losses = {k + f"_denoise": v for k, v in extra_losses.items()}
            losses.update(extra_losses)

        # compute loss for denoising
        elif self.denoise:
            extra_losses = dict()
            extra_losses["loss_bbox_denoise"] = torch.as_tensor(0.0).to("cuda")
            extra_losses["loss_giou_denoise"] = torch.as_tensor(0.0).to("cuda")
            extra_losses["loss_ce_denoise"] = torch.as_tensor(0.0).to("cuda")
            if self.with_segmentation:
                extra_losses["loss_mask_denoise"] = torch.as_tensor(0.0).to("cuda")
                extra_losses["loss_dice_denoise"] = torch.as_tensor(0.0).to("cuda")

            losses.update(extra_losses)

        # in case of auxiliary losses, we repeat loss computation with the output of intermediate layers
        # we do not do denoising for auxiliary outputs so we do not compute if not training
        if "auxiliary_outputs" in outputs and self.training:
            # skip the first auxiliary output (pre-decoder predictions using output_memory_topk OR learned queries)
            # if "intermediates" not in outputs, i.e. if we did not do two-stage pipeline 
            first_auxiliary_output_idx = 0 if "intermediates" in outputs else 1
            for i, auxiliary_output in enumerate(outputs["auxiliary_outputs"]):
                # hungarian matcher to get indices of the matched auxiliary_outputs and targets
                auxiliary_matched_target_indices = self.matcher(auxiliary_output, targets, costs=self.costs)
                for loss in self.losses:
                    extra_losses = self.compute_loss(
                        loss, auxiliary_output, targets, auxiliary_matched_target_indices, average_num_masks_per_device
                    )
                    extra_losses = {k + f"_{i}": v for k, v in extra_losses.items()}
                    losses.update(extra_losses)

                if i >= first_auxiliary_output_idx:
                    if self.denoise and denoise_predictions is not None:
                        auxiliary_predicted_denoise_outputs = predicted_denoise_outputs["auxiliary_outputs"][i]
                        extra_losses = {}
                        for loss in self.denoise_losses:
                            extra_losses.update(
                                self.compute_loss(
                                    loss,
                                    auxiliary_predicted_denoise_outputs,
                                    targets,
                                    denoise_query_target_indices,
                                    average_num_masks_per_device * denoise_queries_per_label,
                                )
                            )
                        extra_losses = {k + f"_denoise_{i}": v for k, v in extra_losses.items()}
                        losses.update(extra_losses)

                    elif self.denoise:
                        extra_losses = dict()
                        extra_losses[f"loss_bbox_denoise_{i}"] = torch.as_tensor(0.0).to("cuda")
                        extra_losses[f"loss_giou_denoise_{i}"] = torch.as_tensor(0.0).to("cuda")
                        extra_losses[f"loss_ce_denoise_{i}"] = torch.as_tensor(0.0).to("cuda")
                        if self.with_segmentation:
                            extra_losses[f"loss_mask_denoise_{i}"] = torch.as_tensor(0.0).to("cuda")
                            extra_losses[f"loss_dice_denoise_{i}"] = torch.as_tensor(0.0).to("cuda")

                        losses.update(extra_losses)

        # initial encoder predictions
        if "intermediates" in outputs:
            intermediate_outputs = outputs["intermediates"]
            intermediate_matched_target_indices = self.matcher(intermediate_outputs, targets, costs=self.costs)
            for loss in self.losses:
                extra_losses = self.compute_loss(
                    loss, intermediate_outputs, targets, 
                    intermediate_matched_target_indices, average_num_masks_per_device
                )
                extra_losses = {k + f"_intermediate": v for k, v in extra_losses.items()}
                losses.update(extra_losses)

        return losses


class PlainDETR_Set_Loss(nn.Module):
    """
    Computes the loss for PlainDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, reparam=False):
        """
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            loss_bbox_type: how to perform loss_bbox
        """
        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.loss_bbox_type = "l1" if (not reparam) else "reparam"

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss
        """
        assert "pred_logits" in outputs, "Predictions must contain pred_logits"
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes: [B, Q] with values in [0, num_classes]
        # where num_classes is no-object class
        target_classes = torch.full(
            src_logits.shape[:2],  # [B, Q]
            self.num_classes,  # fill with no-object class
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        # target_classes_onehot: [B, Q, num_classes+1]
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        # scatter_(dim=2, index, 1) writes a 1 at the appropriate
        # class channel, 0 elsewhere
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # focal loss is applied only over the real classes, not the no-object class
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes.
        For logging purposes only. It doesn't propagate gradients.
        """
        pred_logits = outputs["pred_logits"]
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=pred_logits.device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes: L1 regression loss and the GIoU loss.
        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 6].
        The target boxes are expected in format (center_x, center_y, center_z, h, w, d), normalized by the image size.
        """
        assert "pred_boxes" in outputs, "Predictions must contain pred_boxes"

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        if self.loss_bbox_type == "l1":
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        elif self.loss_bbox_type == "reparam":
            src_deltas = outputs["pred_deltas"][idx]
            src_boxes_old = outputs["pred_boxes_old"][idx]
            target_deltas = bbox2delta(src_boxes_old, target_boxes)
            loss_bbox = F.l1_loss(src_deltas, target_deltas, reduction="none")
        else:
            raise NotImplementedError

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - generalized_box_iou_diag(
            box_cxcyczwhd_to_xyzxyz(src_boxes),
            box_cxcyczwhd_to_xyzxyz(target_boxes)
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w, d]
        """
        assert "pred_masks" in outputs, "Predictions must contain pred_masks"

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        source_masks = outputs["pred_masks"]
        masks = [target["masks"] for target in targets]
        # TODO: use valid to mask invalid areas due to padding in loss
        target_masks, valid = batch_tensors(masks)
        target_masks = target_masks.to(source_masks)

        source_masks = source_masks[src_idx]
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(
            source_masks[:, None],
            size=target_masks.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )

        # src_masks/target_masks: (N, D, H, W) -> (N, D*H*W)
        src_masks = src_masks[:, 0].flatten(1)
        target_masks = target_masks.flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"Unknown loss: {loss}"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"}

        # matching between the outputs of the last layer and the targets
        # matcher sees pred_logits and pred_boxes
        indices = self.matcher(outputs_without_aux, targets)

        # average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_torch_dist_initialized():
            dist.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        if "aux_outputs" in outputs:
            # NOTE: supervises intermediate layers (one2one and one2many)
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs["log"] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            # supervise initial predictions from transformer 2-stage pipeline
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == "masks":
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs["log"] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def build_plainDETR_Set_Loss(
    args,
    num_classes,
    two_stage,
    reparam,
    aux_loss,
    dec_layers,
    masks=False,
):
    matcher = build_plain_detr_matcher(args.matcher_args)

    # --- base loss weights ---
    weight_dict = dict(args.weight_dict).copy()
    if masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    # --- auxiliary decoder layer losses ---
    if aux_loss:
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({f"{k}_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # --- encoder outputs (enc_outputs) ---
    if two_stage:
        enc_weight_dict = {f"{k}_enc": v for k, v in weight_dict.items() if not k.endswith("_enc")}
        weight_dict.update(enc_weight_dict)

    # --- one-to-many ---
    new_dict = {}
    for key, value in weight_dict.items():
        new_dict[key] = value
        new_dict[key + "_one2many"] = value
    weight_dict = new_dict

    # which losses are computed
    losses = ["labels", "boxes", "cardinality"]
    if masks:
        losses.append("masks")

    criterion_args = dict(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        focal_alpha=args.focal_alpha,
        reparam=reparam,
    )
    criterion = PlainDETR_Set_Loss(**criterion_args)
    return criterion


class Mask2FormerSetLoss(nn.Module):
    """
    This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self, 
        num_classes: int,
        matcher: nn.Module,
        loss_weight_dict: dict,
        no_object_loss_weight: float,
        losses: list[str],
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
    ):
        """
        Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            loss_weight_dict: dict containing as key the names of the losses and as values their relative weight.
            no_object_loss_weight: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_weight_dict = loss_weight_dict
        self.no_object_loss_weight = no_object_loss_weight
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.no_object_loss_weight
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        
        # Used to tell HungarianMatcher which costs to use
        self.costs = ["cls", "mask"]        

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # model predictions: (B, num_queries, num_classes)
        source_logits = outputs["pred_logits"].float()

        # idx is a tuple (batch_idx, src_idx), batch_idx is the index of the batch
        # for a given set of matched source and target indices
        # hence idx = (B, num_queries) where each element is batch idx, source query
        query_indices = self._get_query_indices(indices)

        # get the labels for all targets that were matched to the source indices
        # indices is a list of tuples (src_idx, tgt_idx) where src_idx are the indices of
        # the source boxes that were matched to the target boxes, and similar for tgt_idx
        target_labels = torch.cat(
            [
                target["labels"][matched_target_idx]
                for target, (_, matched_target_idx) in zip(targets, indices)
            ]
        ).to(source_logits.device)
        # make tensor (B, num_queries) with values equal to num_classes for all elements
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        # for each (batch_idx, source_idx) tuple we set the corresponding target label
        # in target_classes to the value of target_labels, thus target_classes
        # is now of the form (batch_idx, num_queries) = matched target label
        target_classes[query_indices] = target_labels

        # compute cross entropy loss between source logits (B, num_queries, num_classes) and target_classes (B, num_queries)
        loss_ce = F.cross_entropy(
            source_logits.transpose(1, 2),
            target_classes,
            self.empty_weight.to(source_logits.dtype),
        )
        return {"loss_labels_ce": loss_ce}

    def loss_masks(self, outputs, targets, indices, num_masks):
        """
        Compute mask loss: Focal Loss and Dice Loss.
        """
        query_indices = self._get_query_indices(indices)
        target_class_indices = self._get_target_class_indices(indices)

        source_masks = outputs["pred_masks"][query_indices]
        masks = [target["masks"] for target in targets]

        # TODO: use valid to mask invalid areas due to padding in loss
        target_masks, valid = batch_tensors(masks)
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_class_indices]

        # no need to upsample predictions as we are using normalized coordinates
        # source/target masks: (N, 1, D, H, W)
        source_masks = source_masks[:, None]
        target_masks = target_masks[:, None]

        # Motivated by PointRend & Implicit PointRend
        # train with mask loss calculated on K randomly
        # sampled points instead of whole mask
        with torch.no_grad():
            # sample point_coordinates
            point_coords = get_uncertain_point_coords_with_randomness(
                source_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,  # K
                self.oversample_ratio,
                # ratio of points that are sampled via importance sampling
                self.importance_sample_ratio,
            )
            # samples from target mask at point_coords
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        # samples from source mask at point_coords
        point_logits = point_sample(
            source_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        # compute losses: cross entropy classifcation loss and dice mask loss
        losses = {
            "loss_mask_ce": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_mask_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del source_masks, target_masks
        return losses

    def _get_query_indices(self, indices):
        batch_indices = torch.cat(
            [torch.full_like(source_idx, i) for i, (source_idx, target_idx) in enumerate(indices)]
        )
        source_indices = torch.cat([source_idx for (source_idx, target_idx) in indices])
        return batch_indices, source_indices

    def _get_target_class_indices(self, indices):
        batch_indices = torch.cat(
            [torch.full_like(target_idx, i) for i, (source_idx, target_idx) in enumerate(indices)]
        )
        target_indices = torch.cat([target_idx for (source_idx, target_idx) in indices])
        return batch_indices, target_indices

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux_data = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # use Hungarian matcher to compute the indices of the matched predictions and targets
        matched_target_indices = self.matcher(outputs_without_aux_data, targets)

        # compute number of target boxes accross all nodes for normalization
        total_num_masks = sum(len(target["labels"]) for target in targets)
        total_num_masks = torch.as_tensor(
            [total_num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )

        if is_torch_dist_initialized():
            torch.distributed.all_reduce(total_num_masks)
        average_num_masks_per_node = torch.clamp(total_num_masks / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, matched_target_indices, average_num_masks_per_node))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["auxiliary_outputs"]):
                # hungarian matcher to get indices of the matched auxiliary_outputs and targets
                auxiliary_matched_target_indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    extra_losses = self.get_loss(loss, aux_outputs, targets, auxiliary_matched_target_indices, average_num_masks_per_node)
                    extra_losses = {k + f"_{i}": v for k, v in extra_losses.items()}
                    losses.update(extra_losses)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "loss_weight_dict: {}".format(self.loss_weight_dict),
            "num_classes: {}".format(self.num_classes),
            "no_object_loss_weight: {}".format(self.no_object_loss_weight),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class MultiLabelBinaryPredictionLoss(nn.Module):
    def __init__(
        self,
        loss_dict: dict[str, dict[str, Any]],
        loss_weight_dict: dict[str, float],
        num_classes: int,
        num_points: int,
        oversample_ratio: int,
        importance_sample_ratio: float,
        ):
        super().__init__()
        self.loss_fns: Dict[str, Callable] = {}
        for loss_name, loss_params in loss_dict.items():
            loss_fn = functools.partial(get_loss_fn(loss_name), **loss_params)
            self.loss_fns[loss_name] = loss_fn
        self.loss_weight_dict = loss_weight_dict
        self.num_classes = num_classes
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_binary_predictions(
        self,
        outputs: Dict[str, torch.Tensor], 
        targets: List[Dict[str, torch.Tensor]]
        ):
        """
        Compute binary mask prediction losses.
        """
        B = len(targets)
        N_classes = targets[0]["masks"].shape[0]

        source_masks = outputs["pred_masks"] # B, N_classes, spatial
        masks = [target["masks"] for target in targets]
        target_masks = torch.stack(masks, dim=0).to(source_masks) # B, N_classes, spatial
        assert N_classes == self.num_classes, f"Expected {self.num_classes} classes, got {N_classes}"

        # Flatten masks along the batch and class dimensions to (B*N_classes, D, H, W)
        source_masks = source_masks.flatten(0, 1) # (B*N_classes, D, H, W)
        target_masks = target_masks.flatten(0, 1) # (B*N_classes, D, H, W)

        # no need to upsample predictions as we are using normalized coordinates
        # source/target masks: (B*N_classes, 1, D, H, W)
        source_masks = source_masks[:, None]
        target_masks = target_masks[:, None]

        # Motivated by PointRend & Implicit PointRend
        # train with mask loss calculated on K randomly
        # sampled points instead of whole mask
        with torch.no_grad():
            # sample point_coordinates
            point_coords = get_uncertain_point_coords_with_randomness(
                source_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,  # K
                self.oversample_ratio,
                # ratio of points that are sampled via importance sampling
                self.importance_sample_ratio,
            ) # B*N_classes, P, 3
            # samples from target mask at point_coords
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        # samples from source mask at point_coords
        point_logits = point_sample(
            source_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {}
        for loss_name, loss_fn in self.loss_fns.items():
            losses[loss_name] = loss_fn(point_logits, point_labels, B*N_classes)
        return losses

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        outputs_without_aux_data = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        losses = self.loss_binary_predictions(outputs_without_aux_data, targets)
        
        if "auxiliary_outputs" in outputs:
            for i, aux_output in enumerate(outputs["auxiliary_outputs"]):
                aux_losses = self.loss_binary_predictions({"pred_masks": aux_output}, targets)
                # NOTE: In the paper they downweight lower resolution features like this. 
                # They actually make it so that they all sum to 1.
                # Here we just downweight them by 0.5**i without normalizing.
                # They note this only provides marginal improvement here: https://github.com/MIC-DKFZ/nnUNet/issues/1417
                aux_loss_weights = {
                    k + f"_{i}": self.loss_weight_dict[k] * 0.5**i 
                    for i, k in enumerate(aux_losses.keys())
                }
                self.loss_weight_dict.update(aux_loss_weights)
                aux_losses = {k + f"_{i}": v for k, v in aux_losses.items()}
                losses.update(aux_losses)

        return losses


# adapted from:
# https://github.com/facebookresearch/dinov3/dinov3/loss
class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        
        self.register_buffer("center", torch.full((1, out_dim), math.nan))
        
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    def init_weights(self) -> None:
        self.center.zero_()

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp, update_centers=True):
        if update_centers:
            self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        # teacher_output: [batch, prototypes]
        teacher_output = teacher_output.float()
        # NOTE: original refereence uses get_subgroup_size() instead of get_world_size()
        world_size = get_world_size() if is_torch_dist_initialized() else 1
        # NOTE: Q is K-by-B for consistency with notations from DINO paper
        Q = torch.exp(teacher_output / teacher_temp).t()
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if is_torch_dist_initialized():
            # NOTE: for distillation do: group=get_process_subgroup()
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for _ in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if is_torch_dist_initialized():
                # NOTE: for distillation do: group=get_process_subgroup()
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_logits, teacher_probs, ignore_diagonal=False):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.

        student_logits: [student crops, batch, prototypes]
        teacher_probs:  [teacher crops, batch, prototypes] must sum to 1 over the last dim

        loss = 0
        count = 0
        for each sample `b` in the batch:
            for each student crop `s` of this sample:
                for each teacher crop `t` of this sample:
                    if ignore_diagonal and s == t:
                        continue
                    loss += cross_entropy(softmax(student_logits[s, b] / student_temp), teacher_probs[t, b])
                    count += 1
        return loss / count
        """
        student_crops, B, K = student_logits.shape
        teacher_crops, _, _ = teacher_probs.shape
        student_logits = F.log_softmax(student_logits.float() / self.student_temp, dim=-1)
        if not ignore_diagonal:
            loss = -torch.einsum("s b k, t b k -> ", student_logits, teacher_probs)
            return loss / (B * student_crops * teacher_crops)
        else:
            loss = -torch.einsum("s b k, t b k -> s t", student_logits, teacher_probs)
            min_st = min(student_crops, teacher_crops)
            loss = torch.diagonal_scatter(loss, loss.new_zeros(min_st))
            return loss.sum() / (B * student_crops * teacher_crops - B * min_st)

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if is_torch_dist_initialized():
            # NOTE: for distillation do: group=get_process_subgroup()
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            # NOTE: original refereence uses get_subgroup_size() instead of get_world_size()
            world_size = get_world_size() if is_torch_dist_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True


def lossfunc(t, s, temp):
    return torch.sum(t.float() * F.log_softmax(s.float() / temp, dim=-1), dim=-1)


# NOTE: This is a module and not a function in the `iBOTPatchLoss` class
# This is because we want to torch.compile it, and torch.compil-ing a single
# function with the `@torch.compile` decorator is bad.
# It's better to `module.compile()` it, as we can control when we enable or
# disable compilation globally.
class SinkhornKnoppTeacher(nn.Module):
    @torch.no_grad()
    def forward(self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3):
        teacher_output = teacher_output.float()
        # world_size = dist.get_world_size() if is_torch_dist_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        # B = Q.shape[1] * world_size # number of samples to assign
        B = n_masked_patches_tensor
        # NOTE: for distillation do: group=get_process_subgroup()
        if is_torch_dist_initialized():
            dist.all_reduce(B)
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if is_torch_dist_initialized():
            # NOTE: for distillation do: group=get_process_subgroup()
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for _ in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if is_torch_dist_initialized():
                # NOTE: for distillation do: group=get_process_subgroup()
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()


class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        
        self.register_buffer("center", torch.full((1, 1, patch_out_dim), math.nan))
        
        self.updated = True
        self.reduce_handle = None
        
        self.len_teacher_patch_tokens = None
        
        self.async_batch_center = None
        
        self.sinkhorn_knopp_teacher = SinkhornKnoppTeacher()
        self.sinkhorn_knopp_teacher.compile()

    def init_weights(self) -> None:
        self.center.zero_()

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp, update_centers=True):
        if update_centers:
            self.apply_center_update()
        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.

        student_patch_tokens: (B, N, D) tensor
        teacher_patch_tokens: (B, N, D) tensor
        student_masks_flat: (B, N) tensor
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = lossfunc(t, s, self.student_temp)
        loss = torch.sum(loss * student_masks_flat.float(), dim=-1) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return -loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
    ):
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        # loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = lossfunc(t, s, self.student_temp)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        return -loss.sum() / student_masks_flat.shape[0]

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True)
        if is_torch_dist_initialized():
            # NOTE: for distillation do: group=get_process_subgroup()
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            # NOTE: original refereence uses get_subgroup_size() instead of get_world_size()
            world_size = get_world_size() if is_torch_dist_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True


class KoLeoLoss(nn.Module):
    """
    Kozachenko-Leonenko entropic loss regularizer from 
    Sablayrolles et al. (2018): "Spreading vectors for similarity search".
    """

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        _, indices = torch.max(dots, dim=1)  # max inner prod -> min distance
        return indices

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.autocast("cuda", enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            indices = self.pairwise_NNs_inner(student_output)
            distances = self.pdist(student_output, student_output[indices])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()
        return loss


class KoLeoLossDistributed(nn.Module):
    """
    Kozachenko-Leonenko entropic loss regularizer from 
    Sablayrolles et al. (2018): Spreading vectors for similarity search.
    """

    def __init__(self, topk=1, loss_group_size: int | None = None):
        super().__init__()
        
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)
        
        self.topk = topk
        # NOTE: Size of the nearest neighbor set. If None, uses global batch size.
        self.loss_group_size = loss_group_size

    def pairwise_NNs_inner(self, x, all_x, rank):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, all_x.t())  # local_B x global_B
        local_B, global_B = dots.shape
        dots.view(-1)[rank * local_B :: (global_B + 1)].fill_(-1)  # Trick to fill diagonal with -1
        _, indices = torch.topk(dots, dim=1, k=self.topk)  # max inner prod -> min distance
        return indices

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.autocast("cuda", enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)  # local_B x D

            if is_torch_dist_initialized():
                all_student_outputs = torch.cat(dist.nn.all_gather(student_output), dim=0)  # global_B x D
                world_size = get_world_size()
                rank = process_rank()
            else:
                all_student_outputs = student_output
                world_size = 1
                rank = 0

            # Group the global batch into groups of size `loss_group_size` and use the features of the group
            # the local rank falls into as the nearest neighbor set for the local rank
            local_B = len(student_output)
            global_B = len(all_student_outputs)
            loss_group_size = self.loss_group_size if self.loss_group_size is not None else global_B
            if loss_group_size % local_B != 0:
                raise ValueError(
                    f"Loss group size size {loss_group_size} must be a multiple of local batch size {local_B}."
                )
            if global_B % loss_group_size != 0:
                raise ValueError(
                    f"Global batch size {global_B} must be divisible by loss group size {loss_group_size}."
                )
            
            n_groups = global_B // loss_group_size
            ranks_per_group = world_size // n_groups
            rank_in_group = rank % ranks_per_group
            group = rank // ranks_per_group
            
            all_student_outputs = all_student_outputs.view(n_groups, loss_group_size, student_output.shape[1])
            all_student_outputs = all_student_outputs[group]  # loss_group_size x D

            with torch.no_grad():
                indices = self.pairwise_NNs_inner(student_output, all_student_outputs, rank_in_group)  # local_B x topk

            student_output_expanded = (
                student_output.unsqueeze(1).repeat(1, self.topk, 1).flatten(0, 1)
            )  # (local_B * topk) x D
            distances = self.pdist(student_output_expanded, all_student_outputs[indices].flatten(0, 1))  # BxD, BxD -> B
            loss = -torch.log(distances.float() + eps).mean()

        return loss


class MultiStepMultiMasksAndIousLoss(nn.Module):
    def __init__(
        self,
        input_fmt,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
        activation_checkpoint=False,
        # True  -> PointRend point-sampling focal/dice on uncertain points
        #          sampled from dense materialized masks (high-res stream).
        # False -> dense per-voxel focal/dice/IoU on materialized masks.
        use_point_sampling=False,
        # How to pick number of points (use_point_sampling=True only)?
        # Find the mask logit resolution loss samples from.
        # E.g., for volume 128×256×512, at stride 4, that is 32×64×128 = 262,144 candidate voxels.
        # 5% => ~13k
        # 10% => ~26k
        # 20% => ~52k
        num_points=20000,
        oversample_ratio=3,
        importance_sample_ratio=0.75,
        # ----- IoU-loss / mask-stream ablation flags ----- #
        # Independent of use_point_sampling (which only controls focal/dice).
        # All default to the legacy behavior so all-off == previous code.
        #
        # sample_iou: compute the IoU target on the same sampled points as
        #   focal/dice instead of the full [N,M,Z,Y,X] grid. Requires
        #   use_point_sampling=True (we reuse its point_logits/point_labels).
        sample_iou=False,
        # soft_iou: compute the IoU target from sigmoid probabilities
        #   instead of a hard `> 0` threshold. Independently selectable for
        #   both the dense and sampled IoU paths. Always runs under
        #   torch.no_grad() so it does not back-prop through src_masks.
        soft_iou=False,
        # low_res_multimasks: feed the low-res prediction stream
        #   (`multistep_pred_multimasks`) to the criterion instead of
        #   `multistep_pred_multimasks_high_res`. Requires
        #   use_point_sampling=True AND sample_iou=True because dense
        #   focal/dice/IoU cannot consume low-res predictions against
        #   full-resolution materialized GT masks.
        low_res_multimasks=False,
    ):
        """
        Computes the multi-step multi-mask and IoU losses.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            focal_alpha: alpha for sigmoid focal loss
            focal_gamma: gamma for sigmoid focal loss
            supervise_all_iou: if True, back-prop iou losses for all predicted masks
            iou_use_l1_loss: use L1 loss instead of MSE loss for iou
            pred_obj_scores: if True, compute loss for object scores
            focal_gamma_obj_score: gamma for sigmoid focal loss on object scores
            focal_alpha_obj_score: alpha for sigmoid focal loss on object scores
        """
        super().__init__()

        self.input_fmt = input_fmt

        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        assert "loss_mask" in self.weight_dict, "loss_mask must be in weight_dict"
        assert "loss_dice" in self.weight_dict, "loss_dice must be in weight_dict"
        assert "loss_iou" in self.weight_dict, "loss_iou must be in weight_dict"
        assert "loss_class" in self.weight_dict, "loss_class must be in weight_dict"

        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

        self.core_loss_key = "step_loss"

        # optimizations
        self.activation_checkpoint = activation_checkpoint
        self.use_point_sampling = use_point_sampling
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # ablation flags (see __init__ docstring above)
        self.sample_iou = sample_iou
        self.soft_iou = soft_iou
        self.low_res_multimasks = low_res_multimasks
        if self.sample_iou:
            assert self.use_point_sampling, (
                "sample_iou=True requires use_point_sampling=True "
                "(sampled IoU reuses focal/dice point_logits/point_labels)"
            )
        if self.low_res_multimasks:
            assert self.use_point_sampling and self.sample_iou, (
                "low_res_multimasks=True requires use_point_sampling=True and "
                "sample_iou=True; dense focal/dice/IoU cannot mix low-res "
                "predictions with full-resolution GT masks"
            )

    def forward(self, outs_batch: List[Dict], target_view: Dict):
        """
        Args:
            outs_batch: list of per-frame outputs from SAM2 tracking.
            target_view: SAM2 preprocessor target view dict with materialized
                `masks` (list[T] of [N,Z,Y,X]), `valid`, and `presence_t`.

        Mask focal/dice/IoU use `valid & presence_t` gates; object-score uses
        `valid` only. Denominators are the global gated row counts / world_size.
        Within each frame, `use_point_sampling` toggles PointRend vs dense voxel loss.
        """
        masks_list: List[torch.Tensor] = target_view["masks"] # list[T] [N,Z,Y,X]
        valid_list: List[torch.Tensor] = target_view["valid"] # list[T] [N] bool, is this a real object or padding?
        presence_list: List[torch.Tensor] = target_view["presence_t"] # list[T] [N] bool, is this object present in this frame?
        T = len(masks_list)
        assert len(outs_batch) == T, "outs_batch length must match num_frames"

        device = outs_batch[0]["multistep_pred_multimasks_high_res"][0].device

        mask_gate_total = torch.zeros((), device=device, dtype=torch.float)
        cls_gate_total = torch.zeros((), device=device, dtype=torch.float)
        for v_t, p_t in zip(valid_list, presence_list):
            mask_gate_total = mask_gate_total + (v_t & p_t).sum().to(torch.float)
            cls_gate_total = cls_gate_total + v_t.sum().to(torch.float)
        if is_torch_dist_initialized():
            dist.all_reduce(mask_gate_total)
            dist.all_reduce(cls_gate_total)
        mask_denom = torch.clamp(mask_gate_total / get_world_size(), min=1.0)
        cls_denom = torch.clamp(cls_gate_total / get_world_size(), min=1.0)

        losses = defaultdict(int)
        for t in range(T):
            outs = outs_batch[t]
            target_masks_t = masks_list[t]                       # [N, Z, Y, X]
            mask_gate = valid_list[t] & presence_list[t]         # [N] bool
            cls_gate = valid_list[t]                             # [N] bool
            if self.activation_checkpoint:
                cur_losses = self._forward_checkpoint(
                    outs, target_masks_t, mask_denom, mask_gate, cls_denom, cls_gate,
                )
            else:
                cur_losses = self._forward(
                    outs, target_masks_t, mask_denom, mask_gate, cls_denom, cls_gate,
                )
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    @torch.no_grad()
    def _sample_points_and_labels(
        self,
        src_masks: torch.Tensor,      # [N, M, Z, Y, X] logits
        target_masks: torch.Tensor,   # [N, 1, Z, Y, X] float/bool
    ):
        """
        Returns:
        point_coords_flat: [N*M, P, 3]  (coords per predicted mask)
        point_labels:      [N, M, P]    (GT sampled at those coords)
        """
        N, M, Z, Y, X = src_masks.shape
        P = self.num_points
        if P <= 0:
            raise ValueError("num_points must be > 0 when use_point_sampling=True")

        # coords are per (N*M) predicted mask
        src_flat = src_masks.detach().reshape(N * M, 1, Z, Y, X)
        point_coords_flat = get_uncertain_point_coords_with_randomness(
            src_flat,
            lambda logits: calculate_uncertainty(logits),
            P,
            self.oversample_ratio,
            self.importance_sample_ratio,
        )  # [N*M, P, 3]

        # Sample GT labels in ONE call per object by stacking coords along the point dimension:
        # [N*M, P, 3] -> [N, M*P, 3]
        point_coords_obj = point_coords_flat.view(N, M * P, 3)
        gt = target_masks  # [N,1,Z,Y,X]
        point_labels_obj = point_sample(gt, point_coords_obj, align_corners=False).squeeze(1)  # [N, M*P]
        point_labels = point_labels_obj.view(N, M, P)  # [N, M, P]

        return point_coords_flat, point_labels

    def _iou_loss(
        self,
        src_masks: torch.Tensor,          # [N, M, Z, Y, X] logits (unused if sample_iou)
        target_masks: torch.Tensor,       # [N, M, Z, Y, X] bool/float (unused if sample_iou)
        pred_ious: torch.Tensor,          # [N, M]
        mask_denom: torch.Tensor,         # 0-dim float
        point_logits: torch.Tensor = None,  # [N, M, P] (required if sample_iou)
        point_labels: torch.Tensor = None,  # [N, M, P] (required if sample_iou)
    ) -> torch.Tensor:
        """IoU head loss = L1/MSE(pred_ious, detached actual_ious) / mask_denom.

        Returns [N, M] per-(row, multimask) loss matching ``iou_loss(...)``'s
        ``loss_on_multimask=True`` contract so call sites are drop-in.

        When ``soft_iou=False`` and ``sample_iou=False`` this is exactly
        ``iou_loss(...)`` with ``loss_on_multimask=True`` -- identical numerics.
        """
        # Fast path: legacy dense hard IoU. Preserved bit-for-bit.
        if not self.soft_iou and not self.sample_iou:
            return iou_loss(
                src_masks, target_masks, pred_ious, mask_denom,
                loss_on_multimask=True, use_l1_loss=self.iou_use_l1_loss,
            )

        with torch.no_grad():
            if self.sample_iou:
                assert point_logits is not None and point_labels is not None, (
                    "sample_iou=True requires point_logits and point_labels"
                )
                pl = point_labels.to(point_logits.dtype)
                if self.soft_iou:
                    probs = point_logits.sigmoid()
                    inter = (probs * pl).sum(-1)
                    union = probs.sum(-1) + pl.sum(-1) - inter
                else:
                    pred_bin = (point_logits > 0).to(point_logits.dtype)
                    inter = (pred_bin * pl).sum(-1)
                    union = pred_bin.sum(-1) + pl.sum(-1) - inter
                actual_ious = inter / union.clamp_min(1e-6)
            else:
                # dense soft IoU on the full [N,M,Z,Y,X] grid.
                probs = src_masks.sigmoid()
                tgt = target_masks.to(probs.dtype)
                probs_f = probs.flatten(2)
                tgt_f = tgt.flatten(2)
                inter = (probs_f * tgt_f).sum(-1)
                union = probs_f.sum(-1) + tgt_f.sum(-1) - inter
                actual_ious = inter / union.clamp_min(1e-6)

        if self.iou_use_l1_loss:
            loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
        else:
            loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
        return loss / mask_denom  # [N, M]

    def _forward(self, outputs: Dict, targets: torch.Tensor, mask_denom, mask_gate, cls_denom, cls_gate):
        """
        Compute the losses related to the masks: the focal loss and the dice loss,
        and also the MAE or MSE loss between predicted IoUs and actual IoUs.

        Here "multistep_pred_multimasks_high_res" is a list of multimasks (tensors
        of shape [N, M, D, H, W], where M could be 1 or larger, corresponding to
        one or multiple predicted masks from a click.

        We back-propagate focal, dice losses only on the prediction channel
        with the lowest focal+dice loss between predicted mask and ground-truth.
        If `supervise_all_iou` is True, we backpropagate ious losses for all predicted masks.
        """
        mask_stream_key = (
            "multistep_pred_multimasks" if self.low_res_multimasks
            else "multistep_pred_multimasks_high_res"
        )
        src_masks_list = outputs[mask_stream_key]
        pred_dtype = src_masks_list[0].dtype
        pred_device = src_masks_list[0].device
        target_masks = targets.unsqueeze(1).to(device=pred_device, dtype=pred_dtype)
        if self.input_fmt == "TZYXC":
            assert target_masks.dim() == 5, f"Expected target_masks shape (N, 1, Z, Y, X), got {target_masks.shape}"
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list), \
            f"Expected len(src_masks_list) == len(ious_list), got {len(src_masks_list)} and {len(ious_list)}"
        assert len(object_score_logits_list) == len(ious_list), \
            f"Expected len(object_score_logits_list) == len(ious_list), got {len(object_score_logits_list)} and {len(ious_list)}"

        # accumulate the loss over prediction steps
        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, target_masks, ious, mask_denom, mask_gate,
                cls_denom, cls_gate, object_score_logits,
            )
        losses[self.core_loss_key] = self.reduce_loss(losses)
        # Cast step_loss to prediction dtype so backward matches model
        losses[self.core_loss_key] = losses[self.core_loss_key].to(src_masks_list[0].dtype)
        return losses

    def _forward_checkpoint(self, outputs, targets, mask_denom, mask_gate, cls_denom, cls_gate):
        mask_stream_key = (
            "multistep_pred_multimasks" if self.low_res_multimasks
            else "multistep_pred_multimasks_high_res"
        )
        src_masks_list = outputs[mask_stream_key]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        pred_dtype = src_masks_list[0].dtype
        pred_device = src_masks_list[0].device
        target_masks = targets.unsqueeze(1).to(device=pred_device, dtype=pred_dtype)  # [N,1,Z,Y,X]

        # weights tensor for dot product
        w = target_masks.new_tensor([
            float(self.weight_dict["loss_mask"]),
            float(self.weight_dict["loss_dice"]),
            float(self.weight_dict["loss_iou"]),
            float(self.weight_dict.get("loss_class", 0.0)),
        ])

        total = target_masks.new_zeros(())
        comp_sum_detached = target_masks.new_zeros(4)

        for src_masks, pred_ious, obj_logits in zip(src_masks_list, ious_list, object_score_logits_list):
            if self.use_point_sampling:
                with torch.no_grad():
                    point_coords_flat, point_labels = self._sample_points_and_labels(
                        src_masks, target_masks
                    )
                comps = checkpoint(
                    self._step_loss_components_points,
                    src_masks, pred_ious, obj_logits, target_masks, mask_denom, mask_gate,
                    cls_denom, cls_gate, point_coords_flat, point_labels,
                    use_reentrant=False,
                    preserve_rng_state=False,  # safe: no randomness inside checkpointed fn
                )
            else:
                comps = checkpoint(
                    self._step_loss_components,
                    src_masks, pred_ious, obj_logits, target_masks, mask_denom, mask_gate,
                    cls_denom, cls_gate,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )

            total = total + (comps * w).sum()
            comp_sum_detached = comp_sum_detached + comps.detach()

        out = {self.core_loss_key: total.to(pred_dtype)}
        out["loss_mask"]  = comp_sum_detached[0]
        out["loss_dice"]  = comp_sum_detached[1]
        out["loss_iou"]   = comp_sum_detached[2]
        out["loss_class"] = comp_sum_detached[3]
        return out

    def _step_loss_components_points(
        self,
        src_masks,            # [N, M, Z, Y, X]
        pred_ious,            # [N, M]
        object_score_logits,  # [N, 1] or [N]
        target_masks_base,    # [N, 1, Z, Y, X]
        mask_denom,           # num_objects, 0-dim cuda float
        mask_gate,
        cls_denom,
        cls_gate,
        point_coords_flat,    # [N*M, P, 3]
        point_labels,         # [N, M, P]
    ):
        N, M, Z, Y, X = src_masks.shape
        P = point_labels.shape[-1]

        src_flat = src_masks.reshape(N * M, 1, Z, Y, X)
        point_logits_flat = point_sample(src_flat, point_coords_flat, align_corners=False).squeeze(1)  # [N*M,P]
        point_logits = point_logits_flat.view(N, M, P)  # [N,M,P]

        loss_multimask = sigmoid_focal_loss(
            point_logits, point_labels, mask_denom,
            alpha=self.focal_alpha, gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_loss(
            point_logits, point_labels, mask_denom,
            loss_on_multimask=True,
        )

        # obj score loss
        if not self.pred_obj_scores:
            loss_class = loss_multimask.new_zeros(())
        else:
            n_obj = object_score_logits.shape[0]
            obj_logits_nm1 = object_score_logits.reshape(n_obj, 1, 1)
            target_obj_nm1 = cls_gate.to(obj_logits_nm1.dtype).reshape(n_obj, 1, 1)
            loss_class_row = sigmoid_focal_loss(
                obj_logits_nm1, target_obj_nm1, cls_denom,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
                loss_on_multimask=True,
            ).squeeze(-1)
            loss_class = (loss_class_row * cls_gate.to(loss_class_row.dtype)).sum()

        if self.sample_iou:
            # IoU target sampled at the same uncertain points as focal/dice.
            loss_multiiou = self._iou_loss(
                src_masks, None, pred_ious, mask_denom,
                point_logits=point_logits, point_labels=point_labels,
            )
        else:
            target_masks_iou = target_masks_base.expand_as(src_masks)
            loss_multiiou = self._iou_loss(
                src_masks, target_masks_iou, pred_ious, mask_denom,
            )

        if loss_multimask.size(1) > 1:
            combo = (
                loss_multimask * float(self.weight_dict["loss_mask"])
                + loss_multidice * float(self.weight_dict["loss_dice"])
            )
            best = torch.argmin(combo, dim=-1)  # [N]
            b = torch.arange(combo.size(0), device=combo.device)
            loss_mask = loss_multimask[b, best].unsqueeze(1)
            loss_dice = loss_multidice[b, best].unsqueeze(1)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[b, best].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        gate = mask_gate.to(loss_mask.dtype).view(-1, 1)
        loss_mask = (loss_mask * gate).sum()
        loss_dice = (loss_dice * gate).sum()
        loss_iou = (loss_iou * gate).sum()

        return torch.stack([loss_mask, loss_dice, loss_iou, loss_class])

    def _step_loss_components(
        self,
        src_masks,
        pred_ious,
        object_score_logits,
        target_masks,
        mask_denom,
        mask_gate,
        cls_denom,
        cls_gate,
    ):
        target_masks = target_masks.expand_as(src_masks)

        # [N, M]
        loss_multimask = sigmoid_focal_loss(
            src_masks, target_masks, mask_denom,
            alpha=self.focal_alpha, gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_loss(
            src_masks, target_masks, mask_denom,
            loss_on_multimask=True,
        )

        if not self.pred_obj_scores:
            loss_class = loss_multimask.new_zeros(())
        else:
            N = object_score_logits.shape[0]
            obj_logits_nm1 = object_score_logits.reshape(N, 1, 1)
            target_obj_nm1 = cls_gate.to(obj_logits_nm1.dtype).reshape(N, 1, 1)
            loss_class_row = sigmoid_focal_loss(
                obj_logits_nm1, target_obj_nm1, cls_denom,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
                loss_on_multimask=True,
            ).squeeze(-1)  # [N]
            loss_class = (loss_class_row * cls_gate.to(loss_class_row.dtype)).sum()

        # [N, M] -- sample_iou is rejected at construction time for this path
        # (use_point_sampling=False), so we always take the dense branch.
        loss_multiiou = self._iou_loss(
            src_masks, target_masks, pred_ious, mask_denom,
        )

        if loss_multimask.size(1) > 1:
            combo = (
                loss_multimask * float(self.weight_dict["loss_mask"])
                + loss_multidice * float(self.weight_dict["loss_dice"])
            )
            best = torch.argmin(combo, dim=-1)  # [N]
            b = torch.arange(combo.size(0), device=combo.device)

            loss_mask = loss_multimask[b, best].unsqueeze(1)  # [N,1]
            loss_dice = loss_multidice[b, best].unsqueeze(1)  # [N,1]
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[b, best].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou  = loss_multiiou

        # Gate by valid & presence (drop padded / absent rows), then reduce.
        gate = mask_gate.to(loss_mask.dtype).view(-1, 1)  # [N,1]
        loss_mask = (loss_mask * gate).sum()
        loss_dice = (loss_dice * gate).sum()
        loss_iou  = (loss_iou  * gate).sum()

        return torch.stack([loss_mask, loss_dice, loss_iou, loss_class])

    def _update_losses(
        self,
        losses,
        src_masks,
        target_masks,
        ious,
        mask_denom,
        mask_gate,
        cls_denom,
        cls_gate,
        object_score_logits,
    ):
        # target_masks is [N,1,Z,Y,X]
        target_masks_base = target_masks

        if self.use_point_sampling:
            # coords/labels are no_grad and depend on src_masks.detach()
            point_coords_flat, point_labels = self._sample_points_and_labels(
                src_masks, target_masks_base
            )
            N, M, Z, Y, X = src_masks.shape
            src_flat = src_masks.reshape(N * M, 1, Z, Y, X)
            point_logits_flat = point_sample(
                src_flat, point_coords_flat, align_corners=False
            ).squeeze(1)  # [N*M,P]
            point_logits = point_logits_flat.view(N, M, -1)  # [N,M,P]
            loss_multimask = sigmoid_focal_loss(
                point_logits, point_labels, mask_denom,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
                loss_on_multimask=True,
            )
            loss_multidice = dice_loss(
                point_logits, point_labels, mask_denom,
                loss_on_multimask=True,
            )
        else:
            target_masks = target_masks_base.expand_as(src_masks)  # [N,M,Z,Y,X]
            loss_multimask = sigmoid_focal_loss(
                src_masks, target_masks, mask_denom,
                alpha=self.focal_alpha, gamma=self.focal_gamma,
                loss_on_multimask=True,
            )
            loss_multidice = dice_loss(
                src_masks, target_masks, mask_denom,
                loss_on_multimask=True,
            )

        if self.pred_obj_scores:
            N = object_score_logits.shape[0]
            obj_logits_nm1 = object_score_logits.reshape(N, 1, 1)
            target_obj_nm1 = cls_gate.to(obj_logits_nm1.dtype).reshape(N, 1, 1)
            loss_class_row = sigmoid_focal_loss(
                obj_logits_nm1, target_obj_nm1, cls_denom,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
                loss_on_multimask=True,
            ).squeeze(-1)  # [N]
            losses["loss_class"] += (loss_class_row * cls_gate.to(loss_class_row.dtype)).sum()

        if self.sample_iou:
            # IoU target sampled at the same uncertain points as focal/dice.
            # Requires use_point_sampling=True (asserted in __init__).
            loss_multiiou = self._iou_loss(
                src_masks, None, ious, mask_denom,
                point_logits=point_logits, point_labels=point_labels,
            )
        else:
            target_masks_iou = target_masks_base.expand_as(src_masks)
            loss_multiiou = self._iou_loss(
                src_masks, target_masks_iou, ious, mask_denom,
            )
        assert loss_multimask.dim() == 2, f"Expected loss_multimask shape (N, M), got {loss_multimask.shape}"
        assert loss_multidice.dim() == 2, f"Expected loss_multidice shape (N, M), got {loss_multidice.shape}"
        assert loss_multiiou.dim() == 2, f"Expected loss_multiiou shape (N, M), got {loss_multiiou.shape}"

        if loss_multimask.size(1) > 1:
            # take the mask indices with the smallest focal + dice loss for back propagation
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            # loss_mask: (N, 1), loss_dice: (N, 1)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            # calculate the iou prediction and slot losses only in the index
            # with the minimum loss for each mask (to be consistent w/ SAM)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        # Gate by valid & presence (drop padded / absent rows). mask_denom is
        # the global (valid & presence) count / world_size.
        gate = mask_gate.to(loss_mask.dtype).view(-1, 1)  # [N,1]
        loss_mask = loss_mask * gate
        loss_dice = loss_dice * gate
        loss_iou = loss_iou * gate

        # sum over batch dimension (losses already divided by mask_denom)
        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss