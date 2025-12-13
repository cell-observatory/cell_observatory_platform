import contextlib
import copy
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.nn.functional import interpolate

from cell_observatory_platform.data.masking.mask_generator import apply_masks
from cell_observatory_platform.data.structures import (
    bbox2delta,
    box_cxcyczwhd_to_xyzxyz,
    box_xyzxyz_to_cxcyczwhd,
    generalized_box_iou,
)
from cell_observatory_platform.models.layers.matchers import build_plain_detr_matcher
from cell_observatory_platform.models.layers.patch_embeddings import PatchEmbedding, calc_num_patches
from cell_observatory_platform.models.layers.utils import (
    batch_tensors,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from cell_observatory_platform.models.ops.losses import (
    batch_dice_loss,
    batch_sigmoid_ce_loss,
    calculate_uncertainty,
    dice_loss,
    sigmoid_ce_loss,
    sigmoid_focal_loss,
)
from cell_observatory_platform.training.helpers import get_patch_sizes
from cell_observatory_platform.utils.context import get_world_size, is_torch_dist_initialized


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
    loss = (targets - predictions) ** 2
    loss = loss.mean(dim=-1)  # mean loss per patch
    loss = loss.sum() / num_patches
    return loss, None


def L1_masked_loss(targets, predictions, num_patches, aux_loss_meta=None):
    # compute loss over masked patches
    loss = torch.abs(targets - predictions)
    loss = loss.mean(dim=-1)  # mean loss per patch
    loss = loss.sum() / num_patches
    return loss, None


# see: https://github.com/facebookresearch/ijepa/main/src/train.py
def smooth_L1_masked_loss(targets, predictions, num_patches, aux_loss_meta=None):
    return F.smooth_l1_loss(targets, predictions), None


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
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcyczwhd_to_xyzxyz(source_boxes), box_cxcyczwhd_to_xyzxyz(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

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
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del source_masks, target_masks
        return losses

    def preprocess_masks(self, mask_dict):
        predicted_denoise_bboxes, denoise_target_indices = (
            mask_dict["predicted_denoise_bboxes"],
            mask_dict["denoise_target_indices"],
        )
        max_query_pad_size, denoise_queries_per_label = (
            mask_dict["max_query_pad_size"],
            mask_dict["denoise_queries_per_label"],
        )
        query_pad_size_per_label = max_query_pad_size // denoise_queries_per_label
        num_targets = denoise_target_indices.numel()
        return predicted_denoise_bboxes, num_targets, query_pad_size_per_label, denoise_queries_per_label

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
            predicted_denoise_bboxes, num_targets, query_pad_size_per_label, denoise_queries_per_label = (
                self.preprocess_masks(denoise_predictions)
            )
            denoise_query_target_indices = []
            for target in targets:
                # we have L target labels and for each label we create denoise_queries_per_label queries
                # hence we have L * denoise_queries_per_label queries in total, here we get their indices
                # however, in decoder each batch element has a different number of target labels, so we need to pad the indices
                # to the max number of denoise queries per label, which is what we do
                target_labels = target["labels"]
                if len(target_labels) > 0:
                    # (num_target_labels, ) = [0,1,...,num_target_labels-1]
                    target_label_indices = torch.arange(0, len(target_labels)).long().cuda()
                    # (1, num_target_labels) -> (denoise_queries_per_label, num_target_labels)
                    # hence we get [[0, 1, ..., num_target_labels-1], ....]
                    target_label_indices = target_label_indices.unsqueeze(0).repeat(denoise_queries_per_label, 1)
                    # (denoise_queries_per_label, num_target_labels) -> (denoise_queries_per_label * num_target_labels, )
                    denoise_query_target_index = target_label_indices.flatten()
                    # build output indices into the predicted denoise queries:
                    # torch.arange(R) gives [0,1,...,R−1], shape: (denoise_queries_per_label,)
                    # multiply by pad_size P to get [r*P, r*P, ..., r*P], start offset for each row of queries
                    # unsqueeze -> (R,1), then broadcast-add t row r becomes [r*P + 0, r*P + 1, ..., r*P + (num_target_labels-1)]]
                    padded_denoise_query_target_index = (
                        (torch.tensor(range(denoise_queries_per_label)) * query_pad_size_per_label)
                        .long()
                        .cuda()
                        .unsqueeze(1)
                    )
                    # broadcast addition: (denoise_queries_per_label, 1) + (1, num_target_labels) -> (denoise_queries_per_label, num_target_labels)
                    # each row r becomes [r*P + 0, r*P + 1, ..., r*P + (num_target_labels-1)]
                    padded_denoise_query_target_index = padded_denoise_query_target_index + target_label_indices
                    padded_denoise_query_target_index = padded_denoise_query_target_index.flatten()
                else:
                    padded_denoise_query_target_index = denoise_query_target_index = torch.tensor([]).long().cuda()
                denoise_query_target_indices.append((padded_denoise_query_target_index, denoise_query_target_index))

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
            losses.update(self.compute_loss(loss, outputs, targets, matched_target_indices, average_num_masks_per_node))

        # compute denosing losses if denoise is enabled
        if self.denoise and denoise_predictions is not None:
            extra_losses = {}
            for loss in self.denoise_losses:
                extra_losses.update(
                    self.compute_loss(
                        loss,
                        predicted_denoise_bboxes,
                        targets,
                        denoise_query_target_indices,
                        average_num_masks_per_node * denoise_queries_per_label,
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
        if "auxiliary_outputs" in outputs:
            first_auxiliary_output_idx = 0 if "intermediate_outputs" in outputs else 1
            for i, auxiliary_output in enumerate(outputs["auxiliary_outputs"]):
                # hungarian matcher to get indices of the matched auxiliary_outputs and targets
                auxiliary_matched_target_indices = self.matcher(auxiliary_output, targets)
                for loss in self.losses:
                    extra_losses = self.compute_loss(
                        loss, auxiliary_output, targets, auxiliary_matched_target_indices, average_num_masks_per_node
                    )
                    extra_losses = {k + f"_{i}": v for k, v in extra_losses.items()}
                    losses.update(extra_losses)

                if i >= first_auxiliary_output_idx:
                    if self.denoise and denoise_predictions is not None:
                        auxiliary_predicted_denoise_bboxes = predicted_denoise_bboxes["auxiliary_outputs"][i]
                        extra_losses = {}
                        for loss in self.denoise_losses:
                            extra_losses.update(
                                self.compute_loss(
                                    loss,
                                    auxiliary_predicted_denoise_bboxes,
                                    targets,
                                    denoise_query_target_indices,
                                    average_num_masks_per_node * denoise_queries_per_label,
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

        if "intermediate_outputs" in outputs:
            intermediate_outputs = outputs["intermediate_outputs"]
            intermediate_matched_target_indices = self.matcher(intermediate_outputs, targets)
            for loss in self.losses:
                extra_losses = self.compute_loss(
                    loss, intermediate_outputs, targets, intermediate_matched_target_indices, average_num_masks_per_node
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
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcyczwhd_to_xyzxyz(src_boxes),
                box_cxcyczwhd_to_xyzxyz(target_boxes),
            )
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
            torch.distributed.all_reduce(num_boxes)
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
        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcyczwhd_to_xyzxyz(source_boxes), box_cxcyczwhd_to_xyzxyz(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

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
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del source_masks, target_masks
        return losses

    def preprocess_masks(self, mask_dict):
        predicted_denoise_bboxes, denoise_target_indices = (
            mask_dict["predicted_denoise_bboxes"],
            mask_dict["denoise_target_indices"],
        )
        max_query_pad_size, denoise_queries_per_label = (
            mask_dict["max_query_pad_size"],
            mask_dict["denoise_queries_per_label"],
        )
        query_pad_size_per_label = max_query_pad_size // denoise_queries_per_label
        num_targets = denoise_target_indices.numel()
        return predicted_denoise_bboxes, num_targets, query_pad_size_per_label, denoise_queries_per_label

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
            predicted_denoise_bboxes, num_targets, query_pad_size_per_label, denoise_queries_per_label = (
                self.preprocess_masks(denoise_predictions)
            )
            denoise_query_target_indices = []
            for target in targets:
                # we have L target labels and for each label we create denoise_queries_per_label queries
                # hence we have L * denoise_queries_per_label queries in total, here we get their indices
                # however, in decoder each batch element has a different number of target labels, so we need to pad the indices
                # to the max number of denoise queries per label, which is what we do
                target_labels = target["labels"]
                if len(target_labels) > 0:
                    # (num_target_labels, ) = [0,1,...,num_target_labels-1]
                    target_label_indices = torch.arange(0, len(target_labels)).long().cuda()
                    # (1, num_target_labels) -> (denoise_queries_per_label, num_target_labels)
                    # hence we get [[0, 1, ..., num_target_labels-1], ....]
                    target_label_indices = target_label_indices.unsqueeze(0).repeat(denoise_queries_per_label, 1)
                    # (denoise_queries_per_label, num_target_labels) -> (denoise_queries_per_label * num_target_labels, )
                    denoise_query_target_index = target_label_indices.flatten()
                    # build output indices into the predicted denoise queries:
                    # torch.arange(R) gives [0,1,...,R−1], shape: (denoise_queries_per_label,)
                    # multiply by pad_size P to get [r*P, r*P, ..., r*P], start offset for each row of queries
                    # unsqueeze -> (R,1), then broadcast-add t row r becomes [r*P + 0, r*P + 1, ..., r*P + (num_target_labels-1)]]
                    padded_denoise_query_target_index = (
                        (torch.tensor(range(denoise_queries_per_label)) * query_pad_size_per_label)
                        .long()
                        .cuda()
                        .unsqueeze(1)
                    )
                    # broadcast addition: (denoise_queries_per_label, 1) + (1, num_target_labels) -> (denoise_queries_per_label, num_target_labels)
                    # each row r becomes [r*P + 0, r*P + 1, ..., r*P + (num_target_labels-1)]
                    padded_denoise_query_target_index = padded_denoise_query_target_index + target_label_indices
                    padded_denoise_query_target_index = padded_denoise_query_target_index.flatten()
                else:
                    padded_denoise_query_target_index = denoise_query_target_index = torch.tensor([]).long().cuda()
                denoise_query_target_indices.append((padded_denoise_query_target_index, denoise_query_target_index))

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
            losses.update(self.compute_loss(loss, outputs, targets, matched_target_indices, average_num_masks_per_node))

        # compute denosing losses if denoise is enabled
        if self.denoise and denoise_predictions is not None:
            extra_losses = {}
            for loss in self.denoise_losses:
                extra_losses.update(
                    self.compute_loss(
                        loss,
                        predicted_denoise_bboxes,
                        targets,
                        denoise_query_target_indices,
                        average_num_masks_per_node * denoise_queries_per_label,
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
        if "auxiliary_outputs" in outputs:
            first_auxiliary_output_idx = 0 if "intermediate_outputs" in outputs else 1
            for i, auxiliary_output in enumerate(outputs["auxiliary_outputs"]):
                # hungarian matcher to get indices of the matched auxiliary_outputs and targets
                auxiliary_matched_target_indices = self.matcher(auxiliary_output, targets)
                for loss in self.losses:
                    extra_losses = self.compute_loss(
                        loss, auxiliary_output, targets, auxiliary_matched_target_indices, average_num_masks_per_node
                    )
                    extra_losses = {k + f"_{i}": v for k, v in extra_losses.items()}
                    losses.update(extra_losses)

                if i >= first_auxiliary_output_idx:
                    if self.denoise and denoise_predictions is not None:
                        auxiliary_predicted_denoise_bboxes = predicted_denoise_bboxes["auxiliary_outputs"][i]
                        extra_losses = {}
                        for loss in self.denoise_losses:
                            extra_losses.update(
                                self.compute_loss(
                                    loss,
                                    auxiliary_predicted_denoise_bboxes,
                                    targets,
                                    denoise_query_target_indices,
                                    average_num_masks_per_node * denoise_queries_per_label,
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

        if "intermediate_outputs" in outputs:
            intermediate_outputs = outputs["intermediate_outputs"]
            intermediate_matched_target_indices = self.matcher(intermediate_outputs, targets)
            for loss in self.losses:
                extra_losses = self.compute_loss(
                    loss, intermediate_outputs, targets, intermediate_matched_target_indices, average_num_masks_per_node
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
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcyczwhd_to_xyzxyz(src_boxes),
                box_cxcyczwhd_to_xyzxyz(target_boxes),
            )
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
            torch.distributed.all_reduce(num_boxes)
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
