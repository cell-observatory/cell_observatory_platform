"""
Adapted from:
https://github.com/pytorch/vision/blob/309bd7a1512ad9ff0e9729fbdad043cb3472e4cb/torchvision/models/detection/_utils.py#L317
https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/matcher.py
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.amp import autocast

from cell_observatory_platform.data.structures import bbox2delta, box_cxcyczwhd_to_xyzxyz, generalized_box_iou
from cell_observatory_platform.models.layers.utils import point_sample, point_sample_labelmap
from cell_observatory_platform.models.ops.losses import batch_dice_loss, batch_sigmoid_ce_loss


def _assign_batched(cost_mats):
    """scipy assignment over per-image cost matrices with ONE device->host
    transfer (per-image .cpu() in the loop serialized B syncs per step) and a
    nan_to_num guard: scipy hard-crashes on a NaN/inf cost from a diverging
    step -- clamp so it surfaces as a (bad) match + loss spike instead.
    """
    if not cost_mats:
        return []
    flat = torch.cat([c.reshape(-1) for c in cost_mats])
    flat = torch.nan_to_num(flat, nan=1e6, posinf=1e6, neginf=-1e6).cpu()
    out, offset = [], 0
    for c in cost_mats:
        n = c.numel()
        out.append(linear_sum_assignment(flat[offset:offset + n].reshape(c.shape).numpy()))
        offset += n
    return out


class HungarianMatcher(nn.Module):
    """
    Computes an assignment between targets and model predictions.

    For efficiency, the targets do not include the no object class. Because of this, there may be
    more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are left unmatched (and thus treated as non-objects/background).
    """

    def __init__(
        self,
        cost_classification: float = 1,
        cost_mask: float = 1,
        cost_mask_dice: float = 1,
        num_points: int = 0,
        cost_box: float = 0,
        cost_box_giou: float = 0,
    ):
        """
        Args:
            cost_classification: Relative weight of the classification error in the matching cost
            cost_mask: Relative weight of the focal loss in the matching cost
            cost_mask_dice: Relative weight of the dice loss in matching cost
        """
        super().__init__()
        self.cost_classification = cost_classification

        self.cost_mask = cost_mask
        self.cost_mask_dice = cost_mask_dice

        self.cost_box = cost_box
        self.cost_box_giou = cost_box_giou

        assert cost_classification != 0 or cost_mask != 0 or cost_mask_dice != 0, "All costs can not be 0."

        self.num_points = num_points

    @torch.no_grad()
    def forward(self, outputs, targets, costs=("cls", "box", "mask"), alpha=0.25, gamma=2.0):
        """
        Args:
            outputs: Dict that contains at least the following entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                "pred_masks": Tensor of dim [batch_size, num_queries, D_pred, H_pred, W_pred] with the predicted masks

            targets: List of targets (where len(targets) = batch_size). Each target is a Dict that contains:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, D_gt, H_gt, W_gt] containing target masks

        Returns:
            List of size batch_size, containing tuples (index_i, index_j) where:
                - index_i is the indices of the selected predictions
                - index_j is the indices of the corresponding selected targets
            For each batch element, we must have that:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]

        cost_mats = []
        for batch_idx in range(batch_size):
            predicted_bboxes = outputs["pred_boxes"][batch_idx]
            if "box" in costs:
                target_bboxes = targets[batch_idx]["boxes"]
                # calculates the p-norm (p=1) distance between each pair of the
                # two collections of tensors
                with autocast(enabled=False, device_type="cuda"):
                    cost_bbox = torch.cdist(predicted_bboxes.float(), target_bboxes.float(), p=1)
                # we omit constant terms in the cost function, so we can just use the negative of the generalized box iou
                cost_box_giou = -generalized_box_iou(
                    box_cxcyczwhd_to_xyzxyz(predicted_bboxes), box_cxcyczwhd_to_xyzxyz(target_bboxes)
                )
            else:
                cost_bbox = torch.tensor(0).to(predicted_bboxes)
                cost_box_giou = torch.tensor(0).to(predicted_bboxes)

            # predicted_logits: [num_queries, num_classes]
            predicted_logits = outputs["pred_logits"][batch_idx].sigmoid()
            targets_labels = targets[batch_idx]["labels"]

            # focal loss
            negative_cost_classification = (
                (1 - alpha) * (predicted_logits**gamma) * (-(1 - predicted_logits + 1e-8).log())
            )
            positive_cost_classification = (
                alpha * ((1 - predicted_logits) ** gamma) * (-(predicted_logits + 1e-8).log())
            )
            cost_classification = (
                positive_cost_classification[:, targets_labels] - negative_cost_classification[:, targets_labels]
            )

            # compute classification cost, contrary to the loss computation, we don't use the NLL
            # but approximate it as: 1 - prob[target class]
            # since constants don't change optimization, we set cost_class = -out_prob[:, target_labels]
            if "mask" in costs and self.num_points > 0:
                predicted_masks = outputs["pred_masks"][batch_idx].unsqueeze(1)
                
                # NOTE: Efficient labelmap-based sampling (avoids materializing binary masks)
                instance_ids = targets[batch_idx]["mask_ids"]
                label_map = targets[batch_idx]["label_map"]
                
                point_coords = torch.rand(1, self.num_points, 3, device=predicted_masks.device)
                
                # predicted_sampled: [N, num_points]
                predicted_sampled = point_sample(
                    predicted_masks,
                    point_coords.repeat(predicted_masks.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)
                
                # target_sampled: [N, num_points]
                target_sampled = point_sample_labelmap(
                    label_map,
                    point_coords,
                    instance_ids,
                )
                
                with autocast(enabled=False, device_type="cuda"):
                    predicted_sampled = predicted_sampled.float()
                    target_sampled = target_sampled.float()
                    cost_mask = batch_sigmoid_ce_loss(predicted_sampled, target_sampled)
                    cost_mask_dice = batch_dice_loss(predicted_sampled, target_sampled)

            else:
                cost_mask = torch.tensor(0).to(predicted_bboxes)
                cost_mask_dice = torch.tensor(0).to(predicted_bboxes)

            C = (
                self.cost_mask * cost_mask
                + self.cost_classification * cost_classification
                + self.cost_mask_dice * cost_mask_dice
                + self.cost_box * cost_bbox
                + self.cost_box_giou * cost_box_giou
            )
            # C: (num_queries, num_target_boxes) -- keep on device; transferred
            # in ONE batched copy below (a per-image .cpu() here serialized B
            # device->host syncs per step).
            cost_mats.append(C.reshape(num_queries, -1))

        matched_masks = _assign_batched(cost_mats)
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in matched_masks
        ]


class Mask2FormerHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, 
        cost_classification: float = 1,
        cost_mask: float = 1,
        cost_mask_dice: float = 1,
        num_points: int = 0
    ):
        """Creates the matcher

        Params:
            cost_classification: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_mask_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_classification = cost_classification
        self.cost_mask = cost_mask
        self.cost_mask_dice = cost_mask_dice

        assert cost_classification != 0 or cost_mask != 0 or cost_mask_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        cost_mats = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_classification = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b].unsqueeze(1)  # [num_queries, 1, D_pred, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = (targets[b]["masks"].unsqueeze(1)).to(out_mask)

            # out_mask = out_mask[:, None]
            # tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 3, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                input=tgt_mask,
                point_coords=point_coords.repeat(tgt_mask.shape[0], 1, 1),
                mode="nearest", # default is bilinear; need nearest for GT binary labels
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False, device_type="cuda"):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_mask_dice = batch_dice_loss(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_classification * cost_classification
                + self.cost_mask_dice * cost_mask_dice
            )
            # keep on device; ONE batched transfer below (see _assign_batched)
            cost_mats.append(C.reshape(num_queries, -1))

        indices = _assign_batched(cost_mats)
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_classification: {}".format(self.cost_classification),
            "cost_mask: {}".format(self.cost_mask),
            "cost_mask_dice: {}".format(self.cost_mask_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

class PlainDETRHungarianMatcher(nn.Module):
    def __init__(self, cost_class: float, cost_bbox: float, cost_giou: float, reparam: bool):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.reparam = reparam

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # focal class cost (same as official)
        alpha, gamma = 0.25, 2.0
        neg = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos[:, tgt_ids] - neg[:, tgt_ids]

        # bbox cost
        if self.reparam:
            out_delta = outputs["pred_deltas"].flatten(0, 1)
            out_bbox_old = outputs["pred_boxes_old"].flatten(0, 1)
            # NOTE: supervise deltas between anchors and true boxes
            tgt_delta = bbox2delta(out_bbox_old, tgt_bbox)
            with autocast(enabled=False, device_type="cuda"):
                cost_bbox = torch.cdist(out_delta.float()[:, None], tgt_delta.float(), p=1).squeeze(1)
        else:
            with autocast(enabled=False, device_type="cuda"):
                cost_bbox = torch.cdist(out_bbox.float(), tgt_bbox.float(), p=1)

        cost_giou = -generalized_box_iou(
            box_cxcyczwhd_to_xyzxyz(out_bbox),
            box_cxcyczwhd_to_xyzxyz(tgt_bbox),
        )

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # de-NaN before scipy (hard-crash on NaN from a diverging step)
        C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=-1e6).view(bs, num_queries, -1).cpu()

        # [B,Q, SUM_i T_i] -> [B, Q, T_i] -> [Q,T_i] -> list of B [(index_i, index_j)] for each q in Q
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_plain_detr_matcher(args):
    return PlainDETRHungarianMatcher(
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
        reparam=(getattr(args, "cost_bbox_type", "l1") == "reparam"),
    )
