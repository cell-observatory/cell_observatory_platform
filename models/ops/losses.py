import torch
from torch import nn
import torch.nn.functional as F


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # p_t is large for hard examples where the model mispredicts
    # i.e. prob is close to 0 for targets=1 or close to 1 for targets=0
    # the degree of loss modulation is controlled by gamma 
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        # upweight/downweight positive vs negative examples
        # if alpha is close to 1, the loss will be more focused 
        # on positive examples and vice versa
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    # dice loss is 1 - Dice_Coeff 
    # dice_coeff = 2 x (object overlap) / (sum of pixels in both masks)
    # hence loss is smaller for larger overlap / IOU
    inputs = inputs.sigmoid()
    # masks: (N, D, H, W) -> (N, D*H*W)
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    # (N,D*H*W)x(N,D*H*W)->(N,D*H*W)->(N,) 
    numerator = 2 * (inputs * targets).sum(-1)
    # (N,) + (N,) -> (N,)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).

    Returns:
        Loss tensor
    """
    # binary_cross_entropy_with_logits returns: (num_masks, num_pixels)
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # (num_masks, num_pixels) -> (num_masks,) -> loss / num_masks
    return loss.mean(1).sum() / num_masks


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    
    Returns:
        Loss tensor
    """
    dhw = inputs.shape[1]

    if dhw == 0:
        # return a zero cost matrix (no mask contribution)
        return torch.zeros(
            inputs.shape[0],
            targets.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
        )

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / dhw


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
    foreground class.
    
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))
