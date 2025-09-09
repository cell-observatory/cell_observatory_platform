import torch
import torch.nn.functional as F


def get_loss_fn(loss_type: str ):
    if loss_type == "l2_masked":
        return L2_masked_loss
    
    elif loss_type == "l1_masked":
        return L1_masked_loss

    elif loss_type == "smooth_l1_masked":
        return smooth_L1_masked_loss

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    

def L2_masked_loss(targets, predictions, masks):
    loss = (targets - predictions) ** 2
    loss = loss.mean(dim=-1)  # mean loss per patch
    loss = loss.sum() / masks.sum()
    return loss


def L1_masked_loss(targets, predictions, masks):
    # compute loss over masked patches
    loss = torch.abs(targets - predictions)
    loss = loss.mean(dim=-1)  # mean loss per patch
    loss = loss.sum() / masks.sum()
    return loss


# see: https://github.com/facebookresearch/ijepa/main/src/train.py
def smooth_L1_masked_loss(targets, predictions, masks):
    return F.smooth_l1_loss(targets, predictions)