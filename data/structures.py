import math
from typing import Tuple

import torch
from torch import Tensor

from cell_observatory_platform.models.ops.roi_align_nd import RoIAlign3DFunction


def mask_ids_to_masks(batch_size, spatial_shape, mask_ids_batch, masks, device):
    """
    Convert per-sample mask IDs to per-sample binary masks.

    Args:
        batch_size (int): Number of samples in the batch.
        spatial_shape (tuple): Shape of the spatial dimensions.
        mask_ids_batch (list[list[int]]): For each sample in the batch, a list of instance IDs.
        masks (torch.Tensor): Tensor containing instance-ID maps.
                              Shape: [B, *spatial] or [*spatial] (then B assumed 1).
        input_format (str): Input format string (e.g. "TZYXC"). Used for sanity checks.
        input_shape (tuple): Shape of the input (no batch), matching input_format.
        device (torch.device): Device for output tensors.

    Returns:
        list[torch.Tensor]: For each sample b, a tensor of shape
                            [NUM_INST_b, *spatial], dtype=bool.
    """
    masks = masks.to(device)

    B = batch_size
    if len(mask_ids_batch) != B:
        raise ValueError(
            f"mask_ids_batch length ({len(mask_ids_batch)}) "
            f"does not match batch size ({B})."
        )

    binary_masks_batch = []
    for b in range(B):
        instance_ids = list(mask_ids_batch[b])
        m = masks[b]

        if len(instance_ids) == 0:
            # No instances: return empty [0, *spatial]
            empty = torch.zeros(
                (0,) + spatial_shape,
                dtype=torch.bool,
                device=device,
            )
            binary_masks_batch.append(empty)
            continue

        ids_tensor = torch.as_tensor(instance_ids, device=device, dtype=m.dtype)
        view_shape = (len(instance_ids),) + (1,) * m.dim()  # [N_inst, 1, 1, ...]
        binary_masks = (m.unsqueeze(0) == ids_tensor.view(view_shape))  # [N_inst, *spatial]
        binary_masks_batch.append(binary_masks.to(torch.bool))

    return binary_masks_batch


def delta2bbox(
    proposals, deltas, max_shape=None, whd_ratio_clip=16 / 1000, clip_border=True, add_ctr_clamp=False, ctr_clamp=32
):
    dxyz = deltas[..., :3]
    whd = deltas[..., 3:]

    pxyz = proposals[..., :3]
    pwhd = proposals[..., 3:]

    dxyz_whd = pwhd * dxyz

    max_ratio = abs(math.log(whd_ratio_clip))
    if add_ctr_clamp:
        dxyz_whd = torch.clamp(dxyz_whd, max=ctr_clamp, min=-ctr_clamp)
        whd = torch.clamp(whd, max=max_ratio)
    else:
        whd = whd.clamp(min=-max_ratio, max=max_ratio)

    gxyz = pxyz + dxyz_whd
    gwhd = pwhd * whd.exp()
    # [N, 3]
    x1y1z1 = gxyz - (gwhd * 0.5)
    x2y2z2 = gxyz + (gwhd * 0.5)
    # [N, 6]
    bboxes = torch.cat([x1y1z1, x2y2z2], dim=-1)

    if clip_border and max_shape is not None:
        bboxes[..., 0::3].clamp_(min=0).clamp_(max=max_shape[1])
        bboxes[..., 1::3].clamp_(min=0).clamp_(max=max_shape[0])
        bboxes[..., 2::3].clamp_(min=0).clamp_(max=max_shape[2])

    return bboxes


def bbox2delta(proposals, gt, means=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
    # hack for matcher
    if proposals.size() != gt.size():
        proposals = proposals[:, None]
        gt = gt[None]

    proposals = proposals.float()
    gt = gt.float()
    px, py, pz, pw, ph, pd = proposals.unbind(-1)
    gx, gy, gz, gw, gh, gd = gt.unbind(-1)

    dx = (gx - px) / (pw + 0.1)
    dy = (gy - py) / (ph + 0.1)
    dz = (gz - pz) / (pd + 0.1)
    dw = torch.log(gw / (pw + 0.1))
    dh = torch.log(gh / (ph + 0.1))
    dd = torch.log(gd / (pd + 0.1))
    deltas = torch.stack([dx, dy, dz, dw, dh, dd], dim=-1)

    # avoid unnecessary sync point if not needed
    if means != (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) or stds != (1.0, 1.0, 1.0, 1.0, 1.0, 1.0):
        means = deltas.new_tensor(means).unsqueeze(0)
        stds = deltas.new_tensor(stds).unsqueeze(0)
        deltas = deltas.sub_(means).div_(stds)

    return deltas


def convert_bbox_format(bboxes, bbox_input_format, bbox_output_format):
    """
    Convert bounding boxes from one format to another.
    Supported formats: 'cxcyczwhd', 'xyzxyz'
    """
    bbox_input_format = bbox_input_format.lower()
    bbox_output_format = bbox_output_format.lower()

    if bbox_input_format == bbox_output_format:
        return bboxes
    if bbox_input_format == "cxcyczwhd" and bbox_output_format == "xyzxyz":
        return box_cxcyczwhd_to_xyzxyz(bboxes)
    elif bbox_input_format == "xyzxyz" and bbox_output_format == "cxcyczwhd":
        return box_xyzxyz_to_cxcyczwhd(bboxes)
    elif bbox_input_format == "zyxzyx" and bbox_output_format == "cxcyczwhd":
        # zyxzyx -> xyzxyz -> cxcyczwhd
        bboxes = bboxes[:, [2, 1, 0, 5, 4, 3]]
        return box_xyzxyz_to_cxcyczwhd(bboxes)
    elif bbox_input_format == "cxcyczwhd" and bbox_output_format == "zyxzyx":
        # cxcyczwhd -> xyzxyz -> zyxzyx
        bboxes = box_cxcyczwhd_to_xyzxyz(bboxes)
        bboxes = bboxes[:, [2, 1, 0, 5, 4, 3]]
        return bboxes
    else:
        raise ValueError(f"Unsupported bbox format conversion from {bbox_input_format} to {bbox_output_format}")


def box_cxcyczwhd_to_xyzxyz(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (cx, cy, cz, w, h, d) format to (x1, y1, z1, x2, y2, z2) format.
    (cx, cy, cz) refers to center of bounding box
    (w, h, d) are width and height of bounding box
    Args:
        boxes (Tensor[N, 6]): boxes in (cx, cy, cz, w, h, d) format which will be converted.

    Returns:
        boxes (Tensor(N, 6)): boxes in (x1, y1, z1, x2, y2, z2) format.
    """
    # We need to change all 6 of them so some temporary variable is needed.
    cx, cy, cz, w, h, d = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    z1 = cz - 0.5 * d
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    z2 = cz + 0.5 * d

    boxes = torch.stack((x1, y1, z1, x2, y2, z2), dim=-1)

    return boxes


# Implementation adapted from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return generalized intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, z1, x2, y2, z2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2`` and ``0 <= z1 < z2``.

    Args:
        boxes1 (Tensor[N, 6]): first set of boxes
        boxes2 (Tensor[M, 6]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise generalized IoU values
        for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union

    lti = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rbi = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    whi = _upcast(rbi - lti).clamp(min=0)  # [N,M,3]
    vol_i = whi[:, :, 0] * whi[:, :, 1] * whi[:, :, 2]

    return iou - (vol_i - union) / vol_i


def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)

    roi_align = RoIAlign3DFunction.apply

    gt_masks_gpu = gt_masks.to("cuda")
    rois_gpu = rois.to("cuda")

    result = roi_align(gt_masks_gpu, rois_gpu, (M, M, M), 1.0)[:, 0]
    result = result.to(gt_masks.device)
    return result


def masks_to_boxes(masks: torch.Tensor) -> Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 6] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 <= x2`` and ``0 <= y1 <= y2`` and ``0 <= z1 < z2``.

    .. warning::

        In most cases the output will guarantee ``x1 < x2`` and ``y1 < y2`` and ``0 <= z1 < z2``. But
        if the input is degenerate, e.g. if a mask is a single row or a single
        column, then the output may have x1 = x2 or y1 = y2 or z1=z2.

    Args:
        masks (Tensor[N, D, H, W]): masks to transform where N is the number of masks
            and (D, H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 6]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 6), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        z, y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.min(z)
        bounding_boxes[index, 3] = torch.max(x)
        bounding_boxes[index, 4] = torch.max(y)
        bounding_boxes[index, 5] = torch.max(z)

    return bounding_boxes


# TODO: reconcile masks_to_boxes and masks_to_boxes_v2
def masks_to_boxes_v2(masks, eps: float = 1e-1) -> Tensor:
    """
    Compute the bounding boxes around the provided masks.
    The masks should be in format [N, D, H, W] where N is
    the number of masks, (D, H, W) are the spatial dimensions.
    Returns a [N, 6] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device)

    d, h, w = masks.shape[-3:]

    z = torch.arange(0, d, dtype=torch.float, device=masks.device)
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    z, y, x = torch.meshgrid(z, y, x, indexing="ij")

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    
    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]

    z_mask = masks * z.unsqueeze(0)
    z_max = z_mask.flatten(1).max(-1)[0]

    x_min = x_mask.masked_fill(~(masks.bool()), float("inf")).flatten(1).min(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), float("inf")).flatten(1).min(-1)[0]
    z_min = z_mask.masked_fill(~(masks.bool()), float("inf")).flatten(1).min(-1)[0]

    mask = torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], 1).to(masks.device, torch.float)
    invalid_mask = (torch.isinf(x_min)) | (torch.isinf(y_min)) | (torch.isinf(z_min))
    mask[invalid_mask] = 0
    return mask


def box_xyzxyz_to_cxcyczwhd(boxes: Tensor) -> Tensor:
    """
    Converts bounding boxes from (x1, y1, z1, x2, y2, z2) format to (cx, cy, cz, w, h, d) format.
    (x1, y1, z1) refer to top left of bounding box
    (x2, y2, z2) refer to bottom right of bounding box
    Args:
        boxes (Tensor[N, 6]): boxes in (x1, y1, z1, x2, y2, z2) format which will be converted.

    Returns:
        boxes (Tensor(N, 6)): boxes in (cx, cy, cz w, h, d) format.
    """
    x1, y1, z1, x2, y2, z2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    cz = (z1 + z2) / 2
    w = x2 - x1
    h = y2 - y1
    d = z2 - z1

    boxes = torch.stack((cx, cy, cz, w, h, d), dim=-1)

    return boxes


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    vol1 = box_volume(boxes1)
    vol2 = box_volume(boxes2)

    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    dims = _upcast(rb - lt).clamp(min=0)  # [N,M,3]
    inter = dims[:, :, 0] * dims[:, :, 1] * dims[:, :, 2]  # [N,M]

    union = vol1[:, None] + vol2 - inter

    return inter, union


def box_volume(boxes: Tensor) -> Tensor:
    boxes = _upcast(boxes)
    return (boxes[:, 3] - boxes[:, 0]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 2])


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications
    # by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def bitmask_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    assert masks.dim() == 4, f"Expected (N, D, H, W), got {masks.shape}"

    N, D, H, W = masks.shape
    device = masks.device

    if N == 0:
        return masks.new_zeros((0, 6), dtype=torch.float32)

    # Treat non-zero as foreground
    if masks.dtype is torch.bool:
        masks_bool = masks
    else:
        masks_bool = masks != 0

    boxes = torch.zeros((N, 6), dtype=torch.float32, device=device)

    # occupancy along each principal axis
    # shapes: (N, W), (N, H), (N, D)
    x_any = masks_bool.any(dim=(1, 2))  # collapse D,H -> occupancy along X
    y_any = masks_bool.any(dim=(1, 3))  # collapse D,W -> occupancy along Y
    z_any = masks_bool.any(dim=(2, 3))  # collapse H,W -> occupancy along Z

    for idx in range(N):
        xs = torch.where(x_any[idx])[0]
        ys = torch.where(y_any[idx])[0]
        zs = torch.where(z_any[idx])[0]

        if xs.numel() and ys.numel() and zs.numel():
            # +1 on the max corner to keep the [min, max+1) convention
            boxes[idx] = torch.tensor(
                [xs[0], ys[0], zs[0], xs[-1] + 1, ys[-1] + 1, zs[-1] + 1],
                dtype=torch.float32,
                device=device,
            )
        # else: leave that row as zeros for empty mask

    return boxes
