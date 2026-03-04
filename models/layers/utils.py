"""
Adapted from:
https://github.com/facebookresearch/detectron2/blob/536dc9d527074e3b15df5f6677ffe1f4e104a4ab/projects/PointRend/point_rend/point_features.py#L63
https://github.com/facebookresearch/detectron2/blob/536dc9d527074e3b15df5f6677ffe1f4e104a4ab/detectron2/layers/wrappers.py#L65
https://github.com/IDEA-Research/MaskDINO/blob/3831d8514a3728535ace8d4ecc7d28044c42dd14/maskdino/utils/misc.py#L49
https://github.com/facebookresearch/dinov3/dinov3/utils/utils.py
"""

import math
from typing import List, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from cell_observatory_platform.data.structures import masks_to_boxes_v2


def reconstruct_full_feature_map(
    x: Tensor,
    mask_indices: Tensor,
    full_length: int,
    mask_token: Tensor,
) -> Tensor:
    """
    Scatter context tokens back into a full-length sequence, filling
    masked positions with a learned mask_token.    
    """
    B, N_ctx, C = x.shape
    full = mask_token.expand(B, full_length, C).clone()
    idx = mask_indices if mask_indices.dim() == 2 else mask_indices[None].expand(B, -1)
    full.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, C), x)
    return full


class Unroll(nn.Module):
    """
    Reorders the tokens such that patches are contiguous in memory.
    E.g., given [B, (H, W), C] and stride of (Sy, Sx), this will re-order the tokens as
                           [B, (Sy, Sx, H // Sy, W // Sx), C]

    This allows operations like Max2d to be computed as x.view(B, Sx*Sy, -1, C).max(dim=1).
    Not only is this faster, but it also makes it easy to support inputs of arbitrary
    dimensions in addition to patch-wise sparsity.

    Performing this operation multiple times in sequence puts entire windows as contiguous
    in memory. For instance, if you applied the stride (2, 2) 3 times, entire windows of
    size 8x8 would be contiguous in memory, allowing operations like mask unit attention
    computed easily and efficiently, while also allowing max to be applied sequentially.

    Note: This means that intermediate values of the model are not in HxW order, so they
    need to be re-rolled if you want to use the intermediate values as a HxW feature map.
    The last block of the network is fine though, since by then the strides are all consumed.
    """ 

    def __init__(
        self,
        input_size: Tuple[int, ...],
        patch_stride: Tuple[int, ...],
        unroll_schedule: List[Tuple[int, ...]],
    ):
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]
        self.schedule = unroll_schedule

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Flattened patch embeddings [B, N, C]
        Output: Patch embeddings [B, N, C] permuted such that [B, 4, N//4, C].max(1) etc. performs MaxPoolNd
        """
        B, _, C = x.shape

        cur_size = self.size
        x = x.view(*([B] + cur_size + [C]))

        for strides in self.schedule:
            # Move patches with the given strides to the batch dimension

            # Create a view of the tensor with the patch stride as separate dims
            # For example in 2d: [B, H // Sy, Sy, W // Sx, Sx, C]
            cur_size = [i // s for i, s in zip(cur_size, strides)]
            new_shape = [B] + sum([[i, s] for i, s in zip(cur_size, strides)], []) + [C]
            x = x.view(new_shape)

            # Move the patch stride into the batch dimension
            # For example in 2d: [B, Sy, Sx, H // Sy, W // Sx, C]
            L = len(new_shape)
            permute = (
                [0] + list(range(2, L - 1, 2)) + list(range(1, L - 1, 2)) + [L - 1]
            )
            x = x.permute(permute)

            # Now finally flatten the relevant dims into the batch dimension
            x = x.flatten(0, len(strides))
            B *= math.prod(strides)

        x = x.reshape(-1, math.prod(self.size), C)
        return x


def undo_windowing(x: torch.Tensor, size: List[int], cur_mu_shape: List[int]) -> torch.Tensor:
    """
    Convert windowed tokens [B, N, *cur_mu_shape, C] back to spatial order [B, *full_spatial, C].
    full_spatial[i] = size[i] * cur_mu_shape[i].
    """
    B, N, *_, C = x.shape
    D = len(size)
    x = x.view(B, *size, *cur_mu_shape, C)
    perm = [0]
    for i in range(D):
        perm.append(1 + i)      # size dim
        perm.append(1 + D + i)  # cur_mu_shape dim
    perm.append(len(x.shape) - 1)
    x = x.permute(perm)
    new_dims = [size[i] * cur_mu_shape[i] for i in range(D)]
    x = x.reshape(B, *new_dims, C)
    return x


def conv_nd(n: int):
    """Factory returning Conv2d for n=2, Conv3d for n=3, etc."""
    if n == 2:
        return nn.Conv2d
    elif n == 3:
        return nn.Conv3d
    else:
        raise NotImplementedError(f"conv_nd not implemented for n={n}")


def do_pool_stride(x: torch.Tensor, stride: int) -> torch.Tensor:
    """Max-pool over groups of `stride` tokens in flattened [B, N, C] tensor."""
    if stride is None or stride <= 1:
        return x
    B, N, C = x.shape
    assert N % stride == 0, f"N={N} must be divisible by stride={stride}"
    x = x.view(B, N // stride, stride, C)
    return x.max(dim=2).values


class Reroll(nn.Module):
    """
    Undos the "unroll" operation so that you can use intermediate features.
    """

    def __init__(
        self,
        input_size: Tuple[int, ...],
        patch_stride: Tuple[int, ...],
        unroll_schedule: List[Tuple[int, ...]],
        stage_ends: List[int],
        q_pool: int,
    ):
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]

        # The first stage has to reverse everything
        # The next stage has to reverse all but the first unroll, etc.
        self.schedule = {}
        size = self.size
        for i in range(stage_ends[-1] + 1):
            self.schedule[i] = unroll_schedule, size
            # schedule unchanged if no pooling at a stage end
            if i in stage_ends[:q_pool]:
                if len(unroll_schedule) > 0:
                    size = [n // s for n, s in zip(size, unroll_schedule[0])]
                unroll_schedule = unroll_schedule[1:]

    def forward(
        self,
        x: torch.Tensor,
        block_idx: int,
        mask: torch.Tensor = None,
        return_windowed: bool = False,
    ) -> torch.Tensor:
        """
        Roll the given tensor back up to spatial order assuming it's from the given block.

        If return_windowed=True:
            - Returns [B, N, *cur_mu_shape, C] (MU-grid layout) without undo_windowing.
        If return_windowed=False:
            - If mask is not None: returns [B, #MUs, MUy, MUx, C].
            - If mask is None: returns [B, H, W, C] after undo_windowing.
        """
        schedule, size = self.schedule[block_idx]
        B, N, C = x.shape

        D = len(size)
        cur_mu_shape = [1] * D

        for strides in schedule:
            # Extract the current patch from N
            x = x.view(B, *strides, N // math.prod(strides), *cur_mu_shape, C)

            # Move that patch into the current MU
            # Example in 2d: [B, Sy, Sx, N//(Sy*Sx), MUy, MUx, C] -> [B, N//(Sy*Sx), Sy, MUy, Sx, MUx, C]
            L = len(x.shape)
            permute = (
                [0, 1 + D]
                + sum(
                    [list(p) for p in zip(range(1, 1 + D), range(1 + D + 1, L - 1))],
                    [],
                )
                + [L - 1]
            )
            x = x.permute(permute)

            # Reshape to [B, N//(Sy*Sx), *MU, C]
            for i in range(D):
                cur_mu_shape[i] *= strides[i]
            x = x.reshape(B, -1, *cur_mu_shape, C)
            N = x.shape[1]

        # Current shape (e.g., 2d: [B, #MUy*#MUx, MUy, MUx, C])
        x = x.view(B, N, *cur_mu_shape, C)

        # Output format controlled explicitly
        if return_windowed:
            return x

        # If masked (legacy behavior), return windowed; else undo_windowing to spatial
        if mask is not None:
            return x
        x = undo_windowing(x, size, cur_mu_shape)
        return x


def cat_keep_shapes(x_list: List[Tensor]) -> Tuple[Tensor, List[Tuple[int]], List[int]]:
    # [B_i, N_i, C_i] -> [(B_i, N_i, C_i), (B_2, N_2, C_2), ...]
    shapes = [x.shape for x in x_list]
    # [B_i, N_i, C_i] -> [B_i*N_i]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    # (B_i, N_i, C_i) ->  (B_i*N_i, C_i) -> [SUM_i (B_i*N_i), C_i]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(flattened: Tensor, shapes: List[Tuple[int]], num_tokens: List[int]) -> List[Tensor]:
    # flattened -> [(B_i*N_i, C_i), (B_2*N_2, C_2), ...]
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    # [(B_i, N_i, C_i), (B_2, N_2, C_2), ...] -> [(B_i, N_i, C_*), (B_2, N_2, C_*), ...]
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    # [(B_i*N_i, C_*), (B_2*N_2, C_*), ...] -> [(B_i, N_i, C_*), (B_2, N_2, C_*), ...]
    outputs_reshaped = [o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)]
    return outputs_reshaped


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def pack_time(x: torch.Tensor, input_format: str, output_format: str = "TCZYX"):
    if input_format == "TZYXC":
        B, T, Z, Y, X, C = x.shape
        if output_format == "TCZYX":
            x = x.permute(0, 1, 5, 2, 3, 4).reshape(B * T, C, Z, Y, X)
            return x, B, T
        else:
            raise ValueError(f"Unsupported output_format {output_format}")
    elif input_format == "TCZYX":
        # x: [B, T, C, Z, Y, X] -> [B*T, C, Z, Y, X]
        B, T, C, Z, Y, X = x.shape
        if output_format == "TCZYX":
            x = x.reshape(B * T, C, Z, Y, X)
            return x, B, T
        else:
            raise ValueError(f"Unsupported output_format {output_format}")
    else:
        raise ValueError(f"Unsupported input_format {input_format}")


def unpack_time(x: torch.Tensor, B: int, T: int, input_format: str, output_format: str):
    if input_format == "TZYXC":
        _, Z, Y, X, C = x.shape
        if output_format == "TZYXC":
            return x.reshape(B, T, Z, Y, X, C)
        elif output_format == "CT":
            return x.reshape((B, T, Z, Y, X, C)).permute(0, 2, 3, 4, 5, 1).reshape(B * Z * Y * X, C, T)
        elif output_format == "TCZYX":
            return x.reshape((B, T, Z, Y, X, C)).permute(0, 1, 5, 2, 3, 4)
        elif output_format == "ZYXCT":
            return x.reshape((B, T, Z, Y, X, C)).permute(0, 2, 3, 4, 5, 1)
        else:
            raise ValueError(f"Unsupported output_format {output_format}")
    elif input_format == "TCZYX":
        _, C, Z, Y, X = x.shape
        if output_format == "TZYXC":
            return x.reshape(B, T, C, Z, Y, X).permute(0, 1, 3, 4, 5, 2)
        elif output_format == "CT":
            return x.reshape((B, T, C, Z, Y, X)).permute(0, 3, 4, 5, 2, 1).reshape(B * Z * Y * X, C, T)
        elif output_format == "TCZYX":
            return x.reshape((B, T, C, Z, Y, X))
        elif output_format == "ZYXCT":
            return x.reshape((B, T, C, Z, Y, X)).permute(0, 3, 4, 5, 2, 1)
        else:
            raise ValueError(f"Unsupported output_format {output_format}")
    else:
        raise ValueError(f"Unsupported input_format {input_format}")


def pack_spatial(x: torch.Tensor, input_format: str, output_format: str = "CT"):
    if input_format == "TZYXC":
        B, T, Z, Y, X, C = x.shape
        if output_format == "CT":
            return x.permute(0, 2, 3, 4, 5, 1).reshape(B * Z * Y * X, C, T)
        else:
            raise ValueError(f"Unsupported output_format {output_format}")
    elif input_format == "TCZYX":
        B, T, C, Z, Y, X = x.shape
        if output_format == "CT":
            return x.permute(0, 3, 4, 5, 2, 1).reshape(B * Z * Y * X, C, T)
        else:
            raise ValueError(f"Unsupported output_format {output_format}")
    else:
        raise ValueError(f"Unsupported input_format {input_format}")


def unpack_spatial(x: torch.Tensor, B: int, input_format: str, input_shape: List[int], output_format: str):
    _, C, T = x.shape
    if input_format == "TZYXC":
        T, Z, Y, X, C = input_shape
    elif input_format == "TCZYX":
        T, C, Z, Y, X = input_shape
    else:
        raise ValueError(f"Unsupported input_format {input_format}")

    if output_format == "TZYXC":
        if input_format == "TZYXC" or input_format == "TCZYX":
            return x.reshape(B, Z, Y, X, C, T).permute(0, 5, 1, 2, 3, 4)
        # in case we want to support 2D+T in future
        else:
            raise ValueError(f"Unsupported input_format {input_format}")
    elif output_format == "TCZYX":
        if input_format == "TZYXC" or input_format == "TCZYX":
            return x.reshape(B, Z, Y, X, C, T).permute(0, 5, 4, 1, 2, 3)
    elif output_format == "CTZYX":
        if input_format == "TZYXC" or input_format == "TCZYX":
            return x.reshape(B, Z, Y, X, C, T).permute(0, 4, 5, 1, 2, 3)
    else:
        raise ValueError(f"Unsupported output_format {output_format}")


def get_reference_points(shapes, valid_ratios, device):
    reference_points_list = []
    for lvl, (D_, H_, W_) in enumerate(shapes):
        # create grid [0.5, 1.5, ..., size_dim - 0.5]
        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            indexing="ij",
        )

        # scaling by valid_ratios adjusts the normalized reference grid so that it
        # only spans the unpadded region, i.e. scale grid to [0, 1]
        ref_z = ref_z.reshape(-1)[None] / (valid_ratios[:, None, lvl, 2] * D_)
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)

        ref = torch.stack((ref_x, ref_y, ref_z), -1)  # [B, D*H*W, 3]
        reference_points_list.append(ref)

    # reference_points: [B, \sum_l D_l*H_l*W_l, 3]
    reference_points = torch.cat(reference_points_list, 1)
    # [B, 1, L, 3] * [B, \sum_l D_l*H_l*W_l, 3] -> [B, \sum_l D_l*H_l*W_l, L, 3]
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, D, H, W) that contains features map on a D x H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 3) or (N, Dgrid, Wgrid, Hgrid, 3) that contains
        [0, 1] x [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Dgrid, Wgrid, Hgrid) that contains
            features for points in `point_coords`. The features are obtained via trilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        # (N, P, 3) -> (N, P, 1, 1, 3)
        point_coords = point_coords.unsqueeze(2).unsqueeze(3)

    if point_coords.dtype != input.dtype:
        point_coords = point_coords.to(dtype=input.dtype)

    # sample points in [-1, 1] x [-1, 1] x [-1, 1] coordinate space
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        # (N, C, P, 1, 1) -> (N, C, P)
        output = output.squeeze(4).squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] x [0,1] coordinate space based on their uncertainty. The unceratinties
    are calculated for each point using 'uncertainty_func' function that takes point's logit
    prediction as input.

    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Dmask, Hmask, Wmask) or (N, 1, Dmask, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importance sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 3) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1, "oversample_ratio must be >= 1"
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0, "importance_sample_ratio must be in [0, 1]"

    num_boxes = coarse_logits.shape[0]
    # NOTE: we oversample points first
    num_sampled = int(num_points * oversample_ratio)

    point_coords = torch.rand(num_boxes, num_sampled, 3, device=coarse_logits.device, dtype=coarse_logits.dtype)
    # NOTE: align_corners passed to grid_sample
    # returns: (N, C, D, H, W)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)

    # It is crucial to calculate uncertainty based on the sampled prediction value for points.
    # Calculating uncertainties of coarse predictions first and sampling them for points leads
    # to incorrect results. To illustrate this:
    # assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value
    # however, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty
    # returns: (N, 1, num_sampled) per point score
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points

    # returns: topk indices of the most uncertain points
    # idx: (N, num_uncertain_points)
    uncertain_points_indices = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    # shift: [0, num_sampled, 2*num_sampled, ..., (num_boxes-1)*num_sampled]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    # uncertain_points_indices: (num_boxes, num_uncertain_points) -> broadcast-add (num_boxes, num_uncertain_points)
    # i.e. for every box, we add a constant offset given by num_sampled * box_index
    # this allows for global indexing into point_coords
    uncertain_points_indices += shift[:, None]

    # point_coords: (num_boxes, num_sampled, 3) -> (num_boxes * num_sampled, 3)
    # -> (num_boxes*num_uncertain_points, 3) -> (num_boxes, num_uncertain_points, 3)
    point_coords = point_coords.view(-1, 3)[uncertain_points_indices.view(-1), :].view(
        num_boxes, num_uncertain_points, 3
    )

    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 3, device=coarse_logits.device, dtype=coarse_logits.dtype),
            ],
            dim=1,
        )

    return point_coords


def point_sample_labelmap_batched(
    labelmap: torch.Tensor,      # [B, Z, Y, X] integer instance labelmap
    point_coords: torch.Tensor,  # [N, K, 3] normalized coords in [0, 1]
    batch_indices: torch.Tensor, # [N] which batch each query belongs to
    instance_ids: torch.Tensor,  # [N] actual labelmap instance IDs to match
    align_corners: bool = False,
) -> torch.Tensor:               # [N, K] binary labels (float)
    """
    Sample binary mask labels directly from an integer labelmap.
    Uses direct integer indexing instead of grid_sample to avoid
    materializing [N, Z, Y, X] intermediate tensors.
    """
    # NOTE: CUDA only supports int32 for indexing
    if labelmap.dtype in (torch.uint8, torch.uint16, torch.int8, torch.int16):
        labelmap = labelmap.to(torch.int32)
    N, K, _ = point_coords.shape
    B, Z, Y, X = labelmap.shape
    device = labelmap.device
    
    # Convert normalized coords [0,1] to integer indices
    # point_coords[:, :, 0] is z, [:, :, 1] is y, [:, :, 2] is x
    if align_corners:
        z_idx = (point_coords[:, :, 0] * (Z - 1)).round().long().clamp(0, Z - 1)
        y_idx = (point_coords[:, :, 1] * (Y - 1)).round().long().clamp(0, Y - 1)
        x_idx = (point_coords[:, :, 2] * (X - 1)).round().long().clamp(0, X - 1)
    else:
        z_idx = (point_coords[:, :, 0] * Z).floor().long().clamp(0, Z - 1)
        y_idx = (point_coords[:, :, 1] * Y).floor().long().clamp(0, Y - 1)
        x_idx = (point_coords[:, :, 2] * X).floor().long().clamp(0, X - 1)
    
    # Expand batch_indices to [N, K]
    b_idx = batch_indices.view(N, 1).expand(N, K)
    
    # Direct 4D indexing: labelmap[b, z, y, x] -> [N, K]
    sampled = labelmap[b_idx, z_idx, y_idx, x_idx]  # [N, K] integers
    
    # Compare with instance IDs
    instance_ids_expanded = instance_ids.view(N, 1).expand(N, K)
    binary_labels = (sampled == instance_ids_expanded).float()
    
    return binary_labels


def point_sample_labelmap(
    labelmap_single: torch.Tensor,  # [Z, Y, X] single batch labelmap
    point_coords: torch.Tensor,     # [1, K, 3] or [K, 3] shared coords
    instance_ids: torch.Tensor,     # [M] instance IDs for M targets
    align_corners: bool = False,
) -> torch.Tensor:                  # [M, K] binary target masks at points
    """
    For matcher: sample target masks for all instances at shared point coords.
    """
    # NOTE: CUDA only supports int32 for indexing
    if labelmap_single.dtype in (torch.uint8, torch.uint16, torch.int8, torch.int16):
        labelmap_single = labelmap_single.to(torch.int32)
    
    Z, Y, X = labelmap_single.shape
    M = instance_ids.shape[0]
    
    # Handle both [1, K, 3] and [K, 3] input
    if point_coords.dim() == 3:
        point_coords = point_coords.squeeze(0)  # [K, 3]
    K = point_coords.shape[0]
    
    # Convert normalized coords to integer indices
    if align_corners:
        z_idx = (point_coords[:, 0] * (Z - 1)).round().long().clamp(0, Z - 1)
        y_idx = (point_coords[:, 1] * (Y - 1)).round().long().clamp(0, Y - 1)
        x_idx = (point_coords[:, 2] * (X - 1)).round().long().clamp(0, X - 1)
    else:
        z_idx = (point_coords[:, 0] * Z).floor().long().clamp(0, Z - 1)
        y_idx = (point_coords[:, 1] * Y).floor().long().clamp(0, Y - 1)
        x_idx = (point_coords[:, 2] * X).floor().long().clamp(0, X - 1)
    
    # Sample labelmap at K points: [K]
    sampled = labelmap_single[z_idx, y_idx, x_idx]  # [K]
    
    # Broadcast compare: [M, 1] == [1, K] -> [M, K]
    instance_ids_col = instance_ids.view(M, 1)
    sampled_row = sampled.view(1, K)
    binary_targets = (sampled_row == instance_ids_col).float()
    
    return binary_targets


def _max_by_axis(img_list):
    maxes = img_list[0]
    for sublist in img_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def batch_tensors(tensors: List[torch.Tensor], pad_value: float = 0.0):
    """
    Batch a list of tensors with shape (N_i, *spatial_dims) into a tensor of
    shape (B, max_N, *spatial_dims) plus a validity mask.

    Args:
        tensors: list of length B, each tensor of shape (N_i, *spatial_dims)
                 e.g. (num_instances_i, D, H, W) for 3D masks.
        pad_value: value to use for padding.

    Returns:
        batched: Tensor of shape (B, max_N, *spatial_dims)
        valid:   Bool tensor of shape (B, max_N), True for real instances.
    """
    assert len(tensors) > 0, "batch_tensors: empty tensor list"

    # Ensure all tensors have the same spatial shape
    spatial_shape = tensors[0].shape[1:]
    for t in tensors:
        if t.shape[1:] != spatial_shape:
            raise ValueError(
                f"All tensors must have the same spatial shape. " f"Got {t.shape[1:]} and {spatial_shape}."
            )

    B = len(tensors)
    max_n = max(t.shape[0] for t in tensors)
    device = tensors[0].device
    dtype = tensors[0].dtype

    # (B, max_N, *spatial_dims)
    batched = tensors[0].new_full((B, max_n, *spatial_shape), pad_value)
    # (B, max_N)
    valid = torch.zeros((B, max_n), dtype=torch.bool, device=device)

    for i, t in enumerate(tensors):
        n = t.shape[0]
        if n == 0:
            continue
        batched[i, :n].copy_(t)
        valid[i, :n] = True

    return batched, valid


def compute_unmasked_ratio(mask):
    _, D, H, W = mask.shape

    valid_D = (~mask).any(dim=(2, 3)).sum(dim=1)  # [B] — any valid pixel in each D-slice
    valid_H = (~mask).any(dim=(1, 3)).sum(dim=1)  # [B] — any valid pixel in each H-slice
    valid_W = (~mask).any(dim=(1, 2)).sum(dim=1)  # [B] — any valid pixel in each W-slice

    valid_ratio_d = valid_D.float() / D
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W

    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h, valid_ratio_d], -1)  # [B, 3]
    return valid_ratio


def c2_xavier_fill(module: nn.Module) -> None:
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}


def sample_box_points(
    input_fmt: str,
    masks: torch.Tensor,
    time_separable: bool = True,
    noise: float = 0.1,  # SAM default
    noise_bound: int = 20,  # SAM default
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> Tuple[np.array, np.array]:
    """
    Sample a noised version of the corners of a given `bbox`

    Inputs:
    - input_fmt: input format, e.g. "ZYXC"
    - masks: [B, 1, D, H, W] masks, dtype=torch.Tensor
    - noise: noise as a fraction of box depth, width and height, dtype=float
    - noise_bound: maximum amount of noise (in pure pixels), dtype=int

    Returns:
    - box_coords: [B, num_pt, 3], contains (z, y, x) coordinates of box corners, dtype=torch.float
    - box_labels: [B, num_pt], label 2 is reserverd for top left and 3 for bottom right corners, dtype=torch.int32
    """
    if input_fmt == "ZYXC" or (input_fmt == "TZYXC" and time_separable):
        device = masks.device
        # box_coords: [B, 6] for masks: [N, D, H, W]
        box_coords = masks_to_boxes_v2(masks.squeeze(1))
        B, _, D, H, W = masks.shape
        # box_labels: [B, 2]
        box_labels = torch.tensor(
            [top_left_label, bottom_right_label], dtype=torch.int, device=device
        ).repeat(B)
        if noise > 0.0:
            if not isinstance(noise_bound, torch.Tensor):
                noise_bound = torch.tensor(noise_bound, device=device)
            # NOTE: masks_to_boxes_v2 returns x1, y1, z1, x2, y2, z2 format
            # bbox_w: [B, 1], bbox_h: [B, 1], bbox_d: [B, 1]
            bbox_w = box_coords[..., 3] - box_coords[..., 0]
            bbox_h = box_coords[..., 4] - box_coords[..., 1]
            bbox_d = box_coords[..., 5] - box_coords[..., 2]
            max_dx = torch.min(bbox_w * noise, noise_bound)
            max_dy = torch.min(bbox_h * noise, noise_bound)
            max_dz = torch.min(bbox_d * noise, noise_bound)
            # bbox_noise: [B, 6] in range [-1, 1]
            box_noise = 2 * torch.rand(B, 1, 6, device=device) - 1
            box_noise = box_noise * torch.stack((max_dx, max_dy, max_dz, max_dx, max_dy, max_dz), dim=-1)

            box_coords = box_coords + box_noise
            img_bounds = (
                torch.tensor([W, H, D, W, H, D], device=device) - 1
            )  # uncentered pixel coords
            box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)  # In place clamping

        box_coords = box_coords.reshape(-1, 2, 3)  # always 2 points (top left and bottom right)
        box_labels = box_labels.reshape(-1, 2)  # always 2 labels (top left and bottom right)
    else:
        raise NotImplementedError(f"Input format {input_fmt} not supported yet.")

    return box_coords, box_labels


def sample_random_points_from_errors(
        input_fmt: str,
        gt_masks: torch.Tensor,
        pred_masks: torch.Tensor,
        time_separable: bool = True,
        num_pt: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample `num_pt` random points (along with their labels) independently from the error regions.

    Inputs:
    - input_fmt: input format, e.g. "ZYXC"
    - gt_masks: [B, 1, D, H, W] masks, dtype=torch.bool
    - pred_masks: [B, 1, D, H, W] masks, dtype=torch.bool or None
    - num_pt: int, number of points to sample independently for each of the B error maps

    Outputs:
    - points: [B, num_pt, 3], dtype=torch.float, contains (x, y, z) coordinates of each sampled point
    - labels: [B, num_pt], dtype=torch.int32, where 1 means positive clicks and 0 means
      negative clicks
    """
    if input_fmt == "ZYXC" or (input_fmt == "TZYXC" and time_separable):
        if pred_masks is None:  # if pred_masks is not provided, treat it as empty
            pred_masks = torch.zeros_like(gt_masks)
        assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1, f"Expected (B, 1, D, H, W), got {gt_masks.shape}"
        assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape, f"Expected (B, 1, D, H, W), got {pred_masks.shape}"
        assert num_pt >= 0, f"num_pt must be >= 0, got {num_pt}"

        B, _, D_im, H_im, W_im = gt_masks.shape
        device = gt_masks.device

        # false positive region, a new point sampled in this region should have
        # negative label to correct the FP error
        fp_masks = ~gt_masks & pred_masks
        # false negative region, a new point sampled in this region should have
        # positive label to correct the FN error
        fn_masks = gt_masks & ~pred_masks
        # whether the prediction completely match the ground-truth on each mask
        # all_correct: [B, 1]
        all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
        # all_correct: [B, 1, 1, 1, 1]
        all_correct = all_correct[..., None, None, None]

        # channel 0 is FP map, while channel 1 is FN map
        pts_noise = torch.rand(B, num_pt, D_im, H_im, W_im, 2, device=device)
        # sample a negative new click from FP region or a positive new click
        # from FN region, depend on where the maximum falls,
        # and in case the predictions are all correct (no FP or FN), we just
        # sample a negative click from the background region
        # sample negative click in FP region OR if all correct and background
        pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
        # sample positive click in FN region
        pts_noise[..., 1] *= fn_masks
        # pts_idx: [B, num_pt]
        pts_idx = pts_noise.flatten(2).argmax(dim=2)
        # labels: [B, num_pt]
        labels = (pts_idx % 2).to(torch.int32)
        pts_idx = pts_idx // 2
        pts_x = pts_idx % W_im
        pts_y = (pts_idx // W_im) % H_im
        pts_z = pts_idx // (W_im * H_im)
        points = torch.stack([pts_x, pts_y, pts_z], dim=2).to(torch.float)
    else:
        raise NotImplementedError(f"Input format {input_fmt} not supported yet.")
    return points, labels


def sample_one_point_from_error_center(
        input_fmt: str,
        gt_masks: torch.Tensor,
        pred_masks: torch.Tensor,
        time_separable: bool = True,
        padding: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample 1 random point (along with its label) from the center of each error region,
    that is, the point with the largest distance to the boundary of each error region.
    This is the RITM sampling method from https://github.com/saic-vul/ritm_interactive_segmentation/blob/master/isegm/inference/clicker.py

    Inputs:
    - input_fmt: input format, e.g. "ZYXC"
    - gt_masks: [B, 1, D, H, W] masks, dtype=torch.bool
    - pred_masks: [B, 1, D, H, W] masks, dtype=torch.bool or None
    - padding: if True, pad with boundary of 1 px for distance transform

    Outputs:
    - points: [B, 1, 3], dtype=torch.float, contains (x, y, z) coordinates of each sampled point
    - labels: [B, 1], dtype=torch.int32, where 1 means positive clicks and 0 means negative clicks
    """
    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1, f"Expected (B, 1, D, H, W), got {gt_masks.shape}"
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape, f"Expected (B, 1, D, H, W), got {pred_masks.shape}"

    if input_fmt == "ZYXC" or (input_fmt == "TZYXC" and time_separable):
        B, _, D_im, H_im, W_im = gt_masks.shape
        device = gt_masks.device

        # false positive region, a new point sampled in this region should have
        # negative label to correct the FP error
        fp_masks = ~gt_masks & pred_masks
        # false negative region, a new point sampled in this region should have
        # positive label to correct the FN error
        fn_masks = gt_masks & ~pred_masks

        all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)  # [B,1]
        all_correct = all_correct[:, 0]  # [B]

        fp_np = fp_masks.cpu().numpy()
        fn_np = fn_masks.cpu().numpy()
        bg_np = (~gt_masks).cpu().numpy()  # background for fallback

        points = torch.zeros(B, 1, 3, dtype=torch.float)
        labels = torch.zeros(B, 1, dtype=torch.int32)  # default negative

        for b in range(B):
            if bool(all_correct[b]):
                # --- fallback: sample a negative click from background (~gt) ---
                bg = bg_np[b, 0]  # [D,H,W] bool
                bg_flat = bg.reshape(-1)
                idxs = np.flatnonzero(bg_flat)
                pt_idx = int(np.random.choice(idxs))
                x = pt_idx % W_im
                y = (pt_idx // W_im) % H_im
                z = pt_idx // (H_im * W_im)

                points[b, 0] = torch.tensor([x, y, z], dtype=torch.float)
                labels[b, 0] = 0
                continue

            # --- normal case: pick the deepest voxel in FN or FP by 3D EDT ---
            fn = fn_np[b, 0]  # [D,H,W] bool
            fp = fp_np[b, 0]  # [D,H,W] bool

            if padding:
                fn_pad = np.pad(fn, ((1, 1), (1, 1), (1, 1)), mode="constant")
                fp_pad = np.pad(fp, ((1, 1), (1, 1), (1, 1)), mode="constant")
            else:
                fn_pad, fp_pad = fn, fp

            fn_dt = distance_transform_edt(fn_pad.astype(np.uint8))
            fp_dt = distance_transform_edt(fp_pad.astype(np.uint8))

            if padding:
                fn_dt = fn_dt[1:-1, 1:-1, 1:-1]
                fp_dt = fp_dt[1:-1, 1:-1, 1:-1]

            fn_flat = fn_dt.reshape(-1)
            fp_flat = fp_dt.reshape(-1)

            fn_argmax = int(np.argmax(fn_flat))
            fp_argmax = int(np.argmax(fp_flat))

            fn_max = float(fn_flat[fn_argmax])
            fp_max = float(fp_flat[fp_argmax])

            # choose whether we correct FN (positive click) or FP (negative click)
            is_positive = fn_max > fp_max
            pt_idx = fn_argmax if is_positive else fp_argmax

            x = pt_idx % W_im
            y = (pt_idx // W_im) % H_im
            z = pt_idx // (H_im * W_im)

            points[b, 0] = torch.tensor([x, y, z], dtype=torch.float)
            labels[b, 0] = int(is_positive)

    else:
        raise NotImplementedError(f"Input format {input_fmt} not supported yet.")

    points = points.to(device)
    labels = labels.to(device)
    return points, labels


def get_next_point(
    input_fmt: str,
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    method: str,
    time_separable: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if method == "uniform":
        return sample_random_points_from_errors(input_fmt, gt_masks, pred_masks, time_separable)
    elif method == "center":
        return sample_one_point_from_error_center(input_fmt, gt_masks, pred_masks, time_separable)
    else:
        raise ValueError(f"unknown sampling method {method}")


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """
    Select up to `max_cond_frame_num` conditioning frames from `cond_frame_outputs`
    that are temporally closest to the current frame at `frame_idx`. Here, we take
    - a) the closest conditioning frame before `frame_idx` (if any);
    - b) the closest conditioning frame after `frame_idx` (if any);
    - c) any other temporally closest conditioning frames until reaching a total
         of `max_cond_frame_num` conditioning frames.

    Outputs:
    - selected_outputs: selected items (keys & values) from `cond_frame_outputs`.
    - unselected_outputs: items (keys & values) not selected in `cond_frame_outputs`.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}

        # the closest conditioning frame before `frame_idx` (if any)
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        # the closest conditioning frame after `frame_idx` (if any)
        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        # add other temporally closest conditioning frames until reaching a total
        # of `max_cond_frame_num` conditioning frames.
        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs