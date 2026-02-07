import logging
import math
import sys
from typing import Optional, Tuple, Literal

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from cell_observatory_platform.models.layers import patch_embeddings
from cell_observatory_platform.training.helpers import get_patch_sizes

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- ---- SinCos Embedding --- ---


def sincos(embed_dim, pos, temperature=10000, dtype=np.float32):
    # exponent: [embed_dim//2]
    exponent = np.arange(embed_dim // 2, dtype=dtype) / (embed_dim / 2.0)
    w = 1.0 / temperature**exponent

    # pos: [sequence_length]
    pos = pos.reshape(-1)
    # outer product: h,c -> hc: [sequence_length, embed_dim//2]
    hc = np.einsum("n,c->nc", pos, w)
    # returns: [sequence_length, embed_dim]
    return np.concatenate([np.sin(hc), np.cos(hc)], axis=1)


def positional_encoding_1d(embed_dim, sequence_length, temperature=10000, cls_token=False, dtype=None):
    """
    N = sequence_length
    Returns:
        [N + 1, embed_dim] if cls_token=True
        [N, embed_dim] else
    """
    dtype = dtype if dtype is not None else np.float32
    pos = np.arange(sequence_length, dtype=dtype)
    emb = sincos(embed_dim=embed_dim, pos=pos, temperature=temperature, dtype=dtype)

    if cls_token:
        return np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    else:
        return emb


def positional_encoding_2d(
    embed_dim, lateral_y_sequence_length, lateral_x_sequence_length, temperature=10000, cls_token=False, dtype=None
):
    """
    N = sequence_length^2
    Returns:
        [N + 1, embed_dim] if cls_token=True
        [N, embed_dim] else
    """
    num_dims = 2
    dtype = dtype if dtype is not None else np.float32
    d = int(np.floor(embed_dim / (2 * num_dims)) * 2)
    pad = embed_dim - (d * num_dims)

    xgrid = np.arange(lateral_x_sequence_length, dtype=dtype)
    ygrid = np.arange(lateral_y_sequence_length, dtype=dtype)
    ygrid, xgrid = np.meshgrid(ygrid, xgrid, indexing="ij")

    # outer product of y/x index for each (y,x) in LxL and frequencies
    yemb = sincos(embed_dim=d, pos=ygrid, temperature=temperature, dtype=dtype)  # (N, d)
    xemb = sincos(embed_dim=d, pos=xgrid, temperature=temperature, dtype=dtype)  # (N, d)
    emb = np.concatenate([yemb, xemb], axis=1)  # (N, d*2)

    if pad > 0:
        emb = np.pad(emb, ((0, 0), (0, pad)), mode="constant", constant_values=0)  # (N, embed_dim)

    if cls_token:
        return np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    else:
        return emb


def positional_encoding_3d(
    embed_dim,
    lateral_x_sequence_length,
    lateral_y_sequence_length,
    axial_sequence_length=None,
    temporal_sequence_length=None,
    temperature=10000,
    cls_token=False,
    dtype=None,
):
    """
    N = lateral_sequence_length^2 * (axial_sequence_length or temporal_sequence_length)
    Returns:
        [N + 1, embed_dim] if cls_token=True
        [N, embed_dim] else
    """
    num_dims = 3
    dtype = dtype if dtype is not None else np.float32
    d = int(np.floor(embed_dim / (2 * num_dims)) * 2)
    pad = embed_dim - (d * num_dims)

    if axial_sequence_length is not None and temporal_sequence_length is not None:
        raise ValueError("Use `positional_encoding_4d` if you have both axial and temporal sequence_length")

    xgrid = np.arange(lateral_x_sequence_length, dtype=dtype)
    ygrid = np.arange(lateral_y_sequence_length, dtype=dtype)

    if axial_sequence_length is not None:
        zgrid = np.arange(axial_sequence_length, dtype=dtype)
    else:
        zgrid = np.arange(temporal_sequence_length, dtype=dtype)

    zgrid, ygrid, xgrid = np.meshgrid(zgrid, ygrid, xgrid, indexing="ij")

    zemb = sincos(embed_dim=d, pos=zgrid, temperature=temperature, dtype=dtype)  # (N, d)
    yemb = sincos(embed_dim=d, pos=ygrid, temperature=temperature, dtype=dtype)  # (N, d)
    xemb = sincos(embed_dim=d, pos=xgrid, temperature=temperature, dtype=dtype)  # (N, d)
    emb = np.concatenate([zemb, yemb, xemb], axis=1)  # (N, d*3)

    if pad > 0:
        emb = np.pad(emb, ((0, 0), (0, pad)), mode="constant", constant_values=0)  # (N, embed_dim)

    if cls_token:
        return np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    else:
        return emb


def positional_encoding_4d(
    embed_dim,
    lateral_x_sequence_length,
    lateral_y_sequence_length,
    axial_sequence_length,
    temporal_sequence_length,
    temperature=10000,
    cls_token=False,
    dtype=None,
):
    """
    N = lateral_sequence_length^2 * axial_sequence_length * temporal_sequence_length
    Returns:
        [N + 1, embed_dim] if cls_token=True
        [N, embed_dim] else
    """
    num_dims = 4
    dtype = dtype if dtype is not None else np.float32
    d = int(np.floor(embed_dim / (2 * num_dims)) * 2)
    pad = embed_dim - (d * num_dims)

    xgrid = np.arange(lateral_x_sequence_length, dtype=dtype)
    ygrid = np.arange(lateral_y_sequence_length, dtype=dtype)
    zgrid = np.arange(axial_sequence_length, dtype=dtype)
    tgrid = np.arange(temporal_sequence_length, dtype=dtype)
    tgrid, zgrid, ygrid, xgrid = np.meshgrid(tgrid, zgrid, ygrid, xgrid, indexing="ij")

    temb = sincos(embed_dim=d, pos=tgrid, temperature=temperature, dtype=dtype)  # (N, d)
    zemb = sincos(embed_dim=d, pos=zgrid, temperature=temperature, dtype=dtype)  # (N, d)
    yemb = sincos(embed_dim=d, pos=ygrid, temperature=temperature, dtype=dtype)  # (N, d)
    xemb = sincos(embed_dim=d, pos=xgrid, temperature=temperature, dtype=dtype)  # (N, d)
    emb = np.concatenate([temb, zemb, yemb, xemb], axis=1)  # (N, d*4)

    if pad > 0:
        emb = np.pad(emb, ((0, 0), (0, pad)), mode="constant", constant_values=0)  # (N, embed_dim)

    if cls_token:
        return np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    else:
        return emb


class PosEmbedding(nn.Module):
    def __init__(
        self,
        input_fmt="TZYXC",
        input_shape=(16, 128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        embed_dim=768,
        channels=1,
        cls_token=False,
        interpolate=False,
    ):
        super().__init__()

        self.input_fmt = input_fmt
        self.input_shape = input_shape

        self.temporal_patch_size, self.axial_patch_size, self.lateral_patch_size = get_patch_sizes(
            input_format=input_fmt, patch_shape=patch_shape
        )

        self.embed_dim = embed_dim
        self.channels = channels
        self.cls_token = cls_token
        assert not self.cls_token, "CLS token not yet supported for PosEmbedding"
        self.interpolate = interpolate

        self.num_patches, self.token_shape = patch_embeddings.calc_num_patches(
            input_fmt=self.input_fmt, input_shape=self.input_shape, patch_shape=patch_shape
        )

        num_patches_pos_embed = self.num_patches + 1 if self.cls_token else self.num_patches
        self.register_buffer("pos_embed", torch.zeros(1, num_patches_pos_embed, self.embed_dim), persistent=True)
        self._init_pos_embed(self.pos_embed.data)

    def _init_pos_embed(self, pos_embed):
        if self.input_fmt == "TZYXC":
            sincos = positional_encoding_4d(
                embed_dim=self.embed_dim,
                temporal_sequence_length=self.input_shape[0] // self.temporal_patch_size,
                axial_sequence_length=self.input_shape[1] // self.axial_patch_size,
                lateral_y_sequence_length=self.input_shape[2] // self.lateral_patch_size,
                lateral_x_sequence_length=self.input_shape[3] // self.lateral_patch_size,
                cls_token=self.cls_token,
            )

        elif self.input_fmt == "ZYXC":
            sincos = positional_encoding_3d(
                embed_dim=self.embed_dim,
                temporal_sequence_length=None,
                axial_sequence_length=self.input_shape[0] // self.axial_patch_size,
                lateral_y_sequence_length=self.input_shape[1] // self.lateral_patch_size,
                lateral_x_sequence_length=self.input_shape[2] // self.lateral_patch_size,
                cls_token=self.cls_token,
            )

        elif self.input_fmt == "TYXC":
            sincos = positional_encoding_3d(
                embed_dim=self.embed_dim,
                axial_sequence_length=None,
                temporal_sequence_length=self.input_shape[0] // self.temporal_patch_size,
                lateral_y_sequence_length=self.input_shape[1] // self.lateral_patch_size,
                lateral_x_sequence_length=self.input_shape[2] // self.lateral_patch_size,
                cls_token=self.cls_token,
            )

        elif self.input_fmt == "YXC":
            sincos = positional_encoding_2d(
                embed_dim=self.embed_dim,
                lateral_y_sequence_length=self.input_shape[0] // self.lateral_patch_size,
                lateral_x_sequence_length=self.input_shape[1] // self.lateral_patch_size,
                cls_token=self.cls_token,
            )

        elif self.input_fmt == "XC":
            sincos = positional_encoding_1d(
                embed_dim=self.embed_dim,
                sequence_length=self.input_shape[0] // self.lateral_patch_size,
                cls_token=self.cls_token,
            )

        else:
            raise NotImplementedError

        logger.info(f"Initializing positional embedding with Sin/Cos encoding:")
        logger.info(f"{self.input_shape=}, {self.input_fmt=}")
        logger.info(f"{self.temporal_patch_size=}, {self.axial_patch_size=}, {self.lateral_patch_size=}")
        logger.info(f"({self.num_patches=}, {self.embed_dim=}) -> {sincos.shape=}")

        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def interpolate_positional_encoding(self, x, pos_embed):
        B = pos_embed.shape[0]
        C = self.embed_dim

        # strip cls token if present
        if self.cls_token and pos_embed.shape[1] == 1 + self.num_patches:
            cls_token, pos = pos_embed[:, :1, :], pos_embed[:, 1:, :]
        else:
            cls_token, pos = None, pos_embed

        # resize ND positional grid while keeping channels-first for interpolate
        def resize_1d(pe, L0, L1):
            if L0 == L1:
                return pe
            # (B,L,C) -> (B,C,L)
            pe = pe.reshape(B, L0, C).permute(0, 2, 1)
            pe = F.interpolate(pe, size=L1, mode="linear", align_corners=False)
            return pe.permute(0, 2, 1).reshape(B, L1, C)

        def resize_2d(pe, Z0, Y0, Z1, Y1):
            if (Z0, Y0) == (Z1, Y1):
                return pe
            # (B,H,W,C) -> (B,C,H,W)
            pe = pe.reshape(B, Z0, Y0, C).permute(0, 3, 1, 2)
            pe = F.interpolate(pe, size=(Z1, Y1), mode="bilinear", align_corners=False)
            return pe.permute(0, 2, 3, 1).reshape(B, Z1 * Y1, C)

        def resize_3d(pe, Z0, Y0, X0, Z1, Y1, X1):
            if (Z0, Y0, X0) == (Z1, Y1, X1):
                return pe
            # (B,D,H,W,C) -> (B,C,D,H,W)
            pe = pe.reshape(B, Z0, Y0, X0, C).permute(0, 4, 1, 2, 3)
            pe = F.interpolate(pe, size=(Z1, Y1, X1), mode="trilinear", align_corners=False)
            return pe.permute(0, 2, 3, 4, 1).reshape(B, Z1 * Y1 * X1, C)

        def resize_T_ZYX_separable(pe, T0, Z0, Y0, X0, T1, Z1, Y1, X1):
            # 3D over ZYX per T, then 1D over T
            if (T0, Z0, Y0, X0) == (T1, Z1, Y1, X1):
                return pe
            # (B, T0, Z0, Y0, X0, C)
            pe = pe.reshape(B, T0, Z0, Y0, X0, C)
            # 3D over ZYX per T -> fold T into batch
            pe = pe.permute(0, 1, 5, 2, 3, 4).reshape(B * T0, C, Z0, Y0, X0)
            pe = F.interpolate(pe, size=(Z1, Y1, X1), mode="trilinear", align_corners=False)
            pe = pe.reshape(B, T0, C, Z1, Y1, X1)
            # 1D over T -> fold (C, Z1, Y1, X1) into channels and interpolate length T
            pe = pe.permute(0, 3, 4, 5, 2, 1).reshape(B, C * Z1 * Y1 * X1, T0)  # (B, C*Z*Y*X, T)
            pe = F.interpolate(pe, size=T1, mode="linear", align_corners=False)
            pe = pe.reshape(B, Z1, Y1, X1, C, T1).permute(0, 5, 1, 2, 3, 4)  # (B, T1, Z1, Y1, X1, C)
            return pe.reshape(B, T1 * Z1 * Y1 * X1, C)

        def resize_T_YX_separable(pe, T0, Y0, X0, T1, Y1, X1):
            # 2D over YX per T, then 1D over T
            if (T0, Y0, X0) == (T1, Y1, X1):
                return pe
            # (B, T0, Y0, X0, C)
            pe = pe.reshape(B, T0, Y0, X0, C)
            # 2D over YX per T -> fold T into batch
            pe = pe.permute(0, 1, 4, 2, 3).reshape(B * T0, C, Y0, X0)
            pe = F.interpolate(pe, size=(Y1, X1), mode="bilinear", align_corners=False)
            pe = pe.reshape(B, T0, C, Y1, X1)
            # 1D over T -> fold (C, Y1, X1) into channels and interpolate length T
            pe = pe.permute(0, 3, 4, 2, 1).reshape(B, C * Y1 * X1, T0)
            pe = F.interpolate(pe, size=T1, mode="linear", align_corners=False)
            pe = pe.reshape(B, Y1, X1, C, T1).permute(0, 4, 1, 2, 3)
            return pe.reshape(B, T1 * Y1 * X1, C)

        T0, Z0, Y0, X0, C0 = self.token_shape

        # compute original & target grid sizes from x and config
        if self.input_fmt == "TZYXC":
            T1 = x.shape[1] // self.temporal_patch_size
            Z1 = x.shape[2] // self.axial_patch_size
            Y1 = x.shape[3] // self.lateral_patch_size
            X1 = x.shape[4] // self.lateral_patch_size

            pos = resize_T_ZYX_separable(pos, T0, Z0, Y0, X0, T1, Z1, Y1, X1)

        elif self.input_fmt == "ZYXC":
            Z1 = x.shape[1] // self.axial_patch_size
            Y1 = x.shape[2] // self.lateral_patch_size
            X1 = x.shape[3] // self.lateral_patch_size

            pos = resize_3d(pos, Z0, Y0, X0, Z1, Y1, X1)

        elif self.input_fmt == "TYXC":
            T1 = x.shape[1] // self.temporal_patch_size
            Y1 = x.shape[2] // self.lateral_patch_size
            X1 = x.shape[3] // self.lateral_patch_size

            pos = resize_T_YX_separable(pos, T0, Y0, X0, T1, Y1, X1)

        elif self.input_fmt == "YXC":
            Y1 = x.shape[1] // self.lateral_patch_size
            X1 = x.shape[2] // self.lateral_patch_size

            pos = resize_2d(pos, Y0, X0, Y1, X1)

        elif self.input_fmt == "XC":
            X1 = x.shape[1] // self.lateral_patch_size

            pos = resize_1d(pos, X0, X1)

        else:
            raise NotImplementedError(f"Unknown input_fmt={self.input_fmt}")

        # restore cls if it existed
        if cls_token is not None:
            pos = torch.cat([cls_token, pos], dim=1)

        return pos

    def gather_pos_table(self, pos_table, patches_used):
        idx = patches_used.to(pos_table.device)
        D = pos_table.size(-1)
        # pos_table_batched: (B, L_full, D) -> (B, L_used, D)
        out = torch.gather(pos_table, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, D))
        return out

    @property
    def table(self) -> torch.Tensor:
        # (L_full, D) without cls
        return self.pos_embed[:, 1:, :] if self.cls_token else self.pos_embed

    def forward(self, x: Optional[torch.Tensor] = None, patches_used: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.interpolate:
            # FIXME: interpolate_positional_encoding assumes x is grid format
            #        but currently we pass in sequence format from modules
            pos_table_interpolated = self.interpolate_positional_encoding(x, self.pos_embed)

            if patches_used is not None:
                # NOTE: gather_pos_table assumes batched pos_table
                pos_table_subset = self.gather_pos_table(pos_table_interpolated, patches_used)
                return pos_table_subset
            else:
                return pos_table_interpolated

        if patches_used is not None:
            pos_table = self.table
            B, L_used = patches_used.shape
            pos_table_batched = pos_table.expand(B, -1, -1)
            return self.gather_pos_table(pos_table_batched, patches_used)

        return self.pos_embed


class PositionalEmbeddingSinCos(nn.Module):
    """
    Adapted from:
    https://github.com/IDEA-Research/MaskDINO/main/maskdino/modeling/pixel_decoder/position_encoding.py
    """

    def __init__(self, num_pos_feats, temperature=10000, normalize=False, scale=None):
        super().__init__()

        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    # TODO: run logic in self.dtype instead of casting
    def _forward_queries(self, x, shape, mask=None):
        out_dtype = x.dtype

        F = int(self.num_pos_feats)
        Fe = F - (F % 2)

        # dim_t: (num_pos_feats,)
        dim_t = torch.arange(Fe, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / Fe)

        # x: (N, L, 3) or (N, L, 6)
        x_embed = x[:, :, 0] * self.scale
        y_embed = x[:, :, 1] * self.scale
        z_embed = x[:, :, 2] * self.scale

        # x: (N, L, num_pos_feats)
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_z = z_embed[:, :, None] / dim_t

        # (N,L,num_pos_feats/2,2) [sin, cos] pairs of positional theta values
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)

        if x.size(-1) == 3:
            pos = torch.cat((pos_z, pos_y, pos_x), dim=2)
            remainder = 3 * F - 3 * Fe
            if remainder > 0:
                pos = torch.cat([pos, pos.new_zeros(pos.shape[0], pos.shape[1], remainder)], dim=2)

        elif x.size(-1) == 6:
            w_embed = x[:, :, 3] * self.scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(-2)

            h_embed = x[:, :, 4] * self.scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(-2)

            d_embed = x[:, :, 5] * self.scale
            pos_d = d_embed[:, :, None] / dim_t
            pos_d = torch.stack((pos_d[:, :, 0::2].sin(), pos_d[:, :, 1::2].cos()), dim=3).flatten(-2)

            pos = torch.cat((pos_z, pos_y, pos_x, pos_w, pos_h, pos_d), dim=2)
            # TODO: decide on alternative order of dimensions
            # pos = torch.cat((pos_z, pos_y, pos_x, pos_d, pos_h, pos_w), dim=2)
            # pos = torch.cat((pos_x, pos_y, pos_z, pos_w, pos_h, pos_d), dim=2)

            remainder = 6 * F - 6 * Fe
            if remainder > 0:
                pos = torch.cat([pos, pos.new_zeros(pos.shape[0], pos.shape[1], remainder)], dim=2)

        else:
            raise ValueError("Unknown x shape(-1):{}".format(x.size(-1)))

        # cast back to input dtype
        if pos.dtype != out_dtype:
            pos = pos.to(out_dtype)

        return pos

    def _forward_image(self, x, shape, mask=None):
        N, C, D, H, W = shape
        out_dtype = x.dtype
        if mask is None:
            mask = torch.zeros((N, D, H, W), device=x.device, dtype=torch.bool)
        not_mask = ~mask

        # cumsum gives a valid position sequence
        # even with padding
        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            # gives count at last z/y/x position => normalizes to [0,1]
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        F = int(self.num_pos_feats)
        Fe = F - (F % 2)

        dim_t = torch.arange(Fe, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / Fe)

        # (N, D, H, W, num_pos_feats)
        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t

        # (N, D, H, W, 2, Fe/2) [sin, cos] pairs of positional theta values
        # for each (N, D, H, W) position -> (N, D, H, W, num_pos_feats)
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=4).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=4).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=4).flatten(4)

        #  (N, D, H, W, 3*num_pos_feats) -> (N, 3*num_pos_feats, D, H, W)
        pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)
        remainder = 3 * F - 3 * Fe
        if remainder > 0:
            pos = torch.cat([pos, pos.new_zeros(N, remainder, D, H, W)], dim=1)

        if pos.dtype != out_dtype:
            pos = pos.to(out_dtype)

        return pos

    def forward(self, x, mask=None):
        if x.dim() == 5:
            # x is a 5D tensor (N, C, D, H, W)
            shape = x.shape
            return self._forward_image(x, shape, mask)
        elif x.dim() == 3:
            # x is a 3D tensor (N, C, L)
            shape = x.shape
            return self._forward_queries(x, shape, mask)
        else:
            raise ValueError(f"Unsupported input tensor shape: {x.shape}. Expected 3D or 5D tensor.")


# --- --- ROPE helpers --- ---


# based on: https://github.com/naver-ai/rope-vit and extended to 3D and 4D
def generate_frequency_spectrum(
    dim: int,
    num_heads: int,
    theta: float = 10.0,
    random_rotation_per_head: bool = True,
    input_fmt: str = "TZYXC",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    if input_fmt == "YXC":
        # assert dim % 4 == 0, "head_dim must be divisible by 4 for 2D ROPE."
        freqs_x, freqs_y = [], []
        # generate frequency spectrum: 1 / (theta ** (4i / d)) for i = 0, ..., d/4 - 1
        mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        for i in range(num_heads):
            # we can either have a random rotation per head, or the same rotation for all heads
            # if random rotation per head: sample uniform distribution on the interval [0,2pi)
            # this way each attention head gets its own 2-D orientation
            # so different heads are initialized with different basis sets
            angles = torch.rand(1) * 2 * torch.pi if random_rotation_per_head else torch.zeros(1)
            # form: (cosϕ,-sinϕ) and (sinϕ,cosϕ) where we use (cos(ϕ+π/2​),sin(ϕ+π/2​))=(−sinϕ,cosϕ)
            # if zeros we get (1,0) and (0,1) as in the standard RoPE
            # after referring to compute_mixed_cis, we see that this implies that the sequence is
            # given by [mag_k(x_i*cosϕ + y_i*sinϕ),mag_k(-x_i*sinϕ+y_i*cosϕ)]
            fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1)
            fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1)
            freqs_x.append(fx)
            freqs_y.append(fy)
        freqs_x = torch.stack(freqs_x, dim=0)
        freqs_y = torch.stack(freqs_y, dim=0)
        # freqs: (2, num_heads, dim//4 * 2)
        freqs = torch.stack([freqs_x, freqs_y], dim=0)

    elif input_fmt == "TYXC" or input_fmt == "ZYXC":
        # the below follows the logic as above but generalized to 3D
        # assert dim % 6 == 0, "head_dim must be divisible by 6 for 3D ROPE."
        J = dim // 6

        base = torch.arange(0, dim, 6)[: (dim // 6)].float() / dim
        mag = theta ** (-base)

        # freqs: (3, num_heads, dim//6*3)
        freqs = torch.empty(3, num_heads, J * 3)

        for h in range(num_heads):
            if random_rotation_per_head:
                M3 = torch.randn(3, 3)
                # generate 3 orthonormal basis vectors
                # in R^3 from qr decomposition A = QR
                Q, _ = torch.linalg.qr(M3)
            else:
                Q = torch.eye(3)

            # build 3 blocks of length J
            blocks = []
            for k in range(3):
                # broadcast: [3,1] * [1,J] -> [3,J]
                blocks.append(Q[:, [k]] @ mag[None, :])
            # blocks: [3, J] -> freqs: [3, num_heads, dim//6]
            freqs[:, h, :] = torch.cat(blocks, dim=-1)

    elif input_fmt == "TZYXC":
        # the below follows the logic as above but generalized to 4D
        # assert dim % 8 == 0, "head_dim must be divisible by 8 for 4D ROPE."
        J = dim // 8

        base = torch.arange(0, dim, 8)[: (dim // 8)].float() / dim
        mag = theta ** (-base)

        # freqs: (4, num_heads, dim//8*4)
        freqs = torch.empty(4, num_heads, J * 4)

        for h in range(num_heads):
            if random_rotation_per_head:
                M4 = torch.randn(4, 4)
                # generate 4 orthonormal basis vectors
                # in R^4 from qr decomposition A = QR
                Q, _ = torch.linalg.qr(M4)
            else:
                Q = torch.eye(4)

            # build 4 blocks of length J
            blocks = []
            for k in range(4):
                # broadcast: [4,1] * [1,J] -> [4,J]
                blocks.append(Q[:, [k]] @ mag[None, :])
            # freqs: [4, num_heads, dim//8*4]
            freqs[:, h, :] = torch.cat(blocks, dim=-1)

    else:
        raise NotImplementedError(f"Unknown input_fmt={input_fmt}")

    return freqs.to(dtype=dtype, device=device)


def generate_custom_freqs(
    dim: int,
    num_heads: int,
    theta: float = 10.0,
    input_fmt: str = "TZYXC",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    min_period: int | None = None,
    max_period: int | None = None,
):
    if theta is not None:
        if input_fmt == "TZYXC":
            periods = theta ** (
                2 * torch.arange(dim // 8, device=device, dtype=dtype) / (dim // 4)
            )  # [D//8]
        elif input_fmt == "ZYXC":
            periods = theta ** (
                2 * torch.arange(dim // 6, device=device, dtype=dtype) / (dim // 3)
            )  # [D//6]
        else:
            raise NotImplementedError(f"Unknown input_fmt={input_fmt}")
    else:
        base = max_period / min_period
        if input_fmt == "TZYXC":
            exponents = torch.linspace(0, 1, dim // 8, device=device, dtype=dtype)  # [D//8] range [0, 1]
        elif input_fmt == "ZYXC":
            exponents = torch.linspace(0, 1, dim // 6, device=device, dtype=dtype)  # [D//6] range [0, 1]
        else:
            raise NotImplementedError(f"Unknown input_fmt={input_fmt}")
        periods = base**exponents  # range [1, max_period / min_period]
        periods = periods / base  # range [min_period / max_period, 1]
        periods = periods * max_period  # range [min_period, max_period]
    return periods


def generate_grid_indices(
    end_x: int,
    end_y: int,
    end_z: Optional[int] = None,
    end_t: Optional[int] = None,
    input_fmt: str = "TZYXC",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    need_T = "T" in input_fmt
    need_Z = "Z" in input_fmt
    need_Y = "Y" in input_fmt
    need_X = "X" in input_fmt

    assert need_X and need_Y, "X and Y must be present in all supported formats."

    T = int(end_t) if need_T else 1
    Z = int(end_z) if need_Z else 1
    Y = int(end_y)
    X = int(end_x)

    N = T * Z * Y * X

    idx = torch.arange(N)
    x = idx % X
    y = (idx // X) % Y
    z = None
    t = None

    if need_Z:
        z = (idx // (X * Y)) % Z
    if need_T:
        t = idx // (X * Y * (Z if need_Z else 1))

    t = t.to(dtype=dtype, device=device) if t is not None else None
    z = z.to(dtype=dtype, device=device) if z is not None else None
    y = y.to(dtype=dtype, device=device)
    x = x.to(dtype=dtype, device=device)

    return (t, z, y, x)


def compute_mixed_cis(
    freqs: torch.Tensor,
    t_x: torch.Tensor,
    t_y: torch.Tensor,
    t_z: Optional[torch.Tensor] = None,
    t_t: Optional[torch.Tensor] = None,
    input_fmt: str = "TZYXC",
):

    def _outer_pos_freq(pos, f):
        if pos.dim() == 1:
            # [N,1] * [H,1,J] -> [H,N,J]
            return pos.unsqueeze(-1) @ f.unsqueeze(-2)
        else:
            # broadcast: [B,1,N,1] * [1,H,1,J] -> [B,H,N,J]
            return pos[:, None, :, None] * f[None, :, None, :]

    with torch.amp.autocast(enabled=False, device_type="cuda"):
        if input_fmt == "YXC":
            # [H,N,Jx] or [B,H,N,Jx]
            fx = _outer_pos_freq(t_x, freqs[0])
            # [H,N,Jy] or [B,H,N,Jy]
            fy = _outer_pos_freq(t_y, freqs[1])
            phases = fx + fy
        elif input_fmt == "TYXC":
            assert t_t is not None, "t_t must be provided for TYXC format"
            fx = _outer_pos_freq(t_x, freqs[0])
            fy = _outer_pos_freq(t_y, freqs[1])
            ft = _outer_pos_freq(t_t, freqs[2])
            phases = fx + fy + ft
        elif input_fmt == "ZYXC":
            assert t_z is not None, "t_z must be provided for ZYXC format"
            fx = _outer_pos_freq(t_x, freqs[0])
            fy = _outer_pos_freq(t_y, freqs[1])
            fz = _outer_pos_freq(t_z, freqs[2])
            phases = fx + fy + fz
        elif input_fmt == "TZYXC":
            assert t_t is not None and t_z is not None, "t_t and t_z must be provided for TZYXC format"
            fx = _outer_pos_freq(t_x, freqs[0])
            fy = _outer_pos_freq(t_y, freqs[1])
            fz = _outer_pos_freq(t_z, freqs[2])
            ft = _outer_pos_freq(t_t, freqs[3])
            phases = fx + fy + fz + ft
        else:
            raise NotImplementedError

        if phases.dtype != torch.float32:  # polar doesn't support bf16
            dtype = phases.dtype
            ones = torch.ones_like(phases, dtype=torch.float32)
            freqs_cis = torch.polar(ones, phases.to(torch.float32)).to(dtype)
        else:
            ones = torch.ones_like(phases)
            freqs_cis = torch.polar(ones, phases)

        return freqs_cis


def make_axial_rope_freqs(
    input_fmt: str,
    input_shape: tuple,
    patch_shape: tuple,
    dim: int,
    theta: float = 100.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute axial RoPE frequencies for a given input geometry.

    This is a top-level helper that should be called once by top-level modules
    (ViT, MaskedEncoder, MaskedPredictor, etc.) and stored as a buffer.

    Args:
        input_fmt: e.g. ``"TZYXC"``, ``"ZYXC"``, ``"TYXC"``.
        input_shape: full input tensor shape matching *input_fmt*.
        patch_shape: patch sizes matching the spatial/temporal axes.
        dim: per-head dimension (``embed_dim // num_heads``).
        theta: RoPE base frequency.
        device: target device (defaults to CPU).

    Returns:
        Complex-valued tensor of shape ``(N_patches, dim // 2)`` (the
        exact last-dim width depends on *input_fmt*).
    """
    if device is None:
        device = torch.device("cpu")

    temporal_patch_size, axial_patch_size, lateral_patch_size = get_patch_sizes(
        input_format=input_fmt, patch_shape=patch_shape
    )

    end_x = input_shape[input_fmt.index("X")] // lateral_patch_size
    end_y = input_shape[input_fmt.index("Y")] // lateral_patch_size

    end_z = (input_shape[input_fmt.index("Z")] // axial_patch_size) if "Z" in input_fmt else None
    end_t = (input_shape[input_fmt.index("T")] // temporal_patch_size) if "T" in input_fmt else None

    return compute_axial_cis(
        dim=dim,
        end_x=end_x,
        end_y=end_y,
        end_z=end_z,
        end_t=end_t,
        input_fmt=input_fmt,
        theta=theta,
        device=device,
    )


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    end_z: int,
    end_t: int,
    input_fmt: str = "TZYXC",
    theta: float = 100.0,
    device: str = "cuda",
):
    # NOTE: in the paper they define: R(n,2t)=e{iθ_{t}​p^{n}_{x}​,R(n,2t+1)=eθ_{t}​p^{n}_{y}
    #       however in the reference code the assignment per embedding dimension is:
    #       [x-slot, x-slot, ..., y-slot, y-slot, ...] i.e. the specific assignment of
    #       x,y positions to dimensions is not interleaved but blockwise

    if input_fmt == "YXC":
        # assert dim % 4 == 0, "head_dim must be divisible by 4 for 2D ROPE."
        mag = 1.0 / (theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim))

        t_t, t_z, t_y, t_x = generate_grid_indices(
            end_x=end_x, end_y=end_y, input_fmt=input_fmt, device=device, dtype=torch.float32
        )

        freqs_x = torch.outer(t_x, mag)
        freqs_y = torch.outer(t_y, mag)

    elif input_fmt == "TYXC":
        # assert dim % 6 == 0, "head_dim must be divisible by 6 for 3D ROPE."
        base = torch.arange(0, dim, 6, device=device)[: (dim // 6)].float() / dim
        mag = theta ** (-base)

        t_t, t_z, t_y, t_x = generate_grid_indices(
            end_x=end_x, end_y=end_y, end_t=end_t, input_fmt=input_fmt, device=device, dtype=torch.float32
        )

        freqs_x = torch.outer(t_x, mag)
        freqs_y = torch.outer(t_y, mag)
        freqs_t = torch.outer(t_t, mag)

    elif input_fmt == "ZYXC":
        # assert dim % 6 == 0, "head_dim must be divisible by 6 for 3D ROPE."
        base = torch.arange(0, dim, 6, device=device)[: (dim // 6)].float() / dim
        mag = theta ** (-base)

        t_t, t_z, t_y, t_x = generate_grid_indices(
            end_x=end_x, end_y=end_y, end_z=end_z, input_fmt=input_fmt, device=device, dtype=torch.float32
        )

        freqs_x = torch.outer(t_x, mag)
        freqs_y = torch.outer(t_y, mag)
        freqs_z = torch.outer(t_z, mag)

    elif input_fmt == "TZYXC":
        # assert dim % 8 == 0, "head_dim must be divisible by 8 for 4D ROPE."
        base = torch.arange(0, dim, 8, device=device)[: (dim // 8)].float() / dim
        mag = theta ** (-base)

        t_t, t_z, t_y, t_x = generate_grid_indices(
            end_x=end_x, end_y=end_y, end_z=end_z, end_t=end_t, input_fmt=input_fmt, device=device, dtype=torch.float32
        )

        freqs_x = torch.outer(t_x, mag)
        freqs_y = torch.outer(t_y, mag)
        freqs_z = torch.outer(t_z, mag)
        freqs_t = torch.outer(t_t, mag)

    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)

    if input_fmt == "YXC":
        return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

    if input_fmt == "TYXC":
        freqs_cis_t = torch.polar(torch.ones_like(freqs_t), freqs_t)
        return torch.cat([freqs_cis_x, freqs_cis_y, freqs_cis_t], dim=-1)

    elif input_fmt == "ZYXC":
        freqs_cis_z = torch.polar(torch.ones_like(freqs_z), freqs_z)
        return torch.cat([freqs_cis_x, freqs_cis_y, freqs_cis_z], dim=-1)

    elif input_fmt == "TZYXC":
        freqs_cis_z = torch.polar(torch.ones_like(freqs_z), freqs_z)
        freqs_cis_t = torch.polar(torch.ones_like(freqs_t), freqs_t)
        return torch.cat([freqs_cis_x, freqs_cis_y, freqs_cis_z, freqs_cis_t], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # freqs_cis: (N, J) branch
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        # freq_cis reshaped to (1, 1, N, J) since x: [B, H, N, J]
        shape = [d if i >= x.ndim - 2 else 1 for i, d in enumerate(x.shape)]
    # freqs_cis: (H, N, J) branch
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        # freq_cis reshaped to (1, H, N, J) since x: [B, H, N, J]
        shape = [d if i >= x.ndim - 3 else 1 for i, d in enumerate(x.shape)]
    # freqs_cis: (B, N, J) branch
    elif freqs_cis.shape == (x.shape[0], x.shape[-2], x.shape[-1]):
        shape = [x.shape[0], 1, x.shape[-2], x.shape[-1]]
    # freqs_cis: (B, H, N, J) branch
    elif freqs_cis.shape == (x.shape[0], x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [x.shape[0], x.shape[-3], x.shape[-2], x.shape[-1]]
    else:
        raise ValueError(f"Unexpected freqs_cis shape: {freqs_cis.shape} for x shape: {x.shape}")
    return freqs_cis.view(*shape)


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, pos_enc: torch.Tensor, rope_type: Literal["mixed", "axial", "custom"]):
    if rope_type == "mixed":
        return apply_rope_v1(xq, xk, pos_enc)
    elif rope_type == "axial":
        return apply_rope_v1(xq, xk, pos_enc)
    elif rope_type == "custom":
        return apply_rope_v2(xq, xk, pos_enc)
    else:
        raise ValueError(f"Unknown rope type: {rope_type}")


def apply_rope_v1(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # xq: [B,H,N,D]
    Jf = freqs_cis.shape[-1]
    De = Jf * 2

    # split off the tail that cannot be rotated cleanly
    xq_even = xq[..., :De]
    xk_even = xk[..., :De]
    xq_tail = xq[..., De:]
    xk_tail = xk[..., De:]

    # xq[:-1]: [B, H, N] => xq reshape: [B, H, N, J, 2] for J=D/2
    # thus xq_: [B, H, N, J] complex and similar for xk
    xq_ = torch.view_as_complex(xq_even.float().reshape(*xq_even.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk_even.float().reshape(*xk_even.shape[:-1], -1, 2))

    # if [N, J] -> reshaped to [1, 1, N, J]
    # if [H, N, J] -> reshaped to [1, H, N, J]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_).to(xq_.device)

    # xq_ * freqs_cis: elementwise complex mult -> [B, H, N, J]
    # then view_as_real -> [B, H, N, J, 2] -> flatten last two dims -> [B, H, N, J]
    xq_rot = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_rot = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    if xq_tail.numel():
        xq_out = torch.cat([xq_rot, xq_tail], dim=-1)
        xk_out = torch.cat([xk_rot, xk_tail], dim=-1)
    else:
        xq_out = xq_rot
        xk_out = xk_rot

    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def apply_rope_v2(q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
    q_dtype = q.dtype
    k_dtype = k.dtype
    sin, cos = rope
    rope_dtype = sin.dtype
    q = q.to(dtype=rope_dtype)
    k = k.to(dtype=rope_dtype)
    N = q.shape[-2]
    prefix = N - sin.shape[-2]
    assert prefix >= 0
    q_prefix = q[:, :, :prefix, :]
    q = apply_rope_half(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
    q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
    k_prefix = k[:, :, :prefix, :]
    k = apply_rope_half(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
    k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
    q = q.to(dtype=q_dtype)
    k = k.to(dtype=k_dtype)
    return q, k


# Helper functions for RoPE:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_half(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
    if rope is None:
        return None

    sin, cos = rope
    assert sin.ndim == cos.ndim
    if sin.ndim == 4:
        # If the rope embedding has a batch dimension (is different for each batch element), index into it
        return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
    else:
        # No batch dimension, do not index
        return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]


# --- --- CUSTOM ROPE Class --- --


# RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights
# Supports two parametrizations of the rope parameters: either using `base` or `min_period` and `max_period`.
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        input_fmt: str,
        embed_dim: int,
        num_heads: int,
        theta: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.input_fmt = input_fmt
        if self.input_fmt == "TZYXC":
            ndim = 4
        elif self.input_fmt == "ZYXC":
            ndim = 3
        else:
            raise NotImplementedError(f"Unknown input_fmt={self.input_fmt}")    

        assert embed_dim % (2 * ndim * num_heads) == 0        
        
        both_periods = min_period is not None and max_period is not None
        if (theta is None and not both_periods) or (theta is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        self.theta = theta
        self.min_period = min_period
        self.max_period = max_period

        self.num_heads = num_heads
        D_head = embed_dim // num_heads        
        self.D_head = D_head
        
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self.device = device
        self.dtype = dtype  # Don't rely on self.periods.dtype
        P = self.D_head // (2 * ndim)
        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.register_buffer("periods", torch.empty(P, device=device, dtype=dtype), persistent=True)
        self._init_weights()

    def _init_weights(self):
        self.periods.data = generate_custom_freqs(
            dim=self.D_head,
            input_fmt=self.input_fmt,
            num_heads=self.num_heads,
            theta=self.theta,
            min_period=self.min_period,
            max_period=self.max_period,
            device=self.device,
            dtype=self.dtype,
        )

    def _max_dim(self, shape: tuple[int, int]) -> int:
        if self.input_fmt == "TZYXC":
            T, Z, Y, X = shape
            max_dim = max(T, Z, Y, X)
            return max_dim
        elif self.input_fmt == "ZYXC":
            Z, Y, X = shape
            max_dim = max(Z, Y, X)
            return max_dim
        else:
            raise NotImplementedError(f"Unknown input_fmt={self.input_fmt}")

    def _min_dim(self, shape: tuple[int, int]) -> int:
        if self.input_fmt == "TZYXC":
            T, Z, Y, X = shape
            min_dim = min(T, Z, Y, X)
            return min_dim
        elif self.input_fmt == "ZYXC":
            Z, Y, X = shape
            min_dim = min(Z, Y, X)
            return min_dim
        else:
            raise NotImplementedError(f"Unknown input_fmt={self.input_fmt}")

    def _get_coords_separate(self, shape: tuple[int, int], dd: dict) -> tuple[Tensor, Tensor]:
        if self.input_fmt == "TZYXC":
            T, Z, Y, X = shape
            coords_t = torch.arange(0.5, T, **dd) / T  # [T]
            coords_z = torch.arange(0.5, Z, **dd) / Z  # [Z]
            coords_y = torch.arange(0.5, Y, **dd) / Y  # [Y]
            coords_x = torch.arange(0.5, X, **dd) / X  # [X]
            return coords_t, coords_z, coords_y, coords_x
        elif self.input_fmt == "ZYXC":
            Z, Y, X = shape
            coords_z = torch.arange(0.5, Z, **dd) / Z  # [Z]
            coords_y = torch.arange(0.5, Y, **dd) / Y  # [Y]
            coords_x = torch.arange(0.5, X, **dd) / X  # [X]
            return coords_z, coords_y, coords_x
        else:
            raise NotImplementedError(f"Unknown input_fmt={self.input_fmt}")

    def _get_coords_max(self, shape: tuple[int, int], dd: dict) -> tuple[Tensor, Tensor]:
        if self.input_fmt == "TZYXC":
            T, Z, Y, X = shape
            max_dim = self._max_dim(shape)
            coords_t = torch.arange(0.5, T, **dd) / max_dim  # [T]
            coords_z = torch.arange(0.5, Z, **dd) / max_dim  # [Z]
            coords_y = torch.arange(0.5, Y, **dd) / max_dim  # [Y]
            coords_x = torch.arange(0.5, X, **dd) / max_dim  # [X]
            return coords_t, coords_z, coords_y, coords_x
        elif self.input_fmt == "ZYXC":
            Z, Y, X = shape
            max_dim = self._max_dim(shape)
            coords_z = torch.arange(0.5, Z, **dd) / max_dim  # [Z]
            coords_y = torch.arange(0.5, Y, **dd) / max_dim  # [Y]
            coords_x = torch.arange(0.5, X, **dd) / max_dim  # [X]
            return coords_z, coords_y, coords_x
        else:
            raise NotImplementedError(f"Unknown input_fmt={self.input_fmt}")

    def _get_coords_min(self, shape: tuple[int, int], dd: dict) -> tuple[Tensor, Tensor]:
        if self.input_fmt == "TZYXC":
            min_dim = self._min_dim(shape)
            T, Z, Y, X = shape
            coords_t = torch.arange(0.5, T, **dd) / min_dim  # [T]
            coords_z = torch.arange(0.5, Z, **dd) / min_dim  # [Z]
            coords_y = torch.arange(0.5, Y, **dd) / min_dim  # [Y]
            coords_x = torch.arange(0.5, X, **dd) / min_dim  # [X]
            return coords_t, coords_z, coords_y, coords_x
        elif self.input_fmt == "ZYXC":
            min_dim = self._min_dim(shape)
            Z, Y, X = shape
            coords_z = torch.arange(0.5, Z, **dd) / min_dim  # [Z]
            coords_y = torch.arange(0.5, Y, **dd) / min_dim  # [Y]
            coords_x = torch.arange(0.5, X, **dd) / min_dim  # [X]
            return coords_z, coords_y, coords_x
        else:
            raise NotImplementedError(f"Unknown input_fmt={self.input_fmt}")

    def _generate_coords(self, coords_per_dim: tuple[int, int]) -> tuple[Tensor, Tensor]:
        if self.input_fmt == "TZYXC":
            coords_t, coords_z, coords_y, coords_x = coords_per_dim
            coords = torch.stack(torch.meshgrid(coords_t, coords_z, coords_y, coords_x, indexing="ij"), dim=-1)  # [T, Z, Y, X, 4]
            coords = coords.flatten(0, 3)  # [T*Z*Y*X, 4]
            coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]
            return coords
        elif self.input_fmt == "ZYXC":
            coords_z, coords_y, coords_x = coords_per_dim
            coords = torch.stack(torch.meshgrid(coords_z, coords_y, coords_x, indexing="ij"), dim=-1)  # [Z, Y, X, 3]
            coords = coords.flatten(0, 2)  # [Z*Y*X, 3]
            coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]
            return coords
        else:
            raise NotImplementedError(f"Unknown input_fmt={self.input_fmt}")

    def _shift_coords(self, coords: Tensor, dd: dict) -> Tensor:
        if self.input_fmt == "TZYXC":
            shift_tzyx = torch.empty(4, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_tzyx[None, :]
        elif self.input_fmt == "ZYXC":
            shift_zyx = torch.empty(3, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_zyx[None, :]
        else:
            raise NotImplementedError(f"Unknown input_fmt={self.input_fmt}")
        return coords
    
    def _jitter_coords(self, coords: Tensor, dd: dict) -> Tensor:
        jitter_max = np.log(self.jitter_coords)
        jitter_min = -jitter_max
        if self.input_fmt == "TZYXC":
            jitter_tzyx = torch.empty(4, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_tzyx[None, :]
        elif self.input_fmt == "ZYXC":
            jitter_zyx = torch.empty(3, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_zyx[None, :]
        else:
            raise NotImplementedError(f"Unknown input_fmt={self.input_fmt}")
        return coords

    def _rescale_coords(self, coords: Tensor, dd: dict) -> Tensor:
        rescale_max = np.log(self.rescale_coords)
        rescale_min = -rescale_max
        rescale = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
        coords *= rescale
        return coords

    def forward(self, shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
        dd = {"device": self.periods.device, "dtype": self.dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            coords_per_dim = self._get_coords_max(shape, dd=dd)
        elif self.normalize_coords == "min":
            coords_per_dim = self._get_coords_min(shape, dd=dd)
        elif self.normalize_coords == "separate":
            coords_per_dim = self._get_coords_separate(shape, dd=dd)
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        coords = self._generate_coords(coords_per_dim)

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            coords = self._shift_coords(coords, dd=dd)

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            coords = self._jitter_coords(coords, dd=dd)

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            coords = self._rescale_coords(coords, dd=dd)

        ndim = coords.shape[1]
        P = self.periods.numel()
        # phases is D_head//2, and then tile(2) -> D_head
        if self.D_head != 2 * ndim * P:
            raise ValueError(
                f"Bad RoPE dims: D_head={self.D_head}, ndim={ndim}, periods={P}. "
                f"Need D_head == 2*ndim*periods == {2*ndim*P}."
            )

        # coords: [N,ndim] -> [N, ndim, 1], periods: [P] -> [1, 1, P] -> [N, ndim, P]
        # where P=D_head // (2 * ndim)
        phases = (2.0 * math.pi) * coords[:, :, None] / self.periods[None, None, :]
        phases = phases.flatten(1, 2)  # [N, D_head//2]

        angles = phases.repeat(1, 2)  # tile(2) -> [N, D_head]
        cos = torch.cos(angles).to(dtype=coords.dtype, device=coords.device)
        sin = torch.sin(angles).to(dtype=coords.dtype, device=coords.device)
        return sin, cos