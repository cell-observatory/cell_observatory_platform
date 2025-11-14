import sys
import logging
from typing import Optional, Tuple

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_observatory_platform.models import patch_embeddings
from cell_observatory_platform.training.helpers import get_patch_sizes

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- ---- SinCos Embedding --- ---


def sincos(embed_dim, pos, temperature=10000, dtype=np.float32):
    # exponent: [embed_dim//2]
    exponent = np.arange(embed_dim // 2, dtype=dtype) / (embed_dim / 2.)
    w = 1. / temperature ** exponent

    # pos: [sequence_length]
    pos = pos.reshape(-1)
    # outer product: h,c -> hc: [sequence_length, embed_dim//2]
    hc = np.einsum('n,c->nc', pos, w)
    # returns: [sequence_length, embed_dim]
    return np.concatenate([np.sin(hc) , np.cos(hc)], axis=1)


def positional_encoding_1d(
    embed_dim,
    sequence_length,
    temperature=10000,
    cls_token=False,
    dtype=None
):
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
    embed_dim,
    lateral_y_sequence_length,
    lateral_x_sequence_length,
    temperature=10000,
    cls_token=False,
    dtype=None
):
    """
    N = sequence_length^2
    Returns:
        [N + 1, embed_dim] if cls_token=True
        [N, embed_dim] else
    """
    num_dims = 2
    dtype = dtype if dtype is not None else np.float32
    d = int(np.floor(embed_dim / (2*num_dims)) * 2)
    pad = embed_dim - (d * num_dims)

    xgrid = np.arange(lateral_x_sequence_length, dtype=dtype)
    ygrid = np.arange(lateral_y_sequence_length, dtype=dtype)
    ygrid, xgrid = np.meshgrid(ygrid, xgrid, indexing='ij')

    # outer product of y/x index for each (y,x) in LxL and frequencies
    yemb = sincos(embed_dim=d, pos=ygrid, temperature=temperature, dtype=dtype)  # (N, d)
    xemb = sincos(embed_dim=d, pos=xgrid, temperature=temperature, dtype=dtype)  # (N, d)
    emb = np.concatenate([yemb, xemb], axis=1)  # (N, d*2)

    if pad > 0:
        emb = np.pad(emb, ((0, 0), (0, pad)), mode='constant', constant_values=0)  # (N, embed_dim)

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
    dtype=None
):
    """
    N = lateral_sequence_length^2 * (axial_sequence_length or temporal_sequence_length)
    Returns:
        [N + 1, embed_dim] if cls_token=True
        [N, embed_dim] else
    """
    num_dims = 3
    dtype = dtype if dtype is not None else np.float32
    d = int(np.floor(embed_dim / (2*num_dims)) * 2)
    pad = embed_dim - (d * num_dims)

    if axial_sequence_length is not None and temporal_sequence_length is not None:
        raise ValueError("Use `positional_encoding_4d` if you have both axial and temporal sequence_length")

    xgrid = np.arange(lateral_x_sequence_length, dtype=dtype)
    ygrid = np.arange(lateral_y_sequence_length, dtype=dtype)

    if axial_sequence_length is not None:
        zgrid = np.arange(axial_sequence_length, dtype=dtype)
    else:
        zgrid = np.arange(temporal_sequence_length, dtype=dtype)

    zgrid, ygrid, xgrid = np.meshgrid(zgrid, ygrid, xgrid, indexing='ij')

    zemb = sincos(embed_dim=d, pos=zgrid, temperature=temperature, dtype=dtype)  # (N, d)
    yemb = sincos(embed_dim=d, pos=ygrid, temperature=temperature, dtype=dtype)  # (N, d)
    xemb = sincos(embed_dim=d, pos=xgrid, temperature=temperature, dtype=dtype)  # (N, d)
    emb = np.concatenate([zemb, yemb, xemb], axis=1)  # (N, d*3)

    if pad > 0:
        emb = np.pad(emb, ((0, 0), (0, pad)), mode='constant', constant_values=0)  # (N, embed_dim)

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
    dtype=None
):
    """
    N = lateral_sequence_length^2 * axial_sequence_length * temporal_sequence_length
    Returns:
        [N + 1, embed_dim] if cls_token=True
        [N, embed_dim] else
    """
    num_dims = 4
    dtype = dtype if dtype is not None else np.float32
    d = int(np.floor(embed_dim / (2*num_dims)) * 2)
    pad = embed_dim - (d * num_dims)

    xgrid = np.arange(lateral_x_sequence_length, dtype=dtype)
    ygrid = np.arange(lateral_y_sequence_length, dtype=dtype)
    zgrid = np.arange(axial_sequence_length, dtype=dtype)
    tgrid = np.arange(temporal_sequence_length, dtype=dtype)
    tgrid, zgrid, ygrid, xgrid = np.meshgrid(tgrid, zgrid, ygrid, xgrid, indexing='ij')

    temb = sincos(embed_dim=d, pos=tgrid, temperature=temperature, dtype=dtype)  # (N, d)
    zemb = sincos(embed_dim=d, pos=zgrid, temperature=temperature, dtype=dtype)  # (N, d)
    yemb = sincos(embed_dim=d, pos=ygrid, temperature=temperature, dtype=dtype)  # (N, d)
    xemb = sincos(embed_dim=d, pos=xgrid, temperature=temperature, dtype=dtype)  # (N, d)
    emb = np.concatenate([temb, zemb, yemb, xemb], axis=1)  # (N, d*4)

    if pad > 0:
        emb = np.pad(emb, ((0, 0), (0, pad)), mode='constant', constant_values=0)  # (N, embed_dim)

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
            input_format=input_fmt,
            patch_shape=patch_shape
        )

        self.embed_dim = embed_dim
        self.channels = channels
        self.cls_token = cls_token
        assert not self.cls_token, "CLS token not yet supported for PosEmbedding"
        self.interpolate = interpolate

        self.num_patches, self.token_shape = patch_embeddings.calc_num_patches(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            patch_shape=patch_shape
        )

        num_patches_pos_embed = self.num_patches + 1 if self.cls_token else self.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches_pos_embed, self.embed_dim),
            requires_grad=False
        )
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
            if L0 == L1: return pe
            # (B,L,C) -> (B,C,L)
            pe = pe.reshape(B, L0, C).permute(0, 2, 1)
            pe = F.interpolate(pe, size=L1, mode='linear', align_corners=False)
            return pe.permute(0, 2, 1).reshape(B, L1, C)

        def resize_2d(pe, Z0, Y0, Z1, Y1):
            if (Z0, Y0) == (Z1, Y1): return pe
            # (B,H,W,C) -> (B,C,H,W)
            pe = pe.reshape(B, Z0, Y0, C).permute(0, 3, 1, 2)             
            pe = F.interpolate(pe, size=(Z1, Y1), mode='bilinear', align_corners=False)
            return pe.permute(0, 2, 3, 1).reshape(B, Z1 * Y1, C)

        def resize_3d(pe, Z0, Y0, X0, Z1, Y1, X1):
            if (Z0, Y0, X0) == (Z1, Y1, X1): return pe
            # (B,D,H,W,C) -> (B,C,D,H,W)
            pe = pe.reshape(B, Z0, Y0, X0, C).permute(0, 4, 1, 2, 3)
            pe = F.interpolate(pe, size=(Z1, Y1, X1), mode='trilinear', align_corners=False)
            return pe.permute(0, 2, 3, 4, 1).reshape(B, Z1 * Y1 * X1, C)

        def resize_T_ZYX_separable(pe, T0, Z0, Y0, X0, T1, Z1, Y1, X1):
            # 3D over ZYX per T, then 1D over T
            if (T0, Z0, Y0, X0) == (T1, Z1, Y1, X1): return pe
            # (B, T0, Z0, Y0, X0, C)
            pe = pe.reshape(B, T0, Z0, Y0, X0, C)
            # 3D over ZYX per T -> fold T into batch
            pe = pe.permute(0, 1, 5, 2, 3, 4).reshape(B * T0, C, Z0, Y0, X0)
            pe = F.interpolate(pe, size=(Z1, Y1, X1), mode='trilinear', align_corners=False)
            pe = pe.reshape(B, T0, C, Z1, Y1, X1)
            # 1D over T -> fold (C, Z1, Y1, X1) into channels and interpolate length T
            pe = pe.permute(0, 3, 4, 5, 2, 1).reshape(B, C * Z1 * Y1 * X1, T0)     # (B, C*Z*Y*X, T)
            pe = F.interpolate(pe, size=T1, mode='linear', align_corners=False)
            pe = pe.reshape(B, Z1, Y1, X1, C, T1).permute(0, 5, 1, 2, 3, 4)        # (B, T1, Z1, Y1, X1, C)
            return pe.reshape(B, T1 * Z1 * Y1 * X1, C)
        
        def resize_T_YX_separable(pe, T0, Y0, X0, T1, Y1, X1):
            # 2D over YX per T, then 1D over T
            if (T0, Y0, X0) == (T1, Y1, X1): return pe
            # (B, T0, Y0, X0, C)
            pe = pe.reshape(B, T0, Y0, X0, C)
            # 2D over YX per T -> fold T into batch
            pe = pe.permute(0, 1, 4, 2, 3).reshape(B * T0, C, Y0, X0)
            pe = F.interpolate(pe, size=(Y1, X1), mode='bilinear', align_corners=False)
            pe = pe.reshape(B, T0, C, Y1, X1)
            # 1D over T -> fold (C, Y1, X1) into channels and interpolate length T
            pe = pe.permute(0, 3, 4, 2, 1).reshape(B, C * Y1 * X1, T0)
            pe = F.interpolate(pe, size=T1, mode='linear', align_corners=False)
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
        out = torch.gather(pos_table, 
                            dim=1,
                            index=idx.unsqueeze(-1).expand(-1, -1, D))
        return out

    @property
    def table(self) -> torch.Tensor:
        # (L_full, D) without cls
        return (self.pos_embed[:, 1:, :]
                if self.cls_token else
                self.pos_embed)

    def forward(self, x: Optional[torch.Tensor] = None,
                patches_used: Optional[torch.Tensor] = None) -> torch.Tensor:
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