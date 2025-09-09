import sys
import logging
from typing import Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import patch_embeddings

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
    lateral_sequence_length,
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

    xgrid = np.arange(lateral_sequence_length, dtype=dtype)
    ygrid = np.arange(lateral_sequence_length, dtype=dtype)
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
    lateral_sequence_length,
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

    xgrid = np.arange(lateral_sequence_length, dtype=dtype)
    ygrid = np.arange(lateral_sequence_length, dtype=dtype)

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
    lateral_sequence_length,
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

    xgrid = np.arange(lateral_sequence_length, dtype=dtype)
    ygrid = np.arange(lateral_sequence_length, dtype=dtype)
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
        input_shape=(1, 16, 128, 128, 128, 1),
        lateral_patch_size=16,
        axial_patch_size=1,
        temporal_patch_size=1,
        embed_dim=768,
        channels=1,
        cls_token=False,
        interpolate=False,
    ):   
        super().__init__()

        self.input_fmt = input_fmt
        self.input_shape = input_shape

        self.axial_patch_size = axial_patch_size
        self.lateral_patch_size = lateral_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.embed_dim = embed_dim
        self.channels = channels
        self.cls_token = cls_token
        self.interpolate = interpolate

        self.num_patches, self.token_shape = patch_embeddings.calc_num_patches(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            lateral_patch_size=self.lateral_patch_size,
            axial_patch_size=self.axial_patch_size,
            temporal_patch_size=self.temporal_patch_size,
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
                temporal_sequence_length=self.input_shape[1] // self.temporal_patch_size,
                axial_sequence_length=self.input_shape[2] // self.axial_patch_size,
                lateral_sequence_length=self.input_shape[3] // self.lateral_patch_size,
                cls_token=self.cls_token,
            )

        elif self.input_fmt == "ZYXC":
            sincos = positional_encoding_3d(
                embed_dim=self.embed_dim,
                temporal_sequence_length=None,
                axial_sequence_length=self.input_shape[1] // self.axial_patch_size,
                lateral_sequence_length=self.input_shape[2] // self.lateral_patch_size,
                cls_token=self.cls_token,
            )

        elif self.input_fmt == "TYXC":
            sincos = positional_encoding_3d(
                embed_dim=self.embed_dim,
                axial_sequence_length=None,
                temporal_sequence_length=self.input_shape[1] // self.temporal_patch_size,
                lateral_sequence_length=self.input_shape[2] // self.lateral_patch_size,
                cls_token=self.cls_token,
            )

        elif self.input_fmt == "YXC":
            sincos = positional_encoding_2d(
                embed_dim=self.embed_dim,
                lateral_sequence_length=self.input_shape[1] // self.lateral_patch_size,
                cls_token=self.cls_token,
            )

        elif self.input_fmt == "XC":
            sincos = positional_encoding_1d(
                embed_dim=self.embed_dim,
                sequence_length=self.input_shape[1] // self.lateral_patch_size,
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
        
        T0, Z0, Y0, X0, C0 = self.token_shape[0:4]

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

    def forward(self, x):
        if self.interpolate:
            return self.interpolate_positional_encoding(x, self.pos_embed)
        else:
            return self.pos_embed


# --- ---- Rotary Embedding Helpers --- ---  

# based on: https://github.com/naver-ai/rope-vit and extended to 3D and 4D
# NOTE: the divisibility assertions may be too strong, we might have to relax
#       and pad or similar

def generate_frequency_spectrum(dim: int, 
                                num_heads: int, 
                                theta: float = 10.0, 
                                random_rotation_per_head: bool = True,
                                input_fmt: str = "TZYXC"
):
    if input_fmt == "XYC":
        assert dim % 4 == 0, "head_dim must be divisible by 4 for 2D ROPE."
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
            fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
            fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
            freqs_x.append(fx)
            freqs_y.append(fy)
        freqs_x = torch.stack(freqs_x, dim=0)
        freqs_y = torch.stack(freqs_y, dim=0)
        # freqs: (2, num_heads, dim//4 * 2)
        freqs = torch.stack([freqs_x, freqs_y], dim=0)
    
    elif input_fmt == "TYXC" or input_fmt == "ZYXC":
        # the below follows the logic as above but generalized to 3D
        assert dim % 6 == 0, "head_dim must be divisible by 6 for 3D ROPE."
        J = dim // 6

        base = torch.arange(0, dim, 6)[: (dim // 6)].float() / dim
        mag = theta ** (-base)

        # freqs: (3, num_heads, dim//6*3)
        freqs = torch.empty(3, num_heads, J*3)

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
            # blocks: [3, 3, J] -> freqs: [3, num_heads, dim//6*3]
            freqs[:, h, :] = torch.cat(blocks, dim=-1)
    
    elif input_fmt == "TZYXC":
        # the below follows the logic as above but generalized to 4D
        assert dim % 8 == 0, "head_dim must be divisible by 8 for 4D ROPE."
        J = dim // 8

        base = torch.arange(0, dim, 8)[: (dim // 8)].float() / dim
        mag = theta ** (-base)

        # freqs: (4, num_heads, dim//8*4)
        freqs = torch.empty(4, num_heads, J*4)

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
    
    return freqs

def generate_grid_indices(
    end_x: int,
    end_y: int,
    end_z: Optional[int] = None,
    end_t: Optional[int] = None,
    input_fmt: str = "TZYXC",
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
    x = (idx % X)
    y = ((idx // X) % Y)
    z = None
    t = None

    if need_Z:
        z = ((idx // (X * Y)) % Z)
    if need_T:
        t = (idx // (X * Y * (Z if need_Z else 1)))

    return (t, z, y, x)

def compute_mixed_cis(freqs: torch.Tensor,
                      num_heads: int,
                      t_x: torch.Tensor,
                      t_y: torch.Tensor,
                      t_z: Optional[torch.Tensor] = None,
                      t_t: Optional[torch.Tensor] = None,
                      input_fmt: str = "TZYXC"
):
    N = t_x.shape[0]
    # no float 16 for this range
    with torch.cuda.autocast(enabled=False):
        if input_fmt == "YXC":
            # [N, 1] @ [num_heads,1,(dim/4)*2] -> [num_heads,N,dim/4*2] -> [N,num_heads,dim/4*2]
            freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
            freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
            # compute e^(i*theta) where theta for each k is given by [mag_k(x_k*cosϕ + y_k*sinϕ),mag_k(-x_k*sinϕ+y_k*cosϕ)]
            # i.e. e^(i*theta) where theta is <(x,y), (basis_vectors for head h)>
            freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
        
        elif input_fmt == "TYXC":
            assert t_t is not None, "t_t must be provided for TYXC format"
            freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
            freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
            freqs_t = (t_t.unsqueeze(-1) @ freqs[2].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
            freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y + freqs_t)

        elif input_fmt == "ZYXC":
            assert t_z is not None, "t_z must be provided for ZYXC format"
            freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
            freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
            freqs_z = (t_z.unsqueeze(-1) @ freqs[2].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
            freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y + freqs_z)

        elif input_fmt == "TZYXC":
            assert t_t is not None and t_z is not None, "t_t and t_z must be provided for TZYXC format"
            freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
            freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
            freqs_z = (t_z.unsqueeze(-1) @ freqs[2].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
            freqs_t = (t_t.unsqueeze(-1) @ freqs[3].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
            freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y + freqs_z + freqs_t)

    return freqs_cis

def compute_axial_cis(dim: int, 
                      end_x: int, 
                      end_y: int,
                      end_z: int, 
                      end_t: int,
                      input_fmt: str = "TZYXC", 
                      theta: float = 100.0
):
    # NOTE: in the paper they define: R(n,2t)=e{iθ_{t}​p^{n}_{x}​,R(n,2t+1)=eθ_{t}​p^{n}_{y}
    #       however in the reference code the assignment per embedding dimension is:
    #       [x-slot, x-slot, ..., y-slot, y-slot, ...] i.e. the specific assignment of 
    #       x,y positions to dimensions is not interleaved but blockwise

    if input_fmt == "YXC":
        assert dim % 4 == 0, "head_dim must be divisible by 4 for 2D frame duplication."
        mag = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        
        t_y, t_x = generate_grid_indices(end_x=end_x, end_y=end_y, input_fmt=input_fmt)

        freqs_x = torch.outer(t_x, mag)
        freqs_y = torch.outer(t_y, mag)

    elif input_fmt == "TYXC":
        assert dim % 6 == 0, "head_dim must be divisible by 6 for 3D frame duplication."
        base = torch.arange(0, dim, 6)[: (dim // 6)].float() / dim
        mag = theta ** (-base)
        
        t_t, t_y, t_x = generate_grid_indices(end_x=end_x, end_y=end_y, end_t=end_t, input_fmt=input_fmt)
        
        freqs_x = torch.outer(t_x, mag)
        freqs_y = torch.outer(t_y, mag)
        freqs_t = torch.outer(t_t, mag)

    elif input_fmt == "ZYXC":
        assert dim % 6 == 0, "head_dim must be divisible by 6 for 3D frame duplication."
        base = torch.arange(0, dim, 6)[: (dim // 6)].float() / dim
        mag = theta ** (-base)
        
        t_z, t_y, t_x = generate_grid_indices(end_x=end_x, end_y=end_y, end_z=end_z, input_fmt=input_fmt)
        
        freqs_x = torch.outer(t_x, mag)
        freqs_y = torch.outer(t_y, mag)
        freqs_z = torch.outer(t_z, mag)

    elif input_fmt == "TZYXC":
        assert dim % 8 == 0, "head_dim must be divisible by 8 for 4D frame duplication."
        base = torch.arange(0, dim, 8)[: (dim // 8)].float() / dim
        mag = theta ** (-base)
        
        t_t, t_z, t_y, t_x = generate_grid_indices(end_x=end_x, 
                                                   end_y=end_y, 
                                                   end_z=end_z, 
                                                   end_t=end_t, 
                                                   input_fmt=input_fmt)
        
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
        shape = [d if i >= x.ndim-2 else 1 for i, d in enumerate(x.shape)]
    # freqs_cis: (N, H, J) branch
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        # freq_cis reshaped to (1, H, N, J) since x: [B, H, N, J]
        shape = [d if i >= x.ndim-3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # xq[:-1]: [B, H, N] => xq reshape: [B, H, N, J, 2] for J=H/2 
    # thus xq_: [B, H, N, J] complex and similar for xk
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # if [N, J] -> reshaped to [1, 1, N, J]
    # if [H, N, J] -> reshaped to [1, H, N, J]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # xq_ * freqs_cis: elementwise complex mult -> [B, H, N, J]
    # then view_as_real -> [B, H, N, J, 2] -> flatten last two dims -> [B, H, N, J]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


# --- ---- --- ---