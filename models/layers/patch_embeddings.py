import sys
import logging
from typing import Optional, Literal

import torch
import torch.nn as nn

from cell_observatory_platform.training.helpers import get_patch_sizes

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calc_num_patches(
    input_fmt="TZYXC",
    input_shape=(16, 128, 128, 128, 2),
    patch_shape: tuple = (4, 16, 16, 16),
):
    temporal_patch_size, axial_patch_size, lateral_patch_size = get_patch_sizes(
        input_format=input_fmt,
        patch_shape=patch_shape
    )

    if input_fmt == "TZYXC":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"
        assert axial_patch_size != None, "axial_patch_size cannot be None"
        assert temporal_patch_size != None, "temporal_patch_size cannot be None"

        t = input_shape[0] // temporal_patch_size
        z = input_shape[1] // axial_patch_size
        y = input_shape[2] // lateral_patch_size
        x = input_shape[3] // lateral_patch_size
        c = input_shape[-1]
        num_patches = t * z * y * x

    elif input_fmt == "ZYXC":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"
        assert axial_patch_size != None, "axial_patch_size cannot be None"

        t = None
        z = input_shape[0] // axial_patch_size
        y = input_shape[1] // lateral_patch_size
        x = input_shape[2] // lateral_patch_size
        c = input_shape[-1]
        num_patches = z * y * x

    elif input_fmt == "TYXC":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"
        assert temporal_patch_size != None, "temporal_patch_size cannot be None"

        z = None
        t = input_shape[0] // temporal_patch_size
        y = input_shape[1] // lateral_patch_size
        x = input_shape[2] // lateral_patch_size
        c = input_shape[-1]
        num_patches = t * y * x

    elif input_fmt == "YXC":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"

        t, z = None, None
        y = input_shape[0] // lateral_patch_size
        x = input_shape[1] // lateral_patch_size
        c = input_shape[-1]
        num_patches = y * x

    elif input_fmt == "XC":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"

        t, z, y = None, None, None
        x = input_shape[0] // lateral_patch_size
        c = input_shape[-1]
        num_patches = x
    
    else:
        raise NotImplementedError("input_fmt not supported: {}".format(input_fmt))

    return num_patches, (t, z, y, x, c)


def compute_num_pixels_per_patch(channels, temporal_patch_size, axial_patch_size, lateral_patch_size, input_fmt):
    pixels_per_patch = channels
    pixels_per_patch *= temporal_patch_size if temporal_patch_size is not None else 1
    pixels_per_patch *= axial_patch_size if axial_patch_size is not None else 1
    pixels_per_patch *= lateral_patch_size**2 if input_fmt is not "XC" else lateral_patch_size
    return pixels_per_patch


# NOTE: timm has optional norm layer after patch embedding
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        input_fmt="ZYXC",
        input_shape=(16, 128, 128, 128, 2),
        patch_shape: tuple = (4, 16, 16, 16),
        embed_dim=768,
        channels=1,
    ):
        super().__init__()

        self.input_fmt = input_fmt
        self.input_shape = input_shape
        self.patch_shape = patch_shape
        
        self.temporal_patch_size, self.axial_patch_size, self.lateral_patch_size = get_patch_sizes(
            input_format=input_fmt,
            patch_shape=patch_shape
        )
        
        self.embed_dim = embed_dim
        self.channels = channels

        if self.input_fmt not in ["TZYXC", "ZYXC"] and self.axial_patch_size is not None:
            raise ValueError("axial_patch_size must not be specified for inputs without Z dimension.")
        elif self.input_fmt not in ["TYXC", "TZYXC"] and self.temporal_patch_size is not None:
            raise ValueError("temporal_patch_size must not be specified for inputs without time dimension.")

        self.num_patches, self.token_shape = calc_num_patches(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            patch_shape=patch_shape,
        )
        self.pixels_per_patch = self._compute_num_pixels_per_patch()

        self.proj = nn.Linear(in_features=self.pixels_per_patch, out_features=self.embed_dim)

    @staticmethod
    def compute_num_pixels_per_patch(channels, 
                                     temporal_patch_size, 
                                     axial_patch_size, 
                                     lateral_patch_size, 
                                     input_format
    ):
        pixels_per_patch = channels
        pixels_per_patch *= temporal_patch_size if temporal_patch_size is not None else 1
        pixels_per_patch *= axial_patch_size if axial_patch_size is not None else 1
        if input_format != "XC":
            pixels_per_patch *= lateral_patch_size ** 2
        else:
            pixels_per_patch *= lateral_patch_size
        return pixels_per_patch

    def _compute_num_pixels_per_patch(self):
        pixels_per_patch = self.channels
        pixels_per_patch *= self.temporal_patch_size if self.temporal_patch_size is not None else 1
        pixels_per_patch *= self.axial_patch_size if self.axial_patch_size is not None else 1
        if self.input_fmt != "XC":
            pixels_per_patch *= self.lateral_patch_size ** 2
        else:
            pixels_per_patch *= self.lateral_patch_size
        return pixels_per_patch
    
    def _patchify(self, inputs, reshape=True, shape=None):
        # support variable size input, e.g. multiresolution inputs
        if shape is not None:
            num_patches, token_shape = calc_num_patches(
                input_fmt=self.input_fmt,
                input_shape=tuple(shape),
                patch_shape=self.patch_shape,
            )
        else:
            num_patches, token_shape = self.num_patches, self.token_shape
   
        return self.patchify(
            inputs,
            reshape=reshape,
            temporal_patch_size=self.temporal_patch_size,
            axial_patch_size=self.axial_patch_size,
            lateral_patch_size=self.lateral_patch_size,
            token_shape=token_shape,
            channels=self.channels,
            num_patches=num_patches,
            pixels_per_patch=self.pixels_per_patch,
            input_format=self.input_fmt,
        )

    @staticmethod
    def patchify(inputs,  
                 input_format, 
                 temporal_patch_size,
                 axial_patch_size,
                 lateral_patch_size,
                 token_shape, 
                 channels,
                 num_patches,
                 pixels_per_patch,
                 reshape=True
    ):
        b = inputs.shape[0]
        t, z, y, x, c = token_shape

        if input_format == "TZYXC":
            if reshape:
                patches = inputs.reshape(shape=(
                    b,
                    t, temporal_patch_size,
                    z, axial_patch_size,
                    y, lateral_patch_size,
                    x, lateral_patch_size,
                    channels,
                ))
                patches = torch.einsum("btizjykxvc->btzyxijkvc", patches)
            else:
                patches = inputs.unfold(1, temporal_patch_size, temporal_patch_size) \
                    .unfold(2, axial_patch_size, axial_patch_size) \
                    .unfold(3, lateral_patch_size, lateral_patch_size) \
                    .unfold(4, lateral_patch_size, lateral_patch_size) \

        elif input_format == "ZYXC":
            if reshape:
                patches = inputs.reshape(shape=(
                    b,
                    z, axial_patch_size,
                    y, lateral_patch_size,
                    x, lateral_patch_size,
                    channels,
                ))
                patches = torch.einsum("bzjykxvc->bzyxjkvc", patches)
            else:
                patches = inputs.unfold(1, axial_patch_size, axial_patch_size) \
                    .unfold(2, lateral_patch_size, lateral_patch_size) \
                    .unfold(3, lateral_patch_size, lateral_patch_size)

        elif input_format == "TYXC":
            if reshape:
                patches = inputs.reshape(shape=(
                    b,
                    t, temporal_patch_size,
                    y, lateral_patch_size,
                    x, lateral_patch_size,
                    channels,
                ))
                patches = torch.einsum("btiykxvc->btyxikvc", patches)
            else:
                patches = inputs.unfold(1, temporal_patch_size, temporal_patch_size) \
                    .unfold(2, lateral_patch_size, lateral_patch_size) \
                    .unfold(3, lateral_patch_size, lateral_patch_size)

        elif input_format == "YXC":
            if reshape:
                patches = inputs.reshape(shape=(
                    b,
                    y, lateral_patch_size,
                    x, lateral_patch_size,
                    channels,
                ))
                patches = torch.einsum("bykxvc->byxkvc", patches)
            else:
                patches = inputs.unfold(1, lateral_patch_size, lateral_patch_size) \
                    .unfold(2, lateral_patch_size, lateral_patch_size)

        elif input_format == "XC":
            if reshape:
                patches = inputs.reshape(shape=(
                    b,
                    x, lateral_patch_size,
                    channels,
                ))
            else:
                patches = inputs.unfold(1, lateral_patch_size, lateral_patch_size)
        else:
            raise NotImplementedError

        # NOTE: if tensor is already in the specified memory format, 
        #       contiguous returns the tensor
        patches = patches.contiguous().view(b, num_patches, pixels_per_patch)
        return patches
    
    def _unpatchify(self, patches: torch.Tensor, out_channels: Optional[int]) -> torch.Tensor:
        return self.unpatchify(patches, 
                               out_channels=out_channels,
                               temporal_patch_size=self.temporal_patch_size,
                               axial_patch_size=self.axial_patch_size,
                               lateral_patch_size=self.lateral_patch_size,
                               token_shape=self.token_shape,
                               input_format=self.input_fmt)

    @staticmethod
    def unpatchify(patches: torch.Tensor, 
                   temporal_patch_size,
                   axial_patch_size,
                   lateral_patch_size,
                   token_shape,
                   input_format,
                   out_channels: Optional[int],
    ) -> torch.Tensor:
        b = patches.shape[0]

        t, z, y, x, c = token_shape
        if out_channels is not None:
            c = out_channels
        Ti = temporal_patch_size if temporal_patch_size is not None else 1
        Zi = axial_patch_size if axial_patch_size is not None else 1
        Li = lateral_patch_size

        if input_format == "TZYXC":
            # Forward patchify did:
            # reshape -> (b, t, Ti, z, Zi, y, Li, x, Li, c)
            # einsum "btizjykxvc->btzyxijkvc"
            # view -> (b, num_patches, pixels_per_patch)
            tensor = patches.view(b, t, z, y, x, Ti, Zi, Li, Li, c)
            tensor = torch.einsum("btzyxijkvc->btizjykxvc", tensor)
            tensor = tensor.reshape(b, t * Ti, z * Zi, y * Li, x * Li, c)
            return tensor.contiguous()

        elif input_format == "ZYXC":
            # Forward patchify did:
            # reshape -> (b, z, Zi, y, Li, x, Li, c)
            # einsum "bzjykxvc->bzyxjkvc" (j=Zi, k=Li, v=Li)
            tensor = patches.view(b, z, y, x, Zi, Li, Li, c)
            tensor = torch.einsum("bzyxjkvc->bzjykxvc", tensor)
            tensor = tensor.reshape(b, z * Zi, y * Li, x * Li, c)
            return tensor.contiguous()

        elif input_format == "TYXC":
            # Forward patchify did:
            # reshape -> (b, t, Ti, y, Li, x, Li, c)
            # einsum "btiykxvc->btyxikvc"
            tensor = patches.view(b, t, y, x, Ti, Li, Li, c)
            tensor = torch.einsum("btyxikvc->btiykxvc", tensor)
            tensor = tensor.reshape(b, t * Ti, y * Li, x * Li, c)
            return tensor.contiguous()

        elif input_format == "YXC":
            # Forward patchify did:
            # reshape -> (b, y, Li, x, Li, c)
            # einsum "bykxvc->byxkvc"
            tensor = patches.view(b, y, x, Li, Li, c)
            tensor = torch.einsum("byxkvc->bykxvc", tensor)
            tensor = tensor.reshape(b, y * Li, x * Li, c)
            return tensor.contiguous()

        elif input_format == "XC":
            # Forward patchify did:
            # reshape -> (b, x, Li, c)
            tensor = patches.view(b, x, Li, c)
            tensor = tensor.reshape(b, x * Li, c)
            return tensor.contiguous()

        else:
            raise NotImplementedError(f"input_fmt not supported: {input_format}")

    def forward(
        self, 
        inputs, 
        return_patches=False, 
        is_patches: bool = False,
        shape=None,
    ):
        if not is_patches:
            patches = self._patchify(inputs, shape=shape)
        else:
            patches = inputs
        
        projections = self.proj(patches)

        if return_patches:
            return projections, patches
        else:
            return projections


# FIXME: not fully supported by models and maskgenerator yet.
class ChannelAdaptivePatchEmbedding(nn.Module):
    """
    Variable-channel patch embedding.
    """

    def __init__(
        self,
        input_fmt: str = "ZYXC",
        patch_shape: tuple = (4, 16, 16, 16),
        embed_dim: int = 768,
        max_channels: int = 32,
        use_channel_embed: bool = True,
        channel_fusion: Literal["concat", "attn_pool"] = "concat",
        attn_pool_num_heads: int = 8,
        attn_drop: float = 0.0,
    ):
        super().__init__()

        self.input_fmt = input_fmt
        self.patch_shape = patch_shape

        self.embed_dim = embed_dim

        self.max_channels = max_channels
        self.use_channel_embed = use_channel_embed
        self.channel_fusion = channel_fusion

        self.temporal_patch_size, self.axial_patch_size, self.lateral_patch_size = get_patch_sizes(
            input_format=input_fmt,
            patch_shape=patch_shape,
        )

        # Compute pixels-per-patch for ONE channel (C stays separate)
        Ti = self.temporal_patch_size if self.temporal_patch_size is not None else 1
        Zi = self.axial_patch_size if self.axial_patch_size is not None else 1
        Li = self.lateral_patch_size
        if Li is None:
            raise ValueError("lateral_patch_size cannot be None")

        P = Ti * Zi * (Li * Li)
        self.pixels_per_patch = int(P)
        self.proj = nn.Linear(self.pixels_per_patch, embed_dim)

        if use_channel_embed:
            self.channel_embed = nn.Embedding(max_channels, embed_dim)
        else:
            self.channel_embed = None

        if channel_fusion == "attn_pool":
            self.pool_ln = nn.LayerNorm(embed_dim)
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=attn_pool_num_heads,
                dropout=attn_drop,
                batch_first=True,
            )
            self.pool_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.pool_ln = None
            self.pool_attn = None
            self.pool_query = None

    def _resolve_channel_ids(
        self, B: int, C: int, device, channel_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Returns [B, C] long ids.
        - None: uses 0..C-1
        - [C]: broadcast to [B,C]
        - [B,C]: use directly
        """
        if channel_ids is None:
            ids = torch.arange(C, device=device, dtype=torch.long)
            ids = ids.unsqueeze(0).expand(B, -1)
        else:
            if channel_ids.ndim == 1:
                if channel_ids.numel() != C:
                    raise ValueError(f"channel_ids has {channel_ids.numel()} but C={C}")
                ids = channel_ids.to(device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            elif channel_ids.ndim == 2:
                if channel_ids.shape != (B, C):
                    raise ValueError(f"channel_ids must be [B,C]={B,C}, got {tuple(channel_ids.shape)}")
                ids = channel_ids.to(device=device, dtype=torch.long)
            else:
                raise ValueError(f"channel_ids must be [C] or [B,C], got ndim={channel_ids.ndim}")

        if ids.max().item() >= self.max_channels or ids.min().item() < 0:
            raise ValueError(
                f"channel_ids out of range for max_channels={self.max_channels}: "
                f"min={int(ids.min())}, max={int(ids.max())}"
            )
        return ids

    def patchify_per_channel(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """
        Returns:
          patches: [B, N, C, P]
          token_shape: (t, z, y, x, c) where t/z may be None depending on fmt
        """
        B = x.shape[0]
        C = x.shape[-1]

        Ti = self.temporal_patch_size
        Zi = self.axial_patch_size
        Li = self.lateral_patch_size

        if self.input_fmt == "TZYXC":
            if x.ndim != 6:
                raise ValueError(f"TZYXC expects [B,T,Z,Y,X,C], got {tuple(x.shape)}")
            T, Z, Y, X = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
            if Ti is None or Zi is None or Li is None:
                raise ValueError("TZYXC requires temporal+axial+lateral patch sizes")
            if (T % Ti) or (Z % Zi) or (Y % Li) or (X % Li):
                raise ValueError(f"Input not divisible by patch sizes: {(T,Z,Y,X)} vs {(Ti,Zi,Li)}")

            t = T // Ti
            z = Z // Zi
            y = Y // Li
            x_ = X // Li
            # [B, t, Ti, z, Zi, y, Li, x, Li, C]
            p = x.reshape(B, t, Ti, z, Zi, y, Li, x_, Li, C)
            # -> [B, t, z, y, x, C, Ti, Zi, Li, Li]
            p = p.permute(0, 1, 3, 5, 7, 9, 2, 4, 6, 8)
            N = t * z * y * x_
            P = Ti * Zi * Li * Li
            patches = p.reshape(B, N, C, P)
            token_shape = (t, z, y, x_, C)
            return patches, token_shape

        elif self.input_fmt == "ZYXC":
            if x.ndim != 5:
                raise ValueError(f"ZYXC expects [B,Z,Y,X,C], got {tuple(x.shape)}")
            Z, Y, X = x.shape[1], x.shape[2], x.shape[3]
            if Zi is None or Li is None:
                raise ValueError("ZYXC requires axial+lateral patch sizes")
            if (Z % Zi) or (Y % Li) or (X % Li):
                raise ValueError(f"Input not divisible by patch sizes: {(Z,Y,X)} vs {(Zi,Li)}")

            z = Z // Zi
            y = Y // Li
            x_ = X // Li
            # [B, z, Zi, y, Li, x, Li, C]
            p = x.reshape(B, z, Zi, y, Li, x_, Li, C)
            # -> [B, z, y, x, C, Zi, Li, Li]
            p = p.permute(0, 1, 3, 5, 7, 2, 4, 6)
            N = z * y * x_
            P = Zi * Li * Li
            patches = p.reshape(B, N, C, P)
            token_shape = (None, z, y, x_, C)
            return patches, token_shape

        elif self.input_fmt == "TYXC":
            if x.ndim != 5:
                raise ValueError(f"TYXC expects [B,T,Y,X,C], got {tuple(x.shape)}")
            T, Y, X = x.shape[1], x.shape[2], x.shape[3]
            if Ti is None or Li is None:
                raise ValueError("TYXC requires temporal+lateral patch sizes")
            if (T % Ti) or (Y % Li) or (X % Li):
                raise ValueError(f"Input not divisible by patch sizes: {(T,Y,X)} vs {(Ti,Li)}")

            t = T // Ti
            y = Y // Li
            x_ = X // Li
            # [B, t, Ti, y, Li, x, Li, C]
            p = x.reshape(B, t, Ti, y, Li, x_, Li, C)
            # -> [B, t, y, x, C, Ti, Li, Li]
            p = p.permute(0, 1, 3, 5, 7, 2, 4, 6)
            N = t * y * x_
            P = Ti * Li * Li
            patches = p.reshape(B, N, C, P)
            token_shape = (t, None, y, x_, C)
            return patches, token_shape

        elif self.input_fmt == "YXC":
            if x.ndim != 4:
                raise ValueError(f"YXC expects [B,Y,X,C], got {tuple(x.shape)}")
            Y, X = x.shape[1], x.shape[2]
            if Li is None:
                raise ValueError("YXC requires lateral patch size")
            if (Y % Li) or (X % Li):
                raise ValueError(f"Input not divisible by patch size: {(Y,X)} vs {Li}")

            y = Y // Li
            x_ = X // Li
            # [B, y, Li, x, Li, C]
            p = x.reshape(B, y, Li, x_, Li, C)
            # -> [B, y, x, C, Li, Li]
            p = p.permute(0, 1, 3, 5, 2, 4)
            N = y * x_
            P = Li * Li
            patches = p.reshape(B, N, C, P)
            token_shape = (None, None, y, x_, C)
            return patches, token_shape

        else:
            raise NotImplementedError(f"input_fmt not supported: {self.input_fmt}")

    def forward(
        self,
        inputs: torch.Tensor,
        return_patches: bool = False,
        channel_ids: Optional[torch.Tensor] = None,
    ):
        patches, token_shape = self.patchify_per_channel(inputs)  # [B,N,C,P]
        B, N, C, P = patches.shape

        if P != self.pixels_per_patch:
            raise RuntimeError(f"pixels_per_patch mismatch: got {P}, expected {self.pixels_per_patch}")

        # [B,N,C,D]
        tokens = self.proj(patches)

        # add channel embedding: [B,C,D] -> broadcast over N
        if self.channel_embed is not None:
            ids = self._resolve_channel_ids(B, C, tokens.device, channel_ids)
            ce = self.channel_embed(ids)              # [B,C,D]
            tokens = tokens + ce.unsqueeze(1)         # [B,N,C,D]

        if self.channel_fusion == "concat":
            out = tokens.reshape(B, N * C, self.embed_dim)

        elif self.channel_fusion == "attn_pool":
            if self.pool_attn is None:
                raise RuntimeError("attn_pool not initialized")
            x = self.pool_ln(tokens).reshape(B * N, C, self.embed_dim)   # [B*N,C,D]
            q = self.pool_query.expand(B * N, -1, -1)                    # [B*N,1,D]
            pooled, _ = self.pool_attn(q, x, x, need_weights=False)
            out = pooled.squeeze(1).reshape(B, N, self.embed_dim)        # [B,N,D]

        else:
            raise ValueError(f"Unsupported channel_fusion: {self.channel_fusion}")

        if return_patches:
            return out, patches, token_shape
        return out