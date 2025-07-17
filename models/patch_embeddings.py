import logging
import sys

import torch
import torch.nn as nn

from models import positional_encoding

logging.basicConfig(
	stream=sys.stdout,
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calc_num_patches(
    input_fmt="TZYXC",
    input_shape=(1, 16, 64, 64, 64, 1),
    lateral_patch_size=1,
    axial_patch_size=1,
    temporal_patch_size=1,
):
    if input_fmt == "TZYXC" or input_fmt == "TZYX":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"
        assert axial_patch_size != None, "axial_patch_size cannot be None"
        assert temporal_patch_size != None, "temporal_patch_size cannot be None"

        t = input_shape[1] // temporal_patch_size
        z = input_shape[2] // axial_patch_size
        y = input_shape[3] // lateral_patch_size
        x = input_shape[4] // lateral_patch_size
        c = input_shape[-1] if input_fmt == "TZYXC" else None
        num_patches = t * z * y * x

    elif input_fmt == "ZYXC" or input_fmt == "ZYX":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"
        assert axial_patch_size != None, "axial_patch_size cannot be None"

        t = None
        z = input_shape[1] // axial_patch_size
        y = input_shape[2] // lateral_patch_size
        x = input_shape[3] // lateral_patch_size
        c = input_shape[-1] if input_fmt == "ZYXC" else None
        num_patches = z * y * x

    elif input_fmt == "TYXC" or input_fmt == "TYX":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"
        assert temporal_patch_size != None, "temporal_patch_size cannot be None"

        z = None
        t = input_shape[1] // temporal_patch_size
        y = input_shape[2] // lateral_patch_size
        x = input_shape[3] // lateral_patch_size
        c = input_shape[-1] if input_fmt == "YXC" else None
        num_patches = t * y * x

    elif input_fmt == "YXC" or input_fmt == "YX":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"

        t, z = None, None
        y = input_shape[1] // lateral_patch_size
        x = input_shape[2] // lateral_patch_size
        c = input_shape[-1] if input_fmt == "YXC" else None
        num_patches = y * x

    elif input_fmt == "XC" or input_fmt == "X":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"

        t, z, y = None, None, None
        x = input_shape[1] // lateral_patch_size
        c = input_shape[-1] if input_fmt == "X" else None
        num_patches = x
    else:
        raise NotImplementedError

    return num_patches, (t, z, y, x, c)


def ZYXC_to(x: torch.Tensor, fmt: str):
    if fmt == "CZYX":
        x = torch.permute(x, (0, 4, 1, 2, 3))
    elif fmt == "LC":
        x = torch.flatten(x, 2)  # -> (B, C, L)
        x = torch.transpose(x, 1, 2)
    elif fmt == "CL":
        x = torch.flatten(x, 2)
    return x


class ConvPatchEmbedding(nn.Module):
    def __init__(
        self,
        input_shape=(1, 6, 64, 64, 1),
        lateral_patch_size=16,
        axial_patch_size=1,
        temporal_patch_size=1,
        embed_dim=768,
    ):
        super().__init__()
        self.input_shape = input_shape # (B, Z, Y, X, C)
        self.lateral_patch_size = lateral_patch_size
        self.axial_patch_size = axial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim
        self.img_size = input_shape[-2]
        self.in_chans = input_shape[-1]
        self.num_frames = input_shape[1]

        self.proj = nn.Conv3d(
            in_channels=self.in_chans,
            out_channels=self.embed_dim,
            kernel_size=(axial_patch_size, lateral_patch_size, lateral_patch_size),
            stride=(axial_patch_size, lateral_patch_size, lateral_patch_size),
        )

    def forward(self, x):
        x = ZYXC_to(x, fmt="CZYX") # (B, Z, Y, X, C) -> (B, C, Z, Y, X)
        x = self.proj(x)
        x = ZYXC_to(x, fmt="LC")
        return x # (B, L, C)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        input_fmt="ZYXC",
        input_shape=(1, 6, 64, 64, 1),
        lateral_patch_size=16,
        axial_patch_size=None,
        temporal_patch_size=None,
        embed_dim=768,
        channels=1,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.input_fmt = input_fmt

        self.lateral_patch_size = lateral_patch_size
        self.axial_patch_size = axial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim
        self.channels = channels

        self.num_patches, self.token_shape = calc_num_patches(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            lateral_patch_size=self.lateral_patch_size,
            axial_patch_size=self.axial_patch_size,
            temporal_patch_size=self.temporal_patch_size,
        )

        self.pixels_per_patch = self.channels
        self.pixels_per_patch *= self.temporal_patch_size if self.temporal_patch_size is not None else 1
        self.pixels_per_patch *= self.axial_patch_size if self.axial_patch_size is not None else 1
        self.pixels_per_patch *= self.lateral_patch_size ** 2

        self.proj = nn.Linear(in_features=self.pixels_per_patch, out_features=self.embed_dim)


    def patchify(self, inputs, reshape=True):
        # logger.info(f"Input shape: {x.shape}")
        # logger.info(f"Token shape: {self.token_shape}")

        b = inputs.shape[0]
        t, z, y, x, c = self.token_shape

        if self.input_fmt == "TZYXC" or self.input_fmt == "TZYX":
            if reshape:
                patches = inputs.reshape(shape=(
                    b,
                    t, self.temporal_patch_size,
                    z, self.axial_patch_size,
                    y, self.lateral_patch_size,
                    x, self.lateral_patch_size,
                    self.channels,
                ))
                patches = torch.einsum("btizjykxvc->btzyxijkvc", patches)
            else:
                patches = inputs.unfold(1, self.temporal_patch_size, self.temporal_patch_size) \
                    .unfold(2, self.axial_patch_size, self.axial_patch_size) \
                    .unfold(3, self.lateral_patch_size, self.lateral_patch_size) \
                    .unfold(4, self.lateral_patch_size, self.lateral_patch_size) \

        elif self.input_fmt == "ZYXC" or self.input_fmt == "ZYX":
            if reshape:
                patches = inputs.reshape(shape=(
                    b,
                    z, self.axial_patch_size,
                    y, self.lateral_patch_size,
                    x, self.lateral_patch_size,
                    self.channels,
                ))
                patches = torch.einsum("bzjykxvc->bzyxjkvc", patches)
            else:
                patches = inputs.unfold(1, self.axial_patch_size, self.axial_patch_size) \
                    .unfold(2, self.lateral_patch_size, self.lateral_patch_size) \
                    .unfold(3, self.lateral_patch_size, self.lateral_patch_size)

        elif self.input_fmt == "TYXC" or self.input_fmt == "TYX":
            if reshape:
                patches = inputs.reshape(shape=(
                    b,
                    t, self.temporal_patch_size,
                    y, self.lateral_patch_size,
                    x, self.lateral_patch_size,
                    self.channels,
                ))
                patches = torch.einsum("btiykxvc->btyxikvc", patches)
            else:
                patches = inputs.unfold(1, self.temporal_patch_size, self.temporal_patch_size) \
                    .unfold(2, self.lateral_patch_size, self.lateral_patch_size) \
                    .unfold(3, self.lateral_patch_size, self.lateral_patch_size)

        elif self.input_fmt == "YXC" or self.input_fmt == "YX":
            if reshape:
                patches = inputs.reshape(shape=(
                    b,
                    y, self.lateral_patch_size,
                    x, self.lateral_patch_size,
                    self.channels,
                ))
                patches = torch.einsum("bykxvc->byxkvc", patches)
            else:
                patches = inputs.unfold(1, self.lateral_patch_size, self.lateral_patch_size) \
                    .unfold(2, self.lateral_patch_size, self.lateral_patch_size) \

        elif self.input_fmt == "XC" or self.input_fmt == "X":
            if reshape:
                patches = inputs.reshape(shape=(
                    b,
                    x, self.lateral_patch_size,
                    self.channels,
                ))
            else:
                patches = inputs.unfold(1, self.lateral_patch_size, self.lateral_patch_size)
        else:
            raise NotImplementedError

        # logger.info(f"Patches: {inputs.shape}")
        patches = patches.contiguous().view(b, self.num_patches, self.pixels_per_patch)
        # logger.info(f"Pixels: {inputs.shape}")
        return patches

    def forward(self, inputs, return_patches=False):
        patches = self.patchify(inputs)
        projections = self.proj(patches)

        if return_patches:
            return projections, patches
        else:
            return projections



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
        self.input_shape = input_shape
        self.input_fmt = input_fmt

        self.lateral_patch_size = lateral_patch_size
        self.axial_patch_size = axial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim
        self.channels = channels
        self.cls_token = cls_token
        self.interpolate = interpolate

        self.num_patches, self.token_shape = calc_num_patches(
            input_fmt=self.input_fmt,
            input_shape=self.input_shape,
            lateral_patch_size=self.lateral_patch_size,
            axial_patch_size=self.axial_patch_size,
            temporal_patch_size=self.temporal_patch_size,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.embed_dim),
            requires_grad=False
        )
        self._init_pos_embed(self.pos_embed.data)

    def _init_pos_embed(self, pos_embed):
        if self.input_fmt == "TZYXC" or self.input_fmt == "TZYX":
            sincos = positional_encoding.positional_encoding_4d(
                embed_dim=self.embed_dim,
                temporal_sequence_length=self.input_shape[1] // self.temporal_patch_size,
                axial_sequence_length=self.input_shape[2] // self.axial_patch_size,
                lateral_sequence_length=self.input_shape[3] // self.lateral_patch_size,
                cls_token=self.cls_token,
            )
        elif self.input_fmt == "ZYXC" or self.input_fmt == "ZYX":
            sincos = positional_encoding.positional_encoding_3d(
                embed_dim=self.embed_dim,
                temporal_sequence_length=None,
                axial_sequence_length=self.input_shape[1] // self.axial_patch_size,
                lateral_sequence_length=self.input_shape[2] // self.lateral_patch_size,
                cls_token=self.cls_token,
            )

        elif self.input_fmt == "TYXC" or self.input_fmt == "TYX":
            sincos = positional_encoding.positional_encoding_3d(
                embed_dim=self.embed_dim,
                axial_sequence_length=None,
                temporal_sequence_length=self.input_shape[1] // self.axial_patch_size,
                lateral_sequence_length=self.input_shape[2] // self.lateral_patch_size,
                cls_token=self.cls_token,
            )

        elif self.input_fmt == "YXC" or self.input_fmt == "YX":
            sincos = positional_encoding.positional_encoding_2d(
                embed_dim=self.embed_dim,
                lateral_sequence_length=self.input_shape[1] // self.lateral_patch_size,
                cls_token=self.cls_token,
            )

        elif self.input_fmt == "XC" or self.input_fmt == "X":
            sincos = positional_encoding.positional_encoding_1d(
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
        if self.input_fmt == "TZYXC" or self.input_fmt == "TZYX":
            temporal_sequence_length = self.input_shape[1] // self.temporal_patch_size
            axial_sequence_length = self.input_shape[2] // self.axial_patch_size
            lateral_sequence_length = self.input_shape[3] // self.lateral_patch_size

            T = x.shape[1] // self.temporal_patch_size
            Z = x.shape[2] // self.axial_patch_size
            Y = x.shape[3] // self.lateral_patch_size
            X = x.shape[4] // self.lateral_patch_size

            if T != temporal_sequence_length or Z != axial_sequence_length or Y != lateral_sequence_length or X != lateral_sequence_length:
                scale_factor = (
                    T / temporal_sequence_length,
                    Z / axial_sequence_length,
                    Y / lateral_sequence_length,
                    X / lateral_sequence_length
                )

                logger.info(f"Interpolating position embedding: {x.shape} [{scale_factor=}]")

                pos_embed = nn.functional.interpolate(
                    pos_embed.reshape(
                        1,
                        temporal_sequence_length,
                        axial_sequence_length,
                        lateral_sequence_length,
                        lateral_sequence_length,
                        self.embed_dim
                    ),
                    scale_factor=scale_factor,
                    mode='trilinear'
                )
                pos_embed = pos_embed.view(1, -1, self.embed_dim)

        elif self.input_fmt == "ZYXC" or self.input_fmt == "ZYX":
            axial_sequence_length = self.input_shape[1] // self.axial_patch_size
            lateral_sequence_length = self.input_shape[2] // self.lateral_patch_size

            Z = x.shape[1] // self.axial_patch_size
            Y = x.shape[2] // self.lateral_patch_size
            X = x.shape[3] // self.lateral_patch_size

            if Z != axial_sequence_length or Y != lateral_sequence_length or X != lateral_sequence_length:
                scale_factor = (
                    Z / axial_sequence_length,
                    Y / lateral_sequence_length,
                    X / lateral_sequence_length
                )

                logger.info(f"Interpolating position embedding: {x.shape} [{scale_factor=}]")

                pos_embed = nn.functional.interpolate(
                    pos_embed.reshape(
                        1,
                        axial_sequence_length,
                        lateral_sequence_length,
                        lateral_sequence_length,
                        self.embed_dim
                    ),
                    scale_factor=scale_factor,
                    mode='trilinear'
                )
                pos_embed = pos_embed.view(1, -1, self.embed_dim)

        elif self.input_fmt == "TYXC" or self.input_fmt == "TYX":
            temporal_sequence_length = self.input_shape[1] // self.temporal_patch_size
            lateral_sequence_length = self.input_shape[2] // self.lateral_patch_size

            T = x.shape[1] // self.temporal_patch_size
            Y = x.shape[2] // self.lateral_patch_size
            X = x.shape[3] // self.lateral_patch_size

            if T != temporal_sequence_length or Y != lateral_sequence_length or X != lateral_sequence_length:
                scale_factor = (
                    T / temporal_sequence_length,
                    Y / lateral_sequence_length,
                    X / lateral_sequence_length
                )

                logger.info(f"Interpolating position embedding: {x.shape} [{scale_factor=}]")

                pos_embed = nn.functional.interpolate(
                    pos_embed.reshape(
                        1,
                        temporal_sequence_length,
                        lateral_sequence_length,
                        lateral_sequence_length,
                        self.embed_dim
                    ),
                    scale_factor=scale_factor,
                    mode='trilinear'
                )

                pos_embed = pos_embed.view(1, -1, self.embed_dim)

        elif self.input_fmt == "YXC" or self.input_fmt == "YX":
            lateral_sequence_length = self.input_shape[1] // self.lateral_patch_size

            Y = x.shape[1] // self.lateral_patch_size
            X = x.shape[2] // self.lateral_patch_size

            if Y != lateral_sequence_length or X != lateral_sequence_length:
                scale_factor = (
                    Y / lateral_sequence_length,
                    X / lateral_sequence_length
                )

                logger.info(f"Interpolating position embedding: {x.shape} [{scale_factor=}]")

                pos_embed = nn.functional.interpolate(
                    pos_embed.reshape(
                        1,
                        lateral_sequence_length,
                        lateral_sequence_length,
                        self.embed_dim
                    ),
                    scale_factor=scale_factor,
                    mode='trilinear'
                )
                pos_embed = pos_embed.view(1, -1, self.embed_dim)

        elif self.input_fmt == "XC" or self.input_fmt == "X":
            lateral_sequence_length = self.input_shape[1] // self.lateral_patch_size

            X = x.shape[1] // self.lateral_patch_size

            if X != lateral_sequence_length:
                scale_factor = X / lateral_sequence_length

                logger.info(f"Interpolating position embedding: {x.shape} [{scale_factor=}]")

                pos_embed = nn.functional.interpolate(
                    pos_embed.reshape(
                        1,
                        lateral_sequence_length,
                        self.embed_dim
                    ),
                    scale_factor=scale_factor,
                    mode='trilinear'
                )
                pos_embed = pos_embed.view(1, -1, self.embed_dim)

        else:
            raise NotImplementedError

        return pos_embed

    def forward(self, x):
        if self.interpolate:
            return self.interpolate_positional_encoding(x, self.pos_embed)
        else:
            return self.pos_embed