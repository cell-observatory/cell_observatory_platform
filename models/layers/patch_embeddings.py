import logging
import sys
from typing import Optional

import torch
import torch.nn as nn

from cell_observatory_platform.training.helpers import get_patch_sizes

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calc_num_patches(
    input_fmt="TZYXC",
    input_shape=(16, 128, 128, 128, 2),
    patch_shape: tuple = (4, 16, 16, 16),
):
    temporal_patch_size, axial_patch_size, lateral_patch_size = get_patch_sizes(
        input_format=input_fmt, patch_shape=patch_shape
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


def patchify(inputs, input_fmt, temporal_patch_size, axial_patch_size, lateral_patch_size, channels, reshape=True):

    if "T" not in input_fmt:
        patch_shape = (axial_patch_size, lateral_patch_size, lateral_patch_size, None)
    else:
        patch_shape = (temporal_patch_size, axial_patch_size, lateral_patch_size, lateral_patch_size, None)

    num_patches, token_shape = calc_num_patches(
        input_fmt=input_fmt,
        input_shape=inputs.shape[1:],
        patch_shape=patch_shape,
    )

    pixels_per_patch = compute_num_pixels_per_patch(
        channels, temporal_patch_size, axial_patch_size, lateral_patch_size, input_fmt
    )

    b = inputs.shape[0]
    t, z, y, x, c = token_shape

    if input_fmt == "TZYXC":
        if reshape:
            patches = inputs.reshape(
                shape=(
                    b,
                    t,
                    temporal_patch_size,
                    z,
                    axial_patch_size,
                    y,
                    lateral_patch_size,
                    x,
                    lateral_patch_size,
                    channels,
                )
            )
            patches = torch.einsum("btizjykxvc->btzyxijkvc", patches)
        else:
            patches = (
                inputs.unfold(1, temporal_patch_size, temporal_patch_size)
                .unfold(2, axial_patch_size, axial_patch_size)
                .unfold(3, lateral_patch_size, lateral_patch_size)
                .unfold(4, lateral_patch_size, lateral_patch_size)
            )
    elif input_fmt == "ZYXC":
        if reshape:
            patches = inputs.reshape(
                shape=(
                    b,
                    z,
                    axial_patch_size,
                    y,
                    lateral_patch_size,
                    x,
                    lateral_patch_size,
                    channels,
                )
            )
            patches = torch.einsum("bzjykxvc->bzyxjkvc", patches)
        else:
            patches = (
                inputs.unfold(1, axial_patch_size, axial_patch_size)
                .unfold(2, lateral_patch_size, lateral_patch_size)
                .unfold(3, lateral_patch_size, lateral_patch_size)
            )

    elif input_fmt == "TYXC":
        if reshape:
            patches = inputs.reshape(
                shape=(
                    b,
                    t,
                    temporal_patch_size,
                    y,
                    lateral_patch_size,
                    x,
                    lateral_patch_size,
                    channels,
                )
            )
            patches = torch.einsum("btiykxvc->btyxikvc", patches)
        else:
            patches = (
                inputs.unfold(1, temporal_patch_size, temporal_patch_size)
                .unfold(2, lateral_patch_size, lateral_patch_size)
                .unfold(3, lateral_patch_size, lateral_patch_size)
            )

    elif input_fmt == "YXC":
        if reshape:
            patches = inputs.reshape(
                shape=(
                    b,
                    y,
                    lateral_patch_size,
                    x,
                    lateral_patch_size,
                    channels,
                )
            )
            patches = torch.einsum("bykxvc->byxkvc", patches)
        else:
            patches = inputs.unfold(1, lateral_patch_size, lateral_patch_size).unfold(
                2, lateral_patch_size, lateral_patch_size
            )

    elif input_fmt == "XC":
        if reshape:
            patches = inputs.reshape(
                shape=(
                    b,
                    x,
                    lateral_patch_size,
                    channels,
                )
            )
        else:
            patches = inputs.unfold(1, lateral_patch_size, lateral_patch_size)
    else:
        raise NotImplementedError

    # NOTE: if tensor is already in the specified memory format,
    #       contiguous returns the tensor
    patches = patches.contiguous().view(b, num_patches, pixels_per_patch)
    return patches


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
        self.input_shape = input_shape
        self.input_fmt = input_fmt

        self.temporal_patch_size, self.axial_patch_size, self.lateral_patch_size = get_patch_sizes(
            input_format=input_fmt, patch_shape=patch_shape
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

    def _compute_num_pixels_per_patch(self):
        pixels_per_patch = self.channels
        pixels_per_patch *= self.temporal_patch_size if self.temporal_patch_size is not None else 1
        pixels_per_patch *= self.axial_patch_size if self.axial_patch_size is not None else 1
        pixels_per_patch *= self.lateral_patch_size**2 if self.input_fmt is not "XC" else self.lateral_patch_size
        return pixels_per_patch

    def patchify(self, inputs, reshape=True):
        b = inputs.shape[0]
        t, z, y, x, c = self.token_shape

        if self.input_fmt == "TZYXC":
            if reshape:
                patches = inputs.reshape(
                    shape=(
                        b,
                        t,
                        self.temporal_patch_size,
                        z,
                        self.axial_patch_size,
                        y,
                        self.lateral_patch_size,
                        x,
                        self.lateral_patch_size,
                        self.channels,
                    )
                )
                patches = torch.einsum("btizjykxvc->btzyxijkvc", patches)
            else:
                patches = (
                    inputs.unfold(1, self.temporal_patch_size, self.temporal_patch_size)
                    .unfold(2, self.axial_patch_size, self.axial_patch_size)
                    .unfold(3, self.lateral_patch_size, self.lateral_patch_size)
                    .unfold(4, self.lateral_patch_size, self.lateral_patch_size)
                )
        elif self.input_fmt == "ZYXC":
            if reshape:
                patches = inputs.reshape(
                    shape=(
                        b,
                        z,
                        self.axial_patch_size,
                        y,
                        self.lateral_patch_size,
                        x,
                        self.lateral_patch_size,
                        self.channels,
                    )
                )
                patches = torch.einsum("bzjykxvc->bzyxjkvc", patches)
            else:
                patches = (
                    inputs.unfold(1, self.axial_patch_size, self.axial_patch_size)
                    .unfold(2, self.lateral_patch_size, self.lateral_patch_size)
                    .unfold(3, self.lateral_patch_size, self.lateral_patch_size)
                )

        elif self.input_fmt == "TYXC":
            if reshape:
                patches = inputs.reshape(
                    shape=(
                        b,
                        t,
                        self.temporal_patch_size,
                        y,
                        self.lateral_patch_size,
                        x,
                        self.lateral_patch_size,
                        self.channels,
                    )
                )
                patches = torch.einsum("btiykxvc->btyxikvc", patches)
            else:
                patches = (
                    inputs.unfold(1, self.temporal_patch_size, self.temporal_patch_size)
                    .unfold(2, self.lateral_patch_size, self.lateral_patch_size)
                    .unfold(3, self.lateral_patch_size, self.lateral_patch_size)
                )

        elif self.input_fmt == "YXC":
            if reshape:
                patches = inputs.reshape(
                    shape=(
                        b,
                        y,
                        self.lateral_patch_size,
                        x,
                        self.lateral_patch_size,
                        self.channels,
                    )
                )
                patches = torch.einsum("bykxvc->byxkvc", patches)
            else:
                patches = inputs.unfold(1, self.lateral_patch_size, self.lateral_patch_size).unfold(
                    2, self.lateral_patch_size, self.lateral_patch_size
                )
        elif self.input_fmt == "XC":
            if reshape:
                patches = inputs.reshape(
                    shape=(
                        b,
                        x,
                        self.lateral_patch_size,
                        self.channels,
                    )
                )
            else:
                patches = inputs.unfold(1, self.lateral_patch_size, self.lateral_patch_size)
        else:
            raise NotImplementedError

        # NOTE: if tensor is already in the specified memory format,
        #       contiguous returns the tensor
        patches = patches.contiguous().view(b, self.num_patches, self.pixels_per_patch)
        return patches

    # @torch.no_grad()
    def unpatchify(self, patches: torch.Tensor, out_channels: Optional[int]) -> torch.Tensor:
        b = patches.shape[0]

        t, z, y, x, c = self.token_shape
        if out_channels is not None:
            c = out_channels
        Ti = self.temporal_patch_size if self.temporal_patch_size is not None else 1
        Zi = self.axial_patch_size if self.axial_patch_size is not None else 1
        Li = self.lateral_patch_size

        if self.input_fmt == "TZYXC":
            # Forward patchify did:
            # reshape -> (b, t, Ti, z, Zi, y, Li, x, Li, c)
            # einsum "btizjykxvc->btzyxijkvc"
            # view -> (b, num_patches, pixels_per_patch)
            tensor = patches.view(b, t, z, y, x, Ti, Zi, Li, Li, c)
            tensor = torch.einsum("btzyxijkvc->btizjykxvc", tensor)
            tensor = tensor.reshape(b, t * Ti, z * Zi, y * Li, x * Li, c)
            return tensor.contiguous()

        elif self.input_fmt == "ZYXC":
            # Forward patchify did:
            # reshape -> (b, z, Zi, y, Li, x, Li, c)
            # einsum "bzjykxvc->bzyxjkvc" (j=Zi, k=Li, v=Li)
            tensor = patches.view(b, z, y, x, Zi, Li, Li, c)
            tensor = torch.einsum("bzyxjkvc->bzjykxvc", tensor)
            tensor = tensor.reshape(b, z * Zi, y * Li, x * Li, c)
            return tensor.contiguous()

        elif self.input_fmt == "TYXC":
            # Forward patchify did:
            # reshape -> (b, t, Ti, y, Li, x, Li, c)
            # einsum "btiykxvc->btyxikvc"
            tensor = patches.view(b, t, y, x, Ti, Li, Li, c)
            tensor = torch.einsum("btyxikvc->btiykxvc", tensor)
            tensor = tensor.reshape(b, t * Ti, y * Li, x * Li, c)
            return tensor.contiguous()

        elif self.input_fmt == "YXC":
            # Forward patchify did:
            # reshape -> (b, y, Li, x, Li, c)
            # einsum "bykxvc->byxkvc"
            tensor = patches.view(b, y, x, Li, Li, c)
            tensor = torch.einsum("byxkvc->bykxvc", tensor)
            tensor = tensor.reshape(b, y * Li, x * Li, c)
            return tensor.contiguous()

        elif self.input_fmt == "XC":
            # Forward patchify did:
            # reshape -> (b, x, Li, c)
            tensor = patches.view(b, x, Li, c)
            tensor = tensor.reshape(b, x * Li, c)
            return tensor.contiguous()

        else:
            raise NotImplementedError(f"input_fmt not supported: {self.input_fmt}")

    def forward(self, inputs, return_patches=False):
        patches = self.patchify(inputs)
        projections = self.proj(patches)

        if return_patches:
            return projections, patches
        else:
            return projections
