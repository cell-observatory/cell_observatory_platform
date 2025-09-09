import sys
import logging

import torch
import torch.nn as nn

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
    if input_fmt == "TZYXC":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"
        assert axial_patch_size != None, "axial_patch_size cannot be None"
        assert temporal_patch_size != None, "temporal_patch_size cannot be None"

        t = input_shape[1] // temporal_patch_size
        z = input_shape[2] // axial_patch_size
        y = input_shape[3] // lateral_patch_size
        x = input_shape[4] // lateral_patch_size
        c = input_shape[-1]
        num_patches = t * z * y * x

    elif input_fmt == "ZYXC":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"
        assert axial_patch_size != None, "axial_patch_size cannot be None"

        t = None
        z = input_shape[1] // axial_patch_size
        y = input_shape[2] // lateral_patch_size
        x = input_shape[3] // lateral_patch_size
        c = input_shape[-1]
        num_patches = z * y * x

    elif input_fmt == "TYXC":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"
        assert temporal_patch_size != None, "temporal_patch_size cannot be None"

        z = None
        t = input_shape[1] // temporal_patch_size
        y = input_shape[2] // lateral_patch_size
        x = input_shape[3] // lateral_patch_size
        c = input_shape[-1]
        num_patches = t * y * x

    elif input_fmt == "YXC":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"

        t, z = None, None
        y = input_shape[1] // lateral_patch_size
        x = input_shape[2] // lateral_patch_size
        c = input_shape[-1]
        num_patches = y * x

    elif input_fmt == "XC":
        assert lateral_patch_size != None, "lateral_patch_size cannot be None"

        t, z, y = None, None, None
        x = input_shape[1] // lateral_patch_size
        c = input_shape[-1]
        num_patches = x
    
    else:
        raise NotImplementedError("input_fmt not supported: {}".format(input_fmt))

    return num_patches, (t, z, y, x, c)


# NOTE: timm has optional norm layer after patch embedding
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

        self.axial_patch_size = axial_patch_size
        self.lateral_patch_size = lateral_patch_size
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
        self.pixels_per_patch = self._compute_num_pixels_per_patch()

        self.proj = nn.Linear(in_features=self.pixels_per_patch, out_features=self.embed_dim)

    def _compute_num_pixels_per_patch(self):
        pixels_per_patch = self.channels
        pixels_per_patch *= self.temporal_patch_size if self.temporal_patch_size is not None else 1
        pixels_per_patch *= self.axial_patch_size if self.axial_patch_size is not None else 1
        pixels_per_patch *= self.lateral_patch_size ** 2 if self.input_fmt is not "XC" else self.lateral_patch_size
        return pixels_per_patch

    def patchify(self, inputs, reshape=True):
        b = inputs.shape[0]
        t, z, y, x, c = self.token_shape

        if self.input_fmt == "TZYXC":
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

        elif self.input_fmt == "ZYXC":
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

        elif self.input_fmt == "TYXC":
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

        elif self.input_fmt == "YXC":
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

        elif self.input_fmt == "XC":
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

        # NOTE: if tensor is already in the specified memory format, 
        #       contiguous returns the tensor
        patches = patches.contiguous().view(b, self.num_patches, self.pixels_per_patch)
        return patches

    def forward(self, inputs, return_patches=False):
        patches = self.patchify(inputs)
        projections = self.proj(patches)

        if return_patches:
            return projections, patches
        else:
            return projections