"""
Adapted from:
https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/memory_encoder.py
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_observatory_platform.models.layers.norm import LayerNorm3d
from cell_observatory_platform.models.layers.utils import DropPath, get_clones


class MaskDownSampler(nn.Module):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(
        self,
        input_fmt: str,
        embed_dim=256,
        kernel_size=4,
        stride=4,
        padding=0,
        total_stride=16,
        activation=nn.GELU,
    ):
        super().__init__()

        self.input_fmt = input_fmt
        
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride, "Total stride must be a power of stride."
        
        if self.input_fmt == "ZYXC":
            self.encoder = nn.Sequential()
            mask_in_chans, mask_out_chans = 1, 1
            for _ in range(num_layers):
                mask_out_chans = mask_in_chans * (stride**2)
                self.encoder.append(
                    nn.Conv3d(
                        mask_in_chans,
                        mask_out_chans,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )
                self.encoder.append(LayerNorm3d(mask_out_chans))
                self.encoder.append(activation())
                mask_in_chans = mask_out_chans

            self.encoder.append(nn.Conv3d(mask_out_chans, embed_dim, kernel_size=1))
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

    def forward(self, x):
        return self.encoder(x)


# Lightly adapted from ConvNext (https://github.com/facebookresearch/ConvNeXt)
class CXBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dim,
        input_fmt: str,
        kernel_size=7,
        padding=3,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    ):
        super().__init__()
        self.input_fmt = input_fmt
        
        if self.input_fmt == "ZYXC":
            self.dwconv = nn.Conv3d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=padding,
                groups=dim if use_dwconv else 1,
            )  # depthwise conv
            self.norm = LayerNorm3d(dim, eps=1e-6)
            self.pwconv1 = nn.Linear(
                dim, 4 * dim
            )  # pointwise/1x1 convs, implemented with linear layers
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(4 * dim, dim)
            self.gamma = (
                nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
                if layer_scale_init_value > 0
                else None
            )
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        if self.input_fmt == "ZYXC":
            x = x.permute(0, 2, 3, 4, 1)
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        if self.input_fmt == "ZYXC":
            x = x.permute(0, 4, 1, 2, 3)
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")
        x = input + self.drop_path(x)
        return x


class Fuser(nn.Module):
    def __init__(
        self, 
        layer, 
        num_layers,
        input_fmt: str,
        dim=None, 
        input_projection=False
    ):
        super().__init__()
        self.input_fmt = input_fmt
        self.proj = nn.Identity()
        self.layers = get_clones(layer, num_layers)

        if input_projection:
            assert dim is not None, "dim must be provided if input_projection is True."
            if self.input_fmt == "ZYXC":
                self.proj = nn.Conv3d(dim, dim, kernel_size=1)
            else:
                raise NotImplementedError(f"Input format {input_fmt} not supported yet.")

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(nn.Module):
    def __init__(
        self,
        input_fmt: str,
        out_dim: int,
        position_encoding: nn.Module,
        in_dim: int = 256,
        # MaskDownSampler args
        mask_embed_dim: int = 256,
        mask_kernel_size: int = 4,
        mask_stride: int = 4,
        mask_padding: int = 0,
        mask_total_stride: int = 16,
        mask_activation: type = nn.GELU,
        # Fuser args (layer is CXBlock)
        fuser_num_layers: int = 1,
        fuser_input_projection: bool = False,
        fuser_dim: int | None = None,
        # CXBlock args for fuser layer
        fuser_layer_kernel_size: int = 7,
        fuser_layer_padding: int = 3,
        fuser_layer_drop_path: float = 0.0,
        fuser_layer_scale_init_value: float = 1e-6,
        fuser_layer_use_dwconv: bool = True,
    ):
        super().__init__()

        self.input_fmt = input_fmt
        fuser_dim = in_dim if fuser_dim is None else fuser_dim

        self.mask_downsampler = MaskDownSampler(
            input_fmt=input_fmt,
            embed_dim=mask_embed_dim,
            kernel_size=mask_kernel_size,
            stride=mask_stride,
            padding=mask_padding,
            total_stride=mask_total_stride,
            activation=mask_activation,
        )

        if self.input_fmt == "ZYXC":
            self.pix_feat_proj = nn.Conv3d(in_dim, in_dim, kernel_size=1)
        else:
            raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

        fuser_layer = CXBlock(
            dim=fuser_dim,
            input_fmt=input_fmt,
            kernel_size=fuser_layer_kernel_size,
            padding=fuser_layer_padding,
            drop_path=fuser_layer_drop_path,
            layer_scale_init_value=fuser_layer_scale_init_value,
            use_dwconv=fuser_layer_use_dwconv,
        )
        self.fuser = Fuser(
            layer=fuser_layer,
            num_layers=fuser_num_layers,
            input_fmt=input_fmt,
            dim=fuser_dim if fuser_input_projection else None,
            input_projection=fuser_input_projection,
        )

        self.position_encoding = position_encoding

        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            if self.input_fmt == "ZYXC":
                self.out_proj = nn.Conv3d(in_dim, out_dim, kernel_size=1)
            else:
                raise NotImplementedError(f"Input format {self.input_fmt} not supported yet.")

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        ## Fuse pix_feats and downsampled masks
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}