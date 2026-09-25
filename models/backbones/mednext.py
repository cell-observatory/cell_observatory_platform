"""
Adapted from:


Roy, S., Koehler, G., Ulrich, C., Baumgartner, M., Petersen, J., Isensee, F., Jaeger, P.F. & Maier-Hein, K. (2023).
MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation.
International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2023.

# ------------------------------------------------------------------------
# MedNeXt
# Copyright (c) 2023 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from omegaconf import DictConfig, OmegaConf
import torch.utils.checkpoint as checkpoint
import logging
import sys

logger = logging.getLogger(__name__)

class MedNeXtBlock(nn.Module):

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        exp_r: int = 4, 
        kernel_size:int = 7, 
        do_res: int | bool = True,
        norm_type: str = 'group',
        n_groups: int | None = None,
        dim: str = '3d',
        grn: bool = True
    ):

        super().__init__()

        self.do_res = do_res

        assert dim in ['2d', '3d'], f"Invalid dimension: {dim}. Must be '2d' or '3d'."
        self.dim = dim
        if self.dim == '2d':
            conv = nn.Conv2d
        elif self.dim == '3d':
            conv = nn.Conv3d
        else:
            raise ValueError(f"Invalid dimension: {dim}. Must be '2d' or '3d'.")
            
        # First convolution layer with DepthWise Convolutions
        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 1,
            padding = kernel_size//2,
            groups = in_channels if n_groups is None else n_groups,
        )

        # Normalization Layer. GroupNorm is used by default.
        if norm_type=='group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels, 
                num_channels=in_channels
                )
        elif norm_type=='layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels, 
                data_format='channels_first'
                )
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}. Must be 'group' or 'layer'.")

        # Second convolution (Expansion) layer with Conv3D 1x1x1
        self.conv2 = conv(
            in_channels = in_channels,
            out_channels = exp_r*in_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )
        
        # GeLU activations
        self.act = nn.GELU()
        
        # Third convolution (Compression) layer with Conv3D 1x1x1
        self.conv3 = conv(
            in_channels = exp_r*in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

        self.grn = grn
        if grn:
            if dim == '3d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1,1), requires_grad=True)
            elif dim == '2d':
                self.grn_beta = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)
                self.grn_gamma = nn.Parameter(torch.zeros(1,exp_r*in_channels,1,1), requires_grad=True)

 
    def forward(self, x, dummy_tensor=None):
        
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn:
            # gamma, beta: learnable affine transform parameters
            # X: input of shape (N,C,H,W,D)
            if self.dim == '3d':
                gx = torch.norm(x1, p=2, dim=(-3, -2, -1), keepdim=True)
            elif self.dim == '2d':
                gx = torch.norm(x1, p=2, dim=(-2, -1), keepdim=True)
            else:
                raise ValueError(f"Invalid dimension: {self.dim}. Must be '2d' or '3d'.")
            nx = gx / (gx.mean(dim=1, keepdim=True)+1e-6)
            x1 = self.grn_gamma * (x1 * nx) + self.grn_beta + x1
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(MedNeXtBlock):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7, 
        do_res: bool | int = False,
        norm_type: str = 'group',
        dim: str = '3d',
        grn: bool = True # 3D Global Response Normalization as seen in MedNeXtV2 paper
    ):

        super().__init__(
            in_channels,
            out_channels,
            exp_r,
            kernel_size,
            do_res=False,   # parent residual incompatible with stride=2; DownBlock uses res_conv
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        else:
            raise ValueError(f"Invalid dimension: {dim}. Must be '2d' or '3d'.")
        self.resample_do_res = do_res
        if do_res:
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
            )

        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size//2,
            groups = in_channels,
        )

    def forward(self, x, dummy_tensor=None):
        
        x1 = super().forward(x)
        
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res

        return x1


class MedNeXtUpBlock(MedNeXtBlock):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7, 
        do_res: bool | int = False, 
        norm_type: str = 'group', 
        dim: str = '3d', 
        grn: bool = True                                 # 3D Global Response Normalization as seen in MedNeXtV2 paper
        ):
        super().__init__(
            in_channels,
            out_channels,
            exp_r,
            kernel_size,
            do_res=False,   # parent residual incompatible with stride=2; UpBlock uses res_conv
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.resample_do_res = do_res
        
        self.dim = dim
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        else:
            raise ValueError(f"Invalid dimension: {dim}. Must be '2d' or '3d'.")
        if do_res:            
            self.res_conv = conv(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
                )

        self.conv1 = conv(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size//2,
            groups = in_channels,
        )


    def forward(self, x: torch.Tensor, dummy_tensor: torch.Tensor | None = None) -> torch.Tensor:
        
        x1 = super().forward(x)
        # Asymmetry but necessary to match shape
        
        if self.dim == '2d':
            x1 = torch.nn.functional.pad(x1, (1,0,1,0))
        elif self.dim == '3d':
            x1 = torch.nn.functional.pad(x1, (1,0,1,0,1,0))
        
        if self.resample_do_res:
            res = self.res_conv(x)
            if self.dim == '2d':
                res = torch.nn.functional.pad(res, (1,0,1,0))
            elif self.dim == '3d':
                res = torch.nn.functional.pad(res, (1,0,1,0,1,0))
            x1 = x1 + res

        return x1


class OutBlock(nn.Module):

    def __init__(self, in_channels: int, n_classes: int, dim: str):
        super().__init__()
        
        if dim == '2d':
            conv = nn.ConvTranspose2d
        elif dim == '3d':
            conv = nn.ConvTranspose3d
        else:
            raise ValueError(f"Invalid dimension: {dim}. Must be '2d' or '3d'.")
        self.conv_out = conv(in_channels, n_classes, kernel_size=1)
    
    def forward(self, x, dummy_tensor=None): 
        return self.conv_out(x)


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))        # beta
        self.bias = nn.Parameter(torch.zeros(normalized_shape))         # gamma
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x, dummy_tensor=False):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x



class MedNeXt(nn.Module):

    def __init__(self, 
        input_shape: List[int],
        input_format: str,
        n_channels: int,
        n_classes: int, 
        exp_r: int | list[int] = 4,                          # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                    # Ofcourse can test kernel_size
        enc_kernel_size: int | None = None,
        dec_kernel_size: int | None = None,
        deep_supervision: bool = False,           # Can be used to test deep supervision
        do_res: bool = False,                     # Can be used to individually test residual connection
        do_res_up_down: bool = False,             # Additional 'res' connection on up and down convs
        checkpoint_style: bool | None = None,            # Either inside block or outside block
        block_counts: list = [2,2,2,2,2,2,2,2,2], # Can be used to test staging ratio: 
                                                  # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nnUNet
        norm_type: str = 'group',
        dim: str = '3d',                                # 2d or 3d
        grn: bool = True                                 # 3D Global Response Normalization as seen in MedNeXtV2 paper
    ):

        super().__init__()
        if input_format != "ZYXC":
            raise NotImplementedError(f"Input format {input_format} not supported yet.")

        in_channels = input_shape[input_format.index("C")]

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        assert dim in ['2d', '3d'], f"Invalid dimension: {dim}. Must be '2d' or '3d'."
        
        # Per-half kernel sizes: kernel_size (default 7) is only a FALLBACK for
        # whichever of enc/dec_kernel_size is unset. The old unconditional
        # `if kernel_size is not None` overwrote both (kernel_size defaults to 7,
        # so the branch always fired), silently ignoring any configured
        # enc_kernel_size / dec_kernel_size.
        if enc_kernel_size is None:
            enc_kernel_size = kernel_size
        if dec_kernel_size is None:
            dec_kernel_size = kernel_size
        if enc_kernel_size is None or dec_kernel_size is None:
            raise ValueError(
                "MedNeXt: kernel_size (fallback) or both enc_kernel_size and "
                "dec_kernel_size must be set."
            )

        if dim == '2d':
            conv = nn.Conv2d
        elif dim == '3d':
            conv = nn.Conv3d
        else:
            raise ValueError(f"Invalid dimension: {dim}. Must be '2d' or '3d'.")
        
        self.stem = conv(in_channels, n_channels, kernel_size=1)
        
        if isinstance(exp_r, int):
            exp_r = [exp_r] * len(block_counts)
        else:
            assert isinstance(exp_r, list), f"exp_r must be a list of integers. Got {type(exp_r)}."
            assert len(exp_r) == len(block_counts), f"Length of exp_r must be equal to length of block_counts. Got {len(exp_r)} and {len(block_counts)}."
        
        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                ) 
            for i in range(block_counts[0])]
        ) 

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2*n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn,
        )
    
        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[1])]
        )

        self.down_1 = MedNeXtDownBlock(
            in_channels=2*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[2])]
        )

        self.down_2 = MedNeXtDownBlock(
            in_channels=4*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )
        
        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )            
            for i in range(block_counts[3])]
        )
        
        self.down_3 = MedNeXtDownBlock(
            in_channels=8*n_channels,
            out_channels=16*n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*16,
                out_channels=n_channels*16,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[4])]
        )

        self.up_3 = MedNeXtUpBlock(
            in_channels=16*n_channels,
            out_channels=8*n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[5])]
        )

        self.up_2 = MedNeXtUpBlock(
            in_channels=8*n_channels,
            out_channels=4*n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[6])]
        )

        self.up_1 = MedNeXtUpBlock(
            in_channels=4*n_channels,
            out_channels=2*n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[7])]
        )

        self.up_0 = MedNeXtUpBlock(
            in_channels=2*n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
                )
            for i in range(block_counts[8])]
        )

        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        # Used to fix PyTorch checkpointing bug (buffer not param to avoid DeepSpeed ZeRO
        # reducing None gradients; see https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9)
        self.register_buffer("dummy_tensor", torch.tensor([1.0])) # TODO: Consider adding use_reentrant=False to checkpoint()

        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels*2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels*4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels*8, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=n_channels*16, n_classes=n_classes, dim=dim)

        self.block_counts = block_counts


    def iterative_checkpoint(self, sequential_block, x) -> torch.Tensor:
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor)
        return x


    def forward_features(self, x) -> Dict[str, torch.Tensor]:
        # TODO: Split this into separate encoder / decoder modules
        
        x = self.stem(x)
        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor)

            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.do_ds:
                x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor)
            dec_x = x_res_3 + x_up_3 
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor)
            del x_res_3, x_up_3

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor)
            dec_x = x_res_2 + x_up_2 
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor)
            del x_res_2, x_up_2

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor)
            dec_x = x_res_1 + x_up_1 
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor)
            del x_res_1, x_up_1

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor)
            dec_x = x_res_0 + x_up_0 
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            del x_res_0, x_up_0, dec_x

            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor)

        else:
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            dec_x = x_res_3 + x_up_3 
            x = self.dec_block_3(dec_x)

            if self.do_ds:
                x_ds_3 = self.out_3(x)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            dec_x = x_res_2 + x_up_2 
            x = self.dec_block_2(dec_x)
            if self.do_ds:
                x_ds_2 = self.out_2(x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            dec_x = x_res_1 + x_up_1 
            x = self.dec_block_1(dec_x)
            if self.do_ds:
                x_ds_1 = self.out_1(x)
            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            dec_x = x_res_0 + x_up_0 
            del x_res_0, x_up_0
            x = self.dec_block_0(dec_x)
            del dec_x

            x = self.out_0(x)

        if self.do_ds:
            return {"pred_masks": x, "auxiliary_outputs": [x_ds_1, x_ds_2, x_ds_3, x_ds_4]}
        else: 
            return {"pred_masks": x}

    def forward(self, data_sample: dict) -> Dict[str, torch.Tensor]:
        x, meta = data_sample['data_tensor'], data_sample['metainfo']

        x = torch.permute(x, (0, -1, 1, 2, 3))  # (B, Z, Y, X, C) -> (B, C, Z, Y, X)
        features = self.forward_features(x)
        # TODO: Split off the decoder into a separate module
        # NOTE: We don't permute channels back because the loss expects (B, N_classes, spatial)
        return features


from cell_observatory_platform.utils.registry import REGISTRY
from cell_observatory_platform.utils.config import build_kwargs as _build_kwargs


@REGISTRY.register("backbone", "mednext")
def BUILD(backbone_wrapper_args: dict, adapter_args: Optional[dict] = None) -> nn.Module:
    if adapter_args is not None:
        raise NotImplementedError("Adapter is not supported for MedNeXt.")
    else:
        if "BUILD" in backbone_wrapper_args:
            backbone_wrapper_args = dict(backbone_wrapper_args)
            backbone_wrapper_args.pop("BUILD")
        if isinstance(backbone_wrapper_args, DictConfig):
            resolved = OmegaConf.to_container(backbone_wrapper_args, resolve=True)
            if isinstance(resolved, dict):
                backbone_wrapper_args = resolved
        logger.info(f"Building MedNeXt with args: {backbone_wrapper_args}")
        return MedNeXt(**_build_kwargs(backbone_wrapper_args))