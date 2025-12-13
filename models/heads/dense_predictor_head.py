"""
Adapted to 3D and 3D+T from:
https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/util/blocks.py
https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py#L118
"""

import functools
import inspect
from typing import Any, Mapping

import torch.nn as nn
import torch.nn.functional as F

from cell_observatory_platform.models.layers.patch_embeddings import PatchEmbedding, calc_num_patches
from cell_observatory_platform.training.helpers import get_patch_sizes


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation, bn, dim, strategy="axial"):
        super().__init__()

        self.bn = bn
        self.groups = 1
        self.dim = dim
        self.strategy = strategy

        if self.dim == 3 and self.strategy == "axial":
            self.spatial_conv1 = nn.Conv3d(
                features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
            )
            self.spatial_conv2 = nn.Conv3d(
                features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
            )

            # TODO: think about how best to handle BN for 3D+T
            if self.bn:
                self.bn1 = nn.BatchNorm3d(features)
                self.bn2 = nn.BatchNorm3d(features)

            self.activation = activation
            self.skip_add = nn.quantized.FloatFunctional()

        else:
            raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

    def forward(self, x):
        out = self.activation(x)

        if self.dim == 3 and self.strategy == "axial":
            out = self.spatial_conv1(out)
        else:
            raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

        if self.bn:
            if self.dim == 3 and self.strategy == "axial":
                out = self.bn1(out)
            else:
                raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

        out = self.activation(out)

        if self.dim == 3 and self.strategy == "axial":
            out = self.spatial_conv2(out)

        else:
            raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

        if self.bn:
            if self.dim == 3 and self.strategy == "axial":
                out = self.bn2(out)
            else:
                raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features,
        dim,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        groups=1,
        strategy="axial",
    ):
        super(FeatureFusionBlock, self).__init__()

        self.strategy = strategy

        self.dim = dim
        self.groups = groups

        self.deconv = deconv
        self.align_corners = align_corners

        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2

        if self.dim == 3 and self.strategy == "axial":
            self.out_spatial_conv = nn.Conv3d(
                features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1
            )

            if self.dim == 3 and self.strategy == "axial":
                self.out_temporal_conv = nn.Identity()
            else:
                raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

            self.resConvUnit_1 = ResidualConvUnit(features, activation, bn, dim)
            self.resConvUnit_2 = ResidualConvUnit(features, activation, bn, dim)

            self.skip_add = nn.quantized.FloatFunctional()

        else:
            raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

    def forward(self, *xs, size=None, scale_factor=None):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConvUnit_1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConvUnit_2(output)

        if self.dim == 3 and self.strategy == "axial":
            output = F.interpolate(
                output,
                size=size if size is not None else None,
                scale_factor=scale_factor if scale_factor is not None else None,
                mode="trilinear",
                align_corners=self.align_corners,
            )

        else:
            raise NotImplementedError("Only 3D input is currently supported.")

        if self.dim == 3 and self.strategy == "axial":
            output = self.out_spatial_conv(output)
            output = self.out_temporal_conv(output)
        else:

            raise NotImplementedError("Only 3D input is currently supported.")

        return output


def _get_fusion_block(features, use_bn, dim, strategy="axial"):
    return FeatureFusionBlock(
        features=features,
        dim=dim,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        strategy=strategy,
    )


def _get_FPNAdapter(in_features, out_features, dim, groups=1, expand=False, strategy="axial"):
    adapter = nn.Module()

    if dim == 3 and strategy == "axial":
        out_features_1, out_features_2, out_features_3 = out_features, out_features, out_features
        if len(in_features) >= 4:
            out_features_4 = out_features
        if expand:
            out_features_1 = out_features
            out_features_2 = out_features * 2
            out_features_3 = out_features * 4
            if len(in_features) >= 4:
                out_features_4 = out_features * 8

        adapter.spatial_conv1 = nn.Conv3d(
            in_features[0], out_features_1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
        adapter.spatial_conv2 = nn.Conv3d(
            in_features[1], out_features_2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
        adapter.spatial_conv3 = nn.Conv3d(
            in_features[2], out_features_3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
        if len(in_features) >= 4:
            adapter.spatial_conv4 = nn.Conv3d(
                in_features[3], out_features_4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
            )
    else:
        raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

    if dim == 3 and strategy == "axial":
        adapter.temporal_conv1 = nn.Identity()
        adapter.temporal_conv2 = nn.Identity()
        adapter.temporal_conv3 = nn.Identity()
        if len(in_features) >= 4:
            adapter.temporal_conv4 = nn.Identity()
    else:
        raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

    return adapter


class DPTHead(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        input_format,
        input_shape,
        patch_shape,
        features=256,
        use_bn=False,
        feature_map_channels=[256, 512, 1024, 1024],
        strategy="axial",
    ):
        super(DPTHead, self).__init__()

        self.strategy = strategy

        self.input_shape = input_shape
        self.input_format = input_format
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.patch_shape = patch_shape
        self.temporal_patch_size, self.axial_patch_size, self.lateral_patch_size = get_patch_sizes(
            input_format=input_format, patch_shape=patch_shape
        )
        self.num_patches, self.token_shape = calc_num_patches(
            input_fmt=self.input_format,
            input_shape=self.input_shape,
            patch_shape=patch_shape,
        )
        pixels_per_patch = PatchEmbedding.compute_num_pixels_per_patch(
            channels=self.output_channels,
            temporal_patch_size=self.temporal_patch_size,
            axial_patch_size=self.axial_patch_size,
            lateral_patch_size=self.lateral_patch_size,
            input_format=self.input_format,
        )
        self.pe_patchify = functools.partial(
            PatchEmbedding.patchify,
            temporal_patch_size=self.temporal_patch_size,
            axial_patch_size=self.axial_patch_size,
            lateral_patch_size=self.lateral_patch_size,
            input_format=self.input_format,
            num_patches=self.num_patches,
            pixels_per_patch=pixels_per_patch,
            token_shape=self.token_shape,
        )

        self.dim = 4 if "T" in input_format else 3
        if self.dim != 3:
            raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

        axis_to_value = dict(zip(input_format, input_shape))
        if axis_to_value.get("T", None) is not None:
            self.spatial_shape = (
                axis_to_value.get("T", None),
                axis_to_value.get("Z", None),
                axis_to_value.get("Y", None),
                axis_to_value.get("X", None),
            )
        else:
            self.spatial_shape = (
                axis_to_value.get("Z", None),
                axis_to_value.get("Y", None),
                axis_to_value.get("X", None),
            )

        self.spatial_patchified_shape = self._get_spatial_patchified_shape(
            self.spatial_shape,
            self.axial_patch_size,
            self.lateral_patch_size,
            self.temporal_patch_size,
            self.input_format,
        )

        if self.dim == 3 and self.strategy == "axial":
            self.spatial_projects = nn.ModuleList(
                [
                    nn.Conv3d(
                        in_channels=self.input_channels,
                        out_channels=feature_map_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                    for feature_map_channel in feature_map_channels
                ]
            )

            self.spatial_resize_layers = nn.ModuleList(
                [
                    nn.ConvTranspose3d(
                        in_channels=feature_map_channels[0],
                        out_channels=feature_map_channels[0],
                        kernel_size=4,
                        stride=4,
                        padding=0,
                    ),
                    nn.ConvTranspose3d(
                        in_channels=feature_map_channels[1],
                        out_channels=feature_map_channels[1],
                        kernel_size=2,
                        stride=2,
                        padding=0,
                    ),
                    nn.Identity(),
                    nn.Conv3d(
                        in_channels=feature_map_channels[3],
                        out_channels=feature_map_channels[3],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                ]
            )

        else:
            raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

        if self.dim == 3 and self.strategy == "axial":
            # FPN adapter and RefineNet
            self.FPNAdapter = _get_FPNAdapter(
                feature_map_channels, features, groups=1, expand=False, dim=self.dim, strategy=self.strategy
            )
            self.FPNAdapter.stem_transpose = None
            self.FPNAdapter.RefineNet_1 = _get_fusion_block(features, use_bn, dim=self.dim)
            self.FPNAdapter.RefineNet_2 = _get_fusion_block(features, use_bn, dim=self.dim)
            self.FPNAdapter.RefineNet_3 = _get_fusion_block(features, use_bn, dim=self.dim)
            self.FPNAdapter.RefineNet_4 = _get_fusion_block(features, use_bn, dim=self.dim)

            # output convolutions
            self.FPNAdapter.output_spatial_conv1 = nn.Conv3d(
                features, features // 2, kernel_size=3, stride=1, padding=1
            )

            if self.dim == 3 and self.strategy == "axial":
                self.FPNAdapter.output_temporal_conv1 = nn.Identity()
            else:
                raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

        else:
            raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

        if self.dim == 3 and self.strategy == "axial":
            self.FPNAdapter.output_spatial_conv2 = nn.Sequential(
                nn.Conv3d(features // 2, self.output_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(True)
            )

            if self.dim == 3 and self.strategy == "axial":
                self.FPNAdapter.output_temporal_conv2 = nn.Identity()
            else:
                raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

        else:
            raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

    def _get_spatial_patchified_shape(
        self, input_shape, axial_patch_size, lateral_patch_size, temporal_patch_size, input_format
    ):
        if input_format == "ZYXC":
            return (
                input_shape[0] // axial_patch_size,
                input_shape[1] // lateral_patch_size,
                input_shape[2] // lateral_patch_size,
            )
        elif input_format == "TZYXC":
            return (
                input_shape[0] // temporal_patch_size,
                input_shape[1] // axial_patch_size,
                input_shape[2] // lateral_patch_size,
                input_shape[3] // lateral_patch_size,
            )
        elif input_format == "TYXC":
            return (
                input_shape[0] // temporal_patch_size,
                input_shape[1] // lateral_patch_size,
                input_shape[2] // lateral_patch_size,
            )
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

    def forward(self, out_features):
        out = []
        # generate feature maps at different levels
        for i, x in enumerate(out_features):
            if self.dim == 3 and self.strategy == "axial":
                B, N, C = x.shape
                x = x.permute(0, 2, 1).reshape(B, C, *self.spatial_patchified_shape)
                x = self.spatial_projects[i](x)
                x = self.spatial_resize_layers[i](x)
                out.append(x)

            else:
                raise NotImplementedError("Only 3D input is currently supported.")

        # FPN adapter
        feature_map_1, feature_map_2, feature_map_3, feature_map_4 = out

        if self.dim == 3 and self.strategy == "axial":
            feature_map_1 = self.FPNAdapter.spatial_conv1(feature_map_1)
            feature_map_1 = self.FPNAdapter.temporal_conv1(feature_map_1)

            feature_map_2 = self.FPNAdapter.spatial_conv2(feature_map_2)
            feature_map_2 = self.FPNAdapter.temporal_conv2(feature_map_2)

            feature_map_3 = self.FPNAdapter.spatial_conv3(feature_map_3)
            feature_map_3 = self.FPNAdapter.temporal_conv3(feature_map_3)

            feature_map_4 = self.FPNAdapter.spatial_conv4(feature_map_4)
            feature_map_4 = self.FPNAdapter.temporal_conv4(feature_map_4)

        else:
            raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

        # RefineNet fusion
        if self.dim == 3 and self.strategy == "axial":
            # feature_map_#: B, C, Z, Y, X
            s3 = feature_map_3.shape[-3:]
            s2 = feature_map_2.shape[-3:]
            s1 = feature_map_1.shape[-3:]

            level_4 = self.FPNAdapter.RefineNet_4(feature_map_4, size=s3)
            level_3 = self.FPNAdapter.RefineNet_3(level_4, feature_map_3, size=s2)
            level_2 = self.FPNAdapter.RefineNet_2(level_3, feature_map_2, size=s1)
            level_1 = self.FPNAdapter.RefineNet_1(level_2, feature_map_1, size=None, scale_factor=2)

        else:
            raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

        if self.dim == 3 and self.strategy == "axial":
            out = self.FPNAdapter.output_spatial_conv1(level_1)
            out = self.FPNAdapter.output_temporal_conv1(out)

            out = F.interpolate(out, self.spatial_shape, mode="trilinear", align_corners=True)

            out = self.FPNAdapter.output_spatial_conv2(out)
            out = self.FPNAdapter.output_temporal_conv2(out)
            # out: [B, C, Z, Y, X] -> [B, Z, Y, X, C]
            out = out.permute(0, 2, 3, 4, 1)

        else:
            raise NotImplementedError("Only Dim=3 with axial strategy is supported.")

        out = self.pe_patchify(
            inputs=out,
            channels=self.output_channels,
        )

        return out


def _extract_model_kwargs(cfg: Mapping[str, Any]) -> dict:
    sig = inspect.signature(DPTHead.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    ignore = {"_target_", "BUILD"}

    kwargs = {}
    for k, v in cfg.items():
        if k in ignore or k not in allowed:
            continue
        kwargs[k] = v
    return kwargs


def BUILD(cfg: Mapping[str, Any]) -> DPTHead:
    """
    Hydra entrypoint for DPTHead.

    Example cfg:

      decoder_args:
        _target_: cell_observatory_platform.models.heads.dense_predictor_head.BUILD
        input_channels: 768   # usually embed_dim of backbone
        output_channels: 2
        input_format: TZYXC
        input_shape: [16, 128, 128, 128, 2]
        patch_shape: [4, 16, 16, 16]
        features: 256
        use_bn: true
        feature_map_channels: [256, 512, 1024, 1024]
        strategy: axial

    """
    return DPTHead(**_extract_model_kwargs(cfg))
