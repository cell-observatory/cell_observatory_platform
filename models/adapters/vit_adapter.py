import inspect
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
from timm.layers.drop import DropPath

from cell_observatory_platform.models.layers.attention import FlashDeformAttn3D, CrossAttention
from cell_observatory_platform.data.data_types import TORCH_DTYPES
from cell_observatory_platform.models.layers.activation import get_activation
from cell_observatory_platform.models.layers.norm import get_norm
from cell_observatory_platform.models.layers.utils import get_reference_points
from cell_observatory_platform.training.helpers import get_patch_sizes

try:
    from ops3d import _C
    OPS3D_AVAILABLE = True
except ImportError:
    OPS3D_AVAILABLE = False


class ConvFFN(nn.Module):
    def __init__(
        self, in_features, dim, hidden_features=None, out_features=None, act_layer="GELU", drop=0.0, strategy="axial"
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(embed_dim=hidden_features, dim=dim, strategy=strategy)
        self.act = get_activation(act_layer)()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, query_level_shapes, query_offsets):
        x = self.fc1(x)
        x = self.dwconv(x, query_level_shapes, query_offsets)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim, embed_dim=768, strategy="axial"):
        super().__init__()

        self.dim = dim
        self.embed_dim = embed_dim
        self.strategy = strategy

        if self.dim == 3 and self.strategy == "axial":
            self.spatial_conv = nn.Conv3d(
                embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim, bias=True
            )
            self.temporal_conv = (
                nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1) if dim == 4 else nn.Identity()
            )
        else:
            raise ValueError(f"Only Dim=3 is supported, " f"got dim={dim}, strategy={strategy}.")

    @staticmethod
    def _split_feature_maps(x, offsets):
        # returns list of [B, Ni, C]
        B, N, C = x.shape
        outs = []
        start = 0
        for n in offsets:
            outs.append(x[:, start : start + n, :])
            start += n
        assert start == N, "Split sizes do not sum to N, " f"got sum={start}, N={N}, offsets={offsets}"
        return outs

    def forward(self, x, query_level_shapes, query_offsets):
        B, N, C = x.shape
        x2, x1, x0 = self._split_feature_maps(x, query_offsets)

        out = []
        for feature_map, g in zip([x2, x1, x0], query_level_shapes):
            if self.dim == 3 and self.strategy == "axial":
                Z, Y, X = g
                y = feature_map.transpose(1, 2).reshape(B, C, Z, Y, X)
                y = self.spatial_conv(y).flatten(2).transpose(1, 2)
            else:
                raise ValueError(
                    f"Only Dim=3 with axial strategy is supported, " f"got dim={self.dim}, strategy={self.strategy}."
                )

            out.append(y)

        return torch.cat(out, dim=1)


class Extractor(nn.Module):
    def __init__(
        self,
        embed_dim,
        dim,
        use_deform_attention=False,
        num_heads=6,
        n_points=4,
        n_levels=1,
        deform_ratio=1.0,
        with_cffn=True,
        cffn_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        norm_layer="LayerNorm",
        strategy="axial",
    ):
        super().__init__()

        self.strategy = strategy

        norm = get_norm(norm_layer)
        self.query_norm = norm(embed_dim, eps=1e-6)
        self.feat_norm = norm(embed_dim, eps=1e-6)

        if use_deform_attention:
            assert dim == 3, "Deformable attention kernel is only supported in 3D currently."
            self.with_deform_attention = True
            self.attn = FlashDeformAttn3D(
                d_model=embed_dim,
                n_levels=n_levels,
                n_heads=num_heads,
                n_points=n_points,
                # ratio=deform_ratio
            )
        else:
            self.with_deform_attention = False
            self.attn = CrossAttention(dim=embed_dim, num_heads=num_heads)

        self.with_cffn = with_cffn
        if with_cffn:
            self.ffn = ConvFFN(
                in_features=embed_dim,
                dim=dim,
                hidden_features=int(embed_dim * cffn_ratio),
                drop=drop,
                strategy=strategy,
            )
            self.ffn_norm = norm(embed_dim, eps=1e-6)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self, query, features, reference_points, spatial_shapes, level_start_index, query_level_shapes, query_offsets
    ):
        if self.with_deform_attention:
            attn = self.attn(
                self.query_norm(query),
                reference_points,
                self.feat_norm(features),
                spatial_shapes,
                level_start_index,
                None,
            )
        else:
            attn = self.attn(self.query_norm(query), self.feat_norm(features))

        query = query + attn

        if self.with_cffn:
            query = query + self.drop_path(self.ffn(self.ffn_norm(query), query_level_shapes, query_offsets))
        return query


class InteractionBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        dim,
        # deformable attention params
        use_deform_attention=False,
        num_heads=6,
        n_points=4,
        n_levels=1,
        norm_layer="LayerNorm",
        drop=0.0,
        drop_path=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        init_values=0.0,
        deform_ratio=1.0,
        extra_extractor=False,
        strategy="axial",
    ):
        super().__init__()

        self.extractor = Extractor(
            dim=dim,
            embed_dim=embed_dim,
            use_deform_attention=use_deform_attention,
            n_levels=n_levels,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            strategy=strategy,
        )

        if extra_extractor:
            self.extra_extractors = nn.Sequential(
                *[
                    Extractor(
                        dim=dim,
                        embed_dim=embed_dim,
                        n_levels=1,
                        use_deform_attention=use_deform_attention,
                        num_heads=num_heads,
                        n_points=n_points,
                        norm_layer=norm_layer,
                        with_cffn=with_cffn,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        strategy=strategy,
                    )
                    for _ in range(2)
                ]
            )

        else:
            self.extra_extractors = None

    def forward(
        self, features, query, reference_points, spatial_shapes, level_start_index, query_level_shapes, query_offsets
    ):
        assert query.shape[1] == reference_points.shape[1]  # Len_q match
        assert spatial_shapes.prod(1).sum().item() == features.shape[1]  # Len_v match

        c = self.extractor(
            query=query,
            reference_points=reference_points,
            features=features,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            query_level_shapes=query_level_shapes,
            query_offsets=query_offsets,
        )

        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                query = extractor(
                    query=query,
                    reference_points=reference_points,
                    features=features,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    query_level_shapes=query_level_shapes,
                    query_offsets=query_offsets,
                )

        return features, query


class Conv3dBNAct(nn.Module):
    def __init__(self, cin, cout, k=(3, 3, 3), s=(1, 2, 2), p=None, bn=True):
        super().__init__()
        if p is None:
            p = tuple(kk // 2 for kk in k)
        self.conv = nn.Conv3d(cin, cout, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm3d(cout) if bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B*T, C, Z, Y, X]
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class SpatialPriorModule(nn.Module):
    def __init__(
        self,
        in_ch=2,
        inplanes=64,
        embed_dim=384,
        dim=3,
        strategy="axial",
        strides={
            "stem1": (2, 2, 2),
            "stem2": (1, 1, 1),
            "stem3": (1, 1, 1),
            "maxpool": 2,
            "stage2": (2, 2, 2),
            "stage3": (2, 2, 2),
            "stage4": (2, 2, 2),
        },
    ):
        super().__init__()

        self.dim = dim
        self.strategy = strategy

        self.strides = strides

        if self.dim == 3 or self.strategy == "axial":
            # stem over ZYX
            self.stem1 = Conv3dBNAct(in_ch, inplanes, k=(3, 3, 3), s=self.strides["stem1"])
            self.stem2 = Conv3dBNAct(inplanes, inplanes, k=(3, 3, 3), s=self.strides["stem2"])
            self.stem3 = Conv3dBNAct(inplanes, inplanes, k=(3, 3, 3), s=self.strides["stem3"])
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=self.strides["maxpool"], padding=1)

            self.stage2 = Conv3dBNAct(inplanes, 2 * inplanes, k=(3, 3, 3), s=self.strides["stage2"])  # /4 in ZYX

            self.stage3 = Conv3dBNAct(2 * inplanes, 4 * inplanes, k=(3, 3, 3), s=self.strides["stage3"])  # /8 in ZYX

            self.stage4 = Conv3dBNAct(4 * inplanes, 4 * inplanes, k=(3, 3, 3), s=self.strides["stage4"])  # /16 in ZYX

            # 1x1x1 projections to embed_dim
            self.fc1 = nn.Conv3d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.fc2 = nn.Conv3d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.fc3 = nn.Conv3d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.fc4 = nn.Conv3d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

        else:
            raise ValueError(
                f"Only Dim=3 or Dim=4 with axial strategy is supported, " f"got dim={dim}, strategy={strategy}."
            )

    def forward(self, x):
        # stem
        if self.dim == 3 and self.strategy == "axial":
            c1 = self.stem1(x)
            c1 = self.stem2(c1)
            c1 = self.stem3(c1)
            c1 = self.maxpool(c1)
        else:
            raise ValueError(
                f"Only Dim=3 or Dim=4 with axial strategy is supported, "
                f"got dim={self.dim}, strategy={self.strategy}."
            )

        # stage2
        if self.dim == 3 and self.strategy == "axial":
            c2 = self.stage2(c1)

        # stage3
        if self.dim == 3 and self.strategy == "axial":
            c3 = self.stage3(c2)

        # stage4
        if self.dim == 3 and self.strategy == "axial":
            c4 = self.stage4(c3)

        # proj. to embed dim
        if self.dim == 3 and self.strategy == "axial":
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)

            c1 = c1.permute(0, 2, 3, 4, 1).flatten(1, 3)
            c2 = c2.permute(0, 2, 3, 4, 1).flatten(1, 3)
            c3 = c3.permute(0, 2, 3, 4, 1).flatten(1, 3)
            c4 = c4.permute(0, 2, 3, 4, 1).flatten(1, 3)

        else:
            raise ValueError("Only Dim=3 or Dim=4 with axial strategy is supported.")

        return c1, c2, c3, c4


class EncoderAdapter(nn.Module):
    def __init__(
        self,
        dim,
        input_shape,
        patch_shape,
        backbone_embed_dim,
        input_format,
        num_backbone_features=4,
        add_vit_feature=True,
        # Spatial Prior Module parameters
        conv_inplane=64,
        # deformable attention parameters
        use_deform_attention=False,
        n_points=4,
        n_levels=1,
        deform_num_heads=16,
        drop_path_rate=0.3,
        init_values=0.0,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        use_extra_extractor=True,
        strategy="axial",
        spatial_prior_module_strides={
            "stem1": (2, 2, 2),
            "stem2": (1, 1, 1),
            "stem3": (1, 1, 1),
            "maxpool": 2,
            "stage2": (2, 2, 2),
            "stage3": (2, 2, 2),
            "stage4": (2, 2, 2),
        },
        dtype: str = "bfloat16",
    ):
        super(EncoderAdapter, self).__init__()

        self.strategy = strategy

        self.dim = dim

        self.dtype = TORCH_DTYPES[dtype].value if isinstance(dtype, str) else dtype

        self.add_vit_feature = add_vit_feature
        self.num_backbone_features = num_backbone_features

        self.use_deform_attention = use_deform_attention
        if use_deform_attention and not MSDEFORM_ATTN_AVAILABLE:
            raise ImportError("Please install the deformable attention module.")

        self.embed_dim = backbone_embed_dim
        self.patch_shape = patch_shape
        self.temporal_patch_size, self.axial_patch_size, self.lateral_patch_size = get_patch_sizes(
            input_format=input_format, patch_shape=self.patch_shape
        )

        self.input_shape = input_shape
        self.input_format = input_format

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

        self.in_channels = axis_to_value.get("C", None)
        assert self.in_channels is not None, "Input format must contain 'C' channel."

        self.spatial_patchified_shape = self._get_spatial_patchified_shape(
            self.spatial_shape,
            self.axial_patch_size,
            self.lateral_patch_size,
            self.temporal_patch_size,
            self.input_format,
        )

        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))

        # spatial prior module
        self.spatial_prior_module_strides = spatial_prior_module_strides
        # all modules in the order of forward pass
        self._spm_module_order = [
            "stem1",
            "stem2",
            "stem3",
            "maxpool",
            "stage2",
            "stage3",
            "stage4",
        ]
        # marks end of a stage
        self._spm_level_keys = ("maxpool", "stage2", "stage3", "stage4")

        self.query_level_shapes, self.query_offsets = self._get_query_metadata()

        self.spatial_prior_module = SpatialPriorModule(
            in_ch=self.in_channels,
            inplanes=conv_inplane,
            embed_dim=self.embed_dim,
            dim=self.dim,
            strategy=self.strategy,
            strides=self.spatial_prior_module_strides,
        )

        # injector/extractor
        self.adapter_block = nn.Sequential(
            *[
                InteractionBlock(
                    dim=self.dim,
                    embed_dim=self.embed_dim,
                    use_deform_attention=self.use_deform_attention,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    n_levels=n_levels,
                    init_values=init_values,
                    drop_path=drop_path_rate,
                    norm_layer="LayerNorm",
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=((True if i == self.num_backbone_features - 1 else False) and use_extra_extractor),
                    strategy=self.strategy,
                )
                for i in range(self.num_backbone_features)
            ]
        )

        if self.dim == 3 and self.strategy == "axial":
            self.up_spatial = nn.ConvTranspose3d(self.embed_dim, self.embed_dim, 2, 2)
        else:
            raise ValueError(
                f"Only Dim=3 with axial strategy is supported, " f"got dim={self.dim}, strategy={self.strategy}."
            )

        self.norm1 = nn.SyncBatchNorm(self.embed_dim)
        self.norm2 = nn.SyncBatchNorm(self.embed_dim)
        self.norm3 = nn.SyncBatchNorm(self.embed_dim)
        self.norm4 = nn.SyncBatchNorm(self.embed_dim)

        if self.use_deform_attention:
            self.apply(self._init_deform_weights)

    def _get_stride(self, key: str, val):
        if isinstance(val, int):
            if self.dim == 3:
                return (1, val, val, val)
        if isinstance(val, (tuple, list, ListConfig)):
            if len(val) == 3:
                z, y, x = val
                return (1, z, y, x)
            elif len(val) == 4:
                t, z, y, x = val
                return (t, z, y, x)
        raise ValueError(f"Bad stride spec for '{key}': {val}")

    def _get_cum_strides_per_stage(self, last_key: str):
        st = sz = sy = sx = 1
        for k in self._spm_module_order:
            if k in self.spatial_prior_module_strides:
                t_, z_, y_, x_ = self._get_stride(k, self.spatial_prior_module_strides[k])
                st *= t_
                sz *= z_
                sy *= y_
                sx *= x_
            if k == last_key:
                break
        return (st, sz, sy, sx)

    def _level_shapes_from_strides(self):
        if self.dim == 3:
            Z, Y, X = self.spatial_shape
            shapes = []
            for key in self._spm_level_keys:
                _, sz, sy, sx = self._get_cum_strides_per_stage(key)
                z = max(Z // sz, 1)
                y = max(Y // sy, 1)
                x = max(X // sx, 1)
                shapes.append((z, y, x))
            return shapes
        else:
            raise ValueError(f"Unsupported dim: {self.dim}")

    def _get_query_metadata(self):

        def _prod(tup):
            p = 1
            for v in tup:
                p *= int(v)
            return p

        shapes = self._level_shapes_from_strides()
        if self.dim == 3:
            offsets = [_prod((z, y, x)) for (z, y, x) in shapes]
        return shapes, offsets

    def _get_spatial_patchified_shape(
        self, spatial_shape, axial_patch_size, lateral_patch_size, temporal_patch_size, input_format
    ):
        if input_format == "ZYXC":
            return (
                spatial_shape[0] // axial_patch_size,
                spatial_shape[1] // lateral_patch_size,
                spatial_shape[2] // lateral_patch_size,
            )
        elif input_format == "TZYXC":
            return (
                spatial_shape[0] // temporal_patch_size,
                spatial_shape[1] // axial_patch_size,
                spatial_shape[2] // lateral_patch_size,
                spatial_shape[3] // lateral_patch_size,
            )
        elif input_format == "TYXC":
            return (
                spatial_shape[0] // temporal_patch_size,
                spatial_shape[1] // lateral_patch_size,
                spatial_shape[2] // lateral_patch_size,
            )
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

    def _init_deform_weights(self, m):
        if isinstance(m, FlashDeformAttn3D):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def _get_deformable_attention_metadata(
        self, device, B, feat_level_list, patch_sizes=[(8, 8, 8), (16, 16, 16), (32, 32, 32)]
    ):
        # spatial_shapes (num_levels, 3)
        spatial_shapes = torch.as_tensor(feat_level_list, dtype=torch.long, device=device)

        # level_start_index (num_levels,)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # NOTE: we are not really using valid ratios here, but need to pass them to MSDeformAttn
        num_levels = len(patch_sizes)
        valid_ratios = torch.ones((B, num_levels, 3), dtype=self.dtype, device=device)

        if self.dim == 3:
            # in DINOV3 they generate 1 level of reference points with three levels
            # worth of reference points (by varying patch sizes) for a single feature map
            # only applicable for MSDeformAttn in 3D
            feat_level_list = []
            for patch_z, patch_y, patch_x in patch_sizes:
                feat_level_list.append(
                    self._get_spatial_patchified_shape(
                        self.spatial_shape, patch_z, patch_y, self.temporal_patch_size, self.input_format
                    )
                )

        reference_points = get_reference_points(feat_level_list, valid_ratios, device)
        return reference_points, spatial_shapes, level_start_index, valid_ratios

    # def _get_deformable_and_ffn_metadata(self, x):
    #     device = x.device
    #     B = x.shape[0]

    #     if self.dim == 3:
    #         grids = [self.spatial_patchified_shape]
    #         reference_points, spatial_shapes, level_start_index, valid_ratios = \
    #             self._get_deformable_attention_metadata(device, B, grids)
    #         return reference_points, spatial_shapes, level_start_index, valid_ratios

    #     else:
    #         raise ValueError(f"Unsupported dim: {self.dim}")

    def _get_deformable_and_ffn_metadata(self, x: torch.Tensor):
        if self.dim != 3:
            raise ValueError(f"Unsupported dim: {self.dim}")

        device = x.device
        B = x.shape[0]

        Zp, Yp, Xp = self.spatial_patchified_shape
        spatial_shapes = torch.as_tensor([(Zp, Yp, Xp)], dtype=torch.long, device=device)
        level_start_index = spatial_shapes.new_zeros((1,))
        valid_ratios = torch.ones((B, 1, 3), dtype=self.dtype, device=device)

        level_shapes = self.query_level_shapes[1:]  # c2–c4

        ref_list = []
        for Z, Y, X in level_shapes:
            z_lin = torch.linspace(0.5, Z - 0.5, Z, device=device, dtype=self.dtype) / float(Z)
            y_lin = torch.linspace(0.5, Y - 0.5, Y, device=device, dtype=self.dtype) / float(Y)
            x_lin = torch.linspace(0.5, X - 0.5, X, device=device, dtype=self.dtype) / float(X)

            zz, yy, xx = torch.meshgrid(z_lin, y_lin, x_lin, indexing="ij")
            ref = torch.stack([zz, yy, xx], dim=-1)  # (Z, Y, X, 3), bf16
            ref = ref.reshape(1, -1, 1, 3)  # (1, Z*Y*X, 1, 3)
            ref_list.append(ref)

        reference_points = torch.cat(ref_list, dim=1)  # (1, Len_q, 1, 3)
        reference_points = reference_points.expand(B, -1, -1, -1)
        reference_points = reference_points * valid_ratios[:, None, :, :]
        return reference_points, spatial_shapes, level_start_index, valid_ratios

    def _upsample_spatial_3d(self, x: torch.Tensor, spatial_size, align_corners=False):
        if tuple(x.shape[-3:]) == tuple(spatial_size):
            return x
        return F.interpolate(x, size=spatial_size, mode="trilinear", align_corners=align_corners)

    def _scale_int(self, val, scale):
        return max(1, int(val * scale))

    def _scale_tuple(self, tup, scale):
        return tuple(max(1, int(s * scale)) for s in tup)

    def fuse_pyramid_additions(
        self,
        c_list,
        outs,
        dim,
        #    spatial_base_shape,
        #    temporal_base_len,
        #    spatial_scales=(4.0, 2.0, 1.0, 0.5),
        #    temporal_scales=(4.0, 2.0, 1.0, 0.5),
        align_corners=False,
        input_format: str = "CTZYX",
    ):
        # target_spatial_shapes = tuple(self._scale_tuple(spatial_base_shape, s) for s in spatial_scales)
        target_spatial_shapes = tuple(c_list[i].shape[2:] for i in range(len(c_list)))

        if dim == 3:
            # Spatial-only 3D upsampling
            scaled = []
            for c, o, tgt_sp in zip(c_list, outs, target_spatial_shapes):
                o_scaled = self._upsample_spatial_3d(o, tgt_sp, align_corners=align_corners)
                scaled.append(c + o_scaled)
            return tuple(scaled)

        else:
            raise ValueError(f"Unsupported dim: {dim}")

    def forward(self, x, features):
        reference_points, spatial_shapes, level_start_index, valid_ratios = self._get_deformable_and_ffn_metadata(x)

        # apply spatial prior module
        if self.dim == 3:
            # x: [B, Z, Y, X, C] -> [B, C, Z, Y, X]
            x = x.permute(0, 4, 1, 2, 3)

        c1, c2, c3, c4 = self.spatial_prior_module(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)

        # c: [B, N, C] for N=SUM (N_i) i=2,3,4 ResNet feature maps
        queries = torch.cat([c2, c3, c4], dim=1)

        bs, _, dim = features[0].shape

        outs = list()
        for i, block in enumerate(self.adapter_block):
            feature = features[i]
            _, queries = block(
                features=feature,
                query=queries,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                query_level_shapes=self.query_level_shapes[1:],
                query_offsets=self.query_offsets[1:],
            )
            # x: [B, N, C] -> [B, C, N] -> [B, C, *spatial_shape/patch_size]
            outs.append(feature.transpose(1, 2).view(bs, dim, *self.spatial_patchified_shape).contiguous())

        # split and reshape queries to c2, c3, c4
        c2 = queries[:, 0 : c2.size(1), :]
        c3 = queries[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = queries[:, c2.size(1) + c3.size(1) :, :]

        c2 = c2.transpose(1, 2).view(bs, dim, *self.query_level_shapes[1]).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, *self.query_level_shapes[2]).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, *self.query_level_shapes[3]).contiguous()

        if self.dim == 3:
            c1 = c1.transpose(1, 2).view(bs, dim, *self.query_level_shapes[0]).contiguous()
            c1 = self.up_spatial(c1)

        if self.add_vit_feature:
            if self.dim == 3:
                c1, c2, c3, c4 = self.fuse_pyramid_additions(
                    (c1, c2, c3, c4),
                    outs,
                    dim=3,
                    # temporal_base_len=None,
                    # spatial_base_shape=tuple(self.spatial_patchified_shape),
                    align_corners=False,
                )

            else:
                raise ValueError(f"Unsupported dim: {self.dim}")

        # final norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        return {"1": f1, "2": f2, "3": f3, "4": f4}


def _extract_model_kwargs(cfg: Mapping[str, Any]) -> dict:
    sig = inspect.signature(EncoderAdapter.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    ignore = {"_target_", "BUILD", "input_channels"}

    kwargs = {}
    for k, v in cfg.items():
        if k in ignore or k not in allowed:
            continue
        kwargs[k] = v
    return kwargs


def BUILD(adapter_args: dict):
    adapter_args = dict(adapter_args)  # make a copy

    input_channels = adapter_args.get("input_channels")
    input_shape = adapter_args.get("input_shape")
    if input_channels is not None:
        assert adapter_args["input_format"][-1] == "C", "Input format must end with 'C' when specifying input_channels."
        adapter_args["input_shape"] = list(input_shape)
        adapter_args["input_shape"][-1] = int(input_channels)
        adapter_args["input_shape"] = tuple(adapter_args["input_shape"])

    adapter_args.pop("input_channels", None)
    return EncoderAdapter(**_extract_model_kwargs(cfg=adapter_args))
