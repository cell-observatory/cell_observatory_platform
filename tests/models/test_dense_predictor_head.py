import pytest
import torch
import torch.nn as nn

from cell_observatory_platform.models.heads.dense_predictor_head import DPTHead, FeatureFusionBlock, ResidualConvUnit


# ResidualConvUnit — shape preservation (3D and 4D)
@pytest.mark.parametrize("bn", [False, True])
@pytest.mark.parametrize("features", [8, 16])
def test_residual_conv_unit_3d_shapes(features, bn):
    act = nn.ReLU(inplace=False)
    m = ResidualConvUnit(features=features, activation=act, bn=bn, dim=3, strategy="axial")
    B, C, Z, Y, X = 2, features, 5, 7, 9
    x = torch.randn(B, C, Z, Y, X)
    y = m(x)
    assert y.shape == x.shape


# @pytest.mark.parametrize("bn", [False, True])
# @pytest.mark.parametrize("features", [8])
# def test_residual_conv_unit_4d_shapes(features, bn):
#     act = nn.ReLU(inplace=False)
#     m = ResidualConvUnit(features=features, activation=act, bn=bn, dim=4, strategy="axial")
#     B, C, T, Z, Y, X = 2, features, 3, 5, 7, 9
#     x = torch.randn(B, T, Z, Y, X, C)
#     y = m(x)
#     assert y.shape == x.shape


# FeatureFusionBlock — upsampling behavior (3D and 4D)
@pytest.mark.parametrize("features", [16, 32])
def test_feature_fusion_block_3d_no_skip(features):
    m = FeatureFusionBlock(
        features=features,
        dim=3,
        activation=nn.ReLU(False),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        strategy="axial",
    )
    B, C, Z, Y, X = 2, features, 4, 6, 7
    scaled_size = (Z * 2, Y * 2, X * 2)
    x = torch.randn(B, C, Z, Y, X)
    y = m(x, size=scaled_size)  # default scale_factor=2
    assert y.shape == (B, C, Z * 2, Y * 2, X * 2)


@pytest.mark.parametrize("features", [16])
def test_feature_fusion_block_3d_with_skip(features):
    m = FeatureFusionBlock(
        features=features,
        dim=3,
        activation=nn.ReLU(False),
        deconv=False,
        bn=True,
        expand=False,
        align_corners=True,
        strategy="axial",
    )
    B, C, Z, Y, X = 1, features, 5, 5, 5
    x0 = torch.randn(B, C, Z, Y, X)
    x1 = torch.randn(B, C, Z, Y, X)
    scaled_size = (Z * 2, Y * 2, X * 2)
    y = m(x0, x1, size=scaled_size)  # default scale_factor=2
    assert y.shape == (B, C, Z * 2, Y * 2, X * 2)


# @pytest.mark.parametrize("features", [8])
# def test_feature_fusion_block_4d_no_skip(features):
#     m = FeatureFusionBlock(
#         features=features,
#         dim=4,
#         activation=nn.ReLU(False),
#         deconv=False,
#         bn=False,
#         expand=False,
#         align_corners=True,
#         strategy="axial"
#     )
#     B, C, T, Z, Y, X = 2, features, 2, 3, 4, 5
#     scaled_size = (T * 2, Z * 2, Y * 2, X * 2)
#     x = torch.randn(B, T, Z, Y, X, C)
#     y = m(x, size=scaled_size)
#     assert y.shape == (B, T * 2, Z * 2, Y * 2, X * 2, C)


# @pytest.mark.parametrize("features", [8])
# def test_feature_fusion_block_4d_with_skip(features):
#     m = FeatureFusionBlock(
#         features=features,
#         dim=4,
#         activation=nn.ReLU(False),
#         deconv=False,
#         bn=True,
#         expand=False,
#         align_corners=True,
#         strategy="axial",
#     )
#     B, C, T, Z, Y, X = 1, features, 3, 3, 3, 3
#     scaled_size = (T * 2, Z * 2, Y * 2, X * 2)
#     x0 = torch.randn(B, T, Z, Y, X, C)
#     x1 = torch.randn(B, T, Z, Y, X, C)
#     y = m(x0, x1, size=scaled_size)
#     assert y.shape == (B, T * 2, Z * 2, Y * 2, X * 2, C)


# # 4D path (layout: TZYXC)
# @pytest.mark.parametrize(
#     "B,C_in,T,Z,Y,X,Tp,Zp,YXp,out_channels",
#     [
#         # Full image: 32×48×64×64; patches: 4×8×8
#         # Tokens: (8 * 6 * 8 * 8) = 3072
#         (2, 32, 32, 48, 64, 64, 4, 8, 8, [64, 128, 256, 256]),
#         # Full image: 48×32×48×48; patches: 6×8×12
#         # Tokens: (8 * 4 * 4 * 4) = 512
#         (1, 32, 48, 32, 48, 48, 6, 8, 12, [32, 64, 128, 128]),
#     ],
# )
# def test_dpthead_4d_output_shape_tzyxc(B, C_in, T, Z, Y, X, Tp, Zp, YXp, out_channels):
#     device = "cpu"
#     input_format = "TZYXC"
#     input_shape = (B, T, Z, Y, X, C_in)

#     head = DPTHead(
#         input_channels=C_in,
#         output_channels=2,
#         input_shape=input_shape,
#         input_format=input_format,
#         temporal_patch_size=Tp,
#         axial_patch_size=Zp,
#         lateral_patch_size=YXp,
#         features=64,
#         use_bn=False,
#         feature_map_channels=out_channels,
#         strategy="axial",
#     ).to(device)

#     Nt = (T // Tp) * (Z // Zp) * (Y // YXp) * (X // YXp)
#     levels = [torch.randn(B, Nt, C_in, device=device) for _ in range(4)]

#     with torch.no_grad():
#         out = head(levels)

#     assert out.shape == (B, T, Z, Y, X, 2)


# 3D path (layout: ZYXC)
@pytest.mark.parametrize(
    "B,C_in,Z,Y,X,Zp,YXp,out_channels",
    [
        # Full image: 64×64×64; patches: 8×8
        # Tokens: (8 * 8 * 8) = 512
        (2, 32, 64, 64, 64, 8, 8, [64, 128, 256, 256]),
        # Full image: 48×32×32; patches: 6×8
        # Tokens: (8 * 4 * 4) = 128
        (1, 32, 48, 32, 32, 6, 8, [32, 64, 128, 128]),
    ],
)
def test_dpthead_3d_output_shape_zyxc(B, C_in, Z, Y, X, Zp, YXp, out_channels):
    device = "cpu"
    input_format = "ZYXC"
    input_shape = (Z, Y, X, C_in)

    head = DPTHead(
        input_channels=C_in,
        output_channels=2,
        input_shape=input_shape,
        input_format=input_format,
        patch_shape=(Zp, YXp, YXp, None),
        features=64,
        use_bn=False,
        feature_map_channels=out_channels,
        strategy="axial",
    ).to(device)

    N = (Z // Zp) * (Y // YXp) * (X // YXp)
    levels = [torch.randn(B, N, C_in, device=device) for _ in range(4)]

    with torch.no_grad():
        out = head(levels)

    pixels_per_patch = Zp * YXp * YXp * 2
    assert out.shape == (B, N, pixels_per_patch)
