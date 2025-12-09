import pytest
import torch

from cell_observatory_platform.models.heads.linear_head import LinearHead
from cell_observatory_platform.models.layers.patch_embeddings import calc_num_patches


def _num_patches(input_format, input_shape, axial_patch, lateral_patch, temporal_patch):
    if "T" not in input_format:
        patch_shape = (axial_patch, lateral_patch, lateral_patch, None)
    else:
        patch_shape = (temporal_patch, axial_patch, lateral_patch, lateral_patch, None)

    num_patches, _ = calc_num_patches(input_fmt=input_format, input_shape=input_shape, patch_shape=patch_shape)
    return int(num_patches)


def _pixels_per_patch(input_format, input_shape, axial_patch, lateral_patch, temporal_patch):
    C = input_shape[-1]
    Zp = axial_patch
    Yp = lateral_patch
    Xp = lateral_patch
    Tp = temporal_patch if "T" in input_format else 1
    return int(Tp * Zp * Yp * Xp * C)


@pytest.mark.parametrize(
    "input_format,input_shape,axial_patch,lateral_patch,temporal_patch,in_dim,bottleneck_dim",
    [
        # TZYXC: (B, T, Z, Y, X, C)
        ("TZYXC", (2, 4, 8, 8, 8, 16), 1, 4, 1, 128, 64),
        # ZYXC:  (B, Z, Y, X, C)
        ("ZYXC", (3, 16, 16, 16, 32), 2, 4, 1, 256, 128),
    ],
)
def test_linear_head_default_shapes(
    input_format, input_shape, axial_patch, lateral_patch, temporal_patch, in_dim, bottleneck_dim
):
    B = input_shape[0]
    L = _num_patches(input_format, input_shape, axial_patch, lateral_patch, temporal_patch)
    P = _pixels_per_patch(input_format, input_shape, axial_patch, lateral_patch, temporal_patch)

    head = LinearHead(
        in_dim=in_dim,
        output_dim=P,
        use_bn=False,
        nlayers=3,
        hidden_dim=max(2 * bottleneck_dim, bottleneck_dim),
        bottleneck_dim=bottleneck_dim,
        mlp_bias=True,
    )

    x = torch.randn(B, L, in_dim)
    with torch.no_grad():
        y = head(x)

    assert y.shape == (B, L, P)


@pytest.mark.parametrize(
    "input_format,input_shape,axial_patch,lateral_patch,temporal_patch,in_dim,bottleneck_dim",
    [
        ("TZYXC", (1, 2, 4, 4, 4, 8), 1, 2, 1, 64, 32),
        ("ZYXC", (2, 8, 8, 8, 16), 2, 2, 1, 128, 64),
    ],
)
def test_linear_head_no_last_layer_shapes(
    input_format, input_shape, axial_patch, lateral_patch, temporal_patch, in_dim, bottleneck_dim
):
    B = input_shape[0]
    L = _num_patches(input_format, input_shape, axial_patch, lateral_patch, temporal_patch)
    P = _pixels_per_patch(input_format, input_shape, axial_patch, lateral_patch, temporal_patch)

    head = LinearHead(
        in_dim=in_dim,
        output_dim=P,
        use_bn=False,
        nlayers=2,
        hidden_dim=max(2 * bottleneck_dim, bottleneck_dim),
        bottleneck_dim=bottleneck_dim,
        mlp_bias=True,
    )

    x = torch.randn(B, L, in_dim)
    with torch.no_grad():
        y = head(x, no_last_layer=True)

    assert y.shape == (B, L, bottleneck_dim)


@pytest.mark.parametrize(
    "input_format,input_shape,axial_patch,lateral_patch,temporal_patch,bottleneck_dim",
    [
        ("TZYXC", (2, 3, 6, 6, 6, 10), 1, 3, 1, 64),
        ("ZYXC", (1, 10, 10, 10, 20), 2, 5, 1, 128),
    ],
)
def test_linear_head_only_last_layer_shapes(
    input_format, input_shape, axial_patch, lateral_patch, temporal_patch, bottleneck_dim
):
    B = input_shape[0]
    L = _num_patches(input_format, input_shape, axial_patch, lateral_patch, temporal_patch)
    P = _pixels_per_patch(input_format, input_shape, axial_patch, lateral_patch, temporal_patch)

    head = LinearHead(
        in_dim=999,
        output_dim=P,
        use_bn=False,
        nlayers=3,
        hidden_dim=bottleneck_dim * 2,
        bottleneck_dim=bottleneck_dim,
        mlp_bias=True,
    )

    x = torch.randn(B, L, bottleneck_dim)
    with torch.no_grad():
        y = head(x, only_last_layer=True)

    assert y.shape == (B, L, P)
