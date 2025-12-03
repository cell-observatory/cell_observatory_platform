import pytest

import torch
import numpy as np

from cell_observatory_platform.models.positional_encoding import (
    sincos,
    positional_encoding_1d,
    positional_encoding_2d,
    positional_encoding_3d,
    positional_encoding_4d,
    PosEmbedding,
)

from cell_observatory_platform.models.rope import (
    generate_frequency_spectrum,
    generate_grid_indices,
    compute_mixed_cis,
    compute_axial_cis,
    apply_rotary_emb,
    reshape_for_broadcast,
)


# ------------------------------------------------------
# helpers
# ------------------------------------------------------


def tokens(fmt, shape, lps, aps, tps):
    if fmt == "XC":
        X = shape[1] // lps
        return X
    if fmt == "YXC":
        Y = shape[1] // lps
        X = shape[2] // lps
        return Y * X
    if fmt == "TYXC":
        T = shape[1] // tps
        Y = shape[2] // lps
        X = shape[3] // lps
        return T * Y * X
    if fmt == "ZYXC":
        Z = shape[1] // aps
        Y = shape[2] // lps
        X = shape[3] // lps
        return Z * Y * X
    if fmt == "TZYXC":
        T = shape[1] // tps
        Z = shape[2] // aps
        Y = shape[3] // lps
        X = shape[4] // lps
        return T * Z * Y * X
    raise ValueError

def fmt_shapes_and_patches():
    return [
        ("XC",    (1, 8, 1),               (2, 1, 1), (1, None)),
        ("YXC",   (1, 4, 4, 1),            (2, 1, 1), (1, 1, None)),
        ("TYXC",  (1, 3, 4, 4, 1),         (2, 1, 1), (2, 1, 1, None)),
        ("ZYXC",  (1, 3, 4, 4, 1),         (2, 1, 1), (1, 1, 1, None)),
        ("TZYXC", (1, 2, 3, 4, 4, 1),      (2, 1, 1), (2, 1, 1, 1, None)),
    ]


# ------------------------------------------------------
# Sin/Cos encodings: shapes only
# ------------------------------------------------------


def test_sincos_fn_shapes():
    emb = sincos(embed_dim=8, pos=np.arange(4, dtype=np.float32))
    assert emb.shape == (4, 8)

def test_posenc_1d_shapes():
    emb = positional_encoding_1d(8, 4, cls_token=False)
    emb_cls = positional_encoding_1d(8, 4, cls_token=True)
    assert emb.shape == (4, 8)
    assert emb_cls.shape == (5, 8)

@pytest.mark.parametrize("embed_dim,lateral", [(16, 3), (32, 2)])
def test_posenc_2d_shapes(embed_dim, lateral):
    emb = positional_encoding_2d(embed_dim, lateral, lateral, cls_token=False)
    emb_cls = positional_encoding_2d(embed_dim, lateral, lateral, cls_token=True)
    assert emb.shape == (lateral * lateral, embed_dim)
    assert emb_cls.shape == (1 + lateral * lateral, embed_dim)

def test_posenc_3d_axial_and_temporal_shapes():
    emb_ax = positional_encoding_3d(24, lateral_x_sequence_length=2, lateral_y_sequence_length=2, axial_sequence_length=3, cls_token=False)
    emb_tm = positional_encoding_3d(24, lateral_x_sequence_length=2, lateral_y_sequence_length=2, temporal_sequence_length=3, cls_token=False)
    assert emb_ax.shape == (3 * 2 * 2, 24)
    assert emb_tm.shape == (3 * 2 * 2, 24)

def test_posenc_4d_shapes():
    emb = positional_encoding_4d(32, 
                                 lateral_x_sequence_length=2, 
                                 lateral_y_sequence_length=2, 
                                 axial_sequence_length=3, 
                                 temporal_sequence_length=2, 
                                 cls_token=False)
    assert emb.shape == (2 * 3 * 2 * 2, 32)


# ------------------------------------------------------
# PosEmbedding forward: no-interp & interp
# ------------------------------------------------------


def _patch_shape_from(fmt, lps, aps, tps):
    if fmt == "TZYXC":
        return (tps, aps, lps, None)
    if fmt == "ZYXC":
        return (aps, lps, None)
    if fmt == "TYXC":
        return (tps, lps, None)
    if fmt == "YXC":
        return (lps, None)
    if fmt == "XC":
        return (lps, None)
    raise ValueError(f"unknown fmt {fmt}")


def fmt_shapes_and_patches():
    cases = [
        ("XC",    (1, 8, 1),               (2, 1, 1)),
        ("YXC",   (1, 4, 4, 1),            (2, 1, 1)),
        ("TYXC",  (1, 3, 4, 4, 1),         (2, 1, 1)),
        ("ZYXC",  (1, 3, 4, 4, 1),         (2, 1, 1)),
        ("TZYXC", (1, 2, 3, 4, 4, 1),      (2, 1, 1)),
    ]
    out = []
    for fmt, shape, (lps, aps, tps) in cases:
        patch_shape = _patch_shape_from(fmt, lps, aps, tps)
        out.append((fmt, shape, (lps, aps, tps), patch_shape))
    return out


@pytest.mark.parametrize("fmt,shape,patches,patch_shape", fmt_shapes_and_patches())
def test_pos_embedding_forward_no_interp_shapes(fmt, shape, patches, patch_shape):
    lps, aps, tps = patches
    pe = PosEmbedding(fmt, shape[1:], patch_shape, embed_dim=16, cls_token=False, interpolate=False) 
    x = torch.zeros(shape)
    y = pe(x)
    assert y.shape == (1, tokens(fmt, shape, lps, aps, tps), 16)


@pytest.mark.parametrize("fmt,shape,patches,patch_shape", fmt_shapes_and_patches())
def test_pos_embedding_forward_interp_identity_shapes(fmt, shape, patches, patch_shape):
    lps, aps, tps = patches
    pe = PosEmbedding(fmt, shape[1:], patch_shape, embed_dim=16, cls_token=False, interpolate=True)
    x = torch.zeros(shape)
    y = pe(x)
    assert y.shape == (1, tokens(fmt, shape, lps, aps, tps), 16)


@pytest.mark.parametrize(
    "fmt,shape,new_shape,patches",
    [
        ("XC",    (1,  8, 1),         (1, 12, 1),        (2, 1, 1)),
        ("YXC",   (1,  4, 4, 1),      (1,  6, 6, 1),     (2, 1, 1)),
        ("TYXC",  (1,  3, 4, 4, 1),   (1,  6, 6, 6, 1),  (2, 1, 1)),
        ("ZYXC",  (1,  3, 4, 4, 1),   (1,  3, 6, 6, 1),  (2, 1, 1)),
        ("TZYXC", (1,  2, 3, 4, 4, 1),(1,  6, 6, 6, 6,1),(2, 1, 1)),
    ],
)
def test_pos_embedding_forward_interp_resized_shapes(fmt, shape, new_shape, patches):
    lps, aps, tps = patches
    patch_shape = _patch_shape_from(fmt, lps, aps, tps)
    pe = PosEmbedding(fmt, shape[1:], patch_shape, embed_dim=16, cls_token=False, interpolate=True)
    x = torch.zeros(new_shape)
    y = pe(x)
    assert y.shape == (1, tokens(fmt, new_shape, lps, aps, tps), 16)


# ------------------------------------------------------
# RoPE: spectrum, mixed vs axial, apply_rotary shapes
# ------------------------------------------------------


@pytest.mark.parametrize(
    "gen_fmt,head_dim,num_heads,axes",
    [
        ("YXC",   8,  2, 2),   # 2D
        ("TYXC", 12, 3, 3),    # 3D (T,Y,X)
        ("ZYXC", 12, 3, 3),    # 3D (Z,Y,X)
        ("TZYXC",16, 4, 4),    # 4D
    ],
)
@pytest.mark.parametrize("random_rotation_per_head", [False, True])
def test_generate_frequency_spectrum_shapes(gen_fmt, head_dim, num_heads, axes, random_rotation_per_head):
    freqs = generate_frequency_spectrum(
        dim=head_dim,
        num_heads=num_heads,
        theta=100.0,
        random_rotation_per_head=random_rotation_per_head,
        input_fmt=gen_fmt,
    )
    assert freqs.shape[0] == axes
    assert freqs.shape[1] == num_heads
    assert freqs.shape[2] == head_dim // 2  # last dim should always compress to D/2

@pytest.mark.parametrize(
    "fmt,gen_fmt,head_dim,end_x,end_y,end_z,end_t",
    [
        ("YXC",   "YXC",   8,  3, 2, None, None),
        ("TYXC",  "TYXC", 12, 3, 2, None, 2),
        ("ZYXC",  "ZYXC", 12, 3, 2, 2, None),
        ("TZYXC", "TZYXC",16, 3, 2, 2, 2),
    ],
)
@pytest.mark.parametrize("random_rotation_per_head", [False, True])
def test_compute_mixed_and_axial_cis_and_apply_rotary_shapes(fmt, 
                                                             gen_fmt, 
                                                             head_dim, 
                                                             end_x, 
                                                             end_y, 
                                                             end_z, 
                                                             end_t, 
                                                             random_rotation_per_head):
    num_heads = 3
    freqs = generate_frequency_spectrum(
        dim=head_dim,
        num_heads=num_heads,
        theta=100.0,
        random_rotation_per_head=random_rotation_per_head,
        input_fmt=gen_fmt,
    )

    t_t, t_z, t_y, t_x = generate_grid_indices(end_x=end_x, 
                                               end_y=end_y, 
                                               end_z=end_z, 
                                               end_t=end_t, 
                                               input_fmt=fmt)
    N = (end_t or 1) * (end_z or 1) * end_y * end_x
    J = head_dim // 2

    # mixed per-head
    freqs_cis_mixed = compute_mixed_cis(freqs, num_heads, t_x, t_y, t_z, t_t, input_fmt=fmt)
    assert freqs_cis_mixed.shape == (num_heads, N, J)

    # axial
    freqs_cis_ax = compute_axial_cis(head_dim, 
                                     end_x, 
                                     end_y, 
                                     (end_z or 1), 
                                     (end_t or 1), 
                                     input_fmt=fmt, 
                                     theta=100.0)
    assert freqs_cis_ax.shape == (N, J)

    # apply_rotary_emb with both broadcast branches
    B, H, D = 2, num_heads, head_dim
    xq = torch.randn(B, H, N, D)
    xk = torch.randn(B, H, N, D)

    # branch 1: [N, J]
    xq1, xk1 = apply_rotary_emb(xq, xk, freqs_cis_ax)
    assert xq1.shape == xq.shape and xk1.shape == xk.shape

    # branch 2: [H, N, J]
    xq2, xk2 = apply_rotary_emb(xq, xk, freqs_cis_mixed)
    assert xq2.shape == xq.shape and xk2.shape == xk.shape

def test_reshape_for_broadcast_shapes():
    B, H, N, J = 2, 2, 6, 4
    dummy_x = torch.randn(B, H, N, J)
    # case 1: [N, J]
    fc1 = torch.polar(torch.ones(N, J), torch.zeros(N, J))
    out1 = reshape_for_broadcast(fc1, dummy_x)
    assert out1.shape == (1, 1, N, J)
    # case 2: [H, N, J]
    fc2 = torch.polar(torch.ones(H, N, J), torch.zeros(H, N, J))
    out2 = reshape_for_broadcast(fc2, dummy_x)
    assert out2.shape == (1, H, N, J)


# ------------------------------------------------------
# Grid indices: shapes only
# ------------------------------------------------------


def test_generate_grid_indices_shapes_all_formats():
    # YXC
    tt, tz, ty, tx = generate_grid_indices(3, 2, input_fmt="YXC")
    assert tt is None and tz is None
    assert ty.numel() == 6 and tx.numel() == 6

    # TYXC
    tt, tz, ty, tx = generate_grid_indices(3, 2, end_t=2, input_fmt="TYXC")
    assert tt.numel() == 12 and ty.numel() == 12 and tx.numel() == 12 and tz is None

    # ZYXC
    tt, tz, ty, tx = generate_grid_indices(3, 2, end_z=2, input_fmt="ZYXC")
    assert tz.numel() == 12 and ty.numel() == 12 and tx.numel() == 12 and tt is None

    # TZYXC
    tt, tz, ty, tx = generate_grid_indices(3, 2, end_z=2, end_t=2, input_fmt="TZYXC")
    assert tt.numel() == 24 and tz.numel() == 24 and ty.numel() == 24 and tx.numel() == 24