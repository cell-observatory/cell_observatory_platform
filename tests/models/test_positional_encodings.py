import os
import pytest
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from cell_observatory_platform.models.layers.positional_encoding import (
    # sincos
    PosEmbedding,
    positional_encoding_1d,
    positional_encoding_2d,
    positional_encoding_3d,
    positional_encoding_4d,
    sincos,
    # rope
    apply_rotary_emb,
    compute_axial_cis,
    compute_mixed_cis,
    generate_frequency_spectrum,
    generate_grid_indices,
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
        ("XC", (1, 8, 1), (2, 1, 1), (1, None)),
        ("YXC", (1, 4, 4, 1), (2, 1, 1), (1, 1, None)),
        ("TYXC", (1, 3, 4, 4, 1), (2, 1, 1), (2, 1, 1, None)),
        ("ZYXC", (1, 3, 4, 4, 1), (2, 1, 1), (1, 1, 1, None)),
        ("TZYXC", (1, 2, 3, 4, 4, 1), (2, 1, 1), (2, 1, 1, 1, None)),
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
    emb_ax = positional_encoding_3d(
        24, lateral_x_sequence_length=2, lateral_y_sequence_length=2, axial_sequence_length=3, cls_token=False
    )
    emb_tm = positional_encoding_3d(
        24, lateral_x_sequence_length=2, lateral_y_sequence_length=2, temporal_sequence_length=3, cls_token=False
    )
    assert emb_ax.shape == (3 * 2 * 2, 24)
    assert emb_tm.shape == (3 * 2 * 2, 24)


def test_posenc_4d_shapes():
    emb = positional_encoding_4d(
        32,
        lateral_x_sequence_length=2,
        lateral_y_sequence_length=2,
        axial_sequence_length=3,
        temporal_sequence_length=2,
        cls_token=False,
    )
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
        ("XC", (1, 8, 1), (2, 1, 1)),
        ("YXC", (1, 4, 4, 1), (2, 1, 1)),
        ("TYXC", (1, 3, 4, 4, 1), (2, 1, 1)),
        ("ZYXC", (1, 3, 4, 4, 1), (2, 1, 1)),
        ("TZYXC", (1, 2, 3, 4, 4, 1), (2, 1, 1)),
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
        ("XC", (1, 8, 1), (1, 12, 1), (2, 1, 1)),
        ("YXC", (1, 4, 4, 1), (1, 6, 6, 1), (2, 1, 1)),
        ("TYXC", (1, 3, 4, 4, 1), (1, 6, 6, 6, 1), (2, 1, 1)),
        ("ZYXC", (1, 3, 4, 4, 1), (1, 3, 6, 6, 1), (2, 1, 1)),
        ("TZYXC", (1, 2, 3, 4, 4, 1), (1, 6, 6, 6, 6, 1), (2, 1, 1)),
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
        ("YXC", 8, 2, 2),  # 2D
        ("TYXC", 12, 3, 3),  # 3D (T,Y,X)
        ("ZYXC", 12, 3, 3),  # 3D (Z,Y,X)
        ("TZYXC", 16, 4, 4),  # 4D
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
        ("YXC", "YXC", 8, 3, 2, None, None),
        ("TYXC", "TYXC", 12, 3, 2, None, 2),
        ("ZYXC", "ZYXC", 12, 3, 2, 2, None),
        ("TZYXC", "TZYXC", 16, 3, 2, 2, 2),
    ],
)
@pytest.mark.parametrize("random_rotation_per_head", [False, True])
def test_compute_mixed_and_axial_cis_and_apply_rotary_shapes(
    fmt, gen_fmt, head_dim, end_x, end_y, end_z, end_t, random_rotation_per_head
):
    num_heads = 3
    freqs = generate_frequency_spectrum(
        dim=head_dim,
        num_heads=num_heads,
        theta=100.0,
        random_rotation_per_head=random_rotation_per_head,
        input_fmt=gen_fmt,
    )

    t_t, t_z, t_y, t_x = generate_grid_indices(end_x=end_x, end_y=end_y, end_z=end_z, end_t=end_t, input_fmt=fmt)
    N = (end_t or 1) * (end_z or 1) * end_y * end_x
    J = head_dim // 2

    # mixed per-head
    freqs_cis_mixed = compute_mixed_cis(freqs, t_x, t_y, t_z, t_t, input_fmt=fmt)
    assert freqs_cis_mixed.shape == (num_heads, N, J)

    # axial
    freqs_cis_ax = compute_axial_cis(head_dim, end_x, end_y, (end_z or 1), (end_t or 1), input_fmt=fmt, theta=100.0)
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


# ------------------------------------------------
# Plotting Sanity Checks for Positional Encodings
# ------------------------------------------------

# ----------------------------
# Plot helpers
# ----------------------------

def _get_outdir(tmp_path: Path) -> Path:
    env = os.environ.get("POSENC_PLOT_DIR", "")
    if env:
        out = Path(env).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out
    return tmp_path

def _savefig(outdir: Path, name: str) -> Path:
    path = outdir / name
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def plot_pe_heatmap(pe: np.ndarray, outdir: Path, name: str):
    # pe: [L, D]
    plt.figure(figsize=(7, 4))
    plt.imshow(pe, aspect="auto", interpolation="nearest")
    plt.xlabel("dim")
    plt.ylabel("position")
    plt.title("PE heatmap: PE[pos, dim]")
    plt.colorbar()
    return _savefig(outdir, name)

def plot_pe_similarity(pe: np.ndarray, outdir: Path, name: str):
    x = torch.from_numpy(pe).float()
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    S = (x @ x.T).cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(S, aspect="auto", interpolation="nearest")
    plt.title("Cosine similarity between positions")
    plt.xlabel("j")
    plt.ylabel("i")
    plt.colorbar()
    return _savefig(outdir, name)

def plot_grid_channel(img2d: np.ndarray, outdir: Path, name: str, title: str):
    plt.figure(figsize=(4, 4))
    plt.imshow(img2d, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    return _savefig(outdir, name)

# ----------------------------
# Tests: SinCos 3D
# ----------------------------

@pytest.mark.skip(reason="For debugging purposes only")
def test_sincos_3d_plots(tmp_path):
    outdir = _get_outdir(tmp_path)

    D = 120
    Z, Y, X = 6, 8, 10
    pe = positional_encoding_3d(
        embed_dim=D,
        lateral_x_sequence_length=X,
        lateral_y_sequence_length=Y,
        axial_sequence_length=Z,
        temporal_sequence_length=None,
        cls_token=False,
    )
    assert pe.shape == (Z * Y * X, D)

    # Per 2D z-slice: heatmap and cosine similarity over the (Y, X) positions at that z
    pe_grid = pe.reshape(Z, Y, X, D)  # [Z, Y, X, D]
    z_slices = [0, Z // 2, Z - 1] if Z >= 3 else list(range(Z))
    for zi in z_slices:
        pe_slice = pe_grid[zi].reshape(Y * X, D)  # [Y*X, D]
        plot_pe_heatmap(pe_slice, outdir, f"sincos_3d_heatmap_z{zi}.png")
        plot_pe_similarity(pe_slice, outdir, f"sincos_3d_similarity_z{zi}.png")

# ----------------------------
# Tests: SinCos 4D
# ----------------------------

@pytest.mark.skip(reason="For debugging purposes only")
def test_sincos_4d_plots(tmp_path):
    outdir = _get_outdir(tmp_path)

    D = 128
    T, Z, Y, X = 3, 4, 6, 5
    pe = positional_encoding_4d(
        embed_dim=D,
        lateral_x_sequence_length=X,
        lateral_y_sequence_length=Y,
        axial_sequence_length=Z,
        temporal_sequence_length=T,
        cls_token=False,
    )
    assert pe.shape == (T * Z * Y * X, D)

    plot_pe_heatmap(pe, outdir, "sincos_4d_heatmap_head.png")
    plot_pe_similarity(pe, outdir, "sincos_4d_similarity.png")