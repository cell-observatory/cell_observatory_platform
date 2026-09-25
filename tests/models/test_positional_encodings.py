import pytest

import numpy as np

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
    _apply_rope_v2_k_only,
    apply_rope,
    apply_rope_v1,
    apply_rope_v2,
    compute_axial_cis,
    compute_mixed_cis,
    generate_frequency_spectrum,
    generate_grid_indices,
    reshape_for_broadcast,
    RopePositionEmbedding,
    PositionEmbeddingRandom,
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


# ------------------------------------------------------
# Sin/Cos encodings: shapes and values
# ------------------------------------------------------


def test_sincos_fn_shapes():
    emb = sincos(embed_dim=8, pos=np.arange(4, dtype=np.float32))
    assert emb.shape == (4, 8)


def test_sincos_values_at_origin_and_unit_circle():
    """The table is [sin | cos] per frequency: zero phase at pos 0, unit circle everywhere."""
    emb = sincos(embed_dim=8, pos=np.array([0.0, 1.0], dtype=np.float32))  # [2, 8] = [sin | cos]
    assert emb.shape == (2, 8)
    # pos 0 -> sin = 0, cos = 1 in every frequency
    assert np.array_equal(emb[0, :4], np.zeros(4)) and np.array_equal(emb[0, 4:], np.ones(4))
    # sin^2 + cos^2 == 1 per frequency
    assert np.allclose(emb[1, :4] ** 2 + emb[1, 4:] ** 2, 1.0, atol=1e-6)
    # lowest frequency is 1 rad / position (exponent 0 -> w = 1)
    assert np.isclose(emb[1, 0], np.sin(1.0)) and np.isclose(emb[1, 4], np.cos(1.0))


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
def test_pos_embedding_interpolate_same_grid_is_identity(fmt, shape, patches, patch_shape):
    """Interpolating the table onto its own grid must return the table unchanged."""
    # interpolate=True is rejected at construction; call the grid-format method directly
    # (what forward's interpolate branch does with patches_used=None).
    pe = PosEmbedding(fmt, shape[1:], patch_shape, embed_dim=16, cls_token=False, interpolate=False)
    y = pe.interpolate_positional_encoding(torch.zeros(shape), pe.pos_embed)
    assert y.shape == pe.pos_embed.shape
    assert torch.equal(y, pe.pos_embed)


def test_pos_embedding_interpolate_true_raises():
    # config-contract guard: sequence-format callers would get silently wrong
    # interpolation, so construction must fail loud.
    with pytest.raises(NotImplementedError, match="grid"):
        PosEmbedding("ZYXC", (4, 4, 4, 1), (1, 2, 2), embed_dim=16, cls_token=False, interpolate=True)


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
    pe = PosEmbedding(fmt, shape[1:], patch_shape, embed_dim=16, cls_token=False, interpolate=False)
    x = torch.zeros(new_shape)
    y = pe.interpolate_positional_encoding(x, pe.pos_embed)
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
def test_rope_cis_and_apply_rope_v1_shapes(
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

    B, H, D = 2, num_heads, head_dim
    xq = torch.randn(B, H, N, D)
    xk = torch.randn(B, H, N, D)

    xq1, xk1 = apply_rope_v1(xq, xk, freqs_cis_ax)
    assert xq1.shape == xq.shape and xk1.shape == xk.shape

    xq2, xk2 = apply_rope_v1(xq, xk, freqs_cis_mixed)
    assert xq2.shape == xq.shape and xk2.shape == xk.shape


def test_apply_rope_v1_preserves_norm_and_is_identity_at_origin():
    """RoPE is a per-pair rotation: norms are preserved, the grid origin has zero phase."""
    torch.manual_seed(0)
    head_dim, end_x, end_y, end_z = 12, 3, 2, 2  # ZYXC: 3 axes * (12//6=2) pairs -> J = 6 = D/2
    cis = compute_axial_cis(head_dim, end_x, end_y, end_z, 1, input_fmt="ZYXC", theta=100.0, device="cpu")
    N = end_x * end_y * end_z
    assert cis.shape == (N, head_dim // 2)
    # token 0 is the grid origin (generate_grid_indices: idx 0 -> x=y=z=0) -> zero phase
    assert torch.allclose(cis[0], torch.ones_like(cis[0]))

    xq, xk = torch.randn(2, 3, N, head_dim), torch.randn(2, 3, N, head_dim)
    xq1, xk1 = apply_rope_v1(xq, xk, cis)
    assert torch.allclose(xq1.norm(dim=-1), xq.norm(dim=-1), atol=1e-5)
    assert torch.allclose(xk1.norm(dim=-1), xk.norm(dim=-1), atol=1e-5)
    assert torch.allclose(xq1[:, :, 0], xq[:, :, 0], atol=1e-6)  # origin untouched
    assert not torch.allclose(xq1[:, :, 1], xq[:, :, 1])  # neighbours rotated


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


# --------------------------------------------------------
# Tests: RopePositionEmbedding (custom RoPE module)
# --------------------------------------------------------


@pytest.mark.parametrize(
    "fmt,shape,embed_dim,num_heads",
    [
        ("ZYXC", (4, 8, 8), 48, 2),
        ("ZYXC", (2, 4, 4), 24, 1),
        ("TZYXC", (2, 4, 8, 8), 64, 2),
        ("TZYXC", (3, 2, 4, 4), 32, 1),
    ],
)
def test_rope_position_embedding_shapes(fmt, shape, embed_dim, num_heads):
    rpe = RopePositionEmbedding(
        input_fmt=fmt,
        embed_dim=embed_dim,
        num_heads=num_heads,
        theta=100.0,
    )
    sin, cos = rpe(shape)
    if fmt == "ZYXC":
        Z, Y, X = shape
        N = Z * Y * X
    else:
        T, Z, Y, X = shape
        N = T * Z * Y * X
    D_head = embed_dim // num_heads
    assert sin.shape == (N, D_head)
    assert cos.shape == (N, D_head)


def test_rope_position_embedding_deterministic():
    rpe = RopePositionEmbedding(
        input_fmt="ZYXC", embed_dim=48, num_heads=2, theta=100.0,
    )
    rpe.eval()
    s1, c1 = rpe((4, 8, 8))
    s2, c2 = rpe((4, 8, 8))
    assert torch.allclose(s1, s2)
    assert torch.allclose(c1, c2)


def test_rope_position_embedding_center_token_and_tiling():
    """Angles are tiled twice across D_head and the grid-centre token carries zero phase."""
    rpe = RopePositionEmbedding(input_fmt="ZYXC", embed_dim=24, num_heads=1, theta=100.0).eval()
    sin, cos = rpe((3, 3, 3))  # normalize_coords="separate": coords in {-2/3, 0, 2/3}
    D = 24
    assert sin.shape == cos.shape == (27, D)
    assert torch.allclose(sin ** 2 + cos ** 2, torch.ones_like(sin), atol=1e-6)
    # angles are tile(2)'d: first and second half identical
    assert torch.equal(sin[:, : D // 2], sin[:, D // 2 :]) and torch.equal(cos[:, : D // 2], cos[:, D // 2 :])
    # center token (z=y=x=1 -> coord 0) has zero phase
    c = 1 * 9 + 1 * 3 + 1
    assert torch.allclose(sin[c], torch.zeros(D), atol=1e-6) and torch.allclose(cos[c], torch.ones(D), atol=1e-6)
    assert not torch.allclose(sin[0], torch.zeros(D))


# --------------------------------------------------------
# Tests: PositionEmbeddingRandom
# --------------------------------------------------------

def test_position_embedding_random_zyxc():
    per = PositionEmbeddingRandom(input_fmt="ZYXC", num_pos_feats=16, time_separable=False)
    out = per((4, 8, 8))
    assert out.shape == (32, 4, 8, 8)  # C=2*num_pos_feats, Z, Y, X


def test_position_embedding_random_tzyxc_time_separable():
    per = PositionEmbeddingRandom(input_fmt="TZYXC", num_pos_feats=16, time_separable=True)
    out = per((4, 8, 8))
    assert out.shape == (32, 4, 8, 8)


def test_position_embedding_random_forward_with_coords():
    per = PositionEmbeddingRandom(input_fmt="TZYXC", num_pos_feats=16, time_separable=True)
    coords = torch.rand(1, 5, 3)
    out = per.forward_with_coords(coords, image_size=(4, 8, 8))
    assert out.shape == (1, 5, 32)


# ------------------------------------------------------
# PositionEmbeddingRandom: prompt PE vs dense grid PE
# ------------------------------------------------------


def test_position_embedding_random_prompt_pe_matches_dense_grid_at_voxel_center():
    """A prompt coordinate at a voxel's center (x+0.5, y+0.5, z+0.5) gets the
    same embedding as that voxel in the dense grid PE."""
    torch.manual_seed(0)
    Z, Y, X = 4, 6, 8
    enc = PositionEmbeddingRandom(input_fmt="ZYXC", num_pos_feats=16, time_separable=False)
    dense = enc((Z, Y, X))  # (C, Z, Y, X)

    voxels = [(0, 0, 0), (1, 2, 3), (3, 5, 7), (2, 0, 5)]
    # samplers emit (x, y, z); +0.5 puts the prompt at the voxel center, which
    # is where the dense grid's PE is evaluated
    coords = torch.tensor(
        [[(x + 0.5, y + 0.5, z + 0.5) for (z, y, x) in voxels]], dtype=torch.float32
    )
    prompt = enc.forward_with_coords(coords, (Z, Y, X))[0]  # (N, C)
    for i, (z, y, x) in enumerate(voxels):
        torch.testing.assert_close(prompt[i], dense[:, z, y, x], atol=1e-5, rtol=1e-5)


# ------------------------------------------------------
# compute_axial_cis: input_fmt validation
# ------------------------------------------------------


def test_compute_axial_cis_rejects_unknown_input_fmt():
    """An unsupported layout string raises instead of silently falling through."""
    with pytest.raises(ValueError, match="input_fmt"):
        compute_axial_cis(dim=24, end_x=2, end_y=2, end_z=2, end_t=2,
                          input_fmt="XYZT", device="cpu")


def test_compute_axial_cis_zyxc_emits_one_row_per_position():
    """ZYXC with a 2x2x2 grid yields 8 frequency rows (one per position)."""
    out = compute_axial_cis(dim=24, end_x=2, end_y=2, end_z=2, end_t=1,
                            input_fmt="ZYXC", device="cpu")
    assert out.shape[0] == 8


# ------------------------------------------------------
# apply_rope: per-side tuple branch and the k-only twin
# ------------------------------------------------------


def _rand_freqs_cis(n, jf, seed):
    g = torch.Generator().manual_seed(seed)
    angles = torch.rand(n, jf, generator=g) * 6.28
    return torch.polar(torch.ones(n, jf), angles)


def test_apply_rope_per_side_tuple_matches_v1_selection():
    """apply_rope with a (freqs_q, freqs_k) tuple equals rotating q with freqs_q
    and k with freqs_k through apply_rope_v1, bit for bit."""
    B, H, N, D = 2, 2, 6, 8
    g = torch.Generator().manual_seed(0)
    q = torch.randn(B, H, N, D, generator=g)
    k = torch.randn(B, H, N, D, generator=g)
    fq = _rand_freqs_cis(N, D // 2, seed=1)
    fk = _rand_freqs_cis(N, D // 2, seed=2)

    q_new, k_new = apply_rope(q, k, (fq, fk), rope_type="axial")
    q_old = apply_rope_v1(q, k, fq)[0]
    k_old = apply_rope_v1(q, k, fk)[1]
    torch.testing.assert_close(q_new, q_old, rtol=0, atol=0)
    torch.testing.assert_close(k_new, k_old, rtol=0, atol=0)


def test_apply_rope_none_side_passes_through_unrotated():
    """A None entry in the per-side tuple leaves that side untouched while the
    other side is rotated."""
    B, H, N, D = 1, 2, 4, 8
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    f = _rand_freqs_cis(N, D // 2, seed=3)

    q_rope, k_same = apply_rope(q, k, (f, None), rope_type="axial")
    assert torch.equal(k_same, k)
    assert not torch.equal(q_rope, q)

    q_same, k_rope = apply_rope(q, k, (None, f), rope_type="axial")
    assert torch.equal(q_same, q)
    assert not torch.equal(k_rope, k)


def test_apply_rope_v2_k_only_matches_v2_and_keeps_prefix_tokens():
    """_apply_rope_v2_k_only rotates k exactly like apply_rope_v2's k output and
    leaves the leading prefix tokens (those without sin/cos rows) unrotated."""
    B, H, N, D, prefix = 2, 2, 6, 8, 2
    g = torch.Generator().manual_seed(4)
    q = torch.randn(B, H, N, D, generator=g)
    k = torch.randn(B, H, N, D, generator=g)
    sin = torch.rand(N - prefix, D, generator=g)
    cos = torch.rand(N - prefix, D, generator=g)

    k_only = _apply_rope_v2_k_only(k, (sin, cos))
    k_ref = apply_rope_v2(q, k, (sin, cos))[1]
    torch.testing.assert_close(k_only, k_ref, rtol=0, atol=0)
    torch.testing.assert_close(k_only[:, :, :prefix, :], k[:, :, :prefix, :], rtol=0, atol=0)
    assert not torch.equal(k_only[:, :, prefix:, :], k[:, :, prefix:, :])
