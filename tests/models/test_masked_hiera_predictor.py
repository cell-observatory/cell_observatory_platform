import math

import pytest
import torch

from cell_observatory_platform.models.backbones.masked_hiera_encoder import MaskedHieraEncoder
from cell_observatory_platform.models.heads.masked_hiera_predictor import (
    MaskedHieraPredictor,
    _mu_raster_perms,
)


ENC_CFG = dict(
    input_fmt="ZYXC",
    input_shape=(32, 64, 64, 1),   # tokens (8,8,8)
    patch_shape=(4, 8, 8, None),
    embed_dim=32,
    num_heads=2,
    drop_path_rate=0.0,
    q_pool=1,
    q_stride=(2, 2, 2),
    stages=(1, 1, 1, 1),
    norm_layer="LayerNorm",
    mask_unit_size=(2, 2, 2),
)


def _build_encoder_and_predictor(prediction_mode="pixels"):
    enc = MaskedHieraEncoder(**ENC_CFG, channel_proj_type="none")
    spec = enc.get_decoder_spec()
    encoder_dim_out = enc.encoder.blocks[-1].dim_out

    pred = MaskedHieraPredictor(
        input_fmt=ENC_CFG["input_fmt"],
        input_shape=ENC_CFG["input_shape"],
        patch_shape=ENC_CFG["patch_shape"],
        encoder_dim_out=encoder_dim_out,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=2,
        decoder_spec=spec,
        prediction_mode=prediction_mode,
    )
    return enc, pred


def test_predictor_pixels_mode():
    enc, pred = _build_encoder_and_predictor(prediction_mode="pixels")
    B = 2
    x = torch.randn((B,) + tuple(ENC_CFG["input_shape"]))

    with torch.no_grad():
        enc_out, patches = enc(x)

    num_mus = enc.num_patches // enc.encoder.flat_mu_size
    keep = max(1, num_mus // 2)
    ctx_idx = torch.stack([torch.randperm(num_mus)[:keep] for _ in range(B)])
    mu_mask = torch.ones(B, num_mus, dtype=torch.bool)

    with torch.no_grad():
        out = pred(enc_out, mu_mask=mu_mask, ctx_idx=ctx_idx)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == B
    total_patches = math.prod(pred.mu_grid) * math.prod(pred.mu_window_patches)
    assert out.shape[1] == total_patches
    assert out.shape[2] == pred.pixels_per_patch


def test_predictor_lowest_level_mode():
    enc = MaskedHieraEncoder(**ENC_CFG, channel_proj_type="none")
    spec = enc.get_decoder_spec()
    encoder_dim_out = enc.encoder.blocks[-1].dim_out
    output_embed_dim = 16

    pred = MaskedHieraPredictor(
        input_fmt=ENC_CFG["input_fmt"],
        input_shape=ENC_CFG["input_shape"],
        patch_shape=ENC_CFG["patch_shape"],
        encoder_dim_out=encoder_dim_out,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=2,
        decoder_spec=spec,
        prediction_mode="lowest_level",
        output_embed_dim=output_embed_dim,
    )
    B = 1
    x = torch.randn((B,) + tuple(ENC_CFG["input_shape"]))

    with torch.no_grad():
        enc_out, patches = enc(x)

    num_mus = enc.num_patches // enc.encoder.flat_mu_size
    keep = max(1, num_mus // 2)
    ctx_idx = torch.stack([torch.randperm(num_mus)[:keep] for _ in range(B)])
    mu_mask = torch.ones(B, num_mus, dtype=torch.bool)

    with torch.no_grad():
        out = pred(enc_out, mu_mask=mu_mask, ctx_idx=ctx_idx)

    assert out.shape[0] == B
    assert out.shape[-1] == output_embed_dim


# --------------------------------------------------------------------------- #
# _tokens_to_patch_tokens: MU-major (mu, tok, sub) -> raster patch order
# --------------------------------------------------------------------------- #


def _reference_patch_index(mu_grid, tok_in_mu, sub, mu_id, tok_id, sub_id):
    """Raster patch index for (mu, tok, sub) multi-indices -- independent loop
    implementation of the layout contract."""
    D = len(mu_grid)

    def unravel(idx, dims):
        coords = []
        for d in reversed(dims):
            coords.append(idx % d)
            idx //= d
        return list(reversed(coords))

    mu_c = unravel(mu_id, mu_grid)
    tok_c = unravel(tok_id, tok_in_mu)
    sub_c = unravel(sub_id, sub)
    patch_grid = [mu_grid[i] * tok_in_mu[i] * sub[i] for i in range(D)]
    patch_c = [
        mu_c[i] * (tok_in_mu[i] * sub[i]) + tok_c[i] * sub[i] + sub_c[i]
        for i in range(D)
    ]
    flat = 0
    for i in range(D):
        flat = flat * patch_grid[i] + patch_c[i]
    return flat


@pytest.mark.parametrize(
    "mu_grid,tok_in_mu,sub",
    [
        ((2, 2, 2), (2, 2, 2), (2, 2, 2)),
        ((1, 2, 2), (2, 2, 1), (1, 1, 2)),
        ((2, 1, 3), (1, 2, 2), (2, 1, 1)),
    ],
)
def test_tokens_to_patch_tokens_is_mu_major_to_raster(mu_grid, tok_in_mu, sub):
    """Every (MU-major token n, sub-patch k) value lands on the raster patch row
    given by the per-axis (mu, tok, sub) -> patch coordinate expansion."""
    pp = 1
    n_mu = math.prod(mu_grid)
    tok_prod = math.prod(tok_in_mu)
    sub_prod = math.prod(sub)
    N = n_mu * tok_prod
    K = sub_prod * pp

    x = torch.arange(N * K, dtype=torch.float32).view(1, N, K)

    out = MaskedHieraPredictor._tokens_to_patch_tokens(
        None, x, mu_grid, tok_in_mu, sub, pp
    )  # (1, n_patches, pp)
    assert out.shape == (1, N * sub_prod, pp)

    for n in range(N):
        mu_id, tok_id = divmod(n, tok_prod)
        for k in range(sub_prod):
            expected_row = _reference_patch_index(mu_grid, tok_in_mu, sub, mu_id, tok_id, k)
            assert out[0, expected_row, 0].item() == x[0, n, k].item(), (
                f"token {n} sub {k} landed wrong (mu_grid={mu_grid})"
            )


def test_tokens_to_patch_tokens_rejects_non_factorable_length():
    """A sequence length that is not prod(mu_grid) * prod(tok_in_mu) raises."""
    with pytest.raises(ValueError, match="_tokens_to_patch_tokens"):
        MaskedHieraPredictor._tokens_to_patch_tokens(
            None, torch.zeros(1, 7, 8), (2, 2, 2), (1, 1, 1), (2, 2, 2), 1
        )


# --------------------------------------------------------------------------- #
# _mu_raster_perms: MU-major <-> raster permutations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "mu_grid,tok_in_mu",
    [((2, 2), (2, 2)), ((1, 2, 2), (2, 2, 2)), ((2, 1, 3), (1, 2, 2)), ((4,), (3,))],
)
def test_mu_raster_perms_are_mutual_inverses_and_match_raster_walk(mu_grid, tok_in_mu):
    """gather and idmap invert each other, and gather equals a brute-force raster
    walk that computes each position's MU-major id from its (mu, tok) coords."""
    gather, idmap = _mu_raster_perms(mu_grid, tok_in_mu, torch.device("cpu"))
    n_tok = math.prod(mu_grid) * math.prod(tok_in_mu)
    assert torch.equal(idmap[gather], torch.arange(n_tok))
    assert torch.equal(gather[idmap], torch.arange(n_tok))

    D = len(mu_grid)
    full = [mu_grid[i] * tok_in_mu[i] for i in range(D)]
    ref = []
    for flat in range(n_tok):
        coords, r = [], flat
        for f in reversed(full):
            coords.append(r % f)
            r //= f
        coords = list(reversed(coords))
        mu_c = [c // tok_in_mu[i] for i, c in enumerate(coords)]
        tok_c = [c % tok_in_mu[i] for i, c in enumerate(coords)]
        mu_id = 0
        for i in range(D):
            mu_id = mu_id * mu_grid[i] + mu_c[i]
        tok_id = 0
        for i in range(D):
            tok_id = tok_id * tok_in_mu[i] + tok_c[i]
        ref.append(mu_id * math.prod(tok_in_mu) + tok_id)
    assert gather.tolist() == ref


def test_mu_major_sequence_needs_gather_before_deformable_attn():
    """Two-MU toy: a delta scattered into the MU-major sequence is only found
    by a raster-referenced deformable query after the gather permutation."""
    from cell_observatory_platform.models.ops.flash_deform_attn import (
        ms_deform_attn_core_pytorch_3d,
    )

    mu_grid, tok_in_mu = (1, 2, 2), (2, 2, 2)
    full = tuple(mu_grid[i] * tok_in_mu[i] for i in range(3))  # (2, 4, 4)
    n_tok = math.prod(full)
    C, heads = 8, 1

    gather, idmap = _mu_raster_perms(mu_grid, tok_in_mu, torch.device("cpu"))

    m_star = 5
    mu_seq = torch.zeros(1, n_tok, C)
    mu_seq[0, m_star] = 1.0

    raster_pos = int(idmap[m_star])
    z = raster_pos // (full[1] * full[2])
    y = (raster_pos // full[2]) % full[1]
    x = raster_pos % full[2]
    # normalized center, (x, y, z) order per the kernel contract
    ref = torch.tensor([(x + 0.5) / full[2], (y + 0.5) / full[1], (z + 0.5) / full[0]])

    def run(value_seq):
        value = value_seq.view(1, n_tok, heads, C)
        shapes = torch.tensor([list(full)], dtype=torch.long)
        loc = ref.view(1, 1, 1, 1, 1, 3)  # (B, Lq, heads, levels, points, 3)
        w = torch.ones(1, 1, heads, 1, 1)
        return ms_deform_attn_core_pytorch_3d(value, shapes, loc, w).view(-1)

    out_raster = run(mu_seq[:, gather])
    assert out_raster.abs().sum() > 0.5, "raster-converted value map must hit the delta"

    out_mu = run(mu_seq)
    assert out_mu.abs().sum() < 0.5, (
        "MU-major value map interpreted as raster should MISS the delta "
        "(this failing means the toy no longer distinguishes the layouts)"
    )
