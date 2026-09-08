"""Exact-math checks for the pretraining memory/perf fixes (CPU, tiny shapes).

Each test re-computes the ORIGINAL formulation by hand from the same modules and
asserts the optimised path returns the same numbers (up to float rounding):

  1. MaskedEncoder (joint embed): gather patch rows BEFORE the pixel->D projection
  2. MaskedPredictor: gather target rows BEFORE norm + output projection
  3. L1/L2 masked losses as one fused reduction
  4. MaskedEncoder(return_patches=False) -> (tokens, None), same tokens
  5. predictor reorder gather via expand (covered by 2)
  6. SAM dense PE cached on the prompt encoder + stride-0 expand into the transformer
  7. blocked mask starts drawn from the seeded generator
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from cell_observatory_platform.data.data_shapes import MULTICHANNEL_HYPERCUBE
from cell_observatory_platform.data.masking.mask_generator import MaskGenerator, MaskModes, apply_masks
from cell_observatory_platform.models.backbones.maskedencoder import MaskedEncoder
from cell_observatory_platform.models.heads.maskedpredictor import MaskedPredictor
from cell_observatory_platform.models.heads.sam_decoder import TwoWayTransformer
from cell_observatory_platform.models.layers.prompt_encoders import PromptEncoder
from cell_observatory_platform.models.meta_arch.jepa import JEPA
from cell_observatory_platform.models.meta_arch.maskedautoencoder import MaskedAutoEncoder
from cell_observatory_platform.training.helpers import get_masked_input_data
from cell_observatory_platform.training.losses import L1_masked_loss, L2_masked_loss

SHAPE = (16, 32, 32, 2)          # Z, Y, X, C
PATCH = (8, 8, 8, None)
N = 2 * 4 * 4                    # 32 patches
D = 16


def _encoder(sincos=True, rope=False):
    torch.manual_seed(0)
    return MaskedEncoder(
        model_template="me", input_fmt="ZYXC", input_shape=SHAPE, patch_shape=PATCH,
        embed_dim=D, depth=1, num_heads=2, mlp_ratio=2.0, drop_path_rate=0.0,
        abs_sincos_enc=sincos, rope_pos_enc=rope, dtype=torch.float32,
    ).eval()


def _masks(B):
    g = torch.Generator().manual_seed(1)
    return torch.stack([torch.randperm(N, generator=g)[: N // 4].sort().values for _ in range(B)])


def _encoder_reference(enc, x, masks, concat=True):
    """The pre-change forward: project ALL patches, add pos, then gather."""
    tokens, patches = enc.patch_embedding(x, return_patches=True)
    if enc.abs_sincos_enc:
        tokens = tokens + enc.pos_embedding(x)
    if masks is not None:
        tokens = apply_masks(tokens, masks, concat=concat)
    tokens = enc.encoder(tokens, masks=masks, pos_enc=enc.freqs_cis)
    return enc.norm(tokens), patches


# ---- 1 / 4: encoder gathers before projecting; return_patches=False ------------


def test_encoder_gather_before_project_matches_project_then_gather():
    for sincos, rope in [(True, False), (False, True), (True, True)]:
        enc = _encoder(sincos, rope)
        x = torch.randn(2, *SHAPE)
        masks = _masks(2)

        ref, ref_patches = _encoder_reference(enc, x, masks)
        out, patches = enc(x, masks=masks)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(patches, ref_patches)          # raw pixels: unchanged, full
        assert patches.shape == (2, N, enc.patch_embedding.pixels_per_patch)

        # list-of-masks (concat) path of the embed step (the transformer stack
        # itself only takes a single [B, K] mask): gather-then-project == project-then-gather
        mask_list = [masks, torch.stack([torch.arange(0, N, 4), torch.arange(1, N, 4)])]
        pe = enc.patch_embedding
        ref_tok, _ = pe(x, return_patches=True)
        if sincos:
            ref_tok = ref_tok + enc.pos_embedding(x)
        ref_tok = apply_masks(ref_tok, mask_list, concat=True)
        out_tok = pe.proj(apply_masks(pe._patchify(x), mask_list, concat=True))
        if sincos:
            out_tok = out_tok + apply_masks(enc.pos_embedding(x).expand(2, -1, -1), mask_list, concat=True)
        assert out_tok.shape == (4, masks.shape[1], D)
        torch.testing.assert_close(out_tok, ref_tok, rtol=1e-5, atol=1e-5)

        # no masks: unchanged
        ref, _ = _encoder_reference(enc, x, None)
        out, _ = enc(x)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_encoder_return_patches_false_returns_none_and_same_tokens():
    enc = _encoder()
    x = torch.randn(2, *SHAPE)
    masks = _masks(2)
    full, patches = enc(x, masks=masks, return_patches=True)
    slim, none = enc(x, masks=masks, return_patches=False)
    assert none is None and patches is not None
    torch.testing.assert_close(slim, full)
    # forward_features unpacks the pair either way
    torch.testing.assert_close(enc.forward_features(x, masks=masks), full)


def test_encoder_projection_only_sees_context_rows():
    """The Linear's saved input (what autograd keeps) is the gathered rows."""
    enc = _encoder()
    seen = {}
    def hook(m, i, o):
        seen["rows"] = i[0].shape[1]
    enc.patch_embedding.proj.register_forward_hook(hook)
    masks = _masks(2)
    enc(torch.randn(2, *SHAPE), masks=masks)
    assert seen["rows"] == masks.shape[1] < N


# ---- 2 / 5: predictor gathers target rows before norm + output projection -------


def _predictor():
    torch.manual_seed(0)
    return MaskedPredictor(
        model_template="mp", input_fmt="ZYXC", input_shape=SHAPE, patch_shape=PATCH,
        input_embed_dim=D, output_embed_dim=1024, embed_dim=D, depth=1, num_heads=2,
        mlp_ratio=2.0, drop_path_rate=0.0, abs_sincos_enc=True, rope_pos_enc=False,
        dtype=torch.float32,
    ).eval()


def test_predictor_output_masks_equals_gather_after_projection():
    dec = _predictor()
    B, K = 2, N // 4
    ctx = _masks(B)                                                     # [B, K] kept
    tgt = torch.stack([torch.tensor(sorted(set(range(N)) - set(c.tolist()))) for c in ctx])
    orig = torch.argsort(torch.cat([ctx, tgt], dim=1), dim=1)           # [B, N] reorder
    used = torch.arange(N).unsqueeze(0).expand(B, -1)
    idx = torch.searchsorted(used, tgt)
    h = torch.randn(B, K, D)

    full = dec(h, original_patch_indices=orig, target_masks=tgt, patches_used=used)
    ref = apply_masks(full, idx)
    out = dec(h, original_patch_indices=orig, target_masks=tgt, patches_used=used, output_masks=idx)
    assert out.shape == (B, N - K, 1024)
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)

    seen = {}
    def hook(m, i, o):
        seen["rows"] = i[0].shape[1]
    dec.output_projection.register_forward_hook(hook)
    dec(h, original_patch_indices=orig, target_masks=tgt, patches_used=used, output_masks=idx)
    assert seen["rows"] == N - K

    # 1-D original_patch_indices (training/helpers.get_masked_input_data) still work
    one = dec(h[:1], original_patch_indices=orig[0], target_masks=tgt[:1], patches_used=used[:1])
    torch.testing.assert_close(one, full[:1], rtol=1e-6, atol=1e-6)


# ---- 3: losses ------------------------------------------------------------------


def _old_l2(t, p, n):
    return ((t - p) ** 2).mean(dim=-1).sum() / n


def _old_l1(t, p, n):
    return torch.abs(t - p).mean(dim=-1).sum() / n


def test_masked_losses_match_old_formulation():
    torch.manual_seed(0)
    t, p = torch.randn(3, 24, 1024), torch.randn(3, 24, 1024)
    n = torch.tensor(24 * 3)
    torch.testing.assert_close(L2_masked_loss(t, p, n)[0], _old_l2(t, p, n), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(L1_masked_loss(t, p, n)[0], _old_l1(t, p, n), rtol=1e-6, atol=1e-6)
    # list form (multiscale) with different last dims per level
    ts, ps = [t, torch.randn(3, 8, 64)], [p, torch.randn(3, 8, 64)]
    ref2 = (_old_l2(ts[0], ps[0], 1) + _old_l2(ts[1], ps[1], 1)) / 96
    ref1 = (_old_l1(ts[0], ps[0], 1) + _old_l1(ts[1], ps[1], 1)) / 96
    torch.testing.assert_close(L2_masked_loss(ts, ps, 96)[0], ref2, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(L1_masked_loss(ts, ps, 96)[0], ref1, rtol=1e-6, atol=1e-6)
    # mixed dtypes promote like the subtraction did (fp32 targets, bf16 predictions)
    pb = p.to(torch.bfloat16)
    torch.testing.assert_close(L2_masked_loss(t, pb, n)[0], _old_l2(t, pb, n), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(L1_masked_loss(t, pb, n)[0], _old_l1(t, pb, n), rtol=1e-5, atol=1e-5)
    # gradient wrt predictions is unchanged
    p1 = p.clone().requires_grad_(True)
    p2 = p.clone().requires_grad_(True)
    L2_masked_loss(t, p1, n)[0].backward()
    _old_l2(t, p2, n).backward()
    torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-6, atol=1e-6)


# ---- end to end: MAE / JEPA forward equals the pre-change composition ------------


def _mae():
    torch.manual_seed(0)
    return MaskedAutoEncoder(
        model_template="mae", input_fmt="ZYXC", input_shape=SHAPE, patch_shape=PATCH,
        embed_dim=D, decoder_embed_dim=D, depth=1, decoder_depth=1, num_heads=2,
        decoder_num_heads=2, drop_path_rate=0.0, abs_sincos_enc=True, rope_pos_enc=False,
        dtype=torch.float32, buffer_device="cpu",
    ).eval()


def test_mae_forward_matches_reference_composition():
    model = _mae()
    (sample,) = get_masked_input_data(model, (1, *SHAPE), device="cpu", mask_ratio=0.75)  # helper builds B=1 masks
    meta, x = sample["metainfo"], sample["data_tensor"]
    ctx, tgt = meta["context_masks"][0], meta["target_masks"][0]
    used, orig = meta["patches_used"][0], meta["original_patch_indices"][0]

    enc, dec = model.masked_encoder, model.masked_decoder
    h, patches = _encoder_reference(enc, x, ctx)
    full = dec(h, original_patch_indices=orig, target_masks=tgt, patches_used=used)
    idx = torch.searchsorted(used, tgt)
    targets, preds = apply_masks(patches, tgt), apply_masks(full, idx)
    ref_loss = _old_l2(targets, preds, meta["masks"][0].sum())

    loss_dict, predictions = model(sample)
    torch.testing.assert_close(predictions, preds, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(loss_dict["step_loss"], ref_loss, rtol=1e-5, atol=1e-5)


def test_jepa_forward_matches_reference_composition():
    torch.manual_seed(0)
    model = JEPA(
        input_fmt="ZYXC", input_shape=SHAPE, patch_shape=PATCH, embed_dim=D,
        predictor_embed_dim=D, depth=1, predictor_depth=1, num_heads=2, predictor_num_heads=2,
        drop_path_rate=0.0, abs_sincos_enc=True, rope_pos_enc=False, dtype=torch.float32,
        buffer_device="cpu",
    ).eval()
    (sample,) = get_masked_input_data(model, (1, *SHAPE), device="cpu", mask_ratio=0.75)  # helper builds B=1 masks
    meta, x = sample["metainfo"], sample["data_tensor"]
    ctx, tgt = meta["context_masks"][0], meta["target_masks"][0]
    used, orig = meta["patches_used"][0], meta["original_patch_indices"][0]

    h, _ = _encoder_reference(model.input_encoder, x, ctx)
    full = model.target_predictor(h, original_patch_indices=orig, target_masks=tgt, patches_used=used)
    with torch.no_grad():
        t_full, _ = _encoder_reference(model.target_encoder, x, None)
    idx = torch.searchsorted(used, tgt)
    targets, preds = apply_masks(t_full, tgt), apply_masks(full, idx)
    ref_loss = _old_l1(targets, preds, meta["masks"][0].sum())

    loss_dict, predictions = model(sample)
    torch.testing.assert_close(predictions, preds, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(loss_dict["step_loss"], ref_loss, rtol=1e-5, atol=1e-5)


# ---- 6: SAM dense PE cache + expand -----------------------------------------------


def test_dense_pe_is_cached_and_exact():
    torch.manual_seed(0)
    pe = PromptEncoder(
        embed_dim=32, mask_in_chans=16, mask_downsample_factor=4,
        input_shape=[1, 32, 64, 64, 2], patch_shape=[1, 16, 16, None], input_format="TZYXC",
    )
    ref = pe.pe_layer(pe.token_shape).unsqueeze(0)
    a = pe.get_dense_pe()
    torch.testing.assert_close(a, ref)
    assert pe.get_dense_pe() is a                              # cached, no recompute
    b = pe.get_dense_pe(dtype=torch.bfloat16)
    assert b.dtype == torch.bfloat16 and pe.get_dense_pe(dtype=torch.bfloat16) is b
    torch.testing.assert_close(b, ref.to(torch.bfloat16))
    assert "_dense_pe_cache" not in pe.state_dict() and not any("_dense_pe" in k for k in pe.state_dict())
    pe.token_shape = [1, 2, 2]                                 # shape change -> rebuilt
    c = pe.get_dense_pe()
    assert c.shape[2:] == (1, 2, 2) and len(pe._dense_pe_cache) == 1


def test_two_way_transformer_reads_expanded_pe_like_repeat_interleave():
    torch.manual_seed(0)
    tr = TwoWayTransformer(depth=1, embedding_dim=32, num_heads=4, mlp_dim=32, input_fmt="TZYXC").eval()
    B, c, z, y, x = 3, 32, 2, 2, 2
    src, tokens = torch.randn(B, c, z, y, x), torch.randn(B, 5, c)
    image_pe = torch.randn(1, c, z, y, x)
    rep = torch.repeat_interleave(image_pe, B, dim=0)
    exp = image_pe.expand(B, *image_pe.shape[1:])
    snapshot = image_pe.clone()
    hs_r, src_r = tr(src, rep, tokens)
    hs_e, src_e = tr(src, exp, tokens)
    torch.testing.assert_close(hs_e, hs_r)
    torch.testing.assert_close(src_e, src_r)
    torch.testing.assert_close(image_pe, snapshot)               # nothing wrote into the PE


# ---- 7: blocked mask block starts come from the seeded generator ------------------


def test_blocked_mask_starts_use_seeded_generator():
    def make():
        return MaskGenerator(
            layout=MULTICHANNEL_HYPERCUBE.TZYXC, input_format="TZYXC",
            input_shape=(4, 32, 32, 32, 2), patch_shape=(1, 8, 8, 8),
            mask_mode=MaskModes.BLOCKED, num_blocks=2,
            temporal_mask_scale=(0.3, 0.6), axial_mask_scale=(0.3, 0.6),
            lateral_mask_scale=(0.3, 0.6), aspect_ratio_scale_hw=(1.0, 1.0),
            device=torch.device("cpu"),
        )
    a, b = make(), make()
    torch.manual_seed(0)
    out_a = a(batch_size=2)
    torch.manual_seed(12345)                                     # different GLOBAL rng state
    out_b = b(batch_size=2)
    assert out_a["masks"].any() and not out_a["masks"].all()
    for k in ("masks", "context_masks", "target_masks"):
        va, vb = out_a[k], out_b[k]
        va = va if torch.is_tensor(va) else va[0]
        vb = vb if torch.is_tensor(vb) else vb[0]
        assert torch.equal(va, vb), k
