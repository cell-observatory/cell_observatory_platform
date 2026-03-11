import math

import pytest
import torch

from cell_observatory_platform.models.backbones.masked_hiera_encoder import MaskedHieraEncoder
from cell_observatory_platform.models.heads.masked_hiera_predictor import MaskedHieraPredictor


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
