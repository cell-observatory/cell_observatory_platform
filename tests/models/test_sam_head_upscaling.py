"""Decoder upscaling as Linear + depth-to-space (exact equivalence).

A `ConvTranspose3d` whose kernel equals its stride has no overlap between output
blocks: each input token is mapped by one linear map onto its own s^3 output
block. `UpShuffle3d` implements exactly that as a single GEMM followed by a
depth-to-space shuffle, channels-last in and out.

These tests pin the equivalence (fp32, allclose 1e-5) at three levels:
  1. the layer itself vs a random `nn.ConvTranspose3d(k=s=2)`, both stages;
  2. the full `MaskDecoder.predict_masks` output, with and without
     `use_high_res_features`;
  3. the default config path is unchanged (`upscaling="conv"`).

CPU-only, tiny shapes.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from cell_observatory_platform.models.heads.sam_head import (
    MaskDecoder,
    UpShuffle3d,
    convert_output_upscaling_state_dict,
)


# --------------------------------------------------------------------------- #
# 1) layer-level equivalence
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("c_in,c_out", [(16, 4), (4, 2)])
def test_upshuffle3d_matches_conv_transpose(c_in, c_out):
    """UpShuffle3d.load_from_conv_transpose reproduces ConvTranspose3d(k=s=2)."""
    torch.manual_seed(0)
    s = 2
    conv = nn.ConvTranspose3d(c_in, c_out, kernel_size=(s, s, s), stride=(s, s, s))
    # randomize the bias so a wrong bias broadcast cannot hide behind ~0
    with torch.no_grad():
        conv.bias.normal_()

    up = UpShuffle3d(c_in, c_out, s=s)
    up.load_from_conv_transpose(conv)

    B, Z, Y, X = 2, 3, 2, 4
    x_cl = torch.randn(B, Z, Y, X, c_in)            # channels-last
    x_cf = x_cl.permute(0, 4, 1, 2, 3).contiguous()  # channels-first

    ref = conv(x_cf).permute(0, 2, 3, 4, 1)          # -> channels-last
    got = up(x_cl)

    assert got.shape == (B, Z * s, Y * s, X * s, c_out)
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5), (
        f"max abs diff {(ref - got).abs().max().item()}"
    )


def test_upshuffle3d_bias_is_broadcast_to_every_output_voxel():
    """A zero-weight conv must reduce to a per-channel constant everywhere."""
    torch.manual_seed(1)
    c_in, c_out, s = 5, 3, 2
    conv = nn.ConvTranspose3d(c_in, c_out, kernel_size=(s, s, s), stride=(s, s, s))
    with torch.no_grad():
        conv.weight.zero_()
        conv.bias.copy_(torch.tensor([1.0, -2.0, 3.5]))

    up = UpShuffle3d(c_in, c_out, s=s)
    up.load_from_conv_transpose(conv)

    out = up(torch.randn(1, 2, 2, 2, c_in))
    expected = torch.tensor([1.0, -2.0, 3.5]).view(1, 1, 1, 1, c_out).expand_as(out)
    assert torch.allclose(out, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# 2) full decoder equivalence
# --------------------------------------------------------------------------- #

def _make_decoder(upscaling: str, use_high_res_features: bool, transformer_dim: int = 32):
    return MaskDecoder(
        input_fmt="TZYXC",
        mask_downsample_factor=4,
        transformer_dim=transformer_dim,
        transformer_depth=1,
        transformer_num_heads=4,
        transformer_mlp_dim=32,
        num_multimask_outputs=1,
        iou_head_depth=2,
        iou_head_hidden_dim=transformer_dim,
        use_high_res_features=use_high_res_features,
        upscaling=upscaling,
    )


def _decoder_inputs(dec, B=2, z=2, y=2, x=2, seed=0):
    torch.manual_seed(seed)
    c = dec.transformer_dim
    image_embeddings = torch.randn(B, c, z, y, x)
    image_pe = torch.randn(1, c, z, y, x)
    sparse = torch.randn(B, 3, c)
    dense = torch.randn(B, c, z, y, x)
    high_res = [
        torch.randn(B, c // 8, 4 * z, 4 * y, 4 * x),  # feat_s0, stride 1 (4x tokens)
        torch.randn(B, c // 4, 2 * z, 2 * y, 2 * x),  # feat_s1, stride 2 (2x tokens)
    ]
    return image_embeddings, image_pe, sparse, dense, high_res


@pytest.mark.parametrize("use_high_res_features", [False, True])
def test_predict_masks_conv_vs_linear_shuffle(use_high_res_features):
    """Full predict_masks output matches after converting the conv weights."""
    torch.manual_seed(7)
    dec_conv = _make_decoder("conv", use_high_res_features).eval()
    dec_shuf = _make_decoder("linear_shuffle", use_high_res_features).eval()

    # Everything except output_upscaling has identical key names, so copy the
    # shared parameters verbatim and convert only the upscaling stack.
    sd = dec_conv.state_dict()
    converted = convert_output_upscaling_state_dict(sd)
    missing, unexpected = dec_shuf.load_state_dict(converted, strict=False)
    assert not missing, f"unfilled params in linear_shuffle decoder: {missing}"
    assert not unexpected, f"leftover conv keys: {unexpected}"

    args = _decoder_inputs(dec_conv, seed=3)
    image_embeddings, image_pe, sparse, dense, high_res = args
    hr = high_res if use_high_res_features else None

    with torch.no_grad():
        m_conv, iou_conv, tok_conv, obj_conv = dec_conv.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=False,
            high_res_features=hr,
        )
        m_shuf, iou_shuf, tok_shuf, obj_shuf = dec_shuf.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=False,
            high_res_features=hr,
        )

    assert m_conv.shape == m_shuf.shape
    assert torch.allclose(m_conv, m_shuf, atol=1e-5, rtol=1e-5), (
        f"masks max abs diff {(m_conv - m_shuf).abs().max().item()}"
    )
    assert torch.allclose(iou_conv, iou_shuf, atol=1e-5, rtol=1e-5)
    assert torch.allclose(tok_conv, tok_shuf, atol=1e-5, rtol=1e-5)
    assert torch.allclose(obj_conv, obj_shuf, atol=1e-5, rtol=1e-5)


def test_predict_masks_linear_shuffle_accepts_channels_last_high_res_features():
    """high_res_features may arrive channels-last; the result is the same."""
    torch.manual_seed(11)
    dec_conv = _make_decoder("conv", True).eval()
    dec_shuf = _make_decoder("linear_shuffle", True).eval()
    dec_shuf.load_state_dict(
        convert_output_upscaling_state_dict(dec_conv.state_dict()), strict=True
    )

    image_embeddings, image_pe, sparse, dense, high_res = _decoder_inputs(dec_conv, seed=5)
    high_res_cl = [f.permute(0, 2, 3, 4, 1).contiguous() for f in high_res]

    with torch.no_grad():
        ref, _, _, _ = dec_conv.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=False,
            high_res_features=high_res,
        )
        got, _, _, _ = dec_shuf.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=False,
            high_res_features=high_res_cl,
        )
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# 3) default path unchanged
# --------------------------------------------------------------------------- #

def test_default_upscaling_is_conv():
    """The default MaskDecoder still builds the ConvTranspose3d stack."""
    dec = MaskDecoder(
        input_fmt="TZYXC",
        mask_downsample_factor=4,
        transformer_dim=32,
        transformer_depth=1,
        transformer_num_heads=4,
        transformer_mlp_dim=32,
        num_multimask_outputs=1,
        iou_head_depth=2,
        iou_head_hidden_dim=32,
    )
    assert dec.upscaling == "conv"
    assert isinstance(dec.output_upscaling[0], nn.ConvTranspose3d)
    assert isinstance(dec.output_upscaling[3], nn.ConvTranspose3d)
    # state-dict keys are unchanged so existing checkpoints still load
    keys = set(dec.state_dict().keys())
    assert "output_upscaling.0.weight" in keys
    assert "output_upscaling.1.ln.weight" in keys


def test_unknown_upscaling_raises():
    with pytest.raises(ValueError):
        _make_decoder("bogus", False)
