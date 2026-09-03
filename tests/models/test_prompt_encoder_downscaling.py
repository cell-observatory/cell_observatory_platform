"""Prompt-encoder mask downscaling as space-to-depth + Linear (exact).

`Conv3d` with kernel == stride reads each `s x s x s` block exactly once, so it
is a space-to-depth reshape followed by a per-token linear map. `DownShuffle3d`
implements that; the 1x1x1 conv that ends the stack is a plain `nn.Linear` on
channels-last.

Tests: layer equivalence vs a random `Conv3d(k=s=2)`, full `_embed_masks`
equivalence after weight conversion, and the unchanged default path.

CPU-only, tiny shapes.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from cell_observatory_platform.models.layers.prompt_encoders import (
    DownShuffle3d,
    PromptEncoder,
    convert_mask_downscaling_state_dict,
)


# --------------------------------------------------------------------------- #
# 1) layer-level equivalence
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("c_in,c_out", [(1, 4), (4, 16)])
def test_downshuffle3d_matches_conv3d(c_in, c_out):
    torch.manual_seed(0)
    s = 2
    conv = nn.Conv3d(c_in, c_out, kernel_size=(s, s, s), stride=(s, s, s))
    with torch.no_grad():
        conv.bias.normal_()

    down = DownShuffle3d(c_in, c_out, s=s)
    down.load_from_conv(conv)

    B, Z, Y, X = 2, 4, 2, 6
    x_cl = torch.randn(B, Z, Y, X, c_in)
    x_cf = x_cl.permute(0, 4, 1, 2, 3).contiguous()

    ref = conv(x_cf).permute(0, 2, 3, 4, 1)
    got = down(x_cl)

    assert got.shape == (B, Z // s, Y // s, X // s, c_out)
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5), (
        f"max abs diff {(ref - got).abs().max().item()}"
    )


def test_downshuffle3d_rejects_non_divisible_extent():
    down = DownShuffle3d(1, 4, s=2)
    with pytest.raises(ValueError):
        down(torch.randn(1, 3, 4, 4, 1))


# --------------------------------------------------------------------------- #
# 2) full prompt-encoder equivalence
# --------------------------------------------------------------------------- #

_INPUT_SHAPE = [1, 16, 16, 16, 1]     # T, Z, Y, X, C
_PATCH_SHAPE = [1, 16, 16, 16, None]


def _make_encoder(mask_downscaling: str, embed_dim: int = 12, mask_in_chans: int = 16):
    return PromptEncoder(
        embed_dim=embed_dim,
        mask_in_chans=mask_in_chans,
        mask_downsample_factor=4,
        input_shape=_INPUT_SHAPE,
        patch_shape=_PATCH_SHAPE,
        input_format="TZYXC",
        mask_downscaling=mask_downscaling,
    )


def test_embed_masks_conv_vs_linear_shuffle():
    torch.manual_seed(3)
    enc_conv = _make_encoder("conv").eval()
    enc_shuf = _make_encoder("linear_shuffle").eval()

    converted = convert_mask_downscaling_state_dict(enc_conv.state_dict())
    missing, unexpected = enc_shuf.load_state_dict(converted, strict=False)
    assert not missing, f"unfilled params: {missing}"
    assert not unexpected, f"leftover conv keys: {unexpected}"

    # (K, 1, Z, Y, X) mask logits at the prompt-encoder's mask_input_size
    masks = torch.randn(3, 1, *enc_conv.mask_input_size)

    with torch.no_grad():
        ref = enc_conv._embed_masks(masks)
        got = enc_shuf._embed_masks(masks)

    assert ref.shape == got.shape, f"{ref.shape} != {got.shape}"
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5), (
        f"max abs diff {(ref - got).abs().max().item()}"
    )


def test_forward_with_mask_prompt_conv_vs_linear_shuffle():
    """The whole `forward` (sparse + dense) matches, so the layout round-trip
    back to channels-first is right."""
    torch.manual_seed(4)
    enc_conv = _make_encoder("conv").eval()
    enc_shuf = _make_encoder("linear_shuffle").eval()
    enc_shuf.load_state_dict(
        convert_mask_downscaling_state_dict(enc_conv.state_dict()), strict=True
    )

    K = 2
    points = torch.rand(K, 1, 3) * 8
    labels = torch.ones(K, 1, dtype=torch.int32)
    masks = torch.randn(K, 1, *enc_conv.mask_input_size)

    with torch.no_grad():
        sp_ref, de_ref = enc_conv(points=(points, labels), boxes=None, masks=masks)
        sp_got, de_got = enc_shuf(points=(points, labels), boxes=None, masks=masks)

    assert torch.allclose(sp_ref, sp_got, atol=1e-5, rtol=1e-5)
    assert de_ref.shape == de_got.shape
    assert torch.allclose(de_ref, de_got, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# 3) default path unchanged
# --------------------------------------------------------------------------- #

def test_default_mask_downscaling_is_conv():
    enc = PromptEncoder(
        embed_dim=12,
        mask_in_chans=16,
        mask_downsample_factor=4,
        input_shape=_INPUT_SHAPE,
        patch_shape=_PATCH_SHAPE,
        input_format="TZYXC",
    )
    assert enc.mask_downscaling_mode == "conv"
    assert isinstance(enc.mask_downscaling[0], nn.Conv3d)
    assert isinstance(enc.mask_downscaling[6], nn.Conv3d)
    keys = set(enc.state_dict().keys())
    assert "mask_downscaling.0.weight" in keys
    assert "mask_downscaling.1.ln.weight" in keys
    assert "mask_downscaling.6.weight" in keys


def test_unknown_mask_downscaling_raises():
    with pytest.raises(ValueError):
        _make_encoder("bogus")
