import pytest
import torch
from omegaconf import OmegaConf

# BUILD registration for "masked_vit" is a decorator side effect of this import
import cell_observatory_platform.models.backbones.maskedencoder  # noqa: F401
from cell_observatory_platform.models.backbones.sam_backbone import SAMBackbone


def _tiny_masked_vit_args():
    return OmegaConf.create({
        "name": "masked_vit",
        "input_fmt": "TZYXC",
        "input_shape": [1, 16, 16, 16, 2],
        "patch_shape": [1, 8, 8, 8],
        "model_template": "mae",
        "embed_dim": 32,
        "depth": 1,
        "num_heads": 2,
        "mlp_ratio": 2,
        "abs_sincos_enc": False,
        "rope_pos_enc": False,
    })


def _tiny_sam_backbone():
    return SAMBackbone(
        backbone_args=_tiny_masked_vit_args(),
        adapter_args=None,
        backbone_embed_dims=[33],   # sincos pe splits dim by 3
        train_backbone=True,
        use_layernorm=False,
        backbone_output_format="sequence",
        input_shape=[1, 16, 16, 16, 2],
        patch_shape=[1, 8, 8, 8],
        input_format="TZYXC",
    )


def test_to_backbone_layout_round_trips_and_preserves_tokens():
    """_to_backbone_layout turns the SAM2 (B*T, C, Z, Y, X) tensor back into the
    channels-last (B, T, Z, Y, X, C) input exactly, so the backbone's tokens are
    identical whether it is fed directly or through the wrapper."""
    wrapper = _tiny_sam_backbone()
    assert wrapper.backbone_consumes_channels_last

    x_cl = torch.randn(2, 1, 16, 16, 16, 2)          # (B, T=1, Z, Y, X, C)
    conv = x_cl.permute(0, 1, 5, 2, 3, 4).flatten(0, 1)   # (B*T, C, Z, Y, X)

    restored = wrapper._to_backbone_layout(conv)
    torch.testing.assert_close(restored, x_cl)

    wrapper.backbone.eval()
    with torch.no_grad():
        tokens_direct = wrapper.backbone.forward_features(x_cl)
        tokens_via_wrapper = wrapper.backbone.forward_features(
            wrapper._to_backbone_layout(conv)
        )
    torch.testing.assert_close(tokens_via_wrapper, tokens_direct)


def test_backbone_rejects_channels_first_without_layout_conversion():
    """Feeding the conv layout straight into the backbone trips the patchify
    layout guard instead of silently scrambling tokens."""
    wrapper = _tiny_sam_backbone()
    conv = torch.randn(2, 2, 16, 16, 16)
    with pytest.raises(ValueError, match="channels-last|D input"):
        wrapper.backbone.forward_features(conv)


def test_to_adapter_layout_moves_channels_last_contiguous():
    """_to_adapter_layout maps (B*T, C, Z, Y, X) to a contiguous (B*T, Z, Y, X, C)."""
    w = object.__new__(SAMBackbone)
    x = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(2, 3, 4, 5, 6)
    y = w._to_adapter_layout(x)
    assert y.shape == (2, 4, 5, 6, 3)
    assert y.is_contiguous()
    torch.testing.assert_close(y[1, 2, 3, 4, 1], x[1, 1, 2, 3, 4])
