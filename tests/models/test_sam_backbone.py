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


# --------------------------------------------------------------------------- #
# channel_ids threading (channel-adaptive patch embed)
# --------------------------------------------------------------------------- #

_VOCAB = {
    "localization": {"<unk>": 0, "membrane": 1},
    "fluorophore": {"<unk>": 0, "mstaygold": 1, "electra2": 2},
}


def _tiny_channel_adaptive_args(fusion="attn_pool"):
    args = _tiny_masked_vit_args()
    args["patch_embed_type"] = "channel_adaptive"
    args["patch_embed_args"] = {
        "channel_fusion": fusion, "attn_pool_num_heads": 2,
        "channel_embed": "factorized", "channel_vocab": _VOCAB, "vocab_extra_slots": 1,
    }
    return args


def _tiny_sam_backbone_channel_adaptive(fusion="attn_pool"):
    return SAMBackbone(
        backbone_args=_tiny_channel_adaptive_args(fusion),
        adapter_args=None,
        backbone_embed_dims=[33],
        train_backbone=True,
        use_layernorm=False,
        backbone_output_format="sequence",
        input_shape=[1, 16, 16, 16, 2],
        patch_shape=[1, 8, 8, 8],
        input_format="TZYXC",
    )


def test_channel_ids_are_expanded_over_frames_and_reach_the_backbone():
    """Per-video ids [B, C, 2] are repeated over the T frames SAM2 flattens into
    the batch (frame order b*T + t) and change the features; a joint backbone
    ignores them."""
    wrapper = _tiny_sam_backbone_channel_adaptive().eval()
    B, T = 2, 2
    conv = torch.randn(B * T, 2, 16, 16, 16)                     # (B*T, C, Z, Y, X)
    ids = torch.tensor([[[1, 1], [0, 2]], [[1, 2], [1, 1]]])     # [B, C, 2]
    with torch.no_grad():
        out = wrapper({"data_tensor": conv, "metainfo": {"channel_ids": ids}})
        other = wrapper({"data_tensor": conv, "metainfo": {"channel_ids": torch.zeros_like(ids)}})
    feat = out["backbone_fpn"][0]
    assert feat.shape == (B * T, 32, 2, 2, 2)      # encoder width; no sam_channel_projection here
    assert not torch.allclose(feat, other["backbone_fpn"][0])
    # frame b*T + t uses video b's ids: swapping the two videos' ids swaps frame features
    with torch.no_grad():
        swapped = wrapper({"data_tensor": conv, "metainfo": {"channel_ids": ids.flip(0)}})["backbone_fpn"][0]
    direct = wrapper._channel_ids_for({"metainfo": {"channel_ids": ids}}, conv)
    assert direct.shape == (B * T, 2, 2) and torch.equal(direct[1], ids[0]) and torch.equal(direct[2], ids[1])
    assert not torch.allclose(swapped[0], feat[0])


def test_channel_ids_batch_must_divide_frames():
    wrapper = _tiny_sam_backbone_channel_adaptive()
    with pytest.raises(ValueError, match="does not divide"):
        wrapper._channel_ids_for(
            {"metainfo": {"channel_ids": torch.zeros(3, 2, 2, dtype=torch.long)}},
            torch.randn(4, 2, 16, 16, 16),
        )


def test_joint_backbone_ignores_channel_ids():
    wrapper = _tiny_sam_backbone().eval()
    conv = torch.randn(1, 2, 16, 16, 16)
    with torch.no_grad():
        a = wrapper({"data_tensor": conv})["backbone_fpn"][0]
        b = wrapper({"data_tensor": conv, "metainfo": {"channel_ids": torch.tensor([[[1, 1], [0, 2]]])}})["backbone_fpn"][0]
    torch.testing.assert_close(a, b)


def test_concat_fusion_is_averaged_over_channels_before_unpatchify():
    """concat yields N*C tokens; SAM2 needs one (B, D, z, y, x) map, so the
    wrapper averages the C tokens of each patch."""
    wrapper = _tiny_sam_backbone_channel_adaptive(fusion="concat").eval()
    conv = torch.randn(1, 2, 16, 16, 16)
    ids = torch.tensor([[[1, 1], [0, 2]]])
    with torch.no_grad():
        tokens = wrapper.backbone.forward_features(wrapper._to_backbone_layout(conv), channel_ids=ids)
        feat = wrapper({"data_tensor": conv, "metainfo": {"channel_ids": ids}})["backbone_fpn"][0]
    assert tokens.shape == (1, 8 * 2, 32)
    assert feat.shape == (1, 32, 2, 2, 2)
    expected = tokens.view(1, 8, 2, 32).mean(2)                       # (B, N, D) before projection
    got = wrapper._unpatchify_if_sequence([tokens])[0]
    torch.testing.assert_close(got, expected.transpose(1, 2).reshape(1, 32, 2, 2, 2))
