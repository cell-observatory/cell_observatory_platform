from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from cell_observatory_platform.models.backbones.masked_hiera_encoder import BUILD, MaskedHieraEncoder

ZYXC_CFG = dict(
    input_fmt="ZYXC",
    input_shape=(64, 64, 64, 1),
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

TZYXC_CFG = dict(
    input_fmt="TZYXC",
    input_shape=(8, 64, 64, 64, 1),
    patch_shape=(2, 4, 8, 8, None),
    embed_dim=32,
    num_heads=2,
    drop_path_rate=0.0,
    q_pool=1,
    q_stride=(1, 2, 2, 2),
    stages=(1, 1, 1, 1),
    norm_layer="LayerNorm",
    mask_unit_size=(1, 2, 2, 2),
)

# tokens after patchify: ZYXC (16, 8, 8) = 1024; TZYXC (4, 16, 8, 8) = 4096; one q_pool of stride 2^3 -> /8
EXPECTED = {
    "ZYXC": dict(final_tokens=128, mu_grid=(8, 4, 4), mu_window=(2, 2, 2), tok_in_mu=(1, 1, 1), ppp=4 * 8 * 8),
    "TZYXC": dict(final_tokens=512, mu_grid=(4, 8, 4, 4), mu_window=(1, 2, 2, 2), tok_in_mu=(1, 1, 1, 1), ppp=2 * 4 * 8 * 8),
}


@pytest.mark.parametrize("cfg", [ZYXC_CFG, TZYXC_CFG], ids=["ZYXC", "TZYXC"])
def test_encoder_forward_no_proj(cfg):
    """Without a channel projection the encoder emits one finite token per pooled mask-unit cell."""
    enc = MaskedHieraEncoder(**cfg, channel_proj_type="none")
    x = torch.randn((2,) + cfg["input_shape"])
    out, patches = enc(x)
    exp = EXPECTED[cfg["input_fmt"]]
    assert out.shape == (2, exp["final_tokens"], enc.encoder.blocks[-1].dim_out)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("cfg", [ZYXC_CFG, TZYXC_CFG], ids=["ZYXC", "TZYXC"])
def test_encoder_forward_fusion(cfg):
    out_dim = 48
    enc = MaskedHieraEncoder(
        **cfg,
        channel_proj_type="fusion",
        multiscale_out_dim=out_dim,
    )
    B = 2
    x = torch.randn((B,) + cfg["input_shape"])

    out, patches = enc(x, return_windowed=True)
    assert isinstance(out, torch.Tensor)
    final_dim = enc.encoder.blocks[-1].dim_out
    assert out.shape[-1] == final_dim


@pytest.mark.parametrize("cfg", [ZYXC_CFG, TZYXC_CFG], ids=["ZYXC", "TZYXC"])
def test_encoder_forward_equalization(cfg):
    out_dim = 48
    enc = MaskedHieraEncoder(
        **cfg,
        channel_proj_type="equalization",
        multiscale_out_dim=out_dim,
    )
    B = 2
    x = torch.randn((B,) + cfg["input_shape"])

    out_list, patches = enc(x)
    assert isinstance(out_list, list)
    for t in out_list:
        assert t.shape[-1] == out_dim


@pytest.mark.parametrize("cfg", [ZYXC_CFG, TZYXC_CFG], ids=["ZYXC", "TZYXC"])
def test_decoder_spec(cfg):
    """The decoder spec pins the mask-unit grid, window, tokens-per-unit, pixels-per-patch and channels."""
    spec = MaskedHieraEncoder(**cfg, channel_proj_type="none").get_decoder_spec()
    exp = EXPECTED[cfg["input_fmt"]]
    assert spec["mu_grid"] == exp["mu_grid"]
    assert spec["mu_window_patches"] == exp["mu_window"]
    assert spec["tok_in_mu"] == exp["tok_in_mu"]
    assert spec["pixels_per_patch"] == exp["ppp"]
    assert spec["in_chans"] == 1


# --------------------------------------------------------------------------- #
# fusion requires windowed intermediates
# --------------------------------------------------------------------------- #


def test_encoder_fusion_requires_windowed_intermediates():
    """channel_proj_type='fusion' consumes [B, N, *mu_shape, C] intermediates, so
    return_windowed=False is refused."""
    enc = MaskedHieraEncoder(**ZYXC_CFG, channel_proj_type="fusion", multiscale_out_dim=48)
    x = torch.randn((1,) + ZYXC_CFG["input_shape"])
    with pytest.raises(AssertionError, match="return_windowed=True"), torch.no_grad():
        enc(x, with_intermediates=True, with_fusion_heads=True, return_windowed=False)


# --------------------------------------------------------------------------- #
# forward_features (wrapper contract) and BUILD
# --------------------------------------------------------------------------- #

SMALL_CFG = dict(
    input_fmt="ZYXC",
    input_shape=(32, 64, 64, 1),   # tokens (8, 8, 8)
    patch_shape=(4, 8, 8, None),
    embed_dim=16,
    num_heads=1,
    drop_path_rate=0.0,
    q_pool=2,
    q_stride=(2, 2, 2),
    stages=(1, 1, 1, 1),
    norm_layer="LayerNorm",
    mask_unit_size=(2, 2, 2),
)


def _small_cfg_as_config(**extra):
    return OmegaConf.create({
        "name": "masked_hiera",
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in SMALL_CFG.items()},
        "channel_proj_type": "equalization",
        "multiscale_out_dim": 24,
        **extra,
    })


def test_forward_features_equalization_returns_channels_first_pyramid():
    """With equalization and q_pool=2, forward_features returns three channels-first
    maps, finest first, each with multiscale_out_dim channels."""
    enc = MaskedHieraEncoder(**SMALL_CFG, channel_proj_type="equalization", multiscale_out_dim=24)
    x = torch.randn(2, *SMALL_CFG["input_shape"])
    feats = enc.forward_features(x)
    assert isinstance(feats, list) and len(feats) == 3   # q_pool=2 -> 3 levels
    grids = [(8, 8, 8), (4, 4, 4), (2, 2, 2)]
    for feat, grid in zip(feats, grids):
        assert feat.shape == (2, 24, *grid)              # [B, C, Z, Y, X]


@pytest.mark.parametrize("proj", ["none", "fusion"])
def test_forward_features_requires_equalization_mode(proj):
    """forward_features is only defined for the equalization (multiscale) projection."""
    enc = MaskedHieraEncoder(**SMALL_CFG, channel_proj_type=proj, multiscale_out_dim=24)
    with pytest.raises(NotImplementedError, match="equalization"):
        enc.forward_features(torch.randn(1, *SMALL_CFG["input_shape"]))


def test_build_passes_channel_proj_type_and_derives_with_intermediates():
    """BUILD is a plain passthrough of the current config keys; with_intermediates
    is derived from channel_proj_type rather than configured."""
    enc = BUILD(_small_cfg_as_config())
    assert enc.channel_proj_type == "equalization"
    assert enc.with_intermediates is True
    feats = enc.forward_features(torch.randn(1, *SMALL_CFG["input_shape"]))
    assert len(feats) == 3


@pytest.mark.parametrize(
    "stale_key", ["return_multiscale", "multiscale_out_indices", "return_intermediates"]
)
def test_build_rejects_unknown_config_key(stale_key):
    """An unknown config key fails loudly at construction (Hiera's signature is
    closed) with the offending key named, instead of silently building a
    single-scale backbone."""
    with pytest.raises(TypeError, match=stale_key):
        BUILD(_small_cfg_as_config(**{stale_key: True}))


_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "models" / "backbones"


def test_shipped_maskdino_hiera_wrapper_config_emits_three_equalized_scales():
    """The shipped masked_hiera_encoder/large_multiscale.yaml (channel_proj_type=
    equalization, q_pool=2) wired through maskdino_backbone_hiera_multiscale.yaml
    builds via REGISTRY with only SIZE overrides and returns three channels-first
    maps, finest first, each with multiscale_out_dim channels."""
    from cell_observatory_platform.utils.registry import REGISTRY
    import cell_observatory_platform.models.backbones.maskdino_backbone  # noqa: F401 (BUILD registration)

    hiera_yaml = OmegaConf.load(_CONFIGS / "masked_hiera_encoder" / "large_multiscale.yaml")
    tiny = OmegaConf.merge(hiera_yaml, OmegaConf.create({
        "input_fmt": "ZYXC", "input_shape": [32, 64, 64, 1], "patch_shape": [4, 8, 8, None],
        "embed_dim": 16, "num_heads": 1, "stages": [1, 1, 1, 1],
        "mask_unit_size": [2, 2, 2], "multiscale_out_dim": 24,
    }))
    wrapper_yaml = OmegaConf.load(_CONFIGS / "maskdino_backbone" / "maskdino_backbone_hiera_multiscale.yaml")
    wrapper_cfg = OmegaConf.merge(wrapper_yaml, OmegaConf.create({
        "backbone_args": tiny, "backbone_embed_dims": [24, 24, 24],
        "input_shape": [32, 64, 64, 1], "patch_shape": [4, 8, 8, None], "input_format": "ZYXC",
    }))
    wrapper = REGISTRY.build("backbone", wrapper_cfg.name, wrapper_cfg)
    assert wrapper.backbone.channel_proj_type == "equalization"

    with torch.no_grad():
        out = wrapper({"data_tensor": torch.randn(1, 32, 64, 64, 1)})
    assert sorted(out) == ["1", "2", "3"]                         # finest -> coarsest
    for key, grid in zip(("1", "2", "3"), ((8, 8, 8), (4, 4, 4), (2, 2, 2))):
        assert out[key].shape == (1, 24, *grid), (key, out[key].shape)
        assert torch.isfinite(out[key]).all()
