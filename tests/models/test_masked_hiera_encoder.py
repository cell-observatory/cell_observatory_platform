import pytest
import torch

from cell_observatory_platform.models.backbones.masked_hiera_encoder import MaskedHieraEncoder

ZYXC_CFG = dict(
    input_fmt="ZYXC",
    input_shape=(128, 128, 128, 1),
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
    input_shape=(16, 128, 128, 128, 1),
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


@pytest.mark.parametrize("cfg", [ZYXC_CFG, TZYXC_CFG], ids=["ZYXC", "TZYXC"])
def test_encoder_forward_no_proj(cfg):
    enc = MaskedHieraEncoder(**cfg, channel_proj_type="none")
    B = 2
    x = torch.randn((B,) + cfg["input_shape"])

    out, patches = enc(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == B
    final_dim = enc.encoder.blocks[-1].dim_out
    assert out.shape[-1] == final_dim


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
    enc = MaskedHieraEncoder(**cfg, channel_proj_type="none")
    spec = enc.get_decoder_spec()
    assert "mu_grid" in spec
    assert "tok_in_mu" in spec
    assert "pixels_per_patch" in spec
    assert "mu_window_patches" in spec
