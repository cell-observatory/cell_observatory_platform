import pytest
import torch

from cell_observatory_platform.models.backbones.convnext import CONFIGS, ConvNeXtV2

# stem /4 then three /2 downsamples -> 32^3 is the smallest input that survives all 4 stages
_SMALL = dict(input_fmt="ZYXC", input_shape=(32, 32, 32, 2), modes=5, drop_path_rate=0.0)


@pytest.mark.parametrize(
    "model_template,return_stage_features",
    [("convnext", False), ("convnext", True), ("convnext-tiny", False)],
)
def test_convnext_forward_shapes(model_template, return_stage_features):
    """A named template overrides explicit depths/dims; the regressor emits (B, modes)
    and forward_features emits one channels-first map per stage at /4, /8, /16, /32."""
    torch.manual_seed(0)
    model = ConvNeXtV2(
        model_template=model_template,
        depths=(1, 1, 1, 1),
        dims=(8, 16, 32, 64),
        return_stage_features=return_stage_features,
        **_SMALL,
    ).eval()
    if model_template in CONFIGS:  # template overrides the explicit depths/dims
        assert model.depths == CONFIGS[model_template]["depths"]
        assert model.dims == CONFIGS[model_template]["dims"]
    else:
        assert model.depths == (1, 1, 1, 1) and model.dims == (8, 16, 32, 64)
    B = 2
    x = torch.randn(B, *_SMALL["input_shape"])  # platform-native channels-last (B, Z, Y, X, C)

    with torch.no_grad():
        if return_stage_features:
            feats = model.forward_features(x)
            expected = [(B, d, s, s, s) for d, s in zip(model.dims, (8, 4, 2, 1))]
            assert [tuple(f.shape) for f in feats] == expected
            assert all(torch.isfinite(f).all() for f in feats)
        else:
            out = model({"data_tensor": x, "metainfo": {}})
            assert out.shape == (B, _SMALL["modes"])
            assert torch.isfinite(out).all()
