import pytest
import torch

from cell_observatory_platform.models.heads.linear_head import LinearHead, LinearProbe, _extract_model_kwargs


@pytest.mark.parametrize("B,L,in_dim,out_dim,bottleneck_dim", [(2, 5, 128, 48, 64), (3, 7, 256, 96, 128)])
def test_linear_head_default_shapes(B, L, in_dim, out_dim, bottleneck_dim):
    head = LinearHead(
        in_dim=in_dim,
        output_dim=out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2 * bottleneck_dim,
        bottleneck_dim=bottleneck_dim,
        mlp_bias=True,
    )
    with torch.no_grad():
        y = head(torch.randn(B, L, in_dim))
    assert y.shape == (B, L, out_dim) and torch.isfinite(y).all()


@pytest.mark.parametrize("B,L,in_dim,bottleneck_dim", [(1, 5, 64, 32), (2, 7, 128, 64)])
def test_linear_head_no_last_layer_returns_l2_normalized_bottleneck(B, L, in_dim, bottleneck_dim):
    """With no_last_layer=True the head returns the MLP bottleneck, L2-normalised per token."""
    head = LinearHead(
        in_dim=in_dim,
        output_dim=11,
        use_bn=False,
        nlayers=2,
        hidden_dim=2 * bottleneck_dim,
        bottleneck_dim=bottleneck_dim,
        mlp_bias=True,
    )
    with torch.no_grad():
        y = head(torch.randn(B, L, in_dim), no_last_layer=True)
    assert y.shape == (B, L, bottleneck_dim)
    torch.testing.assert_close(y.norm(dim=-1), torch.ones(B, L))  # bottleneck is unit-norm


@pytest.mark.parametrize("B,L,out_dim,bottleneck_dim", [(2, 4, 30, 64), (1, 9, 50, 128)])
def test_linear_head_only_last_layer_shapes(B, L, out_dim, bottleneck_dim):
    """With only_last_layer=True the MLP and the normalisation are skipped entirely."""
    head = LinearHead(
        in_dim=999,
        output_dim=out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2 * bottleneck_dim,
        bottleneck_dim=bottleneck_dim,
        mlp_bias=True,
    )
    x = torch.randn(B, L, bottleneck_dim)
    with torch.no_grad():
        y = head(x, only_last_layer=True)
    assert y.shape == (B, L, out_dim)
    torch.testing.assert_close(y, head.last_layer(x))  # MLP and normalisation are skipped


def test_linear_probe_forward_and_extract_kwargs_filters_by_cls():
    """LinearProbe is a single Linear; _extract_model_kwargs maps input_dim -> in_dim and
    keeps only the ctor args of the class being built."""
    probe = LinearProbe(in_dim=8, output_dim=4)
    out = probe(torch.randn(2, 8))
    assert out.shape == (2, 4)

    # allowed-args filtering must come from the class being built
    cfg = {"input_dim": 8, "output_dim": 4, "nlayers": 3, "hidden_dim": 32}
    kwargs = _extract_model_kwargs(cfg, cls=LinearProbe)
    assert kwargs == {"in_dim": 8, "output_dim": 4}
    assert "nlayers" not in kwargs and "hidden_dim" not in kwargs
    LinearProbe(**kwargs)
