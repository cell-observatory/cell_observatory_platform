import pytest
import torch
from torch import nn

from cell_observatory_platform.models.heads.plain_detr_transformer_encoder import (
    TransformerEncoder,
    TransformerEncoderLayer,
)


@pytest.mark.parametrize("normalize_before", [False, True])
def test_transformer_encoder_layer_forward_shapes(normalize_before):
    torch.manual_seed(0)

    d_model = 256
    nhead = 8

    B = 2
    D, H, W = 16, 16, 16
    S = D * H * W  # sequence length

    layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=normalize_before,
    )

    src = torch.randn(B, S, d_model, requires_grad=True)
    pos = torch.randn(B, S, d_model)

    # Optional masks
    src_mask = None  # could also test a [S, S] mask
    src_key_padding_mask = torch.zeros(B, S, dtype=torch.bool)  # no padding

    out = layer(
        src,
        src_mask=src_mask,
        src_key_padding_mask=src_key_padding_mask,
        pos=pos,
    )

    # Shape should be preserved
    assert out.shape == (B, S, d_model)

    # Backward should work
    loss = out.sum()
    loss.backward()
    assert src.grad is not None
    assert src.grad.shape == src.shape


def test_transformer_encoder_stack_shapes():
    torch.manual_seed(0)

    d_model = 256
    nhead = 8
    num_layers = 3

    B = 2
    D, H, W = 8, 8, 8  # slightly smaller to keep it light
    S = D * H * W

    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=512,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    )
    encoder_norm = nn.LayerNorm(d_model)
    encoder = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)

    src = torch.randn(B, S, d_model, requires_grad=True)
    pos = torch.randn(B, S, d_model)

    # key_padding_mask: mark last few tokens as padded in one example
    src_key_padding_mask = torch.zeros(B, S, dtype=torch.bool)
    src_key_padding_mask[0, -10:] = True

    out = encoder(
        src,
        mask=None,
        src_key_padding_mask=src_key_padding_mask,
        pos=pos,
    )

    # Shape preserved
    assert out.shape == (B, S, d_model)

    assert not torch.allclose(out.mean(), src.mean())

    # Backprop check
    loss = out.pow(2).mean()
    loss.backward()
    assert src.grad is not None
    assert src.grad.shape == src.shape


def test_transformer_encoder_no_pos_no_mask():
    """
    Minimal smoke test: no pos embedding, no masks.
    """
    torch.manual_seed(0)

    d_model = 128
    nhead = 8
    num_layers = 2

    B = 1
    D, H, W = 8, 8, 8
    S = D * H * W

    encoder_layer = TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=256,
        dropout=0.0,  # to simplify
        activation="relu",
        normalize_before=False,
    )
    encoder = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=None)

    src = torch.randn(B, S, d_model, requires_grad=True)

    out = encoder(src)  # no masks, no pos

    assert out.shape == (B, S, d_model)

    loss = out.sum()
    loss.backward()
    assert src.grad is not None
