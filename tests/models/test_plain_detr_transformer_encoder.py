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


def test_transformer_encoder_stack_ignores_padded_keys():
    """Perturbing only padded keys must leave every unpadded query's output unchanged."""
    torch.manual_seed(0)
    d_model, nhead, B, S, PAD = 64, 8, 2, 512, 10
    layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.0,
                                    activation="relu", normalize_before=False)
    encoder = TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(d_model)).eval()

    src = torch.randn(B, S, d_model)
    pos = torch.randn(B, S, d_model)
    key_padding = torch.zeros(B, S, dtype=torch.bool)
    key_padding[0, -PAD:] = True

    out_a = encoder(src, mask=None, src_key_padding_mask=key_padding, pos=pos)
    # perturb ONLY the padded keys of sample 0: unpadded outputs must not move
    src_b = src.clone()
    src_b[0, -PAD:] += 5.0
    out_b = encoder(src_b, mask=None, src_key_padding_mask=key_padding, pos=pos)

    assert out_a.shape == (B, S, d_model)
    assert torch.allclose(out_a[0, :-PAD], out_b[0, :-PAD], atol=1e-5)
    assert torch.allclose(out_a[1], out_b[1], atol=1e-6)        # untouched sample identical
    assert not torch.allclose(out_a[0, -PAD:], out_b[0, -PAD:])  # padded queries still see their own input


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


def test_encoder_layer_passes_dropout_to_self_attn():
    """The configured attention dropout reaches the layer's MultiheadAttention."""
    layer = TransformerEncoderLayer(d_model=32, nhead=4, dropout=0.3)
    assert layer.self_attn.dropout == pytest.approx(0.3)
