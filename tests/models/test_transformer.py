import pytest

import torch

from cell_observatory_platform.models.transformer import Attention, RopeAttention, Transformer


# ----------------------- Attention -----------------------


def tokens_from(fmt: str, input_shape: tuple, patch_size: tuple) -> int:
    axes = [ch for ch in fmt if ch in "TZYX"]
    prod = 1
    for ch in axes:
        i = fmt.index(ch)
        size = input_shape[i]
        p = patch_size[i if i < len(patch_size) else -1]
        assert size % p == 0, f"{ch}-axis not divisible by patch: {size} vs {p}"
        prod *= (size // p)
    return prod


@pytest.mark.parametrize("dim,num_heads,qk_norm", [
    (32, 4, False),
    (64, 8, True),
])
@pytest.mark.parametrize("B,L", [
    (2, 17),
    (1, 128),
])
def test_attention_shapes(dim, num_heads, qk_norm, B, L):
    torch.manual_seed(0)
    x = torch.randn(B, L, dim, device='cuda')
    m = Attention(dim=dim,
                  num_heads=num_heads, 
                  qk_norm=qk_norm).to('cuda')
    m.eval()
    y = m(x)
    assert y.shape == (B, L, dim)


# ----------------------- RoPE Attention -----------------------


ROPE_CASES = [
    ("YXC",   (1, 64, 64, 2),           (16,)),        # lateral only
    ("ZYXC",  (1, 8, 64, 64, 2),        (4, 16)),      # axial, lateral
    ("TYXC",  (1, 8, 64, 64, 2),        (4, 16)),      # temporal, lateral
    ("TZYXC", (1, 4, 16, 32, 32, 2),    (2, 8, 16)),   # temporal, axial, lateral
]

@pytest.mark.parametrize("dim,num_heads,rope_mixed", [
    (32, 4, True),
    (32, 4, False),
])
@pytest.mark.parametrize("case", ROPE_CASES, ids=[c[0] for c in ROPE_CASES])
def test_rope_attention_shapes(dim, num_heads, rope_mixed, case):
    input_fmt, input_shape_batched, patch_shape = case
    input_shape = input_shape_batched[1:]
    L = tokens_from(input_fmt, input_shape, patch_shape)
    B = 2

    x = torch.randn(B, L, dim, device='cuda')
    m = RopeAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=True,
        qk_norm=False,
        proj_bias=True,
        att_drop=0.0,
        proj_drop=0.0,
        rope_mixed=rope_mixed,
        rope_theta=10.0,
        input_fmt=input_fmt,
        input_shape=input_shape,
        patch_shape=patch_shape,
    ).to('cuda').eval()

    y = m(x)
    assert y.shape == (B, L, dim)


# ----------------------- Transformer block: with and without RoPE -------------------------------


@pytest.mark.parametrize("rope_pos_enc", [False, True])
@pytest.mark.parametrize("dim,num_heads,mlp_ratio", [
    (32, 4, 2.0),
    (64, 8, 4.0),
])
@pytest.mark.parametrize("case", ROPE_CASES, ids=[c[0] for c in ROPE_CASES])
def test_transformer_shapes(rope_pos_enc, dim, num_heads, mlp_ratio, case):
    input_fmt, input_shape_batched, patch_shape = case
    input_shape = input_shape_batched[1:]
    L = tokens_from(input_fmt, input_shape, patch_shape)
    B = 2

    x = torch.randn(B, L, dim, device='cuda')
    m = Transformer(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        qk_norm=False,
        proj_drop=0.0,
        att_drop=0.0,
        drop_path=0.0,
        rope_pos_enc=rope_pos_enc,
        rope_random_rotation_per_head=True,
        rope_mixed=True,
        rope_theta=10.0,
        input_fmt=input_fmt,
        input_shape=input_shape,
        patch_shape=patch_shape,
        wide_silu=False,
    ).to('cuda')
    m.eval()
    y = m(x)
    assert y.shape == (B, L, dim)
    