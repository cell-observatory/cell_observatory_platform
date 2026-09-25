import pytest
import torch
import torch.nn as nn
from timm.layers import Mlp, RmsNorm, SwiGLU

from cell_observatory_platform.models.layers.attention import LinearKMaskedBias, LinearMaskedBias
from cell_observatory_platform.models.layers.layer_scale import LayerScale
from cell_observatory_platform.models.layers.mlp import Mlp_ListFwdMixin, SwiGLUFFN_ListFwdMixin, get_mlp
from cell_observatory_platform.models.layers.norm import get_norm
from cell_observatory_platform.models.layers.transformer import Attention, RopeAttention, Transformer

DEVICE = torch.device("cpu")

# ----------------------- Attention -----------------------


def tokens_from(fmt: str, input_shape: tuple, patch_size: tuple) -> int:
    axes = [ch for ch in fmt if ch in "TZYX"]
    prod = 1
    for ch in axes:
        i = fmt.index(ch)
        size = input_shape[i]
        p = patch_size[i if i < len(patch_size) else -1]
        assert size % p == 0, f"{ch}-axis not divisible by patch: {size} vs {p}"
        prod *= size // p
    return prod


@pytest.mark.parametrize(
    "dim,num_heads,qk_norm",
    [
        (32, 4, False),
        (64, 8, True),
    ],
)
@pytest.mark.parametrize(
    "B,L",
    [
        (2, 17),
        (1, 128),
    ],
)
def test_attention_shapes(dim, num_heads, qk_norm, B, L):
    torch.manual_seed(0)
    x = torch.randn(B, L, dim, device=DEVICE)
    m = Attention(dim=dim, num_heads=num_heads, qk_norm=qk_norm).to(DEVICE).eval()
    y = m(x)
    assert y.shape == (B, L, dim)
    assert torch.isfinite(y).all()


# ----------------------- RoPE Attention -----------------------


ROPE_CASES = [
    ("YXC", (1, 32, 32, 2), (16,)),  # lateral only
    ("ZYXC", (1, 16, 32, 32, 2), (16, 16)),  # axial, lateral
    ("TYXC", (1, 8, 32, 32, 2), (4, 16)),  # temporal, lateral
    ("TZYXC", (1, 4, 16, 32, 32, 2), (2, 16, 16)),  # temporal, axial, lateral
]


@pytest.mark.parametrize(
    "dim,num_heads,rope_type",
    [(32, 4, "mixed")],  # axial requires pos_enc from top-level module
)
@pytest.mark.parametrize("case", ROPE_CASES, ids=[c[0] for c in ROPE_CASES])
def test_rope_attention_shapes(dim, num_heads, rope_type, case):
    torch.manual_seed(0)
    input_fmt, input_shape_batched, patch_shape = case
    input_shape = input_shape_batched[1:]
    L = tokens_from(input_fmt, input_shape, patch_shape)
    B = 2

    x = torch.randn(B, L, dim, device=DEVICE)
    m = RopeAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=True,
        qk_norm=False,
        proj_bias=True,
        att_drop=0.0,
        proj_drop=0.0,
        rope_type=rope_type,
        rope_theta=10.0,
        input_fmt=input_fmt,
        input_shape=input_shape,
        patch_shape=patch_shape,
    ).to(DEVICE)
    m.init_rope_parameters(device=DEVICE)
    m.eval()

    y = m(x)
    assert y.shape == (B, L, dim)
    assert torch.isfinite(y).all()


# ----------------------- Transformer block: with and without RoPE -------------------------------


@pytest.mark.parametrize("rope_pos_enc", [False, True])
@pytest.mark.parametrize(
    "dim,num_heads,mlp_ratio",
    [
        (32, 4, 2.0),
        (64, 8, 4.0),
    ],
)
@pytest.mark.parametrize("case", ROPE_CASES, ids=[c[0] for c in ROPE_CASES])
def test_transformer_shapes(rope_pos_enc, dim, num_heads, mlp_ratio, case):
    torch.manual_seed(0)
    input_fmt, input_shape_batched, patch_shape = case
    input_shape = input_shape_batched[1:]
    L = tokens_from(input_fmt, input_shape, patch_shape)
    B = 2

    x = torch.randn(B, L, dim, device=DEVICE)
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
        rope_type="mixed" if rope_pos_enc else "axial",
        rope_theta=10.0,
        input_fmt=input_fmt,
        input_shape=input_shape,
        patch_shape=patch_shape,
        wide_silu=False,
    ).to(DEVICE)

    for mod in m.modules():
        if isinstance(mod, RopeAttention):
            mod.init_rope_parameters(device=DEVICE)

    m.eval()

    y = m(x)
    assert y.shape == (B, L, dim)
    assert torch.isfinite(y).all()


# ----------------------- small layers: LayerScale, masked-bias linears -----------------------


def test_layer_scale_gamma_initialized_to_init_values():
    """LayerScale's gamma is a (dim,) parameter filled with init_values at construction."""
    ls = LayerScale(dim=16, init_values=1e-5)
    torch.testing.assert_close(ls.gamma, torch.full((16,), 1e-5))


def test_linear_masked_bias_masks_are_finite_and_zero_k_third():
    """LinearMaskedBias zeroes its whole bias mask; LinearKMaskedBias zeroes only
    the K third of a fused qkv bias and passes the Q and V thirds through."""
    wk = LinearMaskedBias(8, 8, bias=True)
    assert torch.isfinite(wk.bias_mask).all()
    assert torch.equal(wk.bias_mask, torch.zeros(8))
    out = wk(torch.randn(2, 8))
    assert torch.isfinite(out).all()

    qkv = LinearKMaskedBias(8, 24, bias=True)
    assert torch.isfinite(qkv.bias_mask).all()
    assert torch.equal(qkv.bias_mask[8:16], torch.zeros(8))
    assert torch.equal(qkv.bias_mask[:8], torch.ones(8))
    assert torch.equal(qkv.bias_mask[16:], torch.ones(8))


# ----------------------- get_mlp / get_norm factories -----------------------


def test_get_mlp_get_norm_resolve_string_names():
    """Every supported name resolves to the matching class object."""
    assert get_mlp("Mlp") is Mlp
    assert get_mlp("SwiGLU") is SwiGLU
    assert get_mlp("Mlp_ListFwdMixin") is Mlp_ListFwdMixin
    assert get_mlp("SwiGLUFFN_ListFwdMixin") is SwiGLUFFN_ListFwdMixin
    assert get_norm("RmsNorm") is RmsNorm
    assert get_norm("LayerNorm") is nn.LayerNorm
    assert get_norm("SyncBatchNorm") is nn.SyncBatchNorm
    assert get_norm("GroupNorm") is nn.GroupNorm


def test_get_mlp_get_norm_pass_classes_through_unchanged():
    """A class input (including a custom one) is returned as-is, never collapsed
    to the default."""
    assert get_mlp(SwiGLU) is SwiGLU
    assert get_mlp(Mlp) is Mlp
    assert get_mlp(SwiGLUFFN_ListFwdMixin) is SwiGLUFFN_ListFwdMixin
    assert get_norm(nn.LayerNorm) is nn.LayerNorm
    assert get_norm(nn.GroupNorm) is nn.GroupNorm
    assert get_norm(RmsNorm) is RmsNorm

    class Custom(nn.Module):
        pass

    assert get_mlp(Custom) is Custom
    assert get_norm(Custom) is Custom


def test_get_mlp_get_norm_are_idempotent():
    """Resolving an already-resolved class (meta-arch then Encoder) is stable."""
    assert get_mlp(get_mlp("SwiGLU")) is SwiGLU
    assert get_norm(get_norm("LayerNorm")) is nn.LayerNorm


def test_get_mlp_get_norm_reject_unknown_name_and_non_class():
    """Unknown names raise ValueError; non-string non-class inputs raise TypeError."""
    with pytest.raises(ValueError):
        get_mlp("NotAThing")
    with pytest.raises(ValueError):
        get_norm("NotAThing")
    with pytest.raises(TypeError):
        get_mlp(3.14)
    with pytest.raises(TypeError):
        get_norm(3.14)


# ----------------------- wide-SiLU hidden widths -----------------------


def _align8(v: int) -> int:
    return (v + 7) // 8 * 8


def _block(mlp_layer, wide_silu):
    return Transformer(
        dim=48, num_heads=2, mlp_ratio=4.0, wide_silu=wide_silu,
        norm_layer=nn.LayerNorm, act_layer=nn.GELU, mlp_layer=mlp_layer,
        rope_pos_enc=False,
    )


def test_wide_silu_timm_swiglu_hidden_is_align8_two_thirds():
    """timm SwiGLU uses hidden_features as given, so wide_silu hands it
    align8(2/3 * dim * ratio)."""
    blk = _block(SwiGLU, wide_silu=True)
    expected = _align8(int(2 * (48 * 4.0) / 3))
    assert blk.mlp.fc1_g.out_features == expected


def test_wide_silu_listfwdmixin_receives_raw_hidden():
    """SwiGLUFFN_ListFwdMixin applies its own 2/3 reduction internally, so it must
    receive the raw dim*ratio: 48*4 -> 2/3 -> 128 (not a double-reduced 85 -> 88)."""
    blk = _block(SwiGLUFFN_ListFwdMixin, wide_silu=True)
    assert blk.mlp.w1.out_features == 128


def test_without_wide_silu_hidden_is_dim_times_ratio():
    """wide_silu=False keeps the plain dim*ratio hidden width."""
    blk = _block(SwiGLU, wide_silu=False)
    assert blk.mlp.fc1_g.out_features == int(48 * 4.0)


def test_wide_silu_ignored_for_plain_mlp():
    """wide_silu only applies to SwiGLU; a plain Mlp keeps dim*ratio and builds cleanly."""
    blk = _block(Mlp, wide_silu=True)
    assert blk.mlp.fc1.out_features == int(48 * 4.0)


# ----------------------- stochastic depth + per-sample masks -----------------------


def test_forward_list_rejects_masks_with_sample_drop():
    """Per-sample stochastic depth cannot be combined with per-sample masks; the
    list forward refuses the combination loudly."""
    blk = Transformer(dim=32, num_heads=2, sample_drop_ratio=0.5, rope_pos_enc=False)
    blk.train()
    x = [torch.randn(4, 10, 32)]
    with pytest.raises(NotImplementedError, match="masks"):
        blk._forward_list(x, masks=[torch.ones(4, 10, dtype=torch.bool)])


def test_forward_list_sample_drop_runs_without_masks():
    """With masks=None the sample-drop list forward runs and keeps the token shape."""
    blk = Transformer(dim=32, num_heads=2, sample_drop_ratio=0.5,
                      rope_pos_enc=False, mlp_layer=SwiGLUFFN_ListFwdMixin)
    blk.train()
    out = blk._forward_list([torch.randn(4, 10, 32)], masks=None)
    assert out[0].shape == (4, 10, 32)
