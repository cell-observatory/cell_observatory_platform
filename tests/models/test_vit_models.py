import pytest
import torch
import torch.nn as nn
from timm.layers import SwiGLU

from cell_observatory_platform.models.backbones.encoder import Encoder
from cell_observatory_platform.models.backbones.vit import ViT

# (B, T, Z, Y, X, C) / patch (T, Z, Y, X) -> 2*2*2*2 = 16 tokens; C=2 channels
INPUT_SHAPE = (2, 4, 8, 16, 16, 2)
PATCH_SHAPE = (2, 4, 8, 8)
NUM_TOKENS = 16

TEMPLATES = {  # literal expected embed dims (vit-large/huge dropped: too big for a unit test)
    "vit": 32, "vit-tiny": 192, "vit-small": 384, "vit-base": 768,
}


@pytest.mark.parametrize("template,embed_dim", TEMPLATES.items(), ids=list(TEMPLATES))
def test_vit_template_forward_shape(template, embed_dim):
    """Each template resolves its embed dim and emits one finite token feature per patch."""
    torch.manual_seed(0)
    model = ViT(
        model_template=template,
        input_fmt="TZYXC",
        input_shape=INPUT_SHAPE[1:],
        patch_shape=PATCH_SHAPE,
        modes=0,            # head = Identity -> token features come out
        global_pool="",     # no pooling -> (B, N, D)
        embed_dim=32, depth=1, num_heads=2,   # only read by the custom "vit" template
        proj_drop_rate=0.0, att_drop_rate=0.0, drop_path_rate=0.0,
        rope_pos_enc=True, rope_type="axial",
        dtype=torch.float32,
    ).eval()

    x = torch.randn(INPUT_SHAPE)
    with torch.no_grad():
        out = model({"data_tensor": x, "metainfo": {}})

    assert model.embed_dim == embed_dim
    assert out.shape == (INPUT_SHAPE[0], NUM_TOKENS, embed_dim)
    assert torch.isfinite(out).all()
    assert model.get_num_patches() == NUM_TOKENS


# --- Encoder / ViT construction details: layer factories, pooling, wide-SiLU --- #


def _align8(v: int) -> int:
    return (v + 7) // 8 * 8


def test_encoder_builds_swiglu_and_layernorm_from_names_or_classes():
    """mlp_layer / norm_layer given as strings or as classes both produce SwiGLU
    MLPs and LayerNorm norms in the transformer blocks."""
    enc = Encoder(
        embed_dim=32, depth=1, num_heads=2, mlp_ratio=2.0,
        norm_layer="LayerNorm", mlp_layer="SwiGLU",
        rope_pos_enc=False,
        input_fmt="ZYXC", input_shape=(16, 16, 16, 1),
        patch_shape=(8, 8, 8, None),
    )
    blk = enc.transformer_blocks[0]
    assert isinstance(blk.mlp, SwiGLU), type(blk.mlp)
    assert isinstance(blk.norm1, nn.LayerNorm), type(blk.norm1)

    enc2 = Encoder(
        embed_dim=32, depth=1, num_heads=2, mlp_ratio=2.0,
        norm_layer=nn.LayerNorm, mlp_layer=SwiGLU,
        rope_pos_enc=False,
        input_fmt="ZYXC", input_shape=(16, 16, 16, 1),
        patch_shape=(8, 8, 8, None),
    )
    assert isinstance(enc2.transformer_blocks[0].mlp, SwiGLU)
    assert isinstance(enc2.transformer_blocks[0].norm1, nn.LayerNorm)


def _vit(**overrides):
    kwargs = dict(
        model_template="vit",
        input_fmt="TZYXC", input_shape=(4, 16, 16, 16, 1),
        patch_shape=(2, 8, 8, 8),
        modes=0, embed_dim=32, depth=1, num_heads=2, mlp_ratio=2.0,
        global_pool="avg", rope_pos_enc=False, mlp_layer="SwiGLU",
    )
    kwargs.update(overrides)
    return ViT(**kwargs)


def test_vit_avg_pool_includes_first_token():
    """Average pooling runs over every token; token 0 is not dropped as a prefix."""
    vit = _vit()
    x = torch.zeros(1, 4, 8)
    x[0, 0] = 4.0  # sentinel riding ONLY on token 0
    pooled = vit.pool(x)
    # avg over ALL 4 tokens = 1.0; dropping token 0 would give 0.0
    assert torch.allclose(pooled, torch.ones(1, 8))


def test_vit_mlp_wide_silu_flag_reaches_encoder_blocks():
    """mlp_wide_silu=True narrows the SwiGLU hidden width to align8(2/3 * dim * ratio)
    inside the encoder blocks; False keeps dim * ratio."""
    vit = _vit(mlp_wide_silu=True)
    expected = _align8(int(2 * (32 * 2.0) / 3))
    assert vit.encoder.transformer_blocks[0].mlp.fc1_g.out_features == expected
    vit_off = _vit(mlp_wide_silu=False)
    assert vit_off.encoder.transformer_blocks[0].mlp.fc1_g.out_features == int(32 * 2.0)
