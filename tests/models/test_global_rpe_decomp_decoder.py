import pytest
import torch

from cell_observatory_platform.models.heads.global_rpe_decomp_decoder import (
    GlobalCrossAttention,
    GlobalDecoder,
    GlobalDecoderLayer,
)


def _make_basic_3d_setup(
    B=2,
    D=4,
    H=8,
    W=8,
    C=256,
    num_queries=64,
    num_heads=8,
):
    """
    Helper to build:
      - single feature level
      - flattened memory: [B, N, C] with N = D * H * W
      - queries: [B, Q, C]
      - reference points: [B, Q, 1, 6]
    """
    N = D * H * W

    query = torch.randn(B, num_queries, C, requires_grad=True)
    k_flat = torch.randn(B, N, C, requires_grad=True)
    v_flat = torch.randn(B, N, C, requires_grad=True)

    # reference points in [0,1], format (cx, cy, cz, w, h, d)
    reference_points = torch.rand(B, num_queries, 1, 6)

    input_spatial_shapes = [(D, H, W)]

    padding_mask = torch.zeros(B, N, dtype=torch.bool)
    # mark a few tokens as padded in first batch item
    padding_mask[0, -10:] = True

    return {
        "B": B,
        "D": D,
        "H": H,
        "W": W,
        "C": C,
        "N": N,
        "num_queries": num_queries,
        "num_heads": num_heads,
        "query": query,
        "k_flat": k_flat,
        "v_flat": v_flat,
        "reference_points": reference_points,
        "input_spatial_shapes": input_spatial_shapes,
        "padding_mask": padding_mask,
    }


# -----------------------------
# GlobalCrossAttention tests
# -----------------------------


@pytest.mark.parametrize("reparam", [False, True])
@pytest.mark.parametrize("rpe_type", ["linear"])
def test_global_cross_attention_shapes(reparam, rpe_type):
    cfg = _make_basic_3d_setup()

    attn = GlobalCrossAttention(
        dim=cfg["C"],
        num_heads=cfg["num_heads"],
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        rpe_hidden_dim=64,
        rpe_type=rpe_type,
        feature_stride=16,
        reparam=reparam,
    )

    out = attn(
        query=cfg["query"],
        reference_points=cfg["reference_points"],
        k_input_flatten=cfg["k_flat"],
        v_input_flatten=cfg["v_flat"],
        input_spatial_shapes=cfg["input_spatial_shapes"],
        input_padding_mask=cfg["padding_mask"],
    )

    # output should be [B, Q, C]
    assert out.shape == (cfg["B"], cfg["num_queries"], cfg["C"])

    # Simple backward test
    loss = out.sum()
    loss.backward()

    assert cfg["query"].grad is not None
    assert cfg["k_flat"].grad is not None
    assert cfg["v_flat"].grad is not None


# -----------------------------
# GlobalDecoderLayer tests
# -----------------------------


@pytest.mark.parametrize("norm_type", ["pre_norm", "post_norm"])
def test_global_decoder_layer_shapes(norm_type):
    cfg = _make_basic_3d_setup()

    layer = GlobalDecoderLayer(
        d_model=cfg["C"],
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_heads=cfg["num_heads"],
        norm_type=norm_type,
        rpe_hidden_dim=64,
        rpe_type="linear",
        feature_stride=16,
        reparam=False,
    )

    # tgt: [B, Q, C]
    tgt = torch.randn(cfg["B"], cfg["num_queries"], cfg["C"], requires_grad=True)
    query_pos = torch.randn(cfg["B"], cfg["num_queries"], cfg["C"])
    src = cfg["k_flat"].detach().clone().requires_grad_(True)  # [B, N, C]
    src_pos_embed = torch.randn(cfg["B"], cfg["N"], cfg["C"])
    src_spatial_shapes = cfg["input_spatial_shapes"]
    src_padding_mask = cfg["padding_mask"]

    # reference_points_input: [B, Q, L, 6] (single level L=1)
    reference_points_input = torch.rand(cfg["B"], cfg["num_queries"], 1, 6)

    out = layer(
        tgt=tgt,
        query_pos=query_pos,
        reference_points=reference_points_input,
        src=src,
        src_pos_embed=src_pos_embed,
        src_spatial_shapes=src_spatial_shapes,
        src_padding_mask=src_padding_mask,
        self_attn_mask=None,
    )

    assert out.shape == (cfg["B"], cfg["num_queries"], cfg["C"])

    loss = out.sum()
    loss.backward()

    assert tgt.grad is not None
    assert src.grad is not None


# -----------------------------
# GlobalDecoder tests
# -----------------------------


def test_global_decoder_single_layer_no_refinement():
    """
    Smoke test for GlobalDecoder with:
      - single feature level
      - bbox_embed is None (no iterative refinement)
      - return_intermediate=False
    """
    cfg = _make_basic_3d_setup()

    # Build a single decoder layer
    decoder_layer = GlobalDecoderLayer(
        d_model=cfg["C"],
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_heads=cfg["num_heads"],
        norm_type="post_norm",
        rpe_hidden_dim=64,
        rpe_type="linear",
        feature_stride=16,
        reparam=False,
    )

    decoder = GlobalDecoder(
        decoder_layer=decoder_layer,
        num_layers=2,
        return_intermediate=False,
        look_forward_twice=False,
        d_model=cfg["C"],
        norm_type="post_norm",
        reparam=False,
    )

    B = cfg["B"]
    Q = cfg["num_queries"]
    C = cfg["C"]
    N = cfg["N"]

    tgt = torch.randn(B, Q, C, requires_grad=True)
    query_pos = torch.randn(B, Q, C)

    # reference_points: [B, Q, 6] (cx, cy, cz, w, h, d) in [0,1]
    reference_points = torch.rand(B, Q, 6)

    # memory / src: [B, N, C]
    src = torch.randn(B, N, C, requires_grad=True)
    src_pos_embed = torch.randn(B, N, C)

    # single level
    src_spatial_shapes = cfg["input_spatial_shapes"]
    src_level_start_index = None  # not used here

    # src_valid_ratios: [B, L, 3] where L=1
    src_valid_ratios = torch.ones(B, 1, 3)

    src_padding_mask = torch.zeros(B, N, dtype=torch.bool)
    self_attn_mask = None
    max_shape = None

    out, ref_out = decoder(
        tgt=tgt,
        reference_points=reference_points,
        src=src,
        src_pos_embed=src_pos_embed,
        src_spatial_shapes=src_spatial_shapes,
        src_level_start_index=src_level_start_index,
        src_valid_ratios=src_valid_ratios,
        query_pos=query_pos,
        src_padding_mask=src_padding_mask,
        self_attn_mask=self_attn_mask,
        max_shape=max_shape,
    )

    # output: [B, Q, C]
    assert out.shape == (B, Q, C)
    assert ref_out.shape == (B, Q, 6)

    loss = out.sum()
    loss.backward()

    assert tgt.grad is not None
    assert src.grad is not None


def test_global_decoder_return_intermediate():
    """
    Test GlobalDecoder with return_intermediate=True.
    We keep look_forward_twice=False to avoid depending on bbox refinement.
    """
    cfg = _make_basic_3d_setup()

    decoder_layer = GlobalDecoderLayer(
        d_model=cfg["C"],
        d_ffn=512,
        dropout=0.1,
        activation="relu",
        n_heads=cfg["num_heads"],
        norm_type="pre_norm",
        rpe_hidden_dim=64,
        rpe_type="linear",
        feature_stride=16,
        reparam=False,
    )

    num_layers = 3
    decoder = GlobalDecoder(
        decoder_layer=decoder_layer,
        num_layers=num_layers,
        return_intermediate=True,
        look_forward_twice=False,
        d_model=cfg["C"],
        norm_type="pre_norm",
        reparam=False,
    )

    B = cfg["B"]
    Q = cfg["num_queries"]
    C = cfg["C"]
    N = cfg["N"]

    tgt = torch.randn(B, Q, C, requires_grad=True)
    query_pos = torch.randn(B, Q, C)
    reference_points = torch.rand(B, Q, 6)
    src = torch.randn(B, N, C, requires_grad=True)
    src_pos_embed = torch.randn(B, N, C)
    src_spatial_shapes = cfg["input_spatial_shapes"]
    src_level_start_index = None
    src_valid_ratios = torch.ones(B, 1, 3)
    src_padding_mask = torch.zeros(B, N, dtype=torch.bool)

    hs, inter_refs = decoder(
        tgt=tgt,
        reference_points=reference_points,
        src=src,
        src_pos_embed=src_pos_embed,
        src_spatial_shapes=src_spatial_shapes,
        src_level_start_index=src_level_start_index,
        src_valid_ratios=src_valid_ratios,
        query_pos=query_pos,
        src_padding_mask=src_padding_mask,
        self_attn_mask=None,
        max_shape=None,
    )

    # hs: [num_layers, B, Q, C]
    assert hs.shape == (num_layers, B, Q, C)
    # inter_refs: [num_layers, B, Q, 6]
    assert inter_refs.shape == (num_layers, B, Q, 6)

    loss = hs.sum()
    loss.backward()

    assert tgt.grad is not None
    assert src.grad is not None
