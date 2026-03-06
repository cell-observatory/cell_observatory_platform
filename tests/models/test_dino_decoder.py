import pytest
import torch
from torch import nn

from cell_observatory_platform.models.heads.dino_decoder import DeformableTransformerDecoderLayer, TransformerDecoder

try:
    from ops3d import _C
    OPS3D_AVAILABLE = True
except ImportError:
    OPS3D_AVAILABLE = False
    
CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="FlashDeformAttn3D (OPS3D_AVAILABLE) is not installed.",
)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_deformable_transformer_decoder_layer():
    device = torch.device("cuda")

    batch_size = 2
    num_queries = 5
    # NOTE: embed_dim/num_heads needs to be divisible by 8!
    embed_dim = 64
    num_levels = 3
    num_heads = 8

    layer = DeformableTransformerDecoderLayer(
        embed_dim=embed_dim,
        feedforward_dim=64,
        num_levels=num_levels,
        num_heads=num_heads,
    ).to(device)

    # target: (num_queries, bs, embed_dim)
    target = torch.randn(num_queries, batch_size, embed_dim, device=device)
    target_pos = torch.randn_like(target)

    # 3D feature map shapes per level (D, H, W)
    level_shapes = torch.tensor(
        [
            [32, 32, 32],  # finest
            [16, 16, 16],
            [8, 8, 8],  # coarsest
        ],
        dtype=torch.long,
        device=device,
    )  # (num_levels, 3)

    # flatten levels into a single sequence for memory
    num_tokens_per_level = level_shapes.prod(dim=1)  # (num_levels,)
    level_start_index = torch.cumsum(
        torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                num_tokens_per_level[:-1],
            ]
        ),
        dim=0,
    )  # (num_levels,)
    num_tokens = int(num_tokens_per_level.sum().item())

    # memory: (dhw, bs, embed_dim)
    memory = torch.randn(num_tokens, batch_size, embed_dim, device=device)

    # reference points per query, per batch, per level, 3D coords in [0, 1]
    # shape: (num_queries, bs, num_levels, 3)
    target_reference_points = torch.rand(num_queries, batch_size, num_levels, 3, device=device)

    # key padding mask for memory: (bs, num_tokens)
    memory_key_padding_mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)

    out = layer(
        target=target,
        target_query_pos_embeddings=target_pos,
        target_reference_points=target_reference_points,
        memory=memory,
        memory_key_padding_mask=memory_key_padding_mask,
        memory_level_start_index=level_start_index,
        # FlashDeformAttn3D expects (n_levels, 3)
        memory_shapes=level_shapes,
    )

    assert out.shape == (num_queries, batch_size, embed_dim)

    # also check that removing self-attn still preserves shapes
    layer.remove_self_attn_modules()
    out_no_self = layer(
        target=target,
        target_query_pos_embeddings=target_pos,
        target_reference_points=target_reference_points,
        memory=memory,
        memory_key_padding_mask=memory_key_padding_mask,
        memory_level_start_index=level_start_index,
        memory_shapes=level_shapes,
    )
    assert out_no_self.shape == (num_queries, batch_size, embed_dim)


@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="FlashDeformAttn3D (OPS3D_AVAILABLE) is not installed.",
)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_transformer_decoder():
    device = torch.device("cuda")

    batch_size = 2
    num_queries = 5
    embed_dim = 64
    num_levels = 3
    num_heads = 8
    query_dim = 6  # (x, y, z, w, h, d)

    base_layer = DeformableTransformerDecoderLayer(
        embed_dim=embed_dim,
        feedforward_dim=64,
        num_levels=num_levels,
        num_heads=num_heads,
    )
    decoder = TransformerDecoder(
        decoder_layer=base_layer,
        num_layers=2,
        norm=nn.LayerNorm(embed_dim),
        embed_dim=embed_dim,
        query_dim=query_dim,
        num_feature_levels=num_levels,
        deformable_decoder=True,
    ).to(device)
    decoder.bbox_embed = None

    # target: (num_queries, bs, embed_dim)
    target = torch.randn(num_queries, batch_size, embed_dim, device=device)

    # same realistic 3D shapes as above: (num_levels, 3)
    level_shapes = torch.tensor(
        [
            [32, 32, 32],
            [16, 16, 16],
            [8, 8, 8],
        ],
        dtype=torch.long,
        device=device,
    )

    # flatten levels for memory
    num_tokens_per_level = level_shapes.prod(dim=1)  # (num_levels,)
    level_start_index = torch.cumsum(
        torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                num_tokens_per_level[:-1],
            ]
        ),
        dim=0,
    )  # (num_levels,)
    num_tokens = int(num_tokens_per_level.sum().item())

    # memory: (dhw, bs, embed_dim)
    memory = torch.randn(num_tokens, batch_size, embed_dim, device=device)

    # initial reference points: (num_queries, bs, query_dim)
    reference_points = torch.rand(num_queries, batch_size, query_dim, device=device)

    # valid ratios per feature level: (B, L, 3) == (w_ratio, h_ratio, d_ratio)
    # all ones -> everything valid (no padding)
    valid_ratios = torch.ones(batch_size, num_levels, 3, device=device)

    outputs, ref_points_out = decoder(
        target=target,
        memory=memory,
        reference_points=reference_points,
        level_start_index=level_start_index,
        shapes=level_shapes,
        valid_ratios=valid_ratios,
    )

    # outputs is a list (per layer) of (bs, num_queries, embed_dim)
    assert len(outputs) == 2
    for out in outputs:
        assert out.shape == (batch_size, num_queries, embed_dim)

    # ref_points_out is a list of reference point tensors; here just the initial one
    assert len(ref_points_out) == 1
    assert ref_points_out[0].shape == (batch_size, num_queries, query_dim)
