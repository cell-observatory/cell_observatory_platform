from types import SimpleNamespace

import pytest
import torch

from cell_observatory_platform.models.heads.plain_detr_transformer import Transformer, TransformerReParam


def _build_dummy_decoder_args(
    d_model: int,
    n_heads: int,
    reparam: bool,
    num_layers: int = 2,
):
    return SimpleNamespace(
        hidden_dim=d_model,
        dim_feedforward=4 * d_model,
        dropout=0.1,
        nheads=n_heads,
        norm_type="post_norm",
        decoder_rpe_hidden_dim=64,
        decoder_rpe_type="linear",
        proposal_in_stride=16,
        reparam=reparam,
        dec_layers=num_layers,
        look_forward_twice=False,
        decoder_use_checkpoint=False,
    )


def _build_dummy_inputs_3d(
    batch_size: int = 2,
    d_model: int = 32,
    D: int = 8,
    H: int = 8,
    W: int = 8,
    num_levels: int = 1,
):
    """
    Build srcs/masks/pos_embeds lists matching Transformer.forward signature.
    """
    srcs, masks, pos_embeds = [], [], []
    for _ in range(num_levels):
        src = torch.randn(batch_size, d_model, D, H, W)
        mask = torch.zeros(batch_size, D, H, W, dtype=torch.bool)
        pos = torch.randn(batch_size, d_model, D, H, W)
        srcs.append(src)
        masks.append(mask)
        pos_embeds.append(pos)
    return srcs, masks, pos_embeds


def test_transformer_two_stage_forward_3d():
    """
    Two-stage (two_stage=True) 3D transformer: non-reparam path.
    """
    batch_size = 2
    d_model = 32
    n_heads = 4
    num_queries = 100
    two_stage_num_proposals = num_queries

    global_decoder_args = _build_dummy_decoder_args(
        d_model=d_model,
        n_heads=n_heads,
        reparam=False,
        num_layers=2,
    )

    model = Transformer(
        d_model=d_model,
        nhead=n_heads,
        num_feature_levels=1,
        two_stage=True,
        two_stage_num_proposals=two_stage_num_proposals,
        decoder_type="global_rpe_decomp",
        global_decoder_args=global_decoder_args,
        add_transformer_encoder=False,
        proposal_feature_levels=1,
    )

    model.decoder.class_embed = torch.nn.ModuleList(
        [torch.nn.Linear(d_model, 1) for _ in range(model.decoder.num_layers + 1)]
    )
    model.decoder.bbox_embed = torch.nn.ModuleList(
        [torch.nn.Linear(d_model, 6) for _ in range(model.decoder.num_layers + 1)]
    )

    srcs, masks, pos_embeds = _build_dummy_inputs_3d(
        batch_size=batch_size,
        d_model=d_model,
        D=8,
        H=8,
        W=8,
        num_levels=1,
    )

    hs, init_ref, inter_refs, enc_cls, enc_coord_unact, enc_delta, out_props, max_shape = model(
        srcs,
        masks,
        pos_embeds,
        query_embed=None,
        self_attn_mask=None,
    )

    # hs: [num_decoder_layers, B, Q, C]
    assert hs.dim() == 4
    num_layers, B_hs, Q_hs, C_hs = hs.shape
    assert B_hs == batch_size
    assert Q_hs == two_stage_num_proposals
    assert C_hs == d_model

    # init_ref: [B, Q, 6]
    assert init_ref.shape == (batch_size, two_stage_num_proposals, 6)

    # inter_refs: [num_layers, B, Q, 6]
    assert inter_refs.shape == (num_layers, batch_size, two_stage_num_proposals, 6)

    # encoder outputs
    assert enc_cls is not None
    assert enc_cls.shape[0] == batch_size

    assert enc_coord_unact is not None
    assert enc_coord_unact.shape[0] == batch_size
    assert enc_coord_unact.shape[-1] == 6

    # non-reparam: enc_delta is None
    assert enc_delta is None

    assert out_props is not None
    assert out_props.shape[0] == batch_size
    assert out_props.shape[-1] == 6

    # base Transformer: max_shape is None
    assert max_shape is None


def test_transformer_reparam_two_stage_forward_3d():
    """
    Two-stage + reparam TransformerReParam path.
    """
    batch_size = 2
    d_model = 32
    n_heads = 4
    num_queries = 100
    two_stage_num_proposals = num_queries

    global_decoder_args = _build_dummy_decoder_args(
        d_model=d_model,
        n_heads=n_heads,
        reparam=True,
        num_layers=2,
    )

    model = TransformerReParam(
        d_model=d_model,
        nhead=n_heads,
        num_feature_levels=1,
        two_stage=True,
        two_stage_num_proposals=two_stage_num_proposals,
        decoder_type="global_rpe_decomp",
        global_decoder_args=global_decoder_args,
        add_transformer_encoder=False,
        proposal_feature_levels=1,
    )

    srcs, masks, pos_embeds = _build_dummy_inputs_3d(
        batch_size=batch_size,
        d_model=d_model,
        D=8,
        H=8,
        W=8,
        num_levels=1,
    )

    # HACK: since we normally initalize class_embed in main plainDETR model,
    #       we have to do it here too
    model.decoder.class_embed = torch.nn.ModuleList(
        [torch.nn.Linear(d_model, 1) for _ in range(model.decoder.num_layers + 1)]
    )
    model.decoder.bbox_embed = torch.nn.ModuleList(
        [torch.nn.Linear(d_model, 6) for _ in range(model.decoder.num_layers + 1)]
    )

    hs, init_ref, inter_refs, enc_cls, enc_coord_unact, enc_delta, out_props, max_shape = model(
        srcs,
        masks,
        pos_embeds,
        query_embed=None,
        self_attn_mask=None,
    )

    # hs: [num_decoder_layers, B, Q, C]
    assert hs.dim() == 4
    num_layers, B_hs, Q_hs, C_hs = hs.shape
    assert B_hs == batch_size
    assert Q_hs == two_stage_num_proposals
    assert C_hs == d_model

    # init_ref: [B, Q, 6]
    assert init_ref.shape == (batch_size, two_stage_num_proposals, 6)

    # inter_refs: [num_layers, B, Q, 6]
    assert inter_refs.shape == (num_layers, batch_size, two_stage_num_proposals, 6)

    # encoder outputs
    assert enc_cls is not None
    assert enc_cls.shape[0] == batch_size

    assert enc_coord_unact is not None
    assert enc_coord_unact.shape[0] == batch_size
    assert enc_coord_unact.shape[-1] == 6

    # reparam: enc_delta is [B, S, 6]
    assert enc_delta is not None
    assert enc_delta.shape[0] == batch_size
    assert enc_delta.shape[-1] == 6

    assert out_props is not None
    assert out_props.shape[0] == batch_size
    assert out_props.shape[-1] == 6

    # reparam version sets a max_shape tuple
    assert max_shape is not None
    assert isinstance(max_shape, tuple) and len(max_shape) == 3
