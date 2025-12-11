import pytest

import torch

from cell_observatory_platform.models.heads.mask2former_head import Mask2FormerHead

try:
    from ops3d import _C
    OPS3D_AVAILABLE = True
except ImportError:
    OPS3D_AVAILABLE = False


def _make_input_shape_dict_for_m2f(c1, c2, c3, c4):
    return {
        "1": {"channels": c1, "stride": 8},
        "2": {"channels": c2, "stride": 16},
        "3": {"channels": c3, "stride": 32},
        "4": {"channels": c4, "stride": 64},
    }


@pytest.mark.cuda
def test_mask2former_head_forward_shapes_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Mask2FormerHead (deformable attention uses GPU)")

    B = 2
    hidden_dim = 64
    num_classes = 7
    num_queries = 20
    decoder_layers = 3

    input_shape = _make_input_shape_dict_for_m2f(48, 64, 96, 128)

    res1 = (32, 32, 32)
    res2 = (16, 16, 16)
    res3 = (8, 8, 8)
    res4 = (4, 4, 4)

    head = Mask2FormerHead(
        input_shape=input_shape,
        input_dim=3,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        decoder_num_queries=num_queries,
        decoder_layers=decoder_layers,
        decoder_nheads=8,
        decoder_dim_feedforward=4 * hidden_dim,
        decoder_pre_norm=False,
        decoder_enforce_input_project=False,
        decoder_num_feature_levels=3,
        pixel_decoder_transformer_dropout=0.0,
        pixel_decoder_transformer_nheads=8,
        pixel_decoder_transformer_dim_feedforward=4 * hidden_dim,
        pixel_decoder_transformer_enc_layers=2,
        pixel_decoder_norm="GroupNorm",
        pixel_decoder_transformer_in_features=("1", "2", "3", "4"),
        pixel_decoder_common_stride=8,
        use_deform_attention=True if OPS3D_AVAILABLE else False,
    ).cuda()

    features = {
        "1": torch.randn(B, input_shape["1"]["channels"], *res1, device="cuda"),
        "2": torch.randn(B, input_shape["2"]["channels"], *res2, device="cuda"),
        "3": torch.randn(B, input_shape["3"]["channels"], *res3, device="cuda"),
        "4": torch.randn(B, input_shape["4"]["channels"], *res4, device="cuda"),
    }

    out = head(features)

    assert isinstance(out, dict)
    assert "pred_logits" in out and "pred_masks" in out and "aux_outputs" in out

    # logits: [B, Q, num_classes+1]
    assert out["pred_logits"].shape == (B, num_queries, num_classes + 1)
    assert out["pred_logits"].is_cuda

    # masks: [B, Q, D, H, W] -> finest level (res1)
    assert out["pred_masks"].shape == (B, num_queries, *res4)
    assert out["pred_masks"].is_cuda

    aux = out["aux_outputs"]
    assert isinstance(aux, list)
    assert len(aux) == decoder_layers
    for a in aux:
        assert a["pred_logits"].shape == (B, num_queries, num_classes + 1)
        assert a["pred_masks"].shape == (B, num_queries, *res4)
        assert a["pred_logits"].is_cuda and a["pred_masks"].is_cuda
