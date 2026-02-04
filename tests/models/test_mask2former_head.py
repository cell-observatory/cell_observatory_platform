import pytest
from typing import Dict
import torch

from cell_observatory_platform.models.heads.mask2former_head import Mask2FormerHead
from cell_observatory_platform.models.heads.pixel_decoders import Mask2FormerPixelDecoder
from cell_observatory_platform.models.heads.mask2former_decoder import MultiScaleMaskedTransformerDecoder

try:
    from ops3d import _C
    OPS3D_AVAILABLE = True
except ImportError:
    OPS3D_AVAILABLE = False


def _make_input_shape_dict_for_m2f(c1, c2, c3, c4) -> Dict[str, Dict[str, int]]:
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
    decoder_layers = 2

    input_shape_metadata = _make_input_shape_dict_for_m2f(48, 64, 96, 128)

    res1 = (32, 32, 32)
    res2 = (16, 16, 16)
    res3 = (8, 8, 8)
    res4 = (4, 4, 4)

    pixel_decoder = Mask2FormerPixelDecoder(
        input_shape_metadata=input_shape_metadata,
        transformer_in_features=["1", "2", "3", "4"],
        target_min_stride=8,
        total_num_feature_levels=3,
        transformer_encoder_dropout=0.0,
        transformer_encoder_num_heads=8,
        transformer_encoder_dim_feedforward=4 * hidden_dim,
        transformer_encoder_layers=2,
        conv_dim=hidden_dim,
        mask_dim=hidden_dim,
        norm="GroupNorm",
        use_deform_attention=True if OPS3D_AVAILABLE else False,
    )

    predictor = MultiScaleMaskedTransformerDecoder(
        input_dim=3,
        in_channels=hidden_dim,
        mask_classification=True,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        decoder_nheads=8,
        dim_feedforward=4 * hidden_dim,
        decoder_num_layers=decoder_layers,
        decoder_pre_norm=False,
        mask_dim=hidden_dim,
        enforce_input_project=False,
        num_feature_levels=3,
    )

    head = Mask2FormerHead(
        pixel_decoder=pixel_decoder,
        predictor=predictor,
        num_classes=num_classes,
    ).cuda()

    features = {
        "1": torch.randn(B, input_shape_metadata["1"]["channels"], *res1, device="cuda"),
        "2": torch.randn(B, input_shape_metadata["2"]["channels"], *res2, device="cuda"),
        "3": torch.randn(B, input_shape_metadata["3"]["channels"], *res3, device="cuda"),
        "4": torch.randn(B, input_shape_metadata["4"]["channels"], *res4, device="cuda"),
    }

    out = head(features)

    assert isinstance(out, dict)
    assert "pred_logits" in out and "pred_masks" in out and "auxiliary_outputs" in out

    # logits: [B, Q, num_classes+1]
    assert out["pred_logits"].shape == (B, num_queries, num_classes + 1)
    assert out["pred_logits"].is_cuda

    # masks: [B, Q, D, H, W] -> finest level (res1)
    assert out["pred_masks"].shape == (B, num_queries, *res4)
    assert out["pred_masks"].is_cuda

    aux = out["auxiliary_outputs"]
    assert isinstance(aux, list)
    assert len(aux) == decoder_layers
    for a in aux:
        assert a["pred_logits"].shape == (B, num_queries, num_classes + 1)
        assert a["pred_masks"].shape == (B, num_queries, *res4)
        assert a["pred_logits"].is_cuda and a["pred_masks"].is_cuda
