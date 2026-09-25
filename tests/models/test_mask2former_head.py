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

# deformable attention is the only GPU-only piece; without the kernel the head runs on CPU
pytestmark = [pytest.mark.cuda] if OPS3D_AVAILABLE else []
DEVICE = torch.device("cuda" if OPS3D_AVAILABLE else "cpu")


def _make_input_shape_dict_for_m2f(c1, c2, c3, c4) -> Dict[str, Dict[str, int]]:
    return {
        "1": {"channels": c1, "stride": 8},
        "2": {"channels": c2, "stride": 16},
        "3": {"channels": c3, "stride": 32},
        "4": {"channels": c4, "stride": 64},
    }


def test_mask2former_head_forward_shapes():
    """pixel decoder + transformer predictor emit logits (B, Q, K+1), masks at the finest
    feature resolution and one auxiliary prediction per decoder layer."""
    torch.manual_seed(0)
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
        use_deform_attention=OPS3D_AVAILABLE,
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
    ).to(DEVICE)

    features = {
        k: torch.randn(B, input_shape_metadata[k]["channels"], *res, device=DEVICE)
        for k, res in zip(("1", "2", "3", "4"), (res1, res2, res3, res4))
    }

    out = head(features)

    assert isinstance(out, dict)
    assert "pred_logits" in out and "pred_masks" in out and "auxiliary_outputs" in out

    # logits: [B, Q, num_classes+1]
    assert out["pred_logits"].shape == (B, num_queries, num_classes + 1)
    assert out["pred_logits"].device.type == DEVICE.type

    # masks: [B, Q, D, H, W] -> finest level (res1)
    assert out["pred_masks"].shape == (B, num_queries, *res1)
    assert out["pred_masks"].device.type == DEVICE.type
    assert torch.isfinite(out["pred_logits"]).all() and torch.isfinite(out["pred_masks"]).all()

    aux = out["auxiliary_outputs"]
    assert isinstance(aux, list)
    assert len(aux) == decoder_layers
    for a in aux:
        assert a["pred_logits"].shape == (B, num_queries, num_classes + 1)
        assert a["pred_masks"].shape == (B, num_queries, *res1)
        assert a["pred_logits"].device.type == DEVICE.type and a["pred_masks"].device.type == DEVICE.type
