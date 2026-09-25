import pytest

import torch
import torch.nn as nn

from omegaconf import OmegaConf
from hydra.utils import get_method

from cell_observatory_platform.models.meta_arch.plainDETR import PlainDETRReParam


class _DummyBackbone(nn.Module):
    """
    Minimal backbone stub: reads only B/device/dtype from data_tensor and emits a
    single zero feature map of a fixed spatial size.
    """
    def __init__(self, backbone_embed_dim: int, out_shape=(8, 16, 16)):
        super().__init__()
        self.backbone_embed_dim = backbone_embed_dim
        self.out_shape = out_shape

    def forward(self, samples):
        data = samples["data_tensor"]
        B = data.shape[0]
        D, H, W = self.out_shape
        device = data.device
        dtype = data.dtype

        x = torch.zeros(B, self.backbone_embed_dim, D, H, W, device=device, dtype=dtype)
        mask = torch.zeros(B, D, H, W, device=device, dtype=torch.bool)
        return [{"x": x, "mask": mask}]


@pytest.mark.cuda
def test_plain_detr_reparam_forward_shapes_bf16():
    """A bf16 two-stage reparam forward emits Q logits/boxes per image, encoder outputs and a positive loss."""
    torch.manual_seed(0)
    NUM_Q, NUM_CLASSES, BACKBONE_DIM = 20, 2, 64
    cfg = OmegaConf.create({
        "transformer_args": {
            "BUILD": "cell_observatory_platform.models.heads.plain_detr_transformer.BUILD",
            "d_model": 384, "nheads": 8, "num_feature_levels": 1, "two_stage": True,
            "norm_type": "pre_norm", "decoder_type": "global_rpe_decomp",
            "proposal_feature_levels": 3, "proposal_in_stride": 8, "proposal_tgt_strides": [4, 8, 16],
            "proposal_min_size": 16, "add_transformer_encoder": True, "dim_feedforward": 512,
            "dropout": 0.0, "activation": "relu", "normalize_before": True, "num_encoder_layers": 1,
            "global_decoder_args": {
                "hidden_dim": 384, "nheads": 8, "dropout": 0.0, "proposal_in_stride": 8,
                "norm_type": "pre_norm", "dim_feedforward": 512, "num_heads": 8, "qkv_bias": True,
                "qk_scale": None, "attn_drop": 0.0, "proj_drop": 0.0, "dec_layers": 2,
                "look_forward_twice": True, "rpe_hidden_dim": 128, "rpe_type": "linear", "reparam": True,
            },
        },
        "criterion_args": {
            "BUILD": "cell_observatory_platform.training.losses.build_plainDETR_Set_Loss",
            "weight_dict": {"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0}, "focal_alpha": 0.25,
            "matcher_args": {"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0, "cost_bbox_type": "reparam"},
        },
        "backbone_embed_dim": BACKBONE_DIM, "num_classes": NUM_CLASSES, "num_feature_levels": 1,
        "with_box_refine": True, "two_stage": True, "aux_loss": False,
        "num_queries_one2one": NUM_Q, "num_queries_one2many": 0, "k_one2many": 0,
        "mixed_selection": True, "reparam": True, "lambda_one2many": 1.0, "normalize_pos_encodings": True,
    })
    t_cfg = dict(cfg["transformer_args"], reparam=True, num_queries_one2one=NUM_Q,
                 num_queries_one2many=0, mixed_selection=True)
    transformer = get_method(t_cfg["BUILD"])(t_cfg)
    loss_module = get_method(cfg["criterion_args"]["BUILD"])(
        cfg["criterion_args"], num_classes=NUM_CLASSES, two_stage=True, reparam=True,
        aux_loss=False, dec_layers=transformer.decoder.num_layers)
    init_kwargs = {k: v for k, v in cfg.items() if k not in ("transformer_args", "criterion_args")}
    model = PlainDETRReParam(backbone=_DummyBackbone(BACKBONE_DIM), transformer=transformer,
                             loss_module=loss_module, **init_kwargs).cuda().bfloat16().eval()

    B = 1   # _DummyBackbone only reads B/device/dtype from data_tensor, emits an (8,16,16) feature map
    samples = {
        "data_tensor": torch.randn(B, 8, 16, 16, 1, dtype=torch.bfloat16, device="cuda"),
        "metainfo": {
            "padding_mask": torch.zeros(B, 8, 16, 16, dtype=torch.bool, device="cuda"),
            "targets": [{"labels": torch.tensor([1], device="cuda"),
                         "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.2, 0.2, 0.2]], device="cuda")}],
        },
    }
    with torch.no_grad():
        losses, outputs = model(samples)

    assert outputs["pred_logits"].shape == (B, NUM_Q, NUM_CLASSES)
    assert outputs["pred_boxes"].shape == (B, NUM_Q, 6)
    assert outputs["pred_logits"].dtype == torch.bfloat16
    assert torch.isfinite(outputs["pred_logits"].float()).all() and torch.isfinite(outputs["pred_boxes"].float()).all()
    assert "enc_outputs" in outputs and "aux_outputs" not in outputs
    assert torch.isfinite(losses["step_loss"]).all() and losses["step_loss"] > 0
