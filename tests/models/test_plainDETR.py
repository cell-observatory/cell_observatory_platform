import inspect

import pytest

import torch
import torch.nn as nn

from omegaconf import OmegaConf
from hydra.utils import get_method

from cell_observatory_platform.models.meta_arch.plainDETR import PlainDETR, PlainDETRReParam


class _DummyBackbone(nn.Module):
    """
    Minimal backbone stub for the smoke test.
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


def _build_plain_detr_reparam_cfg():
    plain_detr_cfg = {
        "BUILD": "cell_observatory_platform.models.meta_arch.plainDETR.BUILD",
        "backbone_wrapper_args": {
            "BUILD": "cell_observatory_platform.models.backbones.plain_detr_backbone.BUILD",
            "backbone_embed_dims": [1024, 1024, 1024, 1024],
            "train_backbone": True,
            "use_layernorm": True,
            "blocks_to_train": None,
            "out_layers": [10, 14, 18, 22],
            "adapter_out_layers": [1],
            "backbone_args": {
                "model": "FinetuneMaskedAutoEncoder",
                "input_fmt": "ZYXC",
                "input_shape": [128, 256, 256, 2],
                "patch_shape": [16, 16, 16],
                "input_channels": 1,
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
                "mlp_ratio": 4.0,
                "proj_drop_rate": 0.0,
                "att_drop_rate": 0.0,
                "drop_path_rate": 0.1,
                "init_std": 0.02,
                "fixed_dropout_depth": False,
                "norm_layer": "RmsNorm",
                "act_layer": "SiLU",
                "mlp_layer": "SwiGLU",
                "abs_sincos_enc": False,
                "rope_pos_enc": True,
                "rope_random_rotation_per_head": True,
                "rope_mixed": False,
                "rope_theta": 100.0,
                "weight_init_type": "mae",
                "mlp_wide_silu": False,
                "loss_fn": "l2_masked",
                "decoder": "plainDETR",
                "task": "instance_segmentation",
                "output_channels": None,
                "decoder_args": {
                    "encoder_out_layers": [10, 14, 18, 22],
                },
            },
        },
        "adapter_args": {
            "BUILD": "cell_observatory_platform.models.adapters.vit_adapter.BUILD",
            "input_format": "ZYXC",
            "input_shape": [128, 256, 256, 2],
            "patch_shape": [16, 16, 16],
            "input_channels": 1,
            "dtype": "float32",
            "dim": 3,
            "backbone_embed_dim": 1024,
            "num_backbone_features": 4,
            "add_vit_feature": True,
            "conv_inplane": 64,
            "use_deform_attention": False,
            "n_points": 4,
            "n_levels": 1,
            "deform_num_heads": 16,
            "drop_path_rate": 0.3,
            "init_values": 0.0,
            "with_cffn": True,
            "cffn_ratio": 0.5,
            "deform_ratio": 0.5,
            "use_extra_extractor": True,
            "strategy": "axial",
            "spatial_prior_module_strides": {
                "stem1": [2, 2, 2],
                "stem2": [1, 1, 1],
                "stem3": [1, 1, 1],
                "maxpool": 2,
                "stage2": [2, 2, 2],
                "stage3": [2, 2, 2],
                "stage4": [2, 2, 2],
            },
        },
        "transformer_args": {
            "BUILD": "cell_observatory_platform.models.heads.plain_detr_transformer.BUILD",
            "d_model": 384,
            "nheads": 8,
            "num_feature_levels": 1,
            "two_stage": True,
            "norm_type": "pre_norm",
            "decoder_type": "global_rpe_decomp",
            "proposal_feature_levels": 3,
            "proposal_in_stride": 8,
            "proposal_tgt_strides": [4, 8, 16],
            "proposal_min_size": 16,
            "add_transformer_encoder": True,
            "dim_feedforward": 2048,
            "dropout": 0.0,
            "activation": "relu",
            "normalize_before": True,
            "num_encoder_layers": 6,
            "global_decoder_args": {
                "hidden_dim": 384,
                "nheads": 8,
                "dropout": 0.0,
                "proposal_in_stride": 8,
                "norm_type": "pre_norm",
                "dim_feedforward": 2048,
                "num_heads": 8,
                "qkv_bias": True,
                "qk_scale": None,
                "attn_drop": 0.0,
                "proj_drop": 0.0,
                "dec_layers": 6,
                "look_forward_twice": True,
                "rpe_hidden_dim": 512,
                "rpe_type": "linear",
                "reparam": True,
            },
        },
        "criterion_args": {
            "BUILD": "cell_observatory_platform.training.losses.build_plainDETR_Set_Loss",
            "num_classes": 2,
            "weight_dict": {"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0},
            "losses": ["labels", "boxes", "cardinality"],
            "focal_alpha": 0.25,
            "reparam": True,
            "matcher_args": {
                "BUILD": "cell_observatory_platform.models.utils.matchers.build_plain_detr_matcher",
                "cost_class": 2.0,
                "cost_bbox": 5.0,
                "cost_giou": 2.0,
                "cost_bbox_type": "reparam",
            },
        },
        "backbone_embed_dim": 1024,
        "num_classes": 2,
        "num_feature_levels": 1,
        "with_box_refine": True,
        "two_stage": True,
        "aux_loss": False,
        "num_queries_one2one": 200,
        "num_queries_one2many": 0,
        "k_one2many": 0,
        "mixed_selection": True,
        "reparam": True,
        "lambda_one2many": 1.0,
        "normalize_pos_encodings": True,
    }

    return OmegaConf.create(plain_detr_cfg)


def test_plain_detr_reparam_forward_pass_smoke_bf16():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available for PlainDETRReParam")

    model_cfg = _build_plain_detr_reparam_cfg()

    # ------------------------------------------------------------------
    # 1) Dummy backbone
    # ------------------------------------------------------------------
    backbone_embed_dim = model_cfg["backbone_embed_dim"]
    backbone = _DummyBackbone(backbone_embed_dim=backbone_embed_dim)

    # ------------------------------------------------------------------
    # 2) Build transformer
    # ------------------------------------------------------------------
    transformer_cfg = model_cfg["transformer_args"]
    build_transformer = get_method(transformer_cfg["BUILD"])

    reparam = model_cfg.get("reparam", True)
    num_queries_one2one = model_cfg.get("num_queries_one2one")
    num_queries_one2many = model_cfg.get("num_queries_one2many")
    mixed_selection = model_cfg.get("mixed_selection")

    transformer_build_cfg = dict(transformer_cfg)
    transformer_build_cfg["reparam"] = reparam
    transformer_build_cfg["num_queries_one2one"] = num_queries_one2one
    transformer_build_cfg["num_queries_one2many"] = num_queries_one2many
    transformer_build_cfg["mixed_selection"] = mixed_selection

    transformer = build_transformer(transformer_build_cfg)

    # ------------------------------------------------------------------
    # 3) Build loss module (criterion)
    # ------------------------------------------------------------------
    crit_cfg = model_cfg["criterion_args"]
    build_loss = get_method(crit_cfg["BUILD"])

    num_classes = model_cfg["num_classes"]
    two_stage = model_cfg.get("two_stage", False)
    aux_loss = model_cfg.get("aux_loss", True)

    loss_module = build_loss(
        crit_cfg,
        num_classes=num_classes,
        two_stage=two_stage,
        reparam=reparam,
        aux_loss=aux_loss,
        dec_layers=transformer.decoder.num_layers,
    )

    # ------------------------------------------------------------------
    # 4) Extract PlainDETR __init__ kwargs from top-level cfg
    # ------------------------------------------------------------------
    sig = inspect.signature(PlainDETR.__init__)
    allowed = set(sig.parameters.keys()) - {"self", "backbone", "transformer", "loss_module"}

    ignore_keys = {
        "_target_",
        "BUILD",
        "backbone_wrapper_args",
        "adapter_args",
        "transformer_args",
        "criterion_args",
    }

    init_kwargs = {}
    for k, v in model_cfg.items():
        if k in ignore_keys:
            continue
        if k in allowed:
            init_kwargs[k] = v

    # ------------------------------------------------------------------
    # 5) Choose PlainDETR vs PlainDETRReParam and instantiate
    # ------------------------------------------------------------------
    plain_detr_cls = PlainDETRReParam if reparam else PlainDETR

    model = plain_detr_cls(
        backbone=backbone,
        transformer=transformer,
        loss_module=loss_module,
        **init_kwargs,
    )

    # ------------------------------------------------------------------
    # 6) Forward pass smoke test
    # ------------------------------------------------------------------
    model = model.cuda().bfloat16()

    batch_size = 1
    D, H, W, C = 128, 256, 256, 1
    data_tensor = torch.randn(batch_size, D, H, W, C, dtype=torch.bfloat16, device="cuda")
    padding_mask = torch.zeros(batch_size, D, H, W, dtype=torch.bool, device="cuda")

    samples = {
        "data_tensor": data_tensor,
        "metainfo": {"padding_mask": padding_mask},
    }

    with torch.no_grad():
        outputs = model._forward(samples)

    assert "pred_logits" in outputs
    assert "pred_boxes" in outputs