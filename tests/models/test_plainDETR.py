import pytest
import torch
from astropy import conf
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cell_observatory_platform.models.meta_arch.plainDETR import PlainDETRReParam

pytestmark = pytest.mark.skip(reason="This module is temporarily disabled till we fix outdated configs")


def _build_plain_detr_reparam_cfg():
    cfg_dict = {
        "BUILD": "cell_observatory_platform.models.meta_arch.plainDETR.BUILD",
        "backbone_wrapper_args": {
            "BUILD": "cell_observatory_platform.models.heads.plain_detr_backbone.BUILD",
            "backbone_embed_dims": [1024, 1024, 1024, 1024],
            "train_backbone": True,
            "use_layernorm": True,
            "blocks_to_train": None,
            "out_layers": [10, 14, 18, 22],
            "adapter_out_layers": [1],
            "backbone_args": {
                "model": "FinetuneMaskedAutoEncoder",
                # pick a reasonable 3D layout + shape
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

    return OmegaConf.create(cfg_dict)


# @pytest.mark.cpu
# def test_plain_detr_reparam_instantiates_from_config():
#     cfg = _build_plain_detr_reparam_cfg()
#     model = instantiate(cfg)

#     assert isinstance(model, PlainDETRReParam)
#     assert model.two_stage is True
#     assert model.with_box_refine is True
#     assert model.num_queries_one2one == 300
#     assert model.num_queries_one2many == 1500


def test_plain_detr_reparam_forward_pass_smoke_bf16():
    # cfg = OmegaConf.load("configs/models/meta_arch/plainDETR/mae_large.yaml").resolve()
    # print(cfg)
    cfg = _build_plain_detr_reparam_cfg()
    model: PlainDETRReParam = instantiate(cfg)
    model = model.to(torch.bfloat16, device="cuda")
    model.eval()

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
