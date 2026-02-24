import pytest
import torch
from torch import nn

from cell_observatory_platform.models.heads.maskdino_decoder import MaskDINODecoder
from cell_observatory_platform.models.heads.maskdino_head import MaskDINOHead
from cell_observatory_platform.models.heads.pixel_decoders import MaskDINOEncoder
from cell_observatory_platform.models.layers.matchers import HungarianMatcher
from cell_observatory_platform.models.meta_arch.maskdino import MaskDINO
from cell_observatory_platform.training import losses as losses_mod
from cell_observatory_platform.training.losses import DETR_Set_Loss

try:
    from ops3d import _C
    OPS3D_AVAILABLE = True
except ImportError:
    OPS3D_AVAILABLE = False



CUDA_AVAILABLE = torch.cuda.is_available()


class DummyBackbone(nn.Module):
    """
    Returns feature maps matching the input_shape we pass to MaskDINOEncoder.

    feature_shapes: dict {feature_name: (D, H, W)}
    channels: number of channels per feature map
    """

    def __init__(self, feature_shapes, channels):
        super().__init__()
        self.feature_shapes = feature_shapes
        self.channels = channels

    def forward(self, x):
        if isinstance(x, dict):
            x = x["data_tensor"]
        B = x.shape[0]
        device = x.device
        dtype = x.dtype
        features = {}
        for name, (D, H, W) in self.feature_shapes.items():
            features[name] = torch.randn(B, self.channels, D, H, W, device=device, dtype=dtype)
        return features

    def forward_features(self, x):
        return self.forward(x)


@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="FlashDeformAttn3D (OPS3D_AVAILABLE) is not installed.",
)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for this test")
def test_maskdino_forward_train(monkeypatch):
    """
    Full forward pass:
      DummyBackbone -> MaskDINOEncoder -> MaskDINODecoder -> DETR_Set_Loss -> MaskDINO
    with no denoising queries, real matcher and real losses.
    """
    # avoid torch.distributed in tests
    monkeypatch.setattr(losses_mod, "get_world_size", lambda: 1)
    monkeypatch.setattr(losses_mod, "is_torch_dist_initialized", lambda: False)

    device = torch.device("cuda")

    batch_size = 1
    in_channels = 16
    conv_dim = 96  # must be divisible by 32 and 3 due to use of GroupNorm
    mask_dim = 8
    num_classes = 3
    num_queries = 5

    D_in = H_in = W_in = 64

    feature_names = ["feat0", "feat1"]
    feature_shapes = {
        "feat0": (32, 32, 32),
        "feat1": (16, 16, 16),
    }

    input_shape_metadata = {
        "feat0": {"stride": 4, "channels": in_channels},
        "feat1": {"stride": 8, "channels": in_channels},
    }

    pixel_decoder = MaskDINOEncoder(
        input_shape_metadata=input_shape_metadata,
        transformer_in_features=feature_names,
        target_min_stride=4,
        total_num_feature_levels=2,  # no extra downsampled levels
        transformer_encoder_dropout=0.0,
        transformer_encoder_num_heads=4,
        transformer_encoder_dim_feedforward=64,
        num_transformer_encoder_layers=2,
        conv_dim=conv_dim,
        mask_dim=mask_dim,
        norm=None,
    ).to(device)

    decoder = MaskDINODecoder(
        in_channels=conv_dim,
        num_classes=num_classes,
        hidden_dim=conv_dim,
        num_queries=num_queries,
        feedforward_dim=64,
        decoder_num_layers=2,
        mask_dim=mask_dim,
        enforce_input_projection=False,
        two_stage_flag=False,
        denoise_queries_flag=False,
        noise_scale=0.0,
        total_denosing_queries=0,
        initialize_box_type=None,
        with_initial_prediction=True,
        learn_query_embeddings=True,
        total_num_feature_levels=2,
        dropout=0.0,
        activation="RELU",
        num_heads=4,
        decoder_num_points=4,
        return_intermediates_decoder=True,
        query_dim=6,
        share_decoder_layers=False,
    ).to(device)

    head = MaskDINOHead(
        num_classes=num_classes,
        pixel_decoders=pixel_decoder,
        decoders=decoder,
    ).to(device)

    backbone = DummyBackbone(
        feature_shapes=feature_shapes,
        channels=in_channels,
    ).to(device)

    matcher = HungarianMatcher().to(device)

    loss_weight_dict = {
        "loss_ce": 1.0,
        "loss_bbox": 1.0,
        "loss_giou": 1.0,
        "loss_mask": 1.0,
        "loss_dice": 1.0,
    }

    criterion = DETR_Set_Loss(
        num_classes=num_classes,
        matcher=matcher,
        loss_weight_dict=loss_weight_dict,
        no_object_loss_weight=0.1,
        losses=["labels", "boxes", "masks"],
        num_points=4,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        denoise=False,
        with_segmentation=True,
        denoise_losses=[],
        semantic_ce_loss=False,  # use focal-loss branch
        focal_alpha=0.25,
    ).to(device)

    model = MaskDINO(
        matcher=matcher,
        backbone=backbone,
        criterion=criterion,
        segmentation_head=head,
        num_queries=num_queries,
        instance_segmentation_flag=True,
        topk_per_image=4,
        focus_on_boxes=False,
    ).to(device)

    data_tensor = torch.randn(batch_size, 1, D_in, H_in, W_in, device=device)

    labels = torch.randint(0, num_classes, (2,), device=device)
    boxes = torch.rand(2, 6, device=device)
    masks = torch.randint(0, 2, (2, D_in, H_in, W_in), device=device, dtype=torch.float32)

    mask_ids = torch.arange(1, 3, device=device)
    label_map = torch.zeros(D_in, H_in, W_in, dtype=torch.long, device=device)
    for i in range(2):
        label_map[masks[i] > 0] = i + 1

    data_sample = {
        "data_tensor": data_tensor,
        "metainfo": {
            "targets": [[{"labels": labels, "boxes": boxes, "masks": masks, "mask_ids": mask_ids, "label_map": label_map}]],
            "image_sizes": [(D_in, H_in, W_in)],
            "orig_image_sizes": [(D_in, H_in, W_in)],
        },
    }

    model.train()
    losses, outputs = model(data_sample)

    # --- outputs sanity ---
    assert "pred_logits" in outputs
    assert "pred_boxes" in outputs
    assert "pred_masks" in outputs

    B, Q, C = outputs["pred_logits"].shape
    assert B == batch_size
    assert Q == num_queries
    assert C == num_classes

    assert outputs["pred_boxes"].shape == (batch_size, num_queries, 6)

    # masks: (B, Q, D', H', W')
    assert outputs["pred_masks"].dim() == 5
    assert outputs["pred_masks"].shape[0] == batch_size
    assert outputs["pred_masks"].shape[1] == num_queries

    # --- losses sanity & weighting ---
    expected_base_keys = set(loss_weight_dict.keys())
    assert expected_base_keys.issubset(set(losses.keys()))
    assert "step_loss" in losses

    for k, v in losses.items():
        assert v.dim() == 0
        assert torch.isfinite(v)
        assert v.requires_grad


@pytest.mark.skipif(
    not OPS3D_AVAILABLE,
    reason="FlashDeformAttn3D (OPS3D_AVAILABLE) is not installed.",
)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for this test")
def test_maskdino_predict():
    device = torch.device("cuda")

    batch_size = 1
    in_channels = 16
    conv_dim = 96
    mask_dim = 8
    num_classes = 3
    num_queries = 5
    topk_per_image = 4

    D_in = H_in = W_in = 64

    feature_names = ["feat0", "feat1"]
    feature_shapes = {
        "feat0": (32, 32, 32),
        "feat1": (16, 16, 16),
    }

    input_shape_metadata = {
        "feat0": {"stride": 4, "channels": in_channels},
        "feat1": {"stride": 8, "channels": in_channels},
    }

    pixel_decoder = MaskDINOEncoder(
        input_shape_metadata=input_shape_metadata,
        transformer_in_features=feature_names,
        target_min_stride=4,
        total_num_feature_levels=2,
        transformer_encoder_dropout=0.0,
        transformer_encoder_num_heads=4,
        transformer_encoder_dim_feedforward=64,
        num_transformer_encoder_layers=2,
        conv_dim=conv_dim,
        mask_dim=mask_dim,
        norm=None,
    ).to(device)

    decoder = MaskDINODecoder(
        in_channels=conv_dim,
        num_classes=num_classes,
        hidden_dim=conv_dim,
        num_queries=num_queries,
        feedforward_dim=64,
        decoder_num_layers=2,
        mask_dim=mask_dim,
        enforce_input_projection=False,
        two_stage_flag=False,
        denoise_queries_flag=False,
        noise_scale=0.0,
        total_denosing_queries=0,
        initialize_box_type=None,
        with_initial_prediction=True,
        learn_query_embeddings=True,
        total_num_feature_levels=2,
        dropout=0.0,
        activation="RELU",
        num_heads=4,
        decoder_num_points=4,
        return_intermediates_decoder=True,
        query_dim=6,
        share_decoder_layers=False,
    ).to(device)

    head = MaskDINOHead(
        num_classes=num_classes,
        pixel_decoders=pixel_decoder,
        decoders=decoder,
    ).to(device)

    backbone = DummyBackbone(
        feature_shapes=feature_shapes,
        channels=in_channels,
    ).to(device)

    class DummyCriterion(nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_weight_dict = {}

        def forward(self, *args, **kwargs):
            return {}

    matcher = HungarianMatcher().to(device)
    criterion = DummyCriterion().to(device)

    model = MaskDINO(
        matcher=matcher,
        backbone=backbone,
        criterion=criterion,
        segmentation_head=head,
        num_queries=num_queries,
        instance_segmentation_flag=True,
        topk_per_image=topk_per_image,
        focus_on_boxes=False,
    ).to(device)

    model.eval()

    data_tensor = torch.randn(batch_size, 1, D_in, H_in, W_in, device=device)
    data_sample = {
        "data_tensor": data_tensor,
        "metainfo": {
            "targets": None,
            "image_sizes": [(D_in, H_in, W_in)],
            "orig_image_sizes": [torch.tensor([D_in, H_in, W_in], dtype=torch.long, device=device)],
        },
    }

    with torch.no_grad():
        predictions = model.predict(data_sample)

    assert len(predictions) == batch_size
    pred = predictions[0]

    assert "masks" in pred
    assert "boxes" in pred
    assert "predicted_labels" in pred

    # masks: (topk, D, H, W) after upsampling
    assert pred["masks"].dim() == 4
    assert pred["masks"].shape[0] == topk_per_image
    assert pred["masks"].shape[1:] == (D_in, H_in, W_in)

    # boxes: (topk, 6)
    assert pred["boxes"].shape == (topk_per_image, 6)

    # scores: (topk,)
    assert pred["predicted_labels"].dim() == 1
    assert pred["predicted_labels"].shape[0] == topk_per_image
    assert torch.isfinite(pred["predicted_labels"]).all()
