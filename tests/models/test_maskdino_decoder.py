import pytest
import torch

from cell_observatory_platform.models.heads.maskdino_decoder import MaskDINODecoder

try:
    from ops3d import _C

    MSDEFORM_ATTN_AVAILABLE = True
except ImportError:
    MSDEFORM_ATTN_AVAILABLE = False


CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_maskdino_gen_encoder_output_proposals():
    device = torch.device("cuda")

    batch_size = 2
    hidden_dim = 64
    num_levels = 3

    # realistic 3D pyramid: 32^3, 16^3, 8^3
    level_shapes = torch.tensor(
        [
            [32, 32, 32],
            [16, 16, 16],
            [8, 8, 8],
        ],
        dtype=torch.long,
        device=device,
    )  # (num_levels, 3)

    tokens_per_level = level_shapes.prod(dim=1)  # (num_levels,)
    total_tokens = int(tokens_per_level.sum().item())

    # memory: (N, SUM{D*H*W}, C)
    memory = torch.randn(batch_size, total_tokens, hidden_dim, device=device)
    # no padding
    memory_padding_mask = torch.zeros(batch_size, total_tokens, dtype=torch.bool, device=device)

    output_memory, output_proposals = MaskDINODecoder.gen_encoder_output_proposals(
        memory, memory_padding_mask, level_shapes
    )

    # shapes should be preserved / extended in expected way
    assert output_memory.shape == (batch_size, total_tokens, hidden_dim)
    assert output_proposals.shape == (batch_size, total_tokens, 6)


@pytest.mark.skipif(
    not MSDEFORM_ATTN_AVAILABLE,
    reason="FlashDeformAttn3D (MSDEFORM_ATTN_AVAILABLE) is not installed.",
)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_maskdino_decoder_forward_no_two_stage_no_denoise():
    device = torch.device("cuda")

    batch_size = 2
    in_channels = 64
    hidden_dim = 64
    num_classes = 3
    num_queries = 10
    feedforward_dim = 128
    decoder_num_layers = 2
    mask_dim = 16
    num_feature_levels = 3

    # construct the decoder with:
    # - no two-stage
    # - no denoising
    # - query_dim=6 (x, y, z, w, h, d)
    decoder = MaskDINODecoder(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        feedforward_dim=feedforward_dim,
        decoder_num_layers=decoder_num_layers,
        mask_dim=mask_dim,
        enforce_input_projection=False,
        two_stage_flag=False,
        denoise_queries_flag=False,
        noise_scale=0.0,
        total_denosing_queries=0,
        initialize_box_type=None,
        with_initial_prediction=True,
        learn_query_embeddings=True,
        total_num_feature_levels=num_feature_levels,
        dropout=0.0,
        activation="RELU",
        num_heads=8,
        decoder_num_points=4,
        return_intermediates_decoder=True,
        query_dim=6,
        share_decoder_layers=False,
    ).to(device)

    # 3D FPN-like feature maps: (B, C, D, H, W)
    # order here is [fine, mid, coarse]; the decoder just treats them as levels
    x = [
        torch.randn(batch_size, in_channels, 32, 32, 32, device=device),
        torch.randn(batch_size, in_channels, 16, 16, 16, device=device),
        torch.randn(batch_size, in_channels, 8, 8, 8, device=device),
    ]

    # pixel decoder output: channels must match mask_dim
    # shape: (B, mask_dim, D_mask, H_mask, W_mask)
    pixel_decoder_output = torch.randn(batch_size, mask_dim, 16, 16, 16, device=device)

    masks = None
    targets = None  # no denoising, so not needed

    decoder.eval()  # shapes-only test; no need for training mode
    outputs, denoise_metadata = decoder(x, pixel_decoder_output, masks, targets)

    # top-level keys & shapes
    assert "pred_logits" in outputs
    assert "pred_masks" in outputs
    assert "pred_boxes" in outputs
    assert "auxiliary_outputs" in outputs

    pred_logits = outputs["pred_logits"]
    pred_masks = outputs["pred_masks"]
    pred_boxes = outputs["pred_boxes"]
    aux = outputs["auxiliary_outputs"]

    # logits: (B, num_queries, num_classes)
    assert pred_logits.shape == (batch_size, num_queries, num_classes)

    # masks: (B, num_queries, D_mask, H_mask, W_mask)
    assert pred_masks.shape == (batch_size, num_queries, 16, 16, 16)

    # boxes: (B, num_queries, 6) – (cx, cy, cz, w, h, d) in [0,1]
    assert pred_boxes.shape == (batch_size, num_queries, 6)

    # aux outputs: list of dicts (can be empty, but must be a list)
    assert isinstance(aux, list)
    for entry in aux:
        assert "pred_logits" in entry
        assert "pred_masks" in entry
        assert "pred_boxes" in entry


def _make_pyramid_features(batch_size, in_channels, device):
    # 3D FPN-like pyramid: (B, C, D, H, W)
    return [
        torch.randn(batch_size, in_channels, 32, 32, 32, device=device),
        torch.randn(batch_size, in_channels, 16, 16, 16, device=device),
        torch.randn(batch_size, in_channels, 8, 8, 8, device=device),
    ]


@pytest.mark.skipif(
    not MSDEFORM_ATTN_AVAILABLE,
    reason="FlashDeformAttn3D (MSDEFORM_ATTN_AVAILABLE) is not installed.",
)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_maskdino_decoder_forward_two_stage_no_denoise():
    device = torch.device("cuda")

    batch_size = 2
    in_channels = 64
    hidden_dim = 64
    num_classes = 3
    num_queries = 10
    feedforward_dim = 128
    decoder_num_layers = 2
    mask_dim = 16
    num_feature_levels = 3

    decoder = MaskDINODecoder(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        feedforward_dim=feedforward_dim,
        decoder_num_layers=decoder_num_layers,
        mask_dim=mask_dim,
        enforce_input_projection=False,
        two_stage_flag=True,  # two-stage ON
        denoise_queries_flag=False,  # no denoise
        noise_scale=0.0,
        total_denosing_queries=0,
        initialize_box_type=None,  # no mask->box init
        with_initial_prediction=True,
        learn_query_embeddings=False,  # use encoder proposals as queries
        total_num_feature_levels=num_feature_levels,
        dropout=0.0,
        activation="RELU",
        num_heads=8,
        decoder_num_points=4,
        return_intermediates_decoder=True,
        query_dim=6,
        share_decoder_layers=False,
    ).to(device)

    x = _make_pyramid_features(batch_size, in_channels, device)
    pixel_decoder_output = torch.randn(batch_size, mask_dim, 16, 16, 16, device=device)

    masks = None
    targets = None

    decoder.eval()
    outputs, denoise_metadata = decoder(x, pixel_decoder_output, masks, targets)

    assert denoise_metadata is None

    pred_logits = outputs["pred_logits"]
    pred_masks = outputs["pred_masks"]
    pred_boxes = outputs["pred_boxes"]
    aux = outputs["auxiliary_outputs"]

    assert pred_logits.shape == (batch_size, num_queries, num_classes)
    assert pred_masks.shape == (batch_size, num_queries, 16, 16, 16)
    assert pred_boxes.shape == (batch_size, num_queries, 6)
    assert isinstance(aux, list)
    for entry in aux:
        assert "pred_logits" in entry
        assert "pred_masks" in entry
        assert "pred_boxes" in entry

    # two-stage should also populate 'intermediates'
    assert "intermediates" in outputs
    inter = outputs["intermediates"]
    assert "pred_logits" in inter
    assert "pred_boxes" in inter
    assert "pred_masks" in inter


@pytest.mark.skipif(
    not MSDEFORM_ATTN_AVAILABLE,
    reason="FlashDeformAttn3D (MSDEFORM_ATTN_AVAILABLE) is not installed.",
)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_maskdino_decoder_forward_two_stage_with_box_init_mask2box():
    device = torch.device("cuda")

    batch_size = 2
    in_channels = 64
    hidden_dim = 64
    num_classes = 3
    num_queries = 10
    feedforward_dim = 128
    decoder_num_layers = 2
    mask_dim = 16
    num_feature_levels = 3

    decoder = MaskDINODecoder(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        feedforward_dim=feedforward_dim,
        decoder_num_layers=decoder_num_layers,
        mask_dim=mask_dim,
        enforce_input_projection=False,
        two_stage_flag=True,  # two-stage ON
        denoise_queries_flag=False,
        noise_scale=0.0,
        total_denosing_queries=0,
        initialize_box_type="mask2box",  # hit mask->box initialization path
        with_initial_prediction=True,
        learn_query_embeddings=False,
        total_num_feature_levels=num_feature_levels,
        dropout=0.0,
        activation="RELU",
        num_heads=8,
        decoder_num_points=4,
        return_intermediates_decoder=True,
        query_dim=6,
        share_decoder_layers=False,
    ).to(device)

    x = _make_pyramid_features(batch_size, in_channels, device)
    pixel_decoder_output = torch.randn(batch_size, mask_dim, 16, 16, 16, device=device)

    masks = None
    targets = None

    decoder.eval()
    outputs, denoise_metadata = decoder(x, pixel_decoder_output, masks, targets)

    assert denoise_metadata is None

    pred_logits = outputs["pred_logits"]
    pred_masks = outputs["pred_masks"]
    pred_boxes = outputs["pred_boxes"]

    # same basic shape expectations
    assert pred_logits.shape == (batch_size, num_queries, num_classes)
    assert pred_masks.shape == (batch_size, num_queries, 16, 16, 16)
    assert pred_boxes.shape == (batch_size, num_queries, 6)


@pytest.mark.skipif(
    not MSDEFORM_ATTN_AVAILABLE,
    reason="FlashDeformAttn3D (MSDEFORM_ATTN_AVAILABLE) is not installed.",
)
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is required for these tests")
def test_maskdino_decoder_forward_with_denoise():
    device = torch.device("cuda")

    batch_size = 2
    in_channels = 64
    hidden_dim = 64
    num_classes = 4
    num_queries = 6
    feedforward_dim = 128
    decoder_num_layers = 2
    mask_dim = 16
    num_feature_levels = 3

    decoder = MaskDINODecoder(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        feedforward_dim=feedforward_dim,
        decoder_num_layers=decoder_num_layers,
        mask_dim=mask_dim,
        enforce_input_projection=False,
        two_stage_flag=False,  # no two-stage, use learnable queries
        denoise_queries_flag=True,  # enable denoising pipeline
        noise_scale=0.5,
        total_denosing_queries=8,  # some non-zero number
        initialize_box_type=None,
        with_initial_prediction=True,
        learn_query_embeddings=True,
        total_num_feature_levels=num_feature_levels,
        dropout=0.0,
        activation="RELU",
        num_heads=8,
        decoder_num_points=4,
        return_intermediates_decoder=True,
        query_dim=6,
        share_decoder_layers=False,
    ).to(device)

    x = _make_pyramid_features(batch_size, in_channels, device)
    pixel_decoder_output = torch.randn(batch_size, mask_dim, 16, 16, 16, device=device)

    masks = None

    # simple synthetic targets with at least one GT per image
    targets = []
    for b in range(batch_size):
        num_gt = 2
        labels = torch.randint(low=0, high=num_classes, size=(num_gt,), device=device)
        # normalized 6D boxes: (cx, cy, cz, w, h, d) in [0, 1]
        boxes = torch.rand(num_gt, 6, device=device)
        targets.append({"labels": labels, "boxes": boxes})

    decoder.train()  # needed to hit denoising path
    outputs, denoise_metadata = decoder(x, pixel_decoder_output, masks, targets)

    # denoise pipeline should have been active
    assert denoise_metadata is not None
    assert "max_query_pad_size" in denoise_metadata
    assert "denoise_queries_per_label" in denoise_metadata

    pred_logits = outputs["pred_logits"]
    pred_masks = outputs["pred_masks"]
    pred_boxes = outputs["pred_boxes"]
    aux = outputs["auxiliary_outputs"]

    # After denoise_post_process, outputs should correspond to *non-denoising* queries only
    assert pred_logits.shape[:2] == (batch_size, num_queries)
    assert pred_logits.shape[2] == num_classes

    assert pred_masks.shape[:2] == (batch_size, num_queries)
    assert pred_masks.shape[2:] == (16, 16, 16)

    assert pred_boxes.shape == (batch_size, num_queries, 6)

    assert isinstance(aux, list)
    for entry in aux:
        assert "pred_logits" in entry
        assert "pred_masks" in entry
        assert "pred_boxes" in entry
