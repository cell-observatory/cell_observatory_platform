import pytest
import torch

from cell_observatory_platform.models.heads.maskdino_decoder import MaskDINODecoder


def _make_tiny_decoder(device="cpu"):
    return MaskDINODecoder(
        in_channels=8,
        num_classes=3,
        hidden_dim=16,
        num_queries=4,
        feedforward_dim=32,
        decoder_num_layers=1,
        mask_dim=5,
        enforce_input_projection=False,
        two_stage_flag=False,
        denoise_queries_flag=False,
        noise_scale=0.0,
        total_denosing_queries=0,
        initialize_box_type=None,
        with_initial_prediction=True,
        learn_query_embeddings=True,
        total_num_feature_levels=1,
        dropout=0.0,
        activation="RELU",
        num_heads=2,
        decoder_num_points=2,
        return_intermediates_decoder=True,
        query_dim=6,
        share_decoder_layers=False,
    ).to(device)


def _pyramid(batch_size, in_channels, device):
    # 3D FPN-like pyramid: (B, C, D, H, W), [fine, mid, coarse]
    return [
        torch.randn(batch_size, in_channels, 8, 8, 8, device=device),
        torch.randn(batch_size, in_channels, 4, 4, 4, device=device),
        torch.randn(batch_size, in_channels, 2, 2, 2, device=device),
    ]


def test_forward_prediction_heads_returns_mask_embeddings_without_masks():
    decoder = _make_tiny_decoder()
    output_queries = torch.randn(4, 2, 16)
    pixel_decoder_output = torch.randn(2, 5, 2, 2, 2)

    result = decoder.forward_prediction_heads(
        output_queries,
        pixel_decoder_output,
        predict_masks=False,
        return_mask_embeddings=True,
    )
    assert len(result) == 3
    outputs_class, outputs_mask, mask_embeddings = result

    assert outputs_class.shape == (2, 4, 3)
    assert outputs_mask is None
    assert mask_embeddings is not None
    assert mask_embeddings.shape == (2, 4, 5)


def test_forward_prediction_heads_mask_einsum_parity():
    decoder = _make_tiny_decoder()
    output_queries = torch.randn(4, 2, 16)
    pixel_decoder_output = torch.randn(2, 5, 2, 2, 2)

    result = decoder.forward_prediction_heads(
        output_queries,
        pixel_decoder_output,
        predict_masks=True,
        return_mask_embeddings=True,
    )
    assert len(result) == 3
    outputs_class, outputs_mask, mask_embeddings = result

    assert outputs_mask is not None
    assert mask_embeddings is not None
    expected = torch.einsum("bqc,bcdhw->bqdhw", mask_embeddings, pixel_decoder_output)
    assert outputs_class.shape == (2, 4, 3)
    torch.testing.assert_close(outputs_mask, expected)


def test_maskdino_gen_encoder_output_proposals():
    torch.manual_seed(0)
    device = torch.device("cpu")

    batch_size = 2
    hidden_dim = 64

    # small 3D pyramid: 8^3, 4^3, 2^3
    level_shapes = [(8, 8, 8), (4, 4, 4), (2, 2, 2)]

    total_tokens = sum(d * h * w for d, h, w in level_shapes)

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


def test_gen_encoder_output_proposals_coarser_level_gets_larger_prior():
    """Shapes arrive coarsest-first; the 0.05 * 2^k size prior must index from the
    fine end so the coarsest level carries the LARGEST anchor prior."""
    shapes = [(2, 2, 2), (4, 4, 4)]
    total = sum(d * h * w for d, h, w in shapes)
    memory = torch.zeros(1, total, 8)
    padding = torch.zeros(1, total, dtype=torch.bool)
    _, proposals = MaskDINODecoder.gen_encoder_output_proposals(memory, padding, shapes)

    # invert the logit transform to recover normalized whd
    whd = torch.sigmoid(proposals[..., 3:])
    n_coarse = shapes[0][0] * shapes[0][1] * shapes[0][2]
    coarse = whd[0, :n_coarse]
    fine = whd[0, n_coarse:]
    # finite entries only (out-of-range proposals are masked to inf pre-sigmoid)
    coarse = coarse[torch.isfinite(coarse).all(-1)]
    fine = fine[torch.isfinite(fine).all(-1)]
    assert coarse.numel() > 0 and fine.numel() > 0
    # lvl 0 = coarsest must carry the LARGER prior (0.05 * 2^(L-1-lvl))
    assert torch.allclose(coarse, torch.full_like(coarse, 0.10), atol=1e-6)
    assert torch.allclose(fine, torch.full_like(fine, 0.05), atol=1e-6)


def test_maskdino_decoder_forward_no_two_stage_no_denoise():
    torch.manual_seed(0)
    device = torch.device("cpu")

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
    x = _pyramid(batch_size, in_channels, device)

    # pixel decoder output: channels must match mask_dim
    # shape: (B, mask_dim, D_mask, H_mask, W_mask)
    pixel_decoder_output = torch.randn(batch_size, mask_dim, 4, 4, 4, device=device)

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
    assert pred_masks.shape == (batch_size, num_queries, 4, 4, 4)

    # boxes: (B, num_queries, 6) – (cx, cy, cz, w, h, d) in [0,1]
    assert pred_boxes.shape == (batch_size, num_queries, 6)
    assert ((pred_boxes >= 0) & (pred_boxes <= 1)).all()

    # aux outputs: one entry per decoder layer plus the initial prediction, minus the final one;
    # in eval mode only the final layer's masks are materialised (deep supervision is train-only)
    assert isinstance(aux, list)
    assert len(aux) == decoder_num_layers
    for entry in aux:
        assert entry["pred_logits"].shape == pred_logits.shape
        assert entry["pred_masks"] is None
        assert entry["pred_boxes"].shape == pred_boxes.shape


def test_forward_predict_mask_false_returns_embeddings_and_pixel_features():
    torch.manual_seed(0)
    device = torch.device("cpu")
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
    x = _pyramid(batch_size, in_channels, device)
    pixel_decoder_output = torch.randn(batch_size, mask_dim, 4, 4, 4, device=device)

    decoder.eval()
    outputs, denoise_metadata = decoder(
        x,
        pixel_decoder_output,
        masks=None,
        targets=None,
        predict_mask=False,
    )

    assert denoise_metadata is None
    assert outputs["pred_masks"] is None
    assert outputs["mask_embeddings"].shape == (batch_size, num_queries, mask_dim)
    assert outputs["pixel_decoder_output"] is pixel_decoder_output
    assert outputs["pixel_decoder_output"].shape == (batch_size, mask_dim, 4, 4, 4)


def test_maskdino_decoder_forward_two_stage_no_denoise():
    torch.manual_seed(0)
    device = torch.device("cpu")

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

    x = _pyramid(batch_size, in_channels, device)
    pixel_decoder_output = torch.randn(batch_size, mask_dim, 4, 4, 4, device=device)

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
    assert pred_masks.shape == (batch_size, num_queries, 4, 4, 4)
    assert pred_boxes.shape == (batch_size, num_queries, 6)
    assert isinstance(aux, list)
    assert len(aux) == decoder_num_layers
    for entry in aux:
        assert entry["pred_logits"].shape == pred_logits.shape
        assert entry["pred_masks"] is None  # eval mode: only the final layer's masks are materialised
        assert entry["pred_boxes"].shape == pred_boxes.shape

    # two-stage should also populate 'intermediates' with the encoder's top-k proposals
    assert "intermediates" in outputs
    inter = outputs["intermediates"]
    assert inter["pred_logits"].shape == (batch_size, num_queries, num_classes)
    assert inter["pred_boxes"].shape == (batch_size, num_queries, 6)
    assert ((inter["pred_boxes"] >= 0) & (inter["pred_boxes"] <= 1)).all()
    # the encoder proposals always carry masks (mask2box initialisation reads them)
    assert inter["pred_masks"].shape == (batch_size, num_queries, 4, 4, 4)


def test_maskdino_decoder_forward_two_stage_with_box_init_mask2box():
    torch.manual_seed(0)
    device = torch.device("cpu")

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

    x = _pyramid(batch_size, in_channels, device)
    pixel_decoder_output = torch.randn(batch_size, mask_dim, 4, 4, 4, device=device)

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
    assert pred_masks.shape == (batch_size, num_queries, 4, 4, 4)
    assert pred_boxes.shape == (batch_size, num_queries, 6)
    assert torch.isfinite(pred_boxes).all()
    assert ((pred_boxes >= 0) & (pred_boxes <= 1)).all()


def test_maskdino_decoder_forward_with_denoise():
    torch.manual_seed(0)
    device = torch.device("cpu")

    batch_size = 2
    in_channels = 64
    hidden_dim = 64
    num_classes = 4
    num_queries = 6
    feedforward_dim = 128
    decoder_num_layers = 2
    mask_dim = 16
    num_feature_levels = 3
    total_denosing_queries = 8

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
        total_denosing_queries=total_denosing_queries,
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

    x = _pyramid(batch_size, in_channels, device)
    pixel_decoder_output = torch.randn(batch_size, mask_dim, 4, 4, 4, device=device)

    masks = None

    # simple synthetic targets with at least one GT per image
    num_gt = 2
    targets = []
    for b in range(batch_size):
        labels = torch.randint(low=0, high=num_classes, size=(num_gt,), device=device)
        # normalized 6D boxes: (cx, cy, cz, w, h, d) in [0, 1]
        boxes = torch.rand(num_gt, 6, device=device)
        targets.append({"labels": labels, "boxes": boxes})

    decoder.train()  # needed to hit denoising path
    outputs, denoise_metadata = decoder(x, pixel_decoder_output, masks, targets)

    # denoise pipeline should have been active
    assert denoise_metadata is not None
    # total_denosing_queries shared over the max GT per image -> copies per label,
    # padded to max_labels * copies
    assert denoise_metadata["denoise_queries_per_label"] == total_denosing_queries // num_gt
    assert denoise_metadata["max_query_pad_size"] == num_gt * (total_denosing_queries // num_gt)

    pred_logits = outputs["pred_logits"]
    pred_masks = outputs["pred_masks"]
    pred_boxes = outputs["pred_boxes"]
    aux = outputs["auxiliary_outputs"]

    # denoise queries are stripped again: the returned predictions cover the matching queries only
    assert pred_logits.shape == (batch_size, num_queries, num_classes)
    assert pred_masks.shape == (batch_size, num_queries, 4, 4, 4)
    assert pred_boxes.shape == (batch_size, num_queries, 6)

    assert isinstance(aux, list)
    assert len(aux) == decoder_num_layers
    for entry in aux:
        assert entry["pred_logits"].shape == pred_logits.shape
        assert entry["pred_masks"].shape == pred_masks.shape
        assert entry["pred_boxes"].shape == pred_boxes.shape
