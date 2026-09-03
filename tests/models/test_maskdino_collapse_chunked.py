import torch

from cell_observatory_platform.models.meta_arch.maskdino import MaskDINO


def _make_model(focus_on_boxes, mask_chunk_size=1):
    model = MaskDINO.__new__(MaskDINO)
    model.mask_chunk_size = mask_chunk_size
    model.focus_on_boxes = focus_on_boxes
    return model


def _make_sample():
    pixel_decoder_output = torch.full((2, 2, 2, 2), -5.0)
    pixel_decoder_output[0, 0, 0, 0] = 5.0
    pixel_decoder_output[1, 1, 1, 1] = 5.0
    return {
        "mask_embeddings": torch.eye(2, dtype=torch.float32),
        "pixel_decoder_output": pixel_decoder_output,
        "topk_query_indices": torch.tensor([0, 1], dtype=torch.long),
        "topk_class_scores": torch.tensor([0.2, 0.9], dtype=torch.float32),
        "boxes": torch.zeros(2, 6, dtype=torch.float32),
        "eval_frame_size": (2, 2, 2),
    }


def test_collapse_focus_on_boxes_keeps_topk_scores():
    model = _make_model(focus_on_boxes=True)
    sample = _make_sample()

    pred = model._collapse_sample_chunked(sample)

    torch.testing.assert_close(pred["labels"], sample["topk_class_scores"])
    assert pred["masks"].dtype == torch.uint16
    assert pred["masks"][0, 0, 0].item() == 1
    assert pred["masks"][1, 1, 1].item() == 2


def test_collapse_focus_on_masks_multiplies_mask_confidence():
    """Instance score = class score x mean sigmoid probability inside the binarised mask."""
    model = _make_model(focus_on_boxes=False)
    sample = _make_sample()

    pred = model._collapse_sample_chunked(sample)

    # each query's mask has exactly one voxel with logit +5 (all others -5 are outside the
    # binarised mask), so mask confidence = mean sigmoid inside the mask = sigmoid(5)
    mask_confidence = torch.sigmoid(torch.tensor(5.0))
    torch.testing.assert_close(pred["labels"], sample["topk_class_scores"] * mask_confidence)
    assert pred["masks"].dtype == torch.uint16
    assert pred["masks"][0, 0, 0].item() == 1
    assert pred["masks"][1, 1, 1].item() == 2


def test_collapse_label_ids_follow_ascending_score_order():
    """Label-map id j corresponds to returned row j-1: labels are sorted ascending and
    boxes are permuted by the same order, even when topk scores arrive unsorted."""
    model = _make_model(focus_on_boxes=True, mask_chunk_size=2)  # final score == topk_class_scores

    # 3 disjoint one-voxel masks in a 3x1x1 volume, identity embeddings
    Q = 3
    pix = torch.full((Q, 3, 1, 1), -5.0)
    for q in range(Q):
        pix[q, q, 0, 0] = 5.0
    scores = torch.tensor([0.9, 0.1, 0.5])  # NOT ascending in topk order
    boxes = torch.arange(Q * 6, dtype=torch.float32).reshape(Q, 6)
    sample = {
        "mask_embeddings": torch.eye(Q),
        "pixel_decoder_output": pix,
        "topk_query_indices": torch.arange(Q),
        "topk_class_scores": scores,
        "boxes": boxes,
        "eval_frame_size": (3, 1, 1),
    }
    pred = model._collapse_sample_chunked(sample)

    order = scores.argsort()  # [1, 2, 0]
    # labels ascending, boxes permuted by the same order
    torch.testing.assert_close(pred["labels"], scores[order])
    torch.testing.assert_close(pred["boxes"], boxes[order])
    # map id j corresponds to row j-1: instance `order[j-1]`'s voxel
    label_map = pred["masks"]
    for j in range(1, Q + 1):
        inst = int(order[j - 1])
        assert label_map[inst, 0, 0].item() == j


def test_collapse_higher_score_wins_overlapping_voxel():
    """When two instances claim the same voxel, the higher-score instance owns it."""
    model = _make_model(focus_on_boxes=True)

    # two queries covering the SAME voxel; higher-score instance must own it
    Q = 2
    pix = torch.full((1, 1, 1, 1), 5.0)
    sample = {
        "mask_embeddings": torch.ones(Q, 1),
        "pixel_decoder_output": pix,
        "topk_query_indices": torch.arange(Q),
        "topk_class_scores": torch.tensor([0.3, 0.8]),
        "boxes": torch.zeros(Q, 6),
        "eval_frame_size": (1, 1, 1),
    }
    pred = model._collapse_sample_chunked(sample)
    # ascending ids: 0.3 -> id 1, 0.8 -> id 2; voxel belongs to id 2
    assert pred["masks"][0, 0, 0].item() == 2
