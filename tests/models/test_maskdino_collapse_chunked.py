import torch

from cell_observatory_platform.models.meta_arch.maskdino import MaskDINO
from cell_observatory_platform.models.meta_arch.maskdino import MaskMaterializer


def _make_model(focus_on_boxes):
    model = MaskDINO.__new__(MaskDINO)
    model.mask_chunk_size = 1
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
        "orig_image_size": (2, 2, 2),
    }


def _materialize_sample_logits(sample):
    materializer = MaskMaterializer(
        mask_embeddings=sample["mask_embeddings"],
        pixel_decoder_output=sample["pixel_decoder_output"],
        target_size=sample["orig_image_size"],
        chunk_size=1,
    )
    return materializer.materialize(sample["topk_query_indices"])


def test_collapse_focus_on_boxes_keeps_topk_scores():
    model = _make_model(focus_on_boxes=True)
    sample = _make_sample()

    pred = model._collapse_sample_chunked(sample)

    torch.testing.assert_close(pred["labels"], sample["topk_class_scores"])
    assert pred["masks"].dtype == torch.uint16
    assert pred["masks"][0, 0, 0].item() == 1
    assert pred["masks"][1, 1, 1].item() == 2


def test_collapse_focus_on_masks_multiplies_mask_confidence():
    model = _make_model(focus_on_boxes=False)
    sample = _make_sample()
    logits = _materialize_sample_logits(sample)
    binary = logits > 0
    expected_conf = (logits.sigmoid() * binary.to(logits.dtype)).flatten(1).sum(dim=1) / (
        binary.flatten(1).sum(dim=1).clamp_min(1).to(logits.dtype)
    )
    expected_scores = sample["topk_class_scores"] * expected_conf

    pred = model._collapse_sample_chunked(sample)

    torch.testing.assert_close(pred["labels"], expected_scores)
    assert pred["masks"][0, 0, 0].item() == 1
    assert pred["masks"][1, 1, 1].item() == 2
