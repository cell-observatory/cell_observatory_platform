import pytest
import torch
from torch.nn import functional as F

from cell_observatory_platform.models.meta_arch.maskdino import MaskMaterializer


def _make_materializer_inputs():
    torch.manual_seed(0)
    num_queries = 5
    mask_dim = 3
    low_res_shape = (2, 3, 2)
    target_size = (4, 5, 4)
    mask_embeddings = torch.randn(num_queries, mask_dim)
    pixel_decoder_output = torch.randn(mask_dim, *low_res_shape)
    return mask_embeddings, pixel_decoder_output, target_size


def _naive_materialize(mask_embeddings, pixel_decoder_output, indices, target_size):
    low_res = torch.einsum(
        "qc,cdhw->qdhw",
        mask_embeddings[indices],
        pixel_decoder_output,
    )
    return F.interpolate(
        low_res.unsqueeze(1).float(),
        size=target_size,
        mode="trilinear",
        align_corners=False,
    ).squeeze(1)


def test_materialize_matches_naive_einsum_interpolate():
    mask_embeddings, pixel_decoder_output, target_size = _make_materializer_inputs()
    indices = torch.tensor([0, 2, 4], dtype=torch.long)
    materializer = MaskMaterializer(
        mask_embeddings=mask_embeddings,
        pixel_decoder_output=pixel_decoder_output,
        target_size=target_size,
        chunk_size=2,
    )

    actual = materializer.materialize(indices)
    expected = _naive_materialize(mask_embeddings, pixel_decoder_output, indices, target_size)

    torch.testing.assert_close(actual.float(), expected)


def test_chunks_concat_equals_materialize_for_noncontiguous_indices():
    mask_embeddings, pixel_decoder_output, target_size = _make_materializer_inputs()
    indices = torch.tensor([4, 1, 3, 0], dtype=torch.long)
    materializer = MaskMaterializer(
        mask_embeddings=mask_embeddings,
        pixel_decoder_output=pixel_decoder_output,
        target_size=target_size,
        chunk_size=2,
    )

    yielded_indices = []
    yielded_logits = []
    for chunk_idx, chunk_logits in materializer.chunks(indices):
        yielded_indices.extend(chunk_idx.tolist())
        yielded_logits.append(chunk_logits)

    actual = torch.cat(yielded_logits, dim=0)
    expected = materializer.materialize(indices)

    assert yielded_indices == indices.tolist()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "mask_embeddings,pixel_decoder_output,target_size,chunk_size,error_match",
    [
        (torch.randn(1, 2, 3), torch.randn(2, 2, 2, 2), (4, 4, 4), 1, "mask_embeddings"),
        (torch.randn(3, 2), torch.randn(2, 2, 2), (4, 4, 4), 1, "pixel_decoder_output"),
        (torch.randn(3, 2), torch.randn(3, 2, 2, 2), (4, 4, 4), 1, "mask_dim mismatch"),
        (torch.randn(3, 2), torch.randn(2, 2, 2, 2), (4, 4), 1, "target_size"),
        (torch.randn(3, 2), torch.randn(2, 2, 2, 2), (4, 4, 4), 0, "chunk_size"),
    ],
)
def test_constructor_validation(
    mask_embeddings,
    pixel_decoder_output,
    target_size,
    chunk_size,
    error_match,
):
    with pytest.raises(ValueError, match=error_match):
        MaskMaterializer(
            mask_embeddings=mask_embeddings,
            pixel_decoder_output=pixel_decoder_output,
            target_size=target_size,
            chunk_size=chunk_size,
        )


def test_empty_indices():
    mask_embeddings, pixel_decoder_output, target_size = _make_materializer_inputs()
    materializer = MaskMaterializer(
        mask_embeddings=mask_embeddings,
        pixel_decoder_output=pixel_decoder_output,
        target_size=target_size,
        chunk_size=2,
    )
    empty = torch.empty(0, dtype=torch.long)

    actual = materializer.materialize(empty)

    assert actual.shape == (0, *target_size)
    assert list(materializer.chunks(empty)) == []
