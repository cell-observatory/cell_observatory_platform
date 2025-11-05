import pytest
import torch

from models.preprocessor import TorchPreprocessor, DaliPreprocessor, RayPreprocessor


def _dummy_mask_generator(batch_size: int):
    masks = torch.ones(batch_size, 1, dtype=torch.bool)
    ctx = torch.zeros(batch_size, 1, dtype=torch.bool)
    tgt = torch.ones(batch_size, 1, dtype=torch.bool)
    orig_idx = torch.arange(batch_size)
    channels_to_mask = [0]
    return masks, ctx, tgt, orig_idx, channels_to_mask, None

# -------------------------
# TorchPreprocessor tests
# -------------------------


def test_torch_preprocessor_no_mask_stacks_list_and_keeps_dtype():
    B, C = 3, 4
    tensors = [torch.zeros(C, dtype=torch.float32) for _ in range(B)]
    sample = {"data_tensor": tensors, "metainfo": {"foo": 123}}

    proc = TorchPreprocessor(dtype=torch.float32, with_masking=False, mask_generator=None)
    out = proc(sample, data_time=0.01)

    assert "data_tensor" in out and "metainfo" in out
    assert out["data_tensor"].shape == (B, C)
    assert out["data_tensor"].dtype == torch.float32
    assert out["metainfo"]["foo"] == 123


def test_torch_preprocessor_with_mask_includes_mask_info():
    B, C = 2, 5
    sample = {"data_tensor": torch.ones(B, C, dtype=torch.float32), "metainfo": {"a": 7}}

    proc = TorchPreprocessor(dtype=torch.float32, with_masking=True, mask_generator=_dummy_mask_generator)
    out = proc(sample, data_time=0.02)

    meta = out["metainfo"]
    for k in ("masks", "context_masks", "target_masks", "original_patch_indices", "channels_to_mask"):
        assert k in meta and isinstance(meta[k], list) and len(meta[k]) == 1

    assert isinstance(meta["preprocess_time"], float)
    assert isinstance(meta["masking_time"], float)
    assert isinstance(meta["data_time"], float) and meta["data_time"] == 0.02
    assert meta["a"] == 7


# -------------------------
# DaliPreprocessor tests
# -------------------------


def test_dali_preprocessor_no_mask_minimal():
    B, C = 2, 3
    inputs = torch.zeros(B, C, dtype=torch.float16)
    dali_sample = ({"data_tensor": inputs, "get_item_time": 0.123},)

    proc = DaliPreprocessor(dtype=torch.float16, with_masking=False, mask_generator=None)
    out = proc(dali_sample, data_time=0.0)

    assert torch.equal(out["data_tensor"], inputs)
    assert out["metainfo"] == {}


def test_dali_preprocessor_with_mask_includes_fields_and_timings():
    B, C = 2, 4
    inputs = torch.ones(B, C, dtype=torch.bfloat16)
    dali_sample = ({"data_tensor": inputs, "get_item_time": 0.5},)

    proc = DaliPreprocessor(dtype=torch.bfloat16, with_masking=True, mask_generator=_dummy_mask_generator)
    out = proc(dali_sample, data_time=0.25)

    meta = out["metainfo"]
    for k in ("masks", "context_masks", "target_masks", "original_patch_indices", "channels_to_mask"):
        assert k in meta and isinstance(meta[k], list) and len(meta[k]) == 1
    assert isinstance(meta["preprocess_time"], float)
    assert isinstance(meta["masking_time"], float)
    assert meta["get_item_time"] == 0.5
    assert meta["data_time"] == 0.25


# -------------------------
# RayPreprocessor tests
# -------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for RayPreprocessor tests")
def test_ray_preprocessor_transform_and_masking_on_cuda():
    B, C = 2, 3
    inputs = torch.ones(B, C, dtype=torch.float32, device='cuda')
    sample = {"data_tensor": inputs, "metainfo": {"k": "v"}}

    def add_five(x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda
        return x + 5

    proc = RayPreprocessor(
        dtype=torch.float32,
        with_masking=True,
        mask_generator=_dummy_mask_generator,
        transforms_list=[add_five],
    )

    out = proc(sample, data_time=0.33)
    data = out["data_tensor"]
    meta = out["metainfo"]

    assert data.is_cuda
    assert torch.allclose(data, inputs + 5)

    for k in ("masks", "context_masks", "target_masks", "original_patch_indices", "channels_to_mask"):
        assert k in meta and isinstance(meta[k], list) and len(meta[k]) == 1

    assert isinstance(meta["preprocess_time"], float)
    assert isinstance(meta["masking_time"], float)
    assert isinstance(meta["transform_time"], float)
    assert meta["data_time"] == 0.33


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for RayPreprocessor tests")
def test_ray_preprocessor_no_mask_returns_empty_meta():
    inputs = torch.zeros(2, 2, dtype=torch.float32, device='cuda')
    sample = {"data_tensor": inputs, "metainfo": {}}

    proc = RayPreprocessor(dtype=torch.float32, with_masking=False, mask_generator=_dummy_mask_generator)
    out = proc(sample, data_time=0.0)

    assert out["data_tensor"].is_cuda
    assert out["metainfo"] == {}
