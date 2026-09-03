"""``inference_postprocess``: restore path, box rescale, per-sample records."""

import numpy as np
import pytest
import torch

from cell_observatory_platform.inference.inference_postprocess import (  # private helpers under test
    _per_sample_spatial,
    _rescale_boxes_to_orig,
    _restore_dense_tensor,
    build_records,
    kinds_from_metainfo,
    postprocess,
    to_instance_stack,
    viz_identifier,
)


def _region_metainfo(B):
    return {
        "batch_size_actual": B,
        "roi_id": list(range(B)),
        "tile_name": [f"tile{b}.zarr" for b in range(B)],
        "tile_relative_path": [f"out/{b}" for b in range(B)],
        "time_start": [0] * B, "time_size": [1] * B,
        "z_start": [0] * B, "z_size": [8] * B,
        "y_start": [0] * B, "y_size": [16] * B,
        "x_start": [0] * B, "x_size": [16] * B,
    }


def _meta_for(name, buffer_shape):
    return {"tensor_info": {name: {"shape": list(buffer_shape)}}}


def _sample(proc, orig):
    return {"metainfo": {"image_sizes": torch.tensor([proc]), "orig_image_sizes": torch.tensor([orig])}}


# ---------------------------------------------------------------------------
# build_records / viz_identifier / kinds
# ---------------------------------------------------------------------------


def test_build_records_batch_first_slices_per_sample():
    B = 3
    outs = {
        "data_tensor": np.arange(B * 8 * 16 * 16 * 2).reshape(B, 8, 16, 16, 2),
        "pred_labelmap": np.zeros((B, 8, 16, 16, 1), dtype=np.int32),
    }
    meta = _region_metainfo(B)
    recs = build_records(outs, meta, columns=("tile_relative_path", "tile_name"),
                         image_key="data_tensor")
    assert len(recs) == B
    for b, rec in enumerate(recs):
        assert rec.index == b
        assert rec.image.shape == (8, 16, 16, 2)           # data_tensor lifted, not in preds
        assert "data_tensor" not in rec.preds
        assert rec.preds["pred_labelmap"].shape == (8, 16, 16, 1)
        assert rec.metadata == {"tile_relative_path": f"out/{b}", "tile_name": f"tile{b}.zarr"}
        assert rec.region["roi"] == b
        assert rec.region["coords"] == (0, 1, 0, 8, 0, 16, 0, 16)


def test_missing_batch_size_fails_hard():
    with pytest.raises(KeyError):
        build_records({"x": np.zeros((2, 3))}, {}, columns=())


def test_missing_column_fails_hard():
    meta = {"batch_size_actual": 2}
    with pytest.raises(KeyError):
        build_records({"x": np.zeros((2, 3))}, meta, columns=("nope",), image_key=None)


def test_region_none_when_columns_absent():
    meta = {"batch_size_actual": 1, "tile_relative_path": ["o"], "tile_name": ["t"]}
    recs = build_records({"x": np.zeros((1, 3))}, meta,
                         columns=("tile_relative_path", "tile_name"), image_key=None)
    assert recs[0].region is None


@pytest.mark.parametrize("name, batched, kind, per_sample", [
    ("masks", np.zeros((1, 5, 1, 4, 8, 8), bool), "instance_stack", (5, 1, 4, 8, 8)),   # AMG stack stays whole
    ("seg", np.arange(2 * 4 * 8 * 8).reshape(2, 4, 8, 8, 1).astype(np.int32), "instance_label_map", (4, 8, 8, 1)),
    ("masks", np.zeros((2, 8, 8, 8, 1), np.float32), "dense", (8, 8, 8, 1)),          # no crop to orig_image_sizes
    ("boxes", np.zeros((2, 5, 6), np.float32), "boxes", (5, 6)),
], ids=["instance_stack", "label_map", "dense", "boxes"])
def test_build_records_passes_every_tensor_through_batch_first(name, batched, kind, per_sample):
    """build_records slices batch-first and does nothing else: no explode, no crop,
    no normalization (those are per-handler / saver concerns)."""
    B = batched.shape[0]
    meta = {"batch_size_actual": B, "tile_relative_path": ["o"] * B, "tile_name": ["t"] * B,
            "orig_image_sizes": np.array([[3, 4, 5]] * B),
            "tensor_metadata": {name: {"kind": kind}}}
    recs = build_records({name: batched}, meta, columns=(), image_key=None)
    assert len(recs) == B
    for b, rec in enumerate(recs):
        assert rec.preds[name].shape == per_sample
        np.testing.assert_array_equal(rec.preds[name], batched[b])
        assert rec.kinds[name] == kind and rec.image is None


def test_kinds_from_metainfo():
    meta = {"tensor_metadata": {"a": {"kind": "dense"}, "b": {"kind": "boxes"}}}
    assert kinds_from_metainfo(meta) == {"a": "dense", "b": "boxes"}
    assert kinds_from_metainfo({}) == {}


def test_viz_identifier_region_and_fallback():
    B = 1
    recs = build_records({"x": np.zeros((1, 3))}, _region_metainfo(B),
                         columns=("tile_relative_path", "tile_name"), image_key=None)
    assert viz_identifier(recs[0], rank=2) == "rank002_roi0_tile0.zarr_t0-1_z0-8_y0-16_x0-16"
    # fallback (no region cols). tile_relative_path ALREADY ends in the tile --
    # that is what collapsed the old output_folder + tile_name pair into one
    # column -- so the identifier is that path alone, with separators flattened.
    meta = {"batch_size_actual": 1, "tile_relative_path": ["a/b/t.zarr"], "tile_name": ["t.zarr"]}
    r = build_records({"x": np.zeros((1, 3))}, meta,
                      columns=("tile_relative_path", "tile_name"), image_key=None)
    assert viz_identifier(r[0], rank=0) == "a_b_t"


# ---------------------------------------------------------------------------
# to_instance_stack
# ---------------------------------------------------------------------------


def test_to_instance_stack_shapes():
    assert to_instance_stack(np.zeros((3, 4, 8, 8)), kind="instance_stack").shape == (3, 1, 4, 8, 8)
    assert to_instance_stack(np.zeros((3, 1, 4, 8, 8)), kind="instance_stack").shape == (3, 1, 4, 8, 8)
    assert to_instance_stack(np.zeros((3, 4, 8, 8, 1)), kind="instance_stack").shape == (3, 1, 4, 8, 8)


def test_to_instance_stack_rejects_label_maps():
    # Label maps are rendered natively by the overlay (O(volume) memory, no
    # per-object explosion, no instance-count ceiling); handing one to the
    # stack normalizer is a contract error and must raise loudly.
    lm = np.zeros((4, 8, 8), dtype=np.int32); lm[0, 0, 0] = 7
    with pytest.raises(ValueError, match="natively"):
        to_instance_stack(lm, kind="instance_label_map")


# ---------------------------------------------------------------------------
# _restore_dense_tensor
# ---------------------------------------------------------------------------


def test_restore_dense_tensor_upsamples_integer_map_nearest_and_zero_pads():
    """Integer maps restore with nearest (2x -> 2x2x2 blocks, labels preserved), land
    top-left in the full-tile buffer, and the rest of the buffer is zero-pad."""
    src = torch.arange(1, 9, dtype=torch.int64).reshape(1, 2, 2, 2, 1)        # 8 distinct labels
    out = _restore_dense_tensor(src, "ZYXC", orig_sizes=[(4, 4, 4)], name="masks",
                                proc_sizes=[(2, 2, 2)], outputs_metadata=_meta_for("masks", (6, 6, 6, 1)))
    assert out.shape == (1, 6, 6, 6, 1) and out.dtype == torch.int64
    expected = src[0, ..., 0].repeat_interleave(2, 0).repeat_interleave(2, 1).repeat_interleave(2, 2)
    torch.testing.assert_close(out[0, :4, :4, :4, 0], expected)
    assert out[0, 4:].eq(0).all() and out[0, :, 4:].eq(0).all() and out[0, :, :, 4:].eq(0).all()
    assert set(out.unique().tolist()) == set(range(0, 9))                      # no interpolated labels


def test_restore_dense_tensor_crops_padded_region_before_resize():
    """Only the valid (proc_sizes) region is resized; trailing padding is dropped.
    A constant valid region stays constant under trilinear resize."""
    src = torch.zeros(1, 1, 4, 4, 4, 1)            # (B, T, Z, Y, X, C): valid 2^3 == 3.0, pad == -1
    src[...] = -1.0
    src[0, 0, :2, :2, :2, 0] = 3.0
    out = _restore_dense_tensor(src, "TZYXC", orig_sizes=[(4, 4, 4)], name="img",
                                proc_sizes=[(2, 2, 2)], outputs_metadata=_meta_for("img", (1, 4, 4, 4, 1)))
    assert out.shape == (1, 1, 4, 4, 4, 1)
    torch.testing.assert_close(out[0, 0, ..., 0], torch.full((4, 4, 4), 3.0))   # pad never bled in


def test_restore_dense_tensor_rejects_tile_larger_than_buffer():
    """An original tile larger than the declared full-tile buffer raises instead of
    silently truncating the persisted volume at the saver's crop."""
    t = torch.zeros(1, 4, 4, 4, 1)
    with pytest.raises(ValueError, match="exceeds"):
        _restore_dense_tensor(
            t, "ZYXC",
            orig_sizes=[(16, 16, 16)],               # > buffer 8
            name="masks",
            proc_sizes=[(4, 4, 4)],
            outputs_metadata=_meta_for("masks", (8, 8, 8, 1)),
        )


# ---------------------------------------------------------------------------
# _rescale_boxes_to_orig / _per_sample_spatial
# ---------------------------------------------------------------------------


def test_rescale_boxes_to_orig_scales_xyzxyz_per_sample():
    boxes = torch.tensor([[[1., 1., 1., 2., 2., 2.]], [[1., 1., 1., 2., 2., 2.]]])   # (B=2, N=1, 6)
    out = _rescale_boxes_to_orig(boxes, proc_sizes=[(8, 4, 2), (8, 8, 8)],           # (Z, Y, X)
                                 orig_sizes=[(16, 8, 8), (8, 8, 8)])
    torch.testing.assert_close(out[0, 0], torch.tensor([4., 2., 2., 8., 4., 4.]))    # x*4, y*2, z*2
    torch.testing.assert_close(out[1], boxes[1])                                      # unresized sample untouched
    torch.testing.assert_close(boxes[0, 0], torch.tensor([1., 1., 1., 2., 2., 2.]))  # input not mutated
    with pytest.raises(ValueError, match=r"\(B, N, 6\)"):
        _rescale_boxes_to_orig(torch.zeros(2, 6), [(1, 1, 1)] * 2, [(1, 1, 1)] * 2)


def test_per_sample_spatial_requires_tensor_and_takes_trailing_three():
    assert _per_sample_spatial(torch.tensor([[2, 8, 16, 32], [1, 4, 4, 4]])) == [(8, 16, 32), (4, 4, 4)]  # TZYX
    assert _per_sample_spatial(torch.tensor([[8, 16, 32]])) == [(8, 16, 32)]                           # ZYX
    with pytest.raises(TypeError, match="must be a tensor"):
        _per_sample_spatial([[8, 16, 32]])


# ---------------------------------------------------------------------------
# postprocess
# ---------------------------------------------------------------------------


def test_postprocess_rejects_save_tensor_missing_from_outputs():
    meta = {"save_tensors": {"boxes": {"annotation_type": "sparse", "data_format": "N6"}},
            "tensor_info": {}}
    with pytest.raises(ValueError, match="not among the model outputs"):
        postprocess({"masks": torch.zeros(1, 2, 2, 2, 1)}, _sample((2, 2, 2), (2, 2, 2)), meta)


def test_postprocess_is_identity_when_not_resized():
    meta = {"save_tensors": {"masks": {"annotation_type": "dense", "data_format": "ZYXC"},
                             "boxes": {"annotation_type": "sparse", "data_format": "N6"},
                             "labels": {"annotation_type": "sparse", "data_format": "N"}},
            "tensor_info": {"masks": {"shape": [2, 2, 2, 1]}}}
    masks = torch.randint(0, 5, (1, 2, 2, 2, 1))
    boxes, labels = torch.rand(1, 3, 6), torch.rand(1, 3)
    out = postprocess({"masks": masks.clone(), "boxes": boxes.clone(), "labels": labels},
                      _sample((2, 2, 2), (2, 2, 2)), meta)
    torch.testing.assert_close(out["masks"], masks)            # nearest identity resize, buffer == tile
    torch.testing.assert_close(out["boxes"], boxes)            # scale 1 -> skipped
    assert out["labels"] is labels                             # N-format sparse: untouched
