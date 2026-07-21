import numpy as np
import pytest

from cell_observatory_platform.inference.inference_postprocess import (
    build_records,
    kinds_from_metainfo,
    viz_identifier,
    to_instance_stack,
)


def _region_metainfo(B):
    return {
        "batch_size_actual": B,
        "prepared_id": list(range(B)),
        "tile_name": [f"tile{b}.zarr" for b in range(B)],
        "output_folder": [f"out/{b}" for b in range(B)],
        "time_start": [0] * B, "time_size": [1] * B,
        "z_start": [0] * B, "z_size": [8] * B,
        "y_start": [0] * B, "y_size": [16] * B,
        "x_start": [0] * B, "x_size": [16] * B,
    }


def test_build_records_batch_first_slices_per_sample():
    B = 3
    outs = {
        "data_tensor": np.arange(B * 8 * 16 * 16 * 2).reshape(B, 8, 16, 16, 2),
        "pred_labelmap": np.zeros((B, 8, 16, 16, 1), dtype=np.int32),
    }
    meta = _region_metainfo(B)
    recs = build_records(outs, meta, columns=("output_folder", "tile_name"),
                         image_key="data_tensor")
    assert len(recs) == B
    for b, rec in enumerate(recs):
        assert rec.index == b
        assert rec.image.shape == (8, 16, 16, 2)           # data_tensor lifted, not in preds
        assert "data_tensor" not in rec.preds
        assert rec.preds["pred_labelmap"].shape == (8, 16, 16, 1)
        assert rec.metadata == {"output_folder": f"out/{b}", "tile_name": f"tile{b}.zarr"}
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
    meta = {"batch_size_actual": 1, "output_folder": ["o"], "tile_name": ["t"]}
    recs = build_records({"x": np.zeros((1, 3))}, meta,
                         columns=("output_folder", "tile_name"), image_key=None)
    assert recs[0].region is None


def test_amg_object_stack_stays_whole_in_one_record():
    # AMG: B=1, masks are (1, N, 1, Z, Y, X) batch-first; record 0 holds the (N,1,Z,Y,X) stack.
    N = 5
    outs = {"masks": np.zeros((1, N, 1, 4, 8, 8), dtype=bool)}
    meta = {"batch_size_actual": 1, "output_folder": ["o"], "tile_name": ["t"],
            "tensor_metadata": {"masks": {"kind": "instance_stack"}}}
    recs = build_records(outs, meta, columns=("output_folder", "tile_name"),
                         image_key=None, normalize_instance_masks=True)
    assert len(recs) == 1
    assert recs[0].preds["masks"].shape == (N, 1, 4, 8, 8)   # N objects intact, not split as batch


def test_normalize_instance_masks_flag_explodes_labelmap_for_viz():
    lm = np.zeros((1, 4, 8, 8, 1), dtype=np.int32)   # (B=1, Z, Y, X, 1) labelmap
    lm[0, 0, 0, 0, 0] = 1
    lm[0, 0, 0, 1, 0] = 2
    meta = {"batch_size_actual": 1, "output_folder": ["o"], "tile_name": ["t"],
            "tensor_metadata": {"seg": {"kind": "instance_label_map"}}}
    outs = {"seg": lm}
    # flag off (save): raw labelmap passes through
    off = build_records(outs, meta, columns=(), image_key=None, normalize_instance_masks=False)
    assert off[0].preds["seg"].shape == (4, 8, 8, 1)
    # flag on (viz): exploded to (N=2, 1, Z, Y, X)
    on = build_records(outs, meta, columns=(), image_key=None, normalize_instance_masks=True)
    assert on[0].preds["seg"].shape == (2, 1, 4, 8, 8)


def test_kinds_from_metainfo():
    meta = {"tensor_metadata": {"a": {"kind": "dense"}, "b": {"kind": "boxes"}}}
    assert kinds_from_metainfo(meta) == {"a": "dense", "b": "boxes"}
    assert kinds_from_metainfo({}) == {}


def test_viz_identifier_region_and_fallback():
    B = 1
    recs = build_records({"x": np.zeros((1, 3))}, _region_metainfo(B),
                         columns=("output_folder", "tile_name"), image_key=None)
    ident = viz_identifier(recs[0], rank=2)
    assert ident.startswith("rank002_roi0_tile0.zarr_t0-1_z0-8_y0-16_x0-16")
    # fallback (no region cols)
    meta = {"batch_size_actual": 1, "output_folder": ["a/b"], "tile_name": ["t.zarr"]}
    r = build_records({"x": np.zeros((1, 3))}, meta,
                      columns=("output_folder", "tile_name"), image_key=None)
    assert viz_identifier(r[0], rank=0) == "a_b_t"


def test_to_instance_stack_shapes():
    assert to_instance_stack(np.zeros((3, 4, 8, 8)), kind="instance_stack").shape == (3, 1, 4, 8, 8)
    assert to_instance_stack(np.zeros((3, 1, 4, 8, 8)), kind="instance_stack").shape == (3, 1, 4, 8, 8)
    lm = np.zeros((4, 8, 8), dtype=np.int32); lm[0, 0, 0] = 7
    assert to_instance_stack(lm, kind="instance_label_map").shape == (1, 1, 4, 8, 8)
    empty = to_instance_stack(np.zeros((4, 8, 8), dtype=np.int32), kind="instance_label_map")
    assert empty.shape == (0, 1, 4, 8, 8)
