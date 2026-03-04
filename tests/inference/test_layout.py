"""Unit tests for inference.layout (OutputLayout, compute_layout, validate_layout)."""

import pytest

from cell_observatory_platform.inference.layout import (
    OutputLayout,
    OutputLayoutEntry,
    compute_layout,
    layout_to_manifest_dict,
    validate_layout,
)


# -----------------------------------------------------------------------------
# OutputLayoutEntry
# -----------------------------------------------------------------------------


def test_output_layout_entry_to_dict_roundtrip():
    """OutputLayoutEntry to_dict/from_dict round-trip preserves data."""
    entry = OutputLayoutEntry(
        output_name="pred_semantic",
        output_type="dense_semantic",
        dtype="float32",
        shape=(1, 1, 32, 32, 32),
        order="C",
        offset_bytes=0,
        nbytes=131072,
    )
    d = entry.to_dict()
    restored = OutputLayoutEntry.from_dict(d)
    assert restored == entry


def test_output_layout_entry_from_dict_handles_list_shape():
    """from_dict accepts shape as list and converts to tuple."""
    d = {
        "output_name": "logits",
        "output_type": "raw_logits",
        "dtype": "float32",
        "shape": [1, 100, 2],
        "order": "C",
        "offset_bytes": 0,
        "nbytes": 800,
    }
    entry = OutputLayoutEntry.from_dict(d)
    assert entry.shape == (1, 100, 2)
    assert entry.nbytes == 800


# -----------------------------------------------------------------------------
# OutputLayout
# -----------------------------------------------------------------------------


def test_output_layout_to_dict_roundtrip():
    """OutputLayout to_dict/from_dict round-trip preserves data."""
    entries = [
        OutputLayoutEntry("a", "dense_semantic", "float32", (1, 2, 3), "C", 0, 24),
        OutputLayoutEntry("b", "raw_logits", "float32", (1, 5), "C", 24, 20),
    ]
    layout = OutputLayout(entries=entries, slot_bytes_total=44)
    d = layout.to_dict()
    restored = OutputLayout.from_dict(d)
    assert restored.slot_bytes_total == layout.slot_bytes_total
    assert len(restored.entries) == len(layout.entries)
    for r, e in zip(restored.entries, layout.entries):
        assert r == e


# -----------------------------------------------------------------------------
# compute_layout
# -----------------------------------------------------------------------------


def test_compute_layout_single_output():
    """compute_layout produces valid layout for single output."""
    outputs_metadata = {
        "pred_semantic": {"output_type": "dense_semantic", "shape": [1, 1, 4, 4, 4]},
    }
    output_type_configs = {
        "dense_semantic": {"dtype": "float32", "order": "C"},
    }
    layout = compute_layout(outputs_metadata, output_type_configs)
    assert layout.slot_bytes_total == 1 * 1 * 4 * 4 * 4 * 4  # 256
    assert len(layout.entries) == 1
    assert layout.entries[0].output_name == "pred_semantic"
    assert layout.entries[0].offset_bytes == 0
    assert layout.entries[0].nbytes == 256


def test_compute_layout_multiple_outputs_contiguous_offsets():
    """compute_layout assigns contiguous offsets for multiple outputs."""
    outputs_metadata = {
        "pred_semantic": {"output_type": "dense_semantic", "shape": [1, 1, 8, 8, 8]},  # 2048 bytes
        "pred_logits": {"output_type": "raw_logits", "shape": [1, 50, 2]},  # 400 bytes
    }
    output_type_configs = {
        "dense_semantic": {"dtype": "float32", "order": "C"},
        "raw_logits": {"dtype": "float32", "order": "C"},
    }
    layout = compute_layout(outputs_metadata, output_type_configs)
    assert layout.slot_bytes_total == 2048 + 400
    assert layout.entries[0].offset_bytes == 0
    assert layout.entries[0].nbytes == 2048
    assert layout.entries[1].offset_bytes == 2048
    assert layout.entries[1].nbytes == 400


def test_compute_layout_uses_output_type_dtype_order():
    """compute_layout takes dtype and order from output_type config."""
    outputs_metadata = {
        "x": {"output_type": "half_precision", "shape": [2, 3]},
    }
    output_type_configs = {
        "half_precision": {"dtype": "float16", "order": "F"},
    }
    layout = compute_layout(outputs_metadata, output_type_configs)
    assert layout.entries[0].dtype == "float16"
    assert layout.entries[0].order == "F"
    # float16 = 2 bytes, 2*3 = 6 elements -> 12 bytes
    assert layout.entries[0].nbytes == 12


def test_compute_layout_missing_output_type():
    """compute_layout raises if output_type is missing in outputs_metadata."""
    outputs_metadata = {"x": {"shape": [1, 2, 3]}}
    output_type_configs = {"dense_semantic": {"dtype": "float32", "order": "C"}}
    with pytest.raises(ValueError, match="missing 'output_type'"):
        compute_layout(outputs_metadata, output_type_configs)


def test_compute_layout_missing_shape():
    """compute_layout raises if shape is missing in outputs_metadata."""
    outputs_metadata = {"x": {"output_type": "dense_semantic"}}
    output_type_configs = {"dense_semantic": {"dtype": "float32", "order": "C"}}
    with pytest.raises(ValueError, match="missing 'shape'"):
        compute_layout(outputs_metadata, output_type_configs)


def test_compute_layout_unknown_output_type():
    """compute_layout raises if output_type not in output_type_configs."""
    outputs_metadata = {"x": {"output_type": "unknown_type", "shape": [1, 2]}}
    output_type_configs = {"dense_semantic": {"dtype": "float32", "order": "C"}}
    with pytest.raises(ValueError, match="not found in output_type_configs"):
        compute_layout(outputs_metadata, output_type_configs)


# -----------------------------------------------------------------------------
# validate_layout
# -----------------------------------------------------------------------------


def test_validate_layout_empty_entries():
    """validate_layout raises for empty entries."""
    layout = OutputLayout(entries=[], slot_bytes_total=0)
    with pytest.raises(ValueError, match="at least one entry"):
        validate_layout(layout)


def test_validate_layout_invalid_dtype():
    """validate_layout raises for unsupported dtype."""
    entry = OutputLayoutEntry(
        "x", "dense", "uint64", (1, 2), "C", 0, 8
    )  # uint64 not in allowed set
    layout = OutputLayout(entries=[entry], slot_bytes_total=8)
    with pytest.raises(ValueError, match="dtype.*not supported"):
        validate_layout(layout)


def test_validate_layout_invalid_order():
    """validate_layout raises for unsupported order."""
    entry = OutputLayoutEntry("x", "dense", "float32", (1, 2), "A", 0, 8)
    layout = OutputLayout(entries=[entry], slot_bytes_total=8)
    with pytest.raises(ValueError, match="order.*not supported"):
        validate_layout(layout)


def test_validate_layout_nbytes_mismatch():
    """validate_layout raises when nbytes != prod(shape)*itemsize."""
    entry = OutputLayoutEntry(
        "x", "dense", "float32", (1, 2, 3), "C", 0, 100
    )  # should be 24
    layout = OutputLayout(entries=[entry], slot_bytes_total=100)
    with pytest.raises(ValueError, match="nbytes.*does not match"):
        validate_layout(layout)


def test_validate_layout_sum_mismatch():
    """validate_layout raises when sum(entry.nbytes) != slot_bytes_total."""
    entry = OutputLayoutEntry("x", "dense", "float32", (1, 2), "C", 0, 8)
    layout = OutputLayout(entries=[entry], slot_bytes_total=100)  # should be 8
    with pytest.raises(ValueError, match="sum.*!= slot_bytes_total"):
        validate_layout(layout)


def test_validate_layout_overlapping_ranges():
    """validate_layout raises for overlapping offset ranges."""
    e1 = OutputLayoutEntry("a", "dense", "float32", (1, 2), "C", 0, 8)
    e2 = OutputLayoutEntry("b", "dense", "float32", (1, 2), "C", 4, 8)  # overlaps [0,8)
    layout = OutputLayout(entries=[e1, e2], slot_bytes_total=16)
    with pytest.raises(ValueError, match="Overlapping"):
        validate_layout(layout)


def test_validate_layout_valid_passes():
    """validate_layout does not raise for valid layout."""
    entries = [
        OutputLayoutEntry("a", "dense", "float32", (1, 2), "C", 0, 8),
        OutputLayoutEntry("b", "dense", "float32", (1, 2), "C", 8, 8),
    ]
    layout = OutputLayout(entries=entries, slot_bytes_total=16)
    validate_layout(layout)  # no raise


# -----------------------------------------------------------------------------
# layout_to_manifest_dict
# -----------------------------------------------------------------------------


def test_layout_to_manifest_dict_from_output_layout():
    """layout_to_manifest_dict with OutputLayout returns to_dict()."""
    layout = OutputLayout(
        entries=[
            OutputLayoutEntry("x", "dense", "float32", (1, 2), "C", 0, 8),
        ],
        slot_bytes_total=8,
    )
    manifest = layout_to_manifest_dict(layout)
    assert manifest["slot_bytes_total"] == 8
    assert len(manifest["entries"]) == 1
    assert manifest["entries"][0]["output_name"] == "x"
    assert manifest["entries"][0]["nbytes"] == 8


def test_layout_to_manifest_dict_from_plain_dict():
    """layout_to_manifest_dict with plain dict normalizes to canonical form."""
    d = {
        "slot_bytes_total": 16,
        "entries": [
            {
                "output_name": "a",
                "output_type": "dense",
                "dtype": "float32",
                "shape": [1, 2],
                "order": "C",
                "offset_bytes": 0,
                "nbytes": 8,
            },
            {
                "output_name": "b",
                "output_type": "raw_logits",
                "dtype": "float32",
                "shape": [1, 2],
                "order": "C",
                "offset_bytes": 8,
                "nbytes": 8,
            },
        ],
    }
    manifest = layout_to_manifest_dict(d)
    assert manifest["slot_bytes_total"] == 16
    assert len(manifest["entries"]) == 2


def test_layout_to_manifest_dict_rejects_non_layout_type():
    """layout_to_manifest_dict raises for non-OutputLayout, non-dict."""
    with pytest.raises(TypeError, match="must be OutputLayout or dict"):
        layout_to_manifest_dict("invalid")
