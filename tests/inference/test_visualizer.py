"""Unit tests for InferenceVisualizer and resolve_path."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cell_observatory_platform.inference.visualizer import (
    InferenceVisualizer,
    resolve_path,
    _softmax_last_axis,
)


# -----------------------------------------------------------------------------
# resolve_path
# -----------------------------------------------------------------------------


def test_resolve_path_simple_key():
    """resolve_path with simple dict key."""
    d = {"a": 1, "b": 2}
    assert resolve_path(d, "a") == 1
    assert resolve_path(d, "b") == 2


def test_resolve_path_nested():
    """resolve_path with nested dict."""
    d = {"x": {"y": {"z": 42}}}
    assert resolve_path(d, "x.y.z") == 42


def test_resolve_path_list_index():
    """resolve_path with list index."""
    d = {"items": [10, 20, 30]}
    assert resolve_path(d, "items.0") == 10
    assert resolve_path(d, "items.1") == 20


def test_resolve_path_list_index_bracket():
    """resolve_path with key[i] syntax."""
    d = {"masks": [100, 200]}
    assert resolve_path(d, "masks[0]") == 100
    assert resolve_path(d, "masks[1]") == 200


def test_resolve_path_missing_key_raises():
    """resolve_path with missing key raises KeyError."""
    d = {"a": 1}
    with pytest.raises(KeyError, match="Bad path segment|'c'"):
        resolve_path(d, "c")


def test_resolve_path_empty_path():
    """resolve_path with empty path returns root."""
    d = {"a": 1}
    assert resolve_path(d, "") == d


def test_resolve_path_scalar_attribute():
    """resolve_path with object attribute."""
    class Obj:
        val = 99

    assert resolve_path(Obj(), "val") == 99


# -----------------------------------------------------------------------------
# _softmax_last_axis
# -----------------------------------------------------------------------------


def test_softmax_last_axis_shape():
    """softmax preserves shape and sums to 1 on last axis."""
    x = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    out = _softmax_last_axis(x)
    assert out.shape == x.shape
    np.testing.assert_array_almost_equal(out.sum(axis=-1), np.ones(2))


def test_softmax_last_axis_stability():
    """softmax handles large values without overflow."""
    x = np.array([1000.0, 1001.0, 1002.0])
    out = _softmax_last_axis(x)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_almost_equal(out.sum(), 1.0)


# -----------------------------------------------------------------------------
# InferenceVisualizer handler dispatch
# -----------------------------------------------------------------------------


def test_visualize_unknown_handler_raises():
    """visualize with unknown handler raises ValueError."""
    viz = InferenceVisualizer()
    cfg = {"viz": {"handler": "unknown_handler"}}
    with pytest.raises(ValueError, match="Unknown viz.handler"):
        viz.visualize("out", cfg, np.zeros(3), {}, save_dir=".")


def test_visualize_missing_handler_raises():
    """visualize with no viz.handler raises ValueError."""
    viz = InferenceVisualizer()
    cfg = {"output_type": "dense_semantic"}
    with pytest.raises(ValueError, match="no viz.handler"):
        viz.visualize("out", cfg, np.zeros(3), {}, save_dir=".")


def test_visualize_semantic_map_dispatches():
    """visualize with semantic_map calls save_semantic_predictions."""
    viz = InferenceVisualizer()
    cfg = {"viz": {"handler": "semantic_map"}, "output_type": "dense_semantic"}
    # pred: (1,2,3,1) TZYXC, image: (1,2,3,1)
    data = np.zeros((1, 2, 3, 1), dtype=np.float32)
    context = {
        "identifier": "test_1",
        "save_dir": "/tmp/viz_test",
        "image": np.zeros((1, 2, 3, 1), dtype=np.float32),
        "save_as_volume": False,
        "save_as_pdf": True,
        "z_step_pdf": 1,
        "filetype": "zarr",
    }
    with patch(
        "cell_observatory_platform.inference.utils.save_semantic_predictions"
    ) as mock_save:
        viz.visualize("predictions", cfg, data, context)
    mock_save.assert_called_once()
    call_kw = mock_save.call_args[1]
    assert call_kw["pred_semantic"] is data
    assert call_kw["name"] == "test_1"


def test_visualize_logits_to_probs_applies_softmax_and_delegates():
    """logits_to_probs applies softmax then delegates to semantic_map."""
    viz = InferenceVisualizer()
    cfg = {
        "viz": {"handler": "logits_to_probs", "delegate_to": "semantic_map"},
        "output_type": "raw_logits",
    }
    logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    context = {
        "identifier": "test_logits",
        "save_dir": "/tmp/viz_test",
        "image": np.zeros((1, 2, 3, 1), dtype=np.float32),
        "save_as_volume": False,
        "save_as_pdf": True,
        "z_step_pdf": 1,
        "filetype": "zarr",
    }
    with patch(
        "cell_observatory_platform.inference.utils.save_semantic_predictions"
    ) as mock_save:
        viz.visualize("raw_logits", cfg, logits, context)
    mock_save.assert_called_once()
    call_data = mock_save.call_args[1]["pred_semantic"]
    # delegate receives softmax output
    np.testing.assert_array_almost_equal(call_data.sum(axis=-1), np.ones(2))


def test_visualize_pca_and_patch_cosine_both_registered():
    """Both pca and patch_cosine handlers are registered and share the same implementation."""
    viz = InferenceVisualizer()
    assert "pca" in viz._handlers
    assert "patch_cosine" in viz._handlers
    # Same underlying method (bound methods may differ; compare __func__)
    assert viz._handlers["pca"].__func__ is viz._handlers["patch_cosine"].__func__
