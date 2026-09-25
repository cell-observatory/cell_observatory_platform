"""``inference/amg.py``: 3D crop-box generation, aligned batch iteration and
small-region cleanup (the ``cc3d``-backed tests skip when it is not installed)."""

import numpy as np
import pytest

from cell_observatory_platform.inference.amg import (
    batch_iterator,
    generate_crop_boxes_3d,
    remove_small_regions_3d,
)


# ---------------------------------------------------------------------------
# batch_iterator
# ---------------------------------------------------------------------------


def test_batch_iterator_yields_aligned_chunks():
    """Every input is sliced by the same positions per chunk; the last chunk is
    the remainder. Calling with no inputs at all is a contract error."""
    a, b = [1, 2, 3, 4, 5], ["a", "b", "c", "d", "e"]
    chunks = list(batch_iterator(2, a, b))
    assert chunks == [[[1, 2], ["a", "b"]], [[3, 4], ["c", "d"]], [[5], ["e"]]]
    with pytest.raises(ValueError, match="same size"):
        list(batch_iterator(2))


def test_batch_iterator_rejects_length_mismatch():
    """Per-object arrays are zipped positionally downstream: unequal lengths
    must raise instead of silently misaligning masks with their scores."""
    with pytest.raises(ValueError, match="same size"):
        list(batch_iterator(2, [1, 2, 3], [1, 2]))


# ---------------------------------------------------------------------------
# generate_crop_boxes_3d
# ---------------------------------------------------------------------------


def test_generate_crop_boxes_layer0_is_full_volume_and_layer1_tiles_it():
    """Layer 0 is the whole volume as ``[x0, y0, z0, x1, y1, z1]``; layer 1 splits
    every axis in two, giving 8 in-bounds crops whose starts cover the octants."""
    boxes, layers = generate_crop_boxes_3d((8, 16, 32), n_layers=1, overlap_ratio=0.0)
    assert boxes[0] == [0, 0, 0, 32, 16, 8] and layers[0] == 0
    assert len(boxes) == 1 + 8 and layers[1:] == [1] * 8
    for x0, y0, z0, x1, y1, z1 in boxes[1:]:
        assert 0 <= x0 < x1 <= 32 and 0 <= y0 < y1 <= 16 and 0 <= z0 < z1 <= 8
    assert {tuple(b[:3]) for b in boxes[1:]} == {
        (x, y, z) for x in (0, 16) for y in (0, 8) for z in (0, 4)
    }


def test_generate_crop_boxes_rejects_non_positive_stride():
    """overlap == crop length would give a zero stride (every crop start at 0):
    the generator raises instead of emitting duplicate crops."""
    with pytest.raises(ValueError, match="overlap"):
        generate_crop_boxes_3d((8, 8, 8), n_layers=1, overlap_ratio=1.0)


# ---------------------------------------------------------------------------
# remove_small_regions_3d
# ---------------------------------------------------------------------------


def test_remove_small_regions_holes_fills_hole_in_uint8_mask():
    """A uint8 {0, 255} mask is binarized before the XOR, so a one-voxel hole
    is detected as background and filled in ``holes`` mode."""
    pytest.importorskip("cc3d")
    mask = np.full((5, 5, 5), 255, dtype=np.uint8)
    mask[2, 2, 2] = 0
    cleaned, modified = remove_small_regions_3d(mask, volume_thresh=5, mode="holes")
    assert modified is True
    assert cleaned.astype(bool).all()


def test_remove_small_regions_islands_drops_small_component():
    """``islands`` mode removes foreground components below the volume
    threshold and keeps the large one intact."""
    pytest.importorskip("cc3d")
    mask = np.zeros((6, 6, 6), dtype=np.uint8)
    mask[:4, :4, :4] = 255      # 64 voxels
    mask[5, 5, 5] = 255         # 1 voxel
    cleaned, modified = remove_small_regions_3d(mask, volume_thresh=5, mode="islands")
    assert modified is True
    assert not cleaned[5, 5, 5]
    assert cleaned[:4, :4, :4].all()


def test_remove_small_regions_returns_input_unmodified_when_nothing_is_small():
    """No component below threshold: the SAME array comes back with
    ``modified=False``. Unknown modes are rejected."""
    pytest.importorskip("cc3d")
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[:4, :4, :4] = True
    cleaned, modified = remove_small_regions_3d(mask, volume_thresh=5, mode="islands")
    assert modified is False and cleaned is mask
    with pytest.raises(ValueError, match="not supported"):
        remove_small_regions_3d(mask, volume_thresh=5, mode="blobs")
