"""Label-map-native instance overlay rendering.

The overlay renders ``INSTANCE_LABEL_MAP`` predictions directly from the integer
volume -- O(Y*X + N) per slice via a searchsorted id remap -- instead of exploding
into an O(N * volume) per-object bool stack.
"""

import numpy as np
import pytest

from cell_observatory_platform.inference.utils import (
    _instance_cmap,
    _label_and_cmap_from_instance_masks,
    _label_and_cmap_from_label_map,
    save_instance_predictions,
)


def _random_label_map(shape, ids, seed=0):
    rng = np.random.default_rng(seed)
    vol = rng.choice(np.concatenate([[0], ids]), size=shape)
    return vol


def _explode(vol, ids):
    """The OLD path: per-object bool stack in sorted-id order."""
    return np.stack([vol == i for i in ids], axis=0)


def _labelmap(*ids_at):
    """(Z,Y,X,1) int32 volume with the given ``(z, y, x, id)`` voxels set."""
    vol = np.zeros((3, 8, 8, 1), dtype=np.int32)
    for z, y, x, i in ids_at:
        vol[z, y, x, 0] = i
    return vol


def _stack():
    """Already-normalized (N=2, 1, Z=3, Y=8, X=8) per-object bool stack."""
    stack = np.zeros((2, 1, 3, 8, 8), dtype=bool)
    stack[0, 0, 0, 0, 0] = True
    stack[1, 0, 1, 3, 3] = True
    return stack


@pytest.fixture
def subplot_rows(monkeypatch):
    """Record nrows of every page grid save_instance_predictions builds (utils.py
    ``fig.subplots(nrows, ncols, squeeze=False)``)."""
    from matplotlib.figure import Figure

    rows, real = [], Figure.subplots

    def spy(self, nrows=1, ncols=1, **kw):
        rows.append(nrows)
        return real(self, nrows, ncols, **kw)

    monkeypatch.setattr(Figure, "subplots", spy)
    return rows


class TestLabelMapSliceRender:
    def test_matches_old_explode_then_collapse(self):
        """Old path ordered instances by np.unique too, so the native label image
        (and therefore the color assignment) is bit-identical to explode+collapse."""
        ids = np.array([3, 7, 12, 500], dtype=np.int64)   # non-contiguous on purpose
        vol = _random_label_map((4, 16, 16), ids, seed=1)
        ids_present = np.unique(vol); ids_present = ids_present[ids_present != 0]
        stack = _explode(vol, ids_present)
        for z in range(vol.shape[0]):
            old_lab, _, old_norm = _label_and_cmap_from_instance_masks(stack[:, z])
            new_lab, _, new_norm = _label_and_cmap_from_label_map(vol[z], ids_present)
            np.testing.assert_array_equal(new_lab, old_lab)
            np.testing.assert_array_equal(new_norm.boundaries, old_norm.boundaries)

    def test_color_index_stable_across_slices_and_planes(self):
        """One instance keeps ONE dense index on every slice and every ortho plane
        (ids are remapped against the volume-global sorted id list)."""
        vol = np.zeros((3, 8, 8), dtype=np.int64)
        vol[0, 0, 0] = 40       # only on z=0
        vol[:, 2, 2] = 99       # spans all z
        ids = np.unique(vol); ids = ids[ids != 0]           # [40, 99]
        for z in range(3):
            lab, _, _ = _label_and_cmap_from_label_map(vol[z], ids)
            assert set(np.unique(lab[vol[z] == 99])) == {2}  # 99 -> index 2 everywhere
        # ortho plane through the 99 column: same index
        lab_zy, _, _ = _label_and_cmap_from_label_map(vol[:, :, 2], ids)
        assert set(np.unique(lab_zy[vol[:, :, 2] == 99])) == {2}
        lab0, _, _ = _label_and_cmap_from_label_map(vol[0], ids)
        assert set(np.unique(lab0[vol[0] == 40])) == {1}

    def test_sparse_64bit_ids_no_lut_allocation(self):
        """Huge sparse ids must work -- guards the searchsorted-not-LUT choice
        (a max_id-sized LUT would be ~1 TB here)."""
        ids = np.array([7, 500_000, 2**40], dtype=np.int64)
        vol = np.zeros((2, 4, 4), dtype=np.int64)
        vol[0, 0, 0], vol[0, 1, 1], vol[1, 2, 2] = ids
        vol[1, 3, 3] = 12345                                 # NOT in ids -> background
        lab, cmap, _ = _label_and_cmap_from_label_map(vol[0], ids)
        assert lab[0, 0] == 1 and lab[1, 1] == 2
        lab1, _, _ = _label_and_cmap_from_label_map(vol[1], ids)
        assert lab1[2, 2] == 3
        assert lab1[3, 3] == 0                               # unknown id -> bg
        assert cmap.N == ids.size + 1                        # + transparent bg

    def test_empty_ids_renders_empty(self):
        lab, cmap, _ = _label_and_cmap_from_label_map(np.zeros((4, 4), np.int64), np.zeros(0, np.int64))
        assert lab.shape == (1, 1) and lab.sum() == 0
        assert cmap.N == 1

    def test_float_label_map(self):
        """Restored dense outputs can arrive float (interp path casts); equality
        remap must still hit."""
        vol = np.zeros((4, 4), dtype=np.float32)
        vol[1, 1] = 5.0
        ids = np.array([5.0], dtype=np.float32)
        lab, _, _ = _label_and_cmap_from_label_map(vol, ids)
        assert lab[1, 1] == 1 and lab.sum() == 1

    def test_1000_instances_render_in_o_volume_memory(self):
        """1000 instances on a 2000x2000 slice: peak allocation stays O(Y*X) -- an
        (N, Y, X) bool explosion alone would be 4 GB."""
        import tracemalloc
        ids = np.arange(1, 1001, dtype=np.int64)
        sl = np.random.default_rng(0).choice(np.concatenate([[0], ids]), size=(2000, 2000))   # 32 MB
        tracemalloc.start()
        lab, cmap, _ = _label_and_cmap_from_label_map(sl, ids)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak < 16 * sl.nbytes                               # a few O(Y*X) temporaries, not O(N*Y*X)
        assert lab.shape == (2000, 2000) and cmap.N == 1001
        np.testing.assert_array_equal(lab[sl != 0], sl[sl != 0])   # contiguous ids: identity remap
        np.testing.assert_array_equal(np.unique(lab[sl == 0]), [0])


class TestInstanceCmapParity:
    def test_shared_helper_matches_stack_path(self):
        """_instance_cmap is the single source of colors for both render paths."""
        for n in (1, 5, 20, 21, 300):
            masks = np.zeros((n, 4, 4), dtype=bool)
            for i in range(n):
                masks[i, i % 4, i % 4] = True
            _, cmap_stack, norm_stack = _label_and_cmap_from_instance_masks(masks)
            cmap, norm = _instance_cmap(n)
            np.testing.assert_array_equal(cmap_stack.colors, cmap.colors)
            np.testing.assert_array_equal(norm_stack.boundaries, norm.boundaries)


class TestSaveInstancePredictionsIntegration:
    def _run(self, tmp_path, preds, kinds, targets=None, **kw):
        img = np.zeros((1, 3, 8, 8, 1), dtype=np.float32)   # (T,Z,Y,X,C)
        save_instance_predictions(
            save_dir=tmp_path, identifier="t", image=img,
            preds=preds, targets=targets, kinds=kinds, z_step=1, **kw,
        )
        out = tmp_path / "t_instances.pdf"
        assert out.exists() and out.stat().st_size > 0

    @pytest.mark.parametrize("preds, kinds, kw, expected_rows, expected_pages", [
        ({"masks": _labelmap((0, 0, 0, 11), (1, 2, 2, 22))}, {"masks": "instance_label_map"}, {}, 2, 3),   # bg+masks
        ({"masks": _labelmap((1, 1, 1, 4))[None]}, {"masks": "instance_label_map"}, {}, 2, 3),             # T axis
        ({"masks": _labelmap()}, {"masks": "instance_label_map"}, {}, 1, 3),                               # empty: masks row dropped
        ({"masks": _stack()}, {"masks": "instance_stack"}, {}, 2, 3),                                      # (N,1,Z,Y,X) stack
        ({"masks": _labelmap(*[(z, 4, 4, 9) for z in range(3)])}, {"masks": "instance_label_map"},
         {"ortho": True}, 4, 3),                                                                            # 2 * (bg+masks)
    ], ids=["labelmap", "labelmap_T", "empty_labelmap", "stack", "ortho"])
    def test_render_row_layout(self, tmp_path, subplot_rows, preds, kinds, kw, expected_rows, expected_pages):
        """One page per z (Z=3, z_step=1); rows are bg (+ masks when any instance
        is present), doubled under ortho."""
        self._run(tmp_path, preds, kinds, **kw)                  # asserts the PDF exists
        assert len(subplot_rows) == expected_pages
        assert set(subplot_rows) == {expected_rows}

    def test_singleton_stack_with_multi_timepoint_image_raises(self, tmp_path):
        """A (N,1,Z,Y,X) stack cannot be indexed by time: with a T>1 image the
        renderer fails loudly instead of mis-slicing Z as time."""
        img = np.zeros((3, 4, 8, 8, 1), dtype=np.float32)       # T=3
        stack = np.zeros((2, 1, 4, 8, 8), dtype=bool)           # (N,1,Z,Y,X)
        stack[0, 0, 0, 0, 0] = True
        with pytest.raises(ValueError, match="cannot be indexed by time"):
            save_instance_predictions(
                save_dir=tmp_path, identifier="t", image=img,
                preds={"masks": stack}, targets=None,
                kinds={"masks": "instance_stack"}, z_step=1,
                input_format="TZYXC",
            )

    def test_bad_labelmap_rank_raises(self, tmp_path):
        vol = np.zeros((2, 1, 3, 8, 8, 1), dtype=np.int32)  # rank 6: not a label map
        with pytest.raises(ValueError, match="instance_label_map"):
            self._run(tmp_path, {"masks": vol}, {"masks": "instance_label_map"})
