import pytest
import torch

import cell_observatory_platform.evaluation.evaluate_postprocess as ep


def test_extract_targets_unwraps_batch_and_squeezes_label_map():
    lm = torch.zeros(1, 4, 8, 8, dtype=torch.int32)   # (1,Z,Y,X) stray T axis
    # platform List[List[dict]] batch wrap
    ds = {"metainfo": {"targets": [{"label_map": lm, "labels": torch.tensor([1])}]}}
    out = ep.extract_targets(ds, squeeze_label_map=True)
    assert len(out) == 1
    assert out[0]["label_map"].shape == (4, 8, 8)     # squeezed
    # without squeeze, left as-is
    out2 = ep.extract_targets(ds, squeeze_label_map=False)
    assert out2[0]["label_map"].shape == (1, 4, 8, 8)


def test_resize_label_map_and_masks_noop_when_same_size():
    lm = torch.randint(0, 3, (4, 8, 8), dtype=torch.long)
    assert ep.resize_label_map(lm, (4, 8, 8)) is lm
    up = ep.resize_label_map(lm, (4, 16, 16))
    assert up.shape == (4, 16, 16) and up.dtype == torch.long
    m = torch.zeros(2, 4, 8, 8, dtype=torch.bool)
    assert ep.resize_masks(m, (4, 8, 8)) is m
    um = ep.resize_masks(m, (4, 16, 16))
    assert um.shape == (2, 4, 16, 16) and um.dtype == torch.bool


def test_resize_gt_matches_transform_convention():
    """Eval-time GT resize agrees voxel-for-voxel with the train-time transform.

    Guards the nearest vs nearest-exact half-voxel offset: the transforms warp
    label_map/masks with nearest-exact, so eval GT must use the same grid.

    Ratios matter here -- an exact 2x UPSAMPLE is the one case where the two
    conventions coincide, so both assertions use ratios that separate them
    (a non-integer upsample and a downsample).
    """
    from cell_observatory_platform.data.transforms.utils import (
        resize_label_map as tf_resize_label_map,
        resize_masks as tf_resize_masks,
    )
    lm = torch.randint(0, 5, (4, 8, 8), dtype=torch.long)
    assert torch.equal(
        ep.resize_label_map(lm, (3, 5, 7)), tf_resize_label_map(lm, (3, 5, 7))
    )
    m = torch.randint(0, 2, (3, 4, 8, 8), dtype=torch.bool)
    assert torch.equal(
        ep.resize_masks(m, (2, 4, 4)), tf_resize_masks(m, (2, 4, 4))
    )


def test_gt_boxes_abs_xyzxyz_denorm():
    # normalized cxcyczwhd box centered, full extent -> xyzxyz [0,0,0,W,H,D]
    box = torch.tensor([[0.5, 0.5, 0.5, 1.0, 1.0, 1.0]])
    out = ep.gt_boxes_abs_xyzxyz({"boxes": box}, size=(4, 8, 16), fmt="cxcyczwhd", normalized=True)
    # xyzxyz corners scaled by (W,H,D,W,H,D) = (16,8,4,16,8,4)
    assert out.shape == (1, 6)
    assert torch.allclose(out[0], torch.tensor([0., 0., 0., 16., 8., 4.]))
    # empty passthrough
    empty = torch.zeros(0, 6)
    assert ep.gt_boxes_abs_xyzxyz({"boxes": empty}, (4, 8, 8), "xyzxyz", False).shape == (0, 6)


def test_gt_semantic_map_from_masks_and_labelmap():
    # masks source: two instances of classes 0 and 1 -> class+1 = 1, 2
    masks = torch.zeros(2, 2, 4, 4, dtype=torch.bool)
    masks[0, 0, 0, 0] = True
    masks[1, 0, 0, 1] = True
    gt = ep.gt_semantic_map({"masks": masks, "labels": torch.tensor([0, 1])},
                            size=(2, 4, 4), source="masks")
    assert gt.shape == (2, 4, 4) and gt[0, 0, 0].item() == 1 and gt[0, 0, 1].item() == 2
    # label_map source: instance ids 5,6 -> classes 0,1
    lm = torch.zeros(2, 4, 4, dtype=torch.long)
    lm[0, 0, 0] = 5
    lm[0, 0, 1] = 6
    gt2 = ep.gt_semantic_map({"label_map": lm, "mask_ids": torch.tensor([5, 6]),
                              "labels": torch.tensor([0, 1])}, size=(2, 4, 4), source="label_map")
    assert gt2[0, 0, 0].item() == 1 and gt2[0, 0, 1].item() == 2


def test_gt_masks_for_class_both_sources():
    lm = torch.zeros(2, 4, 4, dtype=torch.long)
    lm[0, 0, 0] = 5
    lm[0, 0, 1] = 6
    tgt = {"label_map": lm, "mask_ids": torch.tensor([5, 6])}
    m = ep.gt_masks_for_class(tgt, torch.tensor([True, True]), size=(2, 4, 4), source="label_map")
    assert m.shape == (2, 2, 4, 4) and m.dtype == torch.bool
    assert torch.equal(m[0], lm == 5) and torch.equal(m[1], lm == 6)

    masks = torch.zeros(3, 2, 4, 4, dtype=torch.bool)
    masks[0, 0, 0, 0] = True
    masks[1, 0, 1, 1] = True
    masks[2, 1, 2, 2] = True
    mm = ep.gt_masks_for_class({"masks": masks}, torch.tensor([True, False, True]), size=(2, 4, 4), source="masks")
    assert mm.shape == (2, 2, 4, 4) and mm.dtype == torch.bool
    assert torch.equal(mm, masks[[0, 2]])     # rows selected by the class mask, in order


def test_extract_targets_rejects_label_map_with_t_above_one():
    """A (T, Z, Y, X) label_map with T > 1 cannot be squeezed; it raises instead
    of being silently truncated to its first frame."""
    sample = {"metainfo": {"targets": [{"label_map": torch.zeros(2, 3, 3, 3)}]}}
    with pytest.raises(ValueError, match="T==1"):
        ep.extract_targets(sample, squeeze_label_map=True)


class TestResizeLabelMapLargeIds:
    """Instance ids never travel through float32, so ids beyond the 24-bit
    mantissa survive the resize exactly and adjacent large ids do not alias."""

    def test_large_ids_survive_exactly(self):
        ids = [0, 3, 2**24 + 3, 2**24 + 4, 2**31 + 5, 10**8 + 3]
        lm = torch.zeros(4, 4, 4, dtype=torch.long)
        flat = lm.view(-1)
        for i, v in enumerate(ids):
            flat[i * 7] = v
        out = ep.resize_label_map(lm, (8, 8, 8))
        assert out.dtype == torch.long
        assert set(out.unique().tolist()) <= set(ids)
        for v in ids[2:]:
            assert (out == v).any(), f"id {v} corrupted or dropped by resize"

    def test_adjacent_large_ids_do_not_alias(self):
        # 16_777_219 and 16_777_220 both round to 16_777_220 in float32.
        a, b = 2**24 + 3, 2**24 + 4
        lm = torch.zeros(2, 2, 4, dtype=torch.long)
        lm[..., :2] = a
        lm[..., 2:] = b
        out = ep.resize_label_map(lm, (4, 4, 8))
        assert (out == a).sum() > 0 and (out == b).sum() > 0
        assert (out == a).sum() == (out == b).sum()


def _loop_semantic(label_map, mask_ids, labels, size):
    gt = torch.zeros_like(label_map, dtype=torch.long)
    for inst_id, cls in zip(mask_ids.tolist(), labels.tolist()):
        gt[label_map == inst_id] = int(cls) + 1
    return ep.resize_label_map(gt, size)


class TestGtSemanticMap:
    """`gt_semantic_map`: the label_map LUT gather equals a per-instance loop,
    huge ids take the loop path, and the masks source is last-write-wins."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_label_map_source_matches_per_instance_loop(self, seed):
        g = torch.Generator().manual_seed(seed)
        shape = (6, 12, 12)
        ids = torch.tensor([3, 7, 500, 1024])  # non-contiguous, background 0
        lm = torch.zeros(shape, dtype=torch.long)
        for i in ids.tolist():
            lm[torch.rand(shape, generator=g) < 0.1] = i
        labels = torch.tensor([0, 1, 2, 1])
        target = {"label_map": lm, "mask_ids": ids, "labels": labels}
        got = ep.gt_semantic_map(target, shape, source="label_map")
        want = _loop_semantic(lm, ids, labels, shape)
        assert torch.equal(got, want)

    def test_ids_above_lut_limit_take_loop_path(self):
        # ids beyond the LUT-size guard (2**24) take the loop path — same result.
        shape = (2, 4, 4)
        big = 2**24 + 5
        lm = torch.zeros(shape, dtype=torch.long)
        lm[0, 0, 0] = big
        target = {
            "label_map": lm,
            "mask_ids": torch.tensor([big]),
            "labels": torch.tensor([2]),
        }
        got = ep.gt_semantic_map(target, shape, source="label_map")
        assert got[0, 0, 0] == 3 and got.sum() == 3

    def test_masks_source_last_write_wins(self):
        masks = torch.zeros(2, 2, 4, 4, dtype=torch.bool)
        masks[0, 0, 0, 0] = True
        masks[1, 0, 0, 0] = True  # overlap: later class must win
        target = {"masks": masks, "labels": torch.tensor([0, 1])}
        got = ep.gt_semantic_map(target, (2, 4, 4), source="masks")
        assert got[0, 0, 0] == 2
