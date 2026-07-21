import torch

import cell_observatory_platform.evaluation.evaluate_postprocess as ep


def test_extract_targets_unwraps_batch_and_squeezes_label_map():
    lm = torch.zeros(1, 4, 8, 8, dtype=torch.int32)   # (1,Z,Y,X) stray T axis
    # platform List[List[dict]] batch wrap
    ds = {"metainfo": {"targets": [[{"label_map": lm, "labels": torch.tensor([1])}]]}}
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
    # label_map: ids 5,6 in class mask -> (2,Z,Y,X) bool
    lm = torch.zeros(2, 4, 4, dtype=torch.long)
    lm[0, 0, 0] = 5
    lm[0, 0, 1] = 6
    tgt = {"label_map": lm, "mask_ids": torch.tensor([5, 6])}
    class_mask = torch.tensor([True, True])
    m = ep.gt_masks_for_class(tgt, class_mask, size=(2, 4, 4), source="label_map")
    assert m.shape == (2, 2, 4, 4) and m.dtype == torch.bool
    assert m[0, 0, 0, 0].item() is True and m[1, 0, 0, 1].item() is True
    # masks source
    masks = torch.zeros(3, 2, 4, 4, dtype=torch.bool)
    mm = ep.gt_masks_for_class({"masks": masks}, torch.tensor([True, False, True]),
                               size=(2, 4, 4), source="masks")
    assert mm.shape == (2, 2, 4, 4)
