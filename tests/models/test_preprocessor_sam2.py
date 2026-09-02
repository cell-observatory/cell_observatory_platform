"""Tests for SAM2VideoPreprocessor target-view fields (labelmap-native path).

`_build_data_views` materializes a random K=max_masks subset of binary masks
on-device and adds labelmap, instance_ids (with sentinel -1 for pad), valid,
presence_t, boxes, and box_format alongside the `masks`/`img_ids` contract.
These tests exercise it through the class directly, bypassing the base
preprocessor `__init__` which expects a Ray-style runtime config.

The K-subset is drawn via `pp.rng` (torch.Generator), so per-row order is not
deterministic; assertions are written set-/membership-wise rather than by row.
"""
from __future__ import annotations

import pytest
import torch

from cell_observatory_platform.models.layers.preprocessor import SAM2VideoPreprocessor, _LazyFrameMasks


def _make_preprocessor(
    max_masks: int, bbox_format: str = "zyxzyx", boxes_normalized: bool = False
) -> SAM2VideoPreprocessor:
    # Bypass __init__: the data-view builder only needs max_masks, bbox_format,
    # boxes_normalized, rng.
    pp = SAM2VideoPreprocessor.__new__(SAM2VideoPreprocessor)
    pp.max_masks = max_masks
    pp.bbox_format = bbox_format
    pp.boxes_normalized = boxes_normalized
    pp.rng = torch.Generator()
    pp.rng.manual_seed(0)
    return pp


def test_lazy_data_view_fields_shapes_and_pad_sentinel():
    device = torch.device("cpu")
    B, T, Z, Y, X = 2, 3, 2, 4, 5
    # rows per video = largest sampled count in the batch (2), capped by max_masks (4)
    K_full = 2

    # Construct a labelmap whose voxels are mostly background (0) plus a few
    # objects with known integer ids placed deterministically.
    labelmap = torch.zeros((B, T, Z, Y, X), dtype=torch.int32, device=device)
    labelmap[0, 0, 0, 0, 0] = 7    # video 0, frame 0: id=7 present
    labelmap[0, 1, 1, 2, 3] = 11   # video 0, frame 1: id=11 present
    labelmap[1, 0, 0, 1, 1] = 13   # video 1, frame 0: id=13 present
    # Video 0 has ids [7, 11]; video 1 has id [13].

    targets = [
        {
            "mask_ids": torch.tensor([7, 11], dtype=torch.long, device=device),
            "boxes": torch.tensor(
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                 [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]],
                dtype=torch.float32, device=device,
            ),
        },
        {
            "mask_ids": torch.tensor([13], dtype=torch.long, device=device),
            "boxes": torch.tensor(
                [[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]],
                dtype=torch.float32, device=device,
            ),
        },
    ]

    pp = _make_preprocessor(max_masks=4, bbox_format="zyxzyx")
    view = pp._build_data_views(
        targets=targets,
        num_frames=T,
        num_videos=B,
        device=device,
        mask_labelmap=labelmap,
    )

    # Top-level keys.
    expected_keys = {
        "num_frames", "num_videos", "masks", "img_ids",
        "labelmaps", "instance_ids", "valid", "presence_t", "boxes", "box_format",
    }
    assert expected_keys.issubset(view.keys()), f"missing keys: {expected_keys - view.keys()}"
    assert view["box_format"] == "zyxzyx"
    assert view["num_frames"] == T and view["num_videos"] == B

    # labelmaps shape and indexing: flat_id = b*T + t -> mask_labelmap[b, t].
    flat = view["labelmaps"]
    assert flat.shape == (B * T, Z, Y, X)
    for b in range(B):
        for t in range(T):
            assert torch.equal(flat[b * T + t], labelmap[b, t].to(torch.int32))

    # Per-frame fields: each list has length T, each tensor has B*K_full rows.
    for t in range(T):
        for key, expected_shape in [
            ("img_ids", (B * K_full,)),
            ("instance_ids", (B * K_full,)),
            ("valid", (B * K_full,)),
            ("presence_t", (B * K_full,)),
            ("boxes", (B * K_full, 6)),
            ("masks", (B * K_full, Z, Y, X)),
        ]:
            assert view[key][t].shape == expected_shape, (
                f"{key}[{t}] shape {tuple(view[key][t].shape)} != {expected_shape}"
            )

    # Padding sentinels: video 0 has 2 real ids (K=2), pads 2; video 1 has 1
    # real id (K=1), pads 3. Real rows carry the sampled ids (order is random,
    # so assert set-wise); pad rows must have instance_ids=-1, valid=False,
    # presence_t=False, boxes=0.
    for t in range(T):
        inst = view["instance_ids"][t]
        valid = view["valid"][t]
        presence = view["presence_t"][t]
        boxes = view["boxes"][t]

        # video 0 slots: rows 0..K_full-1 (2 real, 2 pad). Real ids = {7, 11}.
        assert set(inst[:2].tolist()) == {7, 11}
        assert torch.all(valid[:2])
        assert torch.equal(inst[2:K_full], torch.full((K_full - 2,), -1, dtype=torch.int64))
        assert not torch.any(valid[2:K_full])
        assert not torch.any(presence[2:K_full])
        assert torch.all(boxes[2:K_full] == 0)

        # video 1 slots: rows K_full..2*K_full-1 (1 real, 3 pad).
        offset = K_full
        assert inst[offset].item() == 13
        assert valid[offset]
        assert torch.equal(inst[offset + 1 : 2 * K_full], torch.full((K_full - 1,), -1, dtype=torch.int64))
        assert not torch.any(valid[offset + 1 : 2 * K_full])
        assert not torch.any(presence[offset + 1 : 2 * K_full])
        assert torch.all(boxes[offset + 1 : 2 * K_full] == 0)


def test_lazy_data_view_presence_tracks_frame_membership():
    # presence_t must reflect whether each object's id appears in the frame's
    # labelmap, NOT just whether the row is a real selected object (valid).
    device = torch.device("cpu")
    B, T, Z, Y, X = 1, 3, 2, 3, 4
    K_full = 2

    labelmap = torch.zeros((B, T, Z, Y, X), dtype=torch.int32, device=device)
    labelmap[0, 0, 0, 0, 0] = 5   # frame 0 has id 5 only
    labelmap[0, 1, 1, 2, 3] = 7   # frame 1 has id 7 only
    # frame 2 has neither

    targets = [
        {
            "mask_ids": torch.tensor([5, 7], dtype=torch.long, device=device),
            "boxes": torch.zeros((2, 6), dtype=torch.float32, device=device),
        }
    ]

    pp = _make_preprocessor(max_masks=K_full)
    view = pp._build_data_views(
        targets=targets,
        num_frames=T,
        num_videos=B,
        device=device,
        mask_labelmap=labelmap,
    )

    # The sampled subset (ids 5 and 7) is stable across frames, so locate each
    # id's row once; row order itself is random.
    inst0 = view["instance_ids"][0]
    row5 = int((inst0 == 5).nonzero().item())
    row7 = int((inst0 == 7).nonzero().item())

    # Frame 0: id 5 present, id 7 absent.
    assert view["presence_t"][0][row5] and not view["presence_t"][0][row7]
    # Frame 1: id 5 absent, id 7 present.
    assert (not view["presence_t"][1][row5]) and view["presence_t"][1][row7]
    # Frame 2: both absent.
    assert not view["presence_t"][2][row5] and not view["presence_t"][2][row7]

    # valid stays True for both rows regardless of frame membership.
    for t in range(T):
        assert torch.equal(view["valid"][t], torch.tensor([True, True]))


def test_lazy_data_view_empty_targets_all_pad():
    # Missing targets entry (e.g. inference) should produce a single pad row
    # (rows follow the batch's instance counts, never max_masks).
    device = torch.device("cpu")
    B, T, Z, Y, X = 1, 1, 2, 2, 2
    K_full = 3

    labelmap = torch.randint(0, 4, (B, T, Z, Y, X), dtype=torch.int32, device=device)
    targets: list[dict] = [{}]  # no mask_ids

    pp = _make_preprocessor(max_masks=K_full)
    view = pp._build_data_views(
        targets=targets,
        num_frames=T,
        num_videos=B,
        device=device,
        mask_labelmap=labelmap,
    )

    assert torch.equal(view["instance_ids"][0], torch.tensor([-1], dtype=torch.int64))
    assert not torch.any(view["valid"][0])
    assert not torch.any(view["presence_t"][0])
    assert torch.all(view["boxes"][0] == 0)


# --------------------------------------------------------------------------- #
# forward(): single-source labelmap split off the data_tensor channel
# --------------------------------------------------------------------------- #


def _make_forward_pp(
    max_masks: int,
    expect_mask_channel: bool = True,
    input_shape: tuple = (2, 2, 3, 4, 2),
) -> SAM2VideoPreprocessor:
    pp = SAM2VideoPreprocessor.__new__(SAM2VideoPreprocessor)
    pp.max_masks = max_masks
    pp.bbox_format = "zyxzyx"
    pp.boxes_normalized = False
    pp.rng = torch.Generator()
    pp.rng.manual_seed(0)
    pp.expect_mask_channel = expect_mask_channel
    pp.dtype = torch.float32
    pp.transforms = None
    # forward() splits channels by ROLE, which reads TARGET_ROLES -> _data_types() ->
    # base_dense_data_type / input_format. __init__ normally sets these; this fixture
    # bypasses __init__, so stub them or the property raises AttributeError (masked by
    # nn.Module.__getattr__ into a confusing "no attribute 'TARGET_ROLES'").
    pp.input_format = "TZYXC"
    # _apply_transforms now runs the post-transform spatial guard for every
    # task, which compares against input_shape (T, Z, Y, X, C) -- set it to the
    # POST-transform shape the test expects.
    pp.input_shape = tuple(input_shape)
    pp.base_dense_data_type = {
        "kind": "dense",
        "layout": pp.input_format,
        "role": "input",
        "has_time": True,
    }
    pp.channels = None
    return pp


def test_forward_splits_labelmap_from_channel_and_builds_views():
    # The labelmap rides the last data_tensor channel; forward must split it
    # (int32, pre-cast), attach it per-target, and build views off of it.
    device = torch.device("cpu")
    B, T, Z, Y, X, C = 1, 2, 2, 3, 4, 2

    img = torch.randn(B, T, Z, Y, X, C, device=device)
    lm = torch.zeros(B, T, Z, Y, X, device=device)
    lm[0, 0, 0, 0, 0] = 7
    lm[0, 1, 1, 2, 3] = 11
    inputs = torch.cat([img, lm.unsqueeze(-1)], dim=-1)  # (B,T,Z,Y,X,C+1)

    targets = [
        {
            "mask_ids": torch.tensor([7, 11], dtype=torch.long, device=device),
            "boxes": torch.zeros((2, 6), dtype=torch.float32, device=device),
        }
    ]

    pp = _make_forward_pp(max_masks=4)
    # channel_mapping names the labelmap channel's ROLE. forward used to inject
    # `_cm[C - 1] = "instance_segmentation"` itself on the positional assumption
    # that the labelmap is last; the DB names the channel now, so the fixture
    # supplies what the loader's role table would.
    out = pp.forward(
        {
            "data_tensor": inputs,
            "metainfo": {"targets": targets, "channel_mapping": {C: "instance_masks"}},
        },
        0.0,
        0,
    )
    dv = out["metainfo"]["sam2_views"]

    # Image channel was stripped: flat batch is (T*B, C, Z, Y, X).
    # Platform layout: SAM2 converts to (B*T, C, Z, Y, X) at its own boundary.
    assert out["data_tensor"].shape == (B, T, Z, Y, X, C)

    # The flat labelmaps must equal the int32 channel, frame by frame.
    for b in range(B):
        for t in range(T):
            assert torch.equal(dv["labelmaps"][b * T + t], lm[b, t].to(torch.int32))

    # The per-target label_map was attached from the channel (mutated in place).
    assert "label_map" in targets[0]
    assert torch.equal(targets[0]["label_map"], lm[0].to(torch.int32))

    # Real instances were materialized into the views.
    assert any(bool(torch.any(dv["valid"][t])) for t in range(T))


def test_forward_no_mask_channel_emits_empty_views():
    device = torch.device("cpu")
    B, T, Z, Y, X, C = 1, 2, 2, 3, 4, 2
    inputs = torch.randn(B, T, Z, Y, X, C, device=device)  # no labelmap channel

    pp = _make_forward_pp(max_masks=4, expect_mask_channel=False)
    out = pp.forward({"data_tensor": inputs, "metainfo": {"targets": [{}]}}, 0.0, 0)
    dv = out["metainfo"]["sam2_views"]

    # Platform layout: SAM2 converts to (B*T, C, Z, Y, X) at its own boundary.
    assert out["data_tensor"].shape == (B, T, Z, Y, X, C)
    assert dv["num_frames"] == T and dv["num_videos"] == B
    assert not any(bool(torch.any(dv["valid"][t])) for t in range(dv["num_frames"]))


class _CropLastX:
    """Minimal geometric transform: crop the X axis on image + label_map in
    lockstep, mirroring how Crop warps both entities together."""

    def __init__(self, xkeep: int):
        self.xkeep = xkeep

    def __call__(self, sample: dict) -> dict:
        sample["data_tensor"] = sample["data_tensor"][..., : self.xkeep, :]
        for t in sample["metainfo"].get("targets", []):
            if "label_map" in t:
                t["label_map"] = t["label_map"][..., : self.xkeep]
        return sample


def test_forward_transform_keeps_image_and_labelmap_coherent():
    # With a geometric transform that crops both image and per-target label_map,
    # the post-transform stack/assert must pass and masks must come from the
    # cropped labelmap (no RuntimeError, no shape mismatch).
    device = torch.device("cpu")
    B, T, Z, Y, X, C = 1, 1, 1, 2, 4, 2
    XKEEP = 2

    img = torch.randn(B, T, Z, Y, X, C, device=device)
    lm = torch.zeros(B, T, Z, Y, X, device=device)
    lm[0, 0, 0, 0, 1] = 9  # inside the kept region (x=1 < XKEEP)
    lm[0, 0, 0, 1, 3] = 4  # outside the kept region (x=3 >= XKEEP) -> dropped
    inputs = torch.cat([img, lm.unsqueeze(-1)], dim=-1)

    targets = [
        {
            "mask_ids": torch.tensor([9], dtype=torch.long, device=device),
            "boxes": torch.zeros((1, 6), dtype=torch.float32, device=device),
        }
    ]

    # input_shape reflects the POST-transform spatial shape (the guard in
    # _apply_transforms checks against it): (T, Z, Y, XKEEP, C).
    pp = _make_forward_pp(max_masks=2, input_shape=(T, Z, Y, XKEEP, C))
    pp.transforms = [_CropLastX(XKEEP)]
    out = pp.forward(
        {
            "data_tensor": inputs,
            "metainfo": {"targets": targets, "channel_mapping": {C: "instance_masks"}},
        },
        0.0,
        0,
    )
    dv = out["metainfo"]["sam2_views"]

    # Image and labelmap were cropped identically along X.
    assert out["data_tensor"].shape == (B, T, Z, Y, XKEEP, C)
    assert dv["labelmaps"].shape == (B * T, Z, Y, XKEEP)
    assert torch.equal(targets[0]["label_map"], lm[..., :XKEEP][0].to(torch.int32))

    # masks/instance_ids are built from the CROPPED labelmap: id 9 (x=1) survives, id 4 (x=3) is gone.
    K_FULL = 1                                       # one instance in the batch -> one row (max_masks 2 caps, no pad)
    ids = dv["instance_ids"][0]                      # frame 0, rows = B*K_full
    assert ids.tolist() == [9]
    assert dv["valid"][0].tolist() == [True]
    assert dv["presence_t"][0].tolist() == [True]

    m = dv["masks"][0]                               # _LazyFrameMasks -> (B*K_full, Z, Y, XKEEP) bool
    assert m.shape == (B * K_FULL, Z, Y, XKEEP) and m.dtype == torch.bool
    assert torch.equal(m[0], targets[0]["label_map"][0] == 9)
    assert m[0].sum().item() == 1 and bool(m[0, 0, 0, 1])
    assert not torch.any(dv["labelmaps"] == 4)


# --------------------------------------------------------------------------- #
# _split_channels (shared base helper)
#
# Replaces the removed `_split_labelmap_int32`: the split is now role-driven via
# channel_mapping rather than positional "last channel is the labelmap".
# --------------------------------------------------------------------------- #


def _split_pp() -> SAM2VideoPreprocessor:
    pp = SAM2VideoPreprocessor.__new__(SAM2VideoPreprocessor)
    pp.dtype = torch.float32
    pp.channels = None
    pp.input_format = "TZYXC"
    pp.base_dense_data_type = {
        "kind": "dense", "layout": pp.input_format, "role": "input", "has_time": True,
    }
    pp.bbox_format = "zyxzyx"
    return pp


def test_split_channels_int32_precast_preserves_large_ids():
    """The object-channel tail is cast to int32, not bf16: ids > 4096 must survive."""
    pp = _split_pp()
    img = torch.randn(1, 4, 2)
    lm = torch.tensor([[300.0, 0.0, 4097.0, 0.0]]).unsqueeze(-1)  # ids > bf16-exact
    x = torch.cat([img, lm], dim=-1)  # (1, 4, 3)
    meta = {"channel_mapping": {2: "instance_masks"}}

    images, targets_by_role = pp._split_channels(x, meta)
    labelmap = targets_by_role["instance_masks"]
    assert images.shape == (1, 4, 2)
    assert labelmap.dtype == torch.int32
    assert labelmap.tolist() == [[300, 0, 4097, 0]]


def test_split_channels_no_target_role_returns_image_view():
    """No object-role channel -> no labelmap, and images is a zero-copy view."""
    pp = _split_pp()
    x = torch.randn(1, 4, 3)
    images, targets_by_role = pp._split_channels(x, {"channel_mapping": {}})
    assert targets_by_role == {}
    # A basic-slice view, not the identical object: _split_channels always slices
    # the signal prefix. Shared storage is what matters -- no gather, no copy.
    assert images.data_ptr() == x.data_ptr()
    assert images.shape == x.shape


def _targets(id_lists, boxes=None, device="cpu"):
    out = []
    for i, ids in enumerate(id_lists):
        t = {"mask_ids": torch.tensor(ids, dtype=torch.long, device=device)}
        if boxes is not None:
            t["boxes"] = boxes[i]
        out.append(t)
    return out


def _labelmap(B, T, Z, Y, X, id_lists):
    lm = torch.zeros((B, T, Z, Y, X), dtype=torch.int32)
    for b, ids in enumerate(id_lists):
        for j, i in enumerate(ids):
            lm[b, :, 0, j % Y, (j * 2) % X] = i
    return lm


# --------------------------------------------------------------------------- #
# rows per video follow the batch's instance counts (capped by max_masks)
# --------------------------------------------------------------------------- #

def test_rows_per_video_follow_batch_max_not_max_masks():
    ids = [[1, 2, 3], [4, 5, 6, 7, 8]]
    lm = _labelmap(2, 1, 2, 4, 8, ids)
    view = _make_preprocessor(max_masks=32)._build_data_views(
        targets=_targets(ids), num_frames=1, num_videos=2, device=torch.device("cpu"), mask_labelmap=lm
    )
    assert view["masks"][0].shape[0] == 2 * 5      # 5 = largest count, not 32
    assert view["valid"][0].sum().item() == 8       # 3 + 5 real rows


def test_rows_per_video_capped_by_max_masks():
    ids = [[1, 2, 3, 4, 5, 6]]
    lm = _labelmap(1, 1, 2, 4, 8, ids)
    view = _make_preprocessor(max_masks=4)._build_data_views(
        targets=_targets(ids), num_frames=1, num_videos=1, device=torch.device("cpu"), mask_labelmap=lm
    )
    assert view["masks"][0].shape[0] == 4
    assert view["valid"][0].sum().item() == 4


def test_rows_per_video_at_least_one_with_empty_targets():
    lm = torch.zeros((1, 1, 2, 2, 2), dtype=torch.int32)
    view = _make_preprocessor(max_masks=8)._build_data_views(
        targets=[{"mask_ids": torch.zeros(0, dtype=torch.long)}], num_frames=1, num_videos=1,
        device=torch.device("cpu"), mask_labelmap=lm,
    )
    assert view["masks"][0].shape[0] == 1 and not view["valid"][0].any()


# --------------------------------------------------------------------------- #
# _LazyFrameMasks: per-frame cache
# --------------------------------------------------------------------------- #

def test_lazy_frame_masks_cache_returns_same_tensor_until_release():
    lm = _labelmap(1, 2, 2, 4, 8, [[3, 5]])
    masks = _LazyFrameMasks(
        mask_labelmap=lm, sampled_ids_per_b=[torch.tensor([3, 5])], max_masks=2, device=torch.device("cpu")
    )
    a = masks[0]
    assert masks[0] is a
    assert masks[1] is not a and masks[1] is masks[1]
    masks.release()
    b = masks[0]
    assert b is not a and torch.equal(a, b)


# --------------------------------------------------------------------------- #
# normalized collator boxes are denormalized to voxels in the view
# --------------------------------------------------------------------------- #

def test_normalized_boxes_are_scaled_to_voxels_in_the_view():
    Z, Y, X = 2, 4, 8
    ids = [[1]]
    lm = _labelmap(1, 1, Z, Y, X, ids)
    boxes = [torch.tensor([[0.5, 0.25, 0.5, 0.25, 0.5, 1.0]])]  # cxcyczwhd, normalized
    view = _make_preprocessor(max_masks=2, bbox_format="cxcyczwhd", boxes_normalized=True)._build_data_views(
        targets=_targets(ids, boxes), num_frames=1, num_videos=1, device=torch.device("cpu"), mask_labelmap=lm
    )
    got = view["boxes"][0][0]
    assert torch.allclose(got, torch.tensor([0.5 * X, 0.25 * Y, 0.5 * Z, 0.25 * X, 0.5 * Y, 1.0 * Z]))
    assert view["box_format"] == "cxcyczwhd"


def test_absolute_boxes_are_left_alone():
    ids = [[1]]
    lm = _labelmap(1, 1, 2, 4, 8, ids)
    boxes = [torch.tensor([[0.0, 1.0, 2.0, 1.0, 3.0, 6.0]])]
    view = _make_preprocessor(max_masks=2, bbox_format="zyxzyx", boxes_normalized=False)._build_data_views(
        targets=_targets(ids, boxes), num_frames=1, num_videos=1, device=torch.device("cpu"), mask_labelmap=lm
    )
    assert torch.equal(view["boxes"][0][0], boxes[0][0])
