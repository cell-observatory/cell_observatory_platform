"""Correction loop + IoU target on the stride-4 mask stream.

With `models.meta_arch.sam.correction_on_low_res=True` a training step never
materializes the full-res mask stream: the per-round decoder call skips the
trilinear upsample, correction clicks are sampled against a max-pooled ("any
voxel in the 4^3 cell") GT and then jittered back into full-res voxel
coordinates, and the criterion's dense IoU loss runs on the low-res stream
against the same max-pooled GT (`criterion_args.iou_on_low_res`).

Coverage here:
  1. `gt_low` is exactly the any-in-cell max-pool of the GT.
  2. Sampled clicks land inside a cell whose low-res error label matches, and
     inside the volume.
  3. Flags off: the SAM2 forward still emits the full-res stream, takes the
     pre-change branches (no `gt_low` built, `with_high_res=True`), and matches
     a model built without the flag at all, tensor-for-tensor.
  4. Flags on: the forward runs, emits NO full-res multimask tensor, and the
     loss is finite and backpropagates.

CPU-only. (3)/(4) reuse the smoke fixture's batch builder on CPU; the smoke test
itself is `@pytest.mark.cuda` and cannot run here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from cell_observatory_platform.models.layers.utils import get_next_point
from cell_observatory_platform.models.meta_arch.sam import low_res_cells_to_voxels


_SMOKE_PATH = Path(__file__).resolve().parent / "test_sam2_smoke.py"


def _smoke_module():
    """Import the smoke test module for its config/batch builders only.

    Importing it does NOT run its cuda-marked tests; we just reuse
    `_compose_smoke_cfg` / `_make_data_sample`, both of which already take a
    `device`.
    """
    spec = importlib.util.spec_from_file_location("_sam2_smoke_fixtures", _SMOKE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 1) gt_low == any-in-cell max-pool of the GT
# --------------------------------------------------------------------------- #

def test_gt_low_is_any_voxel_in_cell():
    torch.manual_seed(0)
    K, Z, Y, X, s = 3, 8, 8, 16, 4
    gt = torch.rand(K, 1, Z, Y, X) < 0.05          # sparse foreground
    gt_low = F.max_pool3d(gt.float(), kernel_size=s, stride=s) > 0

    assert gt_low.shape == (K, 1, Z // s, Y // s, X // s)

    # brute-force reference: fold each 4^3 block and OR it
    ref = (
        gt.reshape(K, 1, Z // s, s, Y // s, s, X // s, s)
        .any(dim=7).any(dim=5).any(dim=3)
    )
    assert torch.equal(gt_low, ref)


def test_gt_low_marks_a_cell_for_a_single_voxel():
    s = 4
    gt = torch.zeros(1, 1, 8, 8, 8, dtype=torch.bool)
    gt[0, 0, 5, 2, 7] = True
    gt_low = F.max_pool3d(gt.float(), kernel_size=s, stride=s) > 0
    assert gt_low.sum().item() == 1
    assert gt_low[0, 0, 5 // s, 2 // s, 7 // s]


# --------------------------------------------------------------------------- #
# 2) clicks land in the right cell, in full-res voxel coordinates
# --------------------------------------------------------------------------- #

def test_low_res_cells_map_into_their_own_cell():
    torch.manual_seed(1)
    s = 4
    cells = torch.tensor([[[0.0, 0.0, 0.0]], [[3.0, 1.0, 2.0]]])   # (B, P, 3) as (x, y, z)
    for _ in range(20):
        voxels = low_res_cells_to_voxels(cells, s)
        assert torch.equal(torch.div(voxels, s, rounding_mode="floor"), cells)
        assert voxels.dtype == cells.dtype


def test_sampled_correction_points_match_the_low_res_error_label():
    """`get_next_point` on the stride-4 grid + the cell->voxel map must give a
    click inside the volume whose cell carries the sampled label."""
    torch.manual_seed(2)
    s = 4
    K, Z, Y, X = 2, 8, 8, 16
    zl, yl, xl = Z // s, Y // s, X // s

    gt_low = torch.zeros(K, 1, zl, yl, xl, dtype=torch.bool)
    pred_low = torch.zeros(K, 1, zl, yl, xl, dtype=torch.bool)
    # row 0: one false NEGATIVE (gt=1, pred=0) -> positive click there
    gt_low[0, 0, 1, 1, 2] = True
    # row 1: one false POSITIVE (gt=0, pred=1) -> negative click there
    pred_low[1, 0, 0, 0, 3] = True

    for _ in range(10):
        cells, labels = get_next_point(
            input_fmt="TZYXC", gt_masks=gt_low, pred_masks=pred_low, method="uniform"
        )
        points = low_res_cells_to_voxels(cells, s)

        assert points.shape == (K, 1, 3)
        # inside the FULL-RES volume, in (x, y, z) order
        assert torch.all(points[..., 0] < X) and torch.all(points[..., 0] >= 0)
        assert torch.all(points[..., 1] < Y) and torch.all(points[..., 1] >= 0)
        assert torch.all(points[..., 2] < Z) and torch.all(points[..., 2] >= 0)

        cell_idx = torch.div(points, s, rounding_mode="floor").long()
        for k in range(K):
            cx, cy, cz = cell_idx[k, 0].tolist()
            err_pos = bool(gt_low[k, 0, cz, cy, cx] & ~pred_low[k, 0, cz, cy, cx])
            err_neg = bool(~gt_low[k, 0, cz, cy, cx] & pred_low[k, 0, cz, cy, cx])
            if labels[k, 0].item() == 1:
                assert err_pos, f"row {k}: positive click on a non-false-negative cell"
            else:
                # negative clicks come from false positives, or (when the row is
                # exact) anywhere in the background
                assert err_neg or not bool(gt_low[k, 0, cz, cy, cx])


# --------------------------------------------------------------------------- #
# 3)/(4) end-to-end SAM2 forward on CPU
# --------------------------------------------------------------------------- #

def _build(correction_on_low_res, set_flag: bool = True, rounds: int = 2):
    """Build a tiny SAM2 + preprocessor + batch on CPU.

    `set_flag=False` omits `correction_on_low_res` from the config entirely, so
    the constructor default is exercised (the "today's behaviour" path).
    """
    import cell_observatory_platform.utils._register  # noqa: F401
    from cell_observatory_platform.models.meta_arch.sam import BUILD as BUILD_SAM2
    from cell_observatory_platform.models.layers.preprocessor import SAM2VideoPreprocessor

    smoke = _smoke_module()
    B = T = 1
    Z = Y = X = 32
    C_in = 1
    max_masks = 2

    cfg, _, input_shape, patch_shape = smoke._compose_smoke_cfg(
        T=T, Z=Z, Y=Y, X=X, C_in=C_in, max_masks=max_masks
    )
    sam_cfg = cfg.models.meta_arch.sam
    # correction_on_low_res requires the low-res focal/dice stream
    sam_cfg.criterion_args.use_point_sampling = True
    sam_cfg.criterion_args.low_res_multimasks = True
    sam_cfg.num_correction_pt_per_frame = rounds
    if set_flag:
        sam_cfg.correction_on_low_res = correction_on_low_res
    else:
        assert "correction_on_low_res" in sam_cfg
        del sam_cfg["correction_on_low_res"]

    torch.manual_seed(0)
    model = BUILD_SAM2(cfg).train()

    pp = SAM2VideoPreprocessor(
        transforms_list=None,
        with_masking=False,
        mask_generator=None,
        patch_shape=tuple(patch_shape[:4]),
        dtype="float32",
        input_format="TZYXC",
        input_shape=tuple(input_shape),
        seed=0,
        expect_mask_channel=True,
        max_masks=max_masks,
        require_targets=True,
        bbox_format="zyxzyx",
    )
    data_sample = smoke._make_data_sample(
        B=B, T=T, Z=Z, Y=Y, X=X, C_in=C_in, max_masks=max_masks,
        device=torch.device("cpu"), seed=0,
    )
    return model, pp, data_sample, (Z, Y, X)


def _run(model, pp, data_sample, seed: int = 1234):
    torch.manual_seed(seed)
    processed = pp.forward(data_sample, data_time=0.0, idx=0)
    return model(processed)


def test_flags_off_takes_the_pre_change_branches_and_keeps_the_full_res_stream():
    """Flag explicitly False == flag absent, tensor-for-tensor; the flag-off run
    takes the pre-change branches; and the full-res mask stream is still produced
    at full resolution.

    NOTE: this is not a bit-identity check against pre-change code -- see the
    module docstring's LIMITATION note. Both models here are built from the
    SAME post-change source, so this cannot detect a regression in the shared
    default path; it pins the branch SELECTION, not the branch contents.
    """
    m_off, pp_off, ds_off, (Z, Y, X) = _build(False, set_flag=True)
    m_def, pp_def, ds_def, _ = _build(False, set_flag=False)

    assert m_off.correction_on_low_res is False
    assert m_def.correction_on_low_res is False
    assert m_def.criterion.iou_on_low_res is False

    # identical init (both built under torch.manual_seed(0))
    for (n1, p1), (n2, p2) in zip(m_off.named_parameters(), m_def.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), f"param mismatch at {n1}"

    losses_off, outs_off = _run(m_off, pp_off, ds_off)
    losses_def, outs_def = _run(m_def, pp_def, ds_def)

    for k in losses_off:
        a, b = losses_off[k], losses_def[k]
        assert torch.equal(a.detach(), b.detach()), f"{k}: {a.item()} != {b.item()}"

    # --- branch pinning: the flag-off run must take the pre-change branches ---
    # A fresh build/sample: the preprocessor consumes metainfo["targets"], so a
    # data_sample cannot be fed through pp.forward() twice.
    m_p, pp_p, ds_p, _ = _build(False, set_flag=True)
    cls = type(m_p)
    seen_high_res: list = []
    seen_gt_low: list = []
    orig_heads = cls._forward_sam_heads
    orig_prep = cls.prepare_prompt_inputs

    def _spy_heads(self, *a, **kw):
        seen_high_res.append(kw.get("with_high_res", True))
        return orig_heads(self, *a, **kw)

    def _spy_prep(self, *a, **kw):
        out = orig_prep(self, *a, **kw)
        seen_gt_low.append(out.get("gt_masks_low_per_frame", None))
        return out

    cls._forward_sam_heads = _spy_heads
    cls.prepare_prompt_inputs = _spy_prep
    try:
        _run(m_p, pp_p, ds_p)
    finally:
        cls._forward_sam_heads = orig_heads
        cls.prepare_prompt_inputs = orig_prep

    # no stride-4 GT is built at all when the flag is off
    assert seen_gt_low, "expected prepare_prompt_inputs to be called"
    for g in seen_gt_low:
        assert g == {}, f"flag off must not build the stride-4 GT, got {type(g)} {g!r}"

    # and every _forward_sam_heads call must ask for the full-res upsample
    assert seen_high_res, "expected _forward_sam_heads to be called"
    assert all(v is True for v in seen_high_res), (
        f"flag off must keep with_high_res=True on every call, saw {seen_high_res}"
    )

    # the full-res stream exists and really is full-res
    hi = outs_off[0]["multistep_pred_multimasks_high_res"]
    assert isinstance(hi, list) and len(hi) == 3           # 1 initial + 2 rounds
    for t in hi:
        assert t is not None
        assert tuple(t.shape[-3:]) == (Z, Y, X)
    assert outs_off[0]["pred_masks_high_res"] is not None


def test_flags_on_emits_no_full_res_multimask_and_the_loss_is_finite():
    model, pp, ds, (Z, Y, X) = _build(True, set_flag=True)

    assert model.correction_on_low_res is True
    # the SAM2 flag force-sets the coupled criterion flag
    assert model.criterion.iou_on_low_res is True

    losses, outs = _run(model, pp, ds)

    # NO full-res multimask tensor anywhere in the output dict
    hi = outs[0]["multistep_pred_multimasks_high_res"]
    assert isinstance(hi, list) and len(hi) == 3
    assert all(t is None for t in hi), "correction_on_low_res must skip the upsample"
    assert all(t is None for t in outs[0]["multistep_pred_masks_high_res"])
    assert outs[0]["pred_masks_high_res"] is None

    # the low-res stream is at stride mask_downsample_factor
    s = model.mask_downsample_factor
    lo = outs[0]["multistep_pred_multimasks"]
    for t in lo:
        assert tuple(t.shape[-3:]) == (Z // s, Y // s, X // s)

    for key in ("loss_mask", "loss_dice", "loss_iou", "loss_class"):
        assert torch.isfinite(losses[key]).item(), f"{key} not finite"
    total = losses[model.criterion.core_loss_key]
    assert torch.isfinite(total).item()
    total.backward()
    grads = [
        p.grad for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None and n.startswith("sam_mask_decoder")
    ]
    assert grads, "no gradient reached the mask decoder"
    assert any(g.abs().sum() > 0 for g in grads)


def test_correction_on_low_res_requires_low_res_multimasks():
    """The coupling is asserted, not silently ignored."""
    import cell_observatory_platform.utils._register  # noqa: F401
    from cell_observatory_platform.models.meta_arch.sam import BUILD as BUILD_SAM2

    smoke = _smoke_module()
    cfg, _, _, _ = smoke._compose_smoke_cfg(T=1, Z=32, Y=32, X=32, C_in=1, max_masks=2)
    sam_cfg = cfg.models.meta_arch.sam
    sam_cfg.criterion_args.use_point_sampling = True
    sam_cfg.criterion_args.low_res_multimasks = False   # incompatible
    sam_cfg.correction_on_low_res = True
    with pytest.raises(AssertionError):
        BUILD_SAM2(cfg)


def test_iou_on_low_res_requires_low_res_multimasks():
    from cell_observatory_platform.training.losses import MultiStepMultiMasksAndIousLoss

    with pytest.raises(AssertionError):
        MultiStepMultiMasksAndIousLoss(
            input_fmt="TZYXC",
            weight_dict={"loss_mask": 1.0, "loss_dice": 1.0, "loss_iou": 1.0, "loss_class": 0.0},
            use_point_sampling=True,
            low_res_multimasks=False,
            iou_on_low_res=True,
        )


def test_iou_target_is_max_pooled_to_the_prediction_grid():
    from cell_observatory_platform.training.losses import MultiStepMultiMasksAndIousLoss

    crit = MultiStepMultiMasksAndIousLoss(
        input_fmt="TZYXC",
        weight_dict={"loss_mask": 1.0, "loss_dice": 1.0, "loss_iou": 1.0, "loss_class": 0.0},
        use_point_sampling=True,
        low_res_multimasks=True,
        iou_on_low_res=True,
    )
    target = torch.zeros(2, 1, 8, 8, 16)
    target[0, 0, 5, 2, 7] = 1.0
    ref = torch.zeros(2, 1, 2, 2, 4)          # stride 4

    pooled = crit._iou_target(target, ref)
    assert pooled.shape == (2, 1, 2, 2, 4)
    assert pooled.sum().item() == 1.0
    assert pooled[0, 0, 1, 0, 1] == 1.0

    # full-res reference grid -> untouched (identity)
    assert crit._iou_target(target, torch.zeros(2, 1, 8, 8, 16)) is target
