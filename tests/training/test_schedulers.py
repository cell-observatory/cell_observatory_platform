"""Unit tests for the warmup-stable-decay LR schedule and the cosine
weight-decay schedule: construction-time application, opt-out groups, horizon
clamps, and the post-replay re-apply the trainer performs on resume. CPU-only."""

from types import SimpleNamespace

import pytest
import torch

from cell_observatory_platform.training.schedulers import (
    CosineWeightDecaySchedule,
    WarmupStableDecaySchedule,
)


def _make_wsd(lr=1.0):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=lr)
    sched = WarmupStableDecaySchedule(
        opt, warmup_steps=10, anneal_steps=4, T_max=30,
        start_lr=0.1, ref_lr=lr, final_lr=0.0, update_type="step",
    )
    return opt, sched


def test_cosine_wd_preserves_zero_wd_and_wd_exclude_groups():
    """Scheduling only touches groups that opted into decay: a zero-WD group
    (biases/norms) and a WD_exclude group keep their construction-time values."""
    params = [torch.nn.Parameter(torch.randn(2)) for _ in range(3)]
    opt = torch.optim.SGD(
        [
            {"params": [params[0]], "weight_decay": 0.05},
            {"params": [params[1]], "weight_decay": 0.0},
            {"params": [params[2]], "weight_decay": 0.07, "WD_exclude": True},
        ],
        lr=0.1,
    )
    sched = CosineWeightDecaySchedule(optimizer=opt, ref_wd=0.05, T_max=10, final_wd=0.0)

    for _ in range(5):
        sched.step()

    assert opt.param_groups[0]["weight_decay"] != pytest.approx(0.05)
    assert opt.param_groups[1]["weight_decay"] == 0.0
    assert opt.param_groups[2]["weight_decay"] == pytest.approx(0.07)


def test_wsd_applies_step0_lr_at_construction():
    """The optimizer runs its first step at start_lr, not at the peak lr it
    was constructed with."""
    opt, _ = _make_wsd(lr=1.0)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)


def test_cosine_wd_holds_final_wd_past_tmax():
    """Past T_max the schedule holds final_wd rather than rebounding toward
    ref_wd along the cosine."""
    opt = SimpleNamespace(param_groups=[{"weight_decay": 0.05}])
    sched = CosineWeightDecaySchedule(opt, ref_wd=0.05, T_max=10, final_wd=0.4)
    for _ in range(15):
        wd = sched.step()
    assert wd == pytest.approx(0.4)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.4)


def test_wsd_holds_final_lr_past_horizon():
    """Past warmup + stable + anneal the LR is clamped at final_lr and never
    continues the linear anneal below it."""
    opt = SimpleNamespace(param_groups=[{"lr": 0.0}])
    s = WarmupStableDecaySchedule(
        opt, warmup_steps=10, anneal_steps=10, T_max=40, start_lr=0.0,
        ref_lr=1.0, final_lr=0.1,
    )
    for _ in range(100):
        s.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)


def test_cosine_wd_applies_ref_wd_at_construction_and_keeps_opt_out_groups():
    """Construction stamps ref_wd onto every scheduled group so the first
    optimizer step already decays at ref_wd; zero-WD groups stay at zero."""
    opt = SimpleNamespace(param_groups=[
        {"weight_decay": 0.123},
        {"weight_decay": 0.0},
    ])
    CosineWeightDecaySchedule(opt, ref_wd=0.05, T_max=10, final_wd=0.4)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.05)
    assert opt.param_groups[1]["weight_decay"] == 0.0


def test_wsd_reapply_after_replay_overrides_restored_optimizer_lr():
    """After the resume replay, re-applying the schedule at its current step
    puts the optimizer back on the replayed LR even when the optimizer-state
    restore overwrote the param-group lr with the checkpoint-time value."""
    opt_fresh, sched_fresh = _make_wsd()
    for _ in range(7):
        sched_fresh.step()
    lr_fresh = opt_fresh.param_groups[0]["lr"]
    assert lr_fresh == pytest.approx(0.1 + 0.7 * 0.9)

    opt_resume, sched_resume = _make_wsd()
    for _ in range(7):
        sched_resume.step()
    opt_resume.param_groups[0]["lr"] = 123.0
    sched_resume._apply(sched_resume._step)
    assert opt_resume.param_groups[0]["lr"] == pytest.approx(lr_fresh)
    assert sched_resume._step == sched_fresh._step == 7
