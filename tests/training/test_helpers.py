"""Unit tests for training/helpers.py: the mode-aware best-metric sentinel,
global seeding, stale-actor cleanup, batched loss materialization, and the
resume guards (legacy-sidecar refusal, fresh-run overwrite guard). CPU-only."""

import math
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import cell_observatory_platform.training.helpers as helpers
from cell_observatory_platform.training.helpers import (
    _loss_dict_to_floats,
    initial_best_metric,
    resume_model_state,
    resume_run,
    set_global_seed,
)


def test_initial_best_metric_is_mode_aware():
    """A fresh run's best metric is +inf for val_mode=min and -inf for
    val_mode=max; a missing val_mode defaults to min."""
    cfg_min = OmegaConf.create({"evaluation": {"val_mode": "min"}})
    cfg_max = OmegaConf.create({"evaluation": {"val_mode": "max"}})
    cfg_missing = OmegaConf.create({})
    assert initial_best_metric(cfg_min) == math.inf
    assert initial_best_metric(cfg_max) == -math.inf
    assert initial_best_metric(cfg_missing) == math.inf


def test_set_global_seed_is_deterministic():
    """Re-seeding with the same value replays the python, numpy and torch RNG
    streams exactly."""
    set_global_seed(1234)
    torch_a = torch.randn(8)
    np_a = np.random.rand(8)
    py_a = [random.random() for _ in range(4)]

    set_global_seed(1234)
    torch_b = torch.randn(8)
    np_b = np.random.rand(8)
    py_b = [random.random() for _ in range(4)]

    assert torch.equal(torch_a, torch_b)
    assert np.array_equal(np_a, np_b)
    assert py_a == py_b


def test_kill_stale_actor_noop_when_absent():
    """When no detached actor is registered under the name, nothing is killed."""
    with patch.object(helpers.ray, "get_actor", side_effect=ValueError("nope")), \
         patch.object(helpers.ray, "kill") as kill:
        helpers.kill_stale_actor("save_worker_rank_0")
    kill.assert_not_called()


def test_kill_stale_actor_kills_existing_actor():
    """A leftover detached actor registered under the name is killed."""
    stale = object()
    with patch.object(helpers.ray, "get_actor", return_value=stale), \
         patch.object(helpers.ray, "kill") as kill:
        helpers.kill_stale_actor("viz_worker_rank_0")
    kill.assert_called_once_with(stale)


def test_kill_stale_actor_swallows_kill_failure():
    """A failing ray.kill is logged, not propagated, so startup proceeds."""
    with patch.object(helpers.ray, "get_actor", return_value=object()), \
         patch.object(helpers.ray, "kill", side_effect=RuntimeError("boom")):
        helpers.kill_stale_actor("save_worker_rank_0")


def test_loss_dict_to_floats_matches_item_with_one_device_sync():
    """The batched conversion yields the same floats as per-key .item() while
    issuing exactly one device-to-host transfer for the whole dict."""
    loss_dict = {f"l{i}": torch.tensor(float(i) * 0.5) for i in range(20)}
    loss_dict["plain"] = 3.25
    expected = {k: (float(v.item()) if torch.is_tensor(v) else float(v))
                for k, v in loss_dict.items()}

    cpu_calls = {"n": 0}
    orig_cpu = torch.Tensor.cpu

    def counting_cpu(self, *a, **k):
        cpu_calls["n"] += 1
        return orig_cpu(self, *a, **k)

    with patch.object(torch.Tensor, "cpu", counting_cpu):
        out = helpers._loss_dict_to_floats(loss_dict)
    assert out == expected
    assert cpu_calls["n"] == 1


def test_loss_dict_to_floats_accepts_zero_dim_and_shaped_scalars():
    """Zero-dim tensors and single-element shaped tensors both reduce to floats."""
    out = _loss_dict_to_floats({"a": torch.tensor([2.0]).reshape(()),
                                "b": torch.tensor(4.0)})
    assert out == {"a": 2.0, "b": 4.0}


def test_resume_model_state_refuses_sidecar_without_best_loss(tmp_path):
    """A checkpoint whose sidecar carries best_loss=None (legacy, no lineage)
    is refused instead of being resumed with a fabricated best-metric baseline."""
    cfg = OmegaConf.create({
        "backend": "DEEPSPEED",
        "checkpoint": {"checkpoint_manager": {
            "resume_checkpointdir": str(tmp_path),
            "save_checkpointdir": str(tmp_path),
        }},
        "schedulers": {"epochs": 10},
        "loggers": {"logdir": str(tmp_path)},
    })
    mgr = SimpleNamespace(
        load=lambda: ("ckpt", {"best_loss": None, "epoch": 0, "iter": 0})
    )
    with pytest.raises(ValueError, match="best_loss lineage"):
        resume_model_state(cfg, mgr)


def _fresh_run_cfg(tmp_path):
    return OmegaConf.create({
        "paths": {"outdir": str(tmp_path / "out"), "resume_checkpointdir": None},
        "loggers": {"logdir": str(tmp_path / "logs")},
        "evaluation": {"val_mode": "min"},
        "checkpoint": {"checkpoint_manager": {
            "save_checkpointdir": str(tmp_path / "ckpts"),
            "resume_checkpointdir": None,
            "checkpoint_tag": "best_model",
        }},
    })


def test_resume_run_refuses_fresh_run_over_existing_checkpoint_tag(tmp_path):
    """A fresh (non-resume) run whose save dir already holds the checkpoint tag
    raises instead of silently overwriting it on the first save."""
    cfg = _fresh_run_cfg(tmp_path)
    tag_dir = tmp_path / "ckpts" / "best_model"
    tag_dir.mkdir(parents=True)
    (tag_dir / "mp_rank_00_model_states.pt").write_bytes(b"x")
    trainer = SimpleNamespace(event_recorder=None, checkpoint_manager=None)
    with pytest.raises(RuntimeError, match="overwrite"):
        resume_run(trainer, cfg)


def test_resume_run_fresh_empty_dir_starts_at_zero(tmp_path):
    """A fresh run into empty directories starts at iteration 0 / epoch 0 with
    the mode-aware best sentinel and creates the log and checkpoint dirs."""
    cfg = _fresh_run_cfg(tmp_path)
    trainer = SimpleNamespace(event_recorder=None, checkpoint_manager=None)
    best, it, ep = resume_run(trainer, cfg)
    assert (it, ep) == (0, 0)
    assert best == math.inf
    assert (tmp_path / "logs").is_dir() and (tmp_path / "ckpts").is_dir()
