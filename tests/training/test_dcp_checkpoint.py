"""DCPCheckpointManager save/load roundtrip and resume-probe tests.

Runs single-process on a gloo process group (the shared ``gloo_pg`` fixture
from tests/conftest.py; DCP works non-distributed too, but gloo matches the
production collective path most closely without GPUs).
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from cell_observatory_platform.training.checkpoint import DCPCheckpointManager
from cell_observatory_platform.training.helpers import _resume_dir_has_checkpoint
from cell_observatory_platform.training.optimizers import OptimizersContainer
from cell_observatory_platform.training.schedulers import build_lr_schedulers


class _TrainState:
    """Minimal Stateful standing in for the trainer."""

    def __init__(self):
        self.iteration, self.epoch, self.best = 0, 0, float("inf")

    def state_dict(self):
        return {"iteration": self.iteration, "epoch": self.epoch, "best_metric": self.best}

    def load_state_dict(self, sd):
        self.iteration, self.epoch, self.best = sd["iteration"], sd["epoch"], sd["best_metric"]


def _build(tmp_path, resume_dir=None, keep_latest_k=0, save_period=1):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    optimizers = OptimizersContainer(
        [model], torch.optim.AdamW,
        {"lr": 1e-3, "betas": (0.9, 0.99), "eps": 1e-8, "weight_decay": 0.01,
         "fused": False, "foreach": True},
    )
    schedulers = build_lr_schedulers(
        optimizers,
        {"schedule": "linear_warmup_stable_decay", "warmup": 1, "cooldown": 0,
         "decay_type": "cosine", "cos_min_ratio": 0.1},
        training_steps=100,
        steps_per_epoch=10,
    )
    train_state = _TrainState()
    manager = DCPCheckpointManager(
        model_parts=[model],
        optimizers=optimizers,
        lr_schedulers=schedulers,
        states={"train_state": train_state},
        save_checkpointdir=tmp_path / "checkpoints",
        save_period=save_period,
        keep_latest_k=keep_latest_k,
        resume_checkpointdir=resume_dir,
    )
    return model, optimizers, schedulers, train_state, manager


def _train_steps(model, optimizers, schedulers, n=3):
    for _ in range(n):
        optimizers.zero_grad()
        model(torch.randn(4, 8)).sum().backward()
        optimizers.step()
        schedulers.step()


class TestDCPCheckpointManager:
    def test_save_load_roundtrip_restores_all_states(self, gloo_pg, tmp_path):
        model, optimizers, schedulers, train_state, manager = _build(tmp_path)
        _train_steps(model, optimizers, schedulers)
        train_state.iteration, train_state.epoch, train_state.best = 3, 1, 0.5

        assert manager.save(curr_step=3, last_step=True)
        want_weights = {k: v.clone() for k, v in model.state_dict().items()}
        want_sched = schedulers.state_dict()

        # mutate everything, then restore from the checkpoint
        model2, optimizers2, schedulers2, train_state2, manager2 = _build(
            tmp_path, resume_dir=tmp_path / "checkpoints"
        )
        _train_steps(model2, optimizers2, schedulers2, n=5)

        loaded_step, meta = manager2.load()
        assert loaded_step == 3
        assert meta is not None  # checkpoint_meta.json sidecar written
        for k, v in model2.state_dict().items():
            assert torch.allclose(v, want_weights[k]), f"weight mismatch: {k}"
        assert schedulers2.state_dict() == want_sched
        assert (train_state2.iteration, train_state2.epoch, train_state2.best) == (3, 1, 0.5)

    def test_load_picks_latest_step_dir(self, gloo_pg, tmp_path):
        model, optimizers, schedulers, train_state, manager = _build(tmp_path)
        manager.save(curr_step=1)
        _train_steps(model, optimizers, schedulers)
        manager.save(curr_step=2)

        *_, manager2 = _build(tmp_path, resume_dir=tmp_path / "checkpoints")
        loaded_step, _ = manager2.load()
        assert loaded_step == 2

    def test_save_period_gates_and_last_step_forces(self, gloo_pg, tmp_path):
        *_, manager = _build(tmp_path, save_period=5)
        assert not manager.save(curr_step=3)
        assert manager.save(curr_step=5)
        assert not manager.save(curr_step=5)  # dedupe same step
        assert manager.save(curr_step=7, last_step=True)

    def test_keep_latest_k_purges_stale(self, gloo_pg, tmp_path):
        *_, manager = _build(tmp_path, keep_latest_k=2)
        for step in (1, 2, 3):
            manager.save(curr_step=step)
        kept = sorted(p.name for p in (tmp_path / "checkpoints").glob("step-*"))
        assert kept == ["step-2", "step-3"]

    def test_force_bypasses_period_but_not_dedup(self, gloo_pg, tmp_path):
        *_, manager = _build(tmp_path, save_period=5)
        assert manager.save(curr_step=3, force=True)
        assert not manager.save(curr_step=3, force=True)   # same step, still deduped
        assert not manager.save(curr_step=4)               # period cadence unchanged
        assert manager.save(curr_step=5)

    def test_step_zero_is_not_a_period_hit(self, gloo_pg, tmp_path):
        *_, manager = _build(tmp_path, save_period=5)
        assert not manager.save(curr_step=0)               # before_step probe at iter 0
        assert manager.save(curr_step=0, force=True)       # explicit force still works

    def test_resume_seeds_last_saved_step(self, gloo_pg, tmp_path):
        model, optimizers, schedulers, _, manager = _build(tmp_path, save_period=1)
        _train_steps(model, optimizers, schedulers)
        assert manager.save(curr_step=3)
        *_, resumed = _build(tmp_path, resume_dir=tmp_path / "checkpoints")
        step, _ = resumed.load()
        assert step == 3
        assert not resumed.save(curr_step=3)               # first before_step after resume
        assert resumed.save(curr_step=4)

    def test_both_resume_and_pretrained_dirs_raise(self, tmp_path):
        model = nn.Linear(4, 4)
        with pytest.raises(ValueError, match="Cannot specify both"):
            DCPCheckpointManager(
                model_parts=[model],
                save_checkpointdir=tmp_path,
                resume_checkpointdir=tmp_path,
                pretrained_checkpointdir=tmp_path,
            )

    def test_pretrained_dir_loads_model_only(self, gloo_pg, tmp_path):
        model, optimizers, schedulers, train_state, manager = _build(tmp_path)
        _train_steps(model, optimizers, schedulers)
        train_state.iteration = 3
        manager.save(curr_step=3, last_step=True)
        want_weights = {k: v.clone() for k, v in model.state_dict().items()}

        model2 = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        train_state2 = _TrainState()
        manager2 = DCPCheckpointManager(
            model_parts=[model2],
            states={"train_state": train_state2},
            save_checkpointdir=tmp_path / "fresh",
            pretrained_checkpointdir=tmp_path / "checkpoints",
        )
        loaded_step, _ = manager2.load()
        assert loaded_step == 3
        for k, v in model2.state_dict().items():
            assert torch.allclose(v, want_weights[k])
        assert train_state2.iteration == 0  # model-only: train state untouched


class TestResumeDirProbe:
    def _cfg(self, resume_dir):
        return OmegaConf.create(
            {
                "backend": "TORCHTITAN",
                "checkpoint": {"checkpoint_manager": {"resume_checkpointdir": str(resume_dir)}},
            }
        )

    def test_detects_dcp_layout(self, tmp_path):
        (tmp_path / "step-10").mkdir()
        (tmp_path / "step-10" / ".metadata").touch()
        assert _resume_dir_has_checkpoint(self._cfg(tmp_path))

    def test_incomplete_checkpoint_not_detected(self, tmp_path):
        (tmp_path / "step-10").mkdir()  # no .metadata: interrupted save
        assert not _resume_dir_has_checkpoint(self._cfg(tmp_path))

    def test_empty_dir_not_detected(self, tmp_path):
        assert not _resume_dir_has_checkpoint(self._cfg(tmp_path))
