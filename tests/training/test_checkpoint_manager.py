"""CheckpointManager.load_for_eval: eval-time trainers (TestTrainer /
Inferencer) load weights only via pretrained_checkpointdir; a resume dir or no
dir at all is refused loudly. CPU-only, duck-typed manager."""

from types import SimpleNamespace

import pytest

from cell_observatory_platform.training.checkpoint import CheckpointManager


def _mgr(pretrained=None, resume=None, load=None):
    return SimpleNamespace(
        pretrained_checkpointdir=pretrained,
        resume_checkpointdir=resume,
        load=load or (lambda: ("path", {})),
    )


def test_load_for_eval_uses_pretrained_dir():
    """With pretrained_checkpointdir set, load_for_eval loads and returns the
    checkpoint sidecar metadata."""
    mgr = _mgr(pretrained="/ckpt", load=lambda: ("path", {"epoch": 3}))
    assert CheckpointManager.load_for_eval(mgr, "test") == {"epoch": 3}


def test_load_for_eval_rejects_resume_dir():
    """A resume_checkpointdir (DeepSpeed engine layout) cannot be loaded into
    the raw module at eval time; the error points at pretrained_checkpointdir."""
    with pytest.raises(ValueError, match="pretrained_checkpointdir"):
        CheckpointManager.load_for_eval(_mgr(resume="/resume"), "test")


def test_load_for_eval_refuses_random_init():
    """With neither directory configured, eval refuses to run on randomly
    initialized weights."""
    with pytest.raises(ValueError, match="randomly initialized"):
        CheckpointManager.load_for_eval(_mgr(), "predict")
