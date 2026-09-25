"""Unit tests for checkpoint sidecar metadata helpers."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from cell_observatory_platform.training.checkpoint_metadata import (
    CHECKPOINT_META_FILENAME,
    build_metadata,
    build_model_name_slug,
    build_run_tag,
    default_metadata,
    metadata_path_for_tag,
    read_metadata_json,
    slugify,
    write_metadata_json,
)


def test_slugify_special_chars_and_long():
    assert slugify("Foo / Bar: Baz") == "foo-bar-baz"
    assert slugify("a" * 100) == "a" * 64


def test_build_run_tag_wandb_vs_iso():
    """The run tag is the W&B run id when present, else the slugified
    save timestamp."""
    t = "2026-04-14T12:00:00+00:00"
    assert build_run_tag("4zhhl9i7", t) == "run_4zhhl9i7"
    assert build_run_tag(None, t) == f"run_{slugify(t, max_len=48)}"
    assert build_run_tag(None, t) == "run_2026-04-14t12-00-00-00-00"


def test_build_model_name_slug():
    s = build_model_name_slug("Mask2Former", "run_abc", 3, 1000)
    assert "mask2former" in s
    assert "run_abc" in s
    assert "e3" in s
    assert "i1000" in s


class _Tiny(nn.Module):
    pass


def test_build_metadata_roundtrip_json(tmp_path):
    model = _Tiny()
    cfg = OmegaConf.create(
        {
            "experiment_name": "exp_test",
            "wandb_project": "wp",
            "x": 1,
        }
    )
    meta = build_metadata(
        model=model,
        cfg=cfg,
        epoch=1,
        iter_=2,
        best_loss=0.5,
        wandb_run_id="rid123",
        wandb_entity="ent",
    )
    assert meta["model_class_name"] == "_Tiny"
    assert meta["experiment_name"] == "exp_test"
    assert meta["wandb_project"] == "wp"
    assert meta["wandb_run_id"] == "rid123"
    assert meta["epoch"] == 1
    assert meta["iter"] == 2
    assert meta["schema_version"] == "1"
    assert "model_name_slug" in meta
    assert "hydra_config_hash" in meta

    p = tmp_path / CHECKPOINT_META_FILENAME
    write_metadata_json(p, meta)
    loaded = read_metadata_json(p)
    assert loaded["model_name_slug"] == meta["model_name_slug"]


def test_build_metadata_carries_trainer_state():
    """trainer_state (iteration, epoch, per-hook state) is persisted verbatim
    in the sidecar so resume can restore hook counters."""
    meta = build_metadata(
        model=torch.nn.Linear(2, 2), cfg={"a": 1}, epoch=1, iter_=10,
        best_loss=0.5, trainer_state={"iteration": 10, "epoch": 1,
                                      "hooks": {"EarlyStopHook": {"wait_count": 2}}},
    )
    assert meta["trainer_state"]["hooks"]["EarlyStopHook"]["wait_count"] == 2
    assert meta["trainer_state"]["iteration"] == 10


def test_read_metadata_json_missing(tmp_path):
    p = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError, match="Missing checkpoint metadata"):
        read_metadata_json(p)


def test_read_metadata_json_missing_allow_missing_warns_and_defaults(tmp_path, caplog):
    """allow_missing=True: a legacy checkpoint (no sidecar) loads with a loud
    warning and a synthesized default, instead of hard-failing."""
    import logging

    p = tmp_path / "nope.json"
    with caplog.at_level(logging.WARNING):
        meta = read_metadata_json(p, allow_missing=True)
    # Loud warning emitted.
    assert any("MISSING" in r.message or "missing" in r.message.lower()
               for r in caplog.records)
    # Synthesized default carries every key the consumers dereference
    # (training/helpers.py: best_loss/epoch/iter; training/loops.py: model_name_slug)
    # so a legacy resume cannot KeyError.
    assert meta["synthesized_default"] is True
    assert meta["epoch"] == 0
    assert meta["iter"] == 0
    # best_loss is None for the synthesized default; resume_model_state now
    # REFUSES on this (2026-07-27) rather than fabricating a baseline -- a
    # sidecar-less checkpoint has no best-metric lineage to resume from.
    assert meta["best_loss"] is None
    assert meta["model_name_slug"] == "legacy__unknown"


def test_default_metadata_has_full_schema():
    """default_metadata must cover the same keys real build_metadata emits, so
    no downstream consumer can KeyError on a legacy checkpoint."""
    d = default_metadata(reason="unit-test")
    for key in (
        "schema_version", "saved_at", "model_class_name", "model_name_slug",
        "experiment_name", "wandb_project", "wandb_entity", "wandb_run_id",
        "epoch", "iter", "best_loss", "hydra_config", "hydra_config_hash",
    ):
        assert key in d, key
    assert d["synthesized_default"] is True


def test_metadata_path_for_tag():
    assert metadata_path_for_tag("/tmp/ckpt", "best_model").name == CHECKPOINT_META_FILENAME


def test_channel_vocab_is_surfaced_at_top_level():
    """The frozen channel vocab travels in hydra_config; build_metadata also lifts it
    to a top-level key so warm starts can read it without walking the config."""
    table = {"localization": {"<unk>": 0, "membrane": 1}, "fluorophore": {"<unk>": 0}}
    cfg = OmegaConf.create({"datasets": {"preprocessor": {"channel_vocab": table}}})
    meta = build_metadata(nn.Linear(2, 2), cfg, epoch=0, iter_=0, best_loss=None)
    assert meta["channel_vocab"] == table
    meta_none = build_metadata(nn.Linear(2, 2), OmegaConf.create({}), epoch=0, iter_=0, best_loss=None)
    assert meta_none["channel_vocab"] is None


def test_explicit_channel_vocab_overrides_the_pristine_config():
    """The resolver injects the frozen vocab into the LIVE config after the
    trainer's pristine copy is taken, so the hook passes it explicitly."""
    import torch
    from omegaconf import OmegaConf
    from cell_observatory_platform.training.checkpoint_metadata import build_metadata
    cfg = OmegaConf.create({"experiment_name": "e", "wandb_project": "p",
                            "datasets": {"preprocessor": {"channel_vocab": None}}})
    vocab = {"localization": {"<unk>": 0, "membrane": 1}, "fluorophore": {"<unk>": 0},
             "table_size": {"localization": 17, "fluorophore": 16}}
    meta = build_metadata(model=torch.nn.Linear(2, 2), cfg=cfg, epoch=0, iter_=1, best_loss=None,
                          channel_vocab=OmegaConf.create(vocab))
    assert meta["channel_vocab"] == vocab
    assert meta["hydra_config"]["datasets"]["preprocessor"]["channel_vocab"] is None
    meta2 = build_metadata(model=torch.nn.Linear(2, 2), cfg=cfg, epoch=0, iter_=1, best_loss=None)
    assert meta2["channel_vocab"] is None
