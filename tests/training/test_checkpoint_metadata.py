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
    metadata_path_for_tag,
    read_metadata_json,
    slugify,
    write_metadata_json,
)


def test_slugify_special_chars_and_long():
    assert slugify("Foo / Bar: Baz") == "foo-bar-baz"
    assert slugify("a" * 100) == "a" * 64


def test_build_run_tag_wandb_vs_iso():
    t = "2026-04-14T12:00:00+00:00"
    assert build_run_tag("4zhhl9i7", t) == "run_4zhhl9i7"
    assert build_run_tag(None, t).startswith("run_")


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


def test_read_metadata_json_missing(tmp_path):
    p = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError, match="Missing checkpoint metadata"):
        read_metadata_json(p)


def test_metadata_path_for_tag():
    assert metadata_path_for_tag("/tmp/ckpt", "best_model").name == CHECKPOINT_META_FILENAME
