"""Frozen channel-token vocabulary: derived once, ids never move, checkpoint wins."""
from __future__ import annotations

import json

import pyarrow as pa
import pytest
from omegaconf import OmegaConf

from cell_observatory_platform.data.channel_vocab import (
    UNK,
    VOCAB_FILENAME,
    ChannelVocab,
    channel_embed_enabled,
    fetch_channel_tokens,
    load_vocab_value,
    read_sidecar_vocab,
    resolve_channel_vocab,
)


class _FakeDb:
    """execute_arrow stub returning the DISTINCT (localization, fluorophore) rows."""

    def __init__(self, rows):
        self.rows = rows
        self.sql = ""

    def execute_arrow(self, sql):
        self.sql = sql
        return pa.table({
            "localization": pa.array([r[0] for r in self.rows]),
            "fluorophore": pa.array([r[1] for r in self.rows]),
        })


SYNTH = [
    ("membrane", "mstaygold"),
    ("cytosol", "Electra2"),
    ("cytosol", "mTFP1"),
    ("cytosol", "mCitrine"),
    ("cytosol", "mKOK"),
]


def _cfg(tmp_path, patch_embed_type="channel_adaptive", channel_vocab=None, channel_embed="factorized",
         resume=None, pretrained=None):
    return OmegaConf.create({
        "paths": {
            "outdir": str(tmp_path / "run"),
            "resume_checkpointdir": resume,
            "pretrained_checkpointdir": pretrained,
        },
        "datasets": {"preprocessor": {"name": "sam2_video", "channel_vocab": None}},
        "models": {"backbones": {"masked_encoder": {
            "patch_embed_type": patch_embed_type,
            "patch_embed_args": {"channel_embed": channel_embed, "channel_vocab": channel_vocab},
        }}},
    })


def _sidecar(dirpath, vocab, *, step=None, tag=None, iter_=0, nested=False):
    sub = dirpath / (f"step-{step}" if step is not None else tag)
    sub.mkdir(parents=True)
    meta = {"iter": iter_}
    if nested:
        meta["hydra_config"] = {"datasets": {"preprocessor": {"channel_vocab": vocab}}}
    else:
        meta["channel_vocab"] = vocab
    (sub / "checkpoint_meta.json").write_text(json.dumps(meta))


# --------------------------------------------------------------------------- #
# ChannelVocab
# --------------------------------------------------------------------------- #


def test_empty_vocab_has_unk_at_zero():
    v = ChannelVocab.empty()
    assert v.to_dict() == {"localization": {UNK: 0}, "fluorophore": {UNK: 0}}
    assert v.lookup("localization", "membrane") is None
    assert v.lookup("localization", None) is None


def test_extend_appends_sorted_and_never_moves_existing_ids():
    v, added = ChannelVocab.empty().extend({"localization": ["membrane", "cytosol"], "fluorophore": ["mTFP1", "Electra2"]})
    assert v.to_dict()["localization"] == {UNK: 0, "cytosol": 1, "membrane": 2}
    assert v.to_dict()["fluorophore"] == {UNK: 0, "electra2": 1, "mtfp1": 2}
    assert added == {"localization": ["cytosol", "membrane"], "fluorophore": ["electra2", "mtfp1"]}

    v2, added2 = v.extend({"localization": ["golgi", "membrane"], "fluorophore": []})
    assert v2.to_dict()["localization"] == {UNK: 0, "cytosol": 1, "membrane": 2, "golgi": 3}
    assert added2 == {"localization": ["golgi"], "fluorophore": []}
    assert v.to_dict()["localization"] == {UNK: 0, "cytosol": 1, "membrane": 2}   # immutable


def test_extend_ignores_null_and_blank_tokens():
    v, added = ChannelVocab.empty().extend({"localization": [None, "  ", "Membrane"], "fluorophore": [None]})
    assert v.to_dict()["localization"] == {UNK: 0, "membrane": 1}
    assert added["fluorophore"] == []


def test_lookup_normalizes_case_and_whitespace():
    v, _ = ChannelVocab.empty().extend({"localization": ["membrane"], "fluorophore": ["mTFP1"]})
    assert v.lookup("fluorophore", " MTFP1 ") == 1
    assert v.lookup("localization", "Membrane") == 1


@pytest.mark.parametrize("table,match", [
    ({"localization": {"membrane": 0}, "fluorophore": {UNK: 0}}, "must be id 0"),
    ({"localization": {UNK: 0, "membrane": 2}, "fluorophore": {UNK: 0}}, "contiguous"),
    ({"localization": {UNK: 0, "Membrane": 1}, "fluorophore": {UNK: 0}}, "not normalized"),
    ({"localization": {UNK: 0}}, "missing table"),
])
def test_invalid_tables_are_rejected(table, match):
    with pytest.raises(ValueError, match=match):
        ChannelVocab.from_dict(table)


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #


def test_fetch_reads_the_catalog_ordered_and_normalized():
    db = _FakeDb(SYNTH)
    toks = fetch_channel_tokens(db)
    assert "api.roi_channels" in db.sql and "channel_type = 'data'" in db.sql and "ORDER BY" in db.sql
    assert toks == {
        "localization": ["cytosol", "membrane"],
        "fluorophore": ["electra2", "mcitrine", "mkok", "mstaygold", "mtfp1"],
    }


def test_load_vocab_value_accepts_none_table_or_file(tmp_path):
    assert load_vocab_value(None) is None
    table = ChannelVocab.empty().to_dict()
    assert load_vocab_value(table) == table
    f = tmp_path / VOCAB_FILENAME
    f.write_text(json.dumps(table))
    assert load_vocab_value(str(f)) == table
    with pytest.raises(FileNotFoundError):
        load_vocab_value(str(tmp_path / "missing.json"))


def test_sidecar_newest_step_wins_and_nested_hydra_config_is_read(tmp_path):
    old = ChannelVocab.empty().to_dict()
    new, _ = ChannelVocab.empty().extend({"localization": ["membrane"], "fluorophore": []})
    _sidecar(tmp_path, old, step=8)
    _sidecar(tmp_path, new.to_dict(), step=16, nested=True)
    assert read_sidecar_vocab(tmp_path) == new.to_dict()


def test_sidecar_deepspeed_layout_uses_iter(tmp_path):
    a, _ = ChannelVocab.empty().extend({"localization": ["a"], "fluorophore": []})
    b, _ = ChannelVocab.empty().extend({"localization": ["b"], "fluorophore": []})
    _sidecar(tmp_path, a.to_dict(), tag="latest_model", iter_=100)
    _sidecar(tmp_path, b.to_dict(), tag="best_model", iter_=40)
    assert read_sidecar_vocab(tmp_path) == a.to_dict()


def test_sidecar_missing_dir_or_vocab_is_none(tmp_path):
    assert read_sidecar_vocab(None) is None
    assert read_sidecar_vocab(tmp_path / "nope") is None
    (tmp_path / "step-1").mkdir()
    (tmp_path / "step-1" / "checkpoint_meta.json").write_text(json.dumps({"iter": 1}))
    assert read_sidecar_vocab(tmp_path) is None


# --------------------------------------------------------------------------- #
# resolve_channel_vocab
# --------------------------------------------------------------------------- #


def test_disabled_for_joint_embed_or_no_encoder_node(tmp_path):
    assert not channel_embed_enabled(_cfg(tmp_path, patch_embed_type="joint"))
    assert not channel_embed_enabled(_cfg(tmp_path, channel_embed="none"))
    assert not channel_embed_enabled(OmegaConf.create({"models": {"backbones": {}}}))
    cfg = _cfg(tmp_path, patch_embed_type="joint")
    assert resolve_channel_vocab(cfg, _FakeDb(SYNTH)) is None
    assert cfg.datasets.preprocessor.channel_vocab is None


def test_derives_from_db_writes_file_and_injects_both_nodes(tmp_path):
    cfg = _cfg(tmp_path)
    vocab = resolve_channel_vocab(cfg, _FakeDb(SYNTH))
    table = vocab.to_dict()
    assert table["localization"] == {UNK: 0, "cytosol": 1, "membrane": 2}
    assert table["fluorophore"]["mstaygold"] == 4
    assert table["table_size"] == {"localization": 3 + 16, "fluorophore": 6 + 16}   # default extra slots
    assert OmegaConf.to_container(cfg.datasets.preprocessor.channel_vocab) == table
    assert OmegaConf.to_container(cfg.models.backbones.masked_encoder.patch_embed_args.channel_vocab) == table
    written = json.loads((tmp_path / "run" / VOCAB_FILENAME).read_text())
    assert written == table


def test_config_pin_is_the_base_and_db_only_appends(tmp_path):
    pinned, _ = ChannelVocab.empty().extend({"localization": ["membrane"], "fluorophore": ["mstaygold"]})
    cfg = _cfg(tmp_path, channel_vocab=pinned.to_dict())
    vocab = resolve_channel_vocab(cfg, _FakeDb(SYNTH), write=False)
    t = vocab.to_dict()
    assert t["localization"] == {UNK: 0, "membrane": 1, "cytosol": 2}       # pinned id kept, cytosol appended
    assert t["fluorophore"][ "mstaygold"] == 1 and t["fluorophore"]["electra2"] == 2


def test_config_pin_may_be_a_file(tmp_path):
    pinned, _ = ChannelVocab.empty().extend({"localization": ["membrane"], "fluorophore": []})
    f = tmp_path / VOCAB_FILENAME
    f.write_text(json.dumps(pinned.to_dict()))
    cfg = _cfg(tmp_path, channel_vocab=str(f))
    vocab = resolve_channel_vocab(cfg, None, write=False)
    assert vocab.tables == pinned.tables and vocab.table_size == {"localization": 2 + 16, "fluorophore": 1 + 16}


def test_checkpoint_beats_config_and_db_appends_on_top(tmp_path):
    ckpt, _ = ChannelVocab.empty().extend({"localization": ["golgi", "membrane"], "fluorophore": ["gfp"]})
    ckdir = tmp_path / "ckpt"
    _sidecar(ckdir, ckpt.to_dict(), step=32)
    pinned, _ = ChannelVocab.empty().extend({"localization": ["membrane"], "fluorophore": []})
    cfg = _cfg(tmp_path, channel_vocab=pinned.to_dict(), pretrained=str(ckdir))
    vocab = resolve_channel_vocab(cfg, _FakeDb(SYNTH), write=False)
    t = vocab.to_dict()
    assert t["localization"] == {UNK: 0, "golgi": 1, "membrane": 2, "cytosol": 3}
    assert t["fluorophore"][UNK] == 0 and t["fluorophore"]["gfp"] == 1


def test_resume_dir_beats_pretrained_dir(tmp_path):
    a, _ = ChannelVocab.empty().extend({"localization": ["a"], "fluorophore": []})
    b, _ = ChannelVocab.empty().extend({"localization": ["b"], "fluorophore": []})
    _sidecar(tmp_path / "resume", a.to_dict(), step=1)
    _sidecar(tmp_path / "pre", b.to_dict(), step=1)
    cfg = _cfg(tmp_path, resume=str(tmp_path / "resume"), pretrained=str(tmp_path / "pre"))
    assert resolve_channel_vocab(cfg, None, write=False).tables == a.tables


def test_no_db_and_no_base_gives_unk_only_tables(tmp_path):
    cfg = _cfg(tmp_path)
    vocab = resolve_channel_vocab(cfg, None, write=False)
    assert vocab.tables == ChannelVocab.empty().tables


# --------------------------------------------------------------------------- #
# lifecycle guarantees
# --------------------------------------------------------------------------- #


def test_table_size_is_frozen_with_the_checkpoint_and_survives_appends(tmp_path):
    """A resumed run may append tokens, but the capacity (= checkpointed weight
    shape) never changes, whatever vocab_extra_slots says now."""
    first = resolve_channel_vocab(_cfg(tmp_path), _FakeDb(SYNTH[:1]), write=False)   # 2 loc, 2 fl tokens
    assert first.table_size == {"localization": 2 + 16, "fluorophore": 2 + 16}
    ckdir = tmp_path / "ckpt"
    _sidecar(ckdir, first.to_dict(), step=10)
    cfg = _cfg(tmp_path, resume=str(ckdir))
    cfg.models.backbones.masked_encoder.patch_embed_args["vocab_extra_slots"] = 99
    resumed = resolve_channel_vocab(cfg, _FakeDb(SYNTH), write=False)
    assert resumed.table_size == first.table_size
    assert resumed.tables["localization"] == {UNK: 0, "membrane": 1, "cytosol": 2}
    assert resumed.tables["fluorophore"][ "mstaygold"] == 1 and resumed.size("fluorophore") == 6


def test_appending_past_the_frozen_capacity_is_an_error():
    v = ChannelVocab({UNK: 0, "membrane": 1}, {UNK: 0}, table_size={"localization": 2, "fluorophore": 1})
    with pytest.raises(ValueError, match="overflow"):
        v.extend({"localization": ["golgi"], "fluorophore": []})
    with pytest.raises(ValueError, match="exceed table_size"):
        ChannelVocab({UNK: 0, "membrane": 1}, {UNK: 0}, table_size={"localization": 1, "fluorophore": 1})


def test_inference_never_grows_the_vocab(tmp_path):
    """job_type != train: the checkpoint table is used as is, so a token the model
    never saw maps to <unk> in the preprocessor instead of an untrained row."""
    trained, _ = ChannelVocab.empty().extend({"localization": ["membrane"], "fluorophore": ["mstaygold"]})
    trained = trained.frozen(4)
    ckdir = tmp_path / "ckpt"
    _sidecar(ckdir, trained.to_dict(), step=5)
    cfg = _cfg(tmp_path, pretrained=str(ckdir))
    cfg.job_type = "test"
    vocab = resolve_channel_vocab(cfg, _FakeDb(SYNTH), write=False)
    assert vocab == trained
    assert vocab.lookup("localization", "cytosol") is None


def test_checkpoint_without_sidecar_vocab_refuses_to_rederive(tmp_path):
    ckdir = tmp_path / "ckpt"
    (ckdir / "step-3").mkdir(parents=True)
    (ckdir / "step-3" / "checkpoint_meta.json").write_text(json.dumps({"iter": 3}))
    with pytest.raises(ValueError, match="no checkpoint_meta.json there records a channel_vocab"):
        resolve_channel_vocab(_cfg(tmp_path, pretrained=str(ckdir)), _FakeDb(SYNTH), write=False)
    # a config pin is an acceptable substitute
    pinned = ChannelVocab.empty().frozen(2)
    assert resolve_channel_vocab(_cfg(tmp_path, pretrained=str(ckdir), channel_vocab=pinned.to_dict()),
                                 None, write=False) == pinned
