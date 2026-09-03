"""Frozen channel-token vocabulary for the channel-adaptive patch embedding.

A data channel is described by two normalized text tokens, its ``localization``
(the biology: membrane, cytosol, ...) and its ``fluorophore`` (the dye). The
channel embedding looks each up in its own table, so every token needs a
stable integer id for the life of a model. The DB does not provide one so
the id table is DERIVED once and then FROZEN:

* the id of a token never changes once assigned; new tokens are appended;
* row 0 of each table is ``<unk>``;
* the resolved table is written to ``<outdir>/channel_vocab.json`` and travels
  in every checkpoint sidecar (``checkpoint_meta.json``), so a resumed or
  warm-started run inherits it.

Resolution order in :func:`resolve_channel_vocab`: checkpoint sidecar, then the
config value (an inline table or a path to a written ``channel_vocab.json``),
then the DB catalog (``SELECT DISTINCT localization, fluorophore ... ORDER BY``)
for tokens the base does not know yet.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from omegaconf import DictConfig, OmegaConf, open_dict

from cell_observatory_platform.data.datasets.utils import _normalize_channel_token

logger = logging.getLogger(__name__)

UNK = "<unk>"
KINDS = ("localization", "fluorophore")
VOCAB_FILENAME = "channel_vocab.json"
CHANNEL_CATALOG = "api.roi_channels"
CHECKPOINT_META_FILENAME = "checkpoint_meta.json"


def _validate_table(kind: str, table: Mapping[str, int]) -> dict[str, int]:
    out = {str(k): int(v) for k, v in table.items()}
    if out.get(UNK) != 0:
        raise ValueError(f"channel vocab {kind!r}: {UNK!r} must be id 0, got {out.get(UNK)!r}")
    ids = sorted(out.values())
    if ids != list(range(len(out))):
        raise ValueError(f"channel vocab {kind!r}: ids must be contiguous 0..n-1, got {ids}")
    for tok in out:
        if tok != UNK and _normalize_channel_token(tok) != tok:
            raise ValueError(f"channel vocab {kind!r}: token {tok!r} is not normalized")
    return out


class ChannelVocab:
    """Two append-only ``token -> id`` tables, one per token kind.

    ``table_size`` is the embedding capacity per kind: fixed when the vocab is
    first frozen (tokens + ``vocab_extra_slots``) and never changed afterwards,
    because it IS the shape of the checkpointed embedding weight. Appending past
    it is an error (the weights would no longer load), not a resize.
    """

    def __init__(
        self,
        localization: Mapping[str, int],
        fluorophore: Mapping[str, int],
        table_size: Optional[Mapping[str, int]] = None,
    ):
        self.tables: dict[str, dict[str, int]] = {
            "localization": _validate_table("localization", localization),
            "fluorophore": _validate_table("fluorophore", fluorophore),
        }
        self.table_size: Optional[dict[str, int]] = None
        if table_size is not None:
            self.table_size = {kind: int(table_size[kind]) for kind in KINDS}
            for kind in KINDS:
                if self.table_size[kind] < len(self.tables[kind]):
                    raise ValueError(
                        f"channel vocab {kind!r}: {len(self.tables[kind])} tokens exceed "
                        f"table_size {self.table_size[kind]}"
                    )

    @classmethod
    def empty(cls) -> "ChannelVocab":
        return cls({UNK: 0}, {UNK: 0})

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ChannelVocab":
        missing = [k for k in KINDS if k not in d]
        if missing:
            raise ValueError(f"channel vocab is missing table(s) {missing}; got keys {sorted(d)}")
        return cls(d["localization"], d["fluorophore"], d.get("table_size"))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {kind: dict(self.tables[kind]) for kind in KINDS}
        if self.table_size is not None:
            out["table_size"] = dict(self.table_size)
        return out

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ChannelVocab) and self.to_dict() == other.to_dict()

    def size(self, kind: str) -> int:
        return len(self.tables[kind])

    def capacity(self, kind: str) -> int:
        """Embedding rows for ``kind``: the frozen table size, else the token count."""
        if self.table_size is not None:
            return self.table_size[kind]
        return len(self.tables[kind])

    def frozen(self, extra_slots: int) -> "ChannelVocab":
        """Fix the capacity at tokens + ``extra_slots`` (no-op if already frozen)."""
        if self.table_size is not None:
            return self
        return ChannelVocab(
            self.tables["localization"], self.tables["fluorophore"],
            {kind: len(self.tables[kind]) + int(extra_slots) for kind in KINDS},
        )

    def lookup(self, kind: str, token: Optional[str]) -> Optional[int]:
        """Id for a token, or ``None`` when unknown (a NULL column is unknown)."""
        if token is None:
            return None
        return self.tables[kind].get(_normalize_channel_token(token))

    def extend(self, tokens: Mapping[str, Iterable[Optional[str]]]) -> tuple["ChannelVocab", dict[str, list[str]]]:
        """Append unseen tokens (sorted) and report what was added; ids never move."""
        new_tables = self.to_dict()
        added: dict[str, list[str]] = {kind: [] for kind in KINDS}
        for kind in KINDS:
            table = new_tables[kind]
            fresh = sorted(
                {_normalize_channel_token(t) for t in tokens.get(kind, ()) if t is not None}
                - set(table)
                - {""}
            )
            for tok in fresh:
                table[tok] = len(table)
            added[kind] = fresh
            if self.table_size is not None and len(table) > self.table_size[kind]:
                raise ValueError(
                    f"channel vocab {kind!r} overflow: {len(table)} tokens (new: {fresh}) but the "
                    f"frozen table_size is {self.table_size[kind]} -- the checkpointed embedding has "
                    "no free row. Train a new model with a larger vocab_extra_slots, or map the new "
                    "token to <unk> (unknown_policy: unk)."
                )
        return ChannelVocab(new_tables["localization"], new_tables["fluorophore"], self.table_size), added


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #


def fetch_channel_tokens(db_client) -> dict[str, list[str]]:
    """Every (localization, fluorophore) on a DATA channel, normalized and sorted.

    ``api.roi_channels`` is one row per channel, 
    so the DISTINCT is instant; the training views carry the same
    tokens as ``channel_tags`` but are millions of rows.
    """
    table = db_client.execute_arrow(
        f"""
        SELECT DISTINCT localization, fluorophore
        FROM {CHANNEL_CATALOG}
        WHERE channel_type = 'data'
        ORDER BY fluorophore, localization
        """
    )
    out: dict[str, list[str]] = {}
    for kind in KINDS:
        if kind not in table.column_names:
            out[kind] = []
            continue
        toks = {
            _normalize_channel_token(v) for v in table[kind].to_pylist() if v is not None
        } - {""}
        out[kind] = sorted(toks)
    return out


def _vocab_from_sidecar(meta: Mapping[str, Any]) -> Optional[dict]:
    direct = meta.get("channel_vocab")
    if direct:
        return dict(direct)
    hydra = meta.get("hydra_config") or {}
    try:
        nested = hydra["datasets"]["preprocessor"]["channel_vocab"]
    except (KeyError, TypeError):
        return None
    return dict(nested) if nested else None


def read_sidecar_vocab(checkpoint_dir: Optional[str | Path]) -> Optional[dict]:
    """The vocab recorded in the newest ``checkpoint_meta.json`` under a dir.

    Handles both layouts: DCP (``<dir>/step-N/``, newest step wins) and
    DeepSpeed (``<dir>/<tag>/``, highest recorded ``iter`` wins). ``None`` when
    the dir is unset, empty, or its sidecars carry no vocab.
    """
    if not checkpoint_dir:
        return None
    root = Path(checkpoint_dir)
    if not root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for meta_path in root.glob(f"*/{CHECKPOINT_META_FILENAME}"):
        parent = meta_path.parent.name
        rank: Optional[int] = None
        if parent.startswith("step-"):
            try:
                rank = int(parent.split("-", 1)[1])
            except ValueError:
                rank = None
        if rank is None:
            try:
                rank = int(json.loads(meta_path.read_text(encoding="utf-8")).get("iter") or 0)
            except (OSError, ValueError, TypeError):
                rank = 0
        candidates.append((rank, meta_path))
    for _, meta_path in sorted(candidates, key=lambda t: t[0], reverse=True):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        vocab = _vocab_from_sidecar(meta)
        if vocab:
            return vocab
    return None


def load_vocab_value(value: Any) -> Optional[dict]:
    """A config value for ``channel_vocab``: ``None``, an inline table, or a
    path to a ``channel_vocab.json`` written by an earlier run."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"channel_vocab file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"channel_vocab must be null, a table or a path; got {type(value).__name__}")


def write_vocab_file(outdir: str | Path, vocab: ChannelVocab) -> Path:
    path = Path(outdir) / VOCAB_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(vocab.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------- #
# config glue
# --------------------------------------------------------------------------- #

ENCODER_NODE = "models.backbones.masked_encoder"


def channel_embed_enabled(cfg: DictConfig) -> bool:
    """True when the encoder is channel-adaptive with a token embedding."""
    node = OmegaConf.select(cfg, ENCODER_NODE)
    if node is None or node.get("patch_embed_type", "joint") != "channel_adaptive":
        return False
    args = node.get("patch_embed_args") or {}
    return str(args.get("channel_embed", "factorized")) != "none"


def resolve_channel_vocab(
    cfg: DictConfig,
    db_client=None,
    *,
    write: bool = True,
) -> Optional[ChannelVocab]:
    """Resolve, freeze and inject the vocab; ``None`` when the model has none.

    Precedence: checkpoint sidecar (resume, then pretrained) > config value >
    empty. The DB catalog then only APPENDS tokens the base does not know.
    The resolved table is written into ``cfg`` under both the encoder's
    ``patch_embed_args.channel_vocab`` and ``datasets.preprocessor.channel_vocab``
    so the model sizes its tables and the preprocessor maps tokens from the
    same object, and it lands in the checkpoint sidecar through ``hydra_config``.
    """
    if not channel_embed_enabled(cfg):
        return None
    enc = OmegaConf.select(cfg, ENCODER_NODE)
    args = enc.get("patch_embed_args") or {}
    extra_slots = int(args.get("vocab_extra_slots", 16))
    training = str(cfg.get("job_type", "train")) == "train"

    paths = cfg.get("paths") or {}
    ckpt_dirs = [d for d in (paths.get("resume_checkpointdir"), paths.get("pretrained_checkpointdir")) if d]
    ckpt = None
    for d in ckpt_dirs:
        ckpt = read_sidecar_vocab(d)
        if ckpt is not None:
            break
    pinned = load_vocab_value(args.get("channel_vocab"))
    if ckpt is None and ckpt_dirs and pinned is None:
        raise ValueError(
            f"[channel_vocab] loading a channel-adaptive model from {ckpt_dirs} but no "
            "checkpoint_meta.json there records a channel_vocab, and the config pins none. "
            "Deriving a fresh vocab would silently mis-map channel ids onto the trained "
            "embeddings; pin the run's channel_vocab.json via patch_embed_args.channel_vocab."
        )
    if ckpt is not None and pinned is not None and ckpt != pinned:
        logger.warning(
            "[channel_vocab] config pins a vocab that differs from the checkpoint's; "
            "the checkpoint wins (ids are frozen with the weights). config=%s checkpoint=%s",
            pinned, ckpt,
        )
    base = ckpt if ckpt is not None else pinned
    vocab = ChannelVocab.from_dict(base) if base else ChannelVocab.empty()

    source = "checkpoint" if ckpt is not None else ("config" if pinned is not None else "empty")
    if db_client is not None and training:
        # only a training run may grow the vocab: a new token gets a fresh row that
        # training will fit. At inference an unseen token must map to <unk>, never
        # to an untrained row, so the checkpoint's table is used as is.
        vocab, added = vocab.extend(fetch_channel_tokens(db_client))
        if any(added.values()):
            logger.info("[channel_vocab] appended tokens from %s: %s", CHANNEL_CATALOG, added)
    vocab = vocab.frozen(extra_slots)
    logger.info(
        "[channel_vocab] base=%s localization=%d/%d fluorophore=%d/%d (tokens/table_size)",
        source, vocab.size("localization"), vocab.capacity("localization"),
        vocab.size("fluorophore"), vocab.capacity("fluorophore"),
    )

    outdir = paths.get("outdir")
    if write and outdir:
        path = write_vocab_file(outdir, vocab)
        logger.info("[channel_vocab] written to %s (pin it via patch_embed_args.channel_vocab)", path)

    table = vocab.to_dict()
    with open_dict(cfg):
        if enc.get("patch_embed_args") is None:
            enc["patch_embed_args"] = {}
        enc.patch_embed_args["channel_vocab"] = table
        cfg.datasets.preprocessor["channel_vocab"] = table
    return vocab
