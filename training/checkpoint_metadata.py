"""
Canonical checkpoint sidecar metadata (checkpoint_meta.json) + model_name_slug for inference.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

CHECKPOINT_META_FILENAME = "checkpoint_meta.json"
SCHEMA_VERSION = "1"


def slugify(s: str, max_len: int = 64) -> str:
    if not s:
        return "unknown"
    s = s.lower()
    s = re.sub(r"[/\\:.]+", "-", s)
    s = re.sub(r"[^a-z0-9_-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        s = "unknown"
    return s[:max_len].rstrip("-")


def build_run_tag(wandb_run_id: Optional[str], saved_at_iso: str) -> str:
    if wandb_run_id:
        return f"run_{slugify(wandb_run_id, max_len=32)}"
    return f"run_{slugify(saved_at_iso, max_len=48)}"


def build_model_name_slug(model_class_name: str, run_tag: str, epoch: int, iter_: int) -> str:
    return f"{slugify(model_class_name)}__{run_tag}__e{int(epoch)}_i{int(iter_)}"


def unwrap_model_for_class_name(model: torch.nn.Module) -> str:
    """Resolve real nn.Module class (DeepSpeed wraps .module; torch.compile uses _orig_mod)."""
    m: torch.nn.Module = model
    m = getattr(m, "module", m)
    m = getattr(m, "_orig_mod", m)
    return m.__class__.__name__


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    return str(obj)


def _hydra_config_dict(cfg: Union[DictConfig, Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(cfg, DictConfig):
        raw = OmegaConf.to_container(cfg, resolve=True)
    else:
        raw = dict(cfg)
    return _sanitize_for_json(raw)


def hydra_config_hash(hydra_config: Dict[str, Any]) -> str:
    canonical = json.dumps(hydra_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_metadata(
    model: torch.nn.Module,
    cfg: Union[DictConfig, Mapping[str, Any]],
    epoch: Optional[int],
    iter_: Optional[int],
    best_loss: Optional[float],
    wandb_run_id: Optional[str] = None,
    wandb_entity: Optional[str] = None,
) -> Dict[str, Any]:
    saved_at = datetime.now(timezone.utc).isoformat()
    model_class_name = unwrap_model_for_class_name(model)
    run_tag = build_run_tag(wandb_run_id, saved_at)
    model_name_slug = build_model_name_slug(model_class_name, run_tag, epoch or 0, iter_ or 0)

    hydra_config = _hydra_config_dict(cfg)
    hsh = hydra_config_hash(hydra_config)

    experiment_name = ""
    wandb_project = ""
    if isinstance(cfg, DictConfig):
        experiment_name = str(OmegaConf.select(cfg, "experiment_name") or "")
        wandb_project = str(OmegaConf.select(cfg, "wandb_project") or "")
    else:
        experiment_name = str(cfg.get("experiment_name", "") or "")
        wandb_project = str(cfg.get("wandb_project", "") or "")

    return {
        "schema_version": SCHEMA_VERSION,
        "saved_at": saved_at,
        "model_class_name": model_class_name,
        "model_name_slug": model_name_slug,
        "experiment_name": experiment_name,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_run_id": wandb_run_id,
        "epoch": epoch,
        "iter": iter_,
        "best_loss": best_loss,
        "hydra_config": hydra_config,
        "hydra_config_hash": hsh,
    }


def write_metadata_json(path: Union[str, Path], metadata: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = json.dumps(metadata, indent=2, default=str)
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def read_metadata_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint metadata: {path}. "
            "Expected checkpoint_meta.json next to checkpoint (same tag directory). "
            "Old checkpoints need metadata added or re-saved with new trainer."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def metadata_path_for_tag(checkpoint_root: Union[str, Path], tag: str) -> Path:
    return Path(checkpoint_root) / tag / CHECKPOINT_META_FILENAME
