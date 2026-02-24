"""
Generate Hydra/OmegaConf YAML configs from:
- a base config YAML (template)
- a JSON spec (sweeps + overrides)
- optionally a pretrain config YAML (to auto-grab pretrained experiment_name)

Usage:
  python generate_configs.py --spec .../my_folder/spec.json
"""

from __future__ import annotations

import re
import copy
import json
import argparse
import itertools
from pathlib import Path
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Tuple

import yaml


# -----------------------------
# Built-in presets
# -----------------------------

# Method presets: what differs between JEPA vs MAE regardless of task.
# NOTE: translate target varies by *model family*, so we leave it as a placeholder.
METHOD_PRESETS = {
    "jepa": {
        "checkpoint_manager_template": {
            "ckpt_include_prefixes": ["__BACKBONE_TARGET__"],
            # fill __BACKBONE_TARGET__ based on TASK_METHOD_DEFAULTS / model family
            "ckpt_translate_map": {"input_encoder": "__BACKBONE_TARGET__"},
        },
    },
    "mae": {
        "checkpoint_manager_template": {
            "ckpt_include_prefixes": ["__BACKBONE_TARGET__"],
            "ckpt_translate_map": {"masked_encoder": "__BACKBONE_TARGET__"},
        },
    },
}


# Task+method presets: to inject for each (task folder, method).
#
# Each entry provides:
#  - defaults_before_checkpoint: injected before "/checkpoint/checkpoint"
#  - backbone_target: what replaces "__BACKBONE_TARGET__" in METHOD_PRESETS
TASK_METHOD_DEFAULTS = {
    # -----------------
    # AutoBench: channel_split
    # -----------------
    ("channel_split", "jepa"): {
        "defaults_before_checkpoint": [
            "/models/BUILD_AutoBench",
            "/models/meta_arch/autobench/ChannelSplitAutoBench/jepa_large",
            "/models/backbones/masked_encoder/large",
            "/models/heads/maskedpredictor/large",
        ],
        "defaults_before_optimizations": [
            "/optimizations/models/autobench/ChannelSplitAutoBench/mae_large",
        ],
        "backbone_target": "backbone",
    },
    ("channel_split", "mae"): {
        "defaults_before_checkpoint": [
            "/models/BUILD_AutoBench",
            "/models/meta_arch/autobench/ChannelSplitAutoBench/mae_large",
            "/models/backbones/masked_encoder/large",
            "/models/heads/maskedpredictor/large",
        ],
        "defaults_before_optimizations": [
            "/optimizations/models/autobench/ChannelSplitAutoBench/mae_large",
        ],
        "backbone_target": "backbone",
    },

    # -----------------
    # AutoBench: upsample_space
    # -----------------
    ("upsample_space", "jepa"): {
        "defaults_before_checkpoint": [
            "/models/BUILD_AutoBench",
            "/models/meta_arch/autobench/UpsampleSpaceAutoBench/jepa_large",
            "/models/backbones/masked_encoder/large",
            "/models/heads/maskedpredictor/large",
        ],
        "defaults_before_optimizations": [
            "/optimizations/models/autobench/UpsampleSpaceAutoBench/mae_large",
        ],
        "backbone_target": "backbone",
    },
    ("upsample_space", "mae"): {
        "defaults_before_checkpoint": [
            "/models/BUILD_AutoBench",
            "/models/meta_arch/autobench/UpsampleSpaceAutoBench/mae_large",
            "/models/backbones/masked_encoder/large",
            "/models/heads/maskedpredictor/large",
        ],
        "defaults_before_optimizations": [
            "/optimizations/models/autobench/UpsampleSpaceAutoBench/mae_large",
        ],
        "backbone_target": "backbone",
    },

    # -----------------
    # Finetune: instance segmentation (PlainDETR)
    # -----------------
    ("instance_seg", "jepa"): {
        "defaults_before_checkpoint": [
            "/models/BUILD_plainDETR",
            "/models/meta_arch/plainDETR/mae_large",
            "/models/backbones/masked_encoder/large",
            "/models/backbones/plain_detr_backbone/plain_detr_backbone",
            "/models/heads/plain_detr_transformer/plain_detr_transformer",
        ],
        "defaults_before_optimizations": [
            "/optimizations/models/plainDETR/mae_large",
        ],
        "backbone_target": "backbone.backbone",
    },
    ("instance_seg", "mae"): {
        "defaults_before_checkpoint": [
            "/models/BUILD_plainDETR",
            "/models/meta_arch/plainDETR/mae_large",
            "/models/backbones/masked_encoder/large",
            "/models/backbones/plain_detr_backbone/plain_detr_backbone",
            "/models/heads/plain_detr_transformer/plain_detr_transformer",
        ],
        "defaults_before_optimizations": [
            "/optimizations/models/plainDETR/mae_large",
        ],
        "backbone_target": "backbone.backbone",
    },

    # -----------------
    # Finetune: instance segmentation (MaskDINO)
    # -----------------
    ("instance_seg_maskdino_denoise", "mae"): {
        "defaults_before_checkpoint": [
            "/models/BUILD_maskDINO",
            "/models/meta_arch/maskdino/mae_large_w_masks_denoise",
            "/models/backbones/masked_encoder/large",
            "/models/backbones/maskdino_backbone/maskdino_backbone",
            "/models/heads/pixel_decoder/maskdino_encoder",
            "/models/heads/maskdino_decoder/maskdino_decoder_denoise",
        ],
        "defaults_before_optimizations": [
            "/optimizations/models/maskdino/mae_large",
        ],
        "backbone_target": "backbone.backbone",
    },
    ("instance_seg_maskdino_denoise", "jepa"): {
        "defaults_before_checkpoint": [
            "/models/BUILD_maskDINO",
            "/models/meta_arch/maskdino/mae_large_w_masks_denoise",
            "/models/backbones/masked_encoder/large",
            "/models/backbones/maskdino_backbone/maskdino_backbone",
            "/models/heads/pixel_decoder/maskdino_encoder",
            "/models/heads/maskdino_decoder/maskdino_decoder_denoise",
        ],
        "defaults_before_optimizations": [
            "/optimizations/models/maskdino/mae_large",
        ],
        "backbone_target": "backbone.backbone",
    },
    ("instance_seg_maskdino", "mae"): {
        "defaults_before_checkpoint": [
            "/models/BUILD_maskDINO",
            "/models/meta_arch/maskdino/mae_large_w_masks",
            "/models/backbones/masked_encoder/large",
            "/models/backbones/maskdino_backbone/maskdino_backbone",
            "/models/heads/pixel_decoder/maskdino_encoder",
            "/models/heads/maskdino_decoder/maskdino_decoder_box_init",
        ],
        "defaults_before_optimizations": [
            "/optimizations/models/maskdino/mae_large",
        ],
        "backbone_target": "backbone.backbone",
    },
    ("instance_seg_maskdino", "jepa"): {
        "defaults_before_checkpoint": [
            "/models/BUILD_maskDINO",
            "/models/meta_arch/maskdino/mae_large_w_masks",
            "/models/backbones/masked_encoder/large",
            "/models/backbones/maskdino_backbone/maskdino_backbone",
            "/models/heads/pixel_decoder/maskdino_encoder",
            "/models/heads/maskdino_decoder/maskdino_decoder_box_init",
        ],
        "defaults_before_optimizations": [
            "/optimizations/models/maskdino/mae_large",
        ],
        "backbone_target": "backbone.backbone",
    },
}


# -----------------------------
# Positional Encoding Presets (task-specific)
# Keys: (task_group, pos_encoding_type)
# -----------------------------

VALID_POS_ENCODINGS = ["rope", "sincos"]

TASK_POS_ENCODING_PRESETS = {
    # -----------------
    # channel_split
    # -----------------
    ("channel_split", "rope"): {
        "models.backbones.masked_encoder.abs_sincos_enc": False,
        "models.backbones.masked_encoder.rope_pos_enc": True,
        "models.backbones.masked_encoder.rope_random_rotation_per_head": True,
        "models.backbones.masked_encoder.rope_mixed": False,
        "models.backbones.masked_encoder.rope_theta": 100.0,
        "models.heads.maskedpredictor.abs_sincos_enc": False,
        "models.heads.maskedpredictor.rope_pos_enc": True,
        "models.heads.maskedpredictor.rope_random_rotation_per_head": True,
        "models.heads.maskedpredictor.rope_mixed": False,
        "models.heads.maskedpredictor.rope_theta": 100.0,
    },
    ("channel_split", "sincos"): {
        "models.backbones.masked_encoder.abs_sincos_enc": True,
        "models.backbones.masked_encoder.rope_pos_enc": False,
        "models.heads.maskedpredictor.abs_sincos_enc": True,
        "models.heads.maskedpredictor.rope_pos_enc": False,
    },

    # -----------------
    # upsample_space
    # -----------------
    ("upsample_space", "rope"): {
        "models.backbones.masked_encoder.abs_sincos_enc": False,
        "models.backbones.masked_encoder.rope_pos_enc": True,
        "models.backbones.masked_encoder.rope_random_rotation_per_head": True,
        "models.backbones.masked_encoder.rope_mixed": False,
        "models.backbones.masked_encoder.rope_theta": 100.0,
        "models.heads.maskedpredictor.abs_sincos_enc": False,
        "models.heads.maskedpredictor.rope_pos_enc": True,
        "models.heads.maskedpredictor.rope_random_rotation_per_head": True,
        "models.heads.maskedpredictor.rope_mixed": False,
        "models.heads.maskedpredictor.rope_theta": 100.0,
    },
    ("upsample_space", "sincos"): {
        "models.backbones.masked_encoder.abs_sincos_enc": True,
        "models.backbones.masked_encoder.rope_pos_enc": False,
        "models.heads.maskedpredictor.abs_sincos_enc": True,
        "models.heads.maskedpredictor.rope_pos_enc": False,
    },

    # -----------------
    # instance_seg (PlainDETR)
    # -----------------
    ("instance_seg", "rope"): {
        "models.backbones.masked_encoder.abs_sincos_enc": False,
        "models.backbones.masked_encoder.rope_pos_enc": True,
        "models.backbones.masked_encoder.rope_random_rotation_per_head": True,
        "models.backbones.masked_encoder.rope_mixed": False,
        "models.backbones.masked_encoder.rope_theta": 100.0,
    },
    ("instance_seg", "sincos"): {
        "models.backbones.masked_encoder.abs_sincos_enc": True,
        "models.backbones.masked_encoder.rope_pos_enc": False,
    },

    # -----------------
    # instance_seg_maskdino
    # -----------------
    ("instance_seg_maskdino", "rope"): {
        "models.backbones.masked_encoder.abs_sincos_enc": False,
        "models.backbones.masked_encoder.rope_pos_enc": True,
        "models.backbones.masked_encoder.rope_random_rotation_per_head": True,
        "models.backbones.masked_encoder.rope_mixed": False,
        "models.backbones.masked_encoder.rope_theta": 100.0,
    },
    ("instance_seg_maskdino", "sincos"): {
        "models.backbones.masked_encoder.abs_sincos_enc": True,
        "models.backbones.masked_encoder.rope_pos_enc": False,
    },

    # -----------------
    # instance_seg_maskdino_denoise
    # -----------------
    ("instance_seg_maskdino_denoise", "rope"): {
        "models.backbones.masked_encoder.abs_sincos_enc": False,
        "models.backbones.masked_encoder.rope_pos_enc": True,
        "models.backbones.masked_encoder.rope_random_rotation_per_head": True,
        "models.backbones.masked_encoder.rope_mixed": False,
        "models.backbones.masked_encoder.rope_theta": 100.0,
    },
    ("instance_seg_maskdino_denoise", "sincos"): {
        "models.backbones.masked_encoder.abs_sincos_enc": True,
        "models.backbones.masked_encoder.rope_pos_enc": False,
    },
}


# -----------------------------
# Head Presets
# Maps head name -> defaults path and decoder_args ref (for autobench)
# -----------------------------

HEAD_PRESETS = {
    "maskedpredictor": {
        "defaults_path": "/models/heads/maskedpredictor/large",
        "decoder_args_ref": "${models.heads.maskedpredictor}",
    },
    "linear": {
        "defaults_path": "/models/heads/linear/linear",
        "decoder_args_ref": "${models.heads.linear}",
    },
    "linear_probe": {
        "defaults_path": "/models/heads/linear/linear_probe",
        "decoder_args_ref": "${models.heads.linear}",
    },
}

# Group -> dotted key for meta_arch decoder_args (autobench tasks only)
GROUP_DECODER_ARGS_KEY = {
    "channel_split": "models.meta_arch.autobench.ChannelSplitAutoBench.decoder_args",
    "upsample_space": "models.meta_arch.autobench.UpsampleSpaceAutoBench.decoder_args",
}


# -----------------------------
# Helpers
# -----------------------------

def resolve_path(base_dir: Path, maybe_rel: str) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else (base_dir / p).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    # Strip leading "# @package ..." lines so PyYAML doesn't choke on them.
    text = re.sub(r"(?m)^\s*#\s*@package.*\n", "", text)
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at top-level in YAML: {path}")
    return data


def dump_yaml_with_header(data: Dict[str, Any]) -> str:
    body = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        width=120,
    )
    return "# @package _global_\n" + body


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge src into dst (in-place)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def set_by_dotted_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """
    overrides can contain:
      - dotted keys: {"optimizers.lr": 1e-3}
      - nested dicts: {"checkpoint": {"checkpoint_manager": {...}}}
    """
    for k, v in overrides.items():
        if "." in k and not isinstance(v, dict):
            set_by_dotted_path(cfg, k, v)
        elif "." in k and isinstance(v, dict):
            # still treat as dotted path (set dict at that node)
            set_by_dotted_path(cfg, k, v)
        else:
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                deep_update(cfg[k], v)
            else:
                cfg[k] = v


def ensure_defaults_list(cfg: Dict[str, Any]) -> List[Any]:
    if "defaults" not in cfg or cfg["defaults"] is None:
        cfg["defaults"] = []
    if not isinstance(cfg["defaults"], list):
        raise ValueError("cfg['defaults'] must be a list")
    return cfg["defaults"]


def insert_defaults(
    defaults_list: List[Any],
    entries: List[Any],
    insert_before: str | None = None,
) -> None:
    """
    Insert string entries (deduped) optionally before an anchor string.
    Keeps existing ordering otherwise.
    """
    # Find insertion index
    idx = len(defaults_list)
    if insert_before is not None:
        for i, item in enumerate(defaults_list):
            if item == insert_before:
                idx = i
                break

    # Insert while deduping (only for exact string matches)
    for e in entries:
        if e in defaults_list:
            continue
        defaults_list.insert(idx, e)
        idx += 1


def infer_method(s: str) -> str | None:
    s = (s or "").lower()
    if "jepa" in s:
        return "jepa"
    if "mae" in s:
        return "mae"
    return None


def to_decimal_str(x: Any) -> str:
    """
    Turn JSON number/string into a stable decimal-ish string for names.
    - 1e-3 -> "0.001"
    - 0.01 -> "0.01"
    """
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        # Avoid scientific notation for small floats by going through Decimal(str())
        try:
            d = Decimal(str(x))
        except InvalidOperation:
            return str(x)
        return format(d.normalize(), "f").rstrip("0").rstrip(".") or "0"
    if isinstance(x, str):
        # allow users to pass "1e-3" as a string
        try:
            d = Decimal(x)
            return format(d.normalize(), "f").rstrip("0").rstrip(".") or "0"
        except InvalidOperation:
            return x
    return str(x)


def val_to_name_token(v: Any) -> str:
    s = to_decimal_str(v)
    # 0.001 -> 0p001
    s = s.replace(".", "p")
    s = s.replace("-", "m")
    s = s.replace("+", "")
    s = s.replace("/", "_")
    return s


def make_sweep_descriptor(sweep_kv: List[Tuple[str, Any]]) -> str:
    parts = []
    for k, v in sweep_kv:
        short = k.split(".")[-1]
        parts.append(f"{short}_{val_to_name_token(v)}")
    return "_".join(parts)


# -----------------------------
# Main generation
# -----------------------------

# Generation steps:
# 1) Insert task+method model defaults BEFORE checkpoint block
# 2) Insert method-level defaults BEFORE optimizations block
# 3) User-provided defaults
# 4) Base required lines
# 5) checkpoint_manager (method preset + task/model-family target)
# 6) pretrained_checkpointdir (auto from pretrain experiment name)
# 7) apply sweep values
# 8) experiment_name auto pattern
# 9) user overrides last (so they win)
# 10) filename
# 11) write to file

def generate_from_recipe(recipe: Dict[str, Any], spec_dir: Path, out_root: Path) -> None:
    # NOTE: recipe is the pretrain config YAML and spec_dir is the directory of 
    # the finetuning configs to run for the given pretrain config
    group = recipe["group"]  # e.g. "instance_seg"
    name = recipe["name"]  # e.g. "plainDETR"

    # Grab the base config for the given group and name, see base_configs folder for examples
    base_path = resolve_path(spec_dir, recipe["base_config"])
    base_cfg = load_yaml(base_path)

    # Get the folder name (parent of spec.json, e.g., "exp_01_30_26_ablation_pos_enc")
    # This is used for wandb_project and path construction
    folder_name = spec_dir.name

    # Grab the pretrain config name from the recipe
    pretrain_exp_name = None
    pretrain_cfg_path = recipe.get("pretrain_config")
    if pretrain_cfg_path:
        pretrain_path = resolve_path(spec_dir, pretrain_cfg_path)
        pre_cfg = load_yaml(pretrain_path)
        pretrain_exp_name = pre_cfg.get("experiment_name")

    # Infer the pretraining method from the pretrain config name or recipe name (e.g. "jepa" or "mae")
    method = recipe.get("method") or infer_method(pretrain_exp_name or "") or infer_method(name) or "unknown"
    if method == "unknown":
        raise ValueError(f"Could not infer method from pretrain experiment name or recipe name: '{pretrain_exp_name}', '{name}'.")

    # Grab the sweep and overrides from the recipe
    sweep: Dict[str, List[Any]] = recipe.get("sweep", {}) or {}
    overrides: Dict[str, Any] = recipe.get("overrides", {}) or {}

    # Expand sweep combinations
    sweep_keys = list(sweep.keys())
    sweep_values = [sweep[k] for k in sweep_keys]
    if sweep_keys and any(not isinstance(vs, list) for vs in sweep_values):
        raise ValueError(f"sweep values must be lists. Got: {sweep}")

    combos = list(itertools.product(*sweep_values)) if sweep_keys else [()]

    # Output dir for this recipe group
    group_dir = out_root / group
    group_dir.mkdir(parents=True, exist_ok=True)

    for combo in combos:
        cfg = copy.deepcopy(base_cfg)

        # --- defaults injection
        defaults_list = ensure_defaults_list(cfg)

        # --- defaults + ckpt preset resolution
        tm = TASK_METHOD_DEFAULTS.get((group, method), {})
        m_preset = METHOD_PRESETS.get(method, {})

        # 1) Insert task+method model defaults BEFORE checkpoint block
        insert_defaults(
            defaults_list,
            tm.get("defaults_before_checkpoint", []),
            insert_before="/checkpoint/checkpoint",
        )

        # 2) Insert task-method defaults BEFORE optimizations block
        insert_defaults(
            defaults_list,
            tm.get("defaults_before_optimizations", []),
            insert_before="/optimizations/optimizations",
        )

        # 3) Insert method-level defaults BEFORE optimizations block (from METHOD_PRESETS)
        insert_defaults(
            defaults_list,
            m_preset.get("defaults_before_optimizations", []),
            insert_before="/optimizations/optimizations",
        )

        # 4) User-provided defaults
        user_defaults = recipe.get("defaults_add", {}) or {}
        insert_defaults(defaults_list, user_defaults.get("before_checkpoint", []), insert_before="/checkpoint/checkpoint")
        insert_defaults(defaults_list, user_defaults.get("before_optimizations", []), insert_before="/optimizations/optimizations")
        insert_defaults(defaults_list, user_defaults.get("append", []), insert_before=None)

        # 5) Head preset (if specified in recipe)
        head_name = recipe.get("head")
        if head_name is not None:
            if head_name not in HEAD_PRESETS:
                raise ValueError(
                    f"Unknown head '{head_name}' in recipe for '{group}/{name}'. "
                    f"Must be one of: {list(HEAD_PRESETS.keys())}"
                )
            head_preset = HEAD_PRESETS[head_name]
            head_defaults_path = head_preset["defaults_path"]
            # Remove any existing head default so the new head replaces it
            head_paths = [p["defaults_path"] for p in HEAD_PRESETS.values()]
            for hp in head_paths:
                while hp in defaults_list:
                    defaults_list.remove(hp)
            insert_defaults(defaults_list, [head_defaults_path], insert_before="/checkpoint/checkpoint")
            # Point meta_arch decoder_args at the chosen head
            decoder_args_key = GROUP_DECODER_ARGS_KEY.get(group)
            if decoder_args_key is not None:
                set_by_dotted_path(cfg, decoder_args_key, head_preset["decoder_args_ref"])

        # --- base required lines
        # Ensure deepspeed checkpoint load_universal
        if cfg.get("deepspeed") is None:
            cfg["deepspeed"] = {}
        if cfg["deepspeed"].get("checkpoint") is None:
            cfg["deepspeed"]["checkpoint"] = {}
        cfg["deepspeed"]["checkpoint"]["load_universal"] = True

        # --- checkpoint_manager (method preset + task/model-family target)
        set_ckpt_mgr = recipe.get("set_checkpoint_manager")
        if set_ckpt_mgr is None:
            # If we use a pretrained model, we need to set the checkpoint manager
            set_ckpt_mgr = bool(pretrain_exp_name)

        if set_ckpt_mgr:
            # Set the checkpoint manager in the config
            cfg["checkpoint"] = cfg.get("checkpoint", {})
            cfg["checkpoint"]["checkpoint_manager"] = cfg["checkpoint"].get("checkpoint_manager", {})

            # Build the checkpoint manager from the METHOD_PRESETS template, filling target from TASK_METHOD_DEFAULTS
            template = copy.deepcopy(m_preset.get("checkpoint_manager_template"))
            backbone_target = tm.get("backbone_target")

            # Fill the placeholder in ckpt_include_prefixes list
            prefixes = template.get("ckpt_include_prefixes", [])
            for i, prefix in enumerate(prefixes):
                if prefix == "__BACKBONE_TARGET__":
                    prefixes[i] = backbone_target

            # Fill the placeholder in the checkpoint manager template with the backbone target
            ckpt_map = template.get("ckpt_translate_map", {})
            for k, v in list(ckpt_map.items()):
                if v == "__BACKBONE_TARGET__":
                    ckpt_map[k] = backbone_target

            deep_update(cfg["checkpoint"]["checkpoint_manager"], template)

        # --- wandb_project (use folder name from spec.json location)
        cfg["wandb_project"] = folder_name

        # --- pretrained_checkpointdir (auto from pretrain config folder structure)
        # Pattern: pretrain configs live in folder X, so checkpoints are at:
        #   ${paths.data_path}/pretrained_models/{folder}/{pretrain_exp_name}/checkpoints
        # And finetuned outputs go to:
        #   ${paths.data_path}/finetuned_models/{folder}/{experiment_name}
        set_pretrained_dir = recipe.get("set_pretrained_checkpointdir")
        if set_pretrained_dir is None:
            set_pretrained_dir = bool(pretrain_exp_name)

        if set_pretrained_dir and pretrain_exp_name:
            if cfg.get("paths") is None:
                cfg["paths"] = {}
            cfg["paths"]["pretrained_checkpointdir"] = (
                f"${{paths.data_path}}/pretrained_models/{folder_name}/{pretrain_exp_name}/checkpoints"
            )

        # --- Set finetuned model output directory
        # Pattern: finetuned outputs go to ${paths.data_path}/finetuned_models/{folder}/{experiment_name}
        if cfg.get("paths") is None:
            cfg["paths"] = {}
        cfg["paths"]["outdir"] = f"${{paths.data_path}}/finetuned_models/{folder_name}/${{experiment_name}}"

        # --- apply sweep values
        sweep_kv = []
        for k, v in zip(sweep_keys, combo):
            set_by_dotted_path(cfg, k, v)
            sweep_kv.append((k, v))
        sweep_desc = make_sweep_descriptor(sweep_kv)

        # --- experiment_name auto pattern
        # pattern: pretrain_experiment_name_(TASK)_(SWEEP PARAMS)
        if recipe.get("set_experiment_name", True):
            if pretrain_exp_name:
                exp = f"{pretrain_exp_name}_{group}"
            else:
                exp = f"{name}_{group}"
            if sweep_desc:
                exp = f"{exp}_{sweep_desc}"
            cfg["experiment_name"] = exp

        # --- positional encoding preset (required for downstream tasks, task-specific)
        pos_encoding = recipe.get("pos_encoding")
        if pos_encoding is None:
            raise ValueError(
                f"Recipe for '{group}/{name}' is missing required 'pos_encoding' field. "
                f"Must be one of: {VALID_POS_ENCODINGS}"
            )
        if pos_encoding not in VALID_POS_ENCODINGS:
            raise ValueError(
                f"Unknown pos_encoding '{pos_encoding}' in recipe for '{group}/{name}'. "
                f"Must be one of: {VALID_POS_ENCODINGS}"
            )
        pos_enc_key = (group, pos_encoding)
        if pos_enc_key not in TASK_POS_ENCODING_PRESETS:
            raise ValueError(
                f"No positional encoding preset defined for task '{group}' with encoding '{pos_encoding}'. "
                f"Please add entry to TASK_POS_ENCODING_PRESETS in generate_configs.py"
            )
        pos_enc_overrides = TASK_POS_ENCODING_PRESETS[pos_enc_key]
        apply_overrides(cfg, pos_enc_overrides)

        # --- user overrides last (so they win)
        apply_overrides(cfg, overrides)

        # --- filename
        file_prefix = recipe.get("file_prefix", name)
        fname = f"{file_prefix}.yaml" if not sweep_desc else f"{file_prefix}__{sweep_desc}.yaml"
        out_path = group_dir / fname

        out_path.write_text(dump_yaml_with_header(cfg))

    print(f"[OK] {group}: wrote {len(combos)} configs to {group_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="Path to spec.json")
    ap.add_argument("--out_root", default=None, help="Output root dir (defaults to spec.json parent)")
    args = ap.parse_args()

    spec_path = Path(args.spec).resolve()
    spec_dir = spec_path.parent
    spec = json.loads(spec_path.read_text())

    out_root = Path(args.out_root).resolve() if args.out_root else resolve_path(spec_dir, spec.get("out_root", str(spec_dir)))

    recipes = spec.get("recipes", [])
    if not isinstance(recipes, list) or not recipes:
        raise ValueError("spec.json must contain a non-empty list: { 'recipes': [ ... ] }")

    for r in recipes:
        for required in ["group", "name", "base_config", "pretrain_config"]:
            if required not in r:
                raise ValueError(f"Recipe missing required key '{required}': {r}")
        generate_from_recipe(r, spec_dir=spec_dir, out_root=out_root)


if __name__ == "__main__":
    main()