"""Meta-arch output-contract helpers.

Declarative builders for ``model.output_metadata``.

The inference contract: ``get_output_metadata().tensor_info`` maps ``name -> {shape,
dtype, kind}``. Shapes are PER-SAMPLE (batch stripped); the runtime tensor returned by
``inference_step`` carries a leading ``B`` that the inferencer strips. ``kind`` drives the
inferencer's restore (nearest vs trilinear), buffer sizing, and save/viz routing.

Canonical kinds:
    dense               (T,Z,Y,X,C) / (*,C)  float32   spatial restore (trilinear)
    semantic_map        (Z,Y,X,1)            uint16     spatial restore (nearest)
    instance_label_map  (Z,Y,X,1)            uint16     spatial restore (nearest)
    instance_stack      (N,Z,Y,X,1)          bool       spatial restore (nearest)
    boxes/labels/scores (N,6)/(N,)/(N,)       f32/i64/f32  box rescale / none
    features            (N,C) + level/grid   float32    non-spatial; NOT saved (VLM/downstream)

A variable-length count (e.g. per-instance ``N``) is declared as ``None``; such outputs
are small/sparse (host-side, never SHM) or eval-only — see the fix-sketches doc.
"""

import math
from typing import Any, Optional, Sequence

import torch
from omegaconf import DictConfig


def output_metadata(**entries: dict) -> DictConfig:
    """Assemble ``{"tensor_info": {name: entry, ...}}`` from canonical kind entries."""
    return DictConfig({"tensor_info": dict(entries)})


def dense(shape: Sequence[int]) -> dict:
    return {"shape": tuple(shape), "dtype": "float32", "kind": "dense"}


def features(n: Optional[int], c: int, *, level: Any, token_grid: Sequence) -> dict:
    """Non-spatial token-feature sequence (VLM/downstream). Carries which encoder
    ``level`` it is and that level's ``token_grid`` so a consumer could re-spatialize."""
    return {
        "shape": (n, c),
        "dtype": "float32",
        "kind": "features",
        "level": level,
        "token_grid": tuple(token_grid),
    }


def semantic_map(spatial: Sequence[int]) -> dict:
    return {"shape": (*spatial, 1), "dtype": "uint16", "kind": "semantic_map"}


def instance_label_map(spatial: Sequence[int]) -> dict:
    return {"shape": (*spatial, 1), "dtype": "uint16", "kind": "instance_label_map"}


def instance_stack(n: Optional[int], spatial: Sequence[int]) -> dict:
    return {"shape": (n, *spatial, 1), "dtype": "bool", "kind": "instance_stack"}


def boxes(n: Optional[int]) -> dict:
    return {"shape": (n, 6), "dtype": "float32", "kind": "boxes"}


def labels(n: Optional[int]) -> dict:
    return {"shape": (n,), "dtype": "int64", "kind": "labels"}


def scores(n: Optional[int]) -> dict:
    return {"shape": (n,), "dtype": "float32", "kind": "scores"}


# ---------------------------------------------------------------------------
# Shared output collapse. Not a metadata builder -- this produces the runtime
# tensor that satisfies the ``semantic_map`` declaration above, so every model
# emitting that kind agrees on convention, dtype and channel axis.
# ---------------------------------------------------------------------------

def collapse_to_semantic_map(
    class_scores: torch.Tensor,
    *,
    threshold: float,
    dim: int = 1,
) -> torch.Tensor:
    """Per-class scores -> semantic label map, ``class + 1`` / background ``0``.

    Returns ``(..., 1)`` uint16 with the class axis at ``dim`` replaced by a
    trailing singleton channel -- the ``semantic_map`` kind's runtime form.

    ``class_scores`` may be logits OR probabilities: both ``argmax`` and the
    ``> threshold`` foreground test need only a monotonic score, so a sigmoid is
    never required here. ``threshold`` is therefore mandatory and must be given in
    whatever space the caller's scores live (logit ``0.0`` == probability ``0.5``).

    Note this is single-label by construction: where several classes exceed
    ``threshold``, ``argmax`` keeps one and the rest are dropped.
    """
    foreground = class_scores.amax(dim=dim) > threshold
    labelmap = (class_scores.argmax(dim=dim) + 1) * foreground
    return labelmap.to(torch.uint16).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Shared token-feature extraction for the pretrain models (MAE / JEPA).
#
# inference_step for MAE/JEPA returns full-context per-patch token features for
# downstream (VLM) consumption -- NOT saved. No masking, no unpatchify, no raster
# un-windowing, no scatter: the encoder output is flattened into a token sequence
# [B, N, C] per level (single level for vit / single-scale hiera; every level for
# multiscale hiera). ``token_grid`` metadata lets a consumer re-spatialize if it wants.
# ---------------------------------------------------------------------------

def extract_token_features(encoder, model, data_sample: dict) -> dict:
    """Full-context encode -> flat ``[B, N, C]`` token features per level."""
    inputs = data_sample["data_tensor"]
    spatial_kwargs = data_sample.get("metainfo", {}).get("spatial_kwargs")

    if model.backbone_type == "vit":
        # ViT MaskedEncoder(masks=None) returns all tokens as a [B, N, C] sequence.
        x, _ = encoder(inputs, masks=None, spatial_kwargs=spatial_kwargs)
        return {"features": x}

    # Hiera: fusion asserts return_windowed=True; equalization (multiscale) returns a
    # list of per-level windowed tensors. Flatten each [B, M, *O, C] -> [B, N, C].
    multiscale = bool(getattr(model, "multiscale", False))
    out, _ = encoder(
        inputs, masks=None, ctx_idx=None, with_intermediates=True,
        with_fusion_heads=not multiscale, return_windowed=True,
        spatial_kwargs=spatial_kwargs,
    )
    levels = out if isinstance(out, list) else [out]
    idxs = model.multiscale_level_indices if multiscale else ["final"]
    feats = {}
    for idx, lvl in zip(idxs, levels):
        seq = lvl.reshape(lvl.shape[0], -1, lvl.shape[-1])   # [B, M*prod(O), C]
        feats["features" if not multiscale else f"features_l{idx}"] = seq
    return feats


def build_feature_metadata(model, encoder) -> dict:
    """tensor_info entries for :func:`extract_token_features` (one per level).

    Best-effort/descriptive: these outputs are not saved, so the declaration is for
    programmatic (VLM) introspection, not buffer sizing.
    """
    if model.backbone_type == "vit":
        grid = tuple(d for d in encoder.patch_embedding.token_shape[:-1] if d is not None)
        return {"features": features(int(math.prod(grid)), int(model.embed_dim),
                                     level="final", token_grid=grid)}
    c = int(encoder.multiscale_out_dim)
    if not bool(getattr(model, "multiscale", False)):
        spec = encoder.get_decoder_spec()
        mg, tim = tuple(spec["mu_grid"]), tuple(spec["tok_in_mu"])
        n = int(math.prod(mg) * math.prod(tim))
        return {"features": features(n, c, level="final", token_grid=(mg, tim))}
    out = {}
    for idx, spec in zip(model.multiscale_level_indices, encoder.get_decoder_specs_per_level()):
        mg, tim = tuple(spec["mu_grid"]), tuple(spec["tok_in_mu"])
        n = int(math.prod(mg) * math.prod(tim))
        out[f"features_l{idx}"] = features(n, c, level=idx, token_grid=(mg, tim))
    return out
