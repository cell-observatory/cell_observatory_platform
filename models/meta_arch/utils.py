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
are small/sparse (host-side, never SHM) or eval-only.
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
# Mask2Former query -> dense semantic map reduction. The reduce step that pairs
# with collapse_to_semantic_map above (reduce -> [B,D,H,W,C] -> collapse). The
# mask2former meta-arch's streaming path is verified equivalent to this reference.
# ---------------------------------------------------------------------------

def _reduce_topk_max(
    pred_masks: torch.Tensor,
    pred_logits: torch.Tensor,
    num_classes: int = 1,
    topk_per_image: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    LEGACY reduction: top-k queries per class, combined with max.

    Returns a per-class map in probability space with NO background channel -- callers
    derive background by thresholding. Retained so checkpoints tuned against this
    reduction stay scoreable; prefer ``reduction="canonical"`` for new work.

    Args:
        pred_masks: [B, num_queries, D, H, W] - mask logits (already at target resolution)
        pred_logits: [B, num_queries, num_classes + 1] - indices 0..num_classes-1 = semantic classes, last = no-object
        num_classes: Number of foreground classes (default: 1 for binary segmentation)
        topk_per_image: Number of top queries to use per image (for num_classes=1 case)

    Returns:
        semantic_map: [B, D, H, W, num_classes] - channels-last dense semantic map (per-class)
        average_probability: average probability per class, [B, num_classes]
            in BOTH branches ([B, 1] for the single-class case).
    """
    B, num_queries, D, H, W = pred_masks.shape

    if num_classes == 1:
        # Binary segmentation: single semantic class at index 0, no-object at index 1
        probs = pred_logits.softmax(-1)[..., 0]  # [B, Q] - class 0 (foreground) probability per query
        # Clamp k like the multi-class branch: k > num_queries is a config
        # smell, not a crash-worthy contract violation in ONE of two branches.
        topk_per_image = min(topk_per_image, num_queries)
        # Get top k queries by foreground probability
        topk_indices = probs.topk(k=topk_per_image, dim=1).indices  # [B, K]
        topk_probs = probs.gather(1, topk_indices)  # [B, K]
        # Get masks for top k queries: gather output shape = index shape, so expand index to [B, K, D, H, W]
        topk_indices_expanded = topk_indices.view(B, topk_per_image, 1, 1, 1).expand(B, topk_per_image, D, H, W)
        topk_masks = pred_masks.gather(1, topk_indices_expanded)  # [B, K, D, H, W]
        # Convert mask logits to probabilities
        topk_masks = topk_masks.sigmoid()  # [B, K, D, H, W]
        # Weight masks by foreground probability and sum over top k queries
        # topk_probs: [B, K] -> [B, K, 1, 1, 1] for broadcasting
        semantic = (topk_probs.view(B, topk_per_image, 1, 1, 1) * topk_masks).sum(1, keepdim=True)  # [B, 1, D, H, W]
        # Permute to channels-last: [B, D, H, W, 1]
        semantic = semantic.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 1]
        # keepdim: uniform [B, num_classes] rank across both branches (the
        # docstring always promised [B, 1] here; callers no longer branch).
        average_probability = topk_probs.mean(dim=1, keepdim=True)  # [B, 1]
        return semantic, average_probability
    else:
        # Multi-class segmentation: produce per-class maps
        # pred_logits: [B, Q, num_classes + 1]; indices 0..num_classes-1 = semantic classes, last index = no-object (DETR/Mask2Former convention, see losses.py empty_weight[-1])
        class_probs = pred_logits.softmax(-1)[..., :-1]  # [B, Q, num_classes] - keep all semantic classes, drop no-object

        # For each class, combine masks from queries assigned to that class
        semantic_per_class = []
        average_probability_per_class = []
        for c in range(num_classes):
            # Get probability of class c for each query: [B, Q]
            class_c_probs = class_probs[..., c]  # [B, Q]

            # Get top k queries for this class
            topk_indices = class_c_probs.topk(k=min(topk_per_image, num_queries), dim=1).indices  # [B, K]
            topk_probs = class_c_probs.gather(1, topk_indices)  # [B, K]

            # Get masks for top k queries
            topk_indices_expanded = topk_indices.view(B, -1, 1, 1, 1).expand(B, -1, D, H, W)
            topk_masks = pred_masks.gather(1, topk_indices_expanded)  # [B, K, D, H, W]
            topk_masks = topk_masks.sigmoid()  # [B, K, D, H, W]

            # Weight by class probability and combine (max aggregation for cleaner maps)
            # Use max instead of sum to avoid over-saturation
            weighted_masks = topk_probs.view(B, -1, 1, 1, 1) * topk_masks  # [B, K, D, H, W]
            class_map = weighted_masks.max(dim=1)[0]  # [B, D, H, W] - max over queries
            semantic_per_class.append(class_map)
            average_probability_per_class.append(topk_probs.mean(dim=1))

        # Stack along channel dimension: [B, D, H, W, num_classes]
        semantic = torch.stack(semantic_per_class, dim=-1)  # [B, D, H, W, num_classes]
        average_probability = torch.stack(average_probability_per_class, dim=-1)  # [B, num_classes]
        return semantic, average_probability


def _reduce_canonical(
    pred_masks: torch.Tensor,
    pred_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonical Mask2Former semantic inference.

    Every query contributes, weighted by its class posterior:
    ``semseg[c] = sum_q softmax(logits)[q, c] * sigmoid(mask)[q]``.

    Args:
        pred_masks: ``[B, Q, D, H, W]`` mask logits at target resolution.
        pred_logits: ``[B, Q, num_classes + 1]``; last index is no-object
            (DETR/Mask2Former convention, see losses.py ``empty_weight[-1]``).

    Returns:
        semseg: ``[B, D, H, W, 1 + num_classes]`` channels-last, **channel 0 is
            no-object (background)**, so ``argmax(dim=-1)`` yields a label map in the
            ``class + 1`` / background ``0`` convention with no threshold.
        average_probability: ``[B, num_classes]`` mean per-class posterior over queries.
    """
    if pred_logits.dim() != 3 or pred_masks.dim() != 5:
        raise ValueError(
            f"expected pred_masks (B,Q,D,H,W) and pred_logits (B,Q,C+1); "
            f"got {tuple(pred_masks.shape)} and {tuple(pred_logits.shape)}"
        )
    probs = pred_logits.softmax(-1)                      # [B, Q, C+1], last = no-object
    masks = pred_masks.sigmoid()                         # [B, Q, D, H, W]

    # [B,Q,K] x [B,Q,D,H,W] -> [B,D,H,W,K], K = C+1 with no-object last.
    semseg = torch.einsum("bqk,bqdhw->bdhwk", probs, masks)
    # Roll no-object to channel 0 so argmax gives class+1 / bg-0 directly.
    semseg = torch.cat([semseg[..., -1:], semseg[..., :-1]], dim=-1)

    average_probability = probs[..., :-1].mean(dim=1)    # [B, C]
    return semseg, average_probability


def reduce_queries_to_semantic_map(
    pred_masks: torch.Tensor,
    pred_logits: torch.Tensor,
    num_classes: int = 1,
    topk_per_image: int = 1,
    reduction: str = "canonical",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Mask2Former query outputs to a dense semantic map.

    ``reduction="canonical"`` (default) sums over all queries and returns a
    background-first ``[B, D, H, W, 1 + C]`` map -- collapse with ``argmax(dim=-1)``.
    ``reduction="topk_max"`` is the legacy top-k/max path returning
    ``[B, D, H, W, C]`` with no background channel -- collapse with
    ``collapse_to_semantic_map(threshold=0.5, dim=-1)``.

    The layouts differ because the two derive background differently; the caller must
    pair the reduction with its matching collapse. ``num_classes`` and
    ``topk_per_image`` apply to ``topk_max`` only; the canonical path infers
    ``C = pred_logits.shape[-1] - 1``.
    """
    if reduction == "canonical":
        return _reduce_canonical(pred_masks, pred_logits)
    if reduction == "topk_max":
        return _reduce_topk_max(pred_masks, pred_logits, num_classes, topk_per_image)
    raise ValueError(f"reduction must be 'canonical' or 'topk_max'; got {reduction!r}")


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
