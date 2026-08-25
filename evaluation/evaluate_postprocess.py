"""Evaluation postprocess: per-image GT construction shared by the evaluators.

Owns every read of the per-image target dict (``metainfo["targets"]``) and every GT resize/box
conversion, so the evaluators contain no target-key literals or resize/box helpers of their own.
Also builds the config-only ``unpatchify`` function (no model), mirroring how the preprocessor
builds ``patchify``.
"""

import functools
from typing import Any, Callable, Dict, List

import torch

from cell_observatory_platform.data.structures import box_cxcyczwhd_to_xyzxyz
from cell_observatory_platform.data.transforms.utils import (
    resize_label_map as _resize_label_map_exact,
    resize_masks as _resize_masks_exact,
)


# The one GT-box coordinate whitelist the evaluator imposes (may be extended).
GT_BOX_FORMATS = ("cxcyczwhd", "xyzxyz")


def extract_targets(
    data_sample: dict, *, squeeze_label_map: bool = False
) -> List[Dict[str, Any]]:
    """Per-image target list from ``metainfo["targets"]``.

    ``targets`` arrives as Form S -- a ``List[Dict]`` of length B, straight from every
    producer (see data/data_types.py; no wrap, nothing to unwrap). Shallow-copies each
    target (views, no deepcopy), and optionally squeezes a stray ``(1, Z, Y, X)``
    ``label_map`` down to ``(Z, Y, X)``.
    """
    raw = data_sample["metainfo"]["targets"]
    out: List[Dict[str, Any]] = []
    for t in raw:
        g = dict(t)
        lm = g.get("label_map")
        if squeeze_label_map and torch.is_tensor(lm) and lm.dim() == 4:
            if lm.shape[0] != 1:  # survives python -O: a T>1 map must not be
                raise ValueError(  # silently truncated to its first frame
                    f"expected T==1 label_map, got T={lm.shape[0]}"
                )
            g["label_map"] = lm[0]
        out.append(g)
    return out


def resize_label_map(label_map: torch.Tensor, size: Any) -> torch.Tensor:
    """Nearest-EXACT resize of an int label map to ``(Z, Y, X)``; no-op when already that size.

    Delegates to the data-pipeline kernel (pure ``index_select``, no float
    round-trip) so eval-time GT is warped with the SAME convention as train-time
    GT. Two reasons:

    * alignment -- plain ``mode="nearest"`` is ``floor(i * in/out)``, a half-voxel
      shift relative to the center-aligned image resize; the transforms use
      ``nearest-exact`` (``floor((i + 0.5) * in/out)``). Eval must match, or every
      small instance loses IoU to a systematic offset.
    * exactness -- ids never travel through float32, so the 24-bit mantissa cannot
      alias them (``16_777_219 -> 16_777_220``) regardless of magnitude. Source ids
      ride in on a uint16 channel today and so cannot reach ``2**24``, but index
      selection costs nothing to guarantee it.
    """
    size = tuple(int(s) for s in size)
    if tuple(label_map.shape[-3:]) == size:
        return label_map
    # .to(long): the kernel is dtype-preserving, but the evaluator's
    # searchsorted lookup compares against int64 mask_ids.
    return _resize_label_map_exact(label_map, size).to(torch.long)


def resize_masks(masks: torch.Tensor, size: Any) -> torch.Tensor:
    """Nearest-EXACT resize of a ``(N, Z, Y, X)`` bool mask stack; no-op when already that size.

    Same kernel as the label-map path (see ``resize_label_map``): bool survives
    ``index_select`` untouched, so the ``.float() ... > 0.5`` round-trip is gone --
    and the two GT views resized side by side can no longer disagree on the grid.
    """
    size = tuple(int(s) for s in size)
    if tuple(masks.shape[-3:]) == size:
        return masks
    return _resize_masks_exact(masks, size).bool()


def gt_boxes_abs_xyzxyz(target: Dict[str, Any], size: Any, fmt: str, normalized: bool) -> torch.Tensor:
    """GT boxes -> absolute ``xyzxyz`` (cxcyczwhd conversion + optional denormalize)."""
    boxes = target["boxes"]
    if boxes.numel() == 0:
        return boxes
    boxes = box_cxcyczwhd_to_xyzxyz(boxes) if fmt == "cxcyczwhd" else boxes
    if normalized:
        d, h, w = (int(s) for s in size)
        boxes = boxes * boxes.new_tensor([w, h, d, w, h, d])
    return boxes


def gt_semantic_map(target: Dict[str, Any], size: Any, source: str) -> torch.Tensor:
    """``(Z, Y, X)`` long GT class map (``class + 1``, background 0).

    ``source="masks"`` scatters ``label + 1`` from the per-class binaries the semantic
    preprocessor produced; this is the semantic path. Scattering is last-write-wins, so
    a later class wins any overlap -- fine for a real taxonomy, and the reason a stack
    of overlapping *derived* maps (a source footprint alongside its own partition) must
    not be declared as classes. See preprocessor.build_semantic_targets.

    ``source="label_map"`` maps instance ids to ``class + 1``. For INSTANCE datasets,
    whose targets carry real per-instance class ids; use it to score semantic mIoU
    against instance GT.
    """
    labels = target["labels"]
    if source == "masks":
        masks = target["masks"]  # (N, Z, Y, X)
        gt = torch.zeros(tuple(masks.shape[1:]), dtype=torch.long, device=masks.device)
        # Scatter loop kept for the general case, but for DB-sourced semantic
        # classes the overlap it defends against can no longer occur: semantic GT
        # is one squashed integer channel, one-hotted by build_semantic_targets, so
        # the classes are mutually exclusive and this scatter is exact. Overlap is
        # still possible between DERIVED roles (boundary/foreground are thresholded
        # independently), where last-write-wins remains the semantics -- and a
        # vectorized gather could not reproduce it.
        for i in range(masks.shape[0]):
            gt[masks[i].bool()] = int(labels[i]) + 1
    else:  # label_map
        label_map, mask_ids = target["label_map"], target["mask_ids"]
        # LUT gather: ONE pass over the volume instead of one full-volume
        # Ids absent from mask_ids (incl. background 0) map to 0.
        max_id = int(label_map.max().item()) if label_map.numel() else 0
        if mask_ids.numel():
            max_id = max(max_id, int(mask_ids.max().item()))
        if max_id <= 2**24:
            lut = torch.zeros(max_id + 1, dtype=torch.long, device=label_map.device)
            lut[mask_ids.long()] = labels.long() + 1
            gt = lut[label_map.long()]
        else:
            # TODO: this feels highly unlikely.
            # Sparse/huge ids (e.g. 64-bit tile-global): a max_id-sized LUT is
            # not safe to allocate — fall back to the per-instance loop.
            gt = torch.zeros_like(label_map, dtype=torch.long)
            for inst_id, cls in zip(mask_ids.tolist(), labels.tolist()):
                gt[label_map == inst_id] = int(cls) + 1
    return resize_label_map(gt, size)


def gt_masks_for_class(
    target: Dict[str, Any], gt_class_mask: torch.Tensor, size: Any, source: str
) -> torch.Tensor:
    """``(n_gt_class, Z, Y, X)`` bool GT masks for one class, from masks or label_map."""
    if source == "masks":
        m = target["masks"][gt_class_mask].bool()
    else:  # label_map
        ids = target["mask_ids"][gt_class_mask]
        m = target["label_map"][None] == ids.view(-1, 1, 1, 1)
    return resize_masks(m, size)


# TODO: consider having one unified unpatchify function since this is reused in multiple places.
def build_unpatchify(patch_shape, input_shape, input_format) -> Callable:
    """A config-built ``unpatchify(patches, out_channels)`` -- no model.

    Partials ``PatchEmbedding.unpatchify`` (a pure staticmethod) over the patch sizes / token
    shape derived from config, mirroring ``RayPreprocessor.pe_patchify``.
    """
    from cell_observatory_platform.models.layers.patch_embeddings import (
        PatchEmbedding,
        calc_num_patches,
    )
    from cell_observatory_platform.training.helpers import get_patch_sizes

    t_ps, z_ps, l_ps = get_patch_sizes(input_format=input_format, patch_shape=patch_shape)
    _, token_shape = calc_num_patches(
        input_fmt=input_format, input_shape=input_shape, patch_shape=patch_shape
    )
    return functools.partial(
        PatchEmbedding.unpatchify,
        temporal_patch_size=t_ps,
        axial_patch_size=z_ps,
        lateral_patch_size=l_ps,
        token_shape=token_shape,
        input_format=input_format,
    )
