import sys
import logging
from typing import Any, Optional

import torch
import ujson
import numpy
import tensorstore
from enum import Enum
from nvidia.dali import types

logger = logging.getLogger(__name__)


class TORCH_DTYPES(Enum):
    float32 = torch.float32
    fp32 = torch.float32

    float16 = torch.float16
    fp16 = torch.float16

    bfloat16 = torch.bfloat16
    bf16 = torch.bfloat16

    uint16 = torch.uint16
    u16 = torch.uint16

class NUMPY_DTYPES(Enum):
    float32 = numpy.float32
    fp32 = numpy.float32

    float16 = numpy.float16
    fp16 = numpy.float16

    uint8 = numpy.uint8
    u8 = numpy.uint8

    uint16 = numpy.uint16
    u16 = numpy.uint16

class TENSORSTORE_DTYPES(Enum):
    float32 = tensorstore.float32
    fp32 = tensorstore.float32

    float16 = tensorstore.float16
    fp16 = tensorstore.float16

    bfloat16 = tensorstore.bfloat16
    bf16 = tensorstore.bfloat16

    uint16 = tensorstore.uint16
    u16 = tensorstore.uint16

class DALI_DTYPES(Enum):
    uint16 = types.UINT16
    u16 = types.UINT16

    float16 = types.FLOAT16
    fp16 = types.FLOAT16

    float32 = types.FLOAT
    fp32 = types.FLOAT


class DataKind(str, Enum):
    """Canonical kind of a sample field. Preprocessors declare each field's kind in
    ``data_types``; transforms dispatch behavior on it (one handler per kind).

    ``SEMANTIC_MASKS`` and ``INSTANCE_MASKS`` are FAMILIES: a concrete kind such as
    ``"semantic_masks_membrane"`` resolves (via :func:`kind_family`) to the family, so
    any ``semantic_masks_<name>`` is handled identically.
    """
    DENSE = "dense"
    INSTANCE_MASKS = "instance_masks"
    SEMANTIC_MASKS = "semantic_masks"
    BOXES = "boxes"


# Kinds whose concrete value may carry a "_<name>" suffix (family prefixes).
_KIND_FAMILIES = (DataKind.SEMANTIC_MASKS, DataKind.INSTANCE_MASKS)


def kind_family(kind: str) -> DataKind:
    """Resolve a concrete kind string to its canonical :class:`DataKind` family.
    Family kinds match either the bare family name or a ``<family>_<name>`` suffix. 
    Unknown kinds raise a ValueError.
    """
    for fam in _KIND_FAMILIES:
        if kind == fam.value or kind.startswith(fam.value + "_"):
            return fam
    return DataKind(kind)  # exact kinds; ValueError for anything unknown


# ---------------------------------------------------------------------------
# The ``metainfo["targets"]`` contract: two forms, discriminated by container type.
#
# :class:`DataKind` above says what kind each FIELD inside a target is; this says
# what shape the CONTAINER holding them takes.
#
# ``metainfo["targets"]`` is exactly one of two shapes -- pick whichever is NATURAL
# for the pipeline and keep it end-to-end (no wrapping, no ``targets[0]``
# unwrapping, no conversions except at genuine transformation points):
#
# Form D (role-keyed dict)   ``dict[str role -> (B, ...) tensor]``
#     Batched values; vectorized ops. Used where the whole batch is processed at
#     once:
#       dense/reconstruction  {"denoising": (B, T, Z, Y, X, C)} -> (B, N, ppp*C) patchified
#       semantic maps stage   {"golgi": (B, Z, Y, X), "boundary": (B, Z, Y, X), ...}
#
# Form S (per-sample list)   ``List[Dict[str, Any]]`` of length B
#     One dict per sample; ragged per-sample values. Used where instance counts vary
#     per sample and consumers are per-sample by nature (Hungarian matching, set
#     losses, per-sample evaluators):
#       structured (instance/detection)  [{"boxes": (N_b, 6), "labels": (N_b,),
#                                          "mask_ids": (N_b,), "label_map": (Z, Y, X),
#                                          "masks": (N_b, Z, Y, X)}, ...]
#       semantic packaged                [{"masks": (N_b, Z, Y, X), "labels": (N_b,)}, ...]
#       SAM2 eval GT                     (same keys as structured)
#
# Rules:
#   - The container type IS the form: ``dict`` == Form D, ``list`` == Form S. No
#     marker metadata; nothing may sniff beyond that single distinction (and only
#     generic boundary code -- e.g. the inference validation -- ever needs to test
#     it; pipeline code statically knows its form).
#   - The ONLY in-pipeline conversion is semantic packaging
#     (``build_semantic_targets``: Form D maps in -> Form S set-loss GT out).
#     Everything else keeps one form.
#   - Form D values stay batched -- never decompose per-sample (de-vectorizes the
#     loss and Normalize). Form S stays per-sample -- never stack ragged fields.
#
# Time: NOT SUPPORTED. Both forms are 3D-spatial on the target side -- Form S
#   carries boxes (N, 6) / mask_ids (N,) with no time axis, so a time_size > 1
#   window cannot be represented and parse_annotations_metadata reads a single
#   bucket. The IMAGE path is already 4D-clean.
# ---------------------------------------------------------------------------


def get_role(targets: dict, role: str) -> torch.Tensor:
    """Form-D read with a loud, listing KeyError.

    The drift guard between independently-configured role names (e.g. a
    preprocessor's ``recon_role`` vs ``DeepCopyInputsAsTargets(role=...)``): a
    mismatch names both sides instead of failing as a bare KeyError.
    """
    if role not in targets:
        raise KeyError(
            f"target role {role!r} not found; targets has {list(targets)} "
            f"-- check the transform/preprocessor role configs match"
        )
    return targets[role]


class OutputKind(str, Enum):
    """Canonical kind of a MODEL OUTPUT tensor (declared in the model's output
    metadata; the inference / save / viz side dispatches on it).

    Distinct from :class:`DataKind` (the data/preprocessor-target taxonomy): the
    instance case is split by *representation* here because the two need different
    downstream handling -- a label map is exploded into per-object masks, a stack
    is already per-object. ``DENSE`` covers every dense grid output (reconstruction
    image, denoised image, dense semantic probability map).
    """
    DENSE = "dense"                            # dense volume: recon / image / semantic prob map
    INSTANCE_LABEL_MAP = "instance_label_map"  # integer instance labelmap (one volume, ids)
    INSTANCE_STACK = "instance_stack"          # explicit per-object mask stack (N, ...)
    BOXES = "boxes"                            # coordinate bounding boxes (N, 6)

def parse_annotations_metadata(
    raw: Any, *, window_offset: int = 0
) -> tuple[list[dict], list[dict]]:
    """Split one ``annotations_metadata`` payload into (instance, semantic) leaves.

    The collator builds per-instance targets from the ``instance`` list, and
    the semantic preprocessor reads the ``semantic`` list as the class legend for
    the semantic labelmap channel.

    The payload is now a time-outer / kind-inner dictionary with WINDOW-LOCAL keys -- 
    ``str(timepoint - time_start)``, range ``0 .. time_size-1``, identical 
    on both training views:

        {"0": {"instance": [{local_segmentation_id, object_type_id,
                             object_subtype_ids, bbox_zyxzyx}, ...],
               "semantic": [{local_segmentation_id, object_type_id,
                             object_subtype_ids}, ...]}}

    ``window_offset`` selects exactly ONE bucket, and stays 0: 4D targets are
    unimplemented. The per-sample targets contract above carries no time axis
    (``boxes`` is ``(N, 6)``, ``mask_ids`` is ``(N,)``), so a ``time_size > 1``
    window has nowhere to put frames ``1 .. T-1``. The image path is already
    4D-clean.

    A missing key means "no objects in that box".
    """
    if hasattr(raw, "as_py"):
        raw = raw.as_py()
    if raw is None:
        return [], []
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        payload = raw.strip()
        if not payload or payload.lower() == "null":
            return [], []
        raw = ujson.loads(payload)

    if not isinstance(raw, dict):
        raise ValueError(
            f"annotations_metadata must be a time-keyed JSON object "
            f"({{'0': {{'instance': [...], 'semantic': [...]}}}}), got "
            f"{type(raw).__name__}"
        )

    bucket = raw.get(str(int(window_offset))) or {}
    if not isinstance(bucket, dict):
        raise ValueError(
            f"annotations_metadata['{window_offset}'] must be a "
            f"{{instance, semantic}} object, got {type(bucket).__name__}"
        )
    return (
        [item for item in bucket.get("instance", []) if isinstance(item, dict)],
        [item for item in bucket.get("semantic", []) if isinstance(item, dict)],
    )