import sys
import logging
import torch
import numpy
import tensorstore
from enum import Enum
from nvidia.dali import types

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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

    Exact kinds (``dense``, ``boxes``) map to themselves; family kinds match either
    the bare family name or a ``<family>_<name>`` suffix. Unknown kinds raise.
    """
    for fam in _KIND_FAMILIES:
        if kind == fam.value or kind.startswith(fam.value + "_"):
            return fam
    return DataKind(kind)  # exact kinds; ValueError for anything unknown


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