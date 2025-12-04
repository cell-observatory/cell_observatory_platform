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