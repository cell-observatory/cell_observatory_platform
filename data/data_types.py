import sys
import logging
import torch
import numpy as np
from enum import Enum

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

class NUMPY_DTYPES(Enum):
    float32 = np.float32
    fp32 = np.float32
    float16 = np.float16
    fp16 = np.float16