from typing import Dict

import torch

from cell_observatory_platform.data.structures.data_sample import DataSample
from cell_observatory_platform.data.structures.base_data_element import BaseDataElement


class PreProcessor(torch.nn.Module):
    def __init__(self, sample_object: BaseDataElement = DataSample):
        super().__init__()
        if not isinstance(sample_object, BaseDataElement):
            raise TypeError(f"sample_object must be a subclass of \
                                BaseDataElement, got {type(sample_object)}")
        self.sample_object = sample_object

    def forward(self, data_sample: Dict):
        return self.sample_object.from_dict(data_sample)