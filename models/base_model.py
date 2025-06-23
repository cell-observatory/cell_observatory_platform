import abc
from functools import wraps
from operator import attrgetter
from typing import Dict, Union, List

import torch
from torch.utils.checkpoint import checkpoint

from models.preprocessor import PreProcessor
from cell_observatory_platform.data.structures.base_data_element import BaseDataElement


class BaseModel(torch.nn.Module):
    """
    Base class for Cell Observatory models.
    All models should subclass this class.
    """
    def __init__(self, preprocessor: PreProcessor = None):
        super(BaseModel, self).__init__()
        self.preprocessor = preprocessor 

    @abc.abstractmethod
    def _forward(self, data_sample: BaseDataElement):
        """
        Main forward pass implementation of the model.
        This method should be implemented by subclasses.
        """
        pass    

    def forward(self, data_sample: Dict = None):
        """
        Entry point for forward pass of the model.
        """
        if self.preprocessor is not None:
            data_sample = self.preprocessor(data_sample)
        return self._forward(data_sample)

    def wrap_forward(self, forward):
        @wraps(forward)
        def wrapper(*args):
            return checkpoint(forward, *args)
        return wrapper

    # from: mmengine/runner/activation_checkpointing.py
    def activation_checkpoint(self, modules: Union[List[str], str]):
        """
        Wrap the forward method of the specified modules
        with activation checkpointing.
        """
        if isinstance(modules, str):
            modules = [modules]
        for module_name in modules:
            module = attrgetter(module_name)(self)
            module.forward = self.wrap_forward(module.forward)

    def freeze(self, modules: Union[str, List[str]]):
        """
        Freeze the parameters of the model.
        """
        if isinstance(modules, str):
            modules = [modules]
        for module_name in modules:
            module = attrgetter(module_name)(self)
            for param in module.parameters():
                param.requires_grad = False