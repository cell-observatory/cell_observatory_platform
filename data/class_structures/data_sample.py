import logging
import sys
from typing import Any, Dict, Optional

from cell_observatory_platform.data.class_structures.base_data_element import BaseDataElement
from cell_observatory_platform.data.class_structures.image_list import ImageList

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataSample(BaseDataElement):
    """Unified container for SSL and SL training data."""

    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs):
        super().__init__(metainfo=metainfo, **kwargs)

    @property
    def data_tensor(self) -> ImageList:
        return self._data_tensor

    @data_tensor.setter
    def data_tensor(self, value: ImageList):
        self.set_field(value=value, name="_data_tensor", dtype=ImageList, field_type="data")

    @data_tensor.deleter
    def data_tensor(self):
        del self._data_tensor

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        d["metainfo"] = self.metainfo

        img_list: ImageList = self.data_tensor
        d["data_tensor"] = img_list.tensor
        d["data_tensor_meta"] = img_list._init_args
        return d

    @classmethod
    def from_dict(cls, d):
        inst = cls(metainfo=d["metainfo"])
        args = d["data_tensor_meta"]
        img_list = ImageList(**{k: args[k] for k in args})
        inst.data_tensor = img_list
        return inst
