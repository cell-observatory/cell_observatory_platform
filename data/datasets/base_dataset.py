import abc
import sys
import logging
from typing import Any, Callable, Optional, Sequence, Mapping, Dict

from torch.utils.data import Dataset

from data.databases.supabase_database import SupabaseDatabase
from data.structures.data_sample import DataSample
from data.data_shapes import MULTICHANNEL_3D_HYPERCUBE, MULTICHANNEL_4D_HYPERCUBE


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Transformations:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional)
        Sequence of transforms to apply to each sample
    """

    def __init__(self, transforms: Optional[Sequence[Callable]] = None):
        self.transforms = list(transforms) if transforms is not None else []

    def __call__(self, data_sample: DataSample) -> Optional[DataSample]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data_sample = t(data_sample)
        return data_sample


class BaseDataset(Dataset, metaclass=abc.ABCMeta):
    """Base class for all datasets."""

    def __init__(
            self,
            db: SupabaseDatabase,
            layout: MULTICHANNEL_3D_HYPERCUBE | MULTICHANNEL_4D_HYPERCUBE,
            transforms: Optional[Sequence] = None,
    ):
        super().__init__()


        self.db = db
        self.layout = layout
        self._process_tables()

        self._index = None
        self._build_index()

        self.transforms = Transformations(transforms)

    @abc.abstractmethod
    def _process_tables(self) -> None:
        """Process tables in the database."""
        pass

    @abc.abstractmethod
    def _build_index(self) -> None:
        pass

    @abc.abstractmethod
    def _load_sample(self, idx_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Given one entry from the pre-built index, fetch and return raw data."""
        pass

    @abc.abstractmethod
    def _collate(self, raw: Dict[str, Any]) -> Mapping[str, Any]:
        """Turn raw dict into sample and data objects."""
        pass

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        _data = self._load_sample(self._index[idx])
        data = self._collate(_data)
        return self.transforms(data)