import abc
import sys
import logging
from typing import Any, Callable, Optional, Sequence, Mapping, Dict
from pathlib import Path
import ujson
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate as torch_default_collate

from data.structures.data_sample import DataSample
from data.structures.image_list import cat_image_lists
from data.data_shapes import MULTICHANNEL_HYPERCUBE
from data.data_types import TENSORSTORE_DTYPES, NUMPY_DTYPES, TORCH_DTYPES

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# from: https://github.com/open-mmlab/mmengine/main/mmengine/dataset/utils.py
def default_collate(data_batch: Sequence) -> Any:
    """
    Convert list of data sampled from dataset into
    a batch of data, of type consistent with 
    the type of each element in ``data_batch``.

    Args:
        data_batch (Sequence): Data sampled from dataset.

    Returns:
        Any: Data in the same format as the data element 
        of ``data_batch``, of which tensors have been 
        stacked, and ndarray, int, float have been
        converted to tensors, etc.
    """  
    # NOTE: we assume the each data element in data_batch
    #       is of the same type
    data_item = data_batch[0]
    data_item_type = type(data_item)

    # recursive collate
    if isinstance(data_item, (str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named tuple
        # we transpose the batch to get a tuple of lists
        # recursively collate each list, then lastly
        # rebuild same named tuple type
        return data_item_type(*(default_collate(samples)
                                for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        # check to make sure that the elements 
        # in batch have consistent size
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                'each data_itement in list of batch should be of equal size')
        
        # from [(a, b), (c, d)] to [(a, c), (b, d)]
        transposed = list(zip(*data_batch))

        # from [(a, c), (b, d)] to [collated[a, c], collated[b, d]]
        if isinstance(data_item, tuple):
            # compatible with Pytorch
            return [default_collate(samples)
                    for samples in transposed]  
        else:
            try:
                return data_item_type(
                    [default_collate(samples) for samples in transposed])
            except TypeError:
                # sequence type may not support `__init__(iterable)`
                # (e.g., `range`). Fall back to list.
                return [default_collate(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        # [{"img": img1, "label": label1},
        #  {"img": img2, "label": label2}]
        # to {"img": collate[img1, img2], "label": collate[label1, label2]}
        return data_item_type({
            key: default_collate([d[key] for d in data_batch])
            for key in data_item
        })
    else:
        return torch_default_collate(data_batch)


# each Dataset class defines its own collate function
def collate_base_dataset(samples: list[DataSample]) -> DataSample:
    metainfo = default_collate([s.metainfo for s in samples])
    batch = DataSample(metainfo=metainfo)
    batched_img = cat_image_lists(image_lists=[s.data_tensor for s in samples])
    batch.data_tensor = batched_img
    return batch.to_dict()


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
        hypercubes_dataframe_path: str | Path,
        input_layout: MULTICHANNEL_HYPERCUBE,
        transforms: Optional[Sequence] = None,
        server_folder_path: Optional[Path] = None,
        dtype: NUMPY_DTYPES | TENSORSTORE_DTYPES | TORCH_DTYPES | str = NUMPY_DTYPES.fp16,
    ):
        """
        Args:
            hypercubes_dataframe_path: path to pre-processed hypercubes dataframe from the supabase database
            input_layout: see MULTICHANNEL_HYPERCUBE
            transforms: list of optional transforms to apply to each sample (default: None)
            server_folder_path: path to override default server folder found in the supabase database
                update this path based on where the data is stored on your local machine
        """
        super().__init__()

        self.input_layout = input_layout
        self.server_folder_path = server_folder_path
        self.hypercubes_dataframe_path = Path(hypercubes_dataframe_path)
        self.hypercubes_dataframe, self.hypercubes_dataframe_config = self._process_tables(self.hypercubes_dataframe_path)
        self.dtype = dtype

        self._index = None
        self._build_index()

        self.transforms = Transformations(transforms)

    def _process_tables(self, hypercubes_dataframe_path) -> tuple[pd.DataFrame, Dict]:
        if not hypercubes_dataframe_path.exists():
            raise FileNotFoundError(f"{hypercubes_dataframe_path} does not exist")

        hypercubes = pd.read_csv(hypercubes_dataframe_path, index_col=0, header=0)
        with open(hypercubes_dataframe_path.with_suffix('.json'), 'r') as f:
            configs = ujson.load(f)

        if self.server_folder_path is not None:
            hypercubes['server_folder'] = self.server_folder_path
        return hypercubes, configs

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