import os
from typing import Dict, Any, Tuple
import torch
from dask.dataframe.tests.test_pyarrow_compat import dtype
from torch.utils.data import get_worker_info

from data.data_types import TENSORSTORE_DTYPES
from data.io import read_zarr
from data.structures.data_sample import DataSample
from data.structures.image_list import ImageList, cat_image_lists
from data.datasets.base_dataset import BaseDataset, default_collate


def collate_pretrain_dataset(samples: list["DataSample"]) -> "DataSample":
    metainfo = default_collate([s.metainfo for s in samples])
    batch = DataSample(metainfo=metainfo)    
    # TODO: we can't currently use this collate function since 
    #      it's unclear if we want to do torch.stack on cpu which 
    #       is a result of cat_image_lists
    batched_img = cat_image_lists(image_lists=[s.data_tensor for s in samples])
    batch.data_tensor = batched_img    
    return batch.to_dict()

def simple_collate_pretrain_dataset(samples: list["DataSample"]) -> "DataSample":
    """
    Simple collate function for pretrain dataset.
    """
    metainfo = default_collate([s.metainfo for s in samples])
    # no image list class until we add a helper function that doesn't stack
    # images in the image list
    image_list = [s.data_tensor.tensor for s in samples]
    return {
        'data_tensor': image_list,
        'metainfo': metainfo,
    }

class PretrainDataset(BaseDataset):
    """
    Dataset for pretraining.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zarr_handles_data = {}
        self.dtype = TENSORSTORE_DTYPES[self.dtype].value if isinstance(self.dtype, str) else dtype

    def worker_init_fn(self, worker_id):
        worker_info = get_worker_info()
        # re-open handles in this worker only 
        # important to pass to dataloader
        self.paths = {
            os.path.join(sf, of, tn)
            for sf, of, tn in 
            zip(self.hypercubes_dataframe["server_folder"], 
                self.hypercubes_dataframe["output_folder"], 
                self.hypercubes_dataframe["tile_name"]
            )
        }
        self._zarr_handles_data = {
            p: read_zarr(p, dtype=self.dtype)
            for p in self.paths
        }

    def _build_index(self) -> None:
        # convert df into a list of Python dicts
        self._index = self.hypercubes_dataframe.to_dict(orient="records")

    def _slice_hypercube(self, data_tensor, meta: Dict[str, Any]) -> Tuple[int]:
        t, c = slice(meta["time_start"], meta["time_start"] + meta["time_size"]), slice(0, meta["channel_size"])
        z = slice(meta["z_start"], meta["z_start"] + meta["cube_size"])
        y = slice(meta["y_start"], meta["y_start"] + meta["cube_size"])
        x = slice(meta["x_start"], meta["x_start"] + meta["cube_size"])
        return data_tensor[t, z, y, x, c].read().result()

    def _load_sample(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Read raw image crop into memory."""
        data_tensor = self._zarr_handles_data[
            os.path.join(meta["server_folder"], meta["output_folder"], meta["tile_name"])
        ]
        img = self._slice_hypercube(data_tensor, meta)
        return dict(meta=meta, image=img)

    def _collate(self, _data: Dict[str, Any]) -> DataSample:
        img_tensor = torch.from_numpy(_data["image"])

        if torch.isnan(img_tensor).all() or torch.isinf(img_tensor).all():
            raise ValueError(f"Invalid training data: {_data['meta']}")

        img_sample = ImageList(
            img_tensor,
            layout=self.input_layout,
            image_sizes=[img_tensor.shape]
        )
        
        sample = DataSample(metainfo=_data["meta"])
        sample.data_tensor = img_sample
        return sample