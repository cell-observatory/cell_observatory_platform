import os
from typing import Dict, Any

import torch
from torch.utils.data import get_worker_info

from data.io import read_zarr
from data.structures.data_sample import DataSample
from data.structures.image_list import ImageList, cat_image_lists
from data.datasets.base_dataset import BaseDataset, default_collate


def collate_pretrain_dataset(samples: list["DataSample"]) -> "DataSample":
    metainfo = default_collate([s.metainfo for s in samples])
    batch = DataSample(metainfo=metainfo)    
    batched_img = cat_image_lists(image_lists=[s.data_tensor for s in samples])
    batch.data_tensor = batched_img    
    return batch.to_dict()


class PretrainDataset(BaseDataset):
    """
    Dataset for pretraining.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zarr_handles_data = {}

    def worker_init_fn(self, worker_id):
        worker_info = get_worker_info()
        # re-open handles in this worker only 
        # important to pass to dataloader
        paths = {
            os.path.join(sf, of, tn)
            for sf, of, tn in 
            zip(self.db.data_table["server_folder"], 
                self.db.data_table["output_folder"], 
                self.db.data_table["tile_name"]
            )
        }
        self._zarr_handles_data = {
            p: read_zarr(p)
            for p in paths
        }

    def _process_tables(self) -> None:
        # no-op for now, consider potentially filtering
        # data_table based on some criteria here or
        # removing method
        return self.db.data_table

    def _build_index(self) -> None:
        # convert df into a list of Python dicts
        self._index = self.db.data_table.to_dict(orient="records")

    def _load_sample(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Read raw image crop into memory."""
        data_tensor = self._zarr_handles_data[os.path.join(meta["server_folder"], 
                                                           meta["output_folder"], 
                                                           meta["tile_name"])]
        
        t, c = slice(meta["time_start"], meta["time_start"] + meta["time_size"]), slice(0, meta["channel_size"])
        z = slice(meta["z_start"], meta["z_start"] + meta["cube_size"])
        y = slice(meta["y_start"], meta["y_start"] + meta["cube_size"])
        x = slice(meta["x_start"], meta["x_start"] + meta["cube_size"])

        img = data_tensor[t, z, y, x, c].read().result()  
        return dict(meta=meta, image=img)

    def _collate(self, _data: Dict[str, Any]) -> DataSample:
        img_tensor = torch.tensor(_data["image"], dtype=torch.float32).clone() 
        img_sample = ImageList(img_tensor,
                                layout=self.layout,
                                image_sizes=[img_tensor.shape])
        
        sample = DataSample(metainfo=_data["meta"])
        sample.data_tensor = img_sample
        return sample