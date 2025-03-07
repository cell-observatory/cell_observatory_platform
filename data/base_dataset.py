from torch.utils.data import Dataset

from .data_config import DataConfig
from .fish_database import FishDatabase


class BaseDataset(Dataset):
    def __init__(self,
                 batch_config: DataConfig = None,
                 transform = None):
        """
        PretrainingDataset constructor.
        Args:
            batch_config: DataConfig object contains the shape information for a single batch. If None, default will be used.
            transform: torchvision.transforms.Compose object. Default is None.
        """
        # If batch_config is not provided, use default DataConfig.
        if batch_config is None:
            batch_config = DataConfig()
        self.batch_config = batch_config

        # Augmentations to perform on the data
        self.transform = transform

        # FishDatabase object can slice the underlying raw data using the batch_config information
        self.fds = FishDatabase(batch_config = batch_config)

    def __len__(self):
        return len(self.fds)

    def __getitem__(self, idx):
        item = self.fds[idx]
        if self.transform:
            item = self.transform(item)
        return item