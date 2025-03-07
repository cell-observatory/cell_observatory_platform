from torch.utils.data import Dataset

from .data_config import DataConfig
from .fish_database import FishDatabase


class PretrainingDataset(Dataset):
    def __init__(self, batch_config: DataConfig = None, transform = None):
        if batch_config is None:
            batch_config = DataConfig()

        self.batch_config = batch_config
        self.fds = FishDatabase(batch_config = batch_config)
        self.transform = transform

    def __len__(self):
        return len(self.fds)

    def __getitem__(self, idx):
        item = self.fds[idx]
        if self.transform:
            item = self.transform(item)
        return item