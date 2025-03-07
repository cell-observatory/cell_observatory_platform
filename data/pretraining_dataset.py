from torch.utils.data import Dataset

from .data_config import DataConfig
from .fish_database import FishDatabase


class PretrainingDataset(Dataset):
    def __init__(self, data_config: DataConfig = None, transform = None):
        if data_config is None:
            data_config = DataConfig()

        self.data_config = data_config
        self.fds = FishDatabase(data_config = data_config)
        self.transform = transform

    def __len__(self):
        return len(self.fds)

    def __getitem__(self, idx):
        item = self.fds[idx]
        if self.transform:
            item = self.transform(item)
        return item