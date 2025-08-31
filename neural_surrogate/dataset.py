import torch
from torch.utils.data import Dataset
import numpy as np
import os

class BurgersDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []
        self.file_paths = file_paths
        self._load_data()

    def _load_data(self):
        for file_path in self.file_paths:
            data = np.load(file_path)
            time_series = data['DataGenerator-Mesh-1D-Internal']
            for i in range(time_series.shape[0] - 1):
                self.data.append((time_series[i], time_series[i+1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()