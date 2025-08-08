import torch
from torch.utils.data import Dataset
import numpy as np
import os

class BurgersDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.data_dir = data_dir
        self._load_data()

    def _load_data(self):
        npz_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        for file_name in npz_files:
            file_path = os.path.join(self.data_dir, file_name)
            data = np.load(file_path)
            time_series = data['DataGenerator-Mesh-1D-Internal']
            for i in range(time_series.shape[0] - 1):
                self.data.append((time_series[i], time_series[i+1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()