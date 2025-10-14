import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

from config import (
    UNROLL_STEPS,
    SPLIT,
    DATA_DIR,
    INPUT_SIZE,
    OUTPUT_SIZE
)

class BurgersDatasetBPTTsubdomain(Dataset):
    def __init__(self, file_paths, unroll_steps=UNROLL_STEPS, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE):
        self.data = []
        self.file_paths = file_paths
        self.unroll_steps = unroll_steps
        self.stride = np.min((self.unroll_steps // 2, 10))
        if self.stride == 0:
            self.stride = 1
        self.ghost_cells = input_size - output_size
        self.subdomain_size = input_size
        self.subdomains = 1
        self.subdomain_stride = self.ghost_cells + 1
        self._load_data()

    def _load_data(self):
        for file_path in self.file_paths:
            data = np.load(file_path)
            time_series = data['Solver-Mesh-1D-Internal']

            if self.unroll_steps == -1:  # full sequence
                self.unroll_steps = time_series.shape[0] - 1
                self.stride = 1

            if not time_series.shape[1] - self.ghost_cells <= self.subdomain_size:
                self.subdomains = np.floor((time_series.shape[1] - self.ghost_cells) / self.subdomain_stride).astype(int)
            else:
                self.subdomains = 1
                self.subdomain_stride = 0  # not used

            for i in range(0, time_series.shape[0] - self.unroll_steps, self.stride):
                for s in range(0, self.subdomains):
                    start_idx = s * self.subdomain_stride
                    end_idx = start_idx + self.subdomain_size
                    if end_idx > time_series.shape[1]:
                        end_idx = time_series.shape[1]
                        start_idx = end_idx - self.subdomain_size
                        break  # no more subdomains can be extracted

                    self.data.append((time_series[i, start_idx:end_idx], time_series[i+1:i+1+self.unroll_steps, start_idx:end_idx]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

class BurgersDatasetBPTT(Dataset):
    def __init__(self, file_paths, unroll_steps=UNROLL_STEPS):
        self.data = []
        self.file_paths = file_paths
        self.unroll_steps = unroll_steps
        self.stride = np.min((self.unroll_steps // 2, 10))
        self._load_data()

    def _load_data(self):
        for file_path in self.file_paths:
            data = np.load(file_path)
            time_series = data['Solver-Mesh-1D-Internal']
            if self.stride == 0:
                self.stride = 1

            if self.unroll_steps == -1:  # full sequence
                self.unroll_steps = time_series.shape[0] - 1
                self.stride = 1

            for i in range(0, time_series.shape[0] - self.unroll_steps, self.stride):
                self.data.append((time_series[i], time_series[i+1:i+1+self.unroll_steps]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

class BurgersDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []
        self.file_paths = file_paths
        self._load_data()

    def _load_data(self):
        for file_path in self.file_paths:
            data = np.load(file_path)
            time_series = data['Solver-Mesh-1D-Internal']
            for i in range(time_series.shape[0] - 1):
                self.data.append((time_series[i], time_series[i+1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
    
def train_split_files(data_dir=DATA_DIR, split=SPLIT, seed=42, file_ext='.npz'):
    """
    Returns train_files, val_files split from data_dir.
    """

    random.seed(seed)
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(file_ext)]
    random.shuffle(all_files)
    split_idx = int(split * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Found {len(all_files)} simulations.")
    print(f"Splitting into {len(train_files)} training and {len(val_files)} validation files.")
    return train_files, val_files