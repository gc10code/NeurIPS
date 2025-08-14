import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from src.utils.logging import lprint, LoggingLevels as ll

class DynamicFusionDataset(Dataset):
    def __init__(self, metadata_file: str, descriptor_files: list[str], targets: list[str]):
        """
        Dynamic dataset using metadata.tsv to map samples across descriptor files.

        Args:
            metadata_file (str): Path to metadata.tsv.
            descriptor_files (list[str]): List of paths to descriptor_{target}.tsv files.
            targets (list[str]): List of targets (e.g., ["Density", "FFV", "Rg", "Tc", "Tg"]).
        """
        self.targets = targets
        self.n_targets = len(targets)
        
        # Load metadata.tsv
        metadata = np.loadtxt(metadata_file, delimiter='\t', dtype=str, skiprows=1)
        self.sample_ids = metadata[:, 0]
        self.row_indices = metadata[:, 1:].astype(float)  # Row indices for each target
        
        # Mask for valid targets (rows not equal to -1)
        self.valid_mask = self.row_indices != -1  # [n_samples, n_targets]
        
        # Load descriptors and targets
        self.X_data = []
        self.y_data = []
        self.n_features = None
        for target, descriptor_file in zip(targets, descriptor_files):
            if not Path(descriptor_file).exists():
                raise FileNotFoundError(f"Descriptor file not found: {descriptor_file}")
            data = np.loadtxt(descriptor_file, delimiter='\t')
            n_features = data.shape[1] - 1  # Exclude target column
            if self.n_features is None:
                self.n_features = n_features
            elif n_features != self.n_features:
                raise ValueError(f"Feature mismatch: {target} has {n_features} features, expected {self.n_features}")
            self.X_data.append(torch.tensor(data[:, :-1], dtype=torch.float32))  # [n_samples, 614]
            self.y_data.append(torch.tensor(data[:, -1], dtype=torch.float32).unsqueeze(-1))  # [n_samples, 1]
        
        lprint(ll.INFO, f"DynamicFusionDataset initialized with {len(self.sample_ids)} samples, {self.n_features} features")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        inputs = []
        targets = []
        mask = torch.tensor(self.valid_mask[idx], dtype=torch.bool)  # [n_targets]
        
        for i, target in enumerate(self.targets):
            if self.valid_mask[idx, i]:
                row_idx = int(self.row_indices[idx, i])
                inputs.append(self.X_data[i][row_idx])  # [614]
                targets.append(self.y_data[i][row_idx])  # [1]
            else:
                inputs.append(torch.zeros(self.n_features, dtype=torch.float32))  # Placeholder
                targets.append(torch.zeros(1, dtype=torch.float32))  # Placeholder
        
        return torch.stack(inputs), torch.stack(targets), mask  # [5, 614], [5, 1], [5]