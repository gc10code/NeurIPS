import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional, Tuple, Union
import logging

from src.utils.validators import DataValidator
from src.utils.logging import lprint, LoggingLevels as ll

from src.utils.logging import lprint, LoggingLevels as ll
from typing import Tuple, Optional, Union
import torch

class DataNormalizer:
    """Handles data normalization with checks for pre-normalized data."""
    
    def __init__(self, input_norm: str = 'none', output_norm: str = 'none'):
        valid_norms = ['none', 'zscore', 'minmax']
        if input_norm not in valid_norms or output_norm not in valid_norms:
            raise ValueError(f"Normalization must be one of: {valid_norms}")
        
        self.input_norm = input_norm
        self.output_norm = output_norm
        self.X_mean = None
        self.X_std = None
        self.X_min = None
        self.X_max = None
        self.y_mean = None
        self.y_std = None
        self.y_min = None
        self.y_max = None
    
    def fit_transform(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit and transform input and output tensors with specified normalization."""
        X, y = DataValidator.validate_data(X, y)  # Assumes DataValidator supports torch.Tensor
        
        if self.input_norm != 'none' and self._is_normalized(X):
            lprint(ll.INFO, "Input data appears already normalized, skipping normalization")
            self.input_norm = 'none'
        
        if self.input_norm == 'zscore':
            self.X_mean = torch.mean(X, dim=0)
            self.X_std = torch.std(X, dim=0) + 1e-10  # Prevent division by zero
            X = (X - self.X_mean) / self.X_std
        elif self.input_norm == 'minmax':
            self.X_min = torch.min(X, dim=0).values
            self.X_max = torch.max(X, dim=0).values
            X = (X - self.X_min) / (self.X_max - self.X_min + 1e-10)  # Prevent division by zero
        
        if self.output_norm != 'none' and self._is_normalized(y):
            lprint(ll.INFO, "Output data appears already normalized, skipping normalization")
            self.output_norm = 'none'
        
        if self.output_norm == 'zscore':
            self.y_mean = torch.mean(y, dim=0)
            self.y_std = torch.std(y, dim=0) + 1e-10
            y = (y - self.y_mean) / self.y_std
        elif self.output_norm == 'minmax':
            self.y_min = torch.min(y, dim=0).values
            self.y_max = torch.max(y, dim=0).values
            y = (y - self.y_min) / (self.y_max - self.y_min + 1e-10)
        
        return X.to(dtype=torch.float32), y.to(dtype=torch.float32)
    
    def _is_normalized(self, data: torch.Tensor) -> bool:
        """Check if data is already normalized (mean ~ 0, std ~ 1)."""
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        return torch.all(torch.abs(mean) < 1e-2) and torch.all(torch.abs(std - 1) < 1e-2)
    
    def transform(self, X: torch.Tensor, y: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Transform new data using fitted normalization parameters."""
        if self.input_norm == 'zscore' and self.X_mean is not None and self.X_std is not None:
            X = (X - self.X_mean) / self.X_std
        elif self.input_norm == 'minmax' and self.X_min is not None and self.X_max is not None:
            X = (X - self.X_min) / (self.X_max - self.X_min + 1e-10)
        
        X_tensor = X.to(dtype=torch.float32)
        
        if y is not None:
            if self.output_norm == 'zscore' and self.y_mean is not None and self.y_std is not None:
                y = (y - self.y_mean) / self.y_std
            elif self.output_norm == 'minmax' and self.y_min is not None and self.y_max is not None:
                y = (y - self.y_min) / (self.y_max - self.y_min + 1e-10)
            y_tensor = y.to(dtype=torch.float32)
            return X_tensor, y_tensor
        
        return X_tensor
    
    def inverse_transform_y(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse transform output data."""
        if self.output_norm == 'zscore' and self.y_mean is not None and self.y_std is not None:
            y = y * self.y_std + self.y_mean
        elif self.output_norm == 'minmax' and self.y_min is not None and self.y_max is not None:
            y = y * (self.y_max - self.y_min) + self.y_min
        return y