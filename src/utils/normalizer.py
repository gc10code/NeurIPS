import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional, Tuple, Union
import logging

from src.utils.validators import DataValidator
from src.utils.logging import lprint, LoggingLevels as ll

class DataNormalizer:
    """Handles data normalization with checks for pre-normalized data."""
    
    def __init__(self, input_norm: str = 'none', output_norm: str = 'none'):
        valid_norms = ['none', 'zscore', 'minmax']
        if input_norm not in valid_norms or output_norm not in valid_norms:
            raise ValueError(f"Normalization must be one of: {valid_norms}")
        
        self.input_norm = input_norm
        self.output_norm = output_norm
        self.scaler_X = None
        self.scaler_y = None
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = DataValidator.validate_data(X, y)
        
        if self.input_norm != 'none' and self._is_normalized(X):
            lprint(ll.INFO,  "Input data appears already normalized, skipping normalization")
            self.input_norm = 'none'
        
        if self.input_norm == 'zscore':
            self.scaler_X = StandardScaler()
            X = self.scaler_X.fit_transform(X)
        elif self.input_norm == 'minmax':
            self.scaler_X = MinMaxScaler()
            X = self.scaler_X.fit_transform(X)
        
        if self.output_norm != 'none' and self._is_normalized(y):
            lprint(ll.INFO,  "Output data appears already normalized, skipping normalization")
            self.output_norm = 'none'
        
        if self.output_norm == 'zscore':
            self.scaler_y = StandardScaler()
            if y.ndim == 1:
                y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            else:
                y = self.scaler_y.fit_transform(y)
        elif self.output_norm == 'minmax':
            self.scaler_y = MinMaxScaler()
            if y.ndim == 1:
                y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            else:
                y = self.scaler_y.fit_transform(y)
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def _is_normalized(self, data: np.ndarray) -> bool:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return np.all(np.abs(mean) < 1e-2) and np.all(np.abs(std - 1) < 1e-2)
    
    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.scaler_X is not None:
            X = self.scaler_X.transform(X)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        if y is not None:
            if self.scaler_y is not None:
                y = self.scaler_y.transform(y.reshape(-1, y.shape[1])).reshape(y.shape)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            return X_tensor, y_tensor
        
        return X_tensor
    
    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        if self.scaler_y is not None:
            return self.scaler_y.inverse_transform(y.reshape(-1, y.shape[1])).reshape(y.shape)
        return y