import numpy as np
from typing import Dict, Any, Union, List, Tuple
import logging

from src.utils.activations import validate_activations
from src.utils.logging import lprint, LoggingLevels as ll

class ConfigValidator:
    """Validates configuration parameters."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration dictionary."""
        required_fields = [
            'training_type', 'learning_rate', 'batch_size', 'max_epochs', 'k_folds',
            'patience', 'min_delta', 'shuffle', 'target_error', 'gradient_clipping',
            'input_normalization', 'output_normalization', 'seed', 'num_workers',
            'pin_memory', 'validation_split', 'dropout_prob', 'batch_norm',
            'output_size', 'problem_type', 'input_dir', 'output_dir',
            'memory_check_interval', 'checkpoint_interval'
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")
        
        # Training parameters
        if config['training_type'] not in ['fold', 'split']:
            raise ValueError(f"training_type must be 'fold' or 'split', got {config['training_type']}")
        ConfigValidator._validate_positive_number(config['learning_rate'], 'learning_rate')
        ConfigValidator._validate_positive_integer(config['batch_size'], 'batch_size')
        ConfigValidator._validate_positive_integer(config['max_epochs'], 'max_epochs')
        ConfigValidator._validate_positive_integer(config['k_folds'], 'k_folds')
        ConfigValidator._validate_positive_integer(config['num_workers'], 'num_workers')
        ConfigValidator._validate_non_negative_number(config['validation_split'], 'validation_split')
        
        if config['k_folds'] < 2:
            raise ValueError("k_folds must be at least 2")
        if config['validation_split'] > 0.5:
            raise ValueError("validation_split should not exceed 0.5")
        
        # Network parameters
        if config['hidden_layers'] is not None:
            if not isinstance(config['hidden_layers'], (list, tuple)) or not config['hidden_layers'] or any(h <= 0 for h in config['hidden_layers']):
                raise ValueError(f"hidden_layers must be a non-empty list of positive integers, got {config['hidden_layers']}")
        if config['activations'] is not None:
            validate_activations([config['activations']])
        ConfigValidator._validate_non_negative_number(config['dropout_prob'], 'dropout_prob')
        if config['dropout_prob'] > 0.5:
            raise ValueError(f"dropout_prob must be in [0,0.5], got {config['dropout_prob']}")
        if config['problem_type'] not in ['regression', 'classification']:
            raise ValueError(f"problem_type must be 'regression' or 'classification', got {config['problem_type']}")
    
    @staticmethod
    def _validate_positive_number(value: Union[int, float], name: str) -> None:
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a positive number, got {value}")
    
    @staticmethod
    def _validate_positive_integer(value: int, name: str) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")
    
    @staticmethod
    def _validate_non_negative_number(value: Union[int, float], name: str) -> None:
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a non-negative number, got {value}")

class DataValidator:
    """Validates input data with additional checks for outliers and correlations."""
    
    @staticmethod
    def validate_data(X: np.ndarray, y: np.ndarray, check_outliers: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        
        if y.ndim != 1 and not (y.ndim == 2 and y.shape[1] >= 1):
            raise ValueError(f"y must be 1D or 2D with multiple outputs, got shape {y.shape}")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have same number of samples: {len(X)} vs {len(y)}")
        
        if len(X) == 0:
            raise ValueError("Input data is empty")
        
        X_invalid = np.isnan(X) | np.isinf(X)
        y_invalid = np.isnan(y) | np.isinf(y)
        
        if np.any(X_invalid) or np.any(y_invalid):
            lprint(ll.WARN,  f"Found {np.sum(X_invalid)} invalid values in X and {np.sum(y_invalid)} in y")
            valid_mask = ~(np.any(X_invalid, axis=1) | np.any(y_invalid, axis=1))
            if np.sum(valid_mask) == 0:
                raise ValueError("No valid samples remaining after removing NaN/inf values")
            
            X = X[valid_mask]
            y = y[valid_mask]
            lprint(ll.INFO,  f"Removed invalid samples, remaining: {len(X)}")
        
        X = np.clip(X, -1e10, 1e10)
        y = np.clip(y, -1e10, 1e10)
        
        if check_outliers:
            z_scores = np.abs((X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10))
            if np.any(z_scores > 5):
                lprint(ll.WARN,  "Detected potential outliers in X (z-score > 5)")
        
        corr_matrix = np.corrcoef(X, rowvar=False)
        if np.any(np.abs(corr_matrix - np.eye(corr_matrix.shape[0])) > 0.95):
            lprint(ll.WARN,  "High correlation (>0.95) detected between some features")
        
        return X, y