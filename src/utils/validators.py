import torch
from typing import Dict, Any, Union, List, Tuple
import logging

from src.utils.activations import validate_activations, ACTIVATION_FUNCTIONS
from src.utils.logging import lprint, LoggingLevels as ll

from dataclasses import asdict

def validate_activations(activations: List[str]) -> None:
    """Validate activation functions."""
    valid_activations = list(ACTIVATION_FUNCTIONS.keys())
    for act in activations:
        if isinstance(act, (list, tuple)):
            validate_activations(act)
        elif act is not None and act not in valid_activations:
            lprint(ll.ERROR, f"Invalid activation function: {act}. Must be one of {valid_activations}")
            raise ValueError(f"Invalid activation function: {act}")

class ConfigValidator:
    """Validates configuration parameters for RPropMLP and FusionModel."""

    @staticmethod
    def validate_config(config: Dict[str, Any], config_type: str = 'rprop') -> None:
        """
        Validate configuration dictionary for RPropMLP or FusionModel.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            config_type (str): Type of configuration ('rprop' for RPropMLP, 'fusion' for HRMConfig).

        Raises:
            ValueError: If any required field is missing or invalid.
        """
        # Common required fields for both RPropMLP and FusionModel
        common_required_fields = [
            'training_type', 'learning_rate', 'batch_size', 'max_epochs', 'k_folds',
            'patience', 'min_delta', 'shuffle', 'target_error', 'gradient_clipping',
            'input_normalization', 'output_normalization', 'seed', 'num_workers',
            'pin_memory', 'validation_split', 'dropout_prob', 'batch_norm',
            'output_size', 'problem_type', 'input_dir', 'output_dir',
            'memory_check_interval', 'checkpoint_interval'
        ]

        # Additional required fields for HRMConfig (FusionModel)
        hrm_required_fields = [
            'input_size', 'num_targets', 'low_hidden', 'high_hidden', 
            'act_eps', 'max_high_steps', 'max_low_steps'
        ]

        # Determine required fields based on config_type
        required_fields = common_required_fields
        if config_type == 'fusion':
            required_fields += hrm_required_fields
        if type(config) != dict:
            config = asdict(config)
            
        # Check for missing fields
        for field in required_fields:
            if field not in config:
                lprint(ll.ERROR, f"Missing required config field for {config_type}: {field}")
                raise ValueError(f"Missing required config field: {field}")

        # Validate common fields
        if config['training_type'] not in ['fold', 'split']:
            lprint(ll.ERROR, f"training_type must be 'fold' or 'split', got {config['training_type']}")
            raise ValueError(f"training_type must be 'fold' or 'split', got {config['training_type']}")
        
        ConfigValidator._validate_positive_number(config['learning_rate'], 'learning_rate')
        ConfigValidator._validate_positive_integer(config['batch_size'], 'batch_size')
        ConfigValidator._validate_positive_integer(config['max_epochs'], 'max_epochs')
        ConfigValidator._validate_positive_integer(config['k_folds'], 'k_folds')
        ConfigValidator._validate_positive_integer(config['num_workers'], 'num_workers')
        ConfigValidator._validate_non_negative_number(config['validation_split'], 'validation_split')
        
        if config['k_folds'] < 2:
            lprint(ll.ERROR, "k_folds must be at least 2")
            raise ValueError("k_folds must be at least 2")
        if config['validation_split'] > 0.5:
            lprint(ll.ERROR, "validation_split should not exceed 0.5")
            raise ValueError("validation_split should not exceed 0.5")
        
        # Validate network parameters
        if 'hidden_layers' in config and config['hidden_layers'] is not None:
            if not isinstance(config['hidden_layers'], (list, tuple)) or not config['hidden_layers'] or any(h <= 0 for h in config['hidden_layers']):
                lprint(ll.ERROR, f"hidden_layers must be a non-empty list of positive integers, got {config['hidden_layers']}")
                raise ValueError(f"hidden_layers must be a non-empty list of positive integers, got {config['hidden_layers']}")
        
        if 'activations' in config and config['activations'] is not None:
            validate_activations([config['activations']])
        
        ConfigValidator._validate_non_negative_number(config['dropout_prob'], 'dropout_prob')
        if config['dropout_prob'] > 0.5:
            lprint(ll.ERROR, f"dropout_prob must be in [0,0.5], got {config['dropout_prob']}")
            raise ValueError(f"dropout_prob must be in [0,0.5], got {config['dropout_prob']}")
        
        if config['problem_type'] not in ['regression', 'classification']:
            lprint(ll.ERROR, f"problem_type must be 'regression' or 'classification', got {config['problem_type']}")
            raise ValueError(f"problem_type must be 'regression' or 'classification', got {config['problem_type']}")

        # Validate HRMConfig-specific fields (for FusionModel)
        if config_type == 'fusion':
            ConfigValidator._validate_positive_integer(config['input_size'], 'input_size')
            ConfigValidator._validate_positive_integer(config['num_targets'], 'num_targets')
            ConfigValidator._validate_positive_integer(config['low_hidden'], 'low_hidden')
            ConfigValidator._validate_positive_integer(config['high_hidden'], 'high_hidden')
            ConfigValidator._validate_positive_number(config['act_eps'], 'act_eps')
            ConfigValidator._validate_positive_integer(config['max_high_steps'], 'max_high_steps')
            ConfigValidator._validate_positive_integer(config['max_low_steps'], 'max_low_steps')

            if config['low_hidden'] < 16:
                lprint(ll.WARN, f"low_hidden ({config['low_hidden']}) is small, consider increasing for better capacity")
            if config['high_hidden'] < config['low_hidden']:
                lprint(ll.WARN, f"high_hidden ({config['high_hidden']}) should be at least as large as low_hidden ({config['low_hidden']})")
            if config['act_eps'] > 0.5:
                lprint(ll.ERROR, f"act_eps must be in (0,0.5], got {config['act_eps']}")
                raise ValueError(f"act_eps must be in (0,0.5], got {config['act_eps']}")
            if config['max_high_steps'] < 1 or config['max_high_steps'] > 20:
                lprint(ll.ERROR, f"max_high_steps must be in [1,10], got {config['max_high_steps']}")
                raise ValueError(f"max_high_steps must be in [1,10], got {config['max_high_steps']}")
            if config['max_low_steps'] < 1 or config['max_low_steps'] > 20:
                lprint(ll.ERROR, f"max_low_steps must be in [1,10], got {config['max_low_steps']}")
                raise ValueError(f"max_low_steps must be in [1,10], got {config['max_low_steps']}")

    @staticmethod
    def _validate_positive_number(value: Union[int, float], name: str) -> None:
        if not isinstance(value, (int, float)) or value <= 0:
            lprint(ll.ERROR, f"{name} must be a positive number, got {value}")
            raise ValueError(f"{name} must be a positive number, got {value}")

    @staticmethod
    def _validate_positive_integer(value: int, name: str) -> None:
        if not isinstance(value, int) or value <= 0:
            lprint(ll.ERROR, f"{name} must be a positive integer, got {value}")
            raise ValueError(f"{name} must be a positive integer, got {value}")

    @staticmethod
    def _validate_non_negative_number(value: Union[int, float], name: str) -> None:
        if not isinstance(value, (int, float)) or value < 0:
            lprint(ll.ERROR, f"{name} must be a non-negative number, got {value}")
            raise ValueError(f"{name} must be a non-negative number, got {value}")


class DataValidator:
    """Validates input data with additional checks for outliers and correlations."""
    
    @staticmethod
    def validate_data(X: torch.Tensor, y: torch.Tensor, check_outliers: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate input tensors, checking for shape, NaN/inf, outliers, and correlations."""
        if not isinstance(X, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("X and y must be PyTorch tensors")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D tensor, got shape {X.shape}")
        
        if y.ndim != 1 and not (y.ndim == 2 and y.shape[1] >= 1):
            raise ValueError(f"y must be 1D or 2D with multiple outputs, got shape {y.shape}")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have same number of samples: {len(X)} vs {len(y)}")
        
        if len(X) == 0:
            raise ValueError("Input data is empty")
        
        X_invalid = torch.isnan(X) | torch.isinf(X)
        y_invalid = torch.isnan(y) | torch.isinf(y)
        
        if torch.any(X_invalid) or torch.any(y_invalid):
            lprint(ll.WARN, f"Found {torch.sum(X_invalid)} invalid values in X and {torch.sum(y_invalid)} in y")
            valid_mask = ~(torch.any(X_invalid, dim=1) | torch.any(y_invalid, dim=1))
            if torch.sum(valid_mask) == 0:
                raise ValueError("No valid samples remaining after removing NaN/inf values")
            
            X = X[valid_mask]
            y = y[valid_mask]
            lprint(ll.INFO, f"Removed invalid samples, remaining: {len(X)}")
        
        X = torch.clamp(X, min=-1e10, max=1e10)
        y = torch.clamp(y, min=-1e10, max=1e10)
        
        if check_outliers:
            z_scores = torch.abs((X - torch.mean(X, dim=0)) / (torch.std(X, dim=0) + 1e-10))
            if torch.any(z_scores > 5):
                lprint(ll.WARN, "Detected potential outliers in X (z-score > 5)")
        
        # Compute correlation matrix
        corr_matrix = torch.corrcoef(X)  # PyTorch 2.0+; assumes X is 2D
        eye = torch.eye(corr_matrix.shape[0], device=corr_matrix.device)
        if torch.any(torch.abs(corr_matrix - eye) > 0.95):
            lprint(ll.WARN, "High correlation (>0.95) detected between some features")
        
        return X, y