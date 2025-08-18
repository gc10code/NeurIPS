import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Optional, Tuple, Union
import numba as nb
import math
import gc

from src.models.rprop_mlp import RPropMLP
from src.utils.normalizer import DataNormalizer
from torch.utils.data import DataLoader
from src.utils.system_utils import setup_device
from src.utils.logging import lprint, LoggingLevels as ll

device = setup_device()

import numpy as np
import pandas as pd

from typing import Union, List, Optional, Tuple

def compute_wmae_weights(true_targets: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> torch.Tensor:
    """
    Compute weights for wMAE loss based on the number of valid samples and target ranges.

    Args:
        true_targets (List[Tuple[torch.Tensor, torch.Tensor, int]]): List of tuples containing
            (target tensor, target tensor, number of valid samples) for each target.

    Returns:
        torch.Tensor: Weights for each target, shape [T].
    """
    K = len(true_targets)
    
    # Initialize lists for scales and ranges
    inverse_sqrt_scales = []
    range_norms = []
    
    for target in true_targets:
        valid_samples = target[2]
        target_tensor = target[1]
        
        # Handle case where valid_samples is 0
        if valid_samples == 0:
            inverse_sqrt_scales.append(0.0)  # Assign zero weight to avoid division by zero
        else:
            inverse_sqrt_scales.append(1.0 / (valid_samples ** 0.5))
        
        # Compute range norm, handling case where max == min
        target_range = target_tensor.max() - target_tensor.min()
        if target_range == 0:
            range_norms.append(1.0)  # Avoid division by zero by setting default range
        else:
            range_norms.append(1.0 / target_range)
    
    # Convert to tensors
    inverse_sqrt_scales = torch.tensor(inverse_sqrt_scales)
    range_norms = torch.tensor(range_norms)
    
    # Normalize weights
    weight_normalization = inverse_sqrt_scales.sum()
    if weight_normalization == 0:
        # If all targets have zero valid samples, return equal weights
        weights = torch.ones(K) / K
    else:
        weights = K * range_norms * inverse_sqrt_scales / weight_normalization
    
    return weights

def wMAE_loss(
    predictions: torch.Tensor,
    targets: Union[torch.Tensor, List[torch.Tensor]],
    weights: torch.Tensor,
    valid_mask: Optional[List[torch.Tensor]] = None
) -> torch.Tensor:
    """
    Weighted Mean Absolute Error loss, computed only on real values based on validity mask.

    Args:
        predictions (torch.Tensor): Predicted values, shape [B, T], [B, 1], or [B] for single target.
        targets (torch.Tensor | List[torch.Tensor]): Ground truth values, either [B, T], [B, 1], [B], or list of [B, 1].
        weights (torch.Tensor): Weights for each target, shape [T] or [1] for single target.
        valid_mask (Optional[List[torch.Tensor]]): Validity mask for each target, each [B], where True indicates real values.

    Returns:
        torch.Tensor: Weighted MAE loss, averaged over valid entries.
    """
    if isinstance(targets, list):
        targets = torch.cat(targets, dim=-1)
    
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, targets {targets.shape}")

    absolute_error = torch.abs(predictions - targets)

    if valid_mask is not None:
        if isinstance(valid_mask, list):
            valid_mask = torch.stack(valid_mask, dim=-1)
        if valid_mask.shape != absolute_error.shape:
            raise ValueError(f"Shape mismatch: valid_mask {valid_mask.shape}, absolute_error {absolute_error.shape}")
        absolute_error = absolute_error * valid_mask.float()
        valid_counts = valid_mask.float().sum(dim=0).clamp(min=1)
    else:
        valid_counts = torch.tensor([predictions.shape[0]] * predictions.shape[-1], device=predictions.device)

    if weights.shape[0] != absolute_error.shape[-1]:
        raise ValueError(f"Shape mismatch: weights {weights.shape}, absolute_error {absolute_error.shape}")
    weighted_error = absolute_error * weights
    total_weighted_error = weighted_error.sum()
    total_valid = valid_counts.sum()
    wmae = total_weighted_error / total_valid.clamp(min=1)

    return wmae

def evaluate_model(model: RPropMLP, data_loader: DataLoader, device: torch.device,
                  normalizer: Optional[DataNormalizer] = None,
                  metrics: List[str] = ['mse', 'rmse', 'mae', 'r2', 'wmae', 'mape', 'max_error', 'mean_error', 'std_error']) -> Dict[str, float]:
    try:
        model.eval()
        y_true_all = []
        y_pred_all = []
        
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                
                if model.problem_type == 'regression':
                    outputs = outputs.squeeze()
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    elif outputs.dim() > 1 and outputs.size(1) == 1:
                        outputs = outputs.squeeze(1)
                
                y_pred_all.append(outputs.cpu().numpy())
                y_true_all.append(y.cpu().numpy())
        
        y_pred = np.concatenate(y_pred_all, axis=0)
        y_true = np.concatenate(y_true_all, axis=0)
        
        if normalizer is not None:
            y_pred = normalizer.inverse_transform_y(y_pred)
            y_true = normalizer.inverse_transform_y(y_true)
        
        result_metrics = {}
        for metric in metrics:
            try:
                if metric == 'mse':
                    result_metrics['mse'] = mean_squared_error(y_true, y_pred)
                elif metric == 'rmse':
                    result_metrics['rmse'] = np.sqrt(result_metrics.get('mse', mean_squared_error(y_true, y_pred)))
                elif metric == 'mae':
                    result_metrics['mae'] = mean_absolute_error(y_true, y_pred)
                elif metric == 'r2':
                    result_metrics['r2'] = r2_score(y_true, y_pred)
                elif metric == 'mape':
                    if np.mean(np.abs(y_true)) > 1e-10:
                        result_metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
                    else:
                        result_metrics['mape'] = float('inf')
                elif metric == 'max_error':
                    result_metrics['max_error'] = np.max(np.abs(y_true - y_pred))
                elif metric == 'mean_error':
                    result_metrics['mean_error'] = np.mean(y_true - y_pred)
                elif metric == 'std_error':
                    result_metrics['std_error'] = np.std(y_true - y_pred)
            except Exception as e:
                lprint(ll.WARN,  f"Failed to compute metric {metric}: {str(e)}")
                result_metrics[metric] = float('nan')
        
        return result_metrics
    
    except Exception as e:
        lprint(ll.ERROR,  f"Model evaluation failed: {str(e)}")
        raise