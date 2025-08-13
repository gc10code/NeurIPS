import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Optional

import logging

from src.models.rprop_mlp import RPropMLP
from src.utils.normalizer import DataNormalizer
from torch.utils.data import DataLoader
from src.utils.system_utils import setup_device
from src.utils.logging import lprint, LoggingLevels as ll

device = setup_device()

# Function to compute wMAE weights based on the contest formula
def compute_wmae_weights(true_targets: List[torch.Tensor]) -> torch.Tensor:
    # Number of available values for each property (non-NaN values)
    num_values = [torch.sum(~torch.isnan(target)).item() for target in true_targets]    
    # Compute value ranges (max - min) for each property
    ranges = []
    for target in true_targets:
        valid_values = target[~torch.isnan(target)]
        if len(valid_values) > 0:
            range_t = valid_values.max().item() - valid_values.min().item()
            ranges.append(range_t if range_t > 0 else 1.0)  # Avoid division by zero
        else:
            ranges.append(1.0)  # Default range if no valid values

    # Compute unnormalized weights: 1 / (num_values * sqrt(num_values) * range)
    unnormalized_weights = [
        1.0 / (n * np.sqrt(n) * r) if n > 0 else 1.0
        for n, r in zip(num_values, ranges)
    ]
    # Normalize weights so their sum equals the number of properties (5)
    total = sum(unnormalized_weights)
    weights = [5.0 * w / total if total > 0 else 1.0 / 5.0 for w in unnormalized_weights]
    return torch.tensor(weights, dtype=torch.float64).to(device)

# Weighted MAE loss function using contest formula
def wMAE_loss(predictions: List[torch.Tensor], targets: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    mae = sum(w * torch.mean(torch.abs(pred - target)) for w, pred, target in zip(weights, predictions, targets))
    return mae

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
                elif metric == 'wmae':
                    weights = torch.ones_like(torch.tensor(y_true))
                    result_metrics['wmae'] = wMAE_loss(torch.tensor(y_true), torch.tensor(y_pred), weights).item()
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