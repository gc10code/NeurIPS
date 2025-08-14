import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Optional, Tuple
import numba as nb
import math

from src.models.rprop_mlp import RPropMLP
from src.utils.normalizer import DataNormalizer
from torch.utils.data import DataLoader
from src.utils.system_utils import setup_device
from src.utils.logging import lprint, LoggingLevels as ll

device = setup_device()

import numpy as np
import pandas as pd

def compute_wmae_weights(true_targets: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> torch.Tensor:
    K = len(true_targets)
    
    inverse_sqrt_scales = torch.tensor([1.0 * (target[2] ** 0.5) for target in true_targets])
    range_norms = torch.tensor([(1.0 / (target[1].max() - target[1].min())) for target in true_targets])
    weight_normalization = inverse_sqrt_scales.sum()
    # it is product element * element, not vectorial product!
    weights = K * range_norms * inverse_sqrt_scales / weight_normalization
    
    return weights

# Weighted MAE loss function using contest formula
def wMAE_loss(predictions: List[torch.Tensor], targets: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    predictions = torch.stack(predictions)
    targets = torch.stack(targets)
    # Compute weighted MAE: sum(weights * |predictions - targets|)
    wmae = torch.sum(weights.view(-1, 1) * torch.abs(predictions - targets))
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