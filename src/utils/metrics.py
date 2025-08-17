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

def compute_wmae_weights(true_targets: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> torch.Tensor:
    K = len(true_targets)
    
    inverse_sqrt_scales = torch.tensor([1.0 * (target[2] ** 0.5) for target in true_targets])
    range_norms = torch.tensor([(1.0 / (target[1].max() - target[1].min())) for target in true_targets])
    weight_normalization = inverse_sqrt_scales.sum()
    # it is product element * element, not vectorial product!
    weights = K * range_norms * inverse_sqrt_scales / weight_normalization
    
    return weights


def wMAE_loss(
    predictions: torch.Tensor,
    targets: Union[torch.Tensor ,List[torch.Tensor]],
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
    #lprint(ll.REPORT, f"Mask {valid_mask.di}")
    #lprint(ll.REPORT, f"targets {targets}")
    #lprint(ll.REPORT, f"weights {weights}")
    #lprint(ll.REPORT, f"prediction {predictions}")
    try:
        # Ensure predictions is a tensor
        if not isinstance(predictions, torch.Tensor):
            lprint(ll.ERROR, f"Predictions must be a tensor, got {type(predictions)}")
            raise ValueError("Invalid predictions format")

        # Ensure predictions and targets are on the same device
        device = predictions.device
        if isinstance(targets, torch.Tensor):
            targets = targets.to(device)
        else:
            targets = [t.to(device) for t in targets]
        weights = weights.to(device)

        # Handle single-target case
        if predictions.ndim == 1 or (predictions.ndim == 2 and predictions.shape[1] == 1):
            # Normalize to [B]
            B = predictions.shape[0]
            T = 1
            if predictions.ndim == 2:
                predictions = predictions.squeeze(1)  # [B, 1] -> [B]
            if isinstance(targets, torch.Tensor):
                if targets.ndim == 1:
                    targets = targets  # [B]
                elif targets.ndim == 2 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)  # [B, 1] -> [B]
                else:
                    lprint(ll.ERROR, f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
                    raise ValueError("Shape mismatch between predictions and targets")
            else:
                if len(targets) != 1:
                    lprint(ll.ERROR, f"Expected 1 target for single-target case, got {len(targets)}")
                    raise ValueError("Invalid number of targets")
                targets = targets[0].squeeze(-1)  # [B, 1] -> [B]
            weights = weights[0:1]  # Ensure weights is [1]
            valid_mask = [valid_mask[0]] if valid_mask is not None else None
        elif predictions.ndim == 2:
            # Multi-target case
            B, T = predictions.shape
            if isinstance(targets, torch.Tensor):
                if targets.shape != (B, T):
                    lprint(ll.ERROR, f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
                    raise ValueError("Shape mismatch between predictions and targets")
            else:
                # Convert list of [B, 1] to [B, T]
                targets = torch.stack([t.squeeze(-1) for t in targets], dim=1)  # [B, T]
                if targets.shape != (B, T):
                    lprint(ll.ERROR, f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
                    raise ValueError("Shape mismatch between predictions and targets")
        else:
            lprint(ll.ERROR, f"Invalid predictions shape {predictions.shape}")
            raise ValueError("Invalid predictions shape")

        # Initialize validity mask if not provided
        if valid_mask is None:
            if T == 1:
                valid_mask = [~torch.isnan(targets)] if predictions.ndim == 1 else [~torch.isnan(targets.squeeze(-1))]
            else:
                valid_mask = [~torch.isnan(targets[:, i]) for i in range(T)]
            lprint(ll.WARN, "No valid_mask provided, assuming non-NaN targets are valid")
        # Ensure valid_mask matches the number of targets
        if len(valid_mask) != T:
            lprint(ll.ERROR, f"Valid mask length {len(valid_mask)} does not match number of targets {T}")
            raise ValueError("Invalid valid_mask length")

        # Ensure valid_mask is on the correct device
        valid_mask = [m.to(device) if isinstance(m, torch.Tensor) else torch.tensor(m, dtype=torch.bool, device=device) for m in valid_mask]

        # Compute weighted MAE
        loss = 0.0
        for i in range(T):
            p = predictions[:, i] if T > 1 else predictions  # [B]
            t = targets[:, i] if T > 1 else targets  # [B]
            w = weights[i] if T > 1 else weights[0]  # Scalar
            m = valid_mask[i]  # [B]
            valid = m & (~torch.isnan(t))  # Combine mask with NaN check
            if valid.any():
                mae = torch.abs(p[valid] - t[valid]).mean()
                loss += w * mae
                lprint(ll.DEBUG, f"Target {i if T > 1 else 0}: {valid.sum().item()} valid entries, MAE = {mae.item():.4f}")
                
            else:
                lprint(ll.DEBUG, f"No valid real values for target {i if T > 1 else 0} in batch: Skip")
        

        # Normalize by number of targets
        loss = loss / T if T > 0 else torch.tensor(0.0, device=device)
        lprint(ll.DEBUG, f"Weighted MAE loss: {loss.item():.4f}")

        return loss

    except Exception as e:
        lprint(ll.ERROR, f"Error in wMAE_loss: {str(e)}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


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