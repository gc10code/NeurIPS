import numpy as np
import torch
from typing import Union
from src.models.rprop_mlp import RPropMLP
import torch.nn as nn

def generate_synthetic_X(X_real: torch.Tensor, noise_scale: float = 0.1, n_samples: int = 1000, 
                        seed: int = 42) -> torch.Tensor:
    """
    Generate synthetic descriptors by adding Gaussian noise to real descriptors.

    Args:
        X_real (torch.Tensor): Real input descriptors (shape: [n_samples, n_features]).
        noise_scale (float): Standard deviation of Gaussian noise. Default: 0.1.
        n_samples (int): Number of synthetic samples to generate. Default: 1000.
        seed (int): Random seed for reproducibility. Default: 42.

    Returns:
        torch.Tensor: Synthetic descriptors (shape: [n_samples, n_features]).

    Raises:
        ValueError: If X_real is not a 2D torch.Tensor or n_samples is non-positive.
    """
    if not isinstance(X_real, torch.Tensor):
        raise ValueError(f"X_real must be a torch.Tensor, got {type(X_real)}")
    if X_real.dim() != 2:
        raise ValueError(f"X_real must be a 2D tensor, got {X_real.dim()}D")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Select random samples from X_real with replacement
    indices = np.random.choice(X_real.shape[0], n_samples, replace=True)
    synthetic_X = X_real[indices].clone()

    # Add Gaussian noise
    noise = torch.normal(mean=0.0, std=noise_scale, size=synthetic_X.shape, device=X_real.device)
    synthetic_X = synthetic_X + noise

    # Clip to ensure values stay within valid ranges
    min_vals = X_real.min(dim=0).values
    max_vals = X_real.max(dim=0).values
    synthetic_X = torch.clamp(synthetic_X, min_vals, max_vals)

    return synthetic_X


def generate_synthetic_Y(model: RPropMLP, X: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Generate synthetic targets using the trained RPropMLP model.

    Args:
        model (RPropMLP): Trained RPropMLP model.
        X (torch.Tensor): Input descriptors (shape: [n_samples, n_features]).
        device (torch.device): Device to run the model on (CPU/GPU).

    Returns:
        torch.Tensor: Synthetic targets (shape: [n_samples, output_size]).

    Raises:
        ValueError: If X is not a 2D torch.Tensor or model is not an RPropMLP instance.
    """
    if not isinstance(model, RPropMLP):
        raise ValueError(f"model must be an RPropMLP instance, got {type(model)}")
    if not isinstance(X, torch.Tensor):
        raise ValueError(f"X must be a torch.Tensor, got {type(X)}")
    if X.dim() != 2:
        raise ValueError(f"X must be a 2D tensor, got {X.dim()}D")

    model.eval()
    with torch.no_grad():
        X_tensor = X.to(device, dtype=torch.float32)
        synthetic_Y = model(X_tensor)
    
    # Ensure output shape consistency
    if synthetic_Y.dim() == 1:
        synthetic_Y = synthetic_Y.unsqueeze(1)
    elif synthetic_Y.dim() > 2:
        synthetic_Y = synthetic_Y.squeeze(-1)

    return synthetic_Y


def generate_synthetic_Y_with_uncertainty(model: RPropMLP, X: torch.Tensor, device: torch.device, 
                                         n_samples: int = 10, uncertainty_threshold: float = 0.25) -> torch.Tensor:
    """
    Generate synthetic targets with uncertainty estimation using Monte Carlo dropout.

    Args:
        model (RPropMLP): Trained RPropMLP model.
        X (torch.Tensor): Input descriptors (shape: [n_samples, n_features]).
        device (torch.device): Device to run the model on (CPU/GPU).
        n_samples (int): Number of Monte Carlo samples for uncertainty estimation. Default: 10.
        uncertainty_threshold (float): Quantile threshold for filtering low-uncertainty predictions. Default: 0.25.

    Returns:
        torch.Tensor: Synthetic targets with low uncertainty (shape: [n_filtered_samples, output_size]).

    Raises:
        ValueError: If inputs are invalid or no samples pass the uncertainty threshold.
    """
    if not isinstance(model, RPropMLP):
        raise ValueError(f"model must be an RPropMLP instance, got {type(model)}")
    if not isinstance(X, torch.Tensor):
        raise ValueError(f"X must be a torch.Tensor, got {type(X)}")
    if X.dim() != 2:
        raise ValueError(f"X must be a 2D tensor, got {X.dim()}D")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    model.train()  # Enable dropout for uncertainty estimation
    predictions = []
    with torch.no_grad():
        X_tensor = X.to(device, dtype=torch.float32)
        for _ in range(n_samples):
            pred = model(X_tensor)
            if pred.dim() == 1:
                pred = pred.unsqueeze(1)
            elif pred.dim() > 2:
                pred = pred.squeeze(-1)
            predictions.append(pred.cpu())
    
    predictions = torch.stack(predictions, dim=0)  # Shape: [n_samples, n_data, output_size]
    mean_pred = predictions.mean(dim=0)  # Shape: [n_data, output_size]
    std_pred = predictions.std(dim=0)    # Shape: [n_data, output_size]
    
    # Filter predictions with low uncertainty
    uncertainty_threshold_val = torch.quantile(std_pred, uncertainty_threshold, dim=0)
    high_confidence_mask = (std_pred < uncertainty_threshold_val).all(dim=-1)
    
    if high_confidence_mask.sum() == 0:
        raise ValueError("No samples passed the uncertainty threshold")
    
    return mean_pred[high_confidence_mask]


class WeightedLoss(nn.Module):
    """
    Weighted loss function to prioritize real data over synthetic data.

    Attributes:
        real_weight (float): Weight for real data loss. Default: 1.0.
        synthetic_weight (float): Weight for synthetic data loss. Default: 0.5.
        base_criterion (nn.Module): Base loss function (e.g., SmoothL1Loss or CrossEntropyLoss).
    """
    def __init__(self, real_weight: float = 1.0, synthetic_weight: float = 0.5, 
                 problem_type: str = 'regression'):
        super().__init__()
        self.real_weight = real_weight
        self.synthetic_weight = synthetic_weight
        self.base_criterion = nn.SmoothL1Loss() if problem_type == 'regression' else nn.CrossEntropyLoss()
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, is_synthetic: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss.

        Args:
            outputs (torch.Tensor): Model predictions.
            targets (torch.Tensor): Target values.
            is_synthetic (torch.Tensor): Boolean tensor indicating synthetic samples (1 for synthetic, 0 for real).

        Returns:
            torch.Tensor: Weighted loss.
        """
        if outputs.shape != targets.shape:
            raise ValueError(f"Outputs shape {outputs.shape} does not match targets shape {targets.shape}")
        if is_synthetic.shape[0] != outputs.shape[0]:
            raise ValueError(f"is_synthetic shape {is_synthetic.shape} does not match outputs shape {outputs.shape}")

        loss = self.base_criterion(outputs, targets)
        weights = torch.where(is_synthetic.bool(), self.synthetic_weight, self.real_weight)
        weighted_loss = (loss * weights).mean()
        
        return weighted_loss