import numpy as np
import torch
from typing import List, Dict, Any, Union, Tuple
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from numpy.typing import NDArray
import logging

TensorLike = Union[np.ndarray, torch.Tensor]

def _to_numpy(X: TensorLike) -> np.ndarray:
    """Ensure the dataset is converted to a NumPy array."""
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy() # TODO: Check CUDA compatibility !!!!!!
    elif isinstance(X, np.ndarray):
        return X
    else:
        raise ValueError("Dataset must be a numpy.ndarray or torch.Tensor")

def _compute_correlation_metrics(X_np: np.ndarray, logger: logging.LoggerAdapter) -> Tuple[float, float]:
    """Compute correlation metrics for the dataset with robust error handling."""
    # Controlla per valori inf o NaN
    if np.any(np.isnan(X_np)) or np.any(np.isinf(X_np)):
        logger.warning("Dataset contains NaN or inf values. Returning default correlation metrics.")
        return 0.0, 1.0
    
    # Rimuovi feature con varianza zero
    var = np.var(X_np, axis=0)
    valid_features = var > 0
    if not np.any(valid_features):
        logger.warning("All features have zero variance. Returning default correlation metrics.")
        return 0.0, 1.0
    
    X_np = X_np[:, valid_features]
    
    if X_np.shape[1] > 1:
        try:
            corr_matrix = np.corrcoef(X_np, rowvar=False)
            # Controlla se la matrice contiene NaN o inf
            if np.any(np.isnan(corr_matrix)) or np.any(np.isinf(corr_matrix)):
                logger.warning("Correlation matrix contains NaN or inf values. Returning default correlation metrics.")
                return 0.0, 1.0
            # Media delle correlazioni assolute (ignorando la diagonale)
            corr_matrix = np.abs(corr_matrix)
            np.fill_diagonal(corr_matrix, 0)
            mean_corr = np.mean(corr_matrix)
            # Spread degli autovalori
            eigenvalues = np.linalg.eigvals(corr_matrix)
            if np.any(np.isnan(eigenvalues)) or np.any(np.isinf(eigenvalues)):
                logger.warning("Eigenvalues contain NaN or inf values. Returning default correlation metrics.")
                return 0.0, 1.0
            eig_spread = eigenvalues.max() / (eigenvalues.min() + 1e-10)
        except Exception as e:
            logger.warning(f"Failed to compute correlation metrics: {str(e)}. Returning default correlation metrics.")
            return 0.0, 1.0
    else:
        logger.warning("Dataset has only one feature or all features have zero variance. Returning default correlation metrics.")
        mean_corr = 0.0
        eig_spread = 1.0
    
    return mean_corr, eig_spread

def generate_bayesian_search(X: TensorLike, problem_type: str = 'regression', max_combinations: int = 20, logger: logging.Logger = None) -> Tuple[Dict[str, Any], List[str]]:
    """Generate search space for Bayesian optimization."""
    if logger is None:
        logger = logging.getLogger('BayesianSearch')
    
    report = []
    X_np = _to_numpy(X)
    
    # Check and Clean Dataset
    if np.any(np.isnan(X_np)) or np.any(np.isinf(X_np)):
        logger.warning("Input dataset contains NaN or inf values. Cleaning dataset.")
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    n_samples, n_features = X_np.shape
    feature_var = np.var(X_np, axis=0).mean()
    sparsity = np.mean(X_np == 0)
    feature_sample_ratio = n_features / max(1, n_samples)
    mean_corr, eig_spread = _compute_correlation_metrics(X_np, logger)

    report.append(f"Dataset: {n_samples} samples, {n_features} features.")
    report.append(f"Mean feature variance: {feature_var:.4f}")
    report.append(f"Sparsity: {sparsity:.4f}")
    report.append(f"Feature/Sample ratio: {feature_sample_ratio:.4f}")
    report.append(f"Mean feature correlation: {mean_corr:.4f}, Eigenvalue spread: {eig_spread:.4f}")

    # Learning rate
    lr_range = [1e-4, 1e-1]
    if feature_sample_ratio < 10:
        lr_range = [1e-4, 5e-3]
        report.append(f"Low sample/feature ratio ({feature_sample_ratio:.2f}) → conservative learning rate range {lr_range}.")
    elif feature_sample_ratio < 100:
        lr_range = [5e-4, 1e-2]
        report.append(f"Moderate sample/feature ratio ({feature_sample_ratio:.2f}) → moderate learning rate range {lr_range}.")
    else:
        lr_range = [1e-3, 2e-2]
        report.append(f"High sample/feature ratio ({feature_sample_ratio:.2f}) → aggressive learning rate range {lr_range}.")
    if feature_var > 5.0:
        lr_range = [lr * 0.5 for lr in lr_range]
        report.append(f"High feature variance ({feature_var:.2f}) → halved learning rate range to {lr_range}.")
    if mean_corr > 0.7:
        lr_range = [lr * 0.8 for lr in lr_range]
        report.append(f"High feature correlation ({mean_corr:.2f}) → reduced learning rate range to {lr_range}.")

    # Batch size
    if n_samples < 1000:
        bs_range = [8, 32]
        report.append(f"Small dataset ({n_samples} samples) → batch size range {bs_range}.")
    elif n_samples < 10000:
        bs_range = [16, 64]
        report.append(f"Medium dataset ({n_samples} samples) → batch size range {bs_range}.")
    else:
        bs_range = [32, 128]
        report.append(f"Large dataset ({n_samples} samples) → batch size range {bs_range}.")
    if sparsity > 0.8:
        bs_range = [max(8, bs // 4) for bs in bs_range]
        report.append(f"High sparsity ({sparsity:.2f}) → reduced batch size range to {bs_range}.")
    if mean_corr > 0.7:
        bs_range = [max(8, int(bs * 0.8)) for bs in bs_range]
        report.append(f"High feature correlation ({mean_corr:.2f}) → reduced batch size range to {bs_range}.")

    # Hidden layers
    base_width = n_features
    if feature_var > 5.0:
        base_width = int(base_width * 2.0)
        report.append(f"High variance → doubled base width to {base_width}.")
    elif feature_var < 0.1:
        base_width = int(base_width * 0.7)
        report.append(f"Low variance → reduced base width to {base_width}.")
    if sparsity > 0.8:
        base_width = int(base_width * 0.5)
        report.append(f"High sparsity → halved base width to {base_width}.")
    if mean_corr > 0.7:
        base_width = int(base_width * 0.7)
        report.append(f"High feature correlation → reduced base width to {base_width}.")
    if n_samples > 100000:
        base_width = int(base_width * 1.5)
        report.append(f"Large dataset → increased base width to {base_width}.")

    depth_range = [2, 5]
    decay_factor_range = [0.3, 0.7] if feature_sample_ratio > 1.0 else [0.5, 0.8]
    report.append(f"Depth range: {depth_range}, Decay factor range: {decay_factor_range}")

    # Activations
    activation_range = ['relu', 'gelu', 'swish']
    if problem_type == 'classification':
        activation_range.append('selu')
        report.append("Classification problem → added SELU to activation range.")

    # RProp parameters
    delta_plus_range = [1.0, 1.3] if sparsity < 0.5 else [1.0, 1.2]
    if mean_corr > 0.7:
        delta_plus_range = [dp * 0.9 for dp in delta_plus_range]
        report.append(f"High feature correlation ({mean_corr:.2f}) → reduced delta_plus range to {delta_plus_range}.")

    bayesian_config = {
        'learning_rate': {'range': lr_range, 'type': Real, 'name': 'learning_rate'},
        'batch_size': {'range': bs_range, 'type': Integer, 'name': 'batch_size'},
        'depth': {'range': depth_range, 'type': Integer, 'name': 'depth'},
        'decay_factor': {'range': decay_factor_range, 'type': Real, 'name': 'decay_factor'},
        'activation': {'range': activation_range, 'type': Categorical, 'name': 'activation'},
        'rprop': {
            'delta_plus': {'range': delta_plus_range, 'type': Real, 'name': 'delta_plus'},
            'delta_minus': {'range': [0.4, 0.6], 'type': Real, 'name': 'delta_minus'},
            'delta_min': {'range': [1e-7, 1e-5], 'type': Real, 'name': 'delta_min'},
            'delta_max': {'range': [10.0, 100.0], 'type': Real, 'name': 'delta_max'}
        },
        'base_width': base_width
    }
    report.append(f"Bayesian search space defined: {bayesian_config}")
    return bayesian_config, report