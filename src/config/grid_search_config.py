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
    if problem_type == 'regression':
        activation_range = ['relu', 'leaky_relu', 'prelu', 'mish', 'gelu', 'swish', 'selu']
    elif problem_type == 'classification':
        activation_range = ['sigmoid', 'linear']

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


from skopt.space import Real, Integer, Categorical
import torch
from typing import List

from skopt.space import Real, Integer, Categorical
import torch
from typing import List

def generate_bayesian_search_fusion(X: torch.Tensor, y: List[torch.Tensor], teacher_output_sizes: List[int], 
                            problem_type: str = 'regression', max_combinations: int = 20) -> tuple:
    """
    Generate a Bayesian search space tailored to dataset and teacher model characteristics,
    supporting a variable number of hidden layers.
    
    Args:
        X (torch.Tensor): Input data tensor, shape [n_samples, n_features].
        y (List[torch.Tensor]): List of target tensors, each with shape [n_samples_i, output_dim].
        teacher_output_sizes (List[int]): Output sizes of teacher models.
        problem_type (str): Type of problem ('regression' or 'classification').
        max_combinations (int): Maximum number of combinations to evaluate.
    
    Returns:
        Tuple: (search_space, report)
    """
    # Dataset characteristics
    n_samples = X.shape[0]  # 556 after truncation
    n_features = X.shape[1]  # 613 after adjustment
    n_outputs = len(teacher_output_sizes)  # 5
    min_samples_per_target = min(y_i.shape[0] for y_i in y)  # 556
    target_ranges = [float(y_i.max() - y_i.min()) for y_i in y]
    max_target_range = max(target_ranges) if target_ranges else 1.0
    target_variances = [float(y_i.var()) for y_i in y]
    max_variance = max(target_variances) if target_variances else 1.0

    # Search space
    search_space = [
        Real(1e-5, 5e-4, name='learning_rate', prior='log-uniform'),
        Integer(max(8, min_samples_per_target // 20), min(min_samples_per_target // 4, 64), name='batch_size'),
        Real(0.1, 0.5, name='dropout_prob'),
        Integer(1, 3, name='num_hidden_layers'),
        Integer(max(n_features // 8, 64), min(n_features // 2, 256), name='hidden_size1'),
        Integer(max(n_features // 16, 32), min(n_features // 4, 128), name='hidden_size2'),
        Integer(max(n_features // 32, 16), min(n_features // 8, 64), name='hidden_size3'),
        Categorical([0, 1, 2], name='activation_idx'),
        Real(0.6, 1.0, name='alpha'),
        Real(0.0, min(0.2 / max_variance, 0.5), name='beta')
    ]

    # Report per il logging
    report = (
        f"Bayesian search space defined with {len(search_space)} dimensions:\n"
        f"- learning_rate: [{1e-5}, {5e-4}] (log-uniform)\n"
        f"- batch_size: [{max(8, min_samples_per_target // 20)}, {min(min_samples_per_target // 4, 64)}]\n"
        f"- dropout_prob: [0.1, 0.5]\n"
        f"- num_hidden_layers: [1, 3]\n"
        f"- hidden_size1: [{max(n_features // 8, 64)}, {min(n_features // 2, 256)}]\n"
        f"- hidden_size2: [{max(n_features // 16, 32)}, {min(n_features // 4, 128)}]\n"
        f"- hidden_size3: [{max(n_features // 32, 16)}, {min(n_features // 8, 64)}]\n"
        f"- activation_idx: [0, 1, 2]\n"
        f"- alpha: [0.6, 1.0]\n"
        f"- beta: [0.0, {min(0.2 / max_variance, 0.5):.6f}]\n"
        f"Dataset: n_samples={n_samples}, n_features={n_features}, n_outputs={n_outputs}, "
        f"min_samples_per_target={min_samples_per_target}, target_ranges={target_ranges}, "
        f"target_variances={target_variances}"
    )

    return search_space, report