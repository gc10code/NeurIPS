from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset
import torch
from typing import Tuple
from src.utils.normalizer import DataNormalizer
from src.utils.validators import DataValidator
from src.config.model_config import RPropConfig
from src.utils.exceptions import DataError
from src.utils.logging import lprint, LoggingLevels as ll

def clean_datasets(X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clean input tensors by handling NaN and inf values and removing zero-variance features."""
    if torch.any(torch.isnan(X)) or torch.any(torch.isinf(X)):
        lprint(ll.WARN, "Input tensor X contains NaN or inf values. Cleaning tensor.")
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if torch.any(torch.isnan(y)) or torch.any(torch.isinf(y)):
        lprint(ll.WARN, "Input tensor y contains NaN or inf values. Cleaning tensor.")
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    var = torch.var(X, dim=0)
    valid_features = var > 0
    if not torch.all(valid_features):
        lprint(ll.WARN, f"Removing {torch.sum(~valid_features)} features with zero variance.")
        X = X[:, valid_features]

    return X, y

def prepare_data(config: RPropConfig, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, DataNormalizer]:
    """Prepare and normalize input tensors."""
    try:
        lprint(ll.INFO, "Preparing and normalizing data")
        X, y = DataValidator.validate_data(X, y)  # Assumes validator supports torch.Tensor
        config.output_size = y.shape[1] if y.ndim == 2 else 1
        normalizer = DataNormalizer(
            input_norm=config.input_normalization,
            output_norm=config.output_normalization
        )
        X_norm, y_norm = normalizer.fit_transform(X, y)  # Assumes normalizer supports torch.Tensor
        lprint(ll.INFO, f"Data prepared: X shape={X_norm.shape}, y shape={y_norm.shape}")
        return X_norm, y_norm, normalizer
    except Exception as e:
        lprint(ll.ERROR, f"Data preparation failed: {str(e)}")
        raise DataError(f"Data preparation failed: {str(e)}")

def create_data_loaders(config: RPropConfig, X_norm: torch.Tensor, y_norm: torch.Tensor,
                        batch_size: int, fold: int = None) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoader for training and validation."""
    try:
        lprint(ll.INFO, f"Creating DataLoaders for fold {fold + 1 if fold is not None else 'split'}, batch_size={batch_size}")
        
        # Ensure inputs are float32 tensors
        X_norm = X_norm.to(dtype=torch.float32)
        y_norm = y_norm.to(dtype=torch.float32)
        
        # Adjust batch size for small datasets
        dataset_size = len(X_norm)
        if batch_size > dataset_size // 2:
            lprint(ll.WARN, f"Batch size {batch_size} is too large for dataset size {dataset_size}. Adjusting to {dataset_size // 2}.")
            batch_size = max(2, dataset_size // 2)
        
        if config.training_type == 'split':
            if config.validation_split <= 0:
                raise DataError("validation_split must be > 0 for split training")
            # train_test_split requires numpy arrays, so convert temporarily
            X_np, y_np = X_norm.numpy(), y_norm.numpy()
            X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
                X_np, y_np, test_size=config.validation_split,
                random_state=config.seed
            )
            X_train = torch.tensor(X_train_np, dtype=torch.float32)
            X_val = torch.tensor(X_val_np, dtype=torch.float32)
            y_train = torch.tensor(y_train_np, dtype=torch.float32)
            y_val = torch.tensor(y_val_np, dtype=torch.float32)

        else:
            kf = KFold(n_splits=config.k_folds, shuffle=True, random_state=config.seed)
            train_idx, val_idx = list(kf.split(X_norm))[fold]
            X_train, X_val = X_norm[train_idx], X_norm[val_idx]
            y_train, y_val = y_norm[train_idx], y_norm[val_idx]
            
            # Check fold size
            train_size = len(train_idx)
            if train_size < batch_size:
                lprint(ll.WARN, f"Training fold {fold + 1} has only {train_size} samples, less than batch_size={batch_size}. Adjusting batch_size.")
                batch_size = max(2, train_size)
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Log dataset sizes
        lprint(ll.DEBUG, f"Fold {fold + 1 if fold is not None else 'split'}: Train samples={len(train_dataset)}, Validation samples={len(val_dataset)}")
        
        # Create DataLoaders with drop_last=True
        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=config.shuffle,
                num_workers=min(config.num_workers, torch.multiprocessing.cpu_count()),
                pin_memory=config.pin_memory,
                drop_last=True  # Prevent single-sample batches
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(config.num_workers, torch.multiprocessing.cpu_count()),
                pin_memory=config.pin_memory,
                drop_last=True  # Prevent single-sample batches
            )
        except RuntimeError as mp_e:
            lprint(ll.WARN, f"DataLoader multiprocessing failed with num_workers={config.num_workers}: {str(mp_e)}. Falling back to num_workers=0.")
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=config.shuffle,
                num_workers=0,
                pin_memory=config.pin_memory,
                drop_last=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=config.pin_memory,
                drop_last=True
            )
        
        lprint(ll.INFO, f"DataLoaders created: {len(train_loader)} train batches, {len(val_loader)} validation batches")
        return train_loader, val_loader
    except Exception as e:
        lprint(ll.ERROR, f"Failed to create data loaders: {str(e)}")
        raise DataError(f"Failed to create data loaders: {str(e)}")   