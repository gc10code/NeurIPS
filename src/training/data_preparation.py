from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from typing import Tuple, List
from src.utils.normalizer import DataNormalizer
from src.utils.validators import DataValidator
from src.config.model_config import ModelConfig
from src.utils.exceptions import DataError
from src.utils.logging import lprint, LoggingLevels as ll
from src.preprocessing.fusion_dataset import DynamicFusionDataset

def clean_datasets (X:np.ndarray, y: np.ndarray):
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        lprint(ll.WARN,  "Input dataset X contains NaN or inf values. Cleaning dataset.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        lprint(ll.WARN,  "Input dataset y contains NaN or inf values. Cleaning dataset.")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    var = np.var(X, axis=0)
    valid_features = var > 0
    if not np.all(valid_features):
        lprint(ll.WARN,  f"Removing {np.sum(~valid_features)} features with zero variance.")
        X = X[:, valid_features]

    return X,y

def prepare_data(config: ModelConfig, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, DataNormalizer]:
    """Prepare and normalize input data."""
    try:
        lprint(ll.INFO, "Preparing and normalizing data")
        X, y = DataValidator.validate_data(X, y)
        config.output_size = y.shape[1] if y.ndim == 2 else 1
        normalizer = DataNormalizer(
            input_norm=config.input_normalization,
            output_norm=config.output_normalization
        )
        X_norm, y_norm = normalizer.fit_transform(X, y)
        lprint(ll.INFO, f"Data prepared: X shape={X_norm.shape}, y shape={y_norm.shape}")
        return X_norm, y_norm, normalizer
    except Exception as e:
        lprint(ll.ERROR, f"Data preparation failed: {str(e)}")
        raise DataError(f"Data preparation failed: {str(e)}")

def create_data_loaders(config: ModelConfig, X_norm: torch.Tensor, y_norm: torch.Tensor,
                        batch_size: int, fold: int = None) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoader for training and validation."""
    try:
        lprint(ll.INFO, f"Creating DataLoaders for fold {fold + 1 if fold is not None else 'split'}, batch_size={batch_size}")
        
        # Convert numpy arrays to tensors if not already
        X_norm = torch.tensor(X_norm, dtype=torch.float32) if not isinstance(X_norm, torch.Tensor) else X_norm
        y_norm = torch.tensor(y_norm, dtype=torch.float32) if not isinstance(y_norm, torch.Tensor) else y_norm
        
        # Adjust batch size for small datasets
        dataset_size = len(X_norm)
        if batch_size > dataset_size // 2:
            lprint(ll.WARN, f"Batch size {batch_size} is too large for dataset size {dataset_size}. Adjusting to {dataset_size // 2}.")
            batch_size = max(2, dataset_size // 2)
        
        if config.training_type == 'split':
            if config.validation_split <= 0:
                raise DataError("validation_split must be > 0 for split training")
            X_train, X_val, y_train, y_val = train_test_split(
                X_norm, y_norm, test_size=config.validation_split,
                random_state=config.seed
            )
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
    

def create_dataloader(metadata_file: str, descriptor_files: list[str], targets: list[str], batch_size: int, 
                      shuffle: bool = True, num_workers: int = 4, pin_memory: bool = True) -> DataLoader:
    lprint(ll.INFO, "Creating DataLoader")
    try:
        dataset = DynamicFusionDataset(metadata_file, descriptor_files, targets)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        lprint(ll.INFO, f"DataLoader created with batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}")
        return dataloader
    except Exception as e:
        lprint(ll.ERROR, f"Error in create_dataloader: {str(e)}")
        raise