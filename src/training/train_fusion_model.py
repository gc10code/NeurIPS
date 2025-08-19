from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from src.config.model_config import HRMConfig
from src.utils.logging import lprint, LoggingLevels as ll
from src.models.fusion_model import FusionModel
from src.models.rprop_mlp import RPropMLP
from src.utils.metrics import wMAE_loss, compute_wmae_weights
from src.utils.normalizer import DataNormalizer
from src._global import global_init
import os
import pandas as pd
from pathlib import Path
import optuna
import pickle
import glob

_global = global_init()

@dataclass
class TrainingConfig:
    """Configuration for training the FusionModel and refining teachers."""
    learning_rate_fusion: float = 0.001
    learning_rate_teachers: float = 0.001
    batch_size: int = 32
    max_epochs: int = _global.FUSION_EPOCHS
    patience: int = _global.FUSION_PATIENCE
    min_delta: float = 1e-4
    output_dir: str = "./output"
    checkpoint_interval: int = 50
    validation_split: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class FusionTrainer:
    def __init__(self, fusion_model: FusionModel, teacher_models: List[nn.Module], config: TrainingConfig, hrm_config: HRMConfig, normalizer: DataNormalizer):
        """
        Initialize the trainer for the FusionModel and teacher models.

        Args:
            fusion_model (FusionModel): The fusion model to train.
            teacher_models (List[nn.Module]): List of pre-trained teacher models.
            config (TrainingConfig): Training configuration.
            hrm_config (HRMConfig): HRM configuration.
            normalizer (DataNormalizer): Normalizer for input/output data.
        """
        self.fusion_model = fusion_model.to(config.device)
        self.teacher_models = [teacher.to(config.device) for teacher in teacher_models]
        self.config = config
        self.hrm_config = hrm_config
        self.normalizer = normalizer
        self.device = config.device
        self.optimizer_fusion = optim.Adam(self.fusion_model.parameters(), lr=config.learning_rate_fusion)
        self.optimizers_teachers = [optim.Adam(teacher.parameters(), lr=config.learning_rate_teachers) 
                                   for teacher in self.teacher_models]
        self.scheduler_fusion = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_fusion, mode='min', factor=0.5, patience=_global.FUSION_PATIENCE
        )
        self.schedulers_teachers = [
            optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=_global.FUSION_PATIENCE)
            for opt in self.optimizers_teachers
        ]
        self.criterion = nn.L1Loss(reduction='none')
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_path = os.path.join(config.output_dir, "best_fusion_model.pth")

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], targets: List[str], malus : List[float] = _global.MALUS) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single training step for the fusion model and teachers.

        Args:
            batch: Tuple of (X_fusion, y_fusion, valid_mask).
            targets: List of target names for logging.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Fusion loss, average teacher loss,
                                                                         fusion wMAE, average teacher wMAE.
        """
        if len(targets) != len(malus):
            lprint(ll.ERROR, "tagets has not the same number of malus")
            raise
        
        X_fusion, y_fusion, valid_mask = [x.to(self.device) for x in batch]
        B, T = y_fusion.shape

        # Compute weights for wMAE
        true_targets = [
            (y_fusion[:, i:i+1], y_fusion[:, i:i+1], valid_mask[:, i:i+1].sum().item())
            for i in range(T)
        ]
        weights = compute_wmae_weights(true_targets).to(self.device)

        # Clear gradients
        self.optimizer_fusion.zero_grad()
        for optimizer in self.optimizers_teachers:
            optimizer.zero_grad()

        # Fusion model forward pass
        outputs, info = self.fusion_model(y_fusion)
        fusion_loss = self.compute_loss(outputs, y_fusion, valid_mask)
        fusion_wmae = wMAE_loss(outputs, y_fusion, weights, valid_mask)

        # Teacher forward pass
        teacher_losses = []
        teacher_wmaes = []
        for i, (teacher, target, mal) in enumerate(zip(self.teacher_models, targets, malus)):
            teacher_output = teacher(X_fusion)
            teacher_target = y_fusion[:, i:i+1]
            teacher_mask = valid_mask[:, i:i+1]
            teacher_weight = weights[i:i+1]
            teacher_loss = self.compute_loss(teacher_output, teacher_target, teacher_mask)
            teacher_wmae = wMAE_loss(teacher_output, teacher_target, teacher_weight, teacher_mask)
            teacher_losses.append(teacher_loss * mal)
            teacher_wmaes.append(teacher_wmae)
            if teacher_mask.sum() == 0:
                lprint(ll.DEBUG, f"No valid data for teacher {target} in this batch")

        # Log per-target teacher losses
        for i, (target, loss) in enumerate(zip(targets, teacher_losses)):
            lprint(ll.DEBUG, f"Teacher {target} Train Loss: {loss.item():.4f}")

        # Combine fusion and teacher losses
        total_loss = _global.TEACHERS_TRUST * fusion_loss + _global.TEACHERS_TRUST * sum(teacher_losses) / len(teacher_losses)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.fusion_model.parameters(), self.hrm_config.gradient_clipping)
        for teacher in self.teacher_models:
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), self.hrm_config.gradient_clipping)
        self.optimizer_fusion.step()
        for optimizer in self.optimizers_teachers:
            optimizer.step()

        return fusion_loss, sum(teacher_losses) / len(teacher_losses), fusion_wmae, sum(teacher_wmaes) / len(teacher_wmaes)

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked MSE loss, considering valid_mask to ignore synthetic targets.

        Args:
            outputs (torch.Tensor): Model predictions [B, num_targets] or [B, 1].
            targets (torch.Tensor): Ground truth targets [B, num_targets] or [B, 1].
            valid_mask (torch.Tensor): Boolean mask [B, num_targets] or [B, 1].

        Returns:
            torch.Tensor: Masked loss.
        """
        loss = self.criterion(outputs, targets)
        masked_loss = loss * valid_mask.float()
        return masked_loss.sum() / valid_mask.sum().clamp(min=1)

    def validate(self, val_loader: DataLoader, targets: List[str]) -> Tuple[float, float, float, float]:
        """
        Validate the fusion model and teachers.

        Args:
            val_loader (DataLoader): Validation data loader.
            targets (List[str]): List of target names for logging.

        Returns:
            Tuple[float, float, float, float]: Average fusion validation loss, teacher validation loss,
                                              fusion wMAE, teacher wMAE.
        """
        self.fusion_model.eval()
        for teacher in self.teacher_models:
            teacher.eval()

        total_fusion_loss = 0.0
        total_teacher_loss = 0.0
        total_fusion_wmae = 0.0
        total_teacher_wmae = 0.0
        num_batches = 0
        teacher_losses_per_target = [0.0] * len(targets)

        with torch.no_grad():
            for batch in val_loader:
                X_fusion, y_fusion, valid_mask = [x.to(self.device) for x in batch]
                true_targets = [
                    (y_fusion[:, i:i+1], y_fusion[:, i:i+1], valid_mask[:, i:i+1].sum().item())
                    for i in range(y_fusion.shape[-1])
                ]
                weights = compute_wmae_weights(true_targets).to(self.device)

                outputs, _ = self.fusion_model(y_fusion)
                fusion_loss = self.compute_loss(outputs, y_fusion, valid_mask)
                fusion_wmae = wMAE_loss(outputs, y_fusion, weights, valid_mask)
                total_fusion_loss += fusion_loss.item()
                total_fusion_wmae += fusion_wmae.item()

                teacher_loss = 0.0
                teacher_wmae = 0.0
                for i, (teacher, target) in enumerate(zip(self.teacher_models, targets)):
                    teacher_output = teacher(X_fusion)
                    teacher_target = y_fusion[:, i:i+1]
                    teacher_mask = valid_mask[:, i:i+1]
                    teacher_weight = weights[i:i+1]
                    t_loss = self.compute_loss(teacher_output, teacher_target, teacher_mask).item()
                    t_wmae = wMAE_loss(teacher_output, teacher_target, teacher_weight, teacher_mask).item()
                    teacher_loss += t_loss
                    teacher_wmae += t_wmae
                    teacher_losses_per_target[i] += t_loss
                total_teacher_loss += teacher_loss / len(self.teacher_models)
                total_teacher_wmae += teacher_wmae / len(self.teacher_models)
                num_batches += 1

        for target, t_loss in zip(targets, teacher_losses_per_target):
            lprint(ll.INFO, f"Teacher {target} Validation Loss: {t_loss / num_batches:.4f}")

        self.fusion_model.train()
        for teacher in self.teacher_models:
            teacher.train()

        return total_fusion_loss / num_batches, total_teacher_loss / num_batches, total_fusion_wmae / num_batches, total_teacher_wmae / num_batches

    def compute_prediction_errors(self, val_loader: DataLoader, targets: List[str]) -> None:
        """
        Compute and log the differences between predicted and true values for non-synthetic data.

        Args:
            val_loader (DataLoader): Validation data loader.
            targets (List[str]): List of target names.
        """
        self.fusion_model.eval()
        for teacher in self.teacher_models:
            teacher.eval()

        all_fusion_preds = []
        all_teacher_preds = []
        all_true = []
        all_masks = []

        with torch.no_grad():
            for batch in val_loader:
                X_fusion, y_fusion, valid_mask = [x.to(self.device) for x in batch]
                fusion_preds, _ = self.fusion_model(y_fusion)
                all_fusion_preds.append(fusion_preds.cpu())
                all_true.append(y_fusion.cpu())
                all_masks.append(valid_mask.cpu())

                teacher_preds = []
                for teacher in self.teacher_models:
                    teacher_pred = teacher(X_fusion)
                    teacher_preds.append(teacher_pred.cpu())
                all_teacher_preds.append(torch.cat(teacher_preds, dim=-1))

        fusion_preds = torch.cat(all_fusion_preds, dim=0)
        teacher_preds = torch.cat(all_teacher_preds, dim=0)
        true_values = torch.cat(all_true, dim=0)
        valid_mask = torch.cat(all_masks, dim=0)

        # Denormalize predictions and true values using self.normalizer
        fusion_preds = self.normalizer.inverse_transform_y(fusion_preds)
        teacher_preds = self.normalizer.inverse_transform_y(teacher_preds)
        true_values = self.normalizer.inverse_transform_y(true_values)

        lprint(ll.REPORT, "Prediction Errors for Non-Synthetic Data (Validation Set):")
        for i, target in enumerate(targets):
            mask = valid_mask[:, i]
            if mask.sum() == 0:
                lprint(ll.WARN, f"No non-synthetic data for target {target}")
                continue

            fusion_pred = fusion_preds[:, i][mask]
            teacher_pred = teacher_preds[:, i][mask]
            true_val = true_values[:, i][mask]

            fusion_errors = torch.abs(fusion_pred - true_val)
            teacher_errors = torch.abs(teacher_pred - true_val)

            fusion_mae = fusion_errors.mean().item()
            teacher_mae = teacher_errors.mean().item()
            fusion_rmse = torch.sqrt(torch.mean(fusion_errors ** 2)).item()
            teacher_rmse = torch.sqrt(torch.mean(teacher_errors ** 2)).item()

            lprint(ll.INFO, f"Target {target}:")
            lprint(ll.INFO, f"  Fusion Model MAE: {fusion_mae:.4f}, RMSE: {fusion_rmse:.4f} (based on teacher outputs)")
            lprint(ll.INFO, f"  Teacher Model MAE: {teacher_mae:.4f}, RMSE: {teacher_rmse:.4f} (based on molecular descriptors)")
            lprint(ll.INFO, f"  Number of Non-Synthetic Samples: {mask.sum().item()}")
            lprint(ll.INFO, f"  Sample Differences (First 5, Fusion):")
            for j in range(min(5, len(fusion_pred))):
                lprint(ll.INFO, f"    Sample {j+1}: Predicted = {fusion_pred[j]:.4f}, True = {true_val[j]:.4f}, Error = {fusion_errors[j]:.4f}")
            lprint(ll.INFO, f"  Sample Differences (First 5, Teacher):")
            for j in range(min(5, len(teacher_pred))):
                lprint(ll.INFO, f"    Sample {j+1}: Predicted = {teacher_pred[j]:.4f}, True = {true_val[j]:.4f}, Error = {teacher_errors[j]:.4f}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader, targets: List[str]) -> None:
        """
        Train the fusion model and refine teachers.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            targets (List[str]): List of target names.
        """
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)

        for epoch in range(self.config.max_epochs):
            self.fusion_model.train()
            for teacher in self.teacher_models:
                teacher.train()

            total_fusion_loss = 0.0
            total_teacher_loss = 0.0
            total_fusion_wmae = 0.0
            total_teacher_wmae = 0.0
            num_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}"):
                fusion_loss, teacher_loss, fusion_wmae, teacher_wmae = self.train_step(batch, targets)
                total_fusion_loss += fusion_loss.item()
                total_teacher_loss += teacher_loss.item()
                total_fusion_wmae += fusion_wmae.item()
                total_teacher_wmae += teacher_wmae.item()
                num_batches += 1

            avg_fusion_loss = total_fusion_loss / num_batches
            avg_teacher_loss = total_teacher_loss / num_batches
            avg_fusion_wmae = total_fusion_wmae / num_batches
            avg_teacher_wmae = total_teacher_wmae / num_batches
            lprint(ll.INFO, f"Epoch {epoch+1}: Train Fusion Loss = {avg_fusion_loss:.4f} "
                            f"Train Teacher Loss = {avg_teacher_loss:.4f}")

            val_fusion_loss, val_teacher_loss, val_fusion_wmae, val_teacher_wmae = self.validate(val_loader, targets)
            lprint(ll.REPORT, f"Validation Fusion Loss = {val_fusion_loss:.4f} "
                              f"Validation Teacher Loss = {val_teacher_loss:.4f}")

            self.scheduler_fusion.step(val_fusion_loss)
            for scheduler in self.schedulers_teachers:
                scheduler.step(val_teacher_loss)
            
            # Save best model with HRMConfig if validation wMAE improves
            val_loss = _global.FUSION_TRUST * val_fusion_loss + _global.TEACHERS_TRUST * val_teacher_loss
            
            if val_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save({
                    'state_dict': self.fusion_model.state_dict(),
                    'hrm_config': self.hrm_config.__dict__
                }, self.best_model_path)
                lprint(ll.SUCCESS, f"New best model saved with validation loss {val_loss:.4f} to {self.best_model_path}")
            else:
                self.patience_counter += 1
                lprint(ll.WARN, f"No improvement in validation loss fuction. Patience: {self.patience_counter}/{self.config.patience}")
                if self.patience_counter >= self.config.patience:
                    lprint(ll.EXIT, "Early stopping triggered")
                    break

        lprint(ll.INFO, "Computing prediction errors for non-synthetic data...")
        self.compute_prediction_errors(val_loader, targets)

def predict_single_target(teacher: nn.Module, X: torch.Tensor, normalizer: DataNormalizer, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> torch.Tensor:
    """
    Make a single-target prediction using a specific teacher model.

    Args:
        teacher (nn.Module): The teacher model to use for prediction.
        X (torch.Tensor): Input features [B, N].
        normalizer (DataNormalizer): Normalizer for input/output data.
        device (str): Device to run the prediction on.

    Returns:
        torch.Tensor: Predicted values [B, 1] in the original scale.
    """
    teacher.eval()
    # Normalize input features
    X_normalized = normalizer.transform(X)  # Only X, no y
    X_normalized = X_normalized.to(device)
    
    with torch.no_grad():
        output = teacher(X_normalized)  # [B, 1]
    
    # Denormalize output to original scale
    output = normalizer.inverse_transform_y(output)
    
    return output

def load_fusion_dataset(targets: List[str]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], DataNormalizer]:
    """
    Load fusion dataset from TSV files and create X_fusion, y_fusion, and valid_mask.

    Args:
        targets (List[str]): List of target names (e.g., ["Density", "FFV", "Rg", "Tc", "Tg"]).

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], DataNormalizer]:
            - X_fusion: Tensor of shape (N, D) containing descriptors.
            - y_fusion: List of tensors, each of shape (N, 1), containing target values.
            - valid_mask: List of boolean tensors, each of shape (N,), indicating real values.
            - normalizer: DataNormalizer used for normalization.
    """
    try:
        lprint(ll.INFO, "Loading fusion dataset")
        fusion_descriptors_file = Path("./data/fusion_descriptors.tsv")
        fusion_target_file = Path("./data/fusion_target.tsv")
        fusion_meta_file = Path("./data/fusion_meta.tsv")

        if not all(Path(f).exists() for f in [fusion_descriptors_file, fusion_target_file, fusion_meta_file]):
            missing = [f for f in [fusion_descriptors_file, fusion_target_file, fusion_meta_file] if not Path(f).exists()]
            lprint(ll.ERROR, f"Missing files: {missing}")
            raise ValueError(f"Missing files: {missing}")

        descriptors_df = pd.read_csv(fusion_descriptors_file, sep='\t')
        if 'id' not in descriptors_df.columns:
            lprint(ll.ERROR, "fusion_descriptors.tsv missing 'id' column")
            raise ValueError("fusion_descriptors.tsv missing 'id' column")
        lprint(ll.DEBUG, f"Loaded fusion_descriptors.tsv with shape {descriptors_df.shape}")

        target_df = pd.read_csv(fusion_target_file, sep='\t')
        if 'id' not in target_df.columns:
            lprint(ll.ERROR, "fusion_target.tsv missing 'id' column")
            raise ValueError("fusion_target.tsv missing 'id' column")
        missing_targets = [t for t in targets if t not in target_df.columns]
        if missing_targets:
            lprint(ll.ERROR, f"Missing target columns in fusion_target.tsv: {missing_targets}")
            raise ValueError(f"Missing target columns: {missing_targets}")
        lprint(ll.DEBUG, f"Loaded fusion_target.tsv with shape {target_df.shape}")

        meta_df = pd.read_csv(fusion_meta_file, sep='\t')
        if 'id' not in meta_df.columns:
            lprint(ll.ERROR, "fusion_meta.tsv missing 'id' column")
            raise ValueError("fusion_meta.tsv missing 'id' column")
        missing_valid_columns = [f"{t}_valid" for t in targets if f"{t}_valid" not in meta_df.columns]
        if missing_valid_columns:
            lprint(ll.ERROR, f"Missing validity columns in fusion_meta.tsv: {missing_valid_columns}")
            raise ValueError(f"Missing validity columns: {missing_valid_columns}")
        lprint(ll.DEBUG, f"Loaded fusion_meta.tsv with shape {meta_df.shape}")

        descriptors_ids = set(descriptors_df['id'])
        target_ids = set(target_df['id'])
        meta_ids = set(meta_df['id'])
        if not (descriptors_ids == target_ids == meta_ids):
            lprint(ll.ERROR, "ID mismatch between fusion_descriptors.tsv, fusion_target.tsv, and fusion_meta.tsv")
            raise ValueError("ID mismatch between files")

        descriptors_df = descriptors_df.sort_values('id').reset_index(drop=True)
        target_df = target_df.sort_values('id').reset_index(drop=True)
        meta_df = meta_df.sort_values('id').reset_index(drop=True)

        num_samples = descriptors_df.shape[0]
        if target_df.shape[0] != num_samples or meta_df.shape[0] != num_samples:
            lprint(ll.ERROR, f"Sample count mismatch: descriptors={num_samples}, target={target_df.shape[0]}, meta={meta_df.shape[0]}")
            raise ValueError("Sample count mismatch between files")

        X_fusion = torch.tensor(descriptors_df.drop(columns=['id']).values, dtype=torch.float32)
        lprint(ll.DEBUG, f"X_fusion shape: {X_fusion.shape}")

        y_fusion = [torch.tensor(target_df[target].values[:, None], dtype=torch.float32).clone().detach() for target in targets]
        lprint(ll.DEBUG, f"y_fusion shapes: {[y.shape for y in y_fusion]}")

        # Initialize normalizer and normalize data
        normalizer = DataNormalizer(input_norm='zscore', output_norm='zscore')  

        # Concatenate y_fusion list into a single tensor for normalization
        y_fusion_tensor = torch.cat(y_fusion, dim=1)

        # Normalize X_fusion and y_fusion_tensor
        X_fusion, y_fusion_tensor = normalizer.fit_transform(X_fusion, y_fusion_tensor)

        # Convert y_fusion_tensor back to a list of tensors
        y_fusion = [y_fusion_tensor[:, i:i+1] for i in range(y_fusion_tensor.shape[1])]

        # Save the normalizer for later use
        normalizer_path = os.path.join('./output', 'normalizer.pkl')
        os.makedirs('./output', exist_ok=True)
        with open(normalizer_path, 'wb') as f:
            pickle.dump(normalizer, f)
        lprint(ll.INFO, "Data normalizer saved to output/normalizer.pkl")

        valid_mask = [torch.tensor(meta_df[f"{target}_valid"].values, dtype=torch.bool).clone().detach() for target in targets]
        lprint(ll.DEBUG, f"valid_mask shapes: {[m.shape for m in valid_mask]}")

        num_samples = X_fusion.shape[0]
        if any(y.shape[0] != num_samples for y in y_fusion) or any(m.shape[0] != num_samples for m in valid_mask):
            lprint(ll.ERROR, f"Shape mismatch between X_fusion ({num_samples}), y_fusion {[y.shape[0] for y in y_fusion]}, and valid_mask {[m.shape[0] for m in valid_mask]}")
            raise ValueError("Shape mismatch between X_fusion, y_fusion, and valid_mask")

        for target, mask in zip(targets, valid_mask):
            valid_count = mask.sum().item()
            lprint(ll.INFO, f"{target}: {valid_count} real values, {num_samples - valid_count} imputed")

        lprint(ll.INFO, "Fusion dataset loaded successfully")
        return X_fusion, y_fusion, valid_mask, normalizer

    except Exception as e:
        lprint(ll.ERROR, f"Error loading fusion dataset: {str(e)}")
        raise

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

def objective(trial: optuna.Trial, 
              true_targets: List[Tuple[torch.Tensor, torch.Tensor, int]], 
              targets: List[str], 
              X_fusion: torch.Tensor, 
              y_fusion_list: List[torch.Tensor], 
              valid_mask_list: List[torch.Tensor], 
              device: str, 
              normalizer: DataNormalizer) -> float:
    """
    Objective function for Bayesian hyperparameter search with Optuna.

    Args:
        trial (optuna.Trial): Optuna trial object for sampling hyperparameters.
        true_targets (List[Tuple]): List of tuples (target_tensor, target_tensor, num_valid) for wMAE weights.
        targets (List[str]): List of target names.
        X_fusion (torch.Tensor): Input features for the fusion model.
        y_fusion_list (List[torch.Tensor]): List of target tensors.
        valid_mask_list (List[torch.Tensor]): List of validity masks.
        device (str): Device (CPU/GPU).
        normalizer (DataNormalizer): Normalizer for input/output data.

    Returns:
        float: Mean validation wMAE of the fusion model.
    """
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hrm_config = HRMConfig(
        input_size=1,
        num_targets=len(targets),
        context_size=trial.suggest_int("context_size", 64, 256),
        low_hidden=trial.suggest_int("low_hidden", 128, 512),
        high_hidden=trial.suggest_int("high_hidden", 256, 1024),
        act_eps=0.01,
        max_high_steps=trial.suggest_int("max_high_steps", 5, 20),
        max_low_steps=trial.suggest_int("max_low_steps", 3, 10),
        one_step_detach=False,
        dropout_prob=trial.suggest_float("dropout_prob", 0.0, 0.5),
        batch_norm=trial.suggest_categorical("batch_norm", [True, False]),
        gradient_clipping=trial.suggest_float("gradient_clipping", 0.5, 5.0),
        output_dir="./output",
        training_type='split',
        learning_rate=0.001,
        batch_size=32,
        max_epochs=_global.FUSION_EPOCHS,
        patience=_global.FUSION_PATIENCE,
        validation_split=0.2
    )

    training_config = TrainingConfig(
        learning_rate_fusion=trial.suggest_float("learning_rate_fusion", 1e-4, 1e-2, log=True),
        learning_rate_teachers=trial.suggest_float("learning_rate_teachers", 1e-4, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        max_epochs=_global.FUSION_EPOCHS,
        patience=_global.FUSION_PATIENCE,
        min_delta=1e-4,
        output_dir="./output",
        checkpoint_interval=50,
        validation_split=0.2,
        device=device
    )

    y_fusion = torch.cat(y_fusion_list, dim=1)
    valid_mask = torch.stack(valid_mask_list, dim=1)
    lprint(ll.DEBUG, f"y_fusion shape: {y_fusion.shape}, valid_mask shape: {valid_mask.shape}")

    dataset = TensorDataset(X_fusion, y_fusion, valid_mask)
    train_size = int((1 - training_config.validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    lprint(ll.DEBUG, f"Dataset size: {len(dataset)}, Train size: {train_size}, Val size: {val_size}")

    if val_size == 0:
        lprint(ll.ERROR, "Validation set is empty. Check dataset size or validation_split.")
        raise ValueError("Validation set is empty")

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)

    # Extract validation set tensors for evaluate_saved_models
    if len(val_dataset) == 0:
        lprint(ll.ERROR, "Validation dataset is empty after split")
        raise ValueError("Empty validation dataset")

    # Extract validation set tensors
    X_fusion_val = torch.stack([x[0] for x in val_dataset])
    y_fusion_val = torch.stack([x[1] for x in val_dataset])
    valid_mask_val = torch.stack([x[2] for x in val_dataset])
    lprint(ll.DEBUG, f"X_fusion_val shape: {X_fusion_val.shape}, y_fusion_val shape: {y_fusion_val.shape}, valid_mask_val shape: {valid_mask_val.shape}")

    # Verifica che y_fusion_val abbia due dimensioni
    if len(y_fusion_val.shape) < 2:
        lprint(ll.ERROR, f"y_fusion_val has unexpected shape: {y_fusion_val.shape}. Expected [N, T]")
        raise ValueError(f"y_fusion_val has unexpected shape: {y_fusion_val.shape}")

    y_fusion_val_list = [y_fusion_val[:, i:i+1] for i in range(y_fusion_val.shape[1])]
    valid_mask_val_list = [valid_mask_val[:, i] for i in range(valid_mask_val.shape[1])]
    true_target_val = [
        (y_fusion_val_list[i], y_fusion_val_list[i], valid_mask_val_list[i].sum().item())
        for i in range(len(targets))
    ]

    from training.results_manager import ResultsManager
    teacher_models = []
    for target in targets:
        try:
            model = ResultsManager.load_best_model(f"./output/{target}/best_model.pth")
            teacher_models.append(model)
            lprint(ll.SUCCESS, f"Loaded teacher model for {target}")
        except Exception as e:
            lprint(ll.ERROR, f"Failed to load teacher model for {target}: {str(e)}")
            raise

    fusion_model = FusionModel(hrm_config)
    trainer = FusionTrainer(fusion_model, teacher_models, training_config, hrm_config, normalizer)

    lprint(ll.INFO, f"Starting trial {trial.number}...")
    trainer.train(train_loader, val_loader, targets)

    # Compute validation metrics
    val_fusion_loss, val_teacher_loss, val_fusion_wmae, val_teacher_wmae = trainer.validate(val_loader, targets)
    lprint(ll.INFO, f"Trial {trial.number} validation fusion wMAE: {val_fusion_wmae:.4f}")

    # Save the best model for this trial with HRMConfig
    trial_model_path = os.path.join(training_config.output_dir, f"fusion_model/trial_{trial.number}_best_fusion_model.pth")
    os.makedirs(os.path.join(training_config.output_dir, "fusion_model"), exist_ok=True)
    torch.save({
        'state_dict': trainer.fusion_model.state_dict(),
        'hrm_config': hrm_config.__dict__
    }, trial_model_path)
    lprint(ll.SUCCESS, f"Saved best model for trial {trial.number} to {trial_model_path}")

    # Evaluate the saved model for this trial
    lprint(ll.INFO, f"Evaluating model for trial {trial.number}...")
    wmae = evaluate_model(
        fusion_model=fusion_model,
        X_fusion= X_fusion,
        y_fusion_list=y_fusion_val_list,
        valid_mask_list=valid_mask_val_list,
        true_target=true_target_val,  # Use validation set true_target
        device=device,
        normalizer=normalizer
    )

    # Save trial metrics to TSV
    metrics = {
        'trial': trial.number,
        'val_fusion_loss': val_fusion_loss,
        'val_teacher_loss': val_teacher_loss,
        'val_fusion_wmae': val_fusion_wmae,
        'val_teacher_wmae': val_teacher_wmae,
        'eval_wmae': wmae
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_file = os.path.join(training_config.output_dir, 'trial_metrics.tsv')
    if os.path.exists(metrics_file):
        metrics_df.to_csv(metrics_file, mode='a', header=False, sep='\t', index=False)
    else:
        metrics_df.to_csv(metrics_file, sep='\t', index=False)
    lprint(ll.INFO, f"Saved metrics for trial {trial.number} to {metrics_file}")

    return val_fusion_wmae

def optimize_hyperparameters(targets: List[str], X_fusion: torch.Tensor, 
                            y_fusion_list: List[torch.Tensor], valid_mask_list: List[torch.Tensor], 
                            device: str, normalizer: DataNormalizer, true_targets: List[Tuple[torch.Tensor, torch.Tensor, int]], 
                            n_trials: int = 50) -> Dict[str, Any]:
    """
    Perform Bayesian hyperparameter search using Optuna.

    Args:
        targets (List[str]): List of target names.
        X_fusion (torch.Tensor): Input features for the fusion model.
        y_fusion_list (List[torch.Tensor]): List of target tensors.
        valid_mask_list (List[torch.Tensor]): List of validity masks.
        device (str): Device (CPU/GPU).
        normalizer (DataNormalizer): Normalizer for input/output data.
        true_targets (List[Tuple]): List of tuples (target_tensor, target_tensor, num_valid) for wMAE weights.
        n_trials (int): Number of trials for the search.

    Returns:
        Dict[str, Any]: Best hyperparameters found.
    """
    lprint(ll.INFO, f"Starting Bayesian optimization with {n_trials} trials...")
    # Set seed for Optuna sampler to ensure reproducibility
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, true_targets, targets, X_fusion, y_fusion_list, valid_mask_list, device, normalizer),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False
    )

    lprint(ll.SUCCESS, f"Best trial: {study.best_trial.number}")
    lprint(ll.SUCCESS, f"Best validation wMAE: {study.best_trial.value:.4f}")
    lprint(ll.SUCCESS, "Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        lprint(ll.SUCCESS, f"  {key}: {value}")

    return study.best_trial.params

def evaluate_model(fusion_model: FusionModel, X_fusion: torch.Tensor, y_fusion_list: List[torch.Tensor], 
                  valid_mask_list: List[torch.Tensor], 
                  true_target: List[Tuple[torch.Tensor, torch.Tensor, int]], 
                  device: str, normalizer: DataNormalizer) -> float:
    lprint(ll.INFO, "Evaluating fusion model...")

    # Move model to device and set to evaluation mode
    fusion_model = fusion_model.to(device)
    fusion_model.eval()

    # Compute weights for wMAE
    weights = compute_wmae_weights(true_target).to(device)

    # Prepare inputs
    y_fusion = torch.cat(y_fusion_list, dim=1).to(device)  # Shape [N, T]

 
    # Normalize inputs (X_fusion is not used in fusion_model, but included for consistency)
    X_fusion_normalized, y_fusion_normalized = normalizer.fit_transform(X_fusion.to(device), y_fusion)

    # Ensure y_fusion_normalized is a tensor
    if not isinstance(y_fusion_normalized, torch.Tensor):
        lprint(ll.ERROR, f"y_fusion_normalized is not a tensor, got type {type(y_fusion_normalized)}")
        raise ValueError(f"y_fusion_normalized must be a torch.Tensor, got {type(y_fusion_normalized)}")

    lprint(ll.DEBUG, f"y_fusion_normalized shape: {y_fusion_normalized.shape}")

    # Get predictions
    with torch.no_grad():
        predictions, _ = fusion_model(y_fusion_normalized)  # Shape [N, T]

    # Denormalize predictions
    predictions_denorm = normalizer.inverse_transform_y(predictions)  # Should return a tensor

    # Split predictions into list of tensors for each target
    #predictions_list = [predictions_denorm[:, i:i+1] for i in range(predictions_denorm.shape[1])]  # List of [N, 1]

    # Check if there are any valid data points
    if not any(len(mask.nonzero(as_tuple=True)[0]) > 0 for mask in valid_mask_list):
        lprint(ll.WARN, "No valid data for any target. Returning inf for wMAE.")
        return float('inf')

    # Compute wMAE using lists of tensors
    wmae = wMAE_loss(predictions, y_fusion_list, weights, valid_mask_list).item()

    lprint(ll.REPORT, f"Fusion model wMAE (denormalized) = {wmae:.4f}")

    return wmae

def fusion_main(targets: List[str], true_targets: List[Tuple[torch.Tensor, torch.Tensor, int]], 
                X_fusion: torch.Tensor, y_fusion_list: List[torch.Tensor], 
                valid_mask_list: List[torch.Tensor], device: str):
    """
    Modified version of fusion_main with Bayesian hyperparameter search and evaluation of all saved models.

    Args:
        targets (List[str]): List of target names.
        true_targets (List[Tuple]): List of tuples (target_tensor, target_tensor, num_valid) for wMAE weights.
        X_fusion (torch.Tensor): Input features for the fusion model.
        y_fusion_list (List[torch.Tensor]): List of target tensors.
        valid_mask_list (List[torch.Tensor]): List of validity masks.
        device (str): Device (CPU/GPU).
    """
    # Load normalizer from dataset
    X_fusion, y_fusion_list, valid_mask_list, normalizer = load_fusion_dataset(targets)

    # Perform hyperparameter optimization
    best_params = optimize_hyperparameters(targets, X_fusion, y_fusion_list, valid_mask_list, device, normalizer, true_targets, n_trials=50)

    hrm_config = HRMConfig(
        input_size=1,
        num_targets=len(targets),
        context_size=best_params["context_size"],
        low_hidden=best_params["low_hidden"],
        high_hidden=best_params["high_hidden"],
        act_eps=0.01,
        max_high_steps=best_params["max_high_steps"],
        max_low_steps=best_params["max_low_steps"],
        one_step_detach=False,
        dropout_prob=best_params["dropout_prob"],
        batch_norm=best_params["batch_norm"],
        gradient_clipping=best_params["gradient_clipping"],
        output_dir="./output",
        training_type='split',
        learning_rate=0.001,
        batch_size=best_params["batch_size"],
        max_epochs=_global.FUSION_EPOCHS,
        patience=_global.FUSION_PATIENCE,
        validation_split=0.2
    )

    training_config = TrainingConfig(
        learning_rate_fusion=best_params["learning_rate_fusion"],
        learning_rate_teachers=best_params["learning_rate_teachers"],
        batch_size=best_params["batch_size"],
        max_epochs=_global.FUSION_EPOCHS,
        patience=_global.FUSION_PATIENCE,
        min_delta=1e-4,
        output_dir="./output",
        checkpoint_interval=50,
        validation_split=0.2,
        device=device
    )

    # Split dataset into training and validation sets
    y_fusion = torch.cat(y_fusion_list, dim=1)
    valid_mask = torch.stack(valid_mask_list, dim=1)

    dataset = TensorDataset(X_fusion, y_fusion, valid_mask)
    train_size = int((1 - training_config.validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)

    # Extract validation set tensors
    X_fusion_val = torch.stack([x[0] for x in val_dataset])
    y_fusion_val = torch.stack([x[1] for x in val_dataset])
    valid_mask_val = torch.stack([x[2] for x in val_dataset])
    y_fusion_val_list = [y_fusion_val[:, i:i+1] for i in range(y_fusion_val.shape[1])]
    valid_mask_val_list = [valid_mask_val[:, i] for i in range(valid_mask_val.shape[1])]
    true_target_val = [
        (y_fusion_val_list[i], y_fusion_val_list[i], valid_mask_val_list[i].sum().item())
        for i in range(len(targets))
    ]

    # Load teacher models
    from training.results_manager import ResultsManager
    teacher_models = []
    for target in targets:
        try:
            model = ResultsManager.load_best_model(f"./output/{target}/best_model.pth")
            teacher_models.append(model)
            lprint(ll.SUCCESS, f"Loaded teacher model for {target}")
        except Exception as e:
            lprint(ll.ERROR, f"Failed to load teacher model for {target}: {str(e)}")
            raise

    # Train with best hyperparameters
    fusion_model = FusionModel(hrm_config)
    trainer = FusionTrainer(fusion_model, teacher_models, training_config, hrm_config, normalizer)

    lprint(ll.INFO, "Starting final training with best hyperparameters...")
    trainer.train(train_loader, val_loader, targets)
    lprint(ll.SUCCESS, "Final training completed")

    # Evaluate the final model
    lprint(ll.INFO, "Evaluating final model...")
    final_wmae = evaluate_model(
        fusion_model=fusion_model,
        X_fusion= X_fusion,
        y_fusion_list=y_fusion_val_list,
        valid_mask_list=valid_mask_val_list,
        true_target=true_target_val,
        device=device,
        normalizer=normalizer
    )
    lprint(ll.SUCCESS, f"Final model wMAE: {final_wmae:.4f}")

    # Predict with teacher models on sample data
    sample_X = X_fusion[:10]
    for i, (teacher, target) in enumerate(zip(teacher_models, targets)):
        prediction = predict_single_target(teacher, sample_X, normalizer, device=training_config.device)
        lprint(ll.INFO, f"Teacher {target} predictions: {prediction.squeeze().tolist()}")

    return fusion_model, final_wmae

if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    targets = ["Tg", "FFV", "Tc", "Density", "Rg"]
    X_fusion, y_fusion_list, valid_mask_list, normalizer = load_fusion_dataset(targets)
    
    true_targets = [
        (y_fusion_list[i], y_fusion_list[i], valid_mask_list[i].sum().item())
        for i in range(len(targets))
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fusion_model, final_wmae = fusion_main(targets, true_targets, X_fusion, y_fusion_list, valid_mask_list, device)
    lprint(ll.SUCCESS, f"Fusion model training completed with final wMAE: {final_wmae:.4f}")