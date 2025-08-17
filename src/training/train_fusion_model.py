from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from src.config.model_config import HRMConfig
from src.utils.logging import lprint, LoggingLevels as ll
from src.models.fusion_model import FusionModel
from src.models.rprop_mlp import RPropMLP
import os
import pandas as pd
from pathlib import Path

@dataclass
class TrainingConfig:
    """Configuration for training the FusionModel and refining teachers."""
    learning_rate_fusion: float = 0.001
    learning_rate_teachers: float = 0.001  # Increased from 0.0001 to encourage teacher improvement
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    output_dir: str = "./output"
    checkpoint_interval: int = 50
    validation_split: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class FusionTrainer:
    def __init__(self, fusion_model: FusionModel, teacher_models: List[nn.Module], config: TrainingConfig):
        """
        Initialize the trainer for the FusionModel and teacher models.

        Args:
            fusion_model (FusionModel): The fusion model to train.
            teacher_models (List[nn.Module]): List of pre-trained teacher models.
            config (TrainingConfig): Training configuration.
        """
        self.fusion_model = fusion_model.to(config.device)
        self.teacher_models = [teacher.to(config.device) for teacher in teacher_models]
        self.config = config
        self.device = config.device
        self.optimizer_fusion = optim.Adam(self.fusion_model.parameters(), lr=config.learning_rate_fusion)
        self.optimizers_teachers = [optim.Adam(teacher.parameters(), lr=config.learning_rate_teachers) 
                                   for teacher in self.teacher_models]
        self.scheduler_fusion = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_fusion, mode='min', factor=0.5, patience=5
        )
        self.schedulers_teachers = [
            optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
            for opt in self.optimizers_teachers
        ]
        self.criterion = nn.L1Loss(reduction='none')  # For regression tasks
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_path = os.path.join(config.output_dir, "best_fusion_model.pth")

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
        loss = self.criterion(outputs, targets)  # [B, num_targets] or [B, 1]
        masked_loss = loss * valid_mask.float()  # Apply mask
        return masked_loss.sum() / valid_mask.sum().clamp(min=1)  # Average over valid entries

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], targets: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single training step for the fusion model and teachers.

        Args:
            batch: Tuple of (X_fusion, y_fusion, valid_mask).
            targets: List of target names for logging.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Fusion loss and average teacher loss.
        """
        X_fusion, y_fusion, valid_mask = [x.to(self.device) for x in batch]
        B, T = y_fusion.shape

        # Fusion model forward pass
        self.optimizer_fusion.zero_grad()
        outputs, info = self.fusion_model(y_fusion)  # [B, num_targets]
        fusion_loss = self.compute_loss(outputs, y_fusion, valid_mask)

        # Teacher refinement
        teacher_losses = []
        for i, (teacher, optimizer, target) in enumerate(zip(self.teacher_models, self.optimizers_teachers, targets)):
            optimizer.zero_grad()
            teacher_output = teacher(X_fusion)  # [B, 1]
            teacher_target = y_fusion[:, i:i+1]  # [B, 1]
            teacher_mask = valid_mask[:, i:i+1]  # [B, 1]
            teacher_loss = self.compute_loss(teacher_output, teacher_target, teacher_mask)
            teacher_losses.append(teacher_loss)

            # Backpropagate for teacher if mask has valid entries
            if teacher_mask.sum() > 0:
                teacher_loss.backward()
                optimizer.step()
            else:
                lprint(ll.DEBUG, f"No valid data for teacher {target} in this batch")

        # Backpropagate for fusion model
        fusion_loss.backward()
        self.optimizer_fusion.step()

        # Log per-target teacher losses
        for i, (target, loss) in enumerate(zip(targets, teacher_losses)):
            lprint(ll.DEBUG, f"Teacher {target} Train Loss: {loss.item():.4f}")

        return fusion_loss, sum(teacher_losses) / len(teacher_losses)

    def validate(self, val_loader: DataLoader, targets: List[str]) -> Tuple[float, float]:
        """
        Validate the fusion model and teachers.

        Args:
            val_loader (DataLoader): Validation data loader.
            targets (List[str]): List of target names for logging.

        Returns:
            Tuple[float, float]: Average fusion validation loss and teacher validation loss.
        """
        self.fusion_model.eval()
        for teacher in self.teacher_models:
            teacher.eval()

        total_fusion_loss = 0.0
        total_teacher_loss = 0.0
        num_batches = 0
        teacher_losses_per_target = [0.0] * len(targets)

        with torch.no_grad():
            for batch in val_loader:
                X_fusion, y_fusion, valid_mask = [x.to(self.device) for x in batch]
                outputs, _ = self.fusion_model(y_fusion)  # [B, num_targets]
                fusion_loss = self.compute_loss(outputs, y_fusion, valid_mask)
                total_fusion_loss += fusion_loss.item()

                teacher_loss = 0.0
                for i, (teacher, target) in enumerate(zip(self.teacher_models, targets)):
                    teacher_output = teacher(X_fusion)  # [B, 1]
                    teacher_target = y_fusion[:, i:i+1]  # [B, 1]
                    teacher_mask = valid_mask[:, i:i+1]  # [B, 1]
                    t_loss = self.compute_loss(teacher_output, teacher_target, teacher_mask).item()
                    teacher_loss += t_loss
                    teacher_losses_per_target[i] += t_loss
                total_teacher_loss += teacher_loss / len(self.teacher_models)
                num_batches += 1

        # Log per-target teacher validation losses
        for target, t_loss in zip(targets, teacher_losses_per_target):
            lprint(ll.INFO, f"Teacher {target} Validation Loss: {t_loss / num_batches:.4f}")

        self.fusion_model.train()
        for teacher in self.teacher_models:
            teacher.train()

        return total_fusion_loss / num_batches, total_teacher_loss / num_batches

    def compute_prediction_errors(self, val_loader: DataLoader, targets: List[str]) -> None:
        """
        Compute and log the differences between predicted and true values for non-synthetic data.

        Args:
            val_loader (DataLoader): Validation data loader.
            targets (List[str]): List of target names (e.g., ["Tg", "FFV", "Tc", "Density", "Rg"]).
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
                fusion_preds, _ = self.fusion_model(y_fusion)  # [B, num_targets]
                all_fusion_preds.append(fusion_preds.cpu())
                all_true.append(y_fusion.cpu())
                all_masks.append(valid_mask.cpu())

                teacher_preds = []
                for teacher in self.teacher_models:
                    teacher_pred = teacher(X_fusion)  # [B, 1]
                    teacher_preds.append(teacher_pred.cpu())
                all_teacher_preds.append(torch.cat(teacher_preds, dim=-1))  # [B, num_targets]

        # Concatenate all predictions and true values
        fusion_preds = torch.cat(all_fusion_preds, dim=0)  # [num_samples, num_targets]
        teacher_preds = torch.cat(all_teacher_preds, dim=0)  # [num_samples, num_targets]
        true_values = torch.cat(all_true, dim=0)  # [num_samples, num_targets]
        valid_mask = torch.cat(all_masks, dim=0)  # [num_samples, num_targets]

        # Compute and log errors for non-synthetic data
        lprint(ll.REPORT, "Prediction Errors for Non-Synthetic Data (Validation Set):")
        for i, target in enumerate(targets):
            # Select non-synthetic data
            mask = valid_mask[:, i]  # [num_samples]
            if mask.sum() == 0:
                lprint(ll.WARN, f"No non-synthetic data for target {target}")
                continue

            fusion_pred = fusion_preds[:, i][mask]  # [num_valid]
            teacher_pred = teacher_preds[:, i][mask]  # [num_valid]
            true_val = true_values[:, i][mask]  # [num_valid]

            # Compute absolute errors
            fusion_errors = torch.abs(fusion_pred - true_val)
            teacher_errors = torch.abs(teacher_pred - true_val)

            # Compute metrics
            fusion_mae = fusion_errors.mean().item()
            teacher_mae = teacher_errors.mean().item()
            fusion_rmse = torch.sqrt(torch.mean(fusion_errors ** 2)).item()
            teacher_rmse = torch.sqrt(torch.mean(teacher_errors ** 2)).item()

            # Log detailed results
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
        os.makedirs(self.config.output_dir, exist_ok=True)

        for epoch in range(self.config.max_epochs):
            self.fusion_model.train()
            for teacher in self.teacher_models:
                teacher.train()

            total_fusion_loss = 0.0
            total_teacher_loss = 0.0
            num_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.max_epochs}"):
                fusion_loss, teacher_loss = self.train_step(batch, targets)
                total_fusion_loss += fusion_loss.item()
                total_teacher_loss += teacher_loss.item()
                num_batches += 1

                if num_batches % self.config.checkpoint_interval == 0:
                    torch.save(self.fusion_model.state_dict(), 
                              os.path.join(self.config.output_dir, f"fusion_checkpoint_epoch_{epoch+1}.pth"))
                    for i, teacher in enumerate(self.teacher_models):
                        torch.save(teacher.state_dict(), 
                                  os.path.join(self.config.output_dir, f"teacher_{i}_checkpoint_epoch_{epoch+1}.pth"))

            avg_fusion_loss = total_fusion_loss / num_batches
            avg_teacher_loss = total_teacher_loss / num_batches
            lprint(ll.INFO, f"Epoch {epoch+1}: Train Fusion Loss = {avg_fusion_loss:.4f}, Teacher Loss = {avg_teacher_loss:.4f}")

            # Validation
            val_fusion_loss, val_teacher_loss = self.validate(val_loader, targets)
            lprint(ll.REPORT, f"Validation Fusion Loss = {val_fusion_loss:.4f}, Teacher Loss = {val_teacher_loss:.4f}")

            # Update schedulers
            self.scheduler_fusion.step(val_fusion_loss)
            for scheduler in self.schedulers_teachers:
                scheduler.step(val_teacher_loss)

            # Early stopping
            if val_fusion_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_fusion_loss
                self.patience_counter = 0
                torch.save(self.fusion_model.state_dict(), self.best_model_path)
                lprint(ll.SUCCESS, f"New best model saved with validation loss {val_fusion_loss:.4f}")
            else:
                self.patience_counter += 1
                lprint(ll.WARN, f"No improvement in validation loss. Patience: {self.patience_counter}/{self.config.patience}")
                if self.patience_counter >= self.config.patience:
                    lprint(ll.EXIT, "Early stopping triggered")
                    break

        # Compute and log prediction errors
        lprint(ll.INFO, "Computing prediction errors for non-synthetic data...")
        self.compute_prediction_errors(val_loader, targets)

def predict_single_target(teacher: nn.Module, X: torch.Tensor, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> torch.Tensor:
    """
    Make a single-target prediction using a specific teacher model.

    Args:
        teacher (nn.Module): The teacher model to use for prediction.
        X (torch.Tensor): Input features [B, N].
        device (str): Device to run the prediction on.

        Returns:
            torch.Tensor: Predicted values [B, 1].
    """
    teacher.eval()
    X = X.to(device)
    with torch.no_grad():
        output = teacher(X)  # [B, 1]
    return output

def load_fusion_dataset(targets: List[str]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Load fusion dataset from TSV files and create X_fusion, y_fusion, and valid_mask.

    Args:
        targets (List[str]): List of target names (e.g., ["Density", "FFV", "Rg", "Tc", "Tg"]).

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
            - X_fusion: Tensor of shape (N, D) containing descriptors.
            - y_fusion: List of tensors, each of shape (N, 1), containing target values.
            - valid_mask: List of boolean tensors, each of shape (N,), indicating real values.
    Raises:
        ValueError: If files are missing, columns are invalid, or data shapes mismatch.
    """
    try:
        lprint(ll.INFO, "Loading fusion dataset")
        fusion_descriptors_file = Path("./data/fusion_descriptors.tsv") 
        fusion_target_file = Path("./data/fusion_target.tsv")
        fusion_meta_file = Path("./data/fusion_meta.tsv")

        # Validate inputs
        if not all(Path(f).exists() for f in [fusion_descriptors_file, fusion_target_file, fusion_meta_file]):
            missing = [f for f in [fusion_descriptors_file, fusion_target_file, fusion_meta_file] if not Path(f).exists()]
            lprint(ll.ERROR, f"Missing files: {missing}")
            raise ValueError(f"Missing files: {missing}")

        # Read descriptors
        descriptors_df = pd.read_csv(fusion_descriptors_file, sep='\t')
        if 'id' not in descriptors_df.columns:
            lprint(ll.ERROR, "fusion_descriptors.tsv missing 'id' column")
            raise ValueError("fusion_descriptors.tsv missing 'id' column")
        lprint(ll.DEBUG, f"Loaded fusion_descriptors.tsv with shape {descriptors_df.shape}")

        # Read targets
        target_df = pd.read_csv(fusion_target_file, sep='\t')
        if 'id' not in target_df.columns:
            lprint(ll.ERROR, "fusion_target.tsv missing 'id' column")
            raise ValueError("fusion_target.tsv missing 'id' column")
        missing_targets = [t for t in targets if t not in target_df.columns]
        if missing_targets:
            lprint(ll.ERROR, f"Missing target columns in fusion_target.tsv: {missing_targets}")
            raise ValueError(f"Missing target columns: {missing_targets}")
        lprint(ll.DEBUG, f"Loaded fusion_target.tsv with shape {target_df.shape}")

        # Read meta (for validity masks)
        meta_df = pd.read_csv(fusion_meta_file, sep='\t')
        if 'id' not in meta_df.columns:
            lprint(ll.ERROR, "fusion_meta.tsv missing 'id' column")
            raise ValueError("fusion_meta.tsv missing 'id' column")
        missing_valid_columns = [f"{t}_valid" for t in targets if f"{t}_valid" not in meta_df.columns]
        if missing_valid_columns:
            lprint(ll.ERROR, f"Missing validity columns in fusion_meta.tsv: {missing_valid_columns}")
            raise ValueError(f"Missing validity columns: {missing_valid_columns}")
        lprint(ll.DEBUG, f"Loaded fusion_meta.tsv with shape {meta_df.shape}")

        # Verify ID consistency
        descriptors_ids = set(descriptors_df['id'])
        target_ids = set(target_df['id'])
        meta_ids = set(meta_df['id'])
        if not (descriptors_ids == target_ids == meta_ids):
            lprint(ll.ERROR, "ID mismatch between fusion_descriptors.tsv, fusion_target.tsv, and fusion_meta.tsv")
            raise ValueError("ID mismatch between files")

        # Sort by ID to ensure alignment
        descriptors_df = descriptors_df.sort_values('id').reset_index(drop=True)
        target_df = target_df.sort_values('id').reset_index(drop=True)
        meta_df = meta_df.sort_values('id').reset_index(drop=True)

        # Verify sample count
        num_samples = descriptors_df.shape[0]
        if target_df.shape[0] != num_samples or meta_df.shape[0] != num_samples:
            lprint(ll.ERROR, f"Sample count mismatch: descriptors={num_samples}, target={target_df.shape[0]}, meta={meta_df.shape[0]}")
            raise ValueError("Sample count mismatch between files")

        # Create X_fusion
        X_fusion = torch.tensor(descriptors_df.drop(columns=['id']).values, dtype=torch.float32)
        lprint(ll.DEBUG, f"X_fusion shape: {X_fusion.shape}")

        # Create y_fusion
        y_fusion = [torch.tensor(target_df[target].values[:, None], dtype=torch.float32).clone().detach() for target in targets]
        lprint(ll.DEBUG, f"y_fusion shapes: {[y.shape for y in y_fusion]}")

        # Create valid_mask
        valid_mask = [torch.tensor(meta_df[f"{target}_valid"].values, dtype=torch.bool).clone().detach() for target in targets]
        lprint(ll.DEBUG, f"valid_mask shapes: {[m.shape for m in valid_mask]}")

        # Validate shapes
        num_samples = X_fusion.shape[0]
        if any(y.shape[0] != num_samples for y in y_fusion) or any(m.shape[0] != num_samples for m in valid_mask):
            lprint(ll.ERROR, f"Shape mismatch between X_fusion ({num_samples}), y_fusion {[y.shape[0] for y in y_fusion]}, and valid_mask {[m.shape[0] for m in valid_mask]}")
            raise ValueError("Shape mismatch between X_fusion, y_fusion, and valid_mask")

        # Log valid data statistics
        for target, mask in zip(targets, valid_mask):
            valid_count = mask.sum().item()
            lprint(ll.INFO, f"{target}: {valid_count} real values, {num_samples - valid_count} imputed")

        lprint(ll.INFO, "Fusion dataset loaded successfully")
        return X_fusion, y_fusion, valid_mask

    except Exception as e:
        lprint(ll.ERROR, f"Error loading fusion dataset: {str(e)}")
        raise

def main():
    # Example configuration (adjust based on your needs)
    hrm_config = HRMConfig(
        input_size=1,  # Since FusionModel takes teacher outputs as input
        num_targets=5,
        context_size=128,
        low_hidden=256,
        high_hidden=512,
        act_eps=0.01,
        max_high_steps=10,
        max_low_steps=5,
        dropout_prob=0.1,
        batch_norm=True,
        output_dir="./output"
    )

    training_config = TrainingConfig(
        learning_rate_fusion=0.001,
        learning_rate_teachers=0.001,  # Increased to match fusion model
        batch_size=32,
        max_epochs=100,
        patience=10,
        output_dir="./output"
    )
    targets = ["Tg", "FFV", "Tc", "Density", "Rg"]

    # Load dataset
    X_fusion, y_fusion_list, valid_mask_list = load_fusion_dataset(targets)

    # Convert y_fusion and valid_mask from lists to tensors
    y_fusion = torch.cat(y_fusion_list, dim=1)  # [num_samples, num_targets]
    valid_mask = torch.stack(valid_mask_list, dim=1)  # [num_samples, num_targets]
    lprint(ll.DEBUG, f"y_fusion shape: {y_fusion.shape}, valid_mask shape: {valid_mask.shape}")

    # Create dataset and dataloaders
    dataset = TensorDataset(X_fusion, y_fusion, valid_mask)
    train_size = int((1 - training_config.validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)

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

    # Initialize fusion model
    fusion_model = FusionModel(hrm_config)

    # Initialize trainer
    trainer = FusionTrainer(fusion_model, teacher_models, training_config)

    # Train
    lprint(ll.INFO, "Starting training...")
    trainer.train(train_loader, val_loader, targets)
    lprint(ll.SUCCESS, "Training completed")

    # Example single-target prediction
    sample_X = X_fusion[:10]  # Example input
    for i, (teacher, target) in enumerate(zip(teacher_models, targets)):
        prediction = predict_single_target(teacher, sample_X, device=training_config.device)
        lprint(ll.INFO, f"Teacher {target} predictions: {prediction.squeeze().tolist()}")

if __name__ == "__main__":
    main()