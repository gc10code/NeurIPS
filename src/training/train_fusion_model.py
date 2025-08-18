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
from src._global import global_init
import os
import pandas as pd
from pathlib import Path
import optuna

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
    def __init__(self, fusion_model: FusionModel, teacher_models: List[nn.Module], config: TrainingConfig, hrm_config: HRMConfig):
        """
        Initialize the trainer for the FusionModel and teacher models.

        Args:
            fusion_model (FusionModel): The fusion model to train.
            teacher_models (List[nn.Module]): List of pre-trained teacher models.
            config (TrainingConfig): Training configuration.
            hrm_config (HRMConfig): HRM configuration.
        """
        self.fusion_model = fusion_model.to(config.device)
        self.teacher_models = [teacher.to(config.device) for teacher in teacher_models]
        self.config = config
        self.hrm_config = hrm_config
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
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_path = os.path.join(config.output_dir, "best_fusion_model.pth")

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], targets: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single training step for the fusion model and teachers.

        Args:
            batch: Tuple of (X_fusion, y_fusion, valid_mask).
            targets: List of target names for logging.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Fusion loss, average teacher loss,
                                                                         fusion wMAE, average teacher wMAE.
        """
        X_fusion, y_fusion, valid_mask = [x.to(self.device) for x in batch]
        B, T = y_fusion.shape

        # Calcola i pesi per wMAE
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
        for i, (teacher, target) in enumerate(zip(self.teacher_models, targets)):
            teacher_output = teacher(X_fusion)
            teacher_target = y_fusion[:, i:i+1]
            teacher_mask = valid_mask[:, i:i+1]
            teacher_weight = weights[i:i+1]
            teacher_loss = self.compute_loss(teacher_output, teacher_target, teacher_mask)
            teacher_wmae = wMAE_loss(teacher_output, teacher_target, teacher_weight, teacher_mask)
            teacher_losses.append(teacher_loss)
            teacher_wmaes.append(teacher_wmae)
            if teacher_mask.sum() == 0:
                lprint(ll.DEBUG, f"No valid data for teacher {target} in this batch")

        # Log per-target teacher losses
        for i, (target, loss) in enumerate(zip(targets, teacher_losses)):
            lprint(ll.DEBUG, f"Teacher {target} Train Loss: {loss.item():.4f}")

        # Combine fusion and teacher losses
        total_loss = self.hrm_config.alpha * fusion_loss + self.hrm_config.beta * sum(teacher_losses) / len(teacher_losses)
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

                if num_batches % self.config.checkpoint_interval == 0:
                    torch.save(self.fusion_model.state_dict(), 
                              os.path.join(self.config.output_dir, f"fusion_checkpoint_epoch_{epoch+1}.pth"))
                    for i, teacher in enumerate(self.teacher_models):
                        torch.save(teacher.state_dict(), 
                                  os.path.join(self.config.output_dir, f"teacher_{i}_checkpoint_epoch_{epoch+1}.pth"))

            avg_fusion_loss = total_fusion_loss / num_batches
            avg_teacher_loss = total_teacher_loss / num_batches
            avg_fusion_wmae = total_fusion_wmae / num_batches
            avg_teacher_wmae = total_teacher_wmae / num_batches
            lprint(ll.INFO, f"Epoch {epoch+1}: Train Fusion Loss = {avg_fusion_loss:.4f}, wMAE = {avg_fusion_wmae:.4f}, "
                            f"Teacher Loss = {avg_teacher_loss:.4f}, Teacher wMAE = {avg_teacher_wmae:.4f}")

            val_fusion_loss, val_teacher_loss, val_fusion_wmae, val_teacher_wmae = self.validate(val_loader, targets)
            lprint(ll.REPORT, f"Validation Fusion Loss = {val_fusion_loss:.4f}, wMAE = {val_fusion_wmae:.4f}, "
                              f"Teacher Loss = {val_teacher_loss:.4f}, Teacher wMAE = {val_teacher_wmae:.4f}")

            self.scheduler_fusion.step(val_fusion_loss)
            for scheduler in self.schedulers_teachers:
                scheduler.step(val_teacher_loss)

            if val_fusion_wmae < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_fusion_wmae
                self.patience_counter = 0
                torch.save(self.fusion_model.state_dict(), self.best_model_path)
                lprint(ll.SUCCESS, f"New best model saved with validation wMAE {val_fusion_wmae:.4f}")
            else:
                self.patience_counter += 1
                lprint(ll.WARN, f"No improvement in validation loss. Patience: {self.patience_counter}/{self.config.patience}")
                if self.patience_counter >= self.config.patience:
                    lprint(ll.EXIT, "Early stopping triggered")
                    break

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
        output = teacher(X)
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
        return X_fusion, y_fusion, valid_mask

    except Exception as e:
        lprint(ll.ERROR, f"Error loading fusion dataset: {str(e)}")
        raise

def objective(trial: optuna.Trial, targets: List[str], X_fusion: torch.Tensor, 
              y_fusion_list: List[torch.Tensor], valid_mask_list: List[torch.Tensor], 
              device: str) -> float:
    """
    Funzione obiettivo per la ricerca bayesiana degli iperparametri con Optuna.

    Args:
        trial (optuna.Trial): Oggetto trial di Optuna per campionare gli iperparametri.
        targets (List[str]): Lista dei nomi dei target.
        X_fusion (torch.Tensor): Feature di input per il modello di fusione.
        y_fusion_list (List[torch.Tensor]): Lista di tensori per i target.
        valid_mask_list (List[torch.Tensor]): Lista di maschere di validità.
        device (str): Dispositivo (CPU/GPU).

    Returns:
        float: Perdita di validazione media del modello di fusione.
    """
    hrm_config = HRMConfig(
        input_size=1,
        num_targets=len(targets),
        context_size=trial.suggest_int("context_size", 64, 256),
        low_hidden=trial.suggest_int("low_hidden", 128, 512),
        high_hidden=trial.suggest_int("high_hidden", 256, 1024),
        act_eps=0.01,
        max_high_steps=trial.suggest_int("max_high_steps", 5, 20),
        max_low_steps=trial.suggest_int("max_low_steps", 3, 10),
        alpha=trial.suggest_float("alpha", 0.5, 1.5),
        beta=trial.suggest_float("beta", 0.1, 1.0),
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
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)

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
    trainer = FusionTrainer(fusion_model, teacher_models, training_config, hrm_config)

    lprint(ll.INFO, f"Starting trial {trial.number}...")
    trainer.train(train_loader, val_loader, targets)

    val_fusion_loss, _ = trainer.validate(val_loader, targets)
    lprint(ll.INFO, f"Trial {trial.number} validation fusion loss: {val_fusion_loss:.4f}")

    return val_fusion_loss

def optimize_hyperparameters(targets: List[str], X_fusion: torch.Tensor, 
                            y_fusion_list: List[torch.Tensor], valid_mask_list: List[torch.Tensor], 
                            device: str, n_trials: int = 50) -> Dict[str, Any]:
    """
    Esegue la ricerca bayesiana degli iperparametri utilizzando Optuna.

    Args:
        targets (List[str]): Lista dei nomi dei target.
        X_fusion (torch.Tensor): Feature di input per il modello di fusione.
        y_fusion_list (List[torch.Tensor]): Lista di tensori per i target.
        valid_mask_list (List[torch.Tensor]): Lista di maschere di validità.
        device (str): Dispositivo (CPU/GPU).
        n_trials (int): Numero di prove per la ricerca.

    Returns:
        Dict[str, Any]: Migliori iperparametri trovati.
    """
    lprint(ll.INFO, f"Starting Bayesian optimization with {n_trials} trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, targets, X_fusion, y_fusion_list, valid_mask_list, device),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False
    )

    lprint(ll.SUCCESS, f"Best trial: {study.best_trial.number}")
    lprint(ll.SUCCESS, f"Best validation loss: {study.best_trial.value:.4f}")
    lprint(ll.SUCCESS, "Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        lprint(ll.SUCCESS, f"  {key}: {value}")

    return study.best_trial.params

def fusion_main(targets: List[str], true_target: List[Tuple[torch.Tensor, torch.Tensor, int]], 
                X_fusion: torch.Tensor, y_fusion_list: List[torch.Tensor], 
                valid_mask_list: List[torch.Tensor], device: str):
    """
    Versione modificata di fusion_main con ricerca bayesiana degli iperparametri.

    Args:
        targets (List[str]): Lista dei nomi dei target.
        true_target (List[Tuple]): Lista di tuple (target_tensor, target_tensor, num_valid) per calcolare i pesi wMAE.
        X_fusion (torch.Tensor): Feature di input per il modello di fusione.
        y_fusion_list (List[torch.Tensor]): Lista di tensori per i target.
        valid_mask_list (List[torch.Tensor]): Lista di maschere di validità.
        device (str): Dispositivo (CPU/GPU).
    """
    best_params = optimize_hyperparameters(targets, X_fusion, y_fusion_list, valid_mask_list, device, n_trials=50)

    hrm_config = HRMConfig(
        input_size=1,
        num_targets=len(targets),
        context_size=best_params["context_size"],
        low_hidden=best_params["low_hidden"],
        high_hidden=best_params["high_hidden"],
        act_eps=0.01,
        max_high_steps=best_params["max_high_steps"],
        max_low_steps=best_params["max_low_steps"],
        alpha=best_params["alpha"],
        beta=best_params["beta"],
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

    y_fusion = torch.cat(y_fusion_list, dim=1)
    valid_mask = torch.stack(valid_mask_list, dim=1)

    dataset = TensorDataset(X_fusion, y_fusion, valid_mask)
    train_size = int((1 - training_config.validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)

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
    trainer = FusionTrainer(fusion_model, teacher_models, training_config, hrm_config)

    lprint(ll.INFO, "Starting final training with best hyperparameters...")
    trainer.train(train_loader, val_loader, targets)
    lprint(ll.SUCCESS, "Final training completed")

    sample_X = X_fusion[:10]
    for i, (teacher, target) in enumerate(zip(teacher_models, targets)):
        prediction = predict_single_target(teacher, sample_X, device=training_config.device)
        lprint(ll.INFO, f"Teacher {target} predictions: {prediction.squeeze().tolist()}")

if __name__ == "__main__":
    targets = ["Tg", "FFV", "Tc", "Density", "Rg"]
    X_fusion, y_fusion_list, valid_mask_list = load_fusion_dataset(targets)
    
    true_target = [
        (y_fusion_list[i], y_fusion_list[i], valid_mask_list[i].sum().item())
        for i in range(len(targets))
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fusion_main(targets, true_target, X_fusion, y_fusion_list, valid_mask_list, device)