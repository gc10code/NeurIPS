import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Union
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import numpy as np
from pathlib import Path
import signal
import multiprocessing as mp
from datetime import datetime
import gc

from src.utils.activations import ACTIVATION_FUNCTIONS
from src.models.rprop_mlp import RPropMLP
from src.models.fusion_model import TransformerFusionModel
from src.config.model_config import FusionConfig
from src.config.config_manager import ConfigManager
from src.training.data_preparation import create_dataloader
from src.utils.metrics import wMAE_loss, compute_wmae_weights
import src.utils.system_utils as su
import src.training.train_rprop_mlp as train_rprop
from src.training.data_preparation import prepare_data
from src.training.results_manager import ResultsManager
from src.utils.system_utils import *
from src.utils.logging import lprint, LoggingLevels as ll

# Setup
su.set_seed(42)
device = su.setup_device()

import torch
from src.utils.logging import lprint, LoggingLevels as ll


def generate_bayesian_search(n_samples: int, n_features: int, teacher_output_sizes: List[int], 
                            problem_type: str = 'regression', max_combinations: int = 20) -> tuple:
    search_space = [
        Real(1e-5, 5e-4, name='learning_rate', prior='log-uniform'),
        Integer(8, min(n_samples // 4, 64), name='batch_size'),
        Real(0.1, 0.5, name='dropout_prob'),
        Integer(1, 3, name='num_layers'),
        Integer(64, 256, name='hidden_dim'),
        Integer(1, 5, name='nhead'),
        Real(0.6, 1.0, name='alpha'),
        Real(0.0, 0.000016, name='beta')
    ]

    report = (
        f"Bayesian search space defined with {len(search_space)} dimensions:\n"
        f"- learning_rate: [1e-5, 5e-4]\n"
        f"- batch_size: [8, {min(n_samples // 4, 64)}]\n"
        f"- dropout_prob: [0.1, 0.5]\n"
        f"- num_layers: [1, 3]\n"
        f"- hidden_dim: [64, 256]\n"
        f"- nhead: [1, 5]\n"
        f"- alpha: [0.6, 1.0]\n"
        f"- beta: [0.0, 0.000016]\n"
        f"Dataset: n_samples={n_samples}, n_features={n_features}, n_outputs={len(teacher_output_sizes)}"
    )

    return search_space, report

def train_model(model: TransformerFusionModel, dataloader: DataLoader, teacher_models: list, 
                weights: torch.Tensor, alpha: float, beta: float, config) -> float:
    lprint(ll.INFO, "Starting model training")
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        criterion = wMAE_loss
        
        model.train()
        best_val_loss = float('inf')
        patience = config.patience if hasattr(config, 'patience') else 10
        counter = 0
        
        for epoch in range(config.max_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                inputs = batch[0].to(config.device)  # [batch_size, 5, 614]
                true_targets = batch[1].to(config.device)  # [batch_size, 5, 1]
                mask = batch[2].to(config.device)  # [batch_size, 5]
                
                try:
                    teacher_outputs = []
                    for i, teacher in enumerate(teacher_models):
                        valid_indices = mask[:, i]
                        if valid_indices.any():
                            teacher_out = teacher(inputs[valid_indices, i]).detach()
                            full_out = torch.zeros(inputs.shape[0], 1, device=config.device)
                            full_out[valid_indices] = teacher_out
                            teacher_outputs.append(full_out)
                        else:
                            teacher_outputs.append(torch.zeros(inputs.shape[0], 1, device=config.device))
                except Exception as e:
                    lprint(ll.ERROR, f"Error in epoch {epoch+1}: {str(e)}")
                    raise
                
                optimizer.zero_grad()
                outputs = model(teacher_outputs, mask=mask)  # [batch_size, 5]
                
                loss_sup = 0.0
                valid_batches = 0
                for i in range(len(true_targets[0])):
                    valid_indices = mask[:, i]
                    if valid_indices.any():
                        valid_outputs = outputs[valid_indices, i:i+1]
                        valid_targets = true_targets[valid_indices, i]
                        loss_sup += criterion([valid_outputs], [valid_targets], weights[i:i+1])
                        valid_batches += 1
                loss_sup = loss_sup / max(valid_batches, 1)
                
                loss_dist = sum(torch.mean((o - t) ** 2) for o, t in zip([outputs[:, i:i+1] for i in range(len(teacher_outputs))], teacher_outputs))
                loss = alpha * loss_sup + beta * loss_dist
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            per_target_mae = []
            for i in range(len(true_targets[0])):
                valid_indices = mask[:, i]
                if valid_indices.any():
                    mae = torch.mean(torch.abs(outputs[valid_indices, i] - true_targets[valid_indices, i])).item()
                    per_target_mae.append(mae)
                else:
                    per_target_mae.append(float('nan'))
            lprint(ll.INFO, f"Epoch {epoch+1}/{config.max_epochs}, Loss: {avg_loss:.4f}, Per-target MAE: {per_target_mae}")
            
            if avg_loss < best_val_loss - getattr(config, 'min_delta', 1e-4):
                best_val_loss = avg_loss
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                lprint(ll.INFO, f"Early stopping at epoch {epoch+1}")
                break
        
        lprint(ll.INFO, f"Training completed with best validation loss: {best_val_loss:.4f}")
        return best_val_loss
    except Exception as e:
        lprint(ll.ERROR, f"Error in train_model: {str(e)}")
        raise

from src.utils.logging import lprint, LoggingLevels as ll

def objective_function(params, metadata_file: str, descriptor_files: list[str], targets: list[str], 
                      teacher_models: list, weights: torch.Tensor, config) -> float:
    try:
        lprint(ll.INFO, f"Evaluating objective function with params: {params}")
        lr, batch_size, dropout_prob, num_layers, hidden_dim, nhead, alpha, beta = params
        
        config.learning_rate = lr
        config.batch_size = int(batch_size)
        config.dropout_prob = dropout_prob
        
        dataloader = create_dataloader(metadata_file, descriptor_files, targets, config.batch_size, 
                                      shuffle=config.shuffle, num_workers=config.num_workers, pin_memory=config.pin_memory)
        
        teacher_output_sizes = [teacher.output_size for teacher in teacher_models]
        model = TransformerFusionModel(
            teacher_output_sizes=teacher_output_sizes,
            hidden_dim=int(hidden_dim),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dropout=dropout_prob,
            batch_norm=config.batch_norm
        ).to(config.device)
        lprint(ll.INFO, "Transformer fusion model initialized")
        
        val_loss = train_model(model, dataloader, teacher_models, weights, alpha, beta, config)
        lprint(ll.INFO, f"Objective function evaluation completed with val_loss: {val_loss:.4f}")
        return val_loss
    except Exception as e:
        lprint(ll.ERROR, f"Error in objective_function: {str(e)}")
        raise

from skopt import gp_minimize
from skopt.space import Real, Integer
from pathlib import Path
import gc
import torch
from datetime import datetime
from src.utils.logging import lprint, LoggingLevels as ll

def fusion_model_main(targets: list[str], metadata_file: str, descriptor_files: list[str], project_root: str, 
                     device, max_combination: int = 20):
    lprint(ll.INFO, f"=== Transformer Fusion Model Training Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    try:
        n_models = len(targets)
        teacher_models = []
        teacher_input_sizes = []
        for i, target in enumerate(targets):
            lprint(ll.INFO, f"Loading teacher model {i+1}/{n_models} for target {target}")
            model_path = Path(f"{project_root}/output/{target}/best_model.pth")
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model = ResultsManager.load_best_model(model_path=model_path)
            model.eval()
            model.to(device)
            teacher_models.append(model)
            teacher_input_sizes.append(model.input_size)
            lprint(ll.INFO, f"Teacher model for {target} loaded, input_size={model.input_size}")

        # Verify input size consistency
        if len(set(teacher_input_sizes)) > 1:
            raise ValueError(f"Inconsistent input sizes for teacher models: {teacher_input_sizes}")
        
        weights = compute_wmae_weights(descriptor_files, targets).to(device)
        lprint(ll.INFO, f"Computed wMAE weights: {weights.tolist()}")

        fusion_config = ConfigManager.create_default_config("fusion")
        fusion_config.output_dir = f"{fusion_config.output_dir}/transformer_fusion_model"
        Path(fusion_config.output_dir).mkdir(parents=True, exist_ok=True)
        lprint(ll.INFO, f"Output directory: {fusion_config.output_dir}")

        # Estimate dataset size
        metadata = np.loadtxt(metadata_file, delimiter='\t', dtype=str, skiprows=1)
        n_samples = metadata.shape[0]
        data = np.loadtxt(descriptor_files[0], delimiter='\t')
        n_features = data.shape[1] - 1
        
        teacher_output_sizes = [teacher.output_size for teacher in teacher_models]
        space, report = generate_bayesian_search(n_samples, n_features, teacher_output_sizes, 
                                                problem_type=fusion_config.problem_type, 
                                                max_combinations=max_combination)
        lprint(ll.INFO, report)

        result = gp_minimize(
            lambda params: objective_function(params, metadata_file, descriptor_files, targets, 
                                             teacher_models, weights, fusion_config),
            space,
            n_calls=max_combination,
            random_state=42,
            verbose=True
        )
        lprint(ll.INFO, "Bayesian optimization completed")

        best_params = result.x
        lr, batch_size, dropout_prob, num_layers, hidden_dim, nhead, alpha, beta = best_params
        
        fusion_config.learning_rate = lr
        fusion_config.batch_size = int(batch_size)
        fusion_config.dropout_prob = dropout_prob
        lprint(ll.INFO, f"Best hyperparameters: lr={lr}, batch_size={batch_size}, dropout={dropout_prob}, "
                       f"num_layers={num_layers}, hidden_dim={hidden_dim}, nhead={nhead}, alpha={alpha}, beta={beta}")

        dataloader = create_dataloader(metadata_file, descriptor_files, targets, fusion_config.batch_size, 
                                      shuffle=fusion_config.shuffle, num_workers=fusion_config.num_workers, 
                                      pin_memory=fusion_config.pin_memory)

        final_model = TransformerFusionModel(
            teacher_output_sizes=teacher_output_sizes,
            hidden_dim=int(hidden_dim),
            nhead=int(nhead),
            num_layers=int(num_layers),
            dropout=dropout_prob,
            batch_norm=fusion_config.batch_norm
        ).to(device)
        
        final_loss = train_model(final_model, dataloader, teacher_models, weights, alpha, beta, fusion_config)
        lprint(ll.INFO, f"Final model trained with loss: {final_loss:.4f}")

        output_path = Path(f"{fusion_config.output_dir}/best_transformer_fusion_model.pth")
        torch.save(final_model.state_dict(), output_path)
        lprint(ll.INFO, f"Model saved to {output_path}")

        return final_model, final_loss
    except Exception as e:
        lprint(ll.ERROR, f"Error in fusion_model_main: {str(e)}")
        raise
    finally:
        lprint(ll.INFO, "Cleaning up resources")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()