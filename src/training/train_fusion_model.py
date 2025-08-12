import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Union
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import numpy as np
from pathlib import Path
import logging
import traceback
import signal
import multiprocessing as mp
from datetime import datetime
import gc

from src.utils.activations import ACTIVATION_FUNCTIONS
from src.models.rprop_mlp import RPropMLP
from src.models.fusion_model import FusionModel
from src.config.model_config import FusionConfig
from src.config.config_manager import ConfigManager
from src.config.grid_search_config import generate_bayesian_search
from src.utils.metrics import wMAE_loss, compute_wmae_weights
from src.utils.logging import setup_logging
import src.utils.system_utils as su
import src.training.train_rprop_mlp as train_rprop
from src.training.data_preparation import prepare_data
from src.training.results_manager import ResultsManager
from src.utils.system_utils import *

# Setup
su.set_seed(42)

logger = setup_logging(log_dir="logs")
device = su.setup_device(logger=logger)
logger.info("Initialized random seed and device setup successfully")

def signal_handler(sig, frame):
    """Handle interrupt signals."""
    logger.error(f"Received signal {sig}, shutting down")
    raise SystemExit("Terminated by signal")

def train_model(model: FusionModel, dataloader: DataLoader, teacher_models: List[RPropMLP], 
                weights, alpha: float, beta: float, config: FusionConfig) -> float:
    """Train the fusion model and return the best validation loss."""
    try:
        logger.info("Starting model training")
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        model.train()
        
        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(config.max_epochs):
            total_loss = 0
            try:
                for batch in dataloader:
                    inputs = batch[0].to(device)
                    true_targets = [t.to(device) for t in batch[1:]]
                    
                    optimizer.zero_grad()
                    
                    # Get teacher outputs
                    with torch.no_grad():
                        teacher_outputs = [teacher(inputs).detach() for teacher in teacher_models]
                    
                    # Student model predictions
                    outputs = model(teacher_outputs)
                    
                    # Compute wMAE losses
                    loss_sup = wMAE_loss(outputs, true_targets, weights)
                    loss_dist = wMAE_loss(outputs, teacher_outputs, weights)
                    
                    # Total loss
                    loss = alpha * loss_sup + beta * loss_dist
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clipping)
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                logger.info(f'Epoch [{epoch+1}/{config.max_epochs}], Loss: {avg_loss:.4f}')
                
                # Early stopping
                if avg_loss < best_loss - config.min_delta:
                    best_loss = avg_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                    logger.debug(f"Improved validation loss: {best_loss:.4f} at epoch {epoch+1}")
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        logger.info(f'Early stopping at epoch {epoch+1}')
                        break
            except Exception as e:
                logger.error(f"Error in epoch {epoch+1}: {str(e)}", exc_info=True)
                raise
                
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        logger.info(f"Training completed with best loss: {best_loss:.4f}")
        return best_loss
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}", exc_info=True)
        raise

def create_dataloader(X: torch.Tensor, y: List[torch.Tensor], batch_size: int, 
                     shuffle: bool = True, num_workers: int = 4, pin_memory: bool = True) -> DataLoader:
    """Create a DataLoader from input and target tensors."""
    try:
        logger.info("Creating DataLoader")
        num_workers = min(num_workers, mp.cpu_count())  # Limit workers to CPU core count
        logger.debug(f"Using {num_workers} workers for DataLoader")
        try:
            dataset = TensorDataset(X, *y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                   num_workers=num_workers, pin_memory=pin_memory)
            logger.info(f"DataLoader created with batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}")
            return dataloader
        except RuntimeError as mp_e:
            logger.warning(f"DataLoader multiprocessing failed with {num_workers} workers: {str(mp_e)}. Falling back to num_workers=0.")
            dataset = TensorDataset(X, *y)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                   num_workers=0, pin_memory=pin_memory)
            logger.info(f"DataLoader created with batch_size={batch_size}, shuffle={shuffle}, num_workers=0")
            return dataloader
    except Exception as e:
        logger.error(f"Error in create_dataloader: {str(e)}", exc_info=True)
        raise

def objective_function(params, X: torch.Tensor, y: List[torch.Tensor], 
                     teacher_models: List[RPropMLP], weights, config: FusionConfig) -> float:
    """Objective function for Bayesian optimization."""
    try:
        logger.info(f"Starting objective function evaluation with params: {params}")
        lr, batch_size, dropout_prob, hidden_size1, hidden_size2, activation_idx, alpha, beta = params
        
        # Map activation index to activation functions
        activation_options = [['relu', 'relu'], ['prelu', 'relu'], ['tanh', 'relu']]
        if activation_idx >= len(activation_options):
            logger.error(f"Invalid activation index: {activation_idx}")
            raise ValueError("Invalid activation index")
        activations = activation_options[activation_idx]
        logger.debug(f"Selected activations: {activations}")
        
        # Update config with new parameters
        config.learning_rate = lr
        config.batch_size = int(batch_size)
        config.dropout_prob = dropout_prob
        config.hidden_layers = [int(hidden_size1), int(hidden_size2)]
        config.activations = activations
        logger.debug(f"Config updated with lr={lr}, batch_size={batch_size}, dropout={dropout_prob}, "
                    f"hidden_layers=[{hidden_size1}, {hidden_size2}]")
        
        # Create DataLoader
        dataloader = create_dataloader(X, y, config.batch_size, shuffle=config.shuffle, 
                                     num_workers=config.num_workers, pin_memory=config.pin_memory)
        
        # Initialize fusion model
        teacher_output_sizes = [teacher.output_size for teacher in teacher_models]
        model = FusionModel(
            teacher_output_sizes=teacher_output_sizes,
            hidden_layers=config.hidden_layers,
            activations=config.activations,
            dropout_prob=config.dropout_prob,
            batch_norm=config.batch_norm
        ).to(device)
        logger.info("Fusion model initialized")
        
        # Train model and return validation loss
        val_loss = train_model(model, dataloader, teacher_models, weights, alpha, beta, config)
        logger.info(f"Objective function evaluation completed with val_loss: {val_loss:.4f}")
        return val_loss
    except Exception as e:
        logger.error(f"Error in objective_function: {str(e)}", exc_info=True)
        raise

def fusion_model_main(targets: List[str], true_targets: List[tuple], project_root: str, device, max_combination: int = 20):
    """Main function to train the fusion model with Bayesian optimization."""
    logger.info(f"=== Fusion Model Training Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Step 1: Load and prepare teacher models
        num_targets_list = []
        teacher_models = []
        for i, data in enumerate(true_targets):
            try:
                logger.info(f"Processing teacher model {i+1}/{len(true_targets)} for target {targets[i]}")
                num_targets_list.append(data[2])  # Assuming data[2] is the number of targets
                model_path = Path(f"{project_root}/output/{targets[i]}/best_model.pth")
                
                if not model_path.exists():
                    logger.error(f"Model file not found: {model_path}")
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                    
                model = ResultsManager.load_best_model(model_path=model_path)
                model.eval()
                teacher_models.append(model)
                logger.info(f"Teacher model {i+1} for {targets[i]} loaded and prepared")
            except Exception as e:
                logger.error(f"Error processing teacher model {i+1} for {targets[i]}: {str(e)}", exc_info=True)
                raise
        
        # Compute weights for wMAE loss
        weights = compute_wmae_weights(torch.tensor(num_targets_list, dtype=torch.float64, device=device))
        logger.info(f"Computed wMAE weights: {weights.tolist()}")
        
        # Prepare data for fusion model
        try:
            logger.info("Preparing fusion model data")
            X_fusion = true_targets[0][0]
            y_fusion = [data[1] for data in true_targets]
            X_fusion = torch.tensor(X_fusion, dtype=torch.float32).to(device)
            y_fusion = [torch.tensor(y, dtype=torch.float32).to(device) for y in y_fusion]
            logger.info(f"Fusion model data prepared: X shape={X_fusion.shape}, y shapes={[y.shape for y in y_fusion]}")
        except Exception as e:
            logger.error(f"Error preparing fusion model data: {str(e)}", exc_info=True)
            raise
        
        # Step 2: Initialize fusion config
        try:
            fusion_config = ConfigManager.create_default_config("fusion")
            logger.info("Fusion configuration initialized")
        except Exception as e:
            logger.error(f"Error initializing fusion config: {str(e)}", exc_info=True)
            raise
        
        # Step 3: Define Bayesian optimization search space
        try:
            space, report = generate_bayesian_search(X_fusion, problem_type=fusion_config.problem_type, max_combinations=max_combination)
            logger.info(f"Bayesian search space defined with {max_combination} combinations")
        except Exception as e:
            logger.error(f"Error defining Bayesian search space: {str(e)}", exc_info=True)
            raise
        
        # Step 4: Run Bayesian optimization
        try:
            logger.info("Starting Bayesian optimization")
            result = gp_minimize(
                lambda params: objective_function(params, X_fusion, y_fusion, teacher_models, weights, fusion_config),
                space,
                n_calls=max_combination,
                random_state=42,
                verbose=True
            )
            logger.info("Bayesian optimization completed")
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {str(e)}", exc_info=True)
            raise
        
        # Step 5: Train final model with best hyperparameters
        try:
            best_params = result.x
            lr, batch_size, dropout_prob, hidden_size1, hidden_size2, activation_idx, alpha, beta = best_params
            activation_options = [['relu', 'relu'], ['prelu', 'relu'], ['tanh', 'relu']]
            activations = activation_options[activation_idx]
            
            fusion_config.learning_rate = lr
            fusion_config.batch_size = int(batch_size)
            fusion_config.dropout_prob = dropout_prob
            fusion_config.hidden_layers = [int(hidden_size1), int(hidden_size2)]
            fusion_config.activations = activations
            logger.info(f"Best hyperparameters: lr={lr}, batch_size={batch_size}, dropout={dropout_prob}, "
                       f"hidden_sizes=[{hidden_size1},{hidden_size2}], activations={activations}, alpha={alpha}, beta={beta}")
            
            # Create final DataLoader
            dataloader = create_dataloader(X_fusion, y_fusion, fusion_config.batch_size, 
                                         shuffle=fusion_config.shuffle, 
                                         num_workers=fusion_config.num_workers, 
                                         pin_memory=fusion_config.pin_memory)
            
            # Initialize and train final fusion model
            teacher_output_sizes = [teacher.output_size for teacher in teacher_models]
            final_model = FusionModel(
                teacher_output_sizes=teacher_output_sizes,
                hidden_layers=fusion_config.hidden_layers,
                activations=fusion_config.activations,
                dropout_prob=fusion_config.dropout_prob,
                batch_norm=fusion_config.batch_norm
            ).to(device)
            logger.info("Final fusion model initialized")
            
            final_loss = train_model(final_model, dataloader, teacher_models, weights, alpha, beta, fusion_config)
            logger.info(f"Final model training completed with loss: {final_loss:.4f}")
            
            # Save the final model
            output_path = Path(f"{project_root}/output/fusion_model")
            output_path.mkdir(parents=True, exist_ok=True)
            torch.save(final_model.state_dict(), output_path / "best_fusion_model.pth")
            logger.info(f"Final model saved to {output_path / 'best_fusion_model.pth'}")
            
            return final_model, final_loss
        except Exception as e:
            logger.error(f"Error training final model: {str(e)}", exc_info=True)
            raise
            
    except Exception as e:
        logger.error(f"Error in fusion_model_main: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Cleaning up resources in fusion_model_main")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()