import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from typing import Tuple, Dict, Any, Optional, List
import gc
import os, time
from pathlib import Path
import multiprocessing as mp
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import signal
from datetime import datetime

from src.models.rprop_mlp import RPropMLP
from src.utils.early_stopping import EarlyStopping
from src.utils.metrics import evaluate_model
from src.config.model_config import ModelConfig
from src.utils.exceptions import TrainingError
from src.utils.normalizer import DataNormalizer
from src.training.data_preparation import create_data_loaders
from src.training.results_manager import ResultsManager
from src.utils.system_utils import *
from src.training.data_preparation import *
from src.utils.exceptions import MLPError
from src.config.config_manager import ConfigManager
from src.config.grid_search_config import generate_bayesian_search
from src.utils.logging import lprint, LoggingLevels as ll

import numba as nb

normalizer = DataNormalizer("minmax", "minmax")

def create_model(config: ModelConfig, input_size: int, hidden_layers: List[int], 
                 activations: List[str], device: torch.device) -> RPropMLP:
    """Create and initialize an RPropMLP model."""
    try:
        lprint(ll.DEBUG,  f"Creating RPropMLP model with input_size={input_size}, hidden_layers={hidden_layers}, "
                     f"activations={activations}")
        model = RPropMLP(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=config.output_size,
            activations=activations,
            dropout_prob=config.dropout_prob,
            batch_norm=config.batch_norm,
            problem_type=config.problem_type
        ).to(device)
        lprint(ll.INFO,  "RPropMLP model created successfully")
        return model
    except Exception as e:
        lprint(ll.ERROR,  f"Failed to create model: {str(e)}")
        raise TrainingError(f"Failed to create model: {str(e)}")

def train_epoch(model: RPropMLP, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, device: torch.device, config: ModelConfig) -> float:
    """Train the model for one epoch."""
    try:
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            if config.problem_type == 'regression':
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                elif outputs.dim() > 2:
                    outputs = outputs.squeeze(-1)
            
            loss = criterion(outputs, batch_y)
            
            if torch.isnan(loss):
                lprint(ll.ERROR,  "NaN loss detected during training")
                raise TrainingError("NaN loss detected during training")
            
            loss.backward()
            
            if config.gradient_clipping > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clipping)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_loss = np.mean(train_losses)
        lprint(ll.DEBUG,  f"Epoch training completed, average loss: {avg_loss:.6f}")
        return avg_loss
    except Exception as e:
        lprint(ll.ERROR,  f"Training epoch failed: {str(e)}")
        raise TrainingError(f"Training epoch failed: {str(e)}")

def validate_epoch(model: RPropMLP, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, 
                   device: torch.device, config: ModelConfig) -> float:
    """Validate the model for one epoch."""
    try:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X, val_y = val_X.to(device), val_y.to(device)
                val_outputs = model(val_X)
                
                if config.problem_type == 'regression':
                    if val_outputs.dim() == 1:
                        val_outputs = val_outputs.unsqueeze(1)
                    elif val_outputs.dim() > 2:
                        val_outputs = val_outputs.squeeze(-1)
                
                val_loss = criterion(val_outputs, val_y).item()
                val_losses.append(val_loss)
        
        avg_val_loss = np.mean(val_losses)
        lprint(ll.DEBUG,  f"Epoch validation completed, average loss: {avg_val_loss:.6f}")
        return avg_val_loss
    except Exception as e:
        lprint(ll.ERROR,  f"Validation epoch failed: {str(e)}")
        raise TrainingError(f"Validation epoch failed: {str(e)}")

def train_model(model: RPropMLP, 
                train_loader: torch.utils.data.DataLoader, 
                val_loader: torch.utils.data.DataLoader,
                config: ModelConfig, 
                device: torch.device,
                log_info: Tuple) -> Tuple[RPropMLP, float, Dict[str, List[float]]]:
    """Train model using DataLoader with comprehensive monitoring."""
    try:
        delta_plus = config.plus_delta
        delta_minus = config.minus_delta
        delta_min = config.min_delta
        delta_max = config.max_delta
        
        if not (0 < delta_minus < 1 < delta_plus):
            lprint(ll.ERROR,  f"Invalid RProp parameters: delta_minus={delta_minus} not in (0,1) or delta_plus={delta_plus} <= 1")
            raise TrainingError(f"Invalid RProp: delta_minus={delta_minus} ∉ (0,1) or delta_plus={delta_plus} ≤ 1")
        if not (0 < delta_min < delta_max):
            lprint(ll.ERROR,  f"Invalid RProp parameters: delta_min={delta_min} >= delta_max={delta_max}")
            raise TrainingError(f"Invalid RProp: delta_min={delta_min} >= delta_max={delta_max}")
        
        lprint(ll.INFO,  f"RProp parameters: δ+={delta_plus:.3f}, δ-={delta_minus:.3f}, "
                   f"δ_min={delta_min:.6f}, δ_max={delta_max:.3f}")
        
        optimizer = torch.optim.Rprop(
            model.parameters(),
            lr=config.learning_rate,
            etas=(delta_minus, delta_plus),
            step_sizes=(delta_min, delta_max)
        )
        criterion = nn.MSELoss() if config.problem_type == 'regression' else nn.CrossEntropyLoss()
        
        early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )

        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'grad_norm': []
        }
        
        best_val_loss = ConfigManager.START_MAX_LOSS_VAL
        best_model_state = None
        start_time = time.time()
        
        for epoch in range(config.max_epochs):
            epoch_start = time.time()
            
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config)
            val_loss = validate_epoch(model, val_loader, criterion, device, config)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            history['grad_norm'].append(grad_norm.item() if 'grad_norm' in locals() else 0.0)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                lprint(ll.DEBUG,  f"New best validation loss: {best_val_loss:.6f} at epoch {epoch+1}")
            
            if (epoch + 1) % ConfigManager.LOGGING_EPOCHS == 0 or epoch == 0:
                epoch_time = time.time() - epoch_start
                memory = monitor_memory()
                lprint(ll.INFO,  f"Param_{log_info[0]} Fold_{log_info[1]} "
                           f"Epoch {epoch+1:4d}/{config.max_epochs}: "
                           f"Train={train_loss:.6f}, Val={val_loss:.6f}, "
                           f"GradNorm={history['grad_norm'][-1]:.4f}, LR={history['learning_rate'][-1]:.6f}, "
                           f"Time={epoch_time:.2f}s, Mem={memory['rss_gb']:.1f}GB")
            
            if early_stopping(val_loss):
                lprint(ll.INFO,  f"Early stopping at epoch {epoch+1}")
                break
            
            if val_loss < config.target_error:
                lprint(ll.INFO,  f"Target error {config.target_error} reached at epoch {epoch+1}")
                break
            
            if (epoch + 1) % config.memory_check_interval == 0:
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        total_time = time.time() - start_time
        lprint(ll.INFO,  f"Training completed in {total_time:.2f}s, best val_loss: {best_val_loss:.6f}")
        
        return model, best_val_loss, history
    except Exception as e:
        lprint(ll.ERROR,  f"Training failed: {str(e)}")
        raise TrainingError(f"Training failed: {str(e)}")

def train_fold(args: Tuple[int, int, Tuple, np.ndarray, np.ndarray, ModelConfig, torch.device]) -> Dict[str, Any]:
    fold, param_idx, grid_params, X_norm, y_norm, config, device = args
    lr, batch_size, hidden_layers, activation, delta_plus, delta_minus, delta_min, delta_max = grid_params
    
    prefix = f'Params_{param_idx} Fold_{fold + 1} - '
    
    try:
        lprint(ll.INFO, prefix +f"Starting with: lr={lr}, batch_size={batch_size}, "
                         f"hidden_layers={hidden_layers}, activation={activation}, "
                         f"δ+={delta_plus}, δ-={delta_minus}, δ_min={delta_min}, δ_max={delta_max}")
        
        config = deepcopy(config)
        config.learning_rate = lr
        config.batch_size = batch_size
        config.hidden_layers = hidden_layers
        config.activations = activation
        config.plus_delta = delta_plus
        config.minus_delta = delta_minus
        config.min_delta = delta_min
        config.max_delta = delta_max
        
        if isinstance(activation, list):
            if len(activation) < len(hidden_layers):
                activation = activation + [activation[-1]] * (len(hidden_layers) - len(activation))
            elif len(activation) > len(hidden_layers):
                activation = activation[:len(hidden_layers)]
                
        train_loader, val_loader = create_data_loaders(config, X_norm, y_norm, batch_size, fold)
 
        model = create_model(config, X_norm.shape[1], hidden_layers, activation, device)
        lprint(ll.INFO, prefix +f"Model parameters: {model.get_num_parameters()}")
        
        model, val_loss, history = train_model(model, train_loader, val_loader, config, device, (param_idx, fold + 1))
        
        metrics = evaluate_model(model, val_loader, device, normalizer = normalizer)
        
        fold_results = {
            'fold': fold + 1,
            'param_idx': param_idx,
            'grid_params': {
                'learning_rate': lr,
                'batch_size': batch_size,
                'hidden_layers': hidden_layers,
                'activation': activation,
                'delta_plus': delta_plus,
                'delta_minus': delta_minus,
                'delta_min': delta_min,
                'delta_max': delta_max
            },
            'val_loss': val_loss,
            'metrics': metrics,
            'history': history,
            'model': model
        }
        
        lprint(ll.INFO, prefix +f"Fold {fold + 1} completed with val_loss: {val_loss:.6f}")
        return fold_results
    except Exception as e:
        lprint(ll.ERROR, prefix +f"Training failed for fold {fold + 1}, params {param_idx}: {str(e)}")
        return None

def create_tasks(config: ModelConfig, params: Tuple, X_norm: torch.Tensor, 
                y_norm: torch.Tensor, device: torch.device, param_idx: int) -> List[Tuple]:
    """Create tasks for K-fold cross-validation or split training for a single parameter set."""
    try:
        lprint(ll.DEBUG,  f"Creating tasks for param_idx {param_idx}")
        tasks = []
        output_dir = Path(config.output_dir)
        
        lr, batch_size, depth, decay_factor, activation, delta_plus, delta_minus, delta_min, delta_max = params
        n_features = X_norm.shape[1]
        hidden_layers = [max(4, int(n_features * (decay_factor ** i))) for i in range(depth)]
        batch_size = int(batch_size)
        if config.training_type == 'split':
            config_copy = deepcopy(config)
            config_copy.learning_rate = lr
            config_copy.batch_size = batch_size 
            config_copy.hidden_layers = hidden_layers
            config_copy.activations = [activation] * len(hidden_layers)
            config_copy.plus_delta = delta_plus
            config_copy.minus_delta = delta_minus
            config_copy.min_delta = delta_min
            config_copy.max_delta = delta_max
            tasks.append((0, param_idx, (lr, batch_size, hidden_layers, [activation] * len(hidden_layers), 
                                      delta_plus, delta_minus, delta_min, delta_max), X_norm, y_norm, config_copy, device))
        else:
            config_copy = deepcopy(config)
            config_copy.num_workers = 0
            config_copy.learning_rate = lr
            config_copy.batch_size = batch_size
            config_copy.hidden_layers = hidden_layers
            config_copy.activations = [activation] * len(hidden_layers)
            config_copy.plus_delta = delta_plus
            config_copy.minus_delta = delta_minus
            config_copy.min_delta = delta_min
            config_copy.max_delta = delta_max
            for fold in range(config.k_folds):
                task_config = deepcopy(config_copy)
                tasks.append((fold, param_idx, (lr, batch_size, hidden_layers, [activation] * len(hidden_layers), 
                                              delta_plus, delta_minus, delta_min, delta_max), X_norm, y_norm, task_config, device))
        
        lprint(ll.DEBUG,  f"Created {len(tasks)} tasks for param_idx {param_idx}")
        return tasks
    except Exception as e:
        lprint(ll.ERROR,  f"Error creating tasks: {str(e)}")
        raise

def run_training(tasks: List[Tuple], use_multiprocessing: bool = True) -> List[Dict]:
    """Run training tasks, with optional multiprocessing."""
    try:
        lprint(ll.DEBUG,  f"Multiprocessing start method: {mp.get_start_method()}")
        if use_multiprocessing:
            try:
                mp.set_start_method('spawn', force=True)
                lprint(ll.DEBUG,  f"Set multiprocessing start method to 'spawn'")
            except RuntimeError as e:
                lprint(ll.WARN,  f"Failed to set start method to 'spawn': {str(e)}. Using default method.")
                mp.set_start_method('fork', force=True)
                lprint(ll.DEBUG,  f"Set multiprocessing start method to 'fork'")
            num_processes = min(4, mp.cpu_count(), len(tasks))  # Limit to 4 processes
            lprint(ll.DEBUG,  f"Starting multiprocessing with {num_processes} workers")
            pool = mp.Pool(processes=num_processes)
            try:
                results = pool.map(train_fold, tasks)
                lprint(ll.INFO,  "Multiprocessing completed successfully")
            finally:
                pool.close()
                pool.join()
                lprint(ll.DEBUG,  "Multiprocessing pool closed and joined")
        else:
            lprint(ll.INFO,  "Running in single-threaded mode")
            results = []
            for task in tasks:
                try:
                    result = train_fold(task)
                    results.append(result)
                    lprint(ll.DEBUG,  f"Completed task: {task}")
                except Exception as e:
                    lprint(ll.ERROR,  f"Failed to process task {task}: {str(e)}")
        return [r for r in results if r is not None]
    except Exception as e:
        lprint(ll.WARN,  f"Multiprocessing failed: {str(e)}. Falling back to single-threaded execution.")
        lprint(ll.DEBUG,  "Starting single-threaded execution for all tasks")
        results = []
        for task in tasks:
            try:
                result = train_fold(task)
                results.append(result)
                lprint(ll.DEBUG,  f"Completed task: {task}")
            except Exception as e:
                lprint(ll.ERROR,  f"Failed to process task {task}: {str(e)}")
        lprint(ll.INFO,  f"Single-threaded execution completed with {len(results)} results")
        return [r for r in results if r is not None]

def objective(params: List, config: ModelConfig, X_norm: torch.Tensor, y_norm: torch.Tensor, 
              device: torch.device, output_dir: Path, param_idx: int, log_file: Path) -> float:
    """Objective function for Bayesian optimization."""
    try:
        lprint(ll.SUCCESS,  f"Evaluating params {param_idx}: {params}")
        tasks = create_tasks(config, params, X_norm, y_norm, device, param_idx)
        fold_results = run_training(tasks, use_multiprocessing=True)
        
        if not fold_results:
            lprint(ll.WARN,  f"No valid results for params {params}, returning high loss")
            with open(log_file, "a") as f:
                f.write(f"Index: {param_idx}, Input: {params}, Reason: No valid results\n")
            return ConfigManager.START_MAX_LOSS_VAL
        
        mean_val_loss = np.mean([r['val_loss'] for r in fold_results])
        lprint(ll.REPORT,  f"Evaluated params {param_idx}: {params}, mean val_loss: {mean_val_loss:.6f}")
        
        if np.isnan(mean_val_loss) or np.isinf(mean_val_loss) or abs(mean_val_loss) > 1e10:
            lprint(ll.WARN,  f"Problematic loss value: {mean_val_loss}")
            with open(log_file, "a") as f:
                f.write(f"Index: {param_idx}, Input: {params}, Problematic output: {mean_val_loss}\n")
            return ConfigManager.START_MAX_LOSS_VAL

        ResultsManager.save_fold_results(fold_results, Path(config.output_dir) / "fold_results.tsv")
        return mean_val_loss
    
    except Exception as e:
        lprint(ll.ERROR,  f"Objective evaluation failed for params {params}: {str(e)}")
        with open(log_file, "a") as f:
            f.write(f"Index: {param_idx}, Input: {params}, Exception: {str(e)}\n")
        return ConfigManager.START_MAX_LOSS_VAL


def rprop_mlp_main(target:str, config: ModelConfig, X: np.ndarray, y: np.ndarray, use_multiprocessing: bool = True, max_combinations: int = 100) -> Dict[str, Any]:
    start_time = time.time()
    lprint(ll.INFO,  f"=== RProp MLP Bayesian Optimization Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:        
        check_pytorch_version()
        set_seed(config.seed)
        lprint(ll.INFO,  f"Random seed set to {config.seed}")
        
        device = setup_device()
        X_norm, y_norm, normalizer = prepare_data(config, X, y)
        lprint(ll.INFO,  f"Data normalized: X shape {X_norm.shape}, y shape {y_norm.shape}")
        
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ConfigManager.save_config(config, output_dir)
        
        bayesian_config, report = generate_bayesian_search(X_norm, problem_type=config.problem_type, max_combinations=max_combinations)
        ConfigManager.save_bayesian_search(bayesian_config, report, output_dir)
        lprint(ll.REPORT,  f"Bayesian search configuration saved to {output_dir / 'bayesian_search.pkl'}")
        
        search_space = [
            Real(bayesian_config['learning_rate']['range'][0], bayesian_config['learning_rate']['range'][1], name='learning_rate'),
            Integer(bayesian_config['batch_size']['range'][0], bayesian_config['batch_size']['range'][1], name='batch_size'),
            Integer(bayesian_config['depth']['range'][0], bayesian_config['depth']['range'][-1], name='depth'),
            Real(bayesian_config['decay_factor']['range'][0], bayesian_config['decay_factor']['range'][1], name='decay_factor'),
            Categorical(bayesian_config['activation']['range'], name='activation'),
            Real(bayesian_config['rprop']['delta_plus']['range'][0], bayesian_config['rprop']['delta_plus']['range'][1], name='delta_plus'),
            Real(bayesian_config['rprop']['delta_minus']['range'][0], bayesian_config['rprop']['delta_minus']['range'][1], name='delta_minus'),
            Real(bayesian_config['rprop']['delta_min']['range'][0], bayesian_config['rprop']['delta_min']['range'][1], name='delta_min'),
            Real(bayesian_config['rprop']['delta_max']['range'][0], bayesian_config['rprop']['delta_max']['range'][1], name='delta_max')
        ]
        
        param_idx_counter = [0]
        log_file = Path(config.output_dir) / "logs/problematic_values.log"

        def objective_wrapper(params):
            param_idx_counter[0] += 1
            return objective(params, config, X_norm, y_norm, device, output_dir, param_idx_counter[0], log_file=log_file)
        
        lprint(ll.SUCCESS,  f"Bayesian Optimization started")
        result = gp_minimize(
            objective_wrapper,
            search_space,
            n_calls=max_combinations,
            random_state=config.seed,
            verbose=True
        )
        
        n_features = X_norm.shape[1]
        
        best_params = {
            'learning_rate': result.x[0],
            'batch_size': result.x[1],
            'hidden_layers': [max(4, int(n_features * (result.x[3] ** i))) for i in range(result.x[2])],
            'activation': [result.x[4]] * result.x[2],
            'delta_plus': result.x[5],
            'delta_minus': result.x[6],
            'delta_min': result.x[7],
            'delta_max': result.x[8]
        }
        best_val_loss = result.fun
        
        lprint(ll.SUCCESS,  f"Bayesian Optimization Completed")
        lprint(ll.SUCCESS,  f"Best parameters: {best_params}")
        lprint(ll.SUCCESS,  f"Best validation loss: {best_val_loss:.6f}")
        lprint(ll.SUCCESS,  f"Total training time: {time.time()-start_time:.2f}s")
        report.append(f"Best parameters: {best_params}")
        report.append(f"Best validation loss: {best_val_loss:.6f}")
        
        # Retraining of final model
        lprint(ll.SUCCESS,  f"Retraining of best params model")
        final_config = ConfigManager.create_default_config("rprop")
        final_config = ConfigManager.update_config(final_config, best_params.values())
        final_params = result.x    
        final_tasks = create_tasks(final_config, final_params, X_norm, y_norm, device, 1)
        final_fold_results = run_training(final_tasks, use_multiprocessing=False)  # Disable multiprocessing for stability
        final_model = ResultsManager.best_model_selection(final_fold_results)
        ResultsManager.save_best_model(final_model, Path(final_config.output_dir) / target/ "best_model.pth") 
        lprint(ll.REPORT,  f"Final model retraining completed and saved in {final_config.output_dir}")
        return final_model
    
    except MLPError as e:
        lprint(ll.ERROR,  f"Main function failed: {str(e)}")
        raise
    except Exception as e:
        lprint(ll.ERROR,  f"Unexpected error in main function: {str(e)}")
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()