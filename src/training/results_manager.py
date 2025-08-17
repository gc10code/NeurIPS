from pathlib import Path
import yaml, json, pickle
from typing import Dict, Any, List, Union
import logging
import torch
from skopt.space import Real, Integer, Categorical
from src.utils.io import save_results
from src.utils.exceptions import IOOperationError
from src.config.model_config import RPropConfig, HRMConfig
from src.config.config_manager import ConfigManager
from src.models.rprop_mlp import RPropMLP
import traceback
from src.utils.logging import lprint, LoggingLevels as ll

class ResultsManager:
    """Manages saving and organizing training results."""
    
    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """Convert non-serializable types to serializable formats."""
        if isinstance(obj, (Real, Integer, Categorical)):
            return str(obj.__class__.__name__)  # Convert to strings: 'Real', 'Integer', 'Categorical'
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()  # Convert tensor scalar o array
        elif isinstance(obj, dict):
            return {k: ResultsManager._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ResultsManager._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(ResultsManager._convert_to_serializable(item) for item in obj)
        elif isinstance(obj, float) and obj.is_integer():
            return int(obj)  # Convert float in int (es. 12.0 -> 12)
        return obj

    @staticmethod
    def save_fold_results(fold_results: Dict[str, Any], file_path: Path) -> None:
        """Save results for a single fold."""
        
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)     
            with open(file_path, "w") as f:
                f.write(f"PAR\tFOLD\tMSE\tMAE\tR2\tWMAE\n")
        
        try:
            with open(file_path, "a") as f:
                for fold_result in fold_results:
                    if fold_result:
                        p_idx = fold_result['param_idx']
                        f_idx = fold_result['fold']
                        mae = fold_result['metrics']['mae']
                        mse = fold_result['metrics']['mse']
                        r2 = fold_result['metrics']['r2']
                        formatted_line = f"{p_idx}\t{f_idx}\t{mse}\t{mae}\t{r2}\n"
                        f.write(formatted_line)                    
            lprint(ll.DEBUG, f"Fold results saved to {file_path}")
        except Exception as e:
            lprint(ll.ERROR,  f"Failed to save fold results to {file_path}: {str(e)}")
            raise IOOperationError(f"Failed to save fold results to {file_path}: {str(e)}")

        
    @staticmethod
    def save_best_model(model: RPropMLP, model_path: Path):
        try:
            save_dict = {
                'state_dict': model.state_dict(),
                'input_size': model.input_size,
                'hidden_layers': model.hidden_layers,
                'output_size': model.output_size,
                'activations': model.activations,
                'dropout_prob': model.dropout_prob,
                'batch_norm': model.batch_norm,
                'problem_type': model.problem_type
            }
            with open(model_path, "wb") as f:
                torch.save(save_dict, f)
            lprint(ll.INFO, f"Saved model to {model_path}")
        except (IOError, PermissionError) as e:
            error_msg = f"Failed to save model to {model_path}: {str(e)}"
            lprint(ll.ERROR, error_msg)
            raise IOError(error_msg)

    
    @staticmethod
    def load_best_model(model_path: Path) -> RPropMLP:
        try:
            # Load the saved dictionary
            with open(model_path, "rb") as f:
                save_dict = torch.load(f, map_location='cpu')  # Load to CPU first for flexibility
            
            # Extract architecture parameters
            input_size = save_dict['input_size']
            hidden_layers = save_dict['hidden_layers']
            output_size = save_dict['output_size']
            activations = save_dict['activations']
            dropout_prob = save_dict.get('dropout_prob', 0.3)  # Default if not saved
            batch_norm = save_dict.get('batch_norm', False)
            problem_type = save_dict.get('problem_type', 'regression')
            
            # Initialize model with saved parameters
            model = RPropMLP(
                input_size=input_size,
                hidden_layers=hidden_layers,
                output_size=output_size,
                activations=activations,
                dropout_prob=dropout_prob,
                batch_norm=batch_norm,
                problem_type=problem_type
            )
            
            # Load state_dict
            model.load_state_dict(save_dict['state_dict'])
            model.eval()  # Set to evaluation mode
            lprint(ll.INFO, f"Loaded model from {model_path}")
            return model
        except (IOError, PermissionError, KeyError) as e:
            error_msg = f"Failed to load model from {model_path}: {str(e)}"
            lprint(ll.ERROR, error_msg)
            raise IOError(error_msg)
        
    @staticmethod
    def best_model_selection(fold_results: List[dict]) -> RPropMLP:
        """Select the best model based on validation loss."""
        try:
            if not fold_results:
                lprint(ll.ERROR,  "Empty fold results list provided")
                raise ValueError("Fold results list is empty")
                
            best_model = None
            best_loss = ConfigManager.START_MAX_LOSS_VAL
            for result in fold_results:
                loss = result['val_loss']
                if loss < best_loss:
                    best_loss = loss
                    best_model = result['model']
                    lprint(ll.DEBUG, f"Found better model with validation loss: {best_loss:.4f}")
            
            if best_model is None:
                lprint(ll.ERROR,  "No valid model found in fold results")
                raise ValueError("No valid model found")
                
            lprint(ll.INFO, f"Selected best model with validation loss: {best_loss:.4f}")
            return best_model
        except Exception as e:
            lprint(ll.ERROR,  f"Error in best_model_selection: {str(e)}")
            lprint(ll.DEBUG, f"Stack trace: {traceback.format_exc()}")
            raise