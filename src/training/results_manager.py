from pathlib import Path
import yaml, json, pickle
from typing import Dict, Any, List, Union
import logging
import torch
from skopt.space import Real, Integer, Categorical
from src.utils.io import save_results
from src.utils.exceptions import IOOperationError
from src.config.model_config import ModelConfig, FusionConfig
from src.config.config_manager import ConfigManager
from src.models.rprop_mlp import RPropMLP
import traceback

logger = logging.getLogger('RPropMLP')

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
                        wmae = fold_result['metrics']['wmae']
                        formatted_line = f"{p_idx}\t{f_idx}\t{mse}\t{mae}\t{r2}\t{wmae}\n"
                        f.write(formatted_line)                    
            logger.debug(f"Fold results saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save fold results to {file_path}: {str(e)}")
            raise IOOperationError(f"Failed to save fold results to {file_path}: {str(e)}")

        
    @staticmethod
    def save_best_model(model: torch.nn.Module, output_dir: Path) -> None:
        """Save the best model"""
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            best_result_file = output_dir / 'best_model.pth'
            
            torch.save(model.state_dict(), best_result_file)
            logger.info(f"Best model saved to {best_result_file}")

        except Exception as e:
            error_msg = f"Failed to save best model to {best_result_file}: {str(e)}"
            logger.error(error_msg)
            raise

    @staticmethod
    def load_best_model(model_path:Path)-> torch.nn.Module:
        try:
            model = RPropMLP(input_size=None, hidden_layers=None, output_size=1, activations=None)
            with open(model_path, "rb") as f:
                model.load_state_dict(torch.load(f))
            return model
        except (IOError, PermissionError) as e:
            error_msg = f"Failed to save best result to {model_path}: {str(e)}"
            logger.error(error_msg)
            raise IOOperationError(error_msg)
        
    @staticmethod
    def best_model_selection(fold_results: List[dict]) -> RPropMLP:
        """Select the best model based on validation loss."""
        try:
            if not fold_results:
                logger.error("Empty fold results list provided")
                raise ValueError("Fold results list is empty")
                
            best_model = None
            best_loss = float('inf')
            for result in fold_results:
                loss = result['val_loss']
                if loss < best_loss:
                    best_loss = loss
                    best_model = result['model']
                    logger.debug(f"Found better model with validation loss: {best_loss:.4f}")
            
            if best_model is None:
                logger.error("No valid model found in fold results")
                raise ValueError("No valid model found")
                
            logger.info(f"Selected best model with validation loss: {best_loss:.4f}")
            return best_model
        except Exception as e:
            logger.error(f"Error in best_model_selection: {str(e)}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            raise