import torch
import numpy as np
import json
import pickle
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

from models.rprop_mlp import RPropMLP
from utils.normalizer import DataNormalizer
from src.utils.logging import lprint, LoggingLevels as ll

def save_results(results: Dict[str, Any], output_dir: Path) -> None:
    try:
        output_dir.mkdir(parents=False, exist_ok=True)
        
        results_file = output_dir / 'results.json'
        with results_file.open('w') as f:
            json.dump(results, f, indent=3, default=str)
        
        if 'model' in results:
            model_file = output_dir / 'best_model.pth'
            torch.save(results['model'].state_dict(), model_file)
            lprint(ll.INFO,  f"Model saved to {model_file}")
        
        if 'normalizer' in results:
            normalizer_file = output_dir / 'normalizer.pkl'
            with normalizer_file.open('wb') as f:
                pickle.dump(results['normalizer'], f)
            lprint(ll.INFO,  f"Normalizer saved to {normalizer_file}")
        
        lprint(ll.INFO,  f"Results saved to {results_file}")
    
    except (PermissionError, IOError) as e:
        lprint(ll.ERROR,  f"Failed to save results: {str(e)}")
        raise

def load_config(config_file: str) -> Dict[str, Any]:
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with config_path.open('r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        from .validators import ConfigValidator
        ConfigValidator.validate_config(config)
        lprint(ll.INFO,  f"Configuration loaded from {config_file}")
        return config
        
    except (yaml.YAMLError, json.JSONDecodeError, PermissionError) as e:
        lprint(ll.ERROR,  f"Failed to load configuration: {str(e)}")
        raise

def predict(model: RPropMLP, X: np.ndarray, device: torch.device,
            normalizer: Optional[DataNormalizer] = None) -> np.ndarray:
    try:
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if normalizer is not None:
            X_tensor = normalizer.transform(X)
        
        X_tensor = X_tensor.to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            
            if model.problem_type == 'regression':
                outputs = outputs.squeeze()
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                elif outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
        
        y_pred = outputs.cpu().numpy()
        if normalizer is not None:
            y_pred = normalizer.inverse_transform_y(y_pred)
        
        return y_pred
    
    except Exception as e:
        lprint(ll.ERROR,  f"Prediction failed: {str(e)}")
        raise
