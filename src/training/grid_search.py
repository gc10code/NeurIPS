import itertools
import random
from typing import Dict, Any, List, Tuple
from pathlib import Path
import yaml
import logging

from src.utils.validators import ConfigValidator
from src.utils.activations import validate_activations

logger = logging.getLogger('RPropMLP')

def generate_grid_combinations(grid_search_config: Dict[str, Any], output_dir: Path) -> List[Tuple]:
    for param_list in ['learning_rates', 'batch_sizes', 'hidden_layers', 'activations']:
        if param_list not in grid_search_config or not grid_search_config[param_list]:
            raise ValueError(f"grid_search.{param_list} must be a non-empty list")
    
    for lr in grid_search_config['learning_rates']:
        ConfigValidator._validate_positive_number(lr, 'grid_search.learning_rates')
    for bs in grid_search_config['batch_sizes']:
        ConfigValidator._validate_positive_integer(bs, 'grid_search.batch_sizes')
    for hl in grid_search_config['hidden_layers']:
        if not isinstance(hl, (list, tuple)) or not hl or any(h <= 0 for h in hl):
            raise ValueError(f"grid_search.hidden_layers must contain non-empty lists of positive integers, got {hl}")
    validate_activations(grid_search_config['activations'])
    
    rprop = grid_search_config['rprop']
    for delta in rprop.get('delta_plus', []):
        if delta <= 1:
            raise ValueError(f"delta_plus values must be > 1, got {delta}")
    for delta in rprop.get('delta_minus', []):
        if not (0 < delta < 1):
            raise ValueError(f"delta_minus values must be in (0,1), got {delta}")
    
    grid_params = list(itertools.product(
        grid_search_config['learning_rates'],
        grid_search_config['batch_sizes'],
        grid_search_config['hidden_layers'],
        grid_search_config['activations'],
        grid_search_config['rprop']['delta_plus'],
        grid_search_config['rprop']['delta_minus'],
        grid_search_config['rprop']['delta_min'],
        grid_search_config['rprop']['delta_max']
    ))
    
    if grid_search_config['search_type'] == 'random':
        max_combinations = grid_search_config['max_combinations']
        grid_params = random.sample(grid_params, min(max_combinations, len(grid_params)))
    
    logger.info(f"Generated {len(grid_params)} parameter combinations for grid search")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    combinations_file = output_dir / 'grid_combinations.yaml'
    combinations_list = [
        {
            'learning_rate': params[0],
            'batch_size': params[1],
            'hidden_layers': params[2],
            'activations': params[3],
            'delta_plus': params[4],
            'delta_minus': params[5],
            'delta_min': params[6],
            'delta_max': params[7]
        } for params in grid_params
    ]
    with combinations_file.open('w') as f:
        yaml.safe_dump(combinations_list, f)
    logger.info(f"Grid search combinations saved to {combinations_file}")
    
    return grid_params

