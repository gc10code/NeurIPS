from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import torch


@dataclass
class ModelConfig:
    """Configuration for the neural network and training process."""
    # Training parameters
    training_type: str = 'fold'  # 'fold' or 'split'
    learning_rate: float = 0.01
    batch_size: int = 32
    max_epochs: int = 1000
    k_folds: int = 5
    patience: int = 50
    min_delta: float = 1e-6
    max_delta: float = 50.0
    minus_delta: float = 0.5
    plus_delta: float = 1.2
    shuffle: bool = True
    target_error: float = 1e-6
    gradient_clipping: float = 1.0
    input_normalization: str = 'minmax'
    output_normalization: str = 'minmax'
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    validation_split: float = 0.0
    
    # Network parameters
    hidden_layers: List[int] = None
    activations: List[str] = None
    dropout_prob: float = 0.3
    batch_norm: bool = True
    output_size: int = 1
    problem_type: str = 'regression'
    
    # Directory parameters
    input_dir: str = 'data'
    output_dir: str = 'output'
    
    # Monitoring parameters
    memory_check_interval: int = 10
    checkpoint_interval: int = 100

    def __post_init__(self):
        """Validate configuration after initialization."""
        from utils.validators import ConfigValidator
        ConfigValidator.validate_config(self.__dict__)



@dataclass
class FusionConfig:
    """Configuration for the fusion neural network and training process."""
    # Training parameters
    training_type: str = 'fold'  # 'fold' or 'split'
    learning_rate: float = 0.01
    batch_size: int = 32
    max_epochs: int = 10
    k_folds: int = 2
    patience: int = 20
    min_delta: float = 1e-6
    max_delta: float = 50.0
    minus_delta: float = 0.5
    plus_delta: float = 1.2
    shuffle: bool = True
    target_error: float = 1e-6
    gradient_clipping: float = 1.0
    input_normalization: str = 'minmax'
    output_normalization: str = 'minmax'
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    validation_split: float = 0.0
    
    # Network parameters
    teachers: List[torch.nn.Module] = None
    hidden_layers: List[int] = None
    activations: List[str] = None
    dropout_prob: float = 0.3
    batch_norm: bool = True
    output_size: int = 1
    problem_type: str = 'regression'
    
    # Directory parameters
    input_dir: str = 'data'
    output_dir: str = 'output'
    
    # Monitoring parameters
    memory_check_interval: int = 10
    checkpoint_interval: int = 100

    def __post_init__(self):
        """Validate configuration after initialization."""
        from utils.validators import ConfigValidator
        ConfigValidator.validate_config(self.__dict__)