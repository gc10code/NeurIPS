from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import torch
from src.utils.logging import lprint, LoggingLevels as ll
from src._global import global_init

global_config = global_init()


@dataclass
class RPropConfig:
    """Configuration for the neural network and training process."""
    # Training parameters
    training_type: str = 'fold'  # 'fold' or 'split'
    learning_rate: float = 0.01
    batch_size: int = 32
    max_epochs: int = global_config.MAX_EPOCHS
    k_folds: int = global_config.K_FOLD
    patience: int = global_config.PATIENCE
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
    num_workers: int = 5
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
        config_dict = self.__dict__
        
        try:
            ConfigValidator.validate_config(config_dict, config_type='rprop')
            lprint(ll.INFO, "RProp validated successfully")
        except ValueError as e:
            lprint(ll.ERROR, f"RProp validation failed: {str(e)}")
            raise
    
    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values and validate.

        Args:
            updates (Dict[str, Any]): Dictionary of updates.

        Raises:
            ValueError: If updated configuration is invalid.
        """
        from utils.validators import ConfigValidator
        ConfigValidator.validate_config(self.__dict__)
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                lprint(ll.ERROR, f"Invalid configuration key: {key}")
                raise ValueError(f"Invalid configuration key: {key}")
        
        try:
            ConfigValidator.validate_config(self.__dict__, config_type='rprop')
            lprint(ll.INFO, "Updated HRMConfig validated successfully")
        except ValueError as e:
            lprint(ll.ERROR, f"HRMConfig update failed: {str(e)}")
            raise

@dataclass
class HRMConfig:
    """
    Configuration class for the FusionModel with Hierarchical Recurrent Model (HRM).

    Attributes:
        input_size (int): Number of input features (e.g., 52 for descriptors.tsv).
        num_targets (int): Number of target outputs (e.g., 5 for Density, FFV, Rg, Tc, Tg).
        context_size (int): Size of the context layer in HRM.
        low_hidden (int): Size of the low-level hidden layer in HRM.
        high_hidden (int): Size of the high-level hidden layer in HRM.
        act_eps (float): Activation epsilon for HRM scaling.
        max_high_steps (int): Maximum number of high-level steps in HRM.
        max_low_steps (int): Maximum number of low-level steps in HRM.
        alpha (float): Weight of supervised loss.
        beta (float): Weight of distillation loss.
        training_type (str): Training type ('fold' or 'split').
        learning_rate (float): Learning rate for optimization.
        batch_size (int): Batch size for training.
        max_epochs (int): Maximum number of training epochs.
        k_folds (int): Number of folds for cross-validation.
        patience (int): Patience for early stopping.
        min_delta (float): Minimum improvement for early stopping.
        shuffle (bool): Whether to shuffle data in DataLoader.
        target_error (float): Target error for convergence (optional).
        gradient_clipping (float): Gradient clipping threshold.
        input_normalization (bool): Whether to normalize inputs.
        output_normalization (bool): Whether to normalize outputs.
        seed (int): Random seed for reproducibility.
        num_workers (int): Number of DataLoader workers.
        pin_memory (bool): Whether to use pinned memory in DataLoader.
        validation_split (float): Fraction of data for validation.
        dropout_prob (float): Dropout probability.
        batch_norm (bool): Whether to use batch normalization.
        output_size (int): Output size per target (typically 1 for regression).
        problem_type (str): Problem type ('regression' or 'classification').
        input_dir (str): Directory for input data.
        output_dir (str): Directory for output data and models.
        memory_check_interval (int): Interval for memory usage checks.
        checkpoint_interval (int): Interval for saving model checkpoints.
    """
    # HRM-specific parameters
    input_size: int
    num_targets: int
    
    low_hidden: int
    high_hidden: int
    act_eps: float
    max_high_steps: int
    max_low_steps: int
    one_step_detach: bool = False
    
    # Common parameters
    context_size: int = 128  # Aggiunto: dimensione del contesto per il modello HRM
    training_type: str = 'fold'
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = global_config.MAX_EPOCHS
    k_folds: int = global_config.K_FOLD
    patience: int = global_config.PATIENCE
    min_delta: float = 1e-4
    shuffle: bool = True
    target_error: float = 0.0
    gradient_clipping: float = 1.0
    input_normalization: bool = True
    output_normalization: bool = True
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    validation_split: float = 0.2
    dropout_prob: float = 0.1
    batch_norm: bool = True
    output_size: int = 1
    problem_type: str = 'regression'
    input_dir: str = './data'
    output_dir: str = './output'
    memory_check_interval: int = 10
    checkpoint_interval: int = 50

    def __post_init__(self):
        """Validate the configuration upon initialization."""
        from utils.validators import ConfigValidator
        ConfigValidator.validate_config(self.__dict__)
        config_dict = self.__dict__
        
        try:
            ConfigValidator.validate_config(config_dict, config_type='fusion')
            lprint(ll.INFO, "HRMConfig validated successfully")
        except ValueError as e:
            lprint(ll.ERROR, f"HRMConfig validation failed: {str(e)}")
            raise

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values and validate.

        Args:
            updates (Dict[str, Any]): Dictionary of updates.

        Raises:
            ValueError: If updated configuration is invalid.
        """
        from utils.validators import ConfigValidator
        ConfigValidator.validate_config(self.__dict__)
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                lprint(ll.ERROR, f"Invalid configuration key: {key}")
                raise ValueError(f"Invalid configuration key: {key}")
        
        try:
            ConfigValidator.validate_config(self.__dict__, config_type='fusion')
            lprint(ll.INFO, "Updated HRMConfig validated successfully")
        except ValueError as e:
            lprint(ll.ERROR, f"HRMConfig update failed: {str(e)}")
            raise