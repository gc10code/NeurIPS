from pathlib import Path
import yaml, json, pickle
from typing import Dict, Any, List, Tuple, Union
from datetime import datetime
from skopt.space import Real, Integer, Categorical

from src.config.model_config import RPropConfig, HRMConfig
from src.utils.exceptions import ConfigurationError
from src.utils.logging import lprint, LoggingLevels as ll
from src.utils.validators import ConfigValidator

class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    # General Parameters
    LOGGING_EPOCHS = 100
    START_MAX_LOSS_VAL = 1e10


    @staticmethod
    def create_default_config(config_type: str = 'rprop') -> Union[Dict[str, Any], HRMConfig]:
        """
        Create a default configuration for RPropMLP or FusionModel.

        Args:
            config_type (str): Type of configuration ('rprop' or 'fusion').

        Returns:
            Union[Dict[str, Any], HRMConfig]: Default configuration (dict for rprop, HRMConfig for fusion).

        Raises:
            ValueError: If config_type is invalid or configuration is invalid.
        """
        if config_type not in ['rprop', 'fusion']:
            lprint(ll.ERROR, f"Invalid config_type: {config_type}. Must be 'rprop' or 'fusion'")
            raise ValueError(f"Invalid config_type: {config_type}")

        if config_type == 'rprop':
            config = RPropConfig()
            try:
                config = RPropConfig(
                    hidden_layers=[64, 32],  # Example values
                    activations=['relu'],    # Example values
                )
                return config
            except ValueError as e:
                    lprint(ll.ERROR, f"Configuration validation failed: {str(e)}")
                    raise
        else:  # config_type == 'fusion'
            try:
                config = HRMConfig(
                    input_size=52,  # Matches X_fusion.shape[1]
                    num_targets=5,  # Matches len(targets)
                    low_hidden=64,
                    high_hidden=128,
                    act_eps=0.1,
                    max_high_steps=5,
                    max_low_steps=5,
                    training_type='fold',
                    learning_rate=0.001,
                    batch_size=32,
                    max_epochs=100,
                    k_folds=5,
                    patience=10,
                    min_delta=1e-4,
                    shuffle=True,
                    target_error=0.0,
                    gradient_clipping=1.0,
                    input_normalization=True,
                    output_normalization=True,
                    seed=42,
                    num_workers=4,
                    pin_memory=True,
                    validation_split=0.2,
                    dropout_prob=0.1,
                    batch_norm=True,
                    output_size=1,
                    problem_type='regression',
                    input_dir='./data',
                    output_dir='./output',
                    memory_check_interval=10,
                    checkpoint_interval=50
                )
                lprint(ll.INFO, "Default fusion configuration created successfully")
                return config
            except ValueError as e:
                lprint(ll.ERROR, f"Failed to create default fusion configuration: {str(e)}")
                raise

    @staticmethod
    def update_config(config: Union[Dict[str, Any], HRMConfig, RPropConfig], updates: Dict[str, Any], config_type: str = 'rprop') -> Union[Dict[str, Any], HRMConfig]:
        """
        Update configuration with new values and validate.

        Args:
            config (Union[Dict[str, Any], HRMConfig]): Existing configuration.
            updates (Dict[str, Any]): Dictionary of updates.
            config_type (str): Type of configuration ('rprop' or 'fusion').

        Returns:
            Union[Dict[str, Any], HRMConfig]: Updated and validated configuration.
        """
        if config_type == 'fusion' and isinstance(config, HRMConfig):
            try:
                config.update(updates)
                return config
            except ValueError as e:
                lprint(ll.ERROR, f"HRMConfig update failed: {str(e)}")
                raise
        else:
            updated_config = config
            updated_config.update(updates)
            try:
                ConfigValidator.validate_config(updated_config, config_type=config_type)
                lprint(ll.INFO, f"Updated {config_type} configuration validated successfully")
                return updated_config
            except ValueError as e:
                lprint(ll.ERROR, f"Configuration update failed: {str(e)}")
                raise

    @staticmethod
    def save_config(config: RPropConfig, output_path: Path) -> None:
        """Save configuration to a YAML file."""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            config_path = output_path / 'config.yaml'
            with config_path.open('w') as f:
                yaml.safe_dump(config.__dict__, f, default_flow_style=False)
            lprint(ll.INFO,  f"Configuration saved to {config_path}")
        except (IOError, PermissionError) as e:
            raise ConfigurationError(f"Failed to save configuration to {output_path}: {str(e)}")

    @staticmethod
    def load_config(config_path: Path) -> Dict[str, Any]:
        """Load configuration from a YAML file."""
        try:
            if not config_path.exists():
                raise ConfigurationError(f"Config file not found: {config_path}")
            with config_path.open('r') as f:
                config_dict = yaml.safe_load(f)
            from ..utils.validators import ConfigValidator
            ConfigValidator.validate_config(config_dict)
            lprint(ll.INFO,  f"Configuration loaded from {config_path}")
            return config_dict
        except (yaml.YAMLError, PermissionError) as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {str(e)}")
    

    @staticmethod
    def save_bayesian_search(bayesian_search: Dict[str, Any], report: List[str], output_path: Path) -> None:
        """
        Save the Bayesian search configuration and report to a pickle file for reuse and optionally to a JSON file for readability.

        Args:
            bayesian_search (Dict[str, Any]): The Bayesian search configuration dictionary.
            report (List[str]): List of report messages generated during configuration creation.
            output_path (Path): Directory where the configuration files will be saved.

        Raises:
            ConfigurationError: If saving the configuration fails due to I/O, permission, or serialization issues.
        """
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            pickle_path = output_path / 'bayesian_search.pkl'
            json_path = output_path / 'bayesian_search.json'

            # Salva in formato pickle per preservare la struttura Python
            data_to_save = {
                "bayesian_search_config": bayesian_search,
                "report": report,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(pickle_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            lprint(ll.INFO,  f"Bayesian search configuration saved to {pickle_path}")

        except (IOError, PermissionError) as e:
            lprint(ll.ERROR,  f"Failed to save Bayesian search configuration to {pickle_path}: {str(e)}")
            raise ConfigurationError(f"Failed to save Bayesian search configuration to {pickle_path}: {str(e)}")
        except Exception as e:
            lprint(ll.ERROR,  f"Unexpected error while saving Bayesian search configuration: {str(e)}")
            raise ConfigurationError(f"Unexpected error while saving Bayesian search configuration: {str(e)}")

    @staticmethod
    def load_bayesian_search(pickle_path: Path) -> Tuple[Dict[str, Any], List[str]]:
        """
        Load the Bayesian search configuration and report from a pickle file.

        Args:
            pickle_path (Path): Path to the pickle file containing the configuration.

        Returns:
            Tuple[Dict[str, Any], List[str]]: The Bayesian search configuration and report.

        Raises:
            ConfigurationError: If loading the configuration fails.
        """
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            lprint(ll.INFO,  f"Bayesian search configuration loaded from {pickle_path}")
            return data["bayesian_search_config"], data["report"]
        except (IOError, PermissionError, pickle.PickleError) as e:
            lprint(ll.ERROR,  f"Failed to load Bayesian search configuration from {pickle_path}: {str(e)}")
            raise ConfigurationError(f"Failed to load Bayesian search configuration from {pickle_path}: {str(e)}")
    
    @staticmethod
    def get_params(config:Union[RPropConfig, HRMConfig])-> Tuple:
        return(config.learning_rate,
               config.batch_size,
               len(config.hidden_layers),
               config.dropout_prob,
               config.activations,
               config.plus_delta,
               config.minus_delta,
               config.min_delta, 
               config.max_delta)
