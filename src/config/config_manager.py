from pathlib import Path
import yaml, json, pickle
import logging
from typing import Dict, Any, List, Tuple, Union
from datetime import datetime
from skopt.space import Real, Integer, Categorical

from src.config.model_config import ModelConfig, FusionConfig
from src.utils.exceptions import ConfigurationError

logger = logging.getLogger('RPropMLP')

class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    # General Parameters
    LOGGING_EPOCHS = 100
    START_MAX_LOSS_VAL = 1e10


    @staticmethod
    def create_default_config(type: str = "rprop") -> Union[ModelConfig, FusionConfig]:
        """Create a default ModelConfig instance."""
        try:
            if type == "rprop":
                return ModelConfig()
            elif type == "fusion":
                return FusionConfig()
        except Exception as e:
            raise ConfigurationError(f"Failed to create default configuration: {str(e)}")

    @staticmethod
    def save_config(config: ModelConfig, output_path: Path) -> None:
        """Save configuration to a YAML file."""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            config_path = output_path / 'config.yaml'
            with config_path.open('w') as f:
                yaml.safe_dump(config.__dict__, f, default_flow_style=False)
            logger.info(f"Configuration saved to {config_path}")
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
            logger.info(f"Configuration loaded from {config_path}")
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
            logger.info(f"Bayesian search configuration saved to {pickle_path}")

        except (IOError, PermissionError) as e:
            logger.error(f"Failed to save Bayesian search configuration to {pickle_path}: {str(e)}")
            raise ConfigurationError(f"Failed to save Bayesian search configuration to {pickle_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error while saving Bayesian search configuration: {str(e)}")
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
            logger.info(f"Bayesian search configuration loaded from {pickle_path}")
            return data["bayesian_search_config"], data["report"]
        except (IOError, PermissionError, pickle.PickleError) as e:
            logger.error(f"Failed to load Bayesian search configuration from {pickle_path}: {str(e)}")
            raise ConfigurationError(f"Failed to load Bayesian search configuration from {pickle_path}: {str(e)}")
    
    @staticmethod
    def get_params(config:Union[ModelConfig, FusionConfig])-> Tuple:
        return(config.learning_rate,
               config.batch_size,
               len(config.hidden_layers),
               config.dropout_prob,
               config.activations,
               config.plus_delta,
               config.minus_delta,
               config.min_delta, 
               config.max_delta)
    
    @staticmethod
    def update_config(config:Union[ModelConfig, FusionConfig], args:Tuple)-> Union[ModelConfig, FusionConfig]:
        learning_rate, batch_size, hidden_layers, activations, plus_delta, minus_delta, min_delta, max_delta = args
        
        config.learning_rate = learning_rate 
        config.batch_size = batch_size
        config.hidden_layers = hidden_layers
        config.activations = activations
        config.plus_delta = plus_delta
        config.minus_delta = minus_delta
        config.min_delta = min_delta
        config.max_delta = max_delta

        return config