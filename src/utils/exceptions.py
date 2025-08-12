class MLPError(Exception):
    """Base exception for MLP project errors."""
    pass

class ConfigurationError(MLPError):
    """Raised for configuration-related errors."""
    pass

class DataError(MLPError):
    """Raised for data validation or processing errors."""
    pass

class TrainingError(MLPError):
    """Raised for training-related errors."""
    pass

class IOOperationError(MLPError):
    """Raised for I/O operation errors."""
    pass