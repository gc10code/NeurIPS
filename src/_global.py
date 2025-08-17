from dataclasses import dataclass

@dataclass
class GlobalConfig:
    # General Config
    SEED: int = 42
    DEBUG_MODE : bool = True
    TIME_MODE : bool = False
    PREPROCESSING :bool = False
    PRETREINING : bool = False
    FUSION : bool = True
    
    # RPROP_MLB Config
    MAX_EPOCHS : int = 1000
    K_FOLD : int = 5
    PATIENCE: int = 50
    MAX_COMBINATION : int = 50
    
    # FUSION Config
    MAX_EPOCHS_TEACHER: int = 5
    

def global_init()-> GlobalConfig:
    return GlobalConfig()