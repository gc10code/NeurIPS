from dataclasses import dataclass

@dataclass
class GlobalConfig:
    # General Config
    SEED: int = 42
    DEBUG_MODE : bool = True
    TIME_MODE : bool = False

    # STAGES Config
    PREPROCESSING :bool = False
    PRETREINING : bool = False
    FUSION : bool = True
    
    # RPROP_MLB Config
    MAX_EPOCHS : int = 5
    K_FOLD : int = 2
    PATIENCE: int = 20
    MAX_COMBINATION : int = 20
    
    # FUSION Config
    MAX_EPOCHS_TEACHER: int = 5
    FUSION_EPOCHS : int = 1
    FUSION_PATIENCE: int = 1
    

def global_init()-> GlobalConfig:
    return GlobalConfig()