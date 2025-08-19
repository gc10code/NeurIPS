from dataclasses import dataclass,field
from typing import List

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
    MAX_EPOCHS : int = 1000
    K_FOLD : int = 5
    PATIENCE: int = 50
    MAX_COMBINATION : int = 50
    
    # FUSION Config
    MAX_EPOCHS_TEACHER: int = 1
    FUSION_EPOCHS : int = 50
    FUSION_PATIENCE: int = 5
    FUSION_TRUST: float = 0.8
    TEACHERS_TRUST: float = 0.2
    MALUS: List[float] = field(default_factory=lambda: [1.7, 1.0, 1.0, 1.0, 1.2])
    

def global_init()-> GlobalConfig:
    return GlobalConfig()