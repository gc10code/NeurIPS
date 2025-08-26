from pathlib import Path
from dataclasses import dataclass,field
from typing import List

from src.config.file_tree import NeurIPSFiles
from src.utils.system_utils import setup_device

import torch

TARGETS = ["Tg","FFV","Tc","Density","Rg"]

NEURIPS_FILES = NeurIPSFiles(
    train_supplements= [
        Path("./data/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv"),
        Path("./data/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv"),
        Path("./data/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv"),
        Path("./data/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv")
    ],
    train = Path("./data/neurips-open-polymer-prediction-2025/train.csv"),
    test =  Path("./data/neurips-open-polymer-prediction-2025/test.csv")
)


@dataclass
class GlobalConfig:
    # General Config
    SEED: int = 42
    DEBUG_MODE : bool = False
    TIME_MODE : bool = False
    DEVICE : torch.device = setup_device()
    
    # DATA ANALYSIS
    CALCULATE_DESCRIPTORS: bool = True

    # STAGES Config
    PREPROCESSING :bool = False
    PRETREINING : bool = True
    FUSION : bool = False
    
    # DAE 
    DAE_N_TRIALS: int = 50
    DAE_EPOCHS: int = 500
    DAE_PATIENCE: int = 20
    DAE_IMPORTANCE_THRESHOLD : float = 0.015

    # RPROP_MLB Config
    MAX_EPOCHS : int = 1000
    LOGGING_EPOCHS: int = 100
    K_FOLD : int = 3
    PATIENCE: int = 50
    N_TRIALS : int = 100
    
    # FUSION Config
    MAX_EPOCHS_TEACHER: int = 1
    FUSION_EPOCHS : int = 50
    FUSION_PATIENCE: int = 5
    FUSION_TRUST: float = 0.8
    TEACHERS_TRUST: float = 0.2
    MALUS: List[float] = field(default_factory=lambda: [1.7, 1.0, 1.0, 1.0, 1.2])
    

def global_init()-> GlobalConfig:
    return GlobalConfig()