import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Union
from datetime import datetime
import multiprocessing as mp
import gc, sys, os
import time
from copy import deepcopy
from pathlib import Path
import signal
import traceback
import pandas as pd

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src._global import global_init
from src.preprocessing.file_tree import NeurIPSFiles
from src.config.config_manager import ConfigManager
from src.utils.data_reader import read_tsv_to_tensor
from src.training.data_preparation import clean_datasets
from src.training.train_rprop_mlp import rprop_mlp_main
from src.training.train_fusion_model import fusion_model_main
from src.preprocessing.proccess import preprocessing_main
from src.preprocessing.fusion_dataset import create_fusion_dataset, load_fusion_dataset
from src.utils.system_utils import *
from src.utils.logging import lprint, LoggingLevels as ll
from colorama import init

#================================
# GLOBAL SETTINGS INIT
#================================
init(autoreset=True)
np.seterr(over='ignore')
global_config = global_init()
neruips_files = NeurIPSFiles(
    train_supplements= [
        Path("./data/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv"),
        Path("./data/neurips-open-polymer-prediction-2025/train_supplement/dataset2.csv"),
        Path("./data/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv"),
        Path("./data/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv")
    ],
    train = Path("./data/neurips-open-polymer-prediction-2025/train.csv"),
    test =  Path("./data/neurips-open-polymer-prediction-2025/test.csv")
)
targets = ["Tg","FFV","Tc","Density","Rg"]
device = setup_device()

#==============================
# Helper Functions
#==============================

def load_dataset(target: str, clean = False ) -> Tuple[np.ndarray, np.ndarray]:
    lprint(ll.DEBUG, f"Loading Dataset")
    meta = pd.read_csv("./data/meta.tsv", sep="\t")
    target_ids = meta.loc[meta[target] >= 0, f"{target}"].tolist()
    X_temp = read_tsv_to_tensor("./data/descriptors.tsv")
    X = X_temp[target_ids]
    y = read_tsv_to_tensor(f"./data/target_{target}.tsv")
    if clean:
        X, y = clean_datasets(X, y)
    lprint(ll.DEBUG, f"Loaded data for {target}: X shape={X.shape}, y shape={y.shape}")
    return X, y

#==============================
# MAIN SCRIPT
#==============================
if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    lprint(ll.SUCCESS,  f"=== Main Script Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    lprint(ll.INFO,  f"Device: {device}")
    
    true_targets = []
    weigths = []

    
    try:
        # PREPROCESSING
        if global_config.PREPROCESSING:
            lprint(ll.SUCCESS,  f"=== Preprocessing files ===")
            preprocessing_main(neruips_files, targets)

            lprint(ll.SUCCESS,  f"Preprocessing files done")
        else:
            lprint(ll.SUCCESS,  f"=== Preprocessing files skipped ===")

        # PRETREINING
        if global_config.PRETREINING:
        # Training data loading and pretraining (if PRETRAINING==True)
            for target in targets:
                try:
                    lprint(ll.INFO,  f"Processing target: {target}")
                    config = ConfigManager.create_default_config()
                    config.output_dir = f"{config.output_dir}/{target}"
                    remove_dir(Path(config.output_dir))
                    os.makedirs(config.output_dir, exist_ok=True)
                    lprint(ll.INFO,  f"Output directory created: {config.output_dir}")
                    problematic_values_file = Path(config.output_dir) / "logs/problematic_values.log"
                    problematic_values_file.parent.mkdir(parents=True, exist_ok=True)
                    problematic_values_file.touch(exist_ok=True)
                    lprint(ll.DEBUG,  f"Created problematic values log file: {problematic_values_file}") 
                    X,y =load_dataset(target) 
                    true_targets.append((torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64), X.shape[0]))

                except Exception as e:
                    lprint(ll.ERROR,  f"Data loading failed for {target}: {str(e)}")
                    raise
                
                try:
                    lprint(ll.INFO,  f"Starting pretraining for {target}")
                    # Try with multiprocessing first, fallback to single-threaded if it fails
                    try:
                        model = rprop_mlp_main(target, config, X, y, use_multiprocessing=True, max_combinations=global_config.MAX_COMBINATION)
                    except Exception as mp_e:
                        lprint(ll.WARN,  f"Multiprocessing failed for {target}: {str(mp_e)}. Falling back to single-threaded.")
                        model = rprop_mlp_main(config, X, y, use_multiprocessing=False, max_combinations=global_config.MAX_COMBINATION)
                    lprint(ll.SUCCESS,  f"Pretraining completed for {target}")
                except Exception as e:
                    lprint(ll.ERROR,  f"rprop_mlp_main failed for {target}: {str(e)}")
                    raise
            try:
                fusion_descriptors_file, fusion_meta_file, fusion_target_file = create_fusion_dataset(targets)
            except Exception as e:
                lprint(ll.ERROR,  f"Computation of Fusion Dataset failed: {e}")
                raise
        else: 
            lprint(ll.SUCCESS, f"Skipping pretraining. Set PRETREINING = True if you want to generate pretrained models")

        # FUSION
        if global_config.FUSION:
            try:
                if not global_config.PRETREINING:
                    lprint(ll.INFO, "Creating fusion dataset")
                    try:   
                        for target in targets:
                            X,y =load_dataset(target) 
                            true_targets.append((torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64), X.shape[0]))
                        fusion_descriptors_file, fusion_meta_file, fusion_target_file = create_fusion_dataset(targets)
                    except Exception as e:
                        lprint(ll.ERROR,  f"Computation of Fusion Dataset failed: {e}")
                        raise                     
                X_fusion,y_fusion, valid_mask =load_fusion_dataset(targets)
                for i, (y_i, m_i) in enumerate(zip(y_fusion, valid_mask)):
                    print(y_i, m_i)
                del X, y #
                lprint(ll.INFO, f"Fusion dataset loaded: X shape={X_fusion.shape}, y shapes={[y.shape for y in y_fusion]}, valid mask shapes={[m.shape for m in valid_mask]}")
            except Exception as e:
                lprint(ll.ERROR, f"Fusion dataset creation failed: {str(e)}")
                raise
            
            try:
                lprint(ll.INFO, "Starting fusion model training")
                fusion_config = ConfigManager.create_default_config("fusion")
                fusion_config.output_dir = f"{fusion_config.output_dir}/fusion_model"
                remove_dir(Path(fusion_config.output_dir))
                os.makedirs(fusion_config.output_dir, exist_ok=True)
                
                fusion_model, final_loss = fusion_model_main(targets, true_targets, X_fusion, y_fusion, valid_mask, device, global_config.MAX_COMBINATION)
                lprint(ll.INFO, f"Fusion model training completed with final loss: {final_loss:.4f}")
            except Exception as e:
                lprint(ll.ERROR, f"Fusion model training failed: {str(e)}")
                raise
        else: 
            lprint(ll.SUCCESS, f"Skipping Fusion Model training")

    except Exception as e:
        lprint(ll.ERROR,  f"Script execution failed: {str(e)}")
        raise
    finally:
        lprint(ll.INFO,  "Cleaning up resources")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        lprint(ll.INFO,  "Script execution completed")