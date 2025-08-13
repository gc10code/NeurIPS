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

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.config_manager import ConfigManager
from src.utils.data_reader import read_tsv_to_numpy
from src.training.train_rprop_mlp import rprop_mlp_main
from src.training.train_fusion_model import fusion_model_main
from src.utils.system_utils import *
from src.utils.logging import lprint, LoggingLevels as ll

MAX_COMBINATION = 20
PRETRAINING = True
targets = ["Density", "FFV", "Rg", "Tc", "Tg"]
device = setup_device()

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    lprint(ll.SUCCESS,  f"=== Main Script Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    lprint(ll.INFO,  f"Device: {device}")
    
    true_targets = []
    try:
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
                
                X = read_tsv_to_numpy(f"./data/descriptor_{target}.tsv")
                y = read_tsv_to_numpy(f"./data/value_{target}.tsv")
                lprint(ll.INFO,  f"Loaded data for {target}: X shape={X.shape}, y shape={y.shape}")
                true_targets.append((X, y, X.shape[0]))
            except Exception as e:
                lprint(ll.ERROR,  f"Data loading failed for {target}: {str(e)}", exc_info=True)
                raise

            if PRETRAINING:
                try:
                    lprint(ll.SUCCESS,  f"Starting pretraining for {target}")
                    # Try with multiprocessing first, fallback to single-threaded if it fails
                    try:
                        model = rprop_mlp_main(target, config, X, y, use_multiprocessing=True, max_combinations=MAX_COMBINATION)
                    except Exception as mp_e:
                        lprint(ll.WARN,  f"Multiprocessing failed for {target}: {str(mp_e)}. Falling back to single-threaded.")
                        model = rprop_mlp_main(config, X, y, use_multiprocessing=False, max_combinations=MAX_COMBINATION)
                    lprint(ll.SUCCESS,  f"Pretraining completed for {target}")
                except Exception as e:
                    lprint(ll.ERROR,  f"rprop_mlp_main failed for {target}: {str(e)}", exc_info=True)
                    raise
        
        # Training Fusion Model
        try:
            lprint(ll.SUCCESS,  "=== Starting fusion model training ===")
            fusion_config = ConfigManager.create_default_config("fusion")
            fusion_config.output_dir = f"{fusion_config.output_dir}/fusion_model"
            remove_dir(Path(fusion_config.output_dir))
            os.makedirs(fusion_config.output_dir, exist_ok=True)
            lprint(ll.INFO,  f"Fusion model output directory created: {fusion_config.output_dir}")
            lprint(ll.INFO,  f"Using device: {device}")
            
            fusion_model, final_loss = fusion_model_main(targets, true_targets, project_root, device, MAX_COMBINATION)
            lprint(ll.SUCCESS,  f"=== Fusion model training completed with final loss: {final_loss:.4f} ===")
        except Exception as e:
            lprint(ll.ERROR,  f"Fusion model training failed: {str(e)}", exc_info=True)
            raise
    
    except Exception as e:
        lprint(ll.ERROR,  f"Script execution failed: {str(e)}", exc_info=True)
        raise
    finally:
        lprint(ll.INFO,  "Cleaning up resources")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        lprint(ll.INFO,  "Script execution completed")