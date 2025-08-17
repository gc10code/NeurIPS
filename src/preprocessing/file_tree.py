from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union, Tuple
import pandas as pd
from src.fingerprint.descriptors import calculate_descriptors
from src.fingerprint.descriptors import change_atom_holders
import os, shutil
import numpy as np

from src.utils.logging import lprint , LoggingLevels as ll

@dataclass
class NeurIPSFiles:
    train_supplements: List[Path] 
    train: Path
    test: Path


def preprocessing_files(NeurIPSFiles: NeurIPSFiles, targets: List[str]):
    df_train = pd.read_csv(NeurIPSFiles.train, header=0)
    
    id_col = df_train.iloc[:, 0]
    smiles_temp = df_train.iloc[:, 1]
    smiles = change_atom_holders(smiles_temp.copy())
    smiles_temp = None
    
    values = {}
    missing_indices = {}
    meta_dict = {'id': id_col}
    
    for target in targets:
        series = df_train[target]
        values[target] = series
        missing_idx = series[series.isna()].index.tolist()
        missing_indices[target] = missing_idx
        
        meta_col = [-1 if pd.isna(val) else 0 for val in series]
        count = 0
        for i in range(len(meta_col)):
            if meta_col[i] != -1:
                meta_col[i] = count
                count += 1
        
        meta_dict[target] = meta_col

    meta = pd.DataFrame(meta_dict)
    
    target_series_non_na = {
        target: pd.DataFrame({
            'id': id_col[~df_train[target].isna()].reset_index(drop=True),
            target: df_train[target].dropna().reset_index(drop=True)
        })[['id', target]]
        for target in targets
    }
    
    df_descriptors = calculate_descriptors(smiles)
    df_descriptors.insert(0, 'id', id_col)
    
    return values, missing_indices, meta, target_series_non_na, df_descriptors

