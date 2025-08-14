from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union
import pandas as pd
from src.fingerprint.descriptors import calculate_descriptors
import os, shutil

@dataclass
class NeurIPSFiles:
    train_supplements: List[Path] 
    train: Path
    test: Path


def preprocessing_files(NeurIPSFiles:NeurIPSFiles, targets: List[str]):
    df_train = pd.read_csv(NeurIPSFiles.train, header=0)
    
    id_col = df_train.iloc[:, 0]
    smiles = df_train.iloc[:, 1]
    
    values = {}
    missing_indices = {}
    meta_dict = {'id': id_col}
    
    for target in targets:
        series = df_train[target]
        values[target] = series
        missing_idx = series[series.isna()].index.tolist()
        missing_indices[target] = missing_idx
        # meta column for target
        meta_col = [-1 if pd.isna(val) else 0 for val in series]  # initialization to 0
        # replace with incresing number
        count = 0
        for i in range(len(meta_col)):
            if meta_col[i] != -1:
                meta_col[i] = count
                count += 1
        
        meta_dict[target] = meta_col

    meta = pd.DataFrame(meta_dict)
    
    # Target Series without NaN
    target_series_non_na = {target: df_train[target].dropna().reset_index(drop=True) for target in targets}
    # calculate descriptors df
    df_descriptors = calculate_descriptors(smiles)
    
    return values, missing_indices, meta, target_series_non_na, df_descriptors