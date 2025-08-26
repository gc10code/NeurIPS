import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.logging import lprint, LoggingLevels as ll
from src.config.file_tree import NeurIPSFiles
from src._global import global_init, NEURIPS_FILES, TARGETS

from rdkit import Chem
import pandas as pd
import numpy as np
from mordred import Calculator, descriptors
from typing import List
import torch
import pickle

#==========================================
# 1. Calculate Descriptors and Fingerprints
#==========================================
np.seterr(over='ignore')
_global = global_init()

PATH_DESCRIPTORS = Path("./data/descriptors.pkl")
PATH_RELEVANT_DESCRIPTORS = Path("./data/relevant_descriptors.pkl")

def change_atom_holders(smiles: pd.Series, atom_holders: str = '*', substitution_group: str = 'C') -> pd.Series:
    """
    Substitute atom holders (e.g., '*') in SMILES strings with a specified functional group or molecule.

    Args:
        smiles: Series containing SMILES strings.
        atom_holders: Character representing atom holders (default: '*').
        substitution_group: Group to replace atom holders with (default: 'C').

    Returns:
        Series with substituted SMILES strings.
    """
    return smiles.str.replace(atom_holders, substitution_group, regex=False)

def is_numeric_series(series: pd.Series) -> bool:
    """
    Check if a pandas Series contains only numeric values.

    Args:
        series: Series to check.

    Returns:
        bool: True if the series is numeric.
    """
    s = series.dropna().astype(str)
    return pd.to_numeric(s, errors='coerce').notna().all()

def calculate_descriptors(smiles: pd.Series, save: bool = True, zero_nan_threshold: float = 0.05) -> pd.DataFrame:
    """
    Calculate molecular descriptors for a series of SMILES strings and filter invalid ones.

    Args:
        smiles: Series containing SMILES strings.
        save: Whether to save the descriptors to a file (default: True).
        zero_nan_threshold: Threshold for dropping columns with excessive NaNs or zeros (default: 0.05).

    Returns:
        DataFrame containing calculated descriptors.
    """
    lprint(ll.INFO, "Computing molecular descriptors")
    descriptor_calculator = Calculator(descriptors, ignore_3D=True)
    
    # Convert SMILES to RDKit molecules, handling invalid SMILES
    mol_list = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            lprint(ll.WARN, f"Invalid SMILES string: {s}")
        mol_list.append(mol)
    
    # Calculate descriptors
    df_descriptor = descriptor_calculator.pandas(mol_list)
    
    # Filter for numeric columns
    valid_cols = [col for col in df_descriptor.columns if is_numeric_series(df_descriptor[col])]
    df_descriptor = df_descriptor[valid_cols]
    
    # Convert float columns to np.float32 to match PyTorch default
    df_descriptor = df_descriptor.astype({
        col: np.float32 for col in df_descriptor.select_dtypes(include=['float32', 'float64', 'float']).columns
    })

    # Remove columns with excessive NaNs or zeros
    col_to_remove = []
    for des in df_descriptor.columns:
        col_data = df_descriptor[des]
        zero_count = (col_data == 0.0).sum()
        nan_count = col_data.isnull().sum()
        if nan_count > 0 or (zero_count / len(col_data)) > zero_nan_threshold:
            col_to_remove.append(des)
    df_descriptor.drop(columns=col_to_remove, inplace=True)

    if save:
        with open(PATH_DESCRIPTORS, 'wb') as file:
            pickle.dump(df_descriptor, file)

    return df_descriptor

def preprocessing_files(neurips_files: NeurIPSFiles, targets: List[str]) -> pd.DataFrame:
    """
    Preprocess NeurIPS dataset by calculating descriptors for SMILES strings.

    Args:
        neurips_files: NeurIPSFiles object containing file paths.
        targets: List of target columns.

    Returns:
        DataFrame with descriptors and 'id' column.
    """
    df_train = pd.read_csv(neurips_files.train, header=0)
    
    id_col = df_train.iloc[:, 0]
    smiles_temp = df_train.iloc[:, 1]
    smiles = change_atom_holders(smiles_temp.copy())
    
    df_descriptors = calculate_descriptors(smiles)
    
    # Use pd.concat to avoid fragmentation
    df_descriptors = pd.concat([id_col.rename('id'), df_descriptors], axis=1)
    
    return df_descriptors

def main_data_analysis(neurips_files: NeurIPSFiles, targets: List[str]) -> None:
    """
    Main function to load NeurIPS data, compute descriptors, and run DAE analysis.

    Args:
        neurips_files: NeurIPSFiles object containing file paths.
        targets: List of target columns.
    """
    lprint(ll.INFO, "Loading NeurIPS data")
    
    df_descriptors = preprocessing_files(neurips_files, targets)
    descriptors_list = df_descriptors.columns[1:]
    if not descriptors_list.empty():
        with open(PATH_RELEVANT_DESCRIPTORS, 'wb') as f:
            pickle.dump(descriptors_list, f)
    
if __name__ == "__main__":
    main_data_analysis(NEURIPS_FILES, TARGETS)

