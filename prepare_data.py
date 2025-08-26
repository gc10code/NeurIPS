import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union

sys.path.append(str(Path(__file__).parent / "fingerprint" / "descriptors"))

from src.fingerprint.smiles import change_atom_holders
from src.fingerprint.descriptors import calculate_descriptors
from src.fingerprint.morgan import calculate_morgan_fingerprint
from src.fingerprint.map4 import calculate_map4_fingerprint
from src.utils.logging import lprint, LoggingLevels as ll

np.seterr(over='ignore', invalid='ignore')

def read_dt_list(dt_list_path: str) -> list:
    """Read descriptor names from a file, one per line.s"""
    with open(dt_list_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def add_headers_and_id(matrix: Union[np.ndarray,pd.DataFrame] , 
                       prefix: str, 
                       id_series: pd.Series, 
                       dt_list=None) -> pd.DataFrame:
    """
    Convert a numpy array or pandas DataFrame to a DataFrame with headers and an ID column.
    Args:
        matrix (np.ndarray or pd.DataFrame): Input data.
        prefix (str): Prefix for column names if dt_list is not provided.
        id_series (pd.Series): Series containing IDs to be added as the first column.
        dt_list (list of str, optional): List of column names. If None, default names will be generated.
    Returns:
        pd.DataFrame: DataFrame with headers and ID column.
    """
    if isinstance(matrix, np.ndarray):
        if dt_list:
            columns = dt_list
        else:
            columns = [f"{prefix}_{i}" for i in range(matrix.shape[1])]
        df = pd.DataFrame(matrix, columns=columns)
    else:
        df = matrix.copy()
        if df.columns.dtype == np.int64 or df.columns.astype(str).str.match(r'^\d+$').all():
            if dt_list:
                df.columns = dt_list
            else:
                df.columns = [f"{prefix}_{i}" for i in range(df.shape[1])]

    df.insert(0, 'id', id_series)
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepara descrittori, fingerprint Morgan e MAP4.")
    parser.add_argument("--input", required=True, help="File TSV di input (deve contenere colonna 'smiles').")
    parser.add_argument("--out", required=True, help="Base name per file di output")
    parser.add_argument("--dt-list", help="File con lista di descrittori (uno per riga)")
    parser.add_argument("--fm-bits", type=int, default=2048, help="Numero di bit per Morgan (default: 2048)")
    parser.add_argument("--fmap4-bits", type=int, default=1024, help="Numero di bit per MAP4 (default: 1024)")
    args = parser.parse_args()

    input_path = str(args.input)
    base_name = str(args.out)
    
    # === input ===
    lprint(ll.INFO, f"Reading input file: {input_path}")
    df = pd.read_csv(input_path, sep="\t")
    if "smiles" not in df.columns:
        lprint(ll.ERROR, "File has to have column 'smiles'.")
        sys.exit(1)

    # Column ID
    if 'id' in df.columns:
        id_series = df['id']
    else:
        id_series = df.index
        lprint(ll.INFO, "Missing Column 'id': use idx of DataFrame.")

    # SMILES
    smiles = change_atom_holders(df["smiles"])

    # === Descriptors ===
    
    dt_list = read_dt_list(args.dt_list) if args.dt_list else None
    descriptors = calculate_descriptors(smiles, dt_list)
    descriptors_df = add_headers_and_id(descriptors, "desc", id_series, dt_list)
    descriptors_df.to_csv(f"{base_name}.dt", sep="\t", index=False)
    lprint(ll.INFO, f"Saved Descriptors in {base_name}.dt")

    # === Morgan fp ===
    morgan_fps = calculate_morgan_fingerprint(smiles, n_bits=args.fm_bits)
    morgan_df = add_headers_and_id(morgan_fps, "fm", id_series)
    morgan_df.to_csv(f"{base_name}.fm", sep="\t", index=False)
    lprint(ll.INFO, f"Saved Morgan fingerprints in {base_name}.fm")

    # === MAP4 fp ===
    map4_fps = calculate_map4_fingerprint(smiles, n_bits=args.fmap4_bits)
    map4_df = add_headers_and_id(map4_fps, "map4", id_series)
    map4_df.to_csv(f"{base_name}.fmap4", sep="\t", index=False)
    lprint(ll.INFO, f"Saved MAP4 fingerprint in {base_name}.fmap4")


if __name__ == "__main__":
    main()