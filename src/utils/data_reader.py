import numpy as np
import pandas as pd

def read_tsv_to_numpy(file_path, skip_header=True, dtype=None)-> np.ndarray:
    """Reads a TSV file and returns the data as a numpy matrix."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        
        if skip_header:
            matrix = df.to_numpy(dtype=dtype)
        else:
            matrix = df.to_numpy(dtype=dtype if dtype else str)
            
        return matrix
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def read_tsv_to_numpy_basic(file_path, skip_header=True, dtype=float)->np.ndarray:
    """Alternative version using only numpy (for numeric data)"""
    try:
        skiprows = 1 if skip_header else 0
        matrix = np.loadtxt(file_path, delimiter='\t', skiprows=skiprows, dtype=dtype)
        
        return matrix
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
