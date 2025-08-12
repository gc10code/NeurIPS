import numpy as np
import pandas as pd

def read_tsv_to_numpy(file_path, skip_header=True, dtype=None):
    """
    Reads a TSV file and returns the data as a numpy matrix.
    
    Parameters:
    - file_path (str): path to the TSV file to read
    - skip_header (bool): if True, skips the first row (headers)
    - dtype: data type for the numpy matrix (None for auto-detect)
    
    Returns:
    - numpy.ndarray: numpy matrix with the TSV file data
    """
    try:
        # Method 1: using pandas (more robust)
        df = pd.read_csv(file_path, sep='\t')
        
        if skip_header:
            # Converts only numeric data, excluding headers
            matrix = df.to_numpy(dtype=dtype)
        else:
            # Includes everything as string if header should not be skipped
            matrix = df.to_numpy(dtype=dtype if dtype else str)
            
        return matrix
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def read_tsv_to_numpy_basic(file_path, skip_header=True, dtype=float):
    """
    Alternative version using only numpy (for numeric data).
    
    Parameters:
    - file_path (str): path to the TSV file to read
    - skip_header (bool): if True, skips the first row
    - dtype: data type for the numpy matrix
    
    Returns:
    - numpy.ndarray: numpy matrix with the TSV file data
    """
    try:
        # Method 2: using numpy directly (only for numeric data)
        skiprows = 1 if skip_header else 0
        matrix = np.loadtxt(file_path, delimiter='\t', skiprows=skiprows, dtype=dtype)
        
        return matrix
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Usage example
if __name__ == "__main__":
    # Example with file containing numeric data
    filename = "./data/value_Density.tsv"
    
    # Using pandas (recommended)
    matrix1 = read_tsv_to_numpy(filename, skip_header=True)
    if matrix1 is not None:
        print("Matrix read with pandas:")
        print(matrix1)
        print(f"Dimensions: {matrix1.shape}")
        print(f"Data type: {matrix1.dtype}")
    
    # Using numpy directly
    matrix2 = read_tsv_to_numpy_basic(filename, skip_header=True)
    if matrix2 is not None:
        print("\nMatrix read with numpy:")
        print(matrix2)
        print(f"Dimensions: {matrix2.shape}")
        print(f"Data type: {matrix2.dtype}")