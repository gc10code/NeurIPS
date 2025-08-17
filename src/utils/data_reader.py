import numpy as np
import pandas as pd
import pandas as pd
import torch

def read_tsv_to_tensor(file_path, dtype=torch.float64) -> torch.Tensor:
    """Reads a TSV file and returns the numeric data (from the second column onwards) as a torch.Tensor."""
    try:
        # skip header
        df = pd.read_csv(file_path, sep='\t', header=0)
        # skip ID column
        numeric_data = df.iloc[:, 1:]
        # conver into torch.tensor
        return torch.tensor(numeric_data.values, dtype=dtype)
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return None