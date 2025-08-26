from rdkit import Chem

import pandas as pd

from rdkit.Chem import rdFingerprintGenerator

def calculate_morgan_fingerprint(smiles: pd.Series, radius: int = 2, n_bits: int = 1024) -> pd.DataFrame:
    """
    Generate Morgan fingerprints using the new RDKit MorganGenerator.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fingerprints = []
    for smi in smiles.tolist():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fingerprints.append([0] * n_bits)
            continue
        fp = gen.GetFingerprint(mol)
        fingerprints.append(list(fp))
    return pd.DataFrame(fingerprints)
