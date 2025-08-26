import pandas as pd

def change_atom_holders(smiles:pd.Series,
                        atom_holders = '*',
                        substitution_group = 'C') -> pd.Series:
    """
        Take in input a series that contain a the SMILES
        and substituites the atom holders ('*' or another char that you can set)
        with an functional group or another molecule that you desire to insert.
    """
    return smiles.str.replace(atom_holders, substitution_group)