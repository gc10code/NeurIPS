import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

import itertools
from collections import defaultdict

from mhfp.encoder import MHFPEncoder
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdmolops import GetDistanceMatrix


def to_smiles(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


class MAP4Calculator:

    def __init__(self, dimensions=1024, radius=2, is_counted=False, return_strings=False):
        """
        MAP4 calculator class
        """
        self.dimensions = dimensions
        self.radius = radius
        self.is_counted = is_counted
        self.return_strings = return_strings
        self.encoder = MHFPEncoder(dimensions)

    def calculate(self, mol):
        """Calculates the atom pair minhashed fingerprint

        Arguments:
            mol -- rdkit mol object

        Returns:
            tmap VectorUint -- minhashed fingerprint
        """
        
        atom_env_pairs = self._calculate(mol)
        return self._fold(atom_env_pairs)
       
    def calculate_many(self, mols):
        """ Calculates the atom pair minhashed fingerprint

        Arguments:
            mols -- list of mols

        Returns:
            list of tmap VectorUint -- minhashed fingerprints list
        """

        atom_env_pairs_list = [self._calculate(mol) for mol in mols]
        return [self._fold(pairs) for pairs in atom_env_pairs_list]

    def _calculate(self, mol):
        return self._all_pairs(mol, self._get_atom_envs(mol))

    def _fold(self, pairs):
        fp_hash = self.encoder.hash(set(pairs))
        return self.encoder.fold(fp_hash, self.dimensions)

    def _get_atom_envs(self, mol: Chem.Mol):
        atoms_env = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            for radius in range(1, self.radius + 1):
                if idx not in atoms_env:
                    atoms_env[idx] = []
                atoms_env[idx].append(MAP4Calculator._find_env(mol, idx, radius))
        return atoms_env

    @classmethod
    def _find_env(cls, mol, idx, radius):
        env = rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius, idx)
        atom_map = {}

        submol = Chem.PathToSubmol(mol, env, atomMap=atom_map)
        if idx in atom_map:
            smiles = Chem.MolToSmiles(submol, rootedAtAtom=atom_map[idx], canonical=True, isomericSmiles=False)
            return smiles
        return ''

    def _all_pairs(self, mol, atoms_env):
        atom_pairs = []
        distance_matrix = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        shingle_dict = defaultdict(int)
        for idx1, idx2 in itertools.combinations(range(num_atoms), 2):
            dist = str(int(distance_matrix[idx1][idx2]))

            for i in range(self.radius):
                env_a = atoms_env[idx1][i]
                env_b = atoms_env[idx2][i]

                ordered = sorted([env_a, env_b])

                shingle = '{}|{}|{}'.format(ordered[0], dist, ordered[1])

                if self.is_counted:
                    shingle_dict[shingle] += 1
                    shingle += '|' + str(shingle_dict[shingle])

                atom_pairs.append(shingle.encode('utf-8'))
        return list(set(atom_pairs))


def calculate_map4_fingerprint(smiles: pd.Series, n_bits: int = 1024) -> pd.DataFrame:
    """
    Calculate MAP4 fingerprints for a list of SMILES strings and return as a pandas DataFrame.

    Args:
        smiles_list (list of str): List of SMILES strings.

    Returns:
        pd.DataFrame: DataFrame where each row is a MAP4 fingerprint.
    """

    calc: MAP4Calculator = MAP4Calculator(dimensions=1024)
    fingerprints = []
    valid_indices = []

    for idx, smi in enumerate(smiles.tolist()):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = calc.calculate(mol)
            fingerprints.append(fp)
            valid_indices.append(idx)
        else:
            fingerprints.append([None]*1024)
            valid_indices.append(idx)

    df = pd.DataFrame(fingerprints, index=valid_indices)
    return df

