import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import MACCSkeys

from mordred import Calculator, descriptors

calc = Calculator(descriptors, ignore_3D=True)

MORDRED = Calculator(descriptors, ignore_3D=True)

N_BITS = 167
FINGERPRINT_COLS = [f'fp_{i}' for i in range(N_BITS)]

TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

def smile_to_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)

def gen_train_fpMaccs_descriptorMordred_file (train_file:Path, test_file:Path) -> None:
    # === Generazione full train === 
    full_train_file = f"./data/full_train.csv"
    df_train = pd.read_csv(train_file,  sep='\t')
    df_train['fp'] = df_train['SMILES'].apply(smile_to_maccs)

    fp_matrix = np.stack(df_train['fp'].values)
    fp_df = pd.DataFrame(fp_matrix, columns=FINGERPRINT_COLS)
    df_train_full = pd.concat([df_train[['id', 'SMILES'] + TARGETS].reset_index(drop=True), fp_df], axis=1)
    
    mol_list = [Chem.MolFromSmiles(s) for s in df_train['SMILES']]
    descriptor_df = MORDRED.pandas(mol_list)
    descriptor_df = descriptor_df.astype({col: np.float64 for col in descriptor_df.select_dtypes(include=['float32', 'float']).columns})

    # Normalizzo
    for col in descriptor_df.columns:
        min_val = descriptor_df[col].min()
        max_val = descriptor_df[col].max()
        descriptor_df[col] = (descriptor_df[col] - min_val) / (max_val - min_val)

    df_train_full = pd.concat([df_train_full, descriptor_df], axis=1)
    df_train_full.to_csv(full_train_file, index=False, sep='\t', float_format='%.31f')
    # ===  === === === === ===  ===

    # === Generazione full test === 
    full_test_file = f"./data/full_test.csv"
    df_test = pd.read_csv(test_file)
    df_test['fp'] = df_test['SMILES'].apply(smile_to_maccs)
    fp_test = np.stack(df_test['fp'].values)
    fp_df = pd.DataFrame(fp_test, columns=FINGERPRINT_COLS)
    df_test_full = pd.concat([df_test[['id', 'SMILES']].reset_index(drop=True), fp_df], axis=1)

    mol_list = [Chem.MolFromSmiles(s) for s in df_test['SMILES']]
    descriptor_df = MORDRED.pandas(mol_list)
    descriptor_df = descriptor_df.astype({col: np.float64 for col in descriptor_df.select_dtypes(include=['float32', 'float']).columns})

    # Normalizzo
    for col in descriptor_df.columns:
        min_val = descriptor_df[col].min()
        max_val = descriptor_df[col].max()
        descriptor_df[col] = (descriptor_df[col] - min_val) / (max_val - min_val)

    descriptor_df = descriptor_df.astype({col: np.float64 for col in descriptor_df.select_dtypes(include=['float32', 'float']).columns})
    df_test_full = pd.concat([df_test, descriptor_df], axis=1)
    df_test_full.to_csv(full_test_file, index=False)
    # ===  === === === === ===  ===

def gen_set_for_nn (full_train:Path):
    df = pd.read_csv(full_train, sep='\t')
    for target in TARGETS:
        subset = df[['id', 'SMILES'] + [target] + FINGERPRINT_COLS + descriptors._import_all_descriptors()].copy()
        subset = subset.dropna(subset=[target]).reset_index(drop=True)
        subset.insert(0, 'row_id', subset.index)
        output_file = f"./data/full_train_{target}.tsv"
        subset.to_csv(output_file, index=False, sep='\t', header=True, float_format='%.31f')
        print(f"File generato: {output_file}")

    # === Crea dataset_meta ===
    dfs_target = [pd.read_csv(f'./data/full_train_{target}.tsv', sep='\t') for target in TARGETS]
    df_meta = df[['id', 'SMILES']].copy()

    for i, df_target in enumerate(dfs_target):
        colname = f'row_id_{TARGETS[i]}'
        id_to_row = dict(zip(df_target['id'], df_target['row_id']))
        df_meta[colname] = df_meta['id'].map(id_to_row).astype('Int64')

    df_meta.to_csv('./data/meta.tsv', index=False, sep='\t')
    print("File meta.tsv generato")

    # === Estrai file value e fingerprint separati per ogni target ===
    for i in range(5):
        df_target_copy = dfs_target[i].copy()
        df_value = df_target_copy[[TARGETS[i]]]
        df_fp = df_target_copy[FINGERPRINT_COLS]
        df_descriptor = df_target_copy[descriptors._import_all_descriptors()]

        value_file = f"./data/value_{TARGETS[i]}.tsv"
        fp_file = f"./data/fp_{TARGETS[i]}.tsv"
        descriptor_file = f"./data/descriptor_{TARGETS[i]}.tsv"

        df_value.to_csv(value_file, index=False, sep='\t', header=False, float_format='%.31f')
        df_fp.to_csv(fp_file, index=False, sep='\t', header=False)
        df_descriptor.to_csv(descriptor_file, index=False, sep='\t', header=False, float_format='%.31f')
        print(f"File generati: {value_file}, {fp_file}, {descriptor_file}")


def main():
    #gen_train_fpMaccs_descriptorMordred_file ('/kaggle/input/neurips-open-polymer-prediction-2025/train.csv', '/kaggle/input/neurips-open-polymer-prediction-2025/test.csv')
    gen_train_fpMaccs_descriptorMordred_file ('./data/mega_train.csv', './data/test.csv')
    gen_set_for_nn ('./data/full_train.csv')
    

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  
    main()