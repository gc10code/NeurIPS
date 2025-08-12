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
DESCRIPTOR_COL = ['SpAbs_A', 'SpMax_A', 'SpDiam_A', 'SpAD_A', 'SpMAD_A', 'LogEE_A', 'VE1_A', 'VE2_A', 'VE3_A', 'VR3_A', 'nAtom', 'nHeavyAtom', 'nHetero', 'nH', 'nC', 'ATS0dv', 'ATS1dv', 'ATS2dv', 'ATS3dv', 'ATS4dv', 'ATS5dv', 'ATS6dv', 'ATS0d', 'ATS1d', 'ATS2d', 'ATS3d', 'ATS4d', 'ATS5d', 'ATS6d', 'ATS7d', 'ATS8d', 'ATS0s', 'ATS1s', 'ATS2s', 'ATS3s', 'ATS4s', 'ATS5s', 'ATS6s', 'ATS7s', 'ATS8s', 'ATS0Z', 'ATS1Z', 'ATS2Z', 'ATS3Z', 'ATS4Z', 'ATS5Z', 'ATS6Z', 'ATS7Z', 'ATS8Z', 'ATS0m', 'ATS1m', 'ATS2m', 'ATS3m', 'ATS4m', 'ATS5m', 'ATS6m', 'ATS7m', 'ATS8m', 'ATS0v', 'ATS1v', 'ATS2v', 'ATS3v', 'ATS4v', 'ATS5v', 'ATS6v', 'ATS7v', 'ATS8v', 'ATS0se', 'ATS1se', 'ATS2se', 'ATS3se', 'ATS4se', 'ATS5se', 'ATS6se', 'ATS7se', 'ATS8se', 'ATS0pe', 'ATS1pe', 'ATS2pe', 'ATS3pe', 'ATS4pe', 'ATS5pe', 'ATS6pe', 'ATS7pe', 'ATS8pe', 'ATS0are', 'ATS1are', 'ATS2are', 'ATS3are', 'ATS4are', 'ATS5are', 'ATS6are', 'ATS7are', 'ATS8are', 'ATS0p', 'ATS1p', 'ATS2p', 'ATS3p', 'ATS4p', 'ATS5p', 'ATS6p', 'ATS7p', 'ATS8p', 'ATS0i', 'ATS1i', 'ATS2i', 'ATS3i', 'ATS4i', 'ATS5i', 'ATS6i', 'ATS7i', 'ATS8i', 'AATS0dv', 'AATS1dv', 'AATS2dv', 'AATS3dv', 'AATS4dv', 'AATS0d', 'AATS1d', 'AATS2d', 'AATS3d', 'AATS4d', 'AATS0s', 'AATS1s', 'AATS2s', 'AATS3s', 'AATS4s', 'AATS0Z', 'AATS1Z', 'AATS2Z', 'AATS3Z', 'AATS4Z', 'AATS0m', 'AATS1m', 'AATS2m', 'AATS3m', 'AATS4m', 'AATS0v', 'AATS1v', 'AATS2v', 'AATS3v', 'AATS4v', 'AATS0se', 'AATS1se', 'AATS2se', 'AATS3se', 'AATS4se', 'AATS0pe', 'AATS1pe', 'AATS2pe', 'AATS3pe', 'AATS4pe', 'AATS0are', 'AATS1are', 'AATS2are', 'AATS3are', 'AATS4are', 'AATS0p', 'AATS1p', 'AATS2p', 'AATS3p', 'AATS4p', 'AATS0i', 'AATS1i', 'AATS2i', 'AATS3i', 'AATS4i', 'ATSC0dv', 'ATSC1dv', 'ATSC2dv', 'ATSC3dv', 'ATSC4dv', 'ATSC0d', 'ATSC1d', 'ATSC0s', 'ATSC2s', 'ATSC3s', 'ATSC4s', 'ATSC8s', 'ATSC0Z', 'ATSC0m', 'ATSC4m', 'ATSC0v', 'ATSC4v', 'ATSC0se', 'ATSC0pe', 'ATSC4pe', 'ATSC0are', 'ATSC0p', 'ATSC4p', 'ATSC0i', 'ATSC4i', 'AATSC0dv', 'AATSC1dv', 'AATSC0d', 'AATSC1d', 'AATSC0s', 'AATSC2s', 'AATSC0Z', 'AATSC0m', 'AATSC4m', 'AATSC0v', 'AATSC4v', 'AATSC0se', 'AATSC0pe', 'AATSC0are', 'AATSC0p', 'AATSC0i', 'MATS1dv', 'MATS1d', 'MATS2s', 'GATS1dv', 'GATS2dv', 'GATS3dv', 'GATS4dv', 'GATS1d', 'GATS2d', 'GATS3d', 'GATS4d', 'GATS1s', 'GATS2s', 'GATS3s', 'GATS4s', 'GATS1Z', 'GATS2Z', 'GATS3Z', 'GATS4Z', 'GATS1m', 'GATS2m', 'GATS3m', 'GATS4m', 'GATS1v', 'GATS2v', 'GATS3v', 'GATS4v', 'GATS1se', 'GATS2se', 'GATS3se', 'GATS4se', 'GATS1pe', 'GATS2pe', 'GATS3pe', 'GATS4pe', 'GATS1are', 'GATS2are', 'GATS3are', 'GATS4are', 'GATS1p', 'GATS2p', 'GATS3p', 'GATS4p', 'GATS1i', 'GATS2i', 'GATS3i', 'GATS4i', 'BCUTdv-1h', 'BCUTdv-1l', 'BCUTd-1h', 'BCUTd-1l', 'BCUTs-1h', 'BCUTs-1l', 'BCUTZ-1h', 'BCUTZ-1l', 'BCUTm-1h', 'BCUTm-1l', 'BCUTv-1h', 'BCUTv-1l', 'BCUTse-1h', 'BCUTse-1l', 'BCUTpe-1h', 'BCUTpe-1l', 'BCUTare-1h', 'BCUTare-1l', 'BCUTp-1h', 'BCUTp-1l', 'BCUTi-1h', 'BCUTi-1l', 'BalabanJ', 'SpAbs_DzZ', 'SpMax_DzZ', 'SpDiam_DzZ', 'SpAD_DzZ', 'SpMAD_DzZ', 'LogEE_DzZ', 'SM1_DzZ', 'VE1_DzZ', 'VE2_DzZ', 'VE3_DzZ', 'VR1_DzZ', 'VR2_DzZ', 'VR3_DzZ', 'SpAbs_Dzm', 'SpMax_Dzm', 'SpDiam_Dzm', 'SpAD_Dzm', 'SpMAD_Dzm', 'LogEE_Dzm', 'SM1_Dzm', 'VE1_Dzm', 'VE2_Dzm', 'VE3_Dzm', 'VR1_Dzm', 'VR2_Dzm', 'VR3_Dzm', 'SpAbs_Dzv', 'SpMax_Dzv', 'SpDiam_Dzv', 'SpAD_Dzv', 'SpMAD_Dzv', 'LogEE_Dzv', 'SM1_Dzv', 'VE1_Dzv', 'VE2_Dzv', 'VE3_Dzv', 'VR1_Dzv', 'VR2_Dzv', 'VR3_Dzv', 'SpAbs_Dzse', 'SpMax_Dzse', 'SpDiam_Dzse', 'SpAD_Dzse', 'SpMAD_Dzse', 'LogEE_Dzse', 'SM1_Dzse', 'VE1_Dzse', 'VE2_Dzse', 'VE3_Dzse', 'VR1_Dzse', 'VR2_Dzse', 'VR3_Dzse', 'SpAbs_Dzpe', 'SpMax_Dzpe', 'SpDiam_Dzpe', 'SpAD_Dzpe', 'SpMAD_Dzpe', 'LogEE_Dzpe', 'SM1_Dzpe', 'VE1_Dzpe', 'VE2_Dzpe', 'VE3_Dzpe', 'VR1_Dzpe', 'VR2_Dzpe', 'VR3_Dzpe', 'SpAbs_Dzare', 'SpMax_Dzare', 'SpDiam_Dzare', 'SpAD_Dzare', 'SpMAD_Dzare', 'LogEE_Dzare', 'SM1_Dzare', 'VE1_Dzare', 'VE2_Dzare', 'VE3_Dzare', 'VR1_Dzare', 'VR2_Dzare', 'VR3_Dzare', 'SpAbs_Dzp', 'SpMax_Dzp', 'SpDiam_Dzp', 'SpAD_Dzp', 'SpMAD_Dzp', 'LogEE_Dzp', 'SM1_Dzp', 'VE1_Dzp', 'VE2_Dzp', 'VE3_Dzp', 'VR1_Dzp', 'VR2_Dzp', 'VR3_Dzp', 'SpAbs_Dzi', 'SpMax_Dzi', 'SpDiam_Dzi', 'SpAD_Dzi', 'SpMAD_Dzi', 'LogEE_Dzi', 'SM1_Dzi', 'VE1_Dzi', 'VE2_Dzi', 'VE3_Dzi', 'VR1_Dzi', 'VR2_Dzi', 'VR3_Dzi', 'BertzCT', 'nBonds', 'nBondsO', 'nBondsS', 'nBondsM', 'nBondsKS', 'nBondsKD', 'HybRatio', 'FCSP3', 'Xc-3d', 'Xc-3dv', 'Xpc-4d', 'Xpc-5d', 'Xpc-6d', 'Xpc-4dv', 'Xpc-5dv', 'Xpc-6dv', 'Xp-0d', 'Xp-1d', 'Xp-2d', 'Xp-3d', 'Xp-4d', 'Xp-5d', 'Xp-6d', 'Xp-7d', 'AXp-0d', 'AXp-1d', 'AXp-2d', 'Xp-1dv', 'Xp-2dv', 'Xp-3dv', 'Xp-4dv', 'Xp-5dv', 'Xp-6dv', 'Xp-7dv', 'AXp-1dv', 'AXp-2dv', 'SZ', 'Sm', 'Sv', 'Sse', 'Spe', 'Sare', 'Sp', 'Si', 'MZ', 'Mm', 'Mv', 'Mse', 'Mpe', 'Mare', 'Mp', 'Mi', 'SpAbs_D', 'SpMax_D', 'SpDiam_D', 'SpAD_D', 'SpMAD_D', 'LogEE_D', 'VE1_D', 'VE2_D', 'VE3_D', 'VR1_D', 'VR2_D', 'VR3_D', 'NsCH3', 'SsCH3', 'SssCH2', 'SaaCH', 'SsssCH', 'SdssC', 'SaasC', 'SaaaC', 'SssssC', 'SsssN', 'SssssSi', 'SdsssP', 'SssS', 'SaaS', 'SddssS', 'ECIndex', 'ETA_alpha', 'AETA_alpha', 'ETA_shape_p', 'ETA_shape_y', 'ETA_beta', 'AETA_beta', 'ETA_beta_s', 'AETA_beta_s', 'ETA_beta_ns', 'AETA_beta_ns', 'ETA_eta', 'AETA_eta', 'ETA_eta_L', 'AETA_eta_L', 'ETA_epsilon_1', 'ETA_epsilon_2', 'ETA_epsilon_4', 'ETA_epsilon_5', 'ETA_dEpsilon_D', 'ETA_dBeta', 'AETA_dBeta', 'ETA_psi_1', 'fragCpx', 'IC0', 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'TIC0', 'TIC1', 'TIC2', 'TIC3', 'TIC4', 'TIC5', 'SIC0', 'SIC1', 'SIC2', 'SIC3', 'SIC4', 'SIC5', 'BIC0', 'BIC1', 'BIC2', 'BIC3', 'BIC4', 'BIC5', 'CIC0', 'CIC1', 'CIC2', 'CIC3', 'CIC4', 'CIC5', 'MIC0', 'MIC1', 'MIC2', 'MIC3', 'MIC4', 'MIC5', 'ZMIC0', 'ZMIC1', 'ZMIC2', 'ZMIC3', 'ZMIC4', 'ZMIC5', 'Kier1', 'Kier2', 'FilterItLogS', 'VMcGowan', 'LabuteASA', 'PEOE_VSA6', 'PEOE_VSA7', 'SMR_VSA5', 'SlogP_VSA2', 'SlogP_VSA5', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'MDEC-11', 'MID', 'AMID', 'MID_h', 'AMID_h', 'MID_C', 'AMID_C', 'MPC2', 'MPC3', 'MPC4', 'MPC5', 'MPC6', 'MPC7', 'TMPC10', 'piPC1', 'piPC2', 'piPC3', 'piPC4', 'piPC5', 'piPC6', 'piPC7', 'TpiPC10', 'apol', 'bpol', 'nRot', 'RotRatio', 'SLogP', 'SMR', 'GGI1', 'GGI2', 'GGI3', 'GGI4', 'GGI5', 'JGI1', 'JGI2', 'JGI3', 'JGI4', 'JGI5', 'JGT10', 'Diameter', 'Radius', 'TopoShapeIndex', 'PetitjeanIndex', 'VAdjMat', 'MWC01', 'MWC02', 'MWC03', 'MWC04', 'MWC05', 'MWC06', 'MWC07', 'MWC08', 'MWC09', 'MWC10', 'TMWC10', 'SRW02', 'SRW04', 'SRW06', 'SRW08', 'SRW10', 'TSRW10', 'MW', 'AMW', 'WPath', 'WPol', 'Zagreb1', 'Zagreb2', 'mZagreb1', 'mZagreb2']
MORDRED = Calculator([d for d in calc.descriptors if str(d) in DESCRIPTOR_COL], ignore_3D=True)

N_BITS = 167
FINGERPRINT_COLS = [f'fp_{i}' for i in range(N_BITS)]

TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

def smile_to_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)

def gen_train_fpMaccs_descriptorMordred_file (train_file:Path, test_file:Path):
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
        subset = df[['id', 'SMILES'] + [target] + FINGERPRINT_COLS + DESCRIPTOR_COL].copy()
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
        df_descriptor = df_target_copy[DESCRIPTOR_COL]

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