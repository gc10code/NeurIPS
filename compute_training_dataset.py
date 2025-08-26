import os, sys
from pathlib import Path
from typing import List, Dict, Union, Tuple
from dataclasses import dataclass


import re 
import pandas as pd
from rdkit import Chem

TARGETS = ["Tc","FFV", "Tg","Density","Rg"] 

@dataclass
class NeurIPSFiles:
    train_supplements: List[Path]
    train_supports: List[Path]
    train: Path
    test: Path

neruips_files = NeurIPSFiles(
    train_supplements= [
        Path("./data/neurips-open-polymer-prediction-2025/train_supplement/dataset1.csv"),
        Path("./data/neurips-open-polymer-prediction-2025/train_supplement/dataset3.csv"),
        Path("./data/neurips-open-polymer-prediction-2025/train_supplement/dataset4.csv")
    ],
    train_supports= [
        Path("./data/neurips-open-polymer-prediction-2025/train_supports/dataset5.csv"),
        Path("./data/neurips-open-polymer-prediction-2025/train_supports/dataset6.csv"),
        Path("./data/neurips-open-polymer-prediction-2025/train_supports/dataset7.csv")
    ],

    train = Path("./data/neurips-open-polymer-prediction-2025/train.csv"),
    test =  Path("./data/neurips-open-polymer-prediction-2025/test.csv")
)


def change_atom_holders(smiles:pd.Series, atom_holders = '*', substitution_group = 'C') -> pd.Series:
    """
        Take in input a series that contain a the SMILES
        and substituites the atom holders ('*' or another char that you can set)
        with an functional group or another molecule that you desire to insert.
    """
    return smiles.str.replace(atom_holders, substitution_group)


def supp_file_standardizer(df_train: pd.DataFrame, df_supp_list: List[pd.DataFrame], files_names: List[str]) -> pd.Series:
    """
    Function that standardize the format of supplemetary/support dataset with the main train set.
    """

    # iteraction on support dataset 
    for dfs in range (len(df_supp_list)):
        train_cols = df_train.columns.to_list()
        df_supp = df_supp_list[dfs]
        supp_cols = df_supp.columns.to_list()

        # verify if the train and support cols are the same:
        #   - if the name of the support column is slightly different, it takes the same name as the train column
        for s in range (len(supp_cols)):
            for t in range (len(train_cols)):
                sc = supp_cols[s]
                tc = train_cols[t]
                if sc == tc or re.search(rf"(?:^|| ){re.escape(tc)}(?:|$| )", sc, flags=re.IGNORECASE):
                    supp_cols[s] = train_cols[t]
                    break
        
        cols_to_add = [col for col in train_cols if col not in supp_cols]
        df_supp.columns = supp_cols

        # iteration to add the cols in support set
        for c in cols_to_add:
            # if it is the ID col
            if bool(re.search(rf"(?:^|| ){re.escape('id')}(?:|$| )", c, flags=re.IGNORECASE)):
                for cs in supp_cols:
                    for target in TARGETS:
                        if target==cs:
                            df_supp[c] = [f'{files_names[dfs]}.{dfs}{target}{id}' for id in range (df_supp.shape[0])]
                            continue
            df_supp[c] = [None for _ in range (df_supp.shape[0])]

        df_support = df_supp[train_cols].copy()

        # fill the NA values in target col with -1
        for t in TARGETS:
            df_support[t] = df_support[t].copy().fillna(-1)

        # change the atom holders (*) with a methyl group
        df_support['SMILES'] = change_atom_holders(df_support['SMILES'].copy())

        # eliminate all those smiles with at least an R in it
        df_support.loc[df_support["SMILES"].str.contains('R', regex=False, na=False), "SMILES"] = None
        df_support = df_support.dropna(subset=['SMILES']).reset_index(drop=True)

        print (df_support)

        df_supp_list[dfs] = df_support
    
    return df_supp_list
        


def merge_datasets (files_path: NeurIPSFiles = neruips_files):
    df_train = pd.read_csv(files_path.train, sep=',', header=0)
    if df_train.shape[1]==1:
        print(f"{files_path.train} has not comma formatting. Trying with '\\t'.")
        df_train = pd.read_csv(files_path.train, sep='\t', header=0)
    
    #for t in TARGETS:
        #df_train[t] = df_train[t].copy().fillna(-1)
    
    # substitute the atom holders with the default molecular group 'C' (methyl)
    df_merge = df_train
    smiles = df_merge['SMILES']
    for s in range(smiles.shape[0]):
        df_merge['SMILES'] = change_atom_holders(df_merge['SMILES'].copy())

    df_supp_list = []
    for sup in (files_path.train_supplements + files_path.train_supports):
        print (sup)
        try:
            df_supp = pd.read_csv(sup, sep=',')
            if df_supp.shape[1]==1:
                print(f"{sup} has not comma formatting. Trying with '\\t'.")
                df_supp = pd.read_csv(sup, sep='\t')
    
            df_supp_list.append(df_supp)

        except Exception as e:
            print(f"{e}")
            raise
    
    files_names = []
    for p in (files_path.train_supplements + files_path.train_supports):
        files_names.append(os.path.splitext(os.path.basename(p))[0])
    
    print (f'files names: {files_names}')
    
    df_supp_list = supp_file_standardizer(df_merge, df_supp_list, files_names)

    for df_supp in df_supp_list:
        df_merge = pd.concat([df_merge, df_supp], ignore_index=True)
        df_merge = df_merge.drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)

    return df_merge

    

def merge_analisys (df_merge:pd.DataFrame):
    for t in TARGETS:
        smiles_target = df_merge[['SMILES', t]].copy().dropna()
        n_rows = len(smiles_target)
        row_to_drop = set()

        threshold = smiles_target[t].mean() - smiles_target[t].min()
        for i in range(n_rows-1):
            if i%100==0 and i>0:
                print (f'i == {i}/{n_rows}')
            if i in row_to_drop:
                continue
            for j in range (i+1, n_rows):
                if j in row_to_drop:
                    continue
                t_vi = smiles_target.iloc[i][t] # target value at i-row
                t_vj = smiles_target.iloc[j][t] # target value at j-row

                if abs(t_vi - t_vj) < threshold:
                    # verify if the smiles represents the same molecules
                    if Chem.MolToSmiles(Chem.MolFromSmiles(smiles_target.iloc[i]['SMILES']), canonical=True) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles_target.iloc[j]['SMILES']), canonical=True):
                        row_to_drop.add(j)
        
        print (t, len(row_to_drop))
        df_merge.drop(row_to_drop).reset_index(drop=True)
            
    return df_merge


def generate_preprocess_files (df_merge: pd.DataFrame):
    df_id_smiles = df_merge['id', 'SMILES']
    df_id_target = df_merge[['id'] + TARGETS]

    df_id_smiles.to_csv('id_smiles.tsv', sep='\t', header=0)
    df_id_target.to_csv('id_target.tsv', sep='\t', header=0)



def merge_main():
    df_merge = merge_analisys(merge_datasets())
    df_merge.to_csv('./data/train_merge.csv', sep='\t')
    generate_preprocess_files(df_merge)
    print("finito")


if __name__ == "__main__":
    merge_main()