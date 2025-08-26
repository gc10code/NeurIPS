import os, sys
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed


import re
import pandas as pd
from rdkit import Chem

TARGETS = ["Tg","FFV", "Tc","Density","Rg"] 

@dataclass
class NeurIPSFiles:
    train_supplements: List[Path]
    train_supports: List[Path]
    train: Path
    test: Path

neruips_files = NeurIPSFiles(
    train_supplements= [
        Path("./data/train_supplement/dataset1.csv"),
        Path("./data/train_supplement/dataset3.csv"),
        Path("./data/train_supplement/dataset4.csv")
    ],
    train_supports= [
        Path("./data/train_supports/dataset5.csv"),
        Path("./data/train_supports/dataset6.csv"),
        Path("./data/train_supports/dataset7.csv")
    ],

    train = Path("./data/train.csv"),
    test =  Path("./data/test.csv")
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
                if sc == tc or re.search(rf"(?:^|_| ){re.escape(tc)}(?:_|$| )", sc, flags=re.IGNORECASE):
                    supp_cols[s] = train_cols[t]
                    break
        
        cols_to_add = [col for col in train_cols if col not in supp_cols]
        df_supp.columns = supp_cols

        # iteration to add the cols in support set
        for c in cols_to_add:
            # if it is the ID col
            if bool(re.search(rf"(?:^|_| ){re.escape('id')}(?:_|$| )", c, flags=re.IGNORECASE)):
                for cs in supp_cols:
                    for target in TARGETS:
                        if target==cs:
                            df_supp[c] = [f'{files_names[dfs]}.{dfs}{target}{id}' for id in range (df_supp.shape[0])]
                            continue
            else:
                df_supp[c] = [None for _ in range (df_supp.shape[0])]

        

        # fill the NA values in target col with -1
        for t in TARGETS:
            df_supp[t] = df_supp[t].copy().fillna(-1)

        # change the atom holders (*) with a methyl group
        df_supp['SMILES'] = change_atom_holders(df_supp['SMILES'].copy())

        # eliminate all those smiles with at least an R in it
        df_supp.loc[df_supp["SMILES"].str.contains('R', regex=False, na=False), "SMILES"] = None
        df_supp = df_supp.dropna(subset=['SMILES']).reset_index(drop=True)

        df_support = df_supp[train_cols].copy()

        print (df_support)
        df_supp_list[dfs] = df_support

    return df_supp_list
        


def merge_datasets (files_path: NeurIPSFiles = neruips_files):
    df_train = pd.read_csv(files_path.train, sep=',', header=0)
    if df_train.shape[1]==1:
        print(f"{files_path.train} has not comma formatting. Trying with '\\t'.")
        df_train = pd.read_csv(files_path.train, sep='\t', header=0)
    
    for t in TARGETS:
        df_train[t] = df_train[t].copy().fillna(-1)
    
    # substitute the atom holders with the default molecular group 'C' (methyl)
    df_merge = df_train
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
        df_merge = pd.concat([df_merge.copy(), df_supp], ignore_index=True)
        df_merge = df_merge.copy().drop_duplicates(subset=["SMILES"], keep="first").reset_index(drop=True)

    return df_merge


def _canon_rdkit_smiles(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m, canonical=True) if m is not None else None


def merge_analisys_worker (df_merge:pd.DataFrame, target: str) -> set:
    df_smiles_target = df_merge[['SMILES', target]].copy()
    row_to_drop = set()

    threshold = (df_smiles_target[target].mean() - df_smiles_target[target].min()) / 10

    # convert SMILES in rdkit canonical SMILES
    df_smiles_target['SMILES'] = df_smiles_target['SMILES'].copy().map(_canon_rdkit_smiles)

    # select all those SMILES that are not convertable in canonical
    na_rows = df_smiles_target[df_smiles_target['SMILES'].isna()].index
    row_to_drop.update(na_rows)

    dupl_smile = df_smiles_target.duplicated(subset=['SMILES'], keep='first').to_list()
    dupl_tv = df_smiles_target.duplicated(subset=[target], keep='first').to_list()

    cicles = df_smiles_target.shape[0]
    for i in range(cicles):
        if i%1000==0 and i!=0:
            print (f'{target} i == {i}/{cicles}')

        dup_count = 0
        if dupl_smile[i]:
            if dupl_tv[i] and df_smiles_target.iloc[i][target]!=-1:
                row_to_drop.add(df_smiles_target.index[i]) 
            else:
                for j in range (cicles):
                    thre_bool = df_smiles_target.iloc[i][target] - df_smiles_target.iloc[j][target] < threshold
                    same_smiles = df_smiles_target.iloc[i]['SMILES'] == df_smiles_target.iloc[j]['SMILES']

                    if df_smiles_target.iloc[i][target]!=-1 or df_smiles_target.iloc[j][target]!=-1:
                        if thre_bool and same_smiles and j<i:
                            row_to_drop.add(df_smiles_target.index[i])
                            if dup_count > 0:
                                row_to_drop.add(df_smiles_target.index[j])
                            dup_count += 1

                        elif thre_bool and same_smiles and j>i:
                            row_to_drop.add(df_smiles_target.index[j])

    
    print (f'{target} | row to drop: {len(row_to_drop)}')
    return set(df_smiles_target.index[r] for r in row_to_drop)


def merge_analisys(df_merge: pd.DataFrame):
    # Lancia 5 processi, uno per target
    futures = {}
    with ProcessPoolExecutor(max_workers=5) as ex:
        for t in TARGETS:
            futures[ex.submit(merge_analisys_worker, df_merge[['SMILES', t]], t)] = t

        rows_to_drop_all = set()
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                rows_to_drop = fut.result()
                rows_to_drop_all.update(rows_to_drop)
            except Exception as e:
                print(f"[{t}] failed: {e}")

    print("Total row to drop:", len(rows_to_drop_all))
    # Importante: assegnare il risultato di drop/reset_index
    df_merge = df_merge.drop(index=rows_to_drop_all).reset_index(drop=True)
    return df_merge

def generate_preprocess_files (df_merge: pd.DataFrame):
    df_id_smiles = df_merge[['id', 'SMILES']] 
    df_id_target = df_merge[['id'] + TARGETS]

    df_id_smiles.to_csv('./data/train.ismiles', sep='\t', header=0, index=False)
    print("The file train.ismiles was correctly generated.")
    df_id_target.to_csv('./data/train.tt', sep='\t', header=0, index=False)
    print("The file train.tt was correctly generated.")



def merge_main():
    df_merge = merge_datasets()
    print ('Merge completed.')

    df_merge = df_merge.copy().dropna().reset_index(drop=True)

    df_merge = merge_analisys(df_merge).reset_index(drop=True)
    print ('Merge Analisys completed.')

    df_merge.to_csv('./data/train_merge.csv', sep='\t', index=False)
    generate_preprocess_files(df_merge)
    

if __name__ == "__main__":
    merge_main()