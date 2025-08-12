import sys, os
from pathlib import Path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

supp_Tc = './data/supp_Tc.csv'
df_Tc = pd.read_csv(supp_Tc)
df_Tc.insert(0, 'id', [f'tc{i}' for i in range(len(df_Tc))])

supp_Tg = './data/supp_Tg.csv'
df_Tg = pd.read_csv(supp_Tg).reset_index(drop=True)
df_Tg.insert(0, 'id', [f'tg{i}' for i in range(len(df_Tg))])

supp_FFV = './data/supp_FFV.csv'
df_FFV = pd.read_csv(supp_FFV).reset_index(drop=True)
df_FFV.insert(0, 'id', [f'ffv{i}' for i in range(len(df_FFV))])

df_train = pd.read_csv('./data/train.csv')


def first_non_null(s: pd.Series):
    return s.dropna().iloc[0] if s.notna().any() else pd.NA

cols_base = list(df_train.columns)
extra_cols = ['id', 'Tc', 'Tg', 'FFV']
all_cols = cols_base.copy()
for c in extra_cols:
    if c not in all_cols:
        all_cols.append(c)

f_train = df_train.drop_duplicates(subset='SMILES', keep='first').copy()

f_tc = df_Tc[['SMILES', 'id', 'Tc']].drop_duplicates(subset='SMILES', keep='first').copy()
f_tg = df_Tg[['SMILES', 'id', 'Tg']].drop_duplicates(subset='SMILES', keep='first').copy()
f_ffv = df_FFV[['SMILES', 'id', 'FFV']].drop_duplicates(subset='SMILES', keep='first').copy()

frames = [f_train, f_tc, f_tg, f_ffv]

aligned = []
for f in frames:
    for col in all_cols:
        if col not in f.columns:
            f[col] = pd.NA
    aligned.append(f[['SMILES'] + [c for c in all_cols if c != 'SMILES']])

stack = pd.concat(aligned, ignore_index=True)

agg_map = {col: first_non_null for col in stack.columns if col != 'SMILES'}
mega_train = (
    stack
    .groupby('SMILES', as_index=False)
    .agg(agg_map)
)

if 'id' in mega_train.columns:
    mega_train['id'] = mega_train['id'].astype('string')
    mega_train = mega_train[['id'] + [c for c in mega_train.columns if c != 'id']]

print(mega_train)
mega_train.to_csv('./data/mega_train.csv', index=False, sep='\t')