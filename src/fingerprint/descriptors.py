from rdkit import Chem
from rdkit.Chem import MACCSkeys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mordred import Calculator, descriptors


from src.utils.logging import lprint, LoggingLevels as ll

def is_not_numeric(series: pd.Series) -> bool:
    forbidden_chars = set(chr(i) for i in range(58, 126))  # ASCII 58-125
    s = series.dropna().astype(str)
    is_all_numeric = pd.to_numeric(s, errors='coerce').notna().all()
    has_no_forbidden_chars = s.apply(lambda x: all(c not in forbidden_chars for c in x)).all()
    return is_all_numeric and has_no_forbidden_chars


def filter_descriptors(df_descriptor: pd.DataFrame, 
                       feature_reduction:bool = True, 
                       filter_by_loding:bool = True, 
                       filter_variance: float = 0.9,
                       filter_quantile: float = 0.70,
                       filter_correlation_max:float = 0.85,
                       filter_correlation_min:float = 0.70) -> pd.DataFrame:
    lprint(ll.INFO, "Filter: TRUE --> Filtering descriptors")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_descriptor)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    cum_var = pca.explained_variance_ratio_.cumsum()

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index = df_descriptor.columns
    )

    # if feature_selection = True select features with cumulative variance (fileter_variance)    
    n_pc = len(cum_var)
    if feature_reduction:
        n_pc = next(i for i, v in enumerate(cum_var) if v > filter_variance)+1

    selected_descriptor = list()
    for i in range (n_pc):
        loading_ord = loadings.iloc[:, i].abs().sort_values(ascending=False)
        mean_loading = loading_ord.quantile(filter_quantile)
        j=0
        while loading_ord.iloc[j]>mean_loading:
            j+=1
        selected_descriptor += loading_ord[:j].index.tolist()
        
    selected_descriptor = set(selected_descriptor)
    sel_df_descriptor = df_descriptor[list(selected_descriptor)].copy()
    corr_matrix = sel_df_descriptor.corr()
    diz_best_descriptor_load = {}  
    for i in range(n_pc):
        series = loadings.iloc[:, i].abs().fillna(0.0)
        for feat, val in series.items():
            if val > diz_best_descriptor_load.get(feat, float("-inf")):
                diz_best_descriptor_load[feat] = (val)

    best_descriptor_load = pd.Series(diz_best_descriptor_load).sort_values(ascending=False)
    mean_loading = best_descriptor_load.mean()

    descriptor_to_remove = set()
    for i in range (len(corr_matrix.columns)):
        for j in range (i+1, len(corr_matrix.columns)):

            idx_d1 = best_descriptor_load.index.get_loc(corr_matrix.columns[i])
            idx_d2 = best_descriptor_load.index.get_loc(corr_matrix.columns[j])

            if corr_matrix.iloc[i,j] > filter_correlation_max:
                descriptor_to_remove.add(best_descriptor_load.index[max(idx_d1, idx_d2)])
            
            if filter_by_loding and (filter_correlation_min < corr_matrix.iloc[i,j] < filter_correlation_max):
                if min(best_descriptor_load[corr_matrix.columns[i]], best_descriptor_load[corr_matrix.columns[j]]) < mean_loading:
                    descriptor_to_remove.add(best_descriptor_load.index[max(idx_d1, idx_d2)])


    descriptor_to_remove = list(descriptor_to_remove)
    final_descriptors = [col for col in selected_descriptor if col not in descriptor_to_remove]

    return df_descriptor[final_descriptors]


def calculate_descriptors(smiles: pd.Series, filter_descriptors:bool = True)-> pd.DataFrame:
    "calculate pandas dataframe of descriptor by a Series of SMILES"
    lprint(ll.INFO, "Compute Descriptors")
    descriptor_calculator = Calculator(descriptors, ignore_3D=True)
    mol_list = [Chem.MolFromSmiles(s) for s in smiles]
    df_descriptor = descriptor_calculator.pandas(mol_list)

    valid_cols = [col for col in df_descriptor.columns if is_not_numeric(df_descriptor[col])]
    df_descriptor = df_descriptor[valid_cols]
    df_descriptor = df_descriptor.astype({col: np.float64 for col in df_descriptor.select_dtypes(include=['float32', 'float']).columns})

    col_to_remove = []
    for des in df_descriptor.columns:
        col_data = df_descriptor[des]
        zero_c = (col_data == 0.0).sum()
        if col_data.isnull().any() or (zero_c/len(col_data))>0.049:
            col_to_remove.append(des)
    df_descriptor.drop(columns=col_to_remove, inplace=True)
    
    if filter_descriptors:
        df_descriptor = filter_descriptors(df_descriptor.copy())

    return df_descriptor
