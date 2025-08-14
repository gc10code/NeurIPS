import pandas as pd
from typing import List

from src.preprocessing.file_tree import preprocessing_files, NeurIPSFiles
from src.utils.logging import lprint, LoggingLevels as ll


def preprocessing_main(NeurIPSFiles:NeurIPSFiles, targets: List[str]):
    lprint(ll.INFO, f"Loading NuerIPS Data")
    values, missing_indices, meta, target_series_non_na, df_descriptors = preprocessing_files(NeurIPSFiles, targets)
    for target in targets:
        target_series_non_na[target].to_csv(f"./data/target_{target}.tsv", sep="\t", index=False, header=False)
    df_descriptors.to_csv(f"./data/descriptors.tsv", sep="\t", index=False, header=False)
    meta.to_csv(f"./data/meta.tsv", sep="\t", index=False, header=True)
    lprint(ll.INFO, f"Conversion of NuerIPS Data. Generate training and meta files")

    
    