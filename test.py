import pandas as pd
import numpy as np
import os

def generate_test_datasets(output_dir="test_data", base_name="test"):
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters
    n_molecules = 100
    n_desc_features = 5
    n_morgan_features = 8
    n_map4_features = 8
    n_targets = 2

    # Molecule IDs
    mol_ids = [f"mol_{i}" for i in range(n_molecules)]

    # Generate descriptors (numerical features)
    desc_data = np.random.normal(loc=0, scale=1, size=(n_molecules, n_desc_features))
    df_desc = pd.DataFrame(
        desc_data,
        index=mol_ids,
        columns=[f"desc_{i}" for i in range(n_desc_features)]
    )

    # Generate Morgan fingerprints (binary)
    morgan_data = np.random.choice([0, 1], size=(n_molecules, n_morgan_features), p=[0.8, 0.2])
    df_morgan = pd.DataFrame(
        morgan_data,
        index=mol_ids,
        columns=[f"morgan_{i}" for i in range(n_morgan_features)]
    )

    # Generate MAP4 fingerprints (binary)
    map4_data = np.random.choice([0, 1], size=(n_molecules, n_map4_features), p=[0.8, 0.2])
    df_map4 = pd.DataFrame(
        map4_data,
        index=mol_ids,
        columns=[f"map4_{i}" for i in range(n_map4_features)]
    )

    # Generate targets (regression values with some missing data)
    target_data = np.random.uniform(low=0, high=1500, size=(n_molecules, n_targets))
    # Introduce some missing values (-1)
    mask = np.random.choice([True, False], size=(n_molecules, n_targets), p=[0.1, 0.9])
    target_data[mask] = -1
    df_targets = pd.DataFrame(
        target_data,
        index=mol_ids,
        columns=[f"target_{i}" for i in range(n_targets)]
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save datasets to TSV files
    df_desc.to_csv(os.path.join(output_dir, f"{base_name}.dt"), sep="\t", index=True)
    df_morgan.to_csv(os.path.join(output_dir, f"{base_name}.fm"), sep="\t", index=True)
    df_map4.to_csv(os.path.join(output_dir, f"{base_name}.fmap4"), sep="\t", index=True)
    df_targets.to_csv(os.path.join(output_dir, f"{base_name}.tt"), sep="\t", index=True)

    print(f"Test datasets generated in {output_dir}/ with base name '{base_name}'")

if __name__ == "__main__":
    generate_test_datasets()