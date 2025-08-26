import argparse
import pickle  
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import mean_absolute_error
import os
import joblib
from src.utils.logging import lprint as log_print, LoggingLevels as ll
from src.models.fusion import Config, main_regression, EnsembleTrainer, tune_hyperparameters, save_model, load_model, predict_with_model


# -----------------------------
# Main Function
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train model, tune hyperparameters, and save the best model")
    parser.add_argument('--input-train', required=True, metavar='FILE',
                        help='Base name for input files (without extensions). Expects .dt, .fm, .fmap4, .tt files.')
    parser.add_argument('--out', required=True, help='Output directory to save best model files')
    args = parser.parse_args()

    # Define input file paths based on base name
    descriptors_file = args.input_train + '.dt'
    morgan_file = args.input_train + '.fm'
    map4_file = args.input_train + '.fmap4'
    targets_file = args.input_train + '.tt'

    # Verify that input files exist
    for f in [descriptors_file, morgan_file, map4_file, targets_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Input file {f} does not exist")

    # Load data (aligned with main_regression)
    log_print(ll.INFO, "Loading data files...")
    df_desc = pd.read_csv(descriptors_file, sep="\t", index_col=0)
    df_morgan = pd.read_csv(morgan_file, sep="\t", index_col=0)
    df_map4 = pd.read_csv(map4_file, sep="\t", index_col=0)
    df_targets = pd.read_csv(targets_file, sep="\t", index_col=0)

    # Ensure consistent indices across all dataframes
    ids = df_desc.index.intersection(df_morgan.index).intersection(df_map4.index).intersection(df_targets.index)
    if len(ids) == 0:
        raise ValueError("No common molecule IDs found across input files")
    df_desc = df_desc.loc[ids]
    df_morgan = df_morgan.loc[ids]
    df_map4 = df_map4.loc[ids]
    df_targets = df_targets.loc[ids]

    # Preprocess data (aligned with main_regression for binary MAP4)
    log_print(ll.INFO, "Preprocessing data...")
    df_targets = df_targets.replace(-1, np.nan)
    df_targets = df_targets.fillna(df_targets.median())
    df_map4 = df_map4.astype(float)
    df_morgan = df_morgan.astype(float)

    # Normalize features for hyperparameter tuning
    scaler_desc = StandardScaler()
    scaler_tab = StandardScaler(with_mean=False)  # No mean centering for sparse data
    X_desc = scaler_desc.fit_transform(df_desc.values)
    X_fp = sp.csr_matrix((df_map4 > 0).astype(float).values)  # Treat MAP4 as binary
    X_tab = scaler_tab.fit_transform(sp.csr_matrix((df_morgan > 0).astype(float).values))
    target_names = df_targets.columns.tolist()
    y_dict = {t: df_targets[t].values for t in target_names}

    # Perform hyperparameter tuning
    log_print(ll.INFO, "Starting hyperparameter tuning...")
    tuning_dir = os.path.join(args.out, "tuning")
    best_config, best_mae = tune_hyperparameters(X_desc, X_fp, X_tab, y_dict, target_names, output_dir=tuning_dir)
    log_print(ll.SUCCESS, f"Best config found with MAE: {best_mae:.2f}")

    # Save best config using pickle
    config_save_path = os.path.join(args.out, "best_config.pkl")
    with open(config_save_path, "wb") as f:
        pickle.dump(best_config, f)
    log_print(ll.INFO, f"Best config saved to {config_save_path}")

    # Train model with best config
    log_print(ll.INFO, "Training model with best config...")
    trainer = main_regression(
        descriptor_file=descriptors_file,
        map4_file=map4_file,
        morgan_file=morgan_file,
        targets_file=targets_file,
        config=best_config
    )

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Save best model using save_model
    log_print(ll.INFO, f"Saving best model to {args.out}...")
    model_save_path = os.path.join(args.out, 'model.pkl')
    save_model(trainer, model_save_path)

    # Save scalers for inference (reuse trainer.scalers for consistency)
    joblib.dump(trainer.scalers['desc'], os.path.join(args.out, 'scaler_desc.joblib'))
    joblib.dump(trainer.scalers['fp'], os.path.join(args.out, 'scaler_map4.joblib'))
    joblib.dump(trainer.scalers['tab'], os.path.join(args.out, 'scaler_morgan.joblib'))

    # Verify saved model by loading and predicting on a subset of training data
    log_print(ll.INFO, "Verifying saved model...")
    subset_size = min(10, len(ids))  # Use up to 10 samples for verification
    subset_ids = ids[:subset_size]
    df_desc_subset = df_desc.loc[subset_ids]
    df_map4_subset = df_map4.loc[subset_ids]
    df_morgan_subset = df_morgan.loc[subset_ids]
    df_targets_subset = df_targets.loc[subset_ids]

    # Save subset data to temporary files
    temp_desc_file = os.path.join(args.out, "temp_desc_subset.tsv")
    temp_map4_file = os.path.join(args.out, "temp_map4_subset.tsv")
    temp_morgan_file = os.path.join(args.out, "temp_morgan_subset.tsv")
    df_desc_subset.to_csv(temp_desc_file, sep="\t", index=True)
    df_map4_subset.to_csv(temp_map4_file, sep="\t", index=True)
    df_morgan_subset.to_csv(temp_morgan_file, sep="\t", index=True)

    # Load model and predict
    loaded_trainer = load_model(model_save_path)
    for target in target_names:
        # Predict with original trainer
        X_desc_subset = trainer.scalers['desc'].transform(df_desc_subset.values)
        X_fp_subset = sp.csr_matrix((df_map4_subset > 0).astype(float).values)
        X_tab_subset = trainer.scalers['tab'].transform(sp.csr_matrix((df_morgan_subset > 0).astype(float).values))
        y_pred_before = trainer.predict(X_desc_subset, X_fp_subset, X_tab_subset, target)
        y_true = df_targets_subset[target].values
        mae_before = mean_absolute_error(y_true, y_pred_before)

        # Predict with loaded model
        y_pred_loaded = predict_with_model(
            trainer=loaded_trainer,
            descriptor_file=temp_desc_file,
            map4_file=temp_map4_file,
            morgan_file=temp_morgan_file,
            target_name=target
        )
        mae_loaded = mean_absolute_error(y_true, y_pred_loaded)

        log_print(ll.SUCCESS, f"Verification for {target} - MAE before save: {mae_before:.2f}, MAE after load: {mae_loaded:.2f}")

        # Check prediction consistency
        np.testing.assert_array_almost_equal(
            y_pred_before, y_pred_loaded, decimal=4,
            err_msg=f"Predictions for {target} differ before and after loading"
        )
        log_print(ll.SUCCESS, f"Predictions for {target} are consistent before and after loading")

    # Clean up temporary files
    for f in [temp_desc_file, temp_map4_file, temp_morgan_file]:
        if os.path.exists(f):
            os.remove(f)

    log_print(ll.SUCCESS, f"Best model and scalers saved to {args.out}. Model verification passed.")

if __name__ == "__main__":
    main()