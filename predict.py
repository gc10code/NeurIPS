import argparse
import json
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import joblib
from src.utils.logging import lprint as log_print, LoggingLevels as ll
from src.models.fusion import Config, EnsembleTrainer, RFComponent, TabularMLP


def load_config(config_path: str) -> Config:
    """Load configuration from a JSON file and convert to Config object."""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        # Convert brain_hidden_dims string (e.g., '64_32') to tuple
        if 'brain_hidden_dims' in config_dict and isinstance(config_dict['brain_hidden_dims'], str):
            config_dict['brain_hidden_dims'] = tuple(map(int, config_dict['brain_hidden_dims'].split('_')))
        return Config(**config_dict)
    except Exception as e:
        log_print(ll.ERROR, f"Failed to load config from {config_path}: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Predict using trained model components")
    parser.add_argument('--input-test', required=True, metavar='FILE',
                        help='Base name for input test files (without extensions). Expects .desc, .fm, .fmap4 files.')
    parser.add_argument('--model-dir', required=True, help='Directory containing trained model components and config')
    parser.add_argument('--target', required=True, help='Target name for prediction (e.g., target_0)')
    parser.add_argument('--out', required=True, help='Output directory to save predictions')
    args = parser.parse_args()

    # Define input file paths based on base name
    descriptors_file = args.input_test + '.desc'
    morgan_file = args.input_test + '.fm'
    map4_file = args.input_test + '.fmap4'

    # Verify that input files exist
    for f in [descriptors_file, morgan_file, map4_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Input file {f} does not exist")

    # Verify that model directory and required files exist
    required_model_files = [
        os.path.join(args.model_dir, 'best_config.json'),
        os.path.join(args.model_dir, 'mlp_models.pth'),
        os.path.join(args.model_dir, f'rf_desc_{args.target}.joblib'),
        os.path.join(args.model_dir, f'rf_fp_{args.target}.joblib'),
        os.path.join(args.model_dir, 'scaler_desc.joblib'),
        os.path.join(args.model_dir, 'scaler_map4.joblib'),
        os.path.join(args.model_dir, 'scaler_morgan.joblib')
    ]
    for f in required_model_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Model file {f} does not exist")

    # Load data
    log_print(ll.INFO, "Loading test data files...")
    try:
        df_desc = pd.read_csv(descriptors_file, sep="\t", index_col=0)
        df_morgan = pd.read_csv(morgan_file, sep="\t", index_col=0)
        df_map4 = pd.read_csv(map4_file, sep="\t", index_col=0)
    except Exception as e:
        log_print(ll.ERROR, f"Failed to load test data files: {str(e)}")
        raise

    # Ensure consistent indices across all dataframes
    ids = df_desc.index.intersection(df_morgan.index).intersection(df_map4.index)
    if len(ids) == 0:
        log_print(ll.ERROR, "No common molecule IDs found across test data files")
        raise ValueError("No common molecule IDs found across test data files")
    df_desc = df_desc.loc[ids]
    df_morgan = df_morgan.loc[ids]
    df_map4 = df_map4.loc[ids]
    log_print(ll.INFO, f"Found {len(ids)} common samples after alignment")

    # Preprocess data (aligned with main_regression and main.py)
    log_print(ll.INFO, "Preprocessing test data...")
    try:
        df_map4 = df_map4.astype(float)
        df_morgan = df_morgan.astype(float)
        tab_features = sp.csr_matrix((df_morgan > 0).astype(float).values)  # Sparse matrix for Morgan fingerprints
    except Exception as e:
        log_print(ll.ERROR, f"Test data preprocessing failed: {str(e)}")
        raise

    # Load configuration
    log_print(ll.INFO, f"Loading configuration from {os.path.join(args.model_dir, 'best_config.json')}...")
    try:
        config = load_config(os.path.join(args.model_dir, 'best_config.json'))
    except Exception as e:
        log_print(ll.ERROR, f"Failed to load configuration: {str(e)}")
        raise

    # Initialize EnsembleTrainer
    target_names = [args.target]  # Only the specified target
    n_tab_features = tab_features.shape[1]
    try:
        trainer = EnsembleTrainer(config=config, target_names=target_names, n_tab_features=n_tab_features)
    except Exception as e:
        log_print(ll.ERROR, f"Failed to initialize EnsembleTrainer: {str(e)}")
        raise

    # Load model components
    log_print(ll.INFO, f"Loading model components from {args.model_dir}...")
    try:
        # Load MLP state dictionary
        mlp_state_dicts = torch.load(os.path.join(args.model_dir, 'mlp_models.pth'))
        if args.target not in mlp_state_dicts:
            raise ValueError(f"Target {args.target} not found in MLP state dictionaries")
        mlp = TabularMLP(
            input_dim=2 * config.n_estimators + n_tab_features,
            hidden_dims=config.brain_hidden_dims,
            dropout=config.brain_dropout,
            use_batch_norm=config.use_batch_norm,
            use_residual=config.use_residual
        ).to(config.device)
        mlp.load_state_dict(mlp_state_dicts[args.target])
        trainer.mlps[args.target] = mlp

        # Load RF components
        trainer.components[args.target] = {
            'desc': joblib.load(os.path.join(args.model_dir, f'rf_desc_{args.target}.joblib')),
            'fp': joblib.load(os.path.join(args.model_dir, f'rf_fp_{args.target}.joblib'))
        }

        # Load scalers
        trainer.scalers = {
            'desc': joblib.load(os.path.join(args.model_dir, 'scaler_desc.joblib')),
            'fp': joblib.load(os.path.join(args.model_dir, 'scaler_map4.joblib')),
            'tab': joblib.load(os.path.join(args.model_dir, 'scaler_morgan.joblib'))
        }
    except Exception as e:
        log_print(ll.ERROR, f"Failed to load model components: {str(e)}")
        raise

    # Normalize test features using loaded scalers
    log_print(ll.INFO, "Normalizing test features...")
    try:
        X_desc = trainer.scalers['desc'].transform(df_desc.values)
        X_fp = trainer.scalers['fp'].transform(df_map4.values)
        X_tab = trainer.scalers['tab'].transform(tab_features)
    except Exception as e:
        log_print(ll.ERROR, f"Feature normalization failed: {str(e)}")
        raise

    # Perform predictions
    log_print(ll.INFO, f"Predicting for target {args.target}...")
    try:
        y_pred = trainer.predict(X_desc, X_fp, X_tab, args.target)
        log_print(ll.SUCCESS, f"Predictions completed for {args.target} with {len(y_pred)} samples")
    except Exception as e:
        log_print(ll.ERROR, f"Prediction failed: {str(e)}")
        raise

    # Save predictions
    os.makedirs(args.out, exist_ok=True)
    try:
        pred_df = pd.DataFrame({
            'molecule_id': ids,
            f'predicted_{args.target}': y_pred
        })
        output_file = os.path.join(args.out, f'predictions_{args.target}.csv')
        pred_df.to_csv(output_file, index=False)
        log_print(ll.SUCCESS, f"Predictions saved to {output_file}")
    except Exception as e:
        log_print(ll.ERROR, f"Failed to save predictions: {str(e)}")
        raise

    return y_pred


if __name__ == "__main__":
    main()