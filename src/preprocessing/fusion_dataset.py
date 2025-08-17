import torch
import numpy as np
import pandas as pd
from pathlib import Path
import gc
from typing import List, Tuple
from src.models.rprop_mlp import RPropMLP
from src.utils.logging import lprint, LoggingLevels as ll
from src.utils.system_utils import setup_device
from src.training.results_manager import ResultsManager

from src._global import global_init

device = setup_device()


def create_fusion_dataset(targets: List[str]) -> Tuple[str, str, str]:
    """
    Create fusion dataset files (fusion_descriptors.tsv, fusion_meta.tsv, fusion_target.tsv)
    for compounds with at least one valid target, imputing missing values using pre-trained models.

    Args:
        targets (List[str]): List of target names.

    Returns:
        Tuple[str, str, str]: Paths to fusion_descriptors.tsv, fusion_meta.tsv, fusion_target.tsv.
    """
    lprint(ll.INFO, "Generating Fusion Dataset")
    try:
        meta_file = Path("./data/meta.tsv")
        descriptor_file = Path("./data/descriptors.tsv")
        target_files = [Path(f"./data/target_{target}.tsv") for target in targets]

        # Verify file existence
        for file in [meta_file, descriptor_file] + target_files:
            if not file.exists():
                lprint(ll.ERROR, f"File not found: {file}")
                raise FileNotFoundError(f"File not found: {file}")

        # Read meta.tsv
        lprint(ll.DEBUG, f"Reading meta file: {meta_file}")
        meta_df = pd.read_csv(meta_file, sep='\t')
        required_columns = ['id'] + targets
        if not all(col in meta_df.columns for col in required_columns):
            lprint(ll.ERROR, f"meta.tsv missing required columns: {required_columns}")
            raise ValueError("meta.tsv missing required columns")
        lprint(ll.DEBUG, f"Loaded meta.tsv: shape={meta_df.shape}")

        # Find compounds with at least one valid target
        common_mask = (meta_df[targets] >= 0).any(axis=1)
        common_df = meta_df[common_mask][required_columns]
        if common_df.empty:
            lprint(ll.ERROR, "No compounds with valid data for any target")
            raise ValueError("No valid compounds found")
        lprint(ll.DEBUG, f"Found {len(common_df)} compounds with at least one valid target")

        # Create validity mask (1 for real, 0 for imputed)
        validity_mask = (common_df[targets] >= 0).astype(int)
        validity_mask.columns = [f"{t}_valid" for t in targets]
        common_df = pd.concat([common_df, validity_mask], axis=1)
        lprint(ll.DEBUG, f"common_df shape with validity mask: {common_df.shape}")

        # Define common_indices
        common_indices = meta_df[common_mask].index
        lprint(ll.DEBUG, f"Common indices length: {len(common_indices)}")

        # Read descriptors (with header, includes 'id' column)
        lprint(ll.DEBUG, f"Reading descriptor file: {descriptor_file}")
        descriptors = pd.read_csv(descriptor_file, sep='\t', header=0)
        lprint(ll.DEBUG, f"Loaded descriptors: shape={descriptors.shape}")

        # Verify descriptor shape and presence of 'id' column
        if 'id' not in descriptors.columns:
            lprint(ll.ERROR, "descriptors.tsv missing 'id' column")
            raise ValueError("descriptors.tsv missing 'id' column")
        if descriptors.shape[0] != meta_df.shape[0]:
            lprint(ll.ERROR, f"Descriptor rows ({descriptors.shape[0]}) mismatch meta rows ({meta_df.shape[0]})")
            raise ValueError("Descriptor-meta row count mismatch")

        # Verify ID alignment
        if not meta_df['id'].equals(descriptors['id']):
            lprint(ll.WARN, "ID mismatch between meta_df and descriptors. Attempting to align by 'id'.")
            descriptors = descriptors.set_index('id').reindex(meta_df['id']).reset_index()
            if descriptors['id'].isna().any():
                lprint(ll.ERROR, "Failed to align descriptors with meta_df by 'id'")
                raise ValueError("Descriptor-meta ID alignment failed")

        # Select numeric features (exclude 'id')
        descriptor_columns = [col for col in descriptors.columns if col != 'id']
        fusion_descriptors_numeric = descriptors[descriptor_columns].iloc[common_indices]
        lprint(ll.INFO, f"Fusion descriptors (numeric) shape: {fusion_descriptors_numeric.shape}")

        # Verify numeric features
        if not fusion_descriptors_numeric.select_dtypes(include=np.number).columns.equals(fusion_descriptors_numeric.columns):
            lprint(ll.ERROR, "Non-numeric columns detected in descriptors")
            raise ValueError("Non-numeric columns detected in descriptors")

        # Create fusion_descriptors with 'id' column for saving
        fusion_descriptors = pd.DataFrame(fusion_descriptors_numeric.values, columns=descriptor_columns)
        fusion_descriptors.insert(0, 'id', common_df['id'].values)
        lprint(ll.DEBUG, f"Fusion descriptors with 'id' column: shape={fusion_descriptors.shape}")

        # Load pre-trained models and verify input size
        teacher_models = []
        expected_feature_count = len(descriptor_columns)  # Number of features excluding 'id'
        if fusion_descriptors_numeric.shape[1] != expected_feature_count:
            lprint(ll.ERROR, f"Descriptor feature count mismatch: expected {expected_feature_count}, got {fusion_descriptors_numeric.shape[1]}")
            raise ValueError(f"Descriptor feature count mismatch: expected {expected_feature_count}, got {fusion_descriptors_numeric.shape[1]}")
        
        for target in targets:
            model_path = Path(f"./output/{target}/best_model.pth")
            lprint(ll.DEBUG, f"Loading pre-trained model for {target}: {model_path}")
            if not model_path.exists():
                lprint(ll.ERROR, f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model = ResultsManager.load_best_model(model_path=model_path)
            model.to(device).eval()
            # Verify input size compatibility
            if model.input_size != expected_feature_count:
                lprint(ll.ERROR, f"Input size mismatch for {target}: model expects {model.input_size} features, got {expected_feature_count}")
                raise ValueError(f"Input size mismatch for {target}: expected {model.input_size}, got {expected_feature_count}")
            teacher_models.append(model)
        lprint(ll.INFO, f"Loaded {len(teacher_models)} pre-trained models")

        # Read and align target values, impute missing values
        fusion_targets = []
        descriptors_tensor = torch.tensor(fusion_descriptors_numeric.values, dtype=torch.float32).to(device)
        lprint(ll.DEBUG, f"Descriptors tensor shape: {descriptors_tensor.shape}")
        for target, target_file, model in zip(targets, target_files, teacher_models):
            lprint(ll.INFO, f"Processing target {target}: {target_file}")
            try:
                # Read target file with header
                target_data = pd.read_csv(target_file, sep='\t', header=0)
                lprint(ll.DEBUG, f"Raw target data shape for {target}: {target_data.shape}")
                
                # Verify expected columns (id, target)
                expected_columns = ['id', target]
                if not all(col in target_data.columns for col in expected_columns):
                    lprint(ll.ERROR, f"Target file {target_file} missing expected columns: {expected_columns}")
                    raise ValueError(f"Target file {target_file} missing expected columns")
                
                # Align target data with common_df using 'id'
                target_data = target_data.set_index('id').reindex(common_df['id']).reset_index()
                target_values = np.full((len(common_df), 1), np.nan, dtype=float)  # Initialize with NaN
                valid_mask = ~target_data[target].isna()
                if valid_mask.any():
                    target_values[valid_mask] = target_data.loc[valid_mask, target].values[:, None]
                    lprint(ll.DEBUG, f"Aligned {target}: {np.sum(valid_mask)} real values")
                else:
                    lprint(ll.WARN, f"No valid real values for target {target}")
            except Exception as e:
                lprint(ll.ERROR, f"Failed to read or process target file {target_file}: {str(e)}")
                raise

            # Impute missing values
            invalid_indices = ~valid_mask
            if invalid_indices.any():
                lprint(ll.DEBUG, f"Imputing {np.sum(invalid_indices)} missing values for {target}")
                with torch.no_grad():
                    model.eval()
                    imputed = model(descriptors_tensor[invalid_indices]).cpu().numpy()
                target_values[invalid_indices] = imputed
            fusion_targets.append(target_values)
            lprint(ll.DEBUG, f"Target {target} shape: {target_values.shape}")

        # Combine target values into a single array [N_common, T]
        fusion_targets = np.hstack(fusion_targets)  # [N_common, T]
        lprint(ll.DEBUG, f"Fusion targets shape: {fusion_targets.shape}")

        # Verify consistent number of samples across files
        num_samples = len(common_df)
        if fusion_descriptors.shape[0] != num_samples or fusion_targets.shape[0] != num_samples:
            lprint(ll.ERROR, f"Sample count mismatch: fusion_descriptors={fusion_descriptors.shape[0]}, fusion_targets={fusion_targets.shape[0]}, common_df={num_samples}")
            raise ValueError("Sample count mismatch between fusion_descriptors, fusion_targets, and common_df")

        # Save fusion_descriptors.tsv (with 'id' column)
        fusion_descriptors_file = Path("./data/fusion_descriptors.tsv")
        fusion_descriptors.to_csv(fusion_descriptors_file, sep='\t', index=False)
        lprint(ll.SUCCESS, f"Saved fusion_descriptors.tsv: {fusion_descriptors_file}")

        # Save fusion_meta.tsv (includes validity mask)
        fusion_meta_file = Path("./data/fusion_meta.tsv")
        common_df.to_csv(fusion_meta_file, sep='\t', index=False)
        lprint(ll.SUCCESS, f"Saved fusion_meta.tsv: {fusion_meta_file}")

        # Save fusion_target.tsv
        fusion_target_file = Path("./data/fusion_target.tsv")
        target_df = pd.DataFrame(fusion_targets, columns=targets)
        target_df.insert(0, 'id', common_df['id'].values)
        target_df.to_csv(fusion_target_file, sep='\t', index=False)
        lprint(ll.SUCCESS, f"Saved fusion_target.tsv: {fusion_target_file}")

        return str(fusion_descriptors_file), str(fusion_meta_file), str(fusion_target_file)

    except FileNotFoundError as e:
        lprint(ll.ERROR, f"File not found: {str(e)}")
        raise
    except Exception as e:
        lprint(ll.ERROR, f"Error creating fusion dataset: {str(e)}")
        raise
    finally:
        lprint(ll.INFO, "Cleaning up resources in create_fusion_dataset")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def load_fusion_dataset(targets: List[str]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Load fusion dataset from TSV files and create X_fusion, y_fusion, and valid_mask.

    Args:
        targets (List[str]): List of target names (e.g., ["Density", "FFV", "Rg", "Tc", "Tg"]).

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
            - X_fusion: Tensor of shape (N, D) containing descriptors.
            - y_fusion: List of tensors, each of shape (N, 1), containing target values.
            - valid_mask: List of boolean tensors, each of shape (N,), indicating real values.
    Raises:
        ValueError: If files are missing, columns are invalid, or data shapes mismatch.
    """
    try:
        lprint(ll.INFO, "Loading fusion dataset")
        fusion_descriptors_file = Path("./data/fusion_descriptors.tsv") 
        fusion_target_file = Path("./data/fusion_target.tsv")
        fusion_meta_file = Path("./data/fusion_meta.tsv")

        # Validate inputs
        if not all(Path(f).exists() for f in [fusion_descriptors_file, fusion_target_file, fusion_meta_file]):
            missing = [f for f in [fusion_descriptors_file, fusion_target_file, fusion_meta_file] if not Path(f).exists()]
            lprint(ll.ERROR, f"Missing files: {missing}")
            raise ValueError(f"Missing files: {missing}")

        # Read descriptors
        descriptors_df = pd.read_csv(fusion_descriptors_file, sep='\t')
        if 'id' not in descriptors_df.columns:
            lprint(ll.ERROR, "fusion_descriptors.tsv missing 'id' column")
            raise ValueError("fusion_descriptors.tsv missing 'id' column")
        lprint(ll.DEBUG, f"Loaded fusion_descriptors.tsv with shape {descriptors_df.shape}")

        # Read targets
        target_df = pd.read_csv(fusion_target_file, sep='\t')
        if 'id' not in target_df.columns:
            lprint(ll.ERROR, "fusion_target.tsv missing 'id' column")
            raise ValueError("fusion_target.tsv missing 'id' column")
        missing_targets = [t for t in targets if t not in target_df.columns]
        if missing_targets:
            lprint(ll.ERROR, f"Missing target columns in fusion_target.tsv: {missing_targets}")
            raise ValueError(f"Missing target columns: {missing_targets}")
        lprint(ll.DEBUG, f"Loaded fusion_target.tsv with shape {target_df.shape}")

        # Read meta (for validity masks)
        meta_df = pd.read_csv(fusion_meta_file, sep='\t')
        if 'id' not in meta_df.columns:
            lprint(ll.ERROR, "fusion_meta.tsv missing 'id' column")
            raise ValueError("fusion_meta.tsv missing 'id' column")
        missing_valid_columns = [f"{t}_valid" for t in targets if f"{t}_valid" not in meta_df.columns]
        if missing_valid_columns:
            lprint(ll.ERROR, f"Missing validity columns in fusion_meta.tsv: {missing_valid_columns}")
            raise ValueError(f"Missing validity columns: {missing_valid_columns}")
        lprint(ll.DEBUG, f"Loaded fusion_meta.tsv with shape {meta_df.shape}")

        # Verify ID consistency
        descriptors_ids = set(descriptors_df['id'])
        target_ids = set(target_df['id'])
        meta_ids = set(meta_df['id'])
        if not (descriptors_ids == target_ids == meta_ids):
            lprint(ll.ERROR, "ID mismatch between fusion_descriptors.tsv, fusion_target.tsv, and fusion_meta.tsv")
            raise ValueError("ID mismatch between files")

        # Sort by ID to ensure alignment
        descriptors_df = descriptors_df.sort_values('id').reset_index(drop=True)
        target_df = target_df.sort_values('id').reset_index(drop=True)
        meta_df = meta_df.sort_values('id').reset_index(drop=True)

        # Verify sample count
        num_samples = descriptors_df.shape[0]
        if target_df.shape[0] != num_samples or meta_df.shape[0] != num_samples:
            lprint(ll.ERROR, f"Sample count mismatch: descriptors={num_samples}, target={target_df.shape[0]}, meta={meta_df.shape[0]}")
            raise ValueError("Sample count mismatch between files")

        # Create X_fusion
        X_fusion = torch.tensor(descriptors_df.drop(columns=['id']).values, dtype=torch.float32)
        lprint(ll.DEBUG, f"X_fusion shape: {X_fusion.shape}")

        # Create y_fusion
        y_fusion = [torch.tensor(target_df[target].values[:, None], dtype=torch.float32).clone().detach() for target in targets]
        lprint(ll.DEBUG, f"y_fusion shapes: {[y.shape for y in y_fusion]}")

        # Create valid_mask
        valid_mask = [torch.tensor(meta_df[f"{target}_valid"].values, dtype=torch.bool).clone().detach() for target in targets]
        lprint(ll.DEBUG, f"valid_mask shapes: {[m.shape for m in valid_mask]}")

        # Validate shapes
        num_samples = X_fusion.shape[0]
        if any(y.shape[0] != num_samples for y in y_fusion) or any(m.shape[0] != num_samples for m in valid_mask):
            lprint(ll.ERROR, f"Shape mismatch between X_fusion ({num_samples}), y_fusion {[y.shape[0] for y in y_fusion]}, and valid_mask {[m.shape[0] for m in valid_mask]}")
            raise ValueError("Shape mismatch between X_fusion, y_fusion, and valid_mask")

        # Log valid data statistics
        for target, mask in zip(targets, valid_mask):
            valid_count = mask.sum().item()
            lprint(ll.INFO, f"{target}: {valid_count} real values, {num_samples - valid_count} imputed")

        lprint(ll.INFO, "Fusion dataset loaded successfully")
        return X_fusion, y_fusion, valid_mask

    except Exception as e:
        lprint(ll.ERROR, f"Error loading fusion dataset: {str(e)}")
        raise