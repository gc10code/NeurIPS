from dataclasses import dataclass
import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, FunctionTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pickle
import shap
import unittest
from src.utils.logging import lprint as log_print, LoggingLevels as ll

# -----------------------------
# Configuration
# -----------------------------
MAX_EPOCHS = 1000
PATIENCE = 20  # Early stopping patience

@dataclass
class Config:
    random_seed: int = 42
    num_workers: int = min(4, os.cpu_count() or 1)
    n_estimators: int = 50
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 4
    brain_hidden_dims: List[int] = (64, 32)
    brain_dropout: float = 0.5
    brain_lr: float = 1e-3
    brain_epochs: int = MAX_EPOCHS
    brain_batch_size: int = 64
    target_min: float = 0.0
    target_max: float = 1500.0
    output_dir: str = "output/regression"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_batch_norm: bool = True
    use_residual: bool = True
    rf_mlp_weight: float = 0.5  # Weight for combining RF and MLP predictions

    def __post_init__(self):
        log_print(ll.INFO, f"Using device: {self.device}")
        log_print(ll.INFO, f"Number of workers: {self.num_workers}")

# -----------------------------
# Random Forest Component
# -----------------------------
class RFComponent:
    def __init__(self, config: Config):
        self.config = config
        self.model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            n_jobs=config.num_workers,
            random_state=config.random_seed
        )
        self.scaler = None  # Will be set to RobustScaler or FunctionTransformer

    def fit(self, X, y):
        if sp.issparse(X):
            self.scaler = FunctionTransformer()  # No scaling for sparse binary data
            X_scaled = X
        else:
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return f"Trained RF component with {X.shape[1]} features, {len(y)} samples."

    def get_embedding(self, X):
        X_scaled = self.scaler.transform(X) if self.scaler else X
        return np.stack([tree.predict(X_scaled) for tree in self.model.estimators_], axis=1)

    def predict(self, X):
        X_scaled = self.scaler.transform(X) if self.scaler else X
        return self.model.predict(X_scaled)

# -----------------------------
# Tabular MLP with Residual Connections and Batch Normalization
# -----------------------------
class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.5, use_batch_norm=True, use_residual=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual and len(hidden_dims) > 1
        layers = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            layer = nn.Linear(prev, h)
            layers.append(layer)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.ModuleList(layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                x_out = layer(x)
                if self.use_residual and i > 0 and i % 3 == 0 and x_out.shape == x.shape:
                    x = x + x_out  # Residual connection
                else:
                    x = x_out
            else:
                x = layer(x)
        return self.head(x)

# -----------------------------
# Ensemble Trainer
# -----------------------------
class EnsembleTrainer:
    def __init__(self, config: Config, target_names: List[str], n_tab_features: int):
        self.config = config
        self.target_names = target_names
        self.n_tab_features = n_tab_features
        self.components: Dict[str, Dict[str, RFComponent]] = {t: {} for t in target_names}
        self.mlps: Dict[str, TabularMLP] = {}
        self.scalers: Dict[str, Any] = {}
        os.makedirs(config.output_dir, exist_ok=True)
        log_print(ll.INFO, f"Initialized EnsembleTrainer for {len(target_names)} targets")

    def train_target(self, X_desc, X_fp, X_tab, y_dict: Dict[str, np.ndarray], fold_idx: int = None):
        device = self.config.device
        log_print(ll.INFO, f"Starting training for {len(self.target_names)} targets{' (Fold ' + str(fold_idx) + ')' if fold_idx is not None else ''}")
        metrics = {t: {'train_mae': [], 'val_mae': [], 'best_val_loss': float('inf')} for t in self.target_names}
        log_buffer = []

        for target in tqdm(self.target_names, desc="Training MLPs for targets", leave=True):
            y = np.clip(y_dict[target], self.config.target_min, self.config.target_max)
            X_desc_tr, X_desc_val, X_fp_tr, X_fp_val, X_tab_tr, X_tab_val, y_tr, y_val = train_test_split(
                X_desc, X_fp, X_tab, y, test_size=0.1, random_state=self.config.random_seed
            )

            # Train RF components
            component_desc = RFComponent(self.config)
            component_fp = RFComponent(self.config)
            log_buffer.append(component_desc.fit(X_desc_tr, y_tr))
            log_buffer.append(component_fp.fit(X_fp_tr, y_tr))
            self.components[target]["desc"] = component_desc
            self.components[target]["fp"] = component_fp
            log_buffer.append(f"Trained RF components for {target} with {X_desc_tr.shape[1]} and {X_fp_tr.shape[1]} features, {len(y_tr)} samples.")

            # Generate embeddings
            emb_desc_tr = component_desc.get_embedding(X_desc_tr)
            emb_fp_tr = component_fp.get_embedding(X_fp_tr)
            X_emb_tr = np.concatenate([emb_desc_tr, emb_fp_tr, X_tab_tr.toarray()], axis=1)
            emb_desc_val = component_desc.get_embedding(X_desc_val)
            emb_fp_val = component_fp.get_embedding(X_fp_val)
            X_emb_val = np.concatenate([emb_desc_val, emb_fp_val, X_tab_val.toarray()], axis=1)

            # Train MLP
            mlp = TabularMLP(
                X_emb_tr.shape[1], self.config.brain_hidden_dims, self.config.brain_dropout,
                self.config.use_batch_norm, self.config.use_residual
            ).to(device)
            optimizer = torch.optim.Adam(mlp.parameters(), lr=self.config.brain_lr, weight_decay=1e-4)
            criterion = nn.MSELoss()

            dataset_tr = TensorDataset(
                torch.tensor(X_emb_tr, dtype=torch.float32, device=device),
                torch.tensor(y_tr, dtype=torch.float32, device=device).reshape(-1, 1)
            )
            dataset_val = TensorDataset(
                torch.tensor(X_emb_val, dtype=torch.float32, device=device),
                torch.tensor(y_val, dtype=torch.float32, device=device).reshape(-1, 1)
            )
            loader_tr = DataLoader(dataset_tr, batch_size=self.config.brain_batch_size, shuffle=True, num_workers=0, pin_memory=True)
            loader_val = DataLoader(dataset_val, batch_size=self.config.brain_batch_size, shuffle=False, num_workers=0, pin_memory=True)

            mlp.train()
            best_val_loss = float('inf')
            best_model_state = None
            patience = PATIENCE = 20
            counter = 0
            for epoch in tqdm(range(self.config.brain_epochs), desc=f"MLP Epochs (Target: {target})", leave=False):
                total_loss = 0
                for xb, yb in loader_tr:
                    optimizer.zero_grad(set_to_none=True)
                    y_pred = mlp(xb)
                    loss = criterion(y_pred, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                train_loss = total_loss / len(loader_tr)

                # Validation
                mlp.eval()
                val_loss = 0
                val_mae = 0
                with torch.no_grad():
                    for xb, yb in loader_val:
                        y_pred = mlp(xb)
                        val_loss += criterion(y_pred, yb).item()
                        val_mae += torch.mean(torch.abs(y_pred - yb)).item()
                val_loss /= len(loader_val)
                val_mae /= len(loader_val)
                mlp.train()

                if epoch % 5 == 0:
                    log_buffer.append(
                        f"Target {target} | Epoch {epoch+1}/{self.config.brain_epochs} | "
                        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}"
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = mlp.state_dict()
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        log_buffer.append(f"Early stopping at epoch {epoch+1} for {target}")
                        break

            mlp.load_state_dict(best_model_state)
            self.mlps[target] = mlp

            # Final metrics
            mlp.eval()
            with torch.no_grad():
                y_pred_tr = mlp(torch.tensor(X_emb_tr, dtype=torch.float32, device=device)).cpu().numpy().flatten()
                y_pred_val = mlp(torch.tensor(X_emb_val, dtype=torch.float32, device=device)).cpu().numpy().flatten()
            train_mae = mean_absolute_error(y_tr, y_pred_tr)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            metrics[target]['train_mae'] = train_mae
            metrics[target]['val_mae'] = val_mae
            metrics[target]['best_val_loss'] = best_val_loss
            log_buffer.append(
                f"Target {target} training complete | "
                f"Train MAE: {train_mae:.2f} | Val MAE: {val_mae:.2f} | Best Val Loss: {best_val_loss:.4f}"
            )

            # SHAP explanations
            explainer = shap.DeepExplainer(mlp, torch.tensor(X_emb_tr, dtype=torch.float32, device=device))
            shap_values = explainer.shap_values(torch.tensor(X_emb_val[:100], dtype=torch.float32, device=device))
            log_buffer.append(f"SHAP values computed for {target} (first 100 validation samples)")

        # Print all collected logs after tqdm
        for log in log_buffer:
            log_print(ll.INFO if "SHAP" in log or "Trained RF components" in log or "Epoch" in log else ll.SUCCESS, log)

        # Final summary
        avg_train_mae = np.mean([m['train_mae'] for m in metrics.values()])
        avg_val_mae = np.mean([m['val_mae'] for m in metrics.values()])
        log_print(ll.SUCCESS, (
            f"Training completed for all targets{' (Fold ' + str(fold_idx) + ')' if fold_idx is not None else ''} | "
            f"Average Train MAE: {avg_train_mae:.2f} | Average Val MAE: {avg_val_mae:.2f}"
        ))
        return metrics

    def predict(self, X_desc, X_fp, X_tab, target_name):
        if target_name not in self.target_names:
            raise ValueError(f"{target_name} not in trained targets.")
        component_desc = self.components[target_name]["desc"]
        component_fp = self.components[target_name]["fp"]
        emb_desc = component_desc.get_embedding(X_desc)
        emb_fp = component_fp.get_embedding(X_fp)
        X_emb = np.concatenate([emb_desc, emb_fp, X_tab.toarray()], axis=1)
        mlp = self.mlps[target_name]
        mlp.eval()
        with torch.no_grad():
            X_emb_tensor = torch.tensor(X_emb, dtype=torch.float32, device=self.config.device)
            y_pred_mlp = mlp(X_emb_tensor).cpu().numpy().flatten()
        y_pred_rf = (component_desc.predict(X_desc) + component_fp.predict(X_fp)) / 2
        y_pred = self.config.rf_mlp_weight * y_pred_rf + (1 - self.config.rf_mlp_weight) * y_pred_mlp
        return y_pred

# -----------------------------
# Main Regression Function
# -----------------------------
def main_regression(descriptor_file: str, map4_file: str, morgan_file: str, targets_file: str, config=Config()):
    log_print(ll.INFO, "Loading data...")
    desc_df = pd.read_csv(descriptor_file, sep="\t")
    df_desc = desc_df.set_index(desc_df.columns[0])
    map4_df = pd.read_csv(map4_file, sep="\t")
    df_map4 = map4_df.set_index(map4_df.columns[0])
    morgan_df = pd.read_csv(morgan_file, sep="\t")
    df_morgan = morgan_df.set_index(morgan_df.columns[0])
    targets_df = pd.read_csv(targets_file, sep="\t")
    df_targets = targets_df.set_index(targets_df.columns[0])

    common_ids = df_desc.index.intersection(df_map4.index).intersection(df_morgan.index).intersection(df_targets.index)
    df_desc = df_desc.loc[common_ids]
    df_map4 = df_map4.loc[common_ids]
    df_morgan = df_morgan.loc[common_ids]
    df_targets = df_targets.loc[common_ids]

    log_print(ll.INFO, f"Found {len(common_ids)} common samples after alignment")
    log_print(ll.INFO, "Preprocessing data...")
    df_targets.replace(-1, np.nan, inplace=True)
    df_targets.fillna(df_targets.median(), inplace=True)
    df_map4 = df_map4.astype(float)
    df_morgan = df_morgan.astype(float)
    tab_features = sp.csr_matrix((df_morgan > 0).astype(float).values)
    map4_features = sp.csr_matrix((df_map4 > 0).astype(float).values)  # Treat MAP4 as binary

    scaler_desc = StandardScaler()
    scaler_fp = FunctionTransformer()  # Identity for binary MAP4
    scaler_tab = StandardScaler(with_mean=False)  # No mean centering for sparse data
    X_desc = scaler_desc.fit_transform(df_desc.values)
    X_fp = map4_features  # No scaling for binary MAP4
    X_tab = scaler_tab.fit_transform(tab_features)

    target_names = df_targets.columns.tolist()
    y_dict = {t: df_targets[t].values for t in target_names}

    # K-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=config.random_seed)
    metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_desc)):
        log_print(ll.INFO, f"Training fold {fold_idx + 1}/5")
        X_desc_tr, X_desc_val = X_desc[train_idx], X_desc[val_idx]
        X_fp_tr, X_fp_val = X_fp[train_idx], X_fp[val_idx]
        X_tab_tr, X_tab_val = X_tab[train_idx], X_tab[val_idx]
        y_dict_tr = {t: y_dict[t][train_idx] for t in target_names}
        y_dict_val = {t: y_dict[t][val_idx] for t in target_names}

        trainer = EnsembleTrainer(config=config, target_names=target_names, n_tab_features=X_tab.shape[1])
        fold_metrics = trainer.train_target(X_desc_tr, X_fp_tr, X_tab_tr, y_dict_tr, fold_idx)
        metrics.append(fold_metrics)

        # Validation on fold
        fold_val_mae = 0
        for target in target_names:
            y_pred = trainer.predict(X_desc_val, X_fp_val, X_tab_val, target)
            val_mae = mean_absolute_error(y_dict_val[target], y_pred)
            fold_val_mae += val_mae
        fold_val_mae /= len(target_names)
        log_print(ll.INFO, f"Fold {fold_idx + 1} average validation MAE: {fold_val_mae:.2f}")

    # Save scalers
    with open(os.path.join(config.output_dir, "scaler_desc.pkl"), "wb") as f:
        pickle.dump(scaler_desc, f)
    with open(os.path.join(config.output_dir, "scaler_fp.pkl"), "wb") as f:
        pickle.dump(scaler_fp, f)
    with open(os.path.join(config.output_dir, "scaler_tab.pkl"), "wb") as f:
        pickle.dump(scaler_tab, f)

    # Average metrics across folds
    avg_metrics = {
        t: {
            'train_mae': np.mean([m[t]['train_mae'] for m in metrics]),
            'val_mae': np.mean([m[t]['val_mae'] for m in metrics]),
            'best_val_loss': np.mean([m[t]['best_val_loss'] for m in metrics])
        } for t in target_names
    }
    log_print(ll.SUCCESS, f"Cross-validation completed | Average metrics: {avg_metrics}")

    # Train on full dataset
    trainer = EnsembleTrainer(config=config, target_names=target_names, n_tab_features=X_tab.shape[1])
    trainer.train_target(X_desc, X_fp, X_tab, y_dict)
    trainer.scalers = {'desc': scaler_desc, 'fp': scaler_fp, 'tab': scaler_tab}

    return trainer

# -----------------------------
# Helper Function for TSV Writing
# -----------------------------
def append_to_tsv(file_path: str, data: Dict, first_write: bool = False):
    """Append a dictionary as a row to a TSV file, creating it if it doesn't exist."""
    # Convert data to a flat dictionary for TSV
    flat_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_data[f"{key}_{subkey}"] = subvalue
        else:
            flat_data[key] = value

    # Convert to DataFrame
    df = pd.DataFrame([flat_data])

    # Ensure all values are string-serializable
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x) if not isinstance(x, (int, float, str)) else x)

    # Write to TSV
    mode = 'w' if first_write else 'a'
    header = first_write
    df.to_csv(file_path, sep="\t", index=False, mode=mode, header=header)

# -----------------------------
# Hyperparameter Tuning with Bayesian Optimization
# -----------------------------
def tune_hyperparameters(X_desc, X_fp, X_tab, y_dict, target_names, output_dir="output/tuning") -> Tuple[Config, float]:
    # Initialize TSV file
    tsv_path = os.path.join(output_dir, "tuning_results.tsv")
    first_write = True

    def objective(params):
        nonlocal first_write
        start_time = time.time()
        # Parse brain_hidden_dims from string to tuple
        hidden_dims_str = params[4]
        hidden_dims = tuple(map(int, hidden_dims_str.split('_')))
        
        config = Config(
            n_estimators=int(params[0]),
            max_depth=int(params[1]),
            min_samples_split=int(params[2]),
            min_samples_leaf=int(params[3]),
            brain_hidden_dims=hidden_dims,
            brain_dropout=params[5],
            brain_lr=params[6],
            brain_epochs=MAX_EPOCHS,
            output_dir=output_dir,
            use_batch_norm=params[7],
            use_residual=params[8]
        )

        log_print(ll.INFO, (
            f"Testing config: n_estimators={int(params[0])}, max_depth={int(params[1])}, "
            f"min_samples_split={int(params[2])}, min_samples_leaf={int(params[3])}, "
            f"brain_hidden_dims={hidden_dims}, brain_dropout={params[5]:.3f}, "
            f"brain_lr={params[6]:.5f}, use_batch_norm={params[7]}, use_residual={params[8]}"
        ))

        trainer = EnsembleTrainer(config, target_names, X_tab.shape[1])
        metrics = trainer.train_target(X_desc, X_fp, X_tab, y_dict)
        avg_mae = np.mean([m['val_mae'] for m in metrics.values()])
        training_time = time.time() - start_time

        # Compute RF-specific metrics
        rf_metrics = {}
        for target in target_names:
            # Get RF predictions (average of desc and fp components)
            X_desc_tr, X_desc_val, X_fp_tr, X_fp_val, _, _, y_tr, y_val = train_test_split(
                X_desc, X_fp, X_tab, y_dict[target], test_size=0.1, random_state=config.random_seed
            )
            component_desc = trainer.components[target]["desc"]
            component_fp = trainer.components[target]["fp"]
            y_pred_rf_val = (component_desc.predict(X_desc_val) + component_fp.predict(X_fp_val)) / 2
            rf_val_mae = mean_absolute_error(y_val, y_pred_rf_val)
            rf_metrics[target] = {'rf_val_mae': rf_val_mae}

        # Prepare data for TSV
        trial_data = {
            'trial': len(pd.read_csv(tsv_path, sep="\t").index) + 1 if not first_write else 1,
            'n_estimators': int(params[0]),
            'max_depth': int(params[1]),
            'min_samples_split': int(params[2]),
            'min_samples_leaf': int(params[3]),
            'brain_hidden_dims': hidden_dims_str,  # Store as string for TSV
            'brain_dropout': params[5],
            'brain_lr': params[6],
            'use_batch_norm': params[7],
            'use_residual': params[8],
            'avg_val_mae': avg_mae,
            'training_time': training_time
        }
        # Add per-target metrics
        for target in target_names:
            trial_data[f"{target}_train_mae"] = metrics[target]['train_mae']
            trial_data[f"{target}_val_mae"] = metrics[target]['val_mae']
            trial_data[f"{target}_rf_val_mae"] = rf_metrics[target]['rf_val_mae']
            trial_data[f"{target}_best_val_loss"] = metrics[target]['best_val_loss']

        # Append to TSV
        append_to_tsv(tsv_path, trial_data, first_write=first_write)
        first_write = False

        log_print(ll.INFO, f"Trial {trial_data['trial']}: Average MAE: {avg_mae:.2f}, Training Time: {training_time:.2f}s")
        return avg_mae  # Return MAE as primary objective

    search_space = [
        Integer(50, 200),  # n_estimators
        Integer(5, 20),    # max_depth
        Integer(2, 10),    # min_samples_split
        Integer(1, 4),     # min_samples_leaf
        Categorical(['64_32', '128_64', '256_128']),  # brain_hidden_dims as strings
        Real(0.2, 0.5),    # brain_dropout
        Real(1e-4, 1e-2, prior='log-uniform'),  # brain_lr
        Categorical([True, False]),  # use_batch_norm
        Categorical([True, False])   # use_residual
    ]

    os.makedirs(output_dir, exist_ok=True)
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=20,
        random_state=Config().random_seed,
        verbose=True
    )

    # Parse best parameters
    best_mae = result.fun
    best_params = result.x
    best_hidden_dims = tuple(map(int, best_params[4].split('_')))
    
    best_config = Config(
        n_estimators=int(best_params[0]),
        max_depth=int(best_params[1]),
        min_samples_split=int(best_params[2]),
        min_samples_leaf=int(best_params[3]),
        brain_hidden_dims=best_hidden_dims,
        brain_dropout=best_params[5],
        brain_lr=best_params[6],
        brain_epochs=MAX_EPOCHS,
        output_dir=output_dir,
        use_batch_norm=best_params[7],
        use_residual=best_params[8]
    )

    log_print(ll.SUCCESS, f"Best config found with MAE: {best_mae:.2f}")
    log_print(ll.INFO, (
        f"Best config: n_estimators={int(best_params[0])}, max_depth={int(best_params[1])}, "
        f"min_samples_split={int(best_params[2])}, min_samples_leaf={int(best_params[3])}, "
        f"brain_hidden_dims={best_hidden_dims}, brain_dropout={best_params[5]:.3f}, "
        f"brain_lr={best_params[6]:.5f}, use_batch_norm={best_params[7]}, use_residual={best_params[8]}"
    ))

    with open(os.path.join(output_dir, "best_config.pkl"), "wb") as f:
        pickle.dump(best_config, f)
    log_print(ll.INFO, f"Best config saved to {os.path.join(output_dir, 'best_config.pkl')}")

    return best_config, best_mae

# -----------------------------
# Save Model
# -----------------------------
def save_model(trainer: EnsembleTrainer, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_state = {
        'config': trainer.config,
        'target_names': trainer.target_names,
        'n_tab_features': trainer.n_tab_features,
        'components': {},
        'mlps': {},
        'scalers': trainer.scalers
    }

    for target in trainer.target_names:
        model_state['components'][target] = {
            'desc': {
                'model': trainer.components[target]['desc'].model,
                'scaler': trainer.components[target]['desc'].scaler
            },
            'fp': {
                'model': trainer.components[target]['fp'].model,
                'scaler': trainer.components[target]['fp'].scaler
            }
        }
        model_state['mlps'][target] = trainer.mlps[target].state_dict()

    with open(save_path, 'wb') as f:
        pickle.dump(model_state, f)

    log_print(ll.SUCCESS, f"Model saved to {save_path}")

# -----------------------------
# Load Model
# -----------------------------
def load_model(model_path: str) -> EnsembleTrainer:
    with open(model_path, 'rb') as f:
        model_state = pickle.load(f)

    config = model_state['config']
    target_names = model_state['target_names']
    n_tab_features = model_state['n_tab_features']
    scalers = model_state['scalers']

    trainer = EnsembleTrainer(config=config, target_names=target_names, n_tab_features=n_tab_features)

    for target in target_names:
        trainer.components[target]['desc'] = RFComponent(config)
        trainer.components[target]['desc'].model = model_state['components'][target]['desc']['model']
        trainer.components[target]['desc'].scaler = model_state['components'][target]['desc']['scaler']
        trainer.components[target]['fp'] = RFComponent(config)
        trainer.components[target]['fp'].model = model_state['components'][target]['fp']['model']
        trainer.components[target]['fp'].scaler = model_state['components'][target]['fp']['scaler']
        mlp = TabularMLP(
            input_dim=2 * config.n_estimators + n_tab_features,
            hidden_dims=config.brain_hidden_dims,
            dropout=config.brain_dropout,
            use_batch_norm=config.use_batch_norm,
            use_residual=config.use_residual
        ).to(config.device)
        mlp.load_state_dict(model_state['mlps'][target])
        trainer.mlps[target] = mlp
    trainer.scalers = scalers

    log_print(ll.SUCCESS, f"Model loaded from {model_path}")
    return trainer

# -----------------------------
# Predict with Loaded Model
# -----------------------------
def predict_with_model(trainer: EnsembleTrainer, descriptor_file: str, map4_file: str, morgan_file: str, target_name: str) -> np.ndarray:
    if target_name not in trainer.target_names:
        raise ValueError(f"Target {target_name} not in trained targets: {trainer.target_names}")

    log_print(ll.INFO, "Loading and preprocessing data for prediction...")
    desc_df = pd.read_csv(descriptor_file, sep="\t")
    desc_df = desc_df.set_index(desc_df.columns[0])  # Correctly set index
    map4_df = pd.read_csv(map4_file, sep="\t")
    map4_df = map4_df.set_index(map4_df.columns[0])  # Fixed: Define map4_df before using it
    morgan_df = pd.read_csv(morgan_file, sep="\t")
    morgan_df = morgan_df.set_index(morgan_df.columns[0])  # Fixed: Define morgan_df before using it

    common_ids = desc_df.index.intersection(map4_df.index).intersection(morgan_df.index)
    desc_df = desc_df.loc[common_ids]
    map4_df = map4_df.loc[common_ids]
    morgan_df = morgan_df.loc[common_ids]

    X_desc = trainer.scalers['desc'].transform(desc_df.values)
    X_fp = sp.csr_matrix((map4_df > 0).astype(float).values)  # Treat MAP4 as binary
    X_tab = trainer.scalers['tab'].transform(sp.csr_matrix((morgan_df > 0).astype(float).values))

    log_print(ll.INFO, f"Predicting for target {target_name}...")
    y_pred = trainer.predict(X_desc, X_fp, X_tab, target_name)

    log_print(ll.SUCCESS, f"Predictions completed for {target_name} with {len(y_pred)} samples")
    return y_pred
