# NeurIPS
Neural Network for polymer feature prediction
## Project Description

**NeurIPS** is a machine learning project focused on predicting polymer features using neural networks. The project provides scripts for data preparation, model training, and prediction, enabling streamlined workflows for polymer property analysis.

## Tutorial

### 1. Data Preparation (`prepare_data.py`)

This script processes smiles of polymers and generates datasets suitable for training and evaluation.

**Usage:**
```bash
python3 prepare_data.py --input ./output/train.ismiles  --out output/train --fm-bits  1024 --fmap4-bits 2048
```
- `--input`: Path to the raw data file.
- `--output`: Path to save the processed data.
- `--fm-bits`: 
- `--fmap4-bits`:
- `--dt-list`:

### 2. Model Training (`train.py`)

This script trains a model on the prepared dataset (run prepare_data.py before train.py)

**Usage:**
```bash
python3 train.py --input-train ./output/train --out ./output/model
```
- `--input-train`: base name of imput data for train (.dt, .fm, .fmap4, .tt)
- `--out`: train.py output dir

### 3. Prediction (`predict.py`)

This script uses a trained model to predict polymer features on new data.

**Usage:**
```bash
python3 predict.py --model model.pth --input new_samples.csv --output predictions.csv
```
- `--model`: Path to the trained model file.
- `--input`: Path to the input data for prediction.
- `--output`: Path to save the prediction results.