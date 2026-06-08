# Kaggle Playground Series S6E6 - Predicting Stellar Class

A self-contained pipeline for predicting stellar class (Galaxy, Star, or Quasar) using balanced accuracy on SDSS tabular data.

## Project Structure

```
.
├── configs/            # JSON/YAML configuration files for model parameters
├── data/               # All raw, processed, and external datasets (Git ignored)
│   └── raw/            # Contains train.csv, test.csv, sample_submission.csv
├── experiments/        # Logged metrics and tracking (results.csv)
├── models/             # Saved model weights and OOF predictions (Git ignored)
│   └── oof/
├── notebooks/          # Jupyter notebooks for EDA and rapid prototyping
├── src/                # Source code and utility scripts
│   ├── download_data.py
│   ├── experiment_utils.py
│   ├── submission_budget.py
│   ├── train_s6e6.py
│   └── validate_submission.py
├── pyproject.toml      # Dependency declaration managed by uv
└── README.md
```

## Quick Start

### 1. Download Competition Data

To redownload datasets using the Kaggle API:

```bash
python3 src/download_data.py playground-series-s6e6
```

### 2. Train a Model (LightGBM, CatBoost, XGBoost)

Train a model with 5-fold cross-validation, feature engineering (SDSS color indices), OOF predictions logging, and submission file generation:

```bash
# Train LightGBM baseline
python3 src/train_s6e6.py --model lightgbm

# Train CatBoost baseline
python3 src/train_s6e6.py --model catboost

# Train XGBoost baseline
python3 src/train_s6e6.py --model xgboost
```

### 3. Validate Submissions

Validate a submission CSV file against the sample submission template before uploading it to Kaggle:

```bash
python3 src/validate_submission.py submissions/submission_lightgbm_stellar_baseline.csv --id-col id --target-col class
```

### 4. Check Submission Budget

View recent submissions and remaining daily limit directly from your terminal:

```bash
python3 src/submission_budget.py playground-series-s6e6
```

