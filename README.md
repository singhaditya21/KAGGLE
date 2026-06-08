# Kaggle Workspace Project

A clean, production-grade template for structuring competitive machine learning and data science projects.

## Project Structure

```
.
├── configs/            # JSON/YAML configuration files for model parameters
├── data/               # All raw, processed, and external datasets (Git ignored)
│   ├── raw/
│   └── processed/
├── experiments/        # Logged metrics and tracking (results.csv)
├── models/             # Saved model weights, checkpoints, and OOF predictions (Git ignored)
│   └── oof/
├── notebooks/          # Jupyter notebooks for EDA and rapid prototyping
├── src/                # Source code and utility scripts
│   ├── download_data.py
│   ├── experiment_utils.py
│   ├── submission_budget.py
│   ├── train_baseline.py
│   └── validate_submission.py
├── pyproject.toml      # Dependency declaration managed by uv
└── README.md
```

## Quick Start

### 1. Download Competition Data

To download datasets using the Kaggle API (automatically authenticated with your `KAGGLE_API_TOKEN` environment variable):

```bash
python3 src/download_data.py <competition-name>
```

Example:
```bash
python3 src/download_data.py playground-series-s6e5
```

### 2. Train a Baseline Model

Run the LightGBM cross-validation baseline script. It performs stratified K-fold CV, generates a submission, saves Out-of-Fold (OOF) predictions, and logs results:

```bash
python3 src/train_baseline.py --config configs/lgbm_baseline.json
```

### 3. Validate Submissions

Validate a submission CSV file against the sample submission template before uploading it to Kaggle:

```bash
python3 src/validate_submission.py submissions/submission_lgbm_baseline_v1.csv --id-col id --target-col target
```

### 4. Check Submission Budget

View recent submissions and remaining daily limit directly from your terminal:

```bash
python3 src/submission_budget.py <competition-name>
```

Example:
```bash
python3 src/submission_budget.py playground-series-s6e5
```
