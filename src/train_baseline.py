#!/usr/bin/env python3
"""Template script for training a baseline tabular model on Kaggle datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from experiment_utils import project_root, append_result, utc_now


def train_baseline(config_path: Path | None = None) -> None:
    root = project_root()
    
    # Default parameters
    config = {
        "run_name": "lgbm_baseline",
        "competition": "playground-series-s6e5",
        "id_col": "id",
        "target_col": "target",
        "n_splits": 5,
        "seed": 42,
        "lgb_params": {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 500,
            "random_state": 42,
            "verbose": -1,
        }
    }
    
    if config_path and config_path.exists():
        with open(config_path) as f:
            user_config = json.load(f)
            # Update dictionary recursively or simply
            for k, v in user_config.items():
                if isinstance(v, dict) and k in config:
                    config[k].update(v)
                else:
                    config[k] = v
                    
    print(f"Starting baseline run: {config['run_name']}")
    
    # Load dataset
    train_path = root / "data" / "raw" / "train.csv"
    test_path = root / "data" / "raw" / "test.csv"
    
    if not train_path.exists() or not test_path.exists():
        print(f"Error: train.csv or test.csv not found under data/raw/.")
        print(f"Please run: python3 src/download_data.py {config['competition']}")
        return
        
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    id_col = config["id_col"]
    target_col = config["target_col"]
    
    # Simple feature selection (numeric features only as a fallback baseline)
    features = [c for c in train.columns if c not in [id_col, target_col]]
    
    # Identify categorical features if any
    cat_features = list(train[features].select_dtypes(include=["object", "category"]).columns)
    for c in cat_features:
        train[c] = train[c].astype("category")
        test[c] = test[c].astype("category")
        
    X = train[features]
    y = train[target_col]
    X_test = test[features]
    
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    
    skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["seed"])
    fold_scores = []
    
    print(f"Training LightGBM model with {config['n_splits']} folds...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**config["lgb_params"])
        
        # Fit with early stopping callback if supported
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model.fit(
            X_train, 
            y_train, 
            eval_set=[(X_val, y_val)], 
            callbacks=callbacks
        )
        
        # Predict on OOF
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        # Predict on test
        test_preds += model.predict_proba(X_test)[:, 1] / config["n_splits"]
        
        score = roc_auc_score(y_val, val_preds)
        fold_scores.append(score)
        print(f"Fold {fold} AUC: {score:.6f}")
        
    cv_score = roc_auc_score(y, oof_preds)
    print(f"Overall CV AUC: {cv_score:.6f}")
    
    # Write OOF and Submissions
    submissions_dir = root / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    
    sub_path = submissions_dir / f"submission_{config['run_name']}.csv"
    sub = pd.DataFrame({id_col: test[id_col], target_col: test_preds})
    sub.to_csv(sub_path, index=False)
    
    oof_dir = root / "models" / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"oof_{config['run_name']}.npy"
    np.save(oof_path, oof_preds)
    
    print(f"Saved submission to {sub_path.relative_to(root)}")
    print(f"Saved OOF predictions to {oof_path.relative_to(root)}")
    
    # Append results to experiment log
    result_row = {
        "timestamp_utc": utc_now(),
        "run_name": config["run_name"],
        "competition": config["competition"],
        "model_family": "LightGBM",
        "feature_set": f"Baseline ({len(features)} features)",
        "oof_score": cv_score,
        "fold_scores_json": fold_scores,
        "submission_path": str(sub_path),
        "oof_path": str(oof_path),
        "config_path": str(config_path) if config_path else "",
        "notes": f"Baseline LightGBM run with {config['n_splits']}-fold CV.",
    }
    append_result(root, result_row)
    print("Experiment successfully logged.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM baseline model.")
    parser.add_argument("--config", type=Path, default=None, help="Path to config JSON file")
    args = parser.parse_args()
    
    train_baseline(args.config)


if __name__ == "__main__":
    main()
