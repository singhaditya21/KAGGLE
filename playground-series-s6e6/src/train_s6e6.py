#!/usr/bin/env python3
"""Train a cross-validated model on the Stellar Class (S6E6) dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from experiment_utils import project_root, append_result, utc_now


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Creates astronomical color index features and spatial/redshift transformations."""
    df = df.copy()
    
    # Photometric color differences (ratios in log scale)
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]
    
    df["u_r"] = df["u"] - df["r"]
    df["u_i"] = df["u"] - df["i"]
    df["u_z"] = df["u"] - df["z"]
    
    df["g_i"] = df["g"] - df["i"]
    df["g_z"] = df["g"] - df["z"]
    
    df["r_z"] = df["r"] - df["z"]
    
    # Convert alpha and delta (spherical sky coordinates) to Cartesian coordinates
    alpha_rad = np.radians(df["alpha"])
    delta_rad = np.radians(df["delta"])
    df["x"] = np.cos(delta_rad) * np.cos(alpha_rad)
    df["y"] = np.cos(delta_rad) * np.sin(alpha_rad)
    df["z"] = np.sin(delta_rad)
    
    # Redshift features (cube root handles negatives and compresses long tails)
    df["redshift_cbrt"] = np.cbrt(df["redshift"])
    df["redshift_is_near_zero"] = (np.abs(df["redshift"]) < 0.005).astype(int)
    df["redshift_is_negative"] = (df["redshift"] < 0).astype(int)
    
    # Statistical aggregates across bands
    bands = ["u", "g", "r", "i", "z"]
    df["band_mean"] = df[bands].mean(axis=1)
    df["band_std"] = df[bands].std(axis=1)
    df["band_max"] = df[bands].max(axis=1)
    df["band_min"] = df[bands].min(axis=1)
    df["band_range"] = df["band_max"] - df["band_min"]
    
    return df


def train_s6e6(model_type: str = "lightgbm", config_path: Path | None = None) -> None:
    root = project_root()
    
    # Default configs
    config = {
        "run_name": f"{model_type}_stellar_baseline",
        "competition": "playground-series-s6e6",
        "n_splits": 5,
        "seed": 42,
        "lgb_params": {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": -1,
            "n_estimators": 1000,
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1
        },
        "xgb_params": {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "learning_rate": 0.05,
            "max_depth": 6,
            "n_estimators": 1000,
            "random_state": 42,
            "n_jobs": -1
        },
        "cat_params": {
            "loss_function": "MultiClass",
            "learning_rate": 0.05,
            "depth": 6,
            "iterations": 1000,
            "random_seed": 42,
            "verbose": 100,
            "thread_count": -1
        }
    }
    
    if config_path and config_path.exists():
        with open(config_path) as f:
            user_config = json.load(f)
            for k, v in user_config.items():
                if isinstance(v, dict) and k in config:
                    config[k].update(v)
                else:
                    config[k] = v
                    
    print(f"Starting {model_type.upper()} pipeline for run: {config['run_name']}")
    
    # Path configuration
    train_path = root / "data" / "raw" / "train.csv"
    test_path = root / "data" / "raw" / "test.csv"
    
    if not train_path.exists() or not test_path.exists():
        print("Error: train.csv or test.csv not found.")
        return
        
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Feature Engineering
    print("Engineering features...")
    train = feature_engineering(train)
    test = feature_engineering(test)
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(train["class"])
    print(f"Target classes mapped: {dict(zip(le.classes_, range(3)))}")
    
    # Define features
    exclude_cols = ["id", "class"]
    features = [c for c in train.columns if c not in exclude_cols]
    
    # Encode categorical features
    cat_cols = ["spectral_type", "galaxy_population"]
    for c in cat_cols:
        if c in features:
            train[c] = train[c].astype("category")
            test[c] = test[c].astype("category")
            
    X = train[features]
    X_test = test[features]
    
    oof_preds = np.zeros((len(train), 3))
    test_preds = np.zeros((len(test), 3))
    
    skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["seed"])
    fold_scores = []
    
    print(f"Training {model_type.upper()} with {config['n_splits']}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        if model_type == "lightgbm":
            model = lgb.LGBMClassifier(**config["lgb_params"])
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )
            val_fold_preds = model.predict_proba(X_val)
            test_fold_preds = model.predict_proba(X_test)
            
        elif model_type == "xgboost":
            # Convert categorical types to numeric for XGBoost
            X_train_xgb = X_train.copy()
            X_val_xgb = X_val.copy()
            X_test_xgb = X_test.copy()
            for c in cat_cols:
                X_train_xgb[c] = X_train_xgb[c].cat.codes
                X_val_xgb[c] = X_val_xgb[c].cat.codes
                X_test_xgb[c] = X_test_xgb[c].cat.codes
                
            model = xgb.XGBClassifier(**config["xgb_params"])
            model.fit(
                X_train_xgb, y_train,
                eval_set=[(X_val_xgb, y_val)],
                verbose=False
            )
            val_fold_preds = model.predict_proba(X_val_xgb)
            test_fold_preds = model.predict_proba(X_test_xgb)
            
        elif model_type == "catboost":
            # Convert categorical cols to strings for CatBoost
            X_train_cat = X_train.copy()
            X_val_cat = X_val.copy()
            X_test_cat = X_test.copy()
            for c in cat_cols:
                X_train_cat[c] = X_train_cat[c].astype(str)
                X_val_cat[c] = X_val_cat[c].astype(str)
                X_test_cat[c] = X_test_cat[c].astype(str)
                
            model = CatBoostClassifier(
                **config["cat_params"],
                cat_features=cat_cols
            )
            model.fit(
                X_train_cat, y_train,
                eval_set=(X_val_cat, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
            val_fold_preds = model.predict_proba(X_val_cat)
            test_fold_preds = model.predict_proba(X_test_cat)
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        oof_preds[val_idx] = val_fold_preds
        test_preds += test_fold_preds / config["n_splits"]
        
        # Calculate fold balanced accuracy score
        val_fold_class = np.argmax(val_fold_preds, axis=1)
        score = balanced_accuracy_score(y_val, val_fold_class)
        fold_scores.append(score)
        print(f"Fold {fold} Balanced Accuracy: {score:.6f}")
        
    oof_classes = np.argmax(oof_preds, axis=1)
    cv_score = balanced_accuracy_score(y, oof_classes)
    print(f"Overall CV Balanced Accuracy: {cv_score:.6f}")
    
    # Save submission
    submissions_dir = root / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    sub_path = submissions_dir / f"submission_{config['run_name']}.csv"
    
    test_classes = np.argmax(test_preds, axis=1)
    test_class_labels = le.inverse_transform(test_classes)
    
    sub = pd.DataFrame({"id": test["id"], "class": test_class_labels})
    sub.to_csv(sub_path, index=False)
    
    # Save OOF and test predictions
    oof_dir = root / "models" / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"oof_{config['run_name']}.npy"
    test_preds_path = oof_dir / f"test_preds_{config['run_name']}.npy"
    np.save(oof_path, oof_preds)
    np.save(test_preds_path, test_preds)
    
    print(f"Saved submission to {sub_path.relative_to(root)}")
    print(f"Saved OOF predictions to {oof_path.relative_to(root)}")
    print(f"Saved test predictions to {test_preds_path.relative_to(root)}")
    
    # Append results to experiment log
    result_row = {
        "timestamp_utc": utc_now(),
        "run_name": config["run_name"],
        "competition": config["competition"],
        "model_family": model_type.upper(),
        "feature_set": f"Feature v2 ({len(features)} features: colors, coordinates, redshift transformations)",
        "oof_score": cv_score,
        "fold_scores_json": fold_scores,
        "submission_path": str(sub_path),
        "oof_path": str(oof_path),
        "config_path": str(config_path) if config_path else "",
        "notes": f"Model {model_type} run on Stellar Class (S6E6) with Feature v2.",
    }
    append_result(root, result_row)
    print("Experiment logged successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model on S6E6 Stellar Class dataset.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="lightgbm", 
        choices=["lightgbm", "xgboost", "catboost"],
        help="Model type to train"
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config JSON file")
    args = parser.parse_args()
    
    train_s6e6(args.model, args.config)


if __name__ == "__main__":
    main()
