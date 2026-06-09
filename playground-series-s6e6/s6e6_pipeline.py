#!/usr/bin/env python3
"""Unified training and blending pipeline for Kaggle Playground Series S6E6.

This script runs locally or on Kaggle, engineering features, training 
LightGBM and XGBoost models, and blending their predictions.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb


def get_data_paths() -> tuple[Path, Path, Path]:
    # Detect Kaggle environment
    if os.path.exists("/kaggle/input"):
        train = Path("/kaggle/input/playground-series-s6e6/train.csv")
        test = Path("/kaggle/input/playground-series-s6e6/test.csv")
        sub = Path("submission.csv")
    else:
        # Local workspace paths
        root = Path(__file__).resolve().parent
        train = root / "data" / "raw" / "train.csv"
        test = root / "data" / "raw" / "test.csv"
        sub = root / "submissions" / "submission_blended_lgb_xgb.csv"
    return train, test, sub


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
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


def main() -> None:
    train_path, test_path, sub_path = get_data_paths()
    
    if not train_path.exists():
        print(f"Error: train.csv not found at {train_path}")
        return
        
    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print("Engineering features...")
    train = feature_engineering(train)
    test = feature_engineering(test)
    
    le = LabelEncoder()
    y = le.fit_transform(train["class"])
    print(f"Classes mapped: {dict(zip(le.classes_, range(3)))}")
    
    exclude_cols = ["id", "class"]
    features = [c for c in train.columns if c not in exclude_cols]
    
    # Encode categorical features
    cat_cols = ["spectral_type", "galaxy_population"]
    for c in cat_cols:
        train[c] = train[c].astype("category")
        test[c] = test[c].astype("category")
        
    X = train[features]
    X_test = test[features]
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 1. Train LightGBM
    print("\n=== Training LightGBM ===")
    lgb_oof = np.zeros((len(train), 3))
    lgb_test = np.zeros((len(test), 3))
    
    lgb_params = {
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
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        lgb_oof[val_idx] = model.predict_proba(X_val)
        lgb_test += model.predict_proba(X_test) / n_splits
        
    lgb_score = balanced_accuracy_score(y, np.argmax(lgb_oof, axis=1))
    print(f"LightGBM CV Balanced Accuracy: {lgb_score:.6f}")
    
    # 2. Train XGBoost
    print("\n=== Training XGBoost ===")
    xgb_oof = np.zeros((len(train), 3))
    xgb_test = np.zeros((len(test), 3))
    
    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 1000,
        "random_state": 42,
        "n_jobs": -1
    }
    
    # XGBoost needs categories encoded as integers
    X_xgb = X.copy()
    X_test_xgb = X_test.copy()
    for c in cat_cols:
        X_xgb[c] = X_xgb[c].cat.codes
        X_test_xgb[c] = X_test_xgb[c].cat.codes
        
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_xgb, y)):
        X_train, y_train = X_xgb.iloc[train_idx], y[train_idx]
        X_val, y_val = X_xgb.iloc[val_idx], y[val_idx]
        
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        xgb_oof[val_idx] = model.predict_proba(X_val)
        xgb_test += model.predict_proba(X_test_xgb) / n_splits
        
    xgb_score = balanced_accuracy_score(y, np.argmax(xgb_oof, axis=1))
    print(f"XGBoost CV Balanced Accuracy: {xgb_score:.6f}")
    
    # 3. Blend (54% LGB / 46% XGB)
    print("\n=== Blending predictions ===")
    w_lgb = 0.54
    w_xgb = 0.46
    
    blended_oof = w_lgb * lgb_oof + w_xgb * xgb_oof
    blended_score = balanced_accuracy_score(y, np.argmax(blended_oof, axis=1))
    print(f"Blended CV Balanced Accuracy: {blended_score:.6f}")
    
    # Generate final blended predictions
    blended_test = w_lgb * lgb_test + w_xgb * xgb_test
    test_classes = np.argmax(blended_test, axis=1)
    test_class_labels = le.inverse_transform(test_classes)
    
    print(f"Saving submission to {sub_path}...")
    sub = pd.DataFrame({"id": test["id"], "class": test_class_labels})
    sub.to_csv(sub_path, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
