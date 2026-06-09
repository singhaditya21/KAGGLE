#!/usr/bin/env python3
"""Blend OOF probabilities from LightGBM and CatBoost to find the optimal combination."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from experiment_utils import project_root, append_result, utc_now


def main() -> None:
    root = project_root()
    
    # Load labels
    train_path = root / "data" / "raw" / "train.csv"
    test_path = root / "data" / "raw" / "test.csv"
    if not train_path.exists() or not test_path.exists():
        print("Error: train.csv or test.csv not found.")
        return
        
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    le = LabelEncoder()
    y = le.fit_transform(train["class"])
    
    # Paths to OOF files
    oof_dir = root / "models" / "oof"
    lgb_oof_path = oof_dir / "oof_lightgbm_stellar_baseline.npy"
    cat_oof_path = oof_dir / "oof_catboost_stellar_baseline.npy"
    xgb_oof_path = oof_dir / "oof_xgboost_stellar_baseline.npy"
    
    if not lgb_oof_path.exists() or not cat_oof_path.exists() or not xgb_oof_path.exists():
        print("Error: OOF predictions from all three models must be generated first.")
        print(f"Checking for LightGBM OOF: {lgb_oof_path.exists()}")
        print(f"Checking for CatBoost OOF: {cat_oof_path.exists()}")
        print(f"Checking for XGBoost OOF: {xgb_oof_path.exists()}")
        return
        
    lgb_oof = np.load(lgb_oof_path)
    cat_oof = np.load(cat_oof_path)
    xgb_oof = np.load(xgb_oof_path)
    
    # 3-way Grid Search for optimal blend weights (w1 + w2 + w3 = 1.0)
    best_weights = (0.0, 0.0, 0.0)
    best_score = 0.0
    
    print("Searching for optimal 3-way blend weights (w_lgb * LGB + w_cat * Cat + w_xgb * XGB)...")
    
    # Iterate with step size 0.02 for finer search (approx 1326 combinations)
    for w_lgb in np.linspace(0, 1, 51):
        for w_cat in np.linspace(0, 1 - w_lgb, 51):
            w_xgb = 1.0 - w_lgb - w_cat
            # Ensure sum is exactly 1.0 and weights are non-negative
            if w_xgb < -1e-6:
                continue
            w_xgb = max(w_xgb, 0.0)
            
            blended_oof = w_lgb * lgb_oof + w_cat * cat_oof + w_xgb * xgb_oof
            blended_class = np.argmax(blended_oof, axis=1)
            score = balanced_accuracy_score(y, blended_class)
            
            if score > best_score:
                best_score = score
                best_weights = (w_lgb, w_cat, w_xgb)
                
    w_lgb, w_cat, w_xgb = best_weights
    print(f"Optimal weights -> LightGBM: {w_lgb:.2f}, CatBoost: {w_cat:.2f}, XGBoost: {w_xgb:.2f}")
    print(f"Best 3-way blended OOF Balanced Accuracy: {best_score:.6f}")
    
    # Load test prediction probabilities
    lgb_test_path = oof_dir / "test_preds_lightgbm_stellar_baseline.npy"
    cat_test_path = oof_dir / "test_preds_catboost_stellar_baseline.npy"
    xgb_test_path = oof_dir / "test_preds_xgboost_stellar_baseline.npy"
    
    if not lgb_test_path.exists() or not cat_test_path.exists() or not xgb_test_path.exists():
        print("Warning: Test probability files not found. Cannot generate blended submission.")
        return
        
    lgb_test = np.load(lgb_test_path)
    cat_test = np.load(cat_test_path)
    xgb_test = np.load(xgb_test_path)
    
    # Blend test predictions
    blended_test = w_lgb * lgb_test + w_cat * cat_test + w_xgb * xgb_test
    test_classes = np.argmax(blended_test, axis=1)
    test_class_labels = le.inverse_transform(test_classes)
    
    # Save blended submission
    submissions_dir = root / "submissions"
    sub_path = submissions_dir / "submission_blended_lgb_cat_xgb.csv"
    sub = pd.DataFrame({"id": test["id"], "class": test_class_labels})
    sub.to_csv(sub_path, index=False)
    print(f"Successfully saved blended submission to {sub_path.relative_to(root)}")
    
    # Append results to experiment log
    result_row = {
        "timestamp_utc": utc_now(),
        "run_name": "lgb_cat_xgb_blend",
        "competition": "playground-series-s6e6",
        "model_family": "BLEND",
        "feature_set": "Ensemble (LGB + CatBoost + XGBoost)",
        "oof_score": best_score,
        "fold_scores_json": [],
        "submission_path": str(sub_path),
        "oof_path": "",
        "config_path": "",
        "notes": f"Optimal blend weights: LGB {w_lgb:.2f}, CatBoost {w_cat:.2f}, XGBoost {w_xgb:.2f}",
    }
    append_result(root, result_row)
    print("Blended experiment logged successfully.")


if __name__ == "__main__":
    main()
