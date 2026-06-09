#!/usr/bin/env python3
"""Blend OOF probabilities from LightGBM and CatBoost to find the optimal combination."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from experiment_utils import project_root, append_result, utc_now


import argparse

def main() -> None:
    root = project_root()
    
    parser = argparse.ArgumentParser(description="Blend OOF predictions.")
    parser.add_argument(
        "--use-original", 
        action="store_true", 
        help="Blend models trained with original SDSS17 data appended"
    )
    args = parser.parse_args()
    
    suffix = "_with_original" if args.use_original else ""
    
    # Load labels
    train_path = root / "data" / "raw" / "train.csv"
    test_path = root / "data" / "raw" / "test.csv"
    if not train_path.exists() or not test_path.exists():
        print("Error: train.csv or test.csv not found.")
        return
        
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    if args.use_original:
        orig_path = root / "data" / "external" / "star_classification.csv"
        if orig_path.exists():
            print("Loading original SDSS17 dataset for blending labels...")
            orig = pd.read_csv(orig_path)
            orig_cols = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift", "class"]
            orig = orig[orig_cols].copy()
            
            # Reconstruct synthetic threshold features with 100% precision
            u_r = orig["u"] - orig["r"]
            orig["galaxy_population"] = np.where(u_r > 2.20, "Red_Sequence", "Blue_Cloud")
            
            g_r = orig["g"] - orig["r"]
            orig["spectral_type"] = pd.cut(
                g_r,
                bins=[-np.inf, 0.0, 0.5, 1.0, np.inf],
                labels=["O/B", "A/F", "G/K", "M"]
            ).astype(str)
            
            orig["id"] = -1
            
            train = pd.concat([train, orig], ignore_index=True)
            print(f"Appended {len(orig)} original rows. Combined train shape: {train.shape}")
        else:
            print(f"Warning: Original dataset not found at {orig_path}. Proceeding with synthetic data only.")
            
    le = LabelEncoder()
    y = le.fit_transform(train["class"])
    
    # Paths to OOF files
    oof_dir = root / "models" / "oof"
    lgb_oof_path = oof_dir / f"oof_lightgbm_stellar_baseline{suffix}.npy"
    xgb_oof_path = oof_dir / f"oof_xgboost_stellar_baseline{suffix}.npy"
    
    if not lgb_oof_path.exists() or not xgb_oof_path.exists():
        print("Error: OOF predictions from LightGBM and XGBoost must be generated first.")
        print(f"Checking for LightGBM OOF: {lgb_oof_path.exists()}")
        print(f"Checking for XGBoost OOF: {xgb_oof_path.exists()}")
        return
        
    lgb_oof = np.load(lgb_oof_path)
    xgb_oof = np.load(xgb_oof_path)
    
    # Check if CatBoost is available (optional in 3-way blend if we only ran LGB + XGB)
    cat_oof_path = oof_dir / f"oof_catboost_stellar_baseline{suffix}.npy"
    use_cat = cat_oof_path.exists()
    
    if use_cat:
        print("Including CatBoost in the blend...")
        cat_oof = np.load(cat_oof_path)
    else:
        print("CatBoost OOF not found. Running 2-way blend (LGB + XGB) instead.")
        cat_oof = np.zeros_like(lgb_oof)
        
    # Grid Search for optimal blend weights (w1 + w2 + w3 = 1.0)
    best_weights = (0.0, 0.0, 0.0)
    best_score = 0.0
    
    print(f"Searching for optimal blend weights{suffix}...")
    
    # Iterate with step size 0.02
    for w_lgb in np.linspace(0, 1, 51):
        # If not using CatBoost, w_cat is fixed to 0
        w_cat_max = (1 - w_lgb) if use_cat else 0.0
        w_cat_steps = 51 if use_cat else 1
        
        for w_cat in np.linspace(0, w_cat_max, w_cat_steps):
            w_xgb = 1.0 - w_lgb - w_cat
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
    print(f"Best blended OOF Balanced Accuracy: {best_score:.6f}")
    
    # Load test prediction probabilities
    lgb_test_path = oof_dir / f"test_preds_lightgbm_stellar_baseline{suffix}.npy"
    xgb_test_path = oof_dir / f"test_preds_xgboost_stellar_baseline{suffix}.npy"
    
    if not lgb_test_path.exists() or not xgb_test_path.exists():
        print("Warning: Test probability files not found. Cannot generate blended submission.")
        return
        
    lgb_test = np.load(lgb_test_path)
    xgb_test = np.load(xgb_test_path)
    
    if use_cat:
        cat_test_path = oof_dir / f"test_preds_catboost_stellar_baseline{suffix}.npy"
        cat_test = np.load(cat_test_path) if cat_test_path.exists() else np.zeros_like(lgb_test)
    else:
        cat_test = np.zeros_like(lgb_test)
        
    # Blend test predictions
    blended_test = w_lgb * lgb_test + w_cat * cat_test + w_xgb * xgb_test
    test_classes = np.argmax(blended_test, axis=1)
    test_class_labels = le.inverse_transform(test_classes)
    
    # Save blended submission
    submissions_dir = root / "submissions"
    sub_path = submissions_dir / f"submission_blended_lgb_cat_xgb{suffix}.csv"
    sub = pd.DataFrame({"id": test["id"], "class": test_class_labels})
    sub.to_csv(sub_path, index=False)
    print(f"Successfully saved blended submission to {sub_path.relative_to(root)}")
    
    # Append results to experiment log
    result_row = {
        "timestamp_utc": utc_now(),
        "run_name": f"lgb_cat_xgb_blend{suffix}",
        "competition": "playground-series-s6e6",
        "model_family": "BLEND",
        "feature_set": f"Ensemble (LGB + Cat + XGB){suffix}",
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
