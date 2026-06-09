#!/usr/bin/env python3
"""Multi-class Logistic Regression Stacker using scikit-learn."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

def prob_to_logit(p, eps=1e-15, logit_clip=30.0):
    p = np.clip(p, eps, 1.0 - eps).astype(np.float64)
    return np.clip(np.log(p / (1.0 - p)), -logit_clip, logit_clip).astype(np.float32)

def load_preds(path, expected_rows=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            vals = df.iloc[:, 0].values
            if expected_rows is not None:
                assert len(vals) == expected_rows * 3
            return vals.reshape(-1, 3)
        return df.iloc[:, -3:].values[:expected_rows]
        
    elif path.suffix == ".npy":
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr.mean(axis=0)
        return arr[:expected_rows]
        
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stack OOF predictions.")
    parser.add_argument(
        "--exclude-original", 
        action="store_true", 
        help="Exclude models trained with the original SDSS17 dataset"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    
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
    N = len(y)
    M = len(test)
    
    print(f"Train size: {N:,} | Test size: {M:,}")
    
    # Define models to stack
    # Format: (name, oof_path, test_path)
    cdeotte_dir = root / "data" / "external" / "cdeotte_preds"
    oof_dir = root / "models" / "oof"
    
    models_to_stack = [
        # Chris Deotte's community models
        ('realmlp_v12', cdeotte_dir / 'oof_preds_realmlp0_v12.csv', cdeotte_dir / 'test_preds_realmlp0_v12.csv'),
        ('tabm_v2', cdeotte_dir / 'oof_preds_tabm0_v2.csv', cdeotte_dir / 'test_preds_tabm0_v2.csv'),
        ('realmlp_v10', cdeotte_dir / 'oof_preds_realmlp2_v10.csv', cdeotte_dir / 'test_preds_realmlp2_v10.csv'),
        ('lgbm_v1', cdeotte_dir / 'oof_preds_lgbm5_v1.csv', cdeotte_dir / 'test_preds_lgbm5_v1.csv'),
        ('xgb_v1', cdeotte_dir / 'oof_final_xgb6_v1.csv', cdeotte_dir / 'test_final_xgb6_v1.csv'),
        ('tabm_v1', cdeotte_dir / 'oof_final_tabm1_v1.csv', cdeotte_dir / 'test_final_tabm1_v1.csv'),
        
        # Our own local models
        ('our_lgb', oof_dir / 'oof_lightgbm_stellar_baseline.npy', oof_dir / 'test_preds_lightgbm_stellar_baseline.npy'),
        ('our_xgb', oof_dir / 'oof_xgboost_stellar_baseline.npy', oof_dir / 'test_preds_xgboost_stellar_baseline.npy'),
        ('our_cat', oof_dir / 'oof_catboost_stellar_baseline.npy', oof_dir / 'test_preds_catboost_stellar_baseline.npy'),
        ('our_et', oof_dir / 'oof_extratrees_stellar_baseline.npy', oof_dir / 'test_preds_extratrees_stellar_baseline.npy'),
        ('our_histgb', oof_dir / 'oof_histgb_stellar_baseline.npy', oof_dir / 'test_preds_histgb_stellar_baseline.npy'),
    ]
    
    # Optional local models with original dataset
    if not args.exclude_original:
        optional_models = [
            ('our_lgb_orig', oof_dir / 'oof_lightgbm_stellar_baseline_with_original.npy', oof_dir / 'test_preds_lightgbm_stellar_baseline_with_original.npy'),
            ('our_xgb_orig', oof_dir / 'oof_xgboost_stellar_baseline_with_original.npy', oof_dir / 'test_preds_xgboost_stellar_baseline_with_original.npy'),
        ]
        
        for name, oof_p, test_p in optional_models:
            if oof_p.exists() and test_p.exists():
                models_to_stack.append((name, oof_p, test_p))
            
    loaded_oofs = []
    loaded_tests = []
    model_names = []
    
    print("\nLoading models and converting to logits:")
    for name, oof_p, test_p in models_to_stack:
        try:
            # Load OOF (limit to N rows if using model with original dataset)
            o = load_preds(oof_p, expected_rows=N)
            t = load_preds(test_p, expected_rows=M)
            
            # Convert to logits
            o_logit = prob_to_logit(o)
            t_logit = prob_to_logit(t)
            
            # Verify shapes
            assert o_logit.shape == (N, 3), f"OOF shape {o_logit.shape} != {(N, 3)}"
            assert t_logit.shape == (M, 3), f"Test shape {t_logit.shape} != {(M, 3)}"
            
            # Calculate and print individual score
            score = balanced_accuracy_score(y, np.argmax(o, axis=1))
            print(f"  {name:<15} | CV Balanced Acc: {score:.6f}")
            
            loaded_oofs.append(o_logit)
            loaded_tests.append(t_logit)
            model_names.append(name)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            
    if not loaded_oofs:
        print("Error: No models were successfully loaded.")
        return
        
    X_oof = np.concatenate(loaded_oofs, axis=1)
    X_test = np.concatenate(loaded_tests, axis=1)
    print(f"\nStacked features shape: OOF = {X_oof.shape}, Test = {X_test.shape}")
    
    # Stacking cross-validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Try different values of C
    best_c = 1.0
    best_cv_score = 0.0
    
    c_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    print("\nTuning regularization strength C:")
    for C in c_values:
        oof_preds = np.zeros((N, 3))
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_oof, y)):
            X_tr, y_tr = X_oof[tr_idx], y[tr_idx]
            X_val, y_val = X_oof[val_idx], y[val_idx]
            
            clf = LogisticRegression(
                solver='lbfgs',
                C=C,
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
            clf.fit(X_tr, y_tr)
            oof_preds[val_idx] = clf.predict_proba(X_val)
            
        score = balanced_accuracy_score(y, np.argmax(oof_preds, axis=1))
        print(f"  C = {C:<6} | CV Balanced Acc: {score:.6f}")
        if score > best_cv_score:
            best_cv_score = score
            best_c = C
            
    print(f"\nBest C: {best_c} with CV Balanced Acc: {best_cv_score:.6f}")
    
    # Generate test predictions using best C
    oof_final = np.zeros((N, 3))
    test_final = np.zeros((M, 3))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_oof, y)):
        X_tr, y_tr = X_oof[tr_idx], y[tr_idx]
        X_val, y_val = X_oof[val_idx], y[val_idx]
        
        clf = LogisticRegression(
            solver='lbfgs',
            C=best_c,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X_tr, y_tr)
        oof_final[val_idx] = clf.predict_proba(X_val)
        test_final += clf.predict_proba(X_test) / n_splits
        
    overall_score = balanced_accuracy_score(y, np.argmax(oof_final, axis=1))
    print(f"Overall Stacked CV Balanced Acc (Argmax): {overall_score:.6f}")
    
    # Threshold Optimization for Balanced Accuracy
    print("\nOptimizing decision thresholds for Balanced Accuracy...")
    best_weights = np.array([1.0, 1.0, 1.0])
    best_score = overall_score
    
    # Grid search for multipliers around 1.0
    for w1 in np.linspace(0.85, 1.15, 61):
        for w2 in np.linspace(0.85, 1.15, 61):
            w = np.array([1.0, w1, w2])
            preds = np.argmax(oof_final * w, axis=1)
            score = balanced_accuracy_score(y, preds)
            if score > best_score:
                best_score = score
                best_weights = w
                
    print(f"Optimized CV Balanced Acc: {best_score:.6f} with class multipliers: {best_weights}")
    
    # Save OOF and test predictions of the stacker
    np.save(oof_dir / 'oof_stacked.npy', oof_final)
    np.save(oof_dir / 'test_preds_stacked.npy', test_final)
    print(f"Saved stacked predictions to {oof_dir.relative_to(root)}")
    
    # Apply optimized weights to test predictions
    test_preds_weighted = test_final * best_weights
    test_classes = np.argmax(test_preds_weighted, axis=1)
    
    # Save submission
    sub = pd.DataFrame({
        "id": test["id"],
        "class": le.inverse_transform(test_classes)
    })
    sub_path = root / "submissions" / "submission_stacked.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved ensembled submission to {sub_path}")
    print("\nSubmission value counts:")
    print(sub["class"].value_counts())

if __name__ == "__main__":
    main()
