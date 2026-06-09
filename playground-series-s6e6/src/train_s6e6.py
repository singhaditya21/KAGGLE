#!/usr/bin/env python3
"""Train a cross-validated model on the Stellar Class (S6E6) dataset with advanced Feature Engineering v3."""

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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from experiment_utils import project_root, append_result, utc_now


def make_qbins(df: pd.DataFrame, col: str, n_bins: int, train_ref: pd.DataFrame | None = None) -> pd.Series:
    ref_df = train_ref if train_ref is not None else df
    probs = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.nanquantile(ref_df[col], probs))
    if len(bins) <= 1:
        return pd.Series(0, index=df.index).astype(str)
    codes = np.searchsorted(bins, df[col].values, side="left") - 1
    codes[df[col].values == bins[0]] = 0
    codes[(df[col].values < bins[0]) | (df[col].values > bins[-1]) | np.isnan(df[col].values)] = -1
    codes = np.clip(codes, -1, len(bins) - 2)
    return pd.Series(codes, index=df.index).astype(str)


def feature_engineering_v3(df: pd.DataFrame, train_ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Creates astronomical color index features, physical flux conversions, curvatures, and quantile binnings."""
    df = df.copy()
    ref_df = train_ref if train_ref is not None else df
    
    # Base colors
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
    
    # Sky coordinates to Cartesian unit sphere
    alpha_rad = np.radians(df["alpha"])
    delta_rad = np.radians(df["delta"])
    df["x"] = np.cos(delta_rad) * np.cos(alpha_rad)
    df["y"] = np.cos(delta_rad) * np.sin(alpha_rad)
    df["z"] = np.sin(delta_rad)
    
    # Trigonometric sky coordinates
    df["alpha_sin"] = np.sin(alpha_rad)
    df["alpha_cos"] = np.cos(alpha_rad)
    df["delta_sin"] = np.sin(delta_rad)
    df["delta_cos"] = np.cos(delta_rad)
    
    # Redshift features
    df["redshift_abs"] = df["redshift"].abs()
    df["redshift_log1p_abs"] = np.log1p(df["redshift_abs"])
    df["redshift_cbrt"] = np.cbrt(df["redshift"])
    df["redshift_is_near_zero"] = (df["redshift_abs"] < 0.005).astype(int)
    df["redshift_is_negative"] = (df["redshift"] < 0).astype(int)
    
    # Statistical aggregates across bands
    bands = ["u", "g", "r", "i", "z"]
    df["band_mean"] = df[bands].mean(axis=1)
    df["band_std"] = df[bands].std(axis=1)
    df["band_max"] = df[bands].max(axis=1)
    df["band_min"] = df[bands].min(axis=1)
    df["band_range"] = df["band_max"] - df["band_min"]
    
    # Curvatures
    df["mag_curvature"] = df["u"] - 2 * df["r"] + df["z"]
    df["blue_curvature"] = df["u"] - 2 * df["g"] + df["r"]
    df["red_curvature"] = df["r"] - 2 * df["i"] + df["z"]
    
    # Polar Color coordinates
    df["color_radius_ug_gr"] = np.sqrt(df["u_g"]**2 + df["g_r"]**2)
    df["color_angle_ug_gr"] = np.arctan2(df["u_g"], df["g_r"])
    df["color_radius_ri_iz"] = np.sqrt(df["r_i"]**2 + df["i_z"]**2)
    df["color_angle_ri_iz"] = np.arctan2(df["r_i"], df["i_z"])
    
    # Flux conversions
    for b in bands:
        clipped = np.clip(df[b].values, -30, 30)
        df[f"flux_{b}"] = np.power(10.0, -0.4 * clipped)
        
    flux_cols = [f"flux_{b}" for b in bands]
    df["flux_mean"] = df[flux_cols].mean(axis=1)
    df["flux_std"] = df[flux_cols].std(axis=1)
    df["flux_max"] = df[flux_cols].max(axis=1)
    df["flux_min"] = df[flux_cols].min(axis=1)
    df["flux_range"] = df["flux_max"] - df["flux_min"]
    
    # Categoricals calculated
    df["spectral_type_calc"] = pd.cut(
        df["g_r"],
        bins=[-np.inf, 0.0, 0.5, 1.0, np.inf],
        labels=["O/B", "A/F", "G/K", "M"]
    ).astype(str)
    df["galaxy_population_calc"] = np.where(df["u_r"] > 2.20, "Red_Sequence", "Blue_Cloud")
    df["spectral_x_pop"] = df["spectral_type_calc"] + "__" + df["galaxy_population_calc"]
    
    # Quantile binning & interactions
    for col, bins in [("alpha", 64), ("delta", 64), ("u_g", 64), ("g_r", 64), ("redshift", 64), ("band_mean", 64)]:
        df[f"{col}_qbin{bins}"] = make_qbins(df, col, bins, ref_df)
        
    df["alpha_delta_qbin"] = df["alpha_qbin64"] + "__" + df["delta_qbin64"]
    df["ug_gr_qbin"] = df["u_g_qbin64"] + "__" + df["g_r_qbin64"]
    df["redshift_mean_qbin"] = df["redshift_qbin64"] + "__" + df["band_mean_qbin64"]
    
    # Frequency encoding for categoricals
    cat_cols = [
        "spectral_type_calc", "galaxy_population_calc", "spectral_x_pop",
        "alpha_delta_qbin", "ug_gr_qbin", "redshift_mean_qbin"
    ]
    for c in cat_cols:
        vc = ref_df[c].value_counts()
        df[f"{c}_freq"] = df[c].map(vc).fillna(0).astype("float32")
        df[f"{c}_freq_log"] = np.log1p(df[f"{c}_freq"])
        
    # Replace infs and fill NaNs with median
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.select_dtypes(include=[np.number]).columns:
        median_val = ref_df[c].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[c] = df[c].fillna(median_val)
        
    return df


def add_target_encoding(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    cat_cols: list[str], 
    y_train: np.ndarray, 
    n_splits: int = 5, 
    seed: int = 42, 
    smooth: float = 20.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies out-of-fold target encoding to categorical variables to prevent label leakage."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Add TE columns to both
    for col in cat_cols:
        for c in range(3):
            train_df[f"TE_{col}_class_{c}"] = 0.0
            test_df[f"TE_{col}_class_{c}"] = 0.0
            
    # Priors
    priors = np.bincount(y_train, minlength=3) / len(y_train)
    
    # Out of Fold TE for Train
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    for tr_idx, val_idx in skf.split(train_df, y_train):
        X_tr = train_df.iloc[tr_idx]
        y_tr = y_train[tr_idx]
        
        for col in cat_cols:
            counts = X_tr[col].value_counts()
            
            for c in range(3):
                hits = X_tr[col][y_tr == c].value_counts()
                cnts = counts.reindex(hits.index).fillna(0)
                probs = (hits + priors[c] * smooth) / (cnts + smooth)
                
                train_df.iloc[val_idx, train_df.columns.get_loc(f"TE_{col}_class_{c}")] = \
                    train_df.iloc[val_idx][col].map(probs).fillna(priors[c])
                    
    # Target Encoding for Test (using full Train statistics)
    for col in cat_cols:
        counts = train_df[col].value_counts()
        for c in range(3):
            hits = train_df[col][y_train == c].value_counts()
            cnts = counts.reindex(hits.index).fillna(0)
            probs = (hits + priors[c] * smooth) / (cnts + smooth)
            
            test_df[f"TE_{col}_class_{c}"] = test_df[col].map(probs).fillna(priors[c])
            
    return train_df, test_df


def train_s6e6(
    model_type: str = "lightgbm", 
    config_path: Path | None = None, 
    use_original: bool = False
) -> None:
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
            "verbose": False,
            "thread_count": -1
        },
        "et_params": {
            "n_estimators": 500,
            "max_depth": 15,
            "min_samples_leaf": 4,
            "random_state": 42,
            "n_jobs": -1
        },
        "histgb_params": {
            "max_iter": 500,
            "learning_rate": 0.05,
            "max_depth": 8,
            "random_state": 42
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
                    
    if use_original:
        config["run_name"] = f"{config['run_name']}_with_original"
                    
    print(f"Starting {model_type.upper()} pipeline for run: {config['run_name']}")
    
    train_path = root / "data" / "raw" / "train.csv"
    test_path = root / "data" / "raw" / "test.csv"
    
    if not train_path.exists() or not test_path.exists():
        print("Error: train.csv or test.csv not found.")
        return
        
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    if use_original:
        orig_path = root / "data" / "external" / "star_classification.csv"
        if orig_path.exists():
            print("Loading original SDSS17 dataset...")
            orig = pd.read_csv(orig_path)
            orig_cols = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift", "class"]
            orig = orig[orig_cols].copy()
            orig["id"] = -1
            train = pd.concat([train, orig], ignore_index=True)
            print(f"Appended {len(orig)} original rows. Combined train shape: {train.shape}")
        else:
            print(f"Warning: Original dataset not found at {orig_path}. Proceeding with synthetic data only.")
            
    # Feature Engineering
    print("Engineering features...")
    train = feature_engineering_v3(train)
    test = feature_engineering_v3(test, train_ref=train)
    
    # Target Encoding
    te_cols = ["alpha_delta_qbin", "ug_gr_qbin", "redshift_mean_qbin"]
    le = LabelEncoder()
    y = le.fit_transform(train["class"])
    print(f"Target classes mapped: {dict(zip(le.classes_, range(3)))}")
    
    print("Applying target encoding...")
    train, test = add_target_encoding(train, test, te_cols, y, n_splits=config["n_splits"], seed=config["seed"])
    
    # Define features
    exclude_cols = [
        "id", "class", "spectral_type", "galaxy_population", 
        "spectral_type_calc", "galaxy_population_calc", "spectral_x_pop", 
        "alpha_delta_qbin", "ug_gr_qbin", "redshift_mean_qbin",
        "alpha_qbin64", "delta_qbin64", "u_g_qbin64", "g_r_qbin64", 
        "redshift_qbin64", "band_mean_qbin64"
    ]
    features = [c for c in train.columns if c not in exclude_cols]
    
    X = train[features].copy()
    X_test = test[features].copy()
    
    # We convert categories to integer codes for models that require numeric features (XGBoost, ET, HistGB)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]) or X[c].dtype.name == "category":
            X[c] = X[c].astype("category").cat.codes
            X_test[c] = X_test[c].astype("category").cat.codes
            
    oof_preds = np.zeros((len(train), 3))
    test_preds = np.zeros((len(test), 3))
    
    skf = StratifiedKFold(n_splits=config["n_splits"], shuffle=True, random_state=config["seed"])
    fold_scores = []
    
    # Define sample weights
    sample_weight = np.ones(len(train))
    if use_original:
        is_original = (train["id"] == -1).values
        sample_weight[is_original] = 0.08
        print(f"Applying sample weights: synthetic=1.0, original=0.08")
        
    print(f"Training {model_type.upper()} with {config['n_splits']}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        w_train = sample_weight[train_idx]
        
        if model_type == "lightgbm":
            model = lgb.LGBMClassifier(**config["lgb_params"])
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )
            val_fold_preds = model.predict_proba(X_val)
            test_fold_preds = model.predict_proba(X_test)
            
        elif model_type == "xgboost":
            model = xgb.XGBClassifier(**config["xgb_params"])
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            val_fold_preds = model.predict_proba(X_val)
            test_fold_preds = model.predict_proba(X_test)
            
        elif model_type == "catboost":
            model = CatBoostClassifier(**config["cat_params"])
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
            val_fold_preds = model.predict_proba(X_val)
            test_fold_preds = model.predict_proba(X_test)
            
        elif model_type == "extratrees":
            model = ExtraTreesClassifier(**config["et_params"])
            model.fit(X_train, y_train, sample_weight=w_train)
            val_fold_preds = model.predict_proba(X_val)
            test_fold_preds = model.predict_proba(X_test)
            
        elif model_type == "histgb":
            model = HistGradientBoostingClassifier(**config["histgb_params"])
            model.fit(X_train, y_train, sample_weight=w_train)
            val_fold_preds = model.predict_proba(X_val)
            test_fold_preds = model.predict_proba(X_test)
            
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
        "feature_set": f"Feature v3 ({len(features)} features: flux, curvatures, polar colors, qbins, TE, freq)",
        "oof_score": cv_score,
        "fold_scores_json": fold_scores,
        "submission_path": str(sub_path),
        "oof_path": str(oof_path),
        "config_path": str(config_path) if config_path else "",
        "notes": f"Model {model_type} run on Stellar Class (S6E6) with Feature v3.",
    }
    append_result(root, result_row)
    print("Experiment logged successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model on S6E6 Stellar Class dataset with Feature v3.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="lightgbm", 
        choices=["lightgbm", "xgboost", "catboost", "extratrees", "histgb"],
        help="Model type to train"
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config JSON file")
    parser.add_argument(
        "--use-original", 
        action="store_true", 
        help="Append the original SDSS17 dataset to training data"
    )
    args = parser.parse_args()
    
    train_s6e6(args.model, args.config, args.use_original)


if __name__ == "__main__":
    main()
