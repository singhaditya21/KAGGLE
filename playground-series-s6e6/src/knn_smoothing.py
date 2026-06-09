#!/usr/bin/env python3
"""KNN Graph Probability Smoothing for Stellar Classification (S6E6) Stacking."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

def main():
    root = Path(__file__).resolve().parents[1]
    
    # Load raw datasets
    train_path = root / "data" / "raw" / "train.csv"
    test_path = root / "data" / "raw" / "test.csv"
    
    if not train_path.exists() or not test_path.exists():
        print("Error: train.csv or test.csv not found.")
        return
        
    print("Loading datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    le = LabelEncoder()
    y = le.fit_transform(train["class"])
    
    # 1. Feature Engineering for KNN Distance Metric
    # We use physical properties: colors, redshift, and unit sphere coordinates
    print("Engineering distance features...")
    
    def get_knn_features(df):
        features = pd.DataFrame(index=df.index)
        
        # Colors
        features["u_g"] = df["u"] - df["g"]
        features["g_r"] = df["g"] - df["r"]
        features["r_i"] = df["r"] - df["i"]
        features["i_z"] = df["i"] - df["z"]
        features["u_r"] = df["u"] - df["r"]
        features["g_i"] = df["g"] - df["i"]
        
        # Redshift
        features["redshift"] = df["redshift"]
        features["redshift_abs"] = df["redshift"].abs()
        
        # Coordinates on unit sphere
        alpha_rad = np.radians(df["alpha"])
        delta_rad = np.radians(df["delta"])
        features["x"] = np.cos(delta_rad) * np.cos(alpha_rad)
        features["y"] = np.cos(delta_rad) * np.sin(alpha_rad)
        features["z"] = np.sin(delta_rad)
        
        return features

    X_train_knn = get_knn_features(train)
    X_test_knn = get_knn_features(test)
    
    # Standardize features for isotropic Euclidean distance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_knn)
    X_test_scaled = scaler.transform(X_test_knn)
    
    # 2. Load Stacked Predictions
    oof_dir = root / "models" / "oof"
    oof_path = oof_dir / 'oof_stacked.npy'
    test_preds_path = oof_dir / 'test_preds_stacked.npy'
    
    if not oof_path.exists() or not test_preds_path.exists():
        print("Error: Stacked predictions not found. Please run the stacker first.")
        return
        
    oof_stacked = np.load(oof_path)
    test_preds_stacked = np.load(test_preds_path)
    
    baseline_score = balanced_accuracy_score(y, np.argmax(oof_stacked, axis=1))
    print(f"\nBaseline Stacked CV Balanced Accuracy: {baseline_score:.6f}")
    
    # 3. Optimize KNN Smoothing on OOF Predictions
    # We find K+1 neighbors so we can exclude the point itself (index 0 is the query point itself)
    print("\nFinding nearest neighbors in training set...")
    nn_train = NearestNeighbors(n_neighbors=51, metric='euclidean', n_jobs=-1)
    nn_train.fit(X_train_scaled)
    
    print("Querying neighbors for training set (OOF cross-validation)...")
    distances, indices = nn_train.kneighbors(X_train_scaled)
    
    best_score = baseline_score
    best_k = 0
    best_lambda = 0.0
    
    # Grid search over K (number of neighbors) and lambda (blending weight)
    k_choices = [3, 5, 10, 15, 20, 30, 50]
    lambda_choices = np.linspace(0.0, 0.4, 21)
    
    print("\nGrid searching K and lambda:")
    for k in k_choices:
        # Get neighbor labels excluding self (index 0)
        # indices shape is (N, K+1)
        neighbor_indices = indices[:, 1:k+1] # shape (N, k)
        neighbor_labels = y[neighbor_indices] # shape (N, k)
        
        # Calculate neighborhood class probability distribution
        # For each sample, count label occurrences
        N = len(y)
        P_knn = np.zeros((N, 3))
        for c in range(3):
            P_knn[:, c] = np.sum(neighbor_labels == c, axis=1) / k
            
        for lam in lambda_choices:
            # Blend stacked OOF with KNN OOF
            blended_oof = (1.0 - lam) * oof_stacked + lam * P_knn
            score = balanced_accuracy_score(y, np.argmax(blended_oof, axis=1))
            
            if score > best_score:
                best_score = score
                best_k = k
                best_lambda = lam
                print(f"  -> NEW BEST: K = {k:<2} | lambda = {lam:.2f} | CV Balanced Acc: {score:.6f} (Improvement: {score - baseline_score:+.6f})")
                
    print(f"\nOptimized Parameters: K = {best_k}, lambda = {best_lambda:.2f}")
    print(f"Optimized OOF CV Balanced Accuracy: {best_score:.6f}")
    
    if best_lambda == 0.0:
        print("\nKNN Smoothing did not improve CV score. Keeping baseline stacked predictions.")
        return
        
    # 4. Apply KNN Smoothing to Test Predictions
    print("\nQuerying neighbors for test set...")
    nn_test = NearestNeighbors(n_neighbors=best_k, metric='euclidean', n_jobs=-1)
    nn_test.fit(X_train_scaled)
    
    test_distances, test_indices = nn_test.kneighbors(X_test_scaled)
    
    # Get neighbor labels
    test_neighbor_labels = y[test_indices] # shape (M, best_k)
    
    # Compute neighborhood class probability distribution
    M = len(test)
    P_knn_test = np.zeros((M, 3))
    for c in range(3):
        P_knn_test[:, c] = np.sum(test_neighbor_labels == c, axis=1) / best_k
        
    # Apply optimal blend
    test_preds_smoothed = (1.0 - best_lambda) * test_preds_stacked + best_lambda * P_knn_test
    
    # 5. Apply Decision Threshold Multipliers from the stacker
    # We load the multipliers if we optimized them, or just use the argmax of the blended predictions
    # Wait, let's load the stacked submission file to check the class multipliers or just apply the same class multipliers
    # In task-1705, the class multipliers optimized were: [1.0, 1.045, 0.87]
    # Let's search if we can optimize decision multipliers on the blended OOF predictions!
    print("\nOptimizing decision thresholds on blended predictions...")
    blended_oof_optimal = (1.0 - best_lambda) * oof_stacked + best_lambda * P_knn
    best_weights = np.array([1.0, 1.0, 1.0])
    best_final_score = best_score
    
    for w1 in np.linspace(0.80, 1.20, 81):
        for w2 in np.linspace(0.80, 1.20, 81):
            w = np.array([1.0, w1, w2])
            preds = np.argmax(blended_oof_optimal * w, axis=1)
            score = balanced_accuracy_score(y, preds)
            if score > best_final_score:
                best_final_score = score
                best_weights = w
                
    print(f"Optimized Blended + Threshold CV Balanced Acc: {best_final_score:.6f} with multipliers: {best_weights}")
    
    # Save OOF and test predictions of the smoothed stacker
    np.save(oof_dir / 'oof_stacked_smoothed.npy', blended_oof_optimal)
    np.save(oof_dir / 'test_preds_stacked_smoothed.npy', test_preds_smoothed)
    
    # Apply multipliers to smoothed test predictions
    test_preds_weighted = test_preds_smoothed * best_weights
    test_classes = np.argmax(test_preds_weighted, axis=1)
    
    # Save submission
    sub = pd.DataFrame({
        "id": test["id"],
        "class": le.inverse_transform(test_classes)
    })
    
    sub_path = root / "submissions" / "submission_stacked_smoothed.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved smoothed ensembled submission to {sub_path}")
    print("\nSubmission value counts:")
    print(sub["class"].value_counts())

if __name__ == "__main__":
    main()
