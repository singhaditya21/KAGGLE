#!/usr/bin/env python3
"""Generate pseudo-labels from high-confidence test predictions."""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels from stacked test predictions.")
    parser.add_argument("--threshold", type=float, default=0.999, help="Probability threshold for pseudo-labeling")
    parser.add_argument("--preds-file", type=str, default="test_preds_stacked_smoothed.npy", help="Test predictions numpy file")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    test_path = root / "data" / "raw" / "test.csv"
    train_path = root / "data" / "raw" / "train.csv"
    preds_path = root / "models" / "oof" / args.preds_file

    if not test_path.exists() or not preds_path.exists() or not train_path.exists():
        print("Error: test.csv, train.csv, or predictions file not found.")
        return

    print(f"Loading test dataset from {test_path}...")
    test = pd.read_csv(test_path)
    print(f"Loading train dataset from {train_path} to fit LabelEncoder...")
    train = pd.read_csv(train_path)
    
    le = LabelEncoder()
    le.fit(train["class"])
    print(f"Mapped classes: {list(le.classes_)}")

    print(f"Loading predictions from {preds_path}...")
    preds = np.load(preds_path)
    if preds.ndim == 3:
        preds = preds.mean(axis=0)

    assert len(test) == len(preds), f"Test size ({len(test)}) and prediction size ({len(preds)}) mismatch!"

    # Get max probability and class indices
    max_probs = np.max(preds, axis=1)
    pred_classes = np.argmax(preds, axis=1)

    # Filter by threshold
    mask = max_probs >= args.threshold
    pseudo_count = np.sum(mask)
    print(f"\nFound {pseudo_count:,} out of {len(test):,} test rows with confidence >= {args.threshold} ({pseudo_count / len(test):.2%})")

    if pseudo_count == 0:
        print("No pseudo-labels generated. Try lowering the threshold.")
        return

    # Extract pseudo-labeled rows
    pseudo_df = test[mask].copy()
    pseudo_df["class"] = le.inverse_transform(pred_classes[mask])
    pseudo_df["confidence"] = max_probs[mask]

    # Print distribution
    print("\nPseudo-label class distribution:")
    print(pseudo_df["class"].value_counts())

    # Save to CSV
    out_path = root / "data" / "raw" / "pseudo_labels.csv"
    pseudo_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(pseudo_df):,} pseudo-labeled rows to {out_path.relative_to(root)}")

if __name__ == "__main__":
    main()
