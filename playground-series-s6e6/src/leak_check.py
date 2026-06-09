import pandas as pd
import numpy as np
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    train_path = root / "data" / "raw" / "train.csv"
    test_path = root / "data" / "raw" / "test.csv"
    orig_path = root / "data" / "external" / "star_classification.csv"
    
    if not orig_path.exists():
        print(f"Original dataset not found at {orig_path}")
        return
        
    print("Loading datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    orig = pd.read_csv(orig_path)
    
    print(f"Train size: {len(train)}")
    print(f"Test size: {len(test)}")
    print(f"Original SDSS17 size: {len(orig)}")
    
    # Check for exact matches on alpha and delta coordinates
    print("\nChecking exact coordinate (alpha, delta) matches between test and original...")
    test_coords = test[['alpha', 'delta']].copy()
    orig_coords = orig[['alpha', 'delta', 'class']].copy()
    
    # Round to ensure precision limits don't prevent match
    test_coords['alpha_round'] = test_coords['alpha'].round(6)
    test_coords['delta_round'] = test_coords['delta'].round(6)
    orig_coords['alpha_round'] = orig_coords['alpha'].round(6)
    orig_coords['delta_round'] = orig_coords['delta'].round(6)
    
    merged_exact = pd.merge(test_coords, orig_coords, on=['alpha', 'delta'], how='inner')
    print(f"Exact alpha-delta matches: {len(merged_exact)}")
    
    merged_round = pd.merge(test_coords, orig_coords, on=['alpha_round', 'delta_round'], how='inner')
    print(f"Rounded alpha-delta matches (6 decimals): {len(merged_round)}")
    
    # Check for duplicate coordinates in orig
    orig_dupes = orig_coords.duplicated(subset=['alpha_round', 'delta_round']).sum()
    print(f"Duplicate coordinates in original: {orig_dupes}")

if __name__ == "__main__":
    main()
