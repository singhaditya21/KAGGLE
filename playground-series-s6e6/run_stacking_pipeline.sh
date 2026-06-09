#!/bin/bash
# Stacking & ensembling pipeline with decision threshold optimization
set -e
export PYTHONUNBUFFERED=1

echo "Waiting for local baseline training pipeline (task-1130) to complete..."
# Wait while train_s6e6.py is running without --use-original
while ps aux | grep "train_s6e6.py" | grep -v grep | grep -q -v "use-original"; do
    sleep 15
done
echo "Baseline training pipeline completed!"

# Run the 11-model stacker
echo "=== Running Stacker on 11 baseline models ==="
uv run python src/stacker.py --exclude-original

# Train the original-dataset augmented models with sample weighting
echo "=== Training LightGBM with original dataset + sample weighting ==="
uv run python src/train_s6e6.py --model lightgbm --use-original

echo "=== Training XGBoost with original dataset + sample weighting ==="
uv run python src/train_s6e6.py --model xgboost --use-original

# Run the 13-model stacker
echo "=== Running Stacker on 13 models (including original-data augmented models) ==="
uv run python src/stacker.py

echo "=== Stacking Pipeline Completed Successfully! ==="
