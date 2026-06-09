#!/bin/bash
# Pipeline to train XGBoost, CatBoost, and HistGB with original + pseudo labels,
# then run ensembling and KNN smoothing.
set -e
export PYTHONUNBUFFERED=1

echo "=== 1. Training XGBoost (Original + Pseudo) ==="
uv run python src/train_s6e6.py --model xgboost --use-original --use-pseudo --pseudo-weight 0.2

echo "=== 2. Training CatBoost (Original + Pseudo) ==="
uv run python src/train_s6e6.py --model catboost --use-original --use-pseudo --pseudo-weight 0.2

echo "=== 3. Training HistGradientBoosting (Original + Pseudo) ==="
uv run python src/train_s6e6.py --model histgb --use-original --use-pseudo --pseudo-weight 0.2

echo "=== 4. Running 19-Model Stacker ==="
uv run python src/stacker.py

echo "=== 5. Running KNN Spatial Probability Smoothing ==="
uv run python src/knn_smoothing.py

echo "=== Pipeline Completed Successfully! ==="
