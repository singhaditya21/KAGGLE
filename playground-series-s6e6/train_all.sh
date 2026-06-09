#!/bin/bash
# Train all 5 model families sequentially on Feature v3
set -e
export PYTHONUNBUFFERED=1

echo "=== Training LightGBM ==="
uv run python src/train_s6e6.py --model lightgbm

echo "=== Training XGBoost ==="
uv run python src/train_s6e6.py --model xgboost

echo "=== Training CatBoost ==="
uv run python src/train_s6e6.py --model catboost

echo "=== Training ExtraTrees ==="
uv run python src/train_s6e6.py --model extratrees

echo "=== Training HistGradientBoosting ==="
uv run python src/train_s6e6.py --model histgb

echo "=== All models trained successfully! ==="
