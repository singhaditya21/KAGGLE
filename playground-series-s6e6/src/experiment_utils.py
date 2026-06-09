#!/usr/bin/env python3
"""Shared experiment tracking and Kaggle API helpers."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def validate_submission(
    path: Path, 
    sample_path: Path, 
    id_col: str = "id", 
    target_col: str = "target"
) -> dict[str, Any]:
    """Validates the structure and value range of a submission file."""
    submission = pd.read_csv(path)
    sample = pd.read_csv(sample_path)
    
    if list(submission.columns) != [id_col, target_col]:
        raise ValueError(f"{path.name} columns must be exactly [{id_col}, {target_col}]")
    if len(submission) != len(sample):
        raise ValueError(f"{path.name} row count {len(submission)} != sample {len(sample)}")
    if not submission[id_col].equals(sample[id_col]):
        raise ValueError(f"{path.name} ID order does not match the sample submission")
    if submission[target_col].isna().any():
        raise ValueError(f"{path.name} contains NaN predictions")
        
    # Check if target column is numeric
    if pd.api.types.is_numeric_dtype(submission[target_col]):
        if not np.isfinite(submission[target_col]).all():
            raise ValueError(f"{path.name} contains non-finite predictions")
        return {
            "rows": int(len(submission)),
            "min": float(submission[target_col].min()),
            "max": float(submission[target_col].max()),
            "mean": float(submission[target_col].mean()),
            "std": float(submission[target_col].std()),
        }
    else:
        # Categorical/string targets
        value_counts = submission[target_col].value_counts().to_dict()
        return {
            "rows": int(len(submission)),
            "value_counts": {str(k): int(v) for k, v in value_counts.items()}
        }


def append_result(root: Path, row: dict[str, Any]) -> Path:
    """Appends an experiment's metrics, configuration, and OOF info to results.csv."""
    experiments_dir = root / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    path = experiments_dir / "results.csv"
    
    normalized = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            normalized[key] = json.dumps(value, sort_keys=True)
        else:
            normalized[key] = value
            
    fieldnames = [
        "timestamp_utc",
        "run_name",
        "competition",
        "model_family",
        "feature_set",
        "oof_score",
        "public_score",
        "fold_scores_json",
        "submission_path",
        "oof_path",
        "config_path",
        "notes",
    ]
    
    for key in normalized:
        if key not in fieldnames:
            fieldnames.append(key)
            
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: normalized.get(key, "") for key in fieldnames})
    return path


def fetch_submission_scores(competition: str) -> pd.DataFrame:
    """Fetches submission history from Kaggle CLI, preserving credentials from env if set."""
    cmd = ["kaggle", "competitions", "submissions", competition, "--csv"]
    
    # Propagate the environment variable if present
    env = os.environ.copy()
    
    try:
        output = subprocess.check_output(cmd, text=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Kaggle API Error: {e.output}")
        return pd.DataFrame()
        
    rows = [line for line in output.splitlines() if line.strip()]
    if not rows:
        return pd.DataFrame()
        
    from io import StringIO
    return pd.read_csv(StringIO("\n".join(rows)))


def update_results_with_public_scores(root: Path, competition: str) -> Path:
    """Fetches latest public scores from Kaggle and updates experiments/results.csv."""
    path = root / "experiments" / "results.csv"
    if not path.exists():
        return path
        
    results = pd.read_csv(path)
    submissions = fetch_submission_scores(competition)
    if submissions.empty:
        return path

    score_by_file = {
        str(row.fileName): row.publicScore
        for row in submissions.itertuples()
        if pd.notna(row.publicScore)
    }
    
    for idx, row in results.iterrows():
        submission_path = str(row.get("submission_path", ""))
        file_name = Path(submission_path).name
        if file_name in score_by_file:
            results.loc[idx, "public_score"] = score_by_file[file_name]
            
    results.to_csv(path, index=False)
    return path
