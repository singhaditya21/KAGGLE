#!/usr/bin/env python3
"""Validate a submission file against a sample submission template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_utils import project_root, validate_submission


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Validate a submission file structure.")
    parser.add_argument("submission", type=Path, help="Path to the submission CSV file")
    parser.add_argument(
        "--sample", 
        type=Path, 
        default=root / "data" / "raw" / "sample_submission.csv", 
        help="Path to the sample submission template"
    )
    parser.add_argument("--id-col", type=str, default="id", help="Name of the ID column")
    parser.add_argument("--target-col", type=str, default="target", help="Name of the target/prediction column")
    args = parser.parse_args()

    if not args.submission.exists():
        print(f"Error: Submission file {args.submission} does not exist.")
        return

    if not args.sample.exists():
        print(f"Error: Sample submission template {args.sample} does not exist.")
        print("Please download competition data first or supply the path via --sample.")
        return

    try:
        stats = validate_submission(args.submission, args.sample, args.id_col, args.target_col)
        print("Validation PASSED.")
        print(json.dumps(stats, indent=2))
    except Exception as e:
        print("Validation FAILED:")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
