#!/usr/bin/env python3
"""Check Kaggle daily submission budget usage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import pandas as pd

from experiment_utils import fetch_submission_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Check submission history and daily budget.")
    parser.add_argument("competition", type=str, help="Kaggle competition name")
    parser.add_argument("--limit", type=int, default=5, help="Daily submission limit (default: 5)")
    args = parser.parse_args()

    submissions = fetch_submission_scores(args.competition)
    if submissions.empty:
        print("No submissions found or unable to fetch submission scores.")
        return

    # Check for date columns
    date_col = None
    for col in ["date", "submitted", "submissionDate"]:
        if col in submissions.columns:
            date_col = col
            break
            
    if date_col is None:
        # Just use the first column containing 'date'
        date_cols = [c for c in submissions.columns if "date" in c.lower()]
        if date_cols:
            date_col = date_cols[0]
            
    if date_col:
        submissions["date_parsed"] = pd.to_datetime(submissions[date_col], errors="coerce", utc=True)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent = submissions[submissions["date_parsed"] >= cutoff].copy()
        
        print(f"Recent submissions in last 24h for '{args.competition}':")
        cols_to_print = [c for c in ["fileName", "date", "status", "publicScore"] if c in submissions.columns]
        if "date_parsed" in recent.columns:
            recent["date_str"] = recent["date_parsed"].dt.strftime("%Y-%m-%d %H:%M:%S")
            if "date" in cols_to_print:
                cols_to_print[cols_to_print.index("date")] = "date_str"
                
        print(recent[cols_to_print].to_string(index=False))
        print()
        print(f"Used {len(recent)} submission(s) in the last 24h. Daily limit target: {args.limit}.")
        print(f"Approx remaining now: {max(args.limit - len(recent), 0)}")
    else:
        print("Columns in submissions history:")
        print(submissions.columns.tolist())
        print("Top 5 recent submissions:")
        print(submissions.head(5))


if __name__ == "__main__":
    main()
