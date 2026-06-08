#!/usr/bin/env python3
"""Download and extract Kaggle competition data files."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from experiment_utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and unzip Kaggle competition files.")
    parser.add_argument("competition", type=str, help="The Kaggle competition name (e.g. playground-series-s6e5)")
    parser.add_argument(
        "--dest", 
        type=Path, 
        default=project_root() / "data" / "raw", 
        help="Destination directory for raw data files"
    )
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading competition data for '{args.competition}' into {args.dest.relative_to(project_root())}...")
    
    # Run kaggle command
    cmd = [
        "kaggle",
        "competitions",
        "download",
        args.competition,
        "-p",
        str(args.dest)
    ]
    
    env = os.environ.copy()
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading data: {e}")
        return

    # Check for downloaded zip file
    zip_path = args.dest / f"{args.competition}.zip"
    if zip_path.exists():
        print(f"Unzipping {zip_path.name}...")
        unzip_cmd = ["unzip", "-o", str(zip_path), "-d", str(args.dest)]
        try:
            subprocess.run(unzip_cmd, check=True)
            print("Unzipped files successfully.")
            # Remove zip to clean up space
            os.remove(zip_path)
            print(f"Removed temporary archive {zip_path.name}.")
        except subprocess.CalledProcessError as e:
            print(f"Error unzipping file: {e}")
    else:
        # Check if there are other zip files downloaded
        zips = list(args.dest.glob("*.zip"))
        if zips:
            for z in zips:
                print(f"Unzipping {z.name}...")
                subprocess.run(["unzip", "-o", str(z), "-d", str(args.dest)], check=True)
                os.remove(z)
        else:
            print("No zip file found. Files might have been downloaded directly.")
            
    print("Data download and extraction complete.")


if __name__ == "__main__":
    main()
