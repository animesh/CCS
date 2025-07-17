import pandas as pd
import argparse
import os
from pathlib import Path


def find_evidence_txts(root_dir):
    """Recursively find all evidence.txt files under root_dir."""
    root = Path(root_dir)
    return list(root.rglob('evidence.txt'))


def concat_evidence_txts(input_files, output_file):
    dfs = []
    for i, file in enumerate(input_files):
        print(f"Reading {file} ...")
        df = pd.read_csv(file, sep='\t')
        dfs.append(df)
    print(f"Concatenating {len(dfs)} files ...")
    concat_df = pd.concat(dfs, ignore_index=True)
    print(f"Writing concatenated file to {output_file} ...")
    concat_df.to_csv(output_file, index=False)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Concatenate all evidence.txt files in a directory tree into a single CSV.")
    parser.add_argument('input_dir', help='Path to root directory containing evidence.txt files in subdirectories')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
    args = parser.parse_args()

    # Find all evidence.txt files
    evidence_files = find_evidence_txts(args.input_dir)
    if not evidence_files:
        raise FileNotFoundError(f"No evidence.txt files found in {args.input_dir}")
    print(f"Found {len(evidence_files)} evidence.txt files.")

    concat_evidence_txts(evidence_files, args.output)


if __name__ == "__main__":
    main() 