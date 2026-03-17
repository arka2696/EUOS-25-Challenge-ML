#!/usr/bin/env python3
"""
inspect_merged_euos25.py

Helper script to understand and visualize what the EUOS25 merge script did.

It will:
- Load the four original training CSVs
- Load the merged training CSV
- Print basic summaries (shapes, ID counts, overlaps)
- Reconstruct a merged DataFrame from the four inputs (without canonicalizing)
  and compare it to the provided merged_train.csv
- Save:
    - A small "before vs after" sample table for a few IDs
    - Histograms for each task label (from merged_train.csv)
    - A correlation heatmap of the four task labels

Usage example:

    python inspect_merged_euos25.py \
        --train_fluor480 data/euos25_challenge_train_fluorescence480.csv \
        --train_fluor340 data/euos25_challenge_train_fluorescence340_450.csv \
        --train_trans450 data/euos25_challenge_train_transmittance450.csv \
        --train_trans340 data/euos25_challenge_train_transmittance340.csv \
        --merged_train merged_train.csv \
        --output_dir merge_report

"""

import argparse
import os
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_standardize(
    path: str,
    id_col: str = "ID",
    smiles_col: str = "SMILES",
    label_col: str = "",
    new_label_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Same logic as in merge_euos25_csv.py, but without RDKit / canonicalization.

    - Infers ID, SMILES, label columns if needed.
    - Renames them to: ID, SMILES, <new_label_name>
    """
    df = pd.read_csv(path)

    # Infer ID column if needed
    if id_col not in df.columns:
        candidates = [c for c in df.columns if c.upper() in ("ID", "N")]
        if not candidates:
            raise ValueError(f"Could not find an ID-like column in {path}")
        id_col = candidates[0]

    # Infer SMILES column
    if smiles_col not in df.columns:
        candidates = [c for c in df.columns if c.upper() == "SMILES"]
        if not candidates:
            raise ValueError(f"Could not find SMILES column in {path}")
        smiles_col = candidates[0]

    # Infer label column if not given
    if not label_col:
        candidates = [c for c in df.columns if c not in (id_col, smiles_col)]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot infer label column in {path}; found candidates: {candidates}"
            )
        label_col = candidates[0]

    if new_label_name is None:
        new_label_name = label_col

    df = df[[id_col, smiles_col, label_col]].copy()
    df.columns = ["ID", "SMILES", new_label_name]

    return df


def merge_training_sets(
    fluor480: pd.DataFrame,
    fluor340: pd.DataFrame,
    trans450: pd.DataFrame,
    trans340: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Reconstruct the merged multi-task dataset exactly like merge_euos25_csv.py,
    but without canonicalization.

    Returns:
        merged_df: merged multi-task dataset
        conflict_ids: IDs where SMILES differ across tasks
    """
    merged = fluor480.merge(
        fluor340[["ID", "Fluorescence340_450"]], on="ID", how="inner"
    )
    merged = merged.merge(
        trans450[["ID", "Transmittance450"]], on="ID", how="inner"
    )
    merged = merged.merge(
        trans340[["ID", "Transmittance340"]], on="ID", how="inner"
    )

    id_to_smiles: Dict[int, set] = {}
    for df in (fluor480, fluor340, trans450, trans340):
        for _, row in df.iterrows():
            _id = int(row["ID"])
            sm = row["SMILES"]
            id_to_smiles.setdefault(_id, set()).add(sm)

    conflict_ids = [i for i, sset in id_to_smiles.items() if len(sset) > 1]
    return merged, conflict_ids


def summarize_ids(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name} ===")
    print(f"Shape: {df.shape}")
    print(f"Unique IDs: {df['ID'].nunique()}")
    print(f"Example rows:\n{df.head(3)}")


def plot_histograms(merged: pd.DataFrame, outdir: str) -> None:
    label_cols = [
        "Fluorescence480",
        "Fluorescence340_450",
        "Transmittance450",
        "Transmittance340",
    ]
    existing = [c for c in label_cols if c in merged.columns]
    if not existing:
        print("No expected label columns found in merged_train.csv; skipping histograms.")
        return

    for col in existing:
        plt.figure()
        merged[col].hist(bins=50)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_{col}.png"), dpi=150)
        plt.close()
        print(f"Saved histogram for {col} to hist_{col}.png")


def plot_correlation(merged: pd.DataFrame, outdir: str) -> None:
    label_cols = [
        "Fluorescence480",
        "Fluorescence340_450",
        "Transmittance450",
        "Transmittance340",
    ]
    existing = [c for c in label_cols if c in merged.columns]
    if len(existing) < 2:
        print("Not enough label columns to compute correlation matrix; skipping.")
        return

    corr = merged[existing].corr()
    corr.to_csv(os.path.join(outdir, "label_correlations.csv"))
    print("Saved correlation matrix to label_correlations.csv")

    plt.figure()
    im = plt.imshow(corr.values, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(existing)), existing, rotation=45, ha="right")
    plt.yticks(range(len(existing)), existing)
    plt.title("Correlation between task labels")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "label_correlations.png"), dpi=150)
    plt.close()
    print("Saved correlation heatmap to label_correlations.png")


def save_before_after_sample(
    fluor480: pd.DataFrame,
    fluor340: pd.DataFrame,
    trans450: pd.DataFrame,
    trans340: pd.DataFrame,
    merged: pd.DataFrame,
    outdir: str,
    n_samples: int = 10,
) -> None:
    """
    Create a small table that shows, for a few IDs:
    - Their SMILES and labels in each original CSV
    - Their row in the merged CSV
    """
    # To have a fair comparison, sample IDs that are actually present in the merged dataset
    sample_ids = merged["ID"].drop_duplicates().sample(
        min(n_samples, merged["ID"].nunique()), random_state=42
    )

    # Merge the four original datasets together (wide format)
    combined = fluor480.merge(
        fluor340,
        on="ID",
        how="left",
        suffixes=("_480", "_340"),
    )
    combined = combined.merge(
        trans450,
        on="ID",
        how="left",
        suffixes=("", "_t450"),
    )
    # After previous merge, trans450 label is named "Transmittance450"
    # and SMILES column from trans450 might be called "SMILES_t450" depending on overlaps.
    combined = combined.merge(
        trans340,
        on="ID",
        how="left",
        suffixes=("", "_t340"),
    )

    # To avoid SMILES confusion, keep only one SMILES per source explicitly
    cols_to_keep = [c for c in combined.columns if "SMILES" in c or "ID" in c or "Fluorescence" in c or "Transmittance" in c]
    combined = combined[cols_to_keep]

    sample_before = combined[combined["ID"].isin(sample_ids)].copy()
    sample_after = merged[merged["ID"].isin(sample_ids)].copy()

    sample_before.to_csv(os.path.join(outdir, "sample_before_merge.csv"), index=False)
    sample_after.to_csv(os.path.join(outdir, "sample_after_merge.csv"), index=False)
    print("Saved sample_before_merge.csv and sample_after_merge.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and visualize the EUOS25 merged dataset."
    )
    parser.add_argument(
        "--train_fluor480",
        required=True,
        help="Path to fluorescence 480nm training CSV.",
    )
    parser.add_argument(
        "--train_fluor340",
        required=True,
        help="Path to fluorescence 340/450nm training CSV.",
    )
    parser.add_argument(
        "--train_trans450",
        required=True,
        help="Path to transmittance 450-679nm training CSV.",
    )
    parser.add_argument(
        "--train_trans340",
        required=True,
        help="Path to transmittance 340nm training CSV.",
    )
    parser.add_argument(
        "--merged_train",
        default="merged_train.csv",
        help="Path to merged training CSV (output of merge_euos25_csv.py).",
    )
    parser.add_argument(
        "--output_dir",
        default="merge_diagnostics",
        help="Directory to save plots and sample tables.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading and standardizing original training CSVs...")
    fluor480 = load_and_standardize(
        args.train_fluor480,
        id_col="ID",
        smiles_col="SMILES",
        label_col="Fluorescence",
        new_label_name="Fluorescence480",
    )
    fluor340 = load_and_standardize(
        args.train_fluor340,
        id_col="ID",
        smiles_col="SMILES",
        label_col="Fluorescence",
        new_label_name="Fluorescence340_450",
    )
    trans450 = load_and_standardize(
        args.train_trans450,
        id_col="ID",
        smiles_col="SMILES",
        label_col="Transmittance",
        new_label_name="Transmittance450",
    )
    trans340 = load_and_standardize(
        args.train_trans340,
        id_col="N",  # same as in merge script
        smiles_col="SMILES",
        label_col="Transmittance (qualitative)",
        new_label_name="Transmittance340",
    )

    summarize_ids(fluor480, "Fluorescence 480")
    summarize_ids(fluor340, "Fluorescence 340/450")
    summarize_ids(trans450, "Transmittance 450")
    summarize_ids(trans340, "Transmittance 340 (qualitative)")

    print("\nComputing ID overlaps...")
    set_480 = set(fluor480["ID"])
    set_340 = set(fluor340["ID"])
    set_t450 = set(trans450["ID"])
    set_t340 = set(trans340["ID"])

    all_ids = set_480 | set_340 | set_t450 | set_t340
    intersect_all = set_480 & set_340 & set_t450 & set_t340

    print(f"Total unique IDs across all tasks: {len(all_ids)}")
    print(f"IDs present in ALL four tasks: {len(intersect_all)}")

    print("\nReconstructing merged dataset from the four inputs...")
    reconstructed, conflict_ids = merge_training_sets(
        fluor480, fluor340, trans450, trans340
    )
    print(f"Reconstructed merged shape: {reconstructed.shape}")
    if conflict_ids:
        print(f"SMILES conflicts detected for {len(conflict_ids)} IDs.")
        print(f"Example conflict IDs: {conflict_ids[:10]}")
    else:
        print("No SMILES conflicts detected across tasks.")

    print(f"\nLoading existing merged_train CSV from: {args.merged_train}")
    merged_file = pd.read_csv(args.merged_train)
    summarize_ids(merged_file, "Merged (from file)")

    # Compare reconstructed vs provided merged file
    print("\nComparing reconstructed merged data to merged_train.csv...")
    # To compare, sort by ID and ensure same columns & order where possible
    common_cols = [c for c in reconstructed.columns if c in merged_file.columns]
    rec_sorted = reconstructed[common_cols].sort_values("ID").reset_index(drop=True)
    file_sorted = merged_file[common_cols].sort_values("ID").reset_index(drop=True)

    if rec_sorted.equals(file_sorted):
        print("✅ merged_train.csv matches the reconstructed merged dataset (for common columns).")
    else:
        print("⚠️ merged_train.csv differs from the reconstructed dataset (at least in some rows/columns).")
        # Check simple differences in IDs
        rec_ids = set(reconstructed["ID"])
        file_ids = set(merged_file["ID"])
        print(f"IDs only in reconstructed: {len(rec_ids - file_ids)}")
        print(f"IDs only in merged_train.csv: {len(file_ids - rec_ids)}")

    print("\nSaving 'before vs after' sample tables...")
    save_before_after_sample(
        fluor480, fluor340, trans450, trans340, merged_file, args.output_dir
    )

    print("\nPlotting histograms and correlations for task labels in merged_train.csv...")
    plot_histograms(merged_file, args.output_dir)
    plot_correlation(merged_file, args.output_dir)

    print("\nDone. Check the directory for outputs:\n  ", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
