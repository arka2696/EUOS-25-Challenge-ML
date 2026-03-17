#!/usr/bin/env python3
"""
merge_euos25_csv.py

Merge the four EUOS25 training CSV files into a single multi-task dataset.

Expected input files (default names):

    euos25_challenge_train_fluorescence480.csv
    euos25_challenge_train_fluorescence340_450.csv
    euos25_challenge_train_transmittance450.csv
    euos25_challenge_train_transmittance340.csv
    euos25_challenge_test.csv   (optional, passed through unchanged)

Output:

    merged_train.csv   # ID, SMILES, Fluorescence480, Fluorescence340_450,
                       # Transmittance450, Transmittance340
    cleaned_test.csv   # ID, SMILES (SMILES optionally canonicalised)

Usage example:

    python merge_euos25_csv.py \
        --train_fluor480 euos25_challenge_train_fluorescence480.csv \
        --train_fluor340 euos25_challenge_train_fluorescence340_450.csv \
        --train_trans450 euos25_challenge_train_transmittance450.csv \
        --train_trans340 euos25_challenge_train_transmittance340.csv \
        --test_csv euos25_challenge_test.csv \
        --output_train merged_train.csv \
        --output_test cleaned_test.csv \
        --canonicalize

"""

import argparse
from typing import Optional, Dict, Tuple, List

import pandas as pd

try:
    from rdkit import Chem
except ImportError:
    Chem = None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Return canonical SMILES using RDKit; if RDKit not installed or parsing
    fails, return the original string.
    """
    if Chem is None or pd.isna(smiles):
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def load_and_standardize(
    path: str,
    id_col: str = "ID",
    smiles_col: str = "SMILES",
    label_col: str = "",
    new_label_name: Optional[str] = None,
    canonicalize: bool = False,
) -> pd.DataFrame:
    """Load a CSV and standardize column names.

    - Renames ID / SMILES / label columns to shared names.
    - Optionally canonicalizes SMILES.
    """
    df = pd.read_csv(path)

    # Infer ID column if needed
    if id_col not in df.columns:
        # Some file has "N" instead of "ID"
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
        candidates = [
            c
            for c in df.columns
            if c not in (id_col, smiles_col)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot infer label column in {path}; found candidates: {candidates}"
            )
        label_col = candidates[0]

    # New label name
    if new_label_name is None:
        new_label_name = label_col

    df = df[[id_col, smiles_col, label_col]].copy()
    df.columns = ["ID", "SMILES", new_label_name]

    if canonicalize:
        df["SMILES"] = df["SMILES"].astype(str).apply(canonicalize_smiles)

    return df


def merge_training_sets(
    fluor480: pd.DataFrame,
    fluor340: pd.DataFrame,
    trans450: pd.DataFrame,
    trans340: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[int]]:
    """Merge the four task DataFrames on ID.

    Returns:
        merged_df: merged multi-task dataset
        conflict_ids: IDs where SMILES differ across tasks
    """
    # Inner merge by ID only (SMILES will be checked)
    merged = fluor480.merge(
        fluor340[["ID", "Fluorescence340_450"]], on="ID", how="inner"
    )
    merged = merged.merge(
        trans450[["ID", "Transmittance450"]], on="ID", how="inner"
    )
    merged = merged.merge(
        trans340[["ID", "Transmittance340"]], on="ID", how="inner"
    )

    # At this point we have:
    #   SMILES from fluor480 as "SMILES"
    #   plus labels: Fluorescence480, Fluorescence340_450,
    #                Transmittance450, Transmittance340

    # Check SMILES consistency across original frames
    # Build mapping ID -> set of SMILES from all four sources
    id_to_smiles: Dict[int, set] = {}
    for df in (fluor480, fluor340, trans450, trans340):
        for idx, row in df.iterrows():
            _id = int(row["ID"])
            sm = row["SMILES"]
            id_to_smiles.setdefault(_id, set()).add(sm)

    conflict_ids = [i for i, sset in id_to_smiles.items() if len(sset) > 1]

    return merged, conflict_ids


def main():
    parser = argparse.ArgumentParser(description="Merge EUOS25 training CSVs into a multi-task dataset.")
    parser.add_argument(
        "--train_fluor480",
        default="euos25_challenge_train_fluorescence480.csv",
        help="Path to fluorescence 480nm training CSV.",
    )
    parser.add_argument(
        "--train_fluor340",
        default="euos25_challenge_train_fluorescence340_450.csv",
        help="Path to fluorescence 340/450nm training CSV.",
    )
    parser.add_argument(
        "--train_trans450",
        default="euos25_challenge_train_transmittance450.csv",
        help="Path to transmittance 450-679nm training CSV.",
    )
    parser.add_argument(
        "--train_trans340",
        default="euos25_challenge_train_transmittance340.csv",
        help="Path to transmittance 340nm training CSV.",
    )
    parser.add_argument(
        "--test_csv",
        default="euos25_challenge_test.csv",
        help="Path to test CSV (ID, SMILES).",
    )
    parser.add_argument(
        "--output_train",
        default="merged_train.csv",
        help="Output path for merged training CSV.",
    )
    parser.add_argument(
        "--output_test",
        default="cleaned_test.csv",
        help="Output path for cleaned test CSV.",
    )
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="Canonicalize SMILES with RDKit before merging.",
    )

    args = parser.parse_args()

    print("Loading and standardizing training CSVs...")

    fluor480 = load_and_standardize(
        args.train_fluor480,
        id_col="ID",
        smiles_col="SMILES",
        label_col="Fluorescence",
        new_label_name="Fluorescence480",
        canonicalize=args.canonicalize,
    )
    fluor340 = load_and_standardize(
        args.train_fluor340,
        id_col="ID",
        smiles_col="SMILES",
        label_col="Fluorescence",
        new_label_name="Fluorescence340_450",
        canonicalize=args.canonicalize,
    )
    trans450 = load_and_standardize(
        args.train_trans450,
        id_col="ID",
        smiles_col="SMILES",
        label_col="Transmittance",
        new_label_name="Transmittance450",
        canonicalize=args.canonicalize,
    )
    trans340 = load_and_standardize(
        args.train_trans340,
        id_col="N",  # will be auto-detected if not present
        smiles_col="SMILES",
        label_col="Transmittance (qualitative)",
        new_label_name="Transmittance340",
        canonicalize=args.canonicalize,
    )

    print("Merging training sets on ID...")
    merged, conflict_ids = merge_training_sets(fluor480, fluor340, trans450, trans340)

    print(f"Merged training shape: {merged.shape}")
    if conflict_ids:
        print("WARNING: SMILES conflicts detected for these IDs:")
        print(conflict_ids)
        print(
            "Consider dropping these IDs or inspecting them manually. "
            "They are usually ring-index differences in SMILES."
        )

    # Optional: drop conflicts to be super-safe
    # merged = merged[~merged["ID"].isin(conflict_ids)].copy()

    print(f"Saving merged training CSV to {args.output_train!r}")
    merged.to_csv(args.output_train, index=False)

    # Process test CSV (for consistency / optional canonicalization)
    if args.test_csv:
        print("Loading test CSV...")
        test_df = pd.read_csv(args.test_csv)
        if args.canonicalize and "SMILES" in test_df.columns:
            print("Canonicalizing SMILES in test set...")
            test_df["SMILES"] = test_df["SMILES"].astype(str).apply(canonicalize_smiles)
        print(f"Saving cleaned test CSV to {args.output_test!r}")
        test_df.to_csv(args.output_test, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
