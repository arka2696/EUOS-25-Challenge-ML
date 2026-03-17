"""Basic I/O utilities for reading and writing EUOS25 datasets and results.

The functions in this module handle reading CSV files into pandas DataFrames,
creating train/validation splits and writing out predictions or metrics to disk.
No ML or RDKit specific logic is contained here; see :mod:`euos25.features`
and :mod:`euos25.models` for those responsibilities.
"""

from __future__ import annotations

import os
from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the EUOS25 dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing SMILES strings and labels.

    Returns
    -------
    DataFrame
        Pandas DataFrame with the dataset contents.
    """
    return pd.read_csv(csv_path)


def train_valid_split(
    df: pd.DataFrame,
    labels: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into training and validation subsets.

    The split is stratified on whether at least one label is positive.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    labels : list of str
        Column names containing binary labels.
    test_size : float, default 0.2
        Fraction of the data to allocate to the validation set.
    random_state : int, default 42
        Seed used to shuffle the data before splitting.

    Returns
    -------
    (DataFrame, DataFrame)
        Training and validation DataFrames.
    """
    stratify = (df[labels].sum(axis=1) > 0).astype(int)
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def save_predictions(
    y_pred_proba: pd.DataFrame, out_path: str, index: bool = False
) -> None:
    """Write predicted probabilities to a CSV file.

    Parameters
    ----------
    y_pred_proba : DataFrame
        DataFrame of predictions with rows matching input samples and columns
        corresponding to tasks.
    out_path : str
        Destination path for the CSV file.
    index : bool, default False
        Whether to include the DataFrame index in the output.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    y_pred_proba.to_csv(out_path, index=index)


def save_metrics(metrics: dict, out_path: str) -> None:
    """Save a metrics dictionary to a JSON file.

    Parameters
    ----------
    metrics : dict
        Nested dictionary of evaluation metrics returned from
        :func:`euos25.eval.evaluate_predictions`.
    out_path : str
        File path to write the JSON to.
    """
    import json  # Imported lazily to avoid unnecessary dependency

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
