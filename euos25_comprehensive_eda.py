"""
euos25_comprehensive_eda.py
=================================

This module contains an expanded set of utilities for performing
exploratory data analysis (EDA) on the EUOS25 challenge dataset.
The functions in this module build upon the baseline EDA provided
in ``euos25_eda_and_baselines.py`` by offering richer chemical
descriptors, advanced visualisations and dimensionality reduction.

Key features
------------

* **Comprehensive descriptor calculation**: Compute a wide variety of
  physico‑chemical descriptors using RDKit.  All descriptors
  available in ``rdkit.Chem.Descriptors._descList`` are supported.
  Invalid or non‑numeric values are handled gracefully.

* **Fingerprint calculation**: Generate multiple fingerprint types
  including circular (Morgan/ECFP) and MACCS keys.  Fingerprints
  can be combined with descriptors for downstream modelling.

* **Visualisation utilities**: Plot distributions of descriptors
  conditioned on class labels, correlation heatmaps, t‑SNE/UMAP
  projections of the feature space coloured by labels, and simple
  functional group statistics.  All plots comply with the
  guidelines: one figure per plot, no explicit colour settings.

* **Substructure analysis**: Compute the frequency of selected
  functional groups (e.g. aromatic rings, hetero atoms) in
  positive versus negative classes for each task.

These functions are designed to run efficiently on large datasets
and produce informative graphics for scientists looking to
understand the chemical space of the EUOS25 challenge.  They can
be used interactively in Jupyter notebooks or invoked from the
command line (see the ``__main__`` block for an example).
"""

from __future__ import annotations

import os
import itertools
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

#######################################################################
# Descriptor calculation
#######################################################################

def compute_all_rdkit_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """Compute all RDKit 2D descriptors for a molecule.

    RDKit defines a list of descriptor functions in
    ``Descriptors._descList``.  This function iterates over all
    descriptors and evaluates them on the provided molecule.  Any
    descriptor that raises an exception or returns a non‑finite
    value is stored as ``np.nan``.  The resulting dictionary maps
    descriptor names to numeric values.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule for which to compute descriptors.

    Returns
    -------
    dict
        Mapping from descriptor name to value (float or NaN).
    """
    descs: Dict[str, float] = {}
    for name, func in Descriptors._descList:
        try:
            val = func(mol)
            # Some descriptors return tuples (e.g. Chi indices).  Flatten
            if isinstance(val, tuple):
                # For tuple descriptors, append indices to the name
                for i, v in enumerate(val):
                    descs[f"{name}_{i}"] = float(v) if np.isfinite(v) else np.nan
            else:
                descs[name] = float(val) if np.isfinite(val) else np.nan
        except Exception:
            # If descriptor calculation fails, mark as NaN
            descs[name] = np.nan
    return descs


def compute_fingerprints(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048,
                         use_chirality: bool = True) -> Dict[str, np.ndarray]:
    """Generate multiple fingerprint representations for a molecule.

    This function computes both an ECFP (Morgan) fingerprint and a
    MACCS keys fingerprint.  Additional fingerprint types can be added
    as needed.  The resulting dictionary contains numpy arrays of
    ``dtype=float32`` for easy concatenation with descriptors.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule for which to compute fingerprints.
    radius : int
        Radius for the ECFP fingerprint.
    n_bits : int
        Length of the ECFP bit vector.
    use_chirality : bool
        Whether to include chirality information in ECFP.

    Returns
    -------
    dict
        Mapping from fingerprint name to numpy array of shape
        (n_bits,) or appropriate length for MACCS keys.
    """
    fps: Dict[str, np.ndarray] = {}
    # Morgan (ECFP) fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=use_chirality)
    arr = np.zeros((1, n_bits), dtype=np.float32)
    from rdkit import DataStructs
    DataStructs.ConvertToNumpyArray(fp, arr)
    fps[f"ECFP{radius}_{n_bits}"] = arr.flatten()
    # MACCS keys fingerprint (166 bits)
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros((1, maccs.GetNumBits()), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
    fps["MACCS"] = maccs_arr.flatten()
    return fps


def featurize_smiles(df: pd.DataFrame, smiles_col: str = "smiles", radius: int = 2,
                     n_bits: int = 2048) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Compute descriptors and fingerprints for each SMILES in a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain a column with SMILES strings.
    smiles_col : str
        Name of the column containing SMILES.
    radius : int
        Radius for ECFP fingerprints.
    n_bits : int
        Length of ECFP fingerprints.

    Returns
    -------
    descriptor_df : DataFrame
        DataFrame of RDKit descriptors (one row per molecule).
    fp_dict : dict
        Mapping from fingerprint name to 2D numpy array (n_samples × n_bits).
    """
    descriptor_list: List[Dict[str, float]] = []
    fp_dict: Dict[str, List[np.ndarray]] = {}
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Append NaNs for descriptors and zeros for fingerprints
            descriptor_list.append({name: np.nan for name, _ in Descriptors._descList})
            for key in [f"ECFP{radius}_{n_bits}", "MACCS"]:
                fp_dict.setdefault(key, []).append(np.zeros(n_bits if key.startswith("ECFP") else 166, dtype=np.float32))
            continue
        # Descriptors
        descs = compute_all_rdkit_descriptors(mol)
        descriptor_list.append(descs)
        # Fingerprints
        fps = compute_fingerprints(mol, radius=radius, n_bits=n_bits)
        for key, arr in fps.items():
            fp_dict.setdefault(key, []).append(arr)
    # Convert descriptor list to DataFrame
    descriptor_df = pd.DataFrame(descriptor_list)
    # Convert fingerprint lists to arrays
    fp_arrays: Dict[str, np.ndarray] = {}
    for key, arrs in fp_dict.items():
        fp_arrays[key] = np.stack(arrs, axis=0)
    return descriptor_df, fp_arrays


#######################################################################
# Visualisation utilities
#######################################################################

def plot_descriptor_histograms(desc_df: pd.DataFrame, labels: pd.Series,
                               descriptor_names: Optional[List[str]] = None,
                               max_plots: int = 20, save_dir: Optional[str] = None) -> None:
    """Plot histograms of descriptors stratified by binary labels.

    For each selected descriptor, this function creates a histogram
    showing the distribution of its values for positive and negative
    samples.  Only a subset of descriptors can be plotted to avoid
    generating an excessive number of figures.  The number of
    descriptors displayed can be controlled via ``max_plots`` and
    ``descriptor_names``.

    Parameters
    ----------
    desc_df : DataFrame
        DataFrame of descriptor values (one row per molecule).
    labels : Series
        Binary labels (0/1) used to stratify the histograms.
    descriptor_names : list of str, optional
        Specific descriptor names to plot.  If None, the function
        automatically selects the ``max_plots`` descriptors with the
        highest variance.
    max_plots : int
        Maximum number of descriptors to visualise.
    save_dir : str, optional
        Directory to save the figures.  If None, the plots are
        displayed interactively.
    """
    if descriptor_names is None:
        # Select descriptors with highest variance as a proxy for
        # informativeness
        variances = desc_df.var().sort_values(ascending=False)
        descriptor_names = variances.iloc[:max_plots].index.tolist()
    else:
        descriptor_names = descriptor_names[:max_plots]
    for name in descriptor_names:
        vals = desc_df[name]
        # Skip if all values are NaN
        if vals.dropna().empty:
            continue
        pos = vals[labels == 1].dropna().astype(float)
        neg = vals[labels == 0].dropna().astype(float)
        # Determine common bin edges
        combined = pd.concat([pos, neg], ignore_index=True)
        bins = np.linspace(combined.min(), combined.max(), 40)
        fig, ax = plt.subplots()
        ax.hist(neg, bins=bins, alpha=0.5, label="negatives")
        ax.hist(pos, bins=bins, alpha=0.5, label="positives")
        ax.set_title(f"Distribution of {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
        ax.legend()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"hist_{name}.png"), bbox_inches="tight")
        plt.close(fig)


def plot_correlation_heatmap(desc_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot a correlation heatmap of descriptors.

    Uses a simple imshow to visualise the correlation matrix.  Axis
    labels are rotated for readability.  Note that large numbers of
    descriptors may result in a very dense plot; consider selecting
    a subset of descriptors beforehand.

    Parameters
    ----------
    desc_df : DataFrame
        Descriptor matrix with one row per molecule.
    save_path : str, optional
        File path to save the heatmap.  If None, the plot is
        displayed interactively.
    """
    corr = desc_df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=6)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Descriptor correlation heatmap")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_tsne_projection(features: np.ndarray, labels: np.ndarray,
                         perplexity: float = 30.0, n_components: int = 2,
                         save_path: Optional[str] = None) -> None:
    """Compute and plot a t‑SNE projection of high‑dimensional data.

    The input ``features`` matrix is standardised before applying
    t‑SNE.  Points are coloured according to ``labels`` (binary or
    multi‑class).  Because t‑SNE is stochastic, results may vary
    between runs; consider fixing ``random_state`` if reproducible
    plots are desired.

    Parameters
    ----------
    features : np.ndarray
        2D array of shape (n_samples, n_features).
    labels : np.ndarray
        Array of shape (n_samples,) containing integer labels.
    perplexity : float
        t‑SNE perplexity parameter.
    n_components : int
        Number of output dimensions (2 for a scatter plot).
    save_path : str, optional
        File path to save the plot.  If None, displays interactively.
    """
    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    # Compute t‑SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, init='pca', n_iter=1000)
    embedding = tsne.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    for label in np.unique(labels):
        idx = labels == label
        ax.scatter(embedding[idx, 0], embedding[idx, 1], label=str(label), alpha=0.6)
    ax.set_title("t‑SNE projection of feature space")
    ax.set_xlabel("t‑SNE 1")
    ax.set_ylabel("t‑SNE 2")
    ax.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def functional_group_stats(desc_df: pd.DataFrame, labels: pd.Series,
                           save_path: Optional[str] = None) -> None:
    """Plot basic functional group statistics across classes.

    This helper function computes the mean of several simple
    descriptors (number of aromatic rings, hetero atoms, hydrogen
    bond donors/acceptors) for positives and negatives and displays
    them in a bar chart.  It is intended to give a high‑level view of
    how functional groups differ between classes.

    Parameters
    ----------
    desc_df : DataFrame
        DataFrame of descriptors including at least the columns
        ``NumAromaticRings``, ``NumHeteroatoms``, ``NumHBD`` and
        ``NumHBA``.  If these columns are missing (e.g. from an
        alternative descriptor calculation), the function will
        silently skip them.
    labels : Series
        Binary labels.
    save_path : str, optional
        File path to save the bar chart.  If None, displays
        interactively.
    """
    group_stats: Dict[str, List[float]] = {}
    fg_cols = [c for c in ["NumAromaticRings", "NumHeteroatoms", "NumHBD", "NumHBA"] if c in desc_df.columns]
    if not fg_cols:
        warnings.warn("No functional group columns found; skipping functional group stats plot.")
        return
    for col in fg_cols:
        group_stats[col] = [desc_df.loc[labels == 0, col].mean(), desc_df.loc[labels == 1, col].mean()]
    fig, ax = plt.subplots()
    x = np.arange(len(fg_cols))
    width = 0.35
    neg_means = [group_stats[col][0] for col in fg_cols]
    pos_means = [group_stats[col][1] for col in fg_cols]
    ax.bar(x - width/2, neg_means, width, label="negatives")
    ax.bar(x + width/2, pos_means, width, label="positives")
    ax.set_xticks(x)
    ax.set_xticklabels(fg_cols, rotation=45)
    ax.set_ylabel("Mean value")
    ax.set_title("Functional group statistics by class")
    ax.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


#######################################################################
# Demonstration (synthetic data)
#######################################################################

if __name__ == "__main__":
    """Run a demonstration of the EDA utilities on synthetic data.

    This block generates a small random dataset of alkane SMILES and
    random binary labels to illustrate how the EDA functions can be
    invoked.  It then computes descriptors and fingerprints,
    visualises distributions and correlations, performs a t‑SNE
    projection and plots functional group statistics.  The plots are
    saved into a ``demo_eda_plots`` directory.
    """
    import random
    # Generate synthetic dataset
    n_samples = 300
    smiles_list = []
    rng = np.random.RandomState(0)
    for _ in range(n_samples):
        length = rng.randint(3, 8)
        smiles_list.append("C" * length)
    labels = rng.binomial(1, 0.1, size=n_samples)
    df_demo = pd.DataFrame({"smiles": smiles_list, "label": labels})
    # Compute descriptors and fingerprints
    desc_df, fp_dict = featurize_smiles(df_demo, "smiles", radius=2, n_bits=1024)
    # Plot descriptor distributions for a few high variance descriptors
    os.makedirs("demo_eda_plots", exist_ok=True)
    plot_descriptor_histograms(desc_df, df_demo["label"], max_plots=10, save_dir="demo_eda_plots")
    # Plot correlation heatmap of top descriptors by variance
    # Reduce to 30 descriptors with highest variance to avoid clutter
    top_desc = desc_df.var().sort_values(ascending=False).head(30).index
    plot_correlation_heatmap(desc_df[top_desc], save_path=os.path.join("demo_eda_plots", "corr_heatmap.png"))
    # t‑SNE projection on descriptors (fill NaNs with mean)
    desc_filled = desc_df.fillna(desc_df.mean())
    plot_tsne_projection(desc_filled[top_desc].values, labels, save_path=os.path.join("demo_eda_plots", "tsne.png"))
    # Functional group statistics
    functional_group_stats(desc_df, df_demo["label"], save_path=os.path.join("demo_eda_plots", "fg_stats.png"))