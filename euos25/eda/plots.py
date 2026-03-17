"""Plotting utilities for exploratory data analysis on EUOS25 features.

Functions in this module generate informative charts to visualise class
imbalances, feature distributions, correlations, low‑dimensional embeddings
and functional group counts.  Plots may be shown interactively or saved
to disk depending on the `save_*` arguments.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_class_distribution(
    y: np.ndarray, label_cols: List[str], save_path: Optional[str] = None
) -> None:
    """Plot the distribution of positive vs negative examples for each task."""
    n_tasks = y.shape[1]
    fig, axes = plt.subplots(n_tasks, 1, figsize=(6, 4 * n_tasks))
    if n_tasks == 1:
        axes = [axes]
    for idx, task_name in enumerate(label_cols):
        counts = pd.Series(y[:, idx]).value_counts().sort_index()
        axes[idx].bar(counts.index.astype(str), counts.values)
        axes[idx].set_title(f"Class distribution for {task_name}")
        axes[idx].set_xlabel("Label")
        axes[idx].set_ylabel("Count")
        for i, v in enumerate(counts.values):
            axes[idx].text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def plot_descriptor_histograms(
    X: np.ndarray,
    feature_names: List[str],
    y: np.ndarray,
    label_cols: List[str],
    num_features: int = 10,
    save_dir: Optional[str] = None,
) -> None:
    """Plot histograms of the top `num_features` descriptors by variance."""
    desc_indices = [
        i
        for i, name in enumerate(feature_names)
        if not name.startswith("ECFP") and not name.startswith("MACCS")
    ]
    X_desc = X[:, desc_indices]
    desc_names = [feature_names[i] for i in desc_indices]
    variances = X_desc.var(axis=0)
    top_indices = np.argsort(variances)[::-1][:num_features]
    selected_indices = [desc_indices[i] for i in top_indices]
    selected_names = [feature_names[i] for i in selected_indices]
    n_tasks = y.shape[1]
    for feature_idx, feat_name in zip(selected_indices, selected_names):
        fig, axes = plt.subplots(n_tasks, 1, figsize=(6, 3 * n_tasks))
        if n_tasks == 1:
            axes = [axes]
        for t_idx, task_name in enumerate(label_cols):
            pos_vals = X[y[:, t_idx] == 1, feature_idx]
            neg_vals = X[y[:, t_idx] == 0, feature_idx]
            axes[t_idx].hist(neg_vals, bins=30, alpha=0.5, label="Negative")
            axes[t_idx].hist(pos_vals, bins=30, alpha=0.5, label="Positive")
            axes[t_idx].set_title(f"{feat_name} distribution for {task_name}")
            axes[t_idx].set_xlabel(feat_name)
            axes[t_idx].set_ylabel("Count")
            axes[t_idx].legend()
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"hist_{feat_name}.png"))
        plt.close(fig)


def plot_correlation_heatmap(
    X: np.ndarray, feature_names: List[str], save_path: Optional[str] = None
) -> None:
    """Plot a correlation heatmap of descriptor features."""
    desc_indices = [
        i
        for i, name in enumerate(feature_names)
        if not name.startswith("ECFP") and not name.startswith("MACCS")
    ]
    X_desc = X[:, desc_indices]
    desc_names = [feature_names[i] for i in desc_indices]
    corr = np.corrcoef(X_desc, rowvar=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, interpolation="none")
    ax.set_xticks(range(len(desc_names)))
    ax.set_xticklabels(desc_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(desc_names)))
    ax.set_yticklabels(desc_names, fontsize=6)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Descriptor correlation heatmap")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def tsne_projection(
    X: np.ndarray,
    y: np.ndarray,
    label_cols: List[str],
    perplexity: float = 30.0,
    n_iter: int = 1000,
    save_path: Optional[str] = None,
) -> None:
    """Perform a t‑SNE projection and colour points by task labels."""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError(
            "scikit‑learn is required for t‑SNE projection.  Install scikit‑learn."
        )
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        init="random",
        random_state=42,
    )
    X_2d = tsne.fit_transform(X)
    fig, axes = plt.subplots(1, len(label_cols), figsize=(6 * len(label_cols), 5))
    if len(label_cols) == 1:
        axes = [axes]
    for idx, task_name in enumerate(label_cols):
        pos = y[:, idx] == 1
        neg = y[:, idx] == 0
        axes[idx].scatter(
            X_2d[neg, 0], X_2d[neg, 1], alpha=0.5, label="Negative"
        )
        axes[idx].scatter(
            X_2d[pos, 0], X_2d[pos, 1], alpha=0.5, label="Positive"
        )
        axes[idx].set_title(f"t‑SNE for {task_name}")
        axes[idx].set_xlabel("t‑SNE 1")
        axes[idx].set_ylabel("t‑SNE 2")
        axes[idx].legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)


def functional_group_stats(
    smiles: List[str],
    y: np.ndarray,
    label_cols: List[str],
    save_dir: Optional[str] = None,
) -> None:
    """Compute and plot functional group presence in positive vs negative samples."""
    from rdkit import Chem

    patterns = {
        "Benzene": "[cR]1[cR][cR][cR][cR][cR]1",
        "Nitro": "[N+](=O)[O-]",
        "Hydroxyl": "[OX2H]",
        "Amine": "[NX3;H2,H1;!$(NC=O)]",
        "Carboxyl": "C(=O)[O;H,-]",
    }
    compiled = {
        name: Chem.MolFromSmarts(smarts) for name, smarts in patterns.items()
    }
    group_counts = {task: {} for task in label_cols}
    for group_name, patt in compiled.items():
        if patt is None:
            continue
        for t_idx, task in enumerate(label_cols):
            pos_count = 0
            neg_count = 0
            for smi, label in zip(smiles, y[:, t_idx]):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                has_group = mol.HasSubstructMatch(patt)
                if has_group:
                    if label == 1:
                        pos_count += 1
                    else:
                        neg_count += 1
            group_counts[task][group_name] = (pos_count, neg_count)
    # Plot counts
    for task in label_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        groups = list(group_counts[task].keys())
        pos_vals = [group_counts[task][g][0] for g in groups]
        neg_vals = [group_counts[task][g][1] for g in groups]
        x = np.arange(len(groups))
        ax.bar(x - 0.2, pos_vals, width=0.4, label="Positive")
        ax.bar(x + 0.2, neg_vals, width=0.4, label="Negative")
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha="right")
        ax.set_title(f"Functional group counts for {task}")
        ax.set_ylabel("Count of molecules with group")
        ax.legend()
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"func_groups_{task}.png"))
        plt.close(fig)
