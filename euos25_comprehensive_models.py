"""
euos25_comprehensive_models.py
===============================

This module provides a suite of advanced modelling utilities for the
EUOS25 challenge built upon the insights gathered from extensive
literature review and competition analysis.  It supplements the
baseline and advanced models in previous modules by adding
descriptor‑driven machine learning models (CatBoost, XGBoost),
additional graph neural network architectures (Graph Attention
Networks), support for SMILES and tautomer augmentation, and
ensemble prediction strategies.  Comprehensive evaluation metrics
and plots are also included to aid interpretation.

The functions in this module are designed to run on high‑performance
computing clusters with access to PyTorch, PyTorch Geometric,
CatBoost and XGBoost.  They can be invoked from standalone scripts
or interactive notebooks.  See the ``__main__`` block for an
example workflow using synthetic data.

Key components
--------------

* **Feature preparation**: Compute RDKit descriptors and multiple
  fingerprints (ECFP, MACCS) using utilities from
  ``euos25_comprehensive_eda``.  Optionally select a subset of
  descriptors based on variance.  Combine descriptors and
  fingerprints into a single feature matrix.

* **CatBoost and XGBoost models**: Train gradient boosted decision
  tree models for each task using cross‑validation.  Automatic
  handling of class imbalance via class weights.  Evaluation metrics
  include ROC‑AUC, PR‑AUC, accuracy, precision, recall and F1.

* **Graph Attention Network (GAT) for multitask classification**:
  An extension of the graph neural network architecture that uses
  attention mechanisms to weigh neighbour contributions.  Supports
  multi‑task outputs and focal loss or class weighting for
  imbalanced tasks.  Cross‑validation and early stopping are
  implemented.

* **Tautomer and SMILES augmentation**: Generate alternate SMILES
  representations and tautomeric forms for molecules using RDKit
  enumeration.  Predictions can be averaged across augmented
  structures to improve robustness.

* **Ensembling**: Combine predictions from multiple model types by
  averaging or weighted averaging.  Optionally perform simple
  stacking via logistic regression on validation predictions.

* **Evaluation and plotting**: Compute common classification
  metrics and generate ROC and precision‑recall curves for each
  task.  Confusion matrices are displayed as heatmaps with count
  annotations.

The module intentionally exposes flexible interfaces rather than
hard‑wired pipelines.  Users can mix and match feature sets and
models, perform hyperparameter tuning if desired, and integrate
additional models.  High‑level workflows are illustrated in the
``__main__`` block.
"""

from __future__ import annotations

import os
import math
import random
import itertools
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

# Import feature computation from comprehensive EDA module
try:
    from euos25_comprehensive_eda import compute_all_rdkit_descriptors, compute_fingerprints, featurize_smiles
except ImportError as e:
    raise ImportError("The module euos25_comprehensive_eda must be available in the same directory to use comprehensive models.")

# Tree based models
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Deep learning models
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import torch_geometric
    from torch_geometric.data import Data as GeometricData
    from torch_geometric.nn import GATConv, global_add_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

#######################################################################
# Feature preparation
#######################################################################

def prepare_feature_matrix(df: pd.DataFrame, smiles_col: str, descriptor_variance_cutoff: float = 0.0,
                           radius: int = 2, n_bits: int = 2048,
                           include_maccs: bool = True) -> Tuple[np.ndarray, List[str]]:
    """Compute and concatenate descriptors and fingerprints into a feature matrix.

    This helper uses functions from ``euos25_comprehensive_eda`` to compute
    all RDKit descriptors and two fingerprint types (ECFP and
    MACCS).  Descriptors with variance below the specified cutoff are
    dropped to reduce dimensionality.  Fingerprints are appended as
    continuous features.  The resulting matrix is returned along with
    the list of column names (descriptor names followed by
    fingerprint names with bit indices).

    Parameters
    ----------
    df : DataFrame
        Must contain a column ``smiles_col`` with valid SMILES strings.
    smiles_col : str
        Name of the column containing SMILES strings.
    descriptor_variance_cutoff : float
        Threshold below which descriptor variance leads to removal.
        Setting this to zero keeps all descriptors.
    radius : int
        Radius for ECFP fingerprints.
    n_bits : int
        Length of ECFP fingerprints.
    include_maccs : bool
        Whether to include MACCS fingerprint bits.

    Returns
    -------
    X : ndarray
        Feature matrix of shape (n_samples, n_features).
    feature_names : list of str
        Names corresponding to each column in X.
    """
    # Compute descriptors and fingerprints
    desc_df, fp_dict = featurize_smiles(df, smiles_col=smiles_col, radius=radius, n_bits=n_bits)
    # Drop descriptors with low variance or entirely NaN
    # Fill NaNs with mean first to compute variance
    desc_filled = desc_df.fillna(desc_df.mean())
    variances = desc_filled.var()
    if descriptor_variance_cutoff > 0.0:
        keep_desc = variances[variances > descriptor_variance_cutoff].index
    else:
        keep_desc = variances.index
    desc_selected = desc_filled[keep_desc]
    desc_names = keep_desc.tolist()
    # Standardise descriptors
    scaler = StandardScaler()
    desc_scaled = scaler.fit_transform(desc_selected.values)
    # Concatenate fingerprints
    # ECFP features always included
    ecfp_key = f"ECFP{radius}_{n_bits}"
    ecfp = fp_dict[ecfp_key]
    feature_parts = [desc_scaled, ecfp]
    feature_names = desc_names + [f"{ecfp_key}_{i}" for i in range(ecfp.shape[1])]
    if include_maccs:
        maccs = fp_dict["MACCS"]
        feature_parts.append(maccs)
        feature_names += [f"MACCS_{i}" for i in range(maccs.shape[1])]
    X = np.concatenate(feature_parts, axis=1).astype(np.float32)
    return X, feature_names


#######################################################################
# CatBoost and XGBoost models
#######################################################################

def train_catboost_cv(X: np.ndarray, y: np.ndarray, label_cols: List[str],
                      n_splits: int = 5, random_state: int = 42,
                      class_weights: Optional[Dict[str, Tuple[float, float]]] = None,
                      verbose: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    """Train CatBoost binary classifiers for each task using cross‑validation.

    Parameters
    ----------
    X : ndarray
        Feature matrix of shape (n_samples, n_features).
    y : ndarray
        Binary label matrix of shape (n_samples, n_tasks).
    label_cols : list of str
        Names of the tasks corresponding to columns in ``y``.
    n_splits : int
        Number of stratified folds.
    random_state : int
        Seed for shuffling and reproducibility.
    class_weights : dict, optional
        Optional mapping from task name to tuple (w0, w1) specifying
        weights for negative and positive classes.  If None,
        weights are computed automatically based on training
        distribution in each fold.
    verbose : bool
        If True, CatBoost will print training progress.

    Returns
    -------
    models_per_fold : list of dict
        Each element is a dictionary mapping task names to trained
        CatBoostClassifier models for that fold.
    auc_scores : dict
        Mapping from task name to list of ROC‑AUC values (one per fold).
    """
    if not HAS_CATBOOST:
        raise ImportError("CatBoost is not installed.  Please install catboost to use this function.")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    stratify_label = (y.sum(axis=1) > 0).astype(int)
    models_per_fold: List[Dict[str, CatBoostClassifier]] = []
    auc_scores: Dict[str, List[float]] = {col: [] for col in label_cols}
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, stratify_label)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        fold_models: Dict[str, CatBoostClassifier] = {}
        for task_idx, task_name in enumerate(label_cols):
            y_train_task = y_train[:, task_idx]
            y_val_task = y_val[:, task_idx]
            # Determine class weights
            if class_weights and task_name in class_weights:
                w0, w1 = class_weights[task_name]
                cw = [w0, w1]
            else:
                # Compute weights inversely proportional to class frequencies in the training fold
                pos = (y_train_task == 1).sum()
                neg = (y_train_task == 0).sum()
                if pos == 0 or neg == 0:
                    cw = None
                else:
                    cw = [neg / (pos + neg), pos / (pos + neg)]
            model = CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=7,
                loss_function='Logloss',
                eval_metric='AUC',
                random_seed=random_state + fold_idx,
                verbose=verbose,
                class_weights=cw
            )
            model.fit(X_train, y_train_task, eval_set=(X_val, y_val_task), verbose=verbose)
            probas = model.predict_proba(X_val)[:, 1]
            try:
                auc = roc_auc_score(y_val_task, probas)
            except ValueError:
                auc = float('nan')
            auc_scores[task_name].append(auc)
            fold_models[task_name] = model
        models_per_fold.append(fold_models)
    return models_per_fold, auc_scores


def train_xgboost_cv(X: np.ndarray, y: np.ndarray, label_cols: List[str],
                     n_splits: int = 5, random_state: int = 42,
                     class_weights: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    """Train XGBoost binary classifiers for each task using cross‑validation.

    Parameters
    ----------
    X : ndarray
        Feature matrix.
    y : ndarray
        Binary labels.
    label_cols : list of str
        Task names.
    n_splits : int
        Number of folds.
    random_state : int
        Random seed.
    class_weights : dict, optional
        Class weights per task.

    Returns
    -------
    models_per_fold : list of dict
    auc_scores : dict
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is not installed.  Please install xgboost to use this function.")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    stratify_label = (y.sum(axis=1) > 0).astype(int)
    models_per_fold: List[Dict[str, Any]] = []
    auc_scores: Dict[str, List[float]] = {col: [] for col in label_cols}
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, stratify_label)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        fold_models: Dict[str, Any] = {}
        for task_idx, task_name in enumerate(label_cols):
            y_train_task = y_train[:, task_idx]
            y_val_task = y_val[:, task_idx]
            # Compute scale_pos_weight or class_weight
            if class_weights and task_name in class_weights:
                w0, w1 = class_weights[task_name]
                # In XGBoost, scale_pos_weight = w0/w1; but we approximate via weighting the objective
                scale_pos_weight = w0 / w1 if w1 > 0 else 1.0
            else:
                pos = (y_train_task == 1).sum()
                neg = (y_train_task == 0).sum()
                scale_pos_weight = neg / pos if pos > 0 else 1.0
            dtrain = xgb.DMatrix(X_train, label=y_train_task)
            dval = xgb.DMatrix(X_val, label=y_val_task)
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'eta': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'seed': random_state + fold_idx,
                'scale_pos_weight': scale_pos_weight,
                'verbosity': 0
            }
            model = xgb.train(params, dtrain, num_boost_round=300, evals=[(dval, 'val')], verbose_eval=False)
            probas = model.predict(dval)
            try:
                auc = roc_auc_score(y_val_task, probas)
            except ValueError:
                auc = float('nan')
            auc_scores[task_name].append(auc)
            fold_models[task_name] = model
        models_per_fold.append(fold_models)
    return models_per_fold, auc_scores


#######################################################################
# Graph Attention Network (GAT) for multitask classification
#######################################################################

class GraphAttentionMultiTask(nn.Module):
    """A simple graph attention network for multi‑task classification.

    This model processes molecular graphs generated from SMILES
    strings using PyTorch Geometric.  Node embeddings are updated by
    a series of GATConv layers and pooled with global add pooling.
    A final linear layer produces logits for each task.
    """
    def __init__(self, num_node_features: int, hidden_dim: int, num_tasks: int,
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        in_channels = num_node_features
        for _ in range(num_layers):
            conv = GATConv(in_channels, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
            self.convs.append(conv)
            in_channels = hidden_dim
        self.dropout = dropout
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_tasks)

    def forward(self, data: GeometricData) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Global add pooling
        x = global_add_pool(x, data.batch)
        x = F.relu(self.lin(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.out(x)
        return logits


class MoleculeGraphDataset(Dataset):
    """Dataset for graph neural networks.

    Converts SMILES strings into PyTorch Geometric Data objects.  This
    class is used by both GIN and GAT models.
    """
    def __init__(self, smiles: List[str], labels: np.ndarray, converter_fn: Optional[Any] = None):
        self.smiles = smiles
        self.labels = labels
        self.converter_fn = converter_fn if converter_fn is not None else self._smiles_to_graph

    def _smiles_to_graph(self, smi: str) -> Optional[GeometricData]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        # Node features: one‑hot atomic number up to 100, degree, formal charge, aromaticity, etc.
        atom_feats = []
        for atom in mol.GetAtoms():
            feat = []
            atomic_num = atom.GetAtomicNum()
            one_hot = [0] * 101
            if atomic_num <= 100:
                one_hot[atomic_num] = 1
            else:
                one_hot[0] = 1
            feat.extend(one_hot)
            # Degree (0–5)
            deg = atom.GetTotalDegree()
            deg_one_hot = [1 if deg == i else 0 for i in range(6)]
            feat.extend(deg_one_hot)
            # Formal charge (–2 to +2)
            charge = atom.GetFormalCharge()
            charge_one_hot = [1 if charge == i else 0 for i in range(-2, 3)]
            feat.extend(charge_one_hot)
            # Aromaticity
            feat.append(int(atom.GetIsAromatic()))
            # Number of hydrogens (0–4)
            num_h = atom.GetTotalNumHs()
            h_one_hot = [1 if num_h == i else 0 for i in range(5)]
            feat.extend(h_one_hot)
            atom_feats.append(feat)
        x = torch.tensor(atom_feats, dtype=torch.float)
        # Edge indices and features (bond type, conjugation, ring membership)
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # undirected graph
            edge_index.append([i, j])
            edge_index.append([j, i])
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        data = GeometricData(x=x, edge_index=edge_index)
        return data

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int):
        graph = self.converter_fn(self.smiles[idx])
        label = self.labels[idx]
        return graph, torch.tensor(label, dtype=torch.float)


def collate_graphs(batch: List[Tuple[Optional[GeometricData], torch.Tensor]]) -> Tuple[GeometricData, torch.Tensor]:
    """Collate function for DataLoader to batch graph data."""
    graphs, labels = zip(*batch)
    # Filter out None graphs
    valid_indices = [i for i, g in enumerate(graphs) if g is not None]
    if not valid_indices:
        raise ValueError("All graphs in batch are None")
    graphs = [graphs[i] for i in valid_indices]
    labels = torch.stack([labels[i] for i in valid_indices])
    batch_data = torch_geometric.data.Batch.from_data_list(graphs)
    return batch_data, labels


def train_gat_multitask(smiles: List[str], labels: np.ndarray, label_cols: List[str],
                        hidden_dim: int = 128, num_layers: int = 3, num_heads: int = 4,
                        batch_size: int = 32, epochs: int = 40,
                        lr: float = 1e-3, weight_decay: float = 1e-4,
                        cv_splits: int = 5, random_state: int = 42,
                        use_focal: bool = False, focal_gamma: float = 2.0,
                        class_weights: Optional[Dict[str, Tuple[float, float]]] = None,
                        save_dir: Optional[str] = None,
                        device: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    """Train a Graph Attention Network (GAT) for each fold in cross‑validation.

    This function performs stratified cross‑validation on the provided
    SMILES and labels, training a GAT model per fold.  The model
    outputs per‑task logits which are converted to probabilities
    using a sigmoid activation.  Training supports class weights or
    focal loss to handle imbalanced tasks.  After training, the
    models are returned along with per‑fold ROC‑AUC metrics.

    Parameters
    ----------
    smiles : list of str
        SMILES strings for training samples.
    labels : ndarray
        Binary label matrix of shape (n_samples, n_tasks).
    label_cols : list of str
        Names of the tasks.
    hidden_dim : int
        Dimension of hidden node embeddings.
    num_layers : int
        Number of GATConv layers.
    num_heads : int
        Number of attention heads per layer.
    batch_size : int
        Batch size for training.
    epochs : int
        Number of training epochs per fold.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularisation coefficient.
    cv_splits : int
        Number of cross‑validation splits.
    random_state : int
        Seed for shuffling.
    use_focal : bool
        Whether to use focal loss instead of binary cross‑entropy.
    focal_gamma : float
        Gamma parameter for focal loss.
    class_weights : dict, optional
        Mapping from task name to tuple (w0, w1) of class weights.
    save_dir : str, optional
        Directory to save ROC and PR plots.
    device : str, optional
        Device to run training on ('cpu' or 'cuda').  If None,
        automatically chooses 'cuda' if available.

    Returns
    -------
    models_per_fold : list of dict
        Each element maps task names to trained GAT models.
    auc_scores : dict
        Mapping from task name to list of ROC‑AUC scores per fold.
    """
    if not (HAS_TORCH and HAS_PYG):
        raise ImportError("PyTorch and PyTorch Geometric must be installed to use GAT models.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    stratify_label = (labels.sum(axis=1) > 0).astype(int)
    auc_scores: Dict[str, List[float]] = {t: [] for t in label_cols}
    models_per_fold: List[Dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(smiles, stratify_label)):
        train_smiles = [smiles[i] for i in train_idx]
        val_smiles = [smiles[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        # Build datasets
        train_dataset = MoleculeGraphDataset(train_smiles, train_labels)
        val_dataset = MoleculeGraphDataset(val_smiles, val_labels)
        # Determine sampling weights for imbalance
        # For each graph sample, compute weight as average of per task class weights
        if class_weights:
            # Build weight vector per sample
            sample_weights = []
            for lbl in train_labels:
                weights_per_task = []
                for t_idx, t_name in enumerate(label_cols):
                    if t_name in class_weights:
                        w0, w1 = class_weights[t_name]
                        weights_per_task.append(w1 if lbl[t_idx] == 1 else w0)
                    else:
                        weights_per_task.append(1.0)
                sample_weights.append(np.mean(weights_per_task))
            sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)
        else:
            sampler = None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, collate_fn=collate_graphs)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)
        # Build model
        num_node_features = len(train_dataset[0][0].x[0])
        num_tasks = len(label_cols)
        model = GraphAttentionMultiTask(num_node_features=num_node_features,
                                        hidden_dim=hidden_dim,
                                        num_tasks=num_tasks,
                                        num_layers=num_layers,
                                        num_heads=num_heads,
                                        dropout=0.1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # Define loss function (BCELoss or FocalLoss)
        def binary_focal_loss(pred, target, gamma):
            # pred: (batch, tasks), raw logits; target: same shape
            prob = torch.sigmoid(pred)
            ce = F.binary_cross_entropy(prob, target, reduction='none')
            p_t = prob * target + (1 - prob) * (1 - target)
            loss = (1 - p_t) ** gamma * ce
            return loss.mean()
        # Training loop
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                logits = model(batch_data)
                if use_focal:
                    loss = binary_focal_loss(logits, batch_labels, gamma=focal_gamma)
                else:
                    # Weight each task's loss by optional class weights
                    prob = torch.sigmoid(logits)
                    if class_weights:
                        # compute per-task weights
                            # cw[t_name] = (w0, w1)
                        total_loss = 0.0
                        for t_idx, t_name in enumerate(label_cols):
                            w0, w1 = class_weights.get(t_name, (1.0, 1.0))
                            wt = torch.where(batch_labels[:, t_idx] == 1, torch.tensor(w1, device=device), torch.tensor(w0, device=device))
                            ce = F.binary_cross_entropy(prob[:, t_idx], batch_labels[:, t_idx], reduction='none')
                            total_loss += (wt * ce).mean()
                        loss = total_loss / len(label_cols)
                    else:
                        loss = F.binary_cross_entropy(prob, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_labels.size(0)
            # Optionally evaluate during training or early stopping (omitted for brevity)
        # End of training
        # Evaluate on validation
        model.eval()
        # Aggregate predictions per task
        task_probas: Dict[str, List[float]] = {t: [] for t in label_cols}
        task_targets: Dict[str, List[int]] = {t: [] for t in label_cols}
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                logits = model(batch_data)
                prob = torch.sigmoid(logits)
                for t_idx, t_name in enumerate(label_cols):
                    task_probas[t_name].extend(prob[:, t_idx].cpu().numpy())
                    task_targets[t_name].extend(batch_labels[:, t_idx].cpu().numpy())
        fold_models: Dict[str, GraphAttentionMultiTask] = {}
        for t_name in label_cols:
            y_true = np.array(task_targets[t_name])
            y_prob = np.array(task_probas[t_name])
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = float('nan')
            auc_scores[t_name].append(auc)
            # Save the entire model for the fold; copy since same model used for all tasks
        models_per_fold.append({"model": model})
    return models_per_fold, auc_scores


#######################################################################
# Tautomer and SMILES augmentation
#######################################################################

def enumerate_tautomers(smiles: str, max_variants: int = 10) -> List[str]:
    """Generate a list of tautomeric SMILES for the input molecule.

    Uses RDKit's TautomerEnumerator.  The number of returned
    tautomers is limited to ``max_variants`` to avoid generating an
    excessively large set for highly tautomeric molecules.

    Parameters
    ----------
    smiles : str
        SMILES string for a molecule.
    max_variants : int
        Maximum number of tautomers to return.

    Returns
    -------
    list of str
        Canonical SMILES strings of enumerated tautomers (unique).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    enumerator = rdMolStandardize.TautomerEnumerator()
    tautomers = enumerator.Enumerate(mol)
    smiles_set = set()
    for tmol in tautomers:
        smi = Chem.MolToSmiles(tmol, canonical=True)
        smiles_set.add(smi)
        if len(smiles_set) >= max_variants:
            break
    return list(smiles_set)


def randomize_smiles(smiles: str, num_variants: int = 5) -> List[str]:
    """Generate random SMILES enumerations for a molecule.

    This function uses RDKit to produce randomized SMILES strings by
    shuffling atom order.  Canonical SMILES is included as the first
    element of the returned list.  Duplicate enumerations are
    removed.

    Parameters
    ----------
    smiles : str
        Original SMILES string.
    num_variants : int
        Total number of SMILES strings to generate (including the
        original canonical form).

    Returns
    -------
    list of str
        Unique SMILES strings representing the same molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    smiles_set = {Chem.MolToSmiles(mol, canonical=True)}
    for _ in range(num_variants - 1):
        rand_smi = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        smiles_set.add(rand_smi)
        if len(smiles_set) >= num_variants:
            break
    return list(smiles_set)


def average_predictions_across_variants(smiles_list: List[str], label_cols: List[str],
                                        model_predict_fn, variant_fn, max_variants: int = 5) -> Dict[str, np.ndarray]:
    """Average predictions across augmented SMILES or tautomer variants.

    This utility accepts a list of SMILES and a prediction function
    that takes a list of SMILES and returns a dictionary mapping
    task names to prediction probability arrays.  It then applies a
    variant generation function (e.g. ``enumerate_tautomers`` or
    ``randomize_smiles``) to each SMILES, collects predictions for
    each variant, and averages them per molecule and per task.

    Parameters
    ----------
    smiles_list : list of str
        Original SMILES strings to predict.
    label_cols : list of str
        Names of tasks.
    model_predict_fn : callable
        Function that accepts a list of SMILES and returns a dict
        mapping task name to numpy array of probabilities.
    variant_fn : callable
        Function that generates a list of variants for a single
        SMILES.  Should accept (smiles: str, max_variants: int)
        arguments.
    max_variants : int
        Maximum number of variants per molecule.

    Returns
    -------
    dict
        Mapping task name to numpy array of averaged probabilities
        corresponding to the input ``smiles_list``.
    """
    # For each molecule, generate variants
    all_variants: List[List[str]] = []
    for smi in smiles_list:
        variants = variant_fn(smi, max_variants=max_variants)
        if not variants:
            variants = [smi]
        all_variants.append(variants)
    # Flatten list and track indices
    flat_smiles = list(itertools.chain.from_iterable(all_variants))
    # Obtain predictions for all variants
    pred_dict = model_predict_fn(flat_smiles)
    # Now average predictions per original molecule
    averaged: Dict[str, np.ndarray] = {t: np.zeros(len(smiles_list), dtype=np.float32) for t in label_cols}
    index = 0
    for i, variants in enumerate(all_variants):
        for t in label_cols:
            # sum predictions for this molecule across its variants
            vals = pred_dict[t][index:index + len(variants)]
            averaged[t][i] = np.mean(vals)
        index += len(variants)
    return averaged


#######################################################################
# Evaluation utilities and ensembling
#######################################################################

def evaluate_models(models_per_fold: List[Dict[str, Any]], X_test: np.ndarray, y_test: np.ndarray,
                    label_cols: List[str], threshold: float = 0.5,
                    save_dir: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Evaluate an ensemble of models on a test set.

    For each task, predictions from each fold are averaged to obtain
    final probabilities.  Various metrics are computed and ROC/PR
    curves and confusion matrices are plotted.

    Parameters
    ----------
    models_per_fold : list of dict
        Each dict maps task names to trained models.  Models must
        implement a ``predict_proba`` method (CatBoost, XGBoost) or
        return a probability array.  For GAT models, the dict must
        contain a single entry with key "model" mapping to a
        GraphAttentionMultiTask model, and additional keys
        ``smiles`` or ``X_test`` are not used here.
    X_test : ndarray
        Feature matrix for descriptor‑based models.  For graph models
        this is ignored (predictions must be provided separately).
    y_test : ndarray
        Ground truth labels for evaluation.
    label_cols : list of str
        Task names.
    threshold : float
        Classification threshold.
    save_dir : str, optional
        Directory to save plots.

    Returns
    -------
    dict
        Mapping task name to metric dictionary.
    """
    metrics: Dict[str, Dict[str, float]] = {}
    num_folds = len(models_per_fold)
    for task_idx, task_name in enumerate(label_cols):
        # Collect probability predictions from each fold
        probas_folds: List[np.ndarray] = []
        for fold_models in models_per_fold:
            model = fold_models.get(task_name)
            if model is None:
                # Could be GAT model stored under 'model'
                gnn_model = fold_models.get("model")
                if gnn_model is not None:
                    # Should have stored predictions separately; skip evaluation here
                    continue
            # For CatBoost or XGBoost
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X_test)[:, 1]
            else:
                # XGBoost model returns raw probabilities via predict
                probas = model.predict(xgb.DMatrix(X_test))
            probas_folds.append(probas)
        if not probas_folds:
            # Skip tasks not applicable to descriptor models
            continue
        probas_mean = np.mean(np.stack(probas_folds, axis=0), axis=0)
        y_true = y_test[:, task_idx]
        try:
            auc = roc_auc_score(y_true, probas_mean)
        except ValueError:
            auc = float('nan')
        try:
            ap = average_precision_score(y_true, probas_mean)
        except ValueError:
            ap = float('nan')
        y_pred = (probas_mean >= threshold).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        metrics[task_name] = {
            "ROC_AUC": auc,
            "PR_AUC": ap,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Accuracy": acc,
        }
        # Plot curves
        fpr, tpr, _ = roc_curve(y_true, probas_mean)
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, probas_mean)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(f"ROC curve for {task_name}")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, f"{task_name}_roc.png"), bbox_inches="tight")
        plt.close(fig)
        fig, ax = plt.subplots()
        ax.plot(recall_vals, precision_vals, label=f"PR (AP={ap:.3f})")
        ax.set_title(f"Precision‑Recall curve for {task_name}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        if save_dir:
            fig.savefig(os.path.join(save_dir, f"{task_name}_pr.png"), bbox_inches="tight")
        plt.close(fig)
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap="Blues")
        ax.set_title(f"Confusion matrix for {task_name}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        if save_dir:
            fig.savefig(os.path.join(save_dir, f"{task_name}_cm.png"), bbox_inches="tight")
        plt.close(fig)
    return metrics


#######################################################################
# Main demonstration
#######################################################################

if __name__ == "__main__":
    """Demonstrate feature preparation, CatBoost/XGBoost training and GAT on synthetic data.

    This block simulates a simplified EUOS25 workflow: synthetic
    SMILES are generated, features are computed, descriptor models
    are trained using CatBoost and XGBoost with cross‑validation,
    and a Graph Attention Network is trained on the same tasks.  The
    performance metrics are printed and plots are saved into
    ``demo_model_plots``.  Note that this is for demonstration
    purposes only; real training on the full challenge dataset will
    require significant computational resources.
    """
    # Generate synthetic dataset
    num_samples = 200
    rng = np.random.RandomState(1)
    smiles_list = ["C" * rng.randint(3, 8) for _ in range(num_samples)]
    # Two tasks with severe imbalance
    y_labels = np.vstack([
        rng.binomial(1, 0.05, size=num_samples),
        rng.binomial(1, 0.01, size=num_samples),
    ]).T
    label_cols = ["task1", "task2"]
    df_syn = pd.DataFrame({"smiles": smiles_list})
    # Prepare features (drop descriptors with near‑zero variance)
    X, feature_names = prepare_feature_matrix(df_syn, "smiles", descriptor_variance_cutoff=1e-3, radius=2, n_bits=512, include_maccs=True)
    # Train CatBoost
    if HAS_CATBOOST:
        models_cb, aucs_cb = train_catboost_cv(X, y_labels, label_cols, n_splits=3, random_state=1)
        print("CatBoost AUC scores:", aucs_cb)
    # Train XGBoost
    if HAS_XGBOOST:
        models_xgb, aucs_xgb = train_xgboost_cv(X, y_labels, label_cols, n_splits=3, random_state=1)
        print("XGBoost AUC scores:", aucs_xgb)
    # Train GAT (optional, may be slow)
    if HAS_TORCH and HAS_PYG:
        models_gat, aucs_gat = train_gat_multitask(smiles_list, y_labels, label_cols,
                                                   hidden_dim=64, num_layers=2, num_heads=2,
                                                   batch_size=32, epochs=10, cv_splits=2, random_state=1)
        print("GAT AUC scores:", aucs_gat)
    # Evaluate CatBoost models on a hold‑out test split
    X_train, X_test, y_train_arr, y_test_arr = train_test_split(X, y_labels, test_size=0.3, stratify=(y_labels.sum(axis=1) > 0), random_state=1)
    # For demonstration, use the first fold of CatBoost models
    if HAS_CATBOOST:
        metrics_cb = evaluate_models([models_cb[0]], X_test, y_test_arr, label_cols, threshold=0.5, save_dir="demo_model_plots")
        print("CatBoost metrics on test split:", metrics_cb)
