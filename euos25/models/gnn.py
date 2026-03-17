"""Graph neural network models for EUOS25.

This module implements a multitask Graph Attention Network (GAT) and
a wrapper around AttentiveFP available in PyTorch Geometric.  It also
provides a training function that performs stratified cross‑validation
on SMILES strings converted into graph objects.  Only one model per fold
is stored; tasks share the same set of weights within a fold.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Conditional imports for torch and PyTorch Geometric
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import torch_geometric
    from torch_geometric.nn import GATConv, global_add_pool
    HAS_TORCH = True
    HAS_PYG = True
except ImportError:
    HAS_TORCH = False
    HAS_PYG = False

from ..features.graphs import MoleculeGraphDataset, collate_graphs


class GraphAttentionMultiTask(nn.Module):  # type: ignore[misc]
    """Graph Attention Network for multitask classification."""

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int,
        num_tasks: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        in_channels = num_node_features
        for _ in range(num_layers):
            conv = GATConv(
                in_channels,
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
            )
            self.convs.append(conv)
            in_channels = hidden_dim
        self.dropout = dropout
        self.lin = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_tasks)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, data.batch)
        x = F.relu(self.lin(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.out(x)
        return logits


class AttentiveFPWrapper(nn.Module):  # type: ignore[misc]
    """A wrapper for AttentiveFP for multitask classification."""

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int,
        num_tasks: int,
        num_layers: int = 2,
        timesteps: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric must be installed to use AttentiveFP.")
        from torch_geometric.nn import AttentiveFP  # type: ignore

        self.model = AttentiveFP(
            in_channels=num_node_features,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_layers,
            timesteps=timesteps,
            dropout=dropout,
        )
        self.out = nn.Linear(hidden_dim, num_tasks)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x = self.model(data.x, data.edge_index, data.batch)
        x = F.relu(x)
        logits = self.out(x)
        return logits


def train_gnn_multitask(
    smiles: List[str],
    labels: np.ndarray,
    label_cols: List[str],
    model_type: str = "gat",
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 4,
    batch_size: int = 32,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    cv_splits: int = 5,
    random_state: int = 42,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    class_weights: Optional[Dict[str, Tuple[float, float]]] = None,
    save_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    """Train a graph neural network across cross‑validation folds.

    Parameters mirror the original implementation and support both GAT and
    AttentiveFP architectures.  Class weights may be provided per task to
    mitigate imbalance.  Only one model per fold is stored; tasks share
    weights within a fold.
    """
    if not (HAS_TORCH and HAS_PYG):
        raise ImportError(
            "PyTorch and PyTorch Geometric are required for GNN models."
        )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    # Determine a single stratification label: at least one positive across tasks
    stratify_label = (labels.sum(axis=1) > 0).astype(int)
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    auc_scores: Dict[str, List[float]] = {t: [] for t in label_cols}
    models_per_fold: List[Dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(smiles, stratify_label)):
        train_smiles = [smiles[i] for i in train_idx]
        val_smiles = [smiles[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        train_dataset = MoleculeGraphDataset(train_smiles, train_labels)
        val_dataset = MoleculeGraphDataset(val_smiles, val_labels)
        # Build sampler for class imbalance
        if class_weights:
            sample_weights = []
            for lbl in train_labels:
                per_task_weights = []
                for t_idx, t_name in enumerate(label_cols):
                    if t_name in class_weights:
                        w0, w1 = class_weights[t_name]
                        per_task_weights.append(w1 if lbl[t_idx] == 1 else w0)
                    else:
                        per_task_weights.append(1.0)
                sample_weights.append(np.mean(per_task_weights))
            sampler = WeightedRandomSampler(
                torch.tensor(sample_weights, dtype=torch.double),
                len(sample_weights),
                replacement=True,
            )
        else:
            sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate_graphs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_graphs,
        )
        # Build model
        sample_graph, _ = train_dataset[0]
        num_node_features = sample_graph.x.shape[1]
        num_tasks = len(label_cols)
        if model_type == "gat":
            model = GraphAttentionMultiTask(
                num_node_features=num_node_features,
                hidden_dim=hidden_dim,
                num_tasks=num_tasks,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=0.1,
            )
        elif model_type == "attfp":
            model = AttentiveFPWrapper(
                num_node_features=num_node_features,
                hidden_dim=hidden_dim,
                num_tasks=num_tasks,
                num_layers=num_layers,
                timesteps=num_layers,
                dropout=0.1,
            )
        else:
            raise ValueError("model_type must be 'gat' or 'attfp'")
        model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        def binary_focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
            prob = torch.sigmoid(pred)
            ce = F.binary_cross_entropy(prob, target, reduction="none")
            p_t = prob * target + (1 - prob) * (1 - target)
            loss = ((1 - p_t) ** gamma) * ce
            return loss.mean()

        # Training loop
        for _ in range(epochs):
            model.train()
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                logits = model(batch_data)
                if use_focal:
                    loss = binary_focal_loss(logits, batch_labels, gamma=focal_gamma)
                else:
                    loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
                loss.backward()
                optimizer.step()
        # Evaluate on validation set
        model.eval()
        val_preds = {t: [] for t in label_cols}
        val_true = {t: [] for t in label_cols}
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                logits = model(batch_data)
                prob = torch.sigmoid(logits).cpu().numpy()
                labels_np = batch_labels.cpu().numpy()
                for t_idx, t_name in enumerate(label_cols):
                    val_preds[t_name].extend(prob[:, t_idx])
                    val_true[t_name].extend(labels_np[:, t_idx])
        fold_models: Dict[str, Any] = {}
        for t_idx, t_name in enumerate(label_cols):
            y_true = np.array(val_true[t_name])
            y_score = np.array(val_preds[t_name])
            try:
                auc = roc_auc_score(y_true, y_score)
            except ValueError:
                auc = float("nan")
            auc_scores[t_name].append(auc)
            fold_models[t_name] = model
        models_per_fold.append(fold_models)
    return models_per_fold, auc_scores
