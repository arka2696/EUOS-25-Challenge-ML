"""Graph feature utilities for molecular data.

This module defines a dataset class for converting SMILES strings into
PyTorch Geometric graph objects and a simple collation function for
batching these graphs.  These utilities are used by the GNN training
functions in :mod:`euos25.models.gnn`.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

try:
    import torch
    from torch.utils.data import Dataset
    import torch_geometric
    from torch_geometric.data import Data as GeometricData
except ImportError:
    # Users can still import this module even if torch/geometric are missing,
    # but will receive informative errors when instantiating classes.
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    GeometricData = None  # type: ignore
    torch_geometric = None  # type: ignore

from rdkit import Chem


class MoleculeGraphDataset(Dataset):  # type: ignore[type-arg]
    """Dataset converting SMILES into PyTorch Geometric graph data.

    Labels are stored as floats for multi‑task binary classification.
    """

    def __init__(
        self,
        smiles: List[str],
        labels: Any,
        converter_fn: Optional[Any] = None,
    ):
        if torch is None or torch_geometric is None:
            raise ImportError(
                "PyTorch and PyTorch Geometric must be installed to use MoleculeGraphDataset."
            )
        self.smiles = smiles
        self.labels = labels
        # Default converter is to call _smiles_to_graph
        self.converter_fn = converter_fn if converter_fn is not None else self._smiles_to_graph

    def _smiles_to_graph(self, smi: str) -> Optional[GeometricData]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        atom_feats: List[List[int]] = []
        for atom in mol.GetAtoms():
            feat: List[int] = []
            # One‑hot encode atomic number up to 100
            atomic_num = atom.GetAtomicNum()
            one_hot = [0] * 101
            if atomic_num <= 100:
                one_hot[atomic_num] = 1
            feat.extend(one_hot)
            # One‑hot encode total degree (0–5)
            deg = atom.GetTotalDegree()
            feat.extend([1 if deg == i else 0 for i in range(6)])
            # Formal charge (–2 to +2)
            charge = atom.GetFormalCharge()
            feat.extend([1 if charge == i else 0 for i in range(-2, 3)])
            # Aromatic flag
            feat.append(int(atom.GetIsAromatic()))
            # Number of hydrogens (0–4)
            num_h = atom.GetTotalNumHs()
            feat.extend([1 if num_h == i else 0 for i in range(5)])
            atom_feats.append(feat)
        x = torch.tensor(atom_feats, dtype=torch.float)
        # Build edge index (bidirectional)
        edges: List[Tuple[int, int]] = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.append((i, j))
            edges.append((j, i))
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
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


def collate_graphs(
    batch: List[Tuple[Optional[GeometricData], torch.Tensor]]
) -> Tuple[Any, Any]:
    """Collate a batch of graph samples, filtering out None graphs.

    Parameters
    ----------
    batch : list of tuples
        Each element contains a graph and its label.

    Returns
    -------
    (Batch, Tensor)
        A PyTorch Geometric Batch object containing all valid graphs and a
        tensor of stacked labels.
    """
    graphs, labels = zip(*batch)
    if torch is None or torch_geometric is None:
        raise ImportError(
            "PyTorch and PyTorch Geometric must be installed to collate graphs."
        )
    valid_indices = [i for i, g in enumerate(graphs) if g is not None]
    if not valid_indices:
        raise ValueError("All graphs in the batch are None")
    graphs = [graphs[i] for i in valid_indices]
    labels = torch.stack([labels[i] for i in valid_indices])
    batch_data = torch_geometric.data.Batch.from_data_list(graphs)
    return batch_data, labels
