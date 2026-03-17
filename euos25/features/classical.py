"""
Classical feature computation for EUOS25 (self-contained, RDKit-only).

Upgrades (patch):
- Adds a "descriptor_transform" mechanism so descriptors are standardized using
  TRAIN statistics and the same transform is applied to TEST (no leakage / no mismatch).
- Keeps backward compatibility: default returns (X, names)
- If return_descriptor_transform=True, returns (X, names, transform_dict)
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors


def _smiles_to_mol(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def _morgan_bits_array(mol, radius: int, n_bits: int, use_chirality: bool = False) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=use_chirality)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _morgan_counts_array(mol, radius: int, n_bits: int, clip: int = 10, use_chirality: bool = False) -> np.ndarray:
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits, useChirality=use_chirality)
    arr = np.zeros((n_bits,), dtype=np.float32)
    nz = fp.GetNonzeroElements()
    for k, v in nz.items():
        arr[int(k)] = float(min(int(v), clip))
    return arr


def _maccs_array(mol) -> np.ndarray:
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)  # should be 167
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


_DESCRIPTOR_FUNCS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("RingCount", Descriptors.RingCount),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NHOHCount", Descriptors.NHOHCount),
    ("NOCount", Descriptors.NOCount),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("MolMR", Descriptors.MolMR),
]


def _descriptor_vector(mol) -> np.ndarray:
    vals = []
    for _, fn in _DESCRIPTOR_FUNCS:
        try:
            v = fn(mol)
            if v is None:
                v = 0.0
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                v = 0.0
        except Exception:
            v = 0.0
        vals.append(float(v))
    return np.asarray(vals, dtype=np.float32)


def _standardize_with_params(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
    std_safe = np.where(std > 0, std, 1.0)
    X -= mean.astype(np.float32)
    X /= std_safe.astype(np.float32)


def compute_feature_matrix(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    radius: int = 2,
    n_bits: int = 2048,
    include_maccs: bool = True,
    descriptor_variance_cutoff: float = 0.0,
    *,
    radii: Optional[Sequence[int]] = None,
    use_counts: bool = False,
    count_clip: int = 10,
    include_descriptors: bool = True,
    standardize_descriptors: bool = True,
    use_chirality: bool = False,
    # NEW: pass this when computing TEST so it uses TRAIN transform
    descriptor_transform: Optional[Dict[str, Any]] = None,
    # NEW: return transform when computing TRAIN
    return_descriptor_transform: bool = False,
) -> Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, List[str], Dict[str, Any]]:
    if smiles_col not in df.columns:
        raise ValueError(f"compute_feature_matrix: missing column {smiles_col!r}")

    smiles_list = df[smiles_col].astype(str).tolist()
    n = len(smiles_list)

    if radii is None:
        radii = (2, 3)
    else:
        radii = tuple(int(r) for r in radii)
        if len(radii) == 0:
            radii = (radius,)

    maccs_len = 167 if include_maccs else 0
    raw_desc_len = len(_DESCRIPTOR_FUNCS) if include_descriptors else 0
    fp_len = len(radii) * n_bits

    # descriptor_transform determines which descriptor columns are kept
    keep_mask = None
    if include_descriptors and descriptor_transform is not None:
        keep_mask = np.asarray(descriptor_transform.get("keep_mask", None)) if "keep_mask" in descriptor_transform else None

    desc_len = int(keep_mask.sum()) if (include_descriptors and keep_mask is not None) else raw_desc_len
    n_feat = desc_len + fp_len + maccs_len

    X = np.zeros((n, n_feat), dtype=np.float32)
    bad = 0

    desc_names_all = [f"DESC_{name}" for name, _ in _DESCRIPTOR_FUNCS] if include_descriptors else []
    fp_names: List[str] = []
    for r in radii:
        prefix = f"ECFPcount_r{r}" if use_counts else f"ECFP_r{r}"
        fp_names.extend([f"{prefix}_{i}" for i in range(n_bits)])
    maccs_names = [f"MACCS_{i}" for i in range(167)] if include_maccs else []

    # If keep_mask exists, apply it to descriptor names too
    if include_descriptors and keep_mask is not None:
        desc_names = [desc_names_all[i] for i in range(raw_desc_len) if keep_mask[i]]
    else:
        desc_names = desc_names_all

    names = desc_names + fp_names + maccs_names

    # Fill rows
    for i, smi in enumerate(smiles_list):
        mol = _smiles_to_mol(smi)
        if mol is None:
            bad += 1
            continue

        offset = 0

        # descriptors
        if include_descriptors:
            d = _descriptor_vector(mol)  # raw_desc_len
            if keep_mask is not None:
                d = d[keep_mask]
            X[i, offset : offset + d.shape[0]] = d
            offset += d.shape[0]

        # fingerprints
        for r in radii:
            if use_counts:
                fp = _morgan_counts_array(mol, radius=r, n_bits=n_bits, clip=count_clip, use_chirality=use_chirality)
            else:
                fp = _morgan_bits_array(mol, radius=r, n_bits=n_bits, use_chirality=use_chirality).astype(np.float32)
            X[i, offset : offset + n_bits] = fp
            offset += n_bits

        # MACCS
        if include_maccs:
            maccs = _maccs_array(mol).astype(np.float32)
            if maccs.shape[0] != 167:
                raise RuntimeError(f"Unexpected MACCS length: {maccs.shape[0]} (expected 167)")
            X[i, offset : offset + 167] = maccs

    if bad:
        print(f"[compute_feature_matrix] WARNING: {bad} invalid SMILES -> zero vectors")

    # ---------- Descriptor transform ----------
    out_transform: Dict[str, Any] = {}

    if include_descriptors and raw_desc_len > 0:
        # If no transform passed, we're likely building TRAIN and need to fit it.
        if descriptor_transform is None:
            # Work on raw descriptor block before any keep_mask
            # If keep_mask is None here, X contains all raw_desc_len at front.
            desc_block = X[:, :raw_desc_len].copy()

            # variance cutoff
            if descriptor_variance_cutoff is not None and float(descriptor_variance_cutoff) > 0:
                var = desc_block.var(axis=0, dtype=np.float64)
                keep_mask_fit = var > float(descriptor_variance_cutoff)
                if keep_mask_fit.sum() == 0:
                    keep_mask_fit = np.ones((raw_desc_len,), dtype=bool)
            else:
                keep_mask_fit = np.ones((raw_desc_len,), dtype=bool)

            # rebuild X with kept descriptors
            kept_desc = desc_block[:, keep_mask_fit]
            rest = X[:, raw_desc_len:]
            X = np.concatenate([kept_desc, rest], axis=1).astype(np.float32)

            # update names accordingly
            kept_desc_names = [desc_names_all[i] for i in range(raw_desc_len) if keep_mask_fit[i]]
            names = kept_desc_names + fp_names + maccs_names

            # standardize kept descriptors using TRAIN stats
            if standardize_descriptors and kept_desc.shape[1] > 0:
                mean = kept_desc.mean(axis=0, dtype=np.float64).astype(np.float32)
                std = kept_desc.std(axis=0, dtype=np.float64).astype(np.float32)
                _standardize_with_params(X[:, : kept_desc.shape[1]], mean, std)
            else:
                mean = np.zeros((kept_desc.shape[1],), dtype=np.float32)
                std = np.ones((kept_desc.shape[1],), dtype=np.float32)

            out_transform = {
                "keep_mask": keep_mask_fit.astype(bool),
                "mean": mean,
                "std": std,
                "standardized": bool(standardize_descriptors),
            }

        else:
            # apply provided transform (TEST)
            if keep_mask is None:
                raise ValueError("descriptor_transform was provided but keep_mask is missing.")
            # descriptors are already kept during row fill
            desc_len_now = int(keep_mask.sum())
            if standardize_descriptors and descriptor_transform.get("standardized", True):
                mean = np.asarray(descriptor_transform["mean"], dtype=np.float32)
                std = np.asarray(descriptor_transform["std"], dtype=np.float32)
                _standardize_with_params(X[:, :desc_len_now], mean, std)

    if return_descriptor_transform:
        return X.astype(np.float32), names, out_transform
    return X.astype(np.float32), names
