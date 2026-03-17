"""Data augmentation utilities for molecular SMILES.

This module defines functions to generate tautomeric variants of molecules,
produce randomised SMILES strings by shuffling atom orders and combine both
strategies to expand a dataset.  These augmentations can help improve model
robustness by exposing algorithms to different representations of the same
molecule.
"""

from __future__ import annotations

import random
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def enumerate_tautomers(smiles: List[str], max_tautomers: int = 5) -> List[List[str]]:
    """Enumerate tautomeric forms for each SMILES up to a maximum number.

    Parameters
    ----------
    smiles : list of str
        Input canonical SMILES strings.
    max_tautomers : int, default 5
        Maximum number of tautomeric forms to generate per molecule.

    Returns
    -------
    list of list of str
        A list with one element per input SMILES.  Each element is a list
        of canonical SMILES strings (one per enumerated tautomer).  If
        tautomers cannot be enumerated for a molecule, the original SMILES
        is returned.
    """
    enumerated: List[List[str]] = []
    enumerator = rdMolStandardize.TautomerEnumerator()
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            enumerated.append([smi])
            continue
        tautomers = enumerator.Enumerate(mol)
        smi_list: List[str] = []
        for t in tautomers:
            smi_form = Chem.MolToSmiles(t, canonical=True)
            if smi_form not in smi_list:
                smi_list.append(smi_form)
            if len(smi_list) >= max_tautomers:
                break
        if not smi_list:
            smi_list.append(smi)
        enumerated.append(smi_list)
    return enumerated


def randomize_smiles(smi: str, num_variants: int = 5) -> List[str]:
    """Generate random SMILES variants by permuting atom order.

    The original canonical SMILES is always returned as the first element
    followed by `num_variants - 1` randomised strings.

    Parameters
    ----------
    smi : str
        Input canonical SMILES.
    num_variants : int, default 5
        Total number of SMILES strings to return (including the original).

    Returns
    -------
    list of str
        A list of randomised SMILES strings.
    """
    variants = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=True)]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return variants
    for _ in range(num_variants - 1):
        atoms = list(range(mol.GetNumAtoms()))
        random.shuffle(atoms)
        new_mol = Chem.RenumberAtoms(mol, atoms)
        variants.append(Chem.MolToSmiles(new_mol, canonical=False))
    return variants


def augment_dataset_with_variants(
    smiles: List[str],
    num_variants: int = 3,
    use_tautomers: bool = True,
) -> List[List[str]]:
    """Combine tautomer enumeration and SMILES randomisation to create variants.

    For each canonical SMILES, the canonical form is always included as the
    first element of the returned list.  Additional variants are drawn from
    tautomer enumeration and randomised SMILES until `num_variants` unique
    strings have been collected.

    Parameters
    ----------
    smiles : list of str
        Canonical SMILES for molecules.
    num_variants : int, default 3
        Total number of variants desired per molecule.
    use_tautomers : bool, default True
        Whether to include tautomeric variants.  If false, only randomised
        SMILES strings are generated.

    Returns
    -------
    variants_per_molecule : list of list of str
        For each input SMILES, a list of variant SMILES strings.
    """
    variants_per_mol: List[List[str]] = []
    taut_lists = (
        enumerate_tautomers(smiles, max_tautomers=num_variants)
        if use_tautomers
        else [[s] for s in smiles]
    )
    for canonical, taut_list in zip(smiles, taut_lists):
        variants: List[str] = []
        # Always include canonical
        variants.append(
            Chem.MolToSmiles(Chem.MolFromSmiles(canonical), canonical=True)
        )
        # Add tautomeric forms (excluding canonical) up to the desired count
        for t_smi in taut_list:
            if t_smi != canonical and len(variants) < num_variants:
                variants.append(t_smi)
        # Fill remaining slots with randomised SMILES
        attempts = 0
        while len(variants) < num_variants:
            rnds = randomize_smiles(canonical, num_variants=2)
            for v in rnds:
                if v not in variants and len(variants) < num_variants:
                    variants.append(v)
            attempts += 1
            if attempts > 5 * num_variants:
                break
        variants_per_mol.append(variants)
    return variants_per_mol
