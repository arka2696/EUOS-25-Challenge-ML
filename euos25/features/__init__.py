"""Feature generation subpackage for EUOS25.

This subpackage contains utilities for constructing numerical
representations of molecules and performing simple data augmentation
to expand the training set.  The classical module provides descriptor
and fingerprint features using RDKit, while the augmentation module
offers SMILES randomisation and tautomer enumeration.
"""

from .classical import compute_feature_matrix
from .augmentation import (
    enumerate_tautomers,
    randomize_smiles,
    augment_dataset_with_variants,
)

__all__ = [
    "classical",
    "augmentation",
    "graphs",
    "compute_feature_matrix",
    "enumerate_tautomers",
    "randomize_smiles",
    "augment_dataset_with_variants",
]
