"""Configuration classes and constants for the EUOS25 toolkit.

The :mod:`euos25.config` module defines simple dataclasses to centralise
defaults for dataset columns, fingerprint parameters and cross‑validation
settings.  Users can override these defaults when constructing pipelines.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Euos25Config:
    """Dataclass grouping together default parameters for EUOS25 workflows.

    Attributes
    ----------
    smiles_col : str
        Name of the column containing canonical SMILES strings.
    label_cols : List[str]
        List of task names for the multi‑label classification problem.
    descriptor_variance_cutoff : float
        Variance threshold below which descriptor features will be removed.
    radius : int
        Radius for ECFP fingerprint computation.
    n_bits : int
        Number of bits for ECFP fingerprints.
    include_maccs : bool
        Whether to append MACCS keys to the feature matrix.
    n_splits : int
        Number of folds for cross‑validation.
    random_state : int
        Seed used for shuffling in cross‑validation routines.
    """

    smiles_col: str = "SMILES"
    label_cols: List[str] = field(default_factory=list)
    descriptor_variance_cutoff: float = 0.0
    radius: int = 2
    n_bits: int = 2048
    include_maccs: bool = True
    n_splits: int = 5
    random_state: int = 42

    def copy(self, **overrides: Optional[object]) -> "Euos25Config":
        """Return a copy of this configuration, optionally overriding fields.

        Parameters
        ----------
        overrides : dict
            Keyword arguments matching dataclass fields to override.

        Returns
        -------
        Euos25Config
            A new instance with the same values except where overridden.
        """
        params = self.__dict__.copy()
        params.update(overrides)
        return Euos25Config(**params)  # type: ignore[arg-type]
