# Top-level package for the EUOS25 modular toolkit.
"""
euos25
========

This package provides a modular toolkit for working with the EU‑OPENSCREEN
EUOS25 challenge dataset.  It exposes independent submodules for configuration,
I/O, feature generation, EDA plotting, model training and evaluation.  Each
component has a narrow scope to simplify maintenance and encourage reuse.

Typical usage is to import only the pieces you need.  For example:

.. code-block:: python

    from euos25.config import Euos25Config
    from euos25.features.classical import compute_feature_matrix
    from euos25.models.classical import train_random_forest_cv
    from euos25.eval import evaluate_predictions

"""

__all__ = [
    "config",
    "io",
    "features",
    "eda",
    "models",
    "eval",
    "ensemble",
]
