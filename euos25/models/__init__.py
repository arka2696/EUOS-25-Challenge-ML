"""Model training subpackage for EUOS25.

This subpackage contains implementations of classical machine‑learning
algorithms (random forests, extra trees, PLS regression and optional CatBoost,
XGBoost and LightGBM) and deep learning architectures based on graph neural
networks.  Each module exposes a consistent interface to fit models via
cross‑validation and return both the models per fold and the associated
evaluation scores.
"""

from .classical import (
    train_random_forest_cv,
    train_extra_trees_cv,
    train_pls_regression_cv,
    train_catboost_cv,
    train_xgboost_cv,
    train_lightgbm_cv,
)
#from .gnn import train_gnn_multitask

__all__ = [
    "classical",                   #I have removed the "gnn"
    "train_random_forest_cv",
    "train_extra_trees_cv",
    "train_pls_regression_cv",
    "train_catboost_cv",
    "train_xgboost_cv",
    "train_lightgbm_cv",
    "train_gnn_multitask",
]
