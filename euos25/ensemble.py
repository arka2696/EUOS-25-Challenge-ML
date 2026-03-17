from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_decomposition import PLSRegression

try:
    from lightgbm import LGBMClassifier  # type: ignore
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier  # type: ignore
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import xgboost as xgb  # type: ignore
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def average_model_predictions(
    models_per_fold: List[Dict[str, Any]],
    X: np.ndarray,
    label_cols: List[str],
) -> np.ndarray:
    n_samples = X.shape[0]
    n_tasks = len(label_cols)
    preds = np.zeros((n_samples, n_tasks), dtype=np.float32)

    n_folds = 0
    for fold_models in models_per_fold:
        for t_idx, t_name in enumerate(label_cols):
            model = fold_models[t_name]

            if isinstance(model, (RandomForestClassifier, ExtraTreesClassifier)):
                probas = model.predict_proba(X)[:, 1]
            elif HAS_LGBM and isinstance(model, LGBMClassifier):
                probas = model.predict_proba(X)[:, 1]
            elif HAS_CATBOOST and isinstance(model, CatBoostClassifier):
                probas = model.predict_proba(X)[:, 1]
            elif HAS_XGBOOST and isinstance(model, xgb.Booster):
                dtest = xgb.DMatrix(X)
                probas = model.predict(dtest)
            elif isinstance(model, PLSRegression):
                probas = np.clip(model.predict(X).ravel(), 0.0, 1.0)
            else:
                raise TypeError(f"Unknown model type {type(model)}")

            preds[:, t_idx] += probas.astype(np.float32)

        n_folds += 1

    preds /= max(n_folds, 1)
    return preds
