"""
Classical machine learning models for EUOS25.

Cross-validated training routines for:
- RandomForest
- ExtraTrees
- PLSRegression (probability-like outputs via clipping)
- Optional: CatBoost, XGBoost, LightGBM

Each training function returns:
    (models_per_fold, auc_scores, oof_pred)

OOF = Out-Of-Fold predictions:
Every training sample gets a prediction from a model that did NOT train on it.

Notes:
- Multi-task dataset (n_samples, n_tasks). We train one binary model per task.
- CV stratification uses a single label: whether the sample has ANY positive task.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.cross_decomposition import PLSRegression

# Optional imports
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

try:
    from lightgbm import LGBMClassifier  # type: ignore
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# -----------------------------
# Helpers
# -----------------------------
def _compute_class_weight_sklearn(y: np.ndarray) -> Optional[Dict[int, float]]:
    """
    sklearn-style class_weight using the classic "balanced" formula:
        w_c = n_samples / (n_classes * n_samples_in_class_c)
    """
    y = y.astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    n = pos + neg
    if pos == 0 or neg == 0:
        return None
    return {0: n / (2.0 * neg), 1: n / (2.0 * pos)}


def _compute_scale_pos_weight(y: np.ndarray) -> Optional[float]:
    """For XGBoost/LightGBM: scale_pos_weight = neg/pos."""
    y = y.astype(int)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    return float(neg) / float(pos)


def _cv_split_indices(y: np.ndarray, n_splits: int, random_state: int):
    """Stratify by whether a sample has ANY positive task."""
    stratify_label = (y.sum(axis=1) > 0).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return skf, stratify_label


def _init_oof(n_samples: int, n_tasks: int) -> np.ndarray:
    """OOF matrix filled with NaNs so we can sanity-check coverage."""
    return np.full((n_samples, n_tasks), np.nan, dtype=np.float32)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC can fail if fold has only one class."""
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def _catboost_class_weights_from_sklearn(class_weight: Optional[Dict[int, float]]) -> Optional[List[float]]:
    """
    Patch: CatBoost prefers list weights [w0, w1] (not sklearn dict).
    """
    if class_weight is None:
        return None
    w0 = float(class_weight.get(0, 1.0))
    w1 = float(class_weight.get(1, 1.0))
    return [w0, w1]


def _fit_lgbm_with_early_stopping(
    model: "LGBMClassifier",
    X_train,
    y_train,
    X_val,
    y_val,
    early_stopping_rounds: int,
):
    """
    Patch: robust early stopping across LightGBM versions.
    """
    try:
        import lightgbm as lgb  # type: ignore
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=callbacks,
        )
    except Exception:
        # Older versions use early_stopping_rounds param
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )


# -----------------------------
# RandomForest
# -----------------------------
def train_random_forest_cv(
    X: np.ndarray,
    y: np.ndarray,
    label_cols: List[str],
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], np.ndarray]:
    skf, stratify_label = _cv_split_indices(y, n_splits=n_splits, random_state=random_state)

    n_samples = X.shape[0]
    n_tasks = len(label_cols)
    oof_pred = _init_oof(n_samples, n_tasks)

    auc_scores: Dict[str, List[float]] = {t: [] for t in label_cols}
    models_per_fold: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, stratify_label)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_models: Dict[str, Any] = {}
        for task_idx, task in enumerate(label_cols):
            y_train_task = y_train[:, task_idx].astype(int)
            y_val_task = y_val[:, task_idx].astype(int)

            class_weight = _compute_class_weight_sklearn(y_train_task)

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=random_state + fold_idx,
                n_jobs=n_jobs,
            )
            model.fit(X_train, y_train_task)

            probas = model.predict_proba(X_val)[:, 1].astype(np.float32)
            oof_pred[val_idx, task_idx] = probas

            auc_scores[task].append(_safe_auc(y_val_task, probas))
            fold_models[task] = model

        models_per_fold.append(fold_models)

    return models_per_fold, auc_scores, oof_pred


# -----------------------------
# ExtraTrees
# -----------------------------
def train_extra_trees_cv(
    X: np.ndarray,
    y: np.ndarray,
    label_cols: List[str],
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], np.ndarray]:
    skf, stratify_label = _cv_split_indices(y, n_splits=n_splits, random_state=random_state)

    n_samples = X.shape[0]
    n_tasks = len(label_cols)
    oof_pred = _init_oof(n_samples, n_tasks)

    auc_scores: Dict[str, List[float]] = {t: [] for t in label_cols}
    models_per_fold: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, stratify_label)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_models: Dict[str, Any] = {}
        for task_idx, task in enumerate(label_cols):
            y_train_task = y_train[:, task_idx].astype(int)
            y_val_task = y_val[:, task_idx].astype(int)

            class_weight = _compute_class_weight_sklearn(y_train_task)

            model = ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=random_state + fold_idx,
                n_jobs=n_jobs,
            )
            model.fit(X_train, y_train_task)

            probas = model.predict_proba(X_val)[:, 1].astype(np.float32)
            oof_pred[val_idx, task_idx] = probas

            auc_scores[task].append(_safe_auc(y_val_task, probas))
            fold_models[task] = model

        models_per_fold.append(fold_models)

    return models_per_fold, auc_scores, oof_pred


# -----------------------------
# PLSRegression
# -----------------------------
def train_pls_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    label_cols: List[str],
    n_components: int = 10,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], np.ndarray]:
    skf, stratify_label = _cv_split_indices(y, n_splits=n_splits, random_state=random_state)

    n_samples = X.shape[0]
    n_tasks = len(label_cols)
    oof_pred = _init_oof(n_samples, n_tasks)

    auc_scores: Dict[str, List[float]] = {t: [] for t in label_cols}
    models_per_fold: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, stratify_label)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_models: Dict[str, Any] = {}
        for task_idx, task in enumerate(label_cols):
            y_train_task = y_train[:, task_idx].astype(float)
            y_val_task = y_val[:, task_idx].astype(float)

            pos = float((y_train_task == 1).sum())
            neg = float((y_train_task == 0).sum())
            if pos > 0 and neg > 0:
                w_pos = (pos + neg) / (2.0 * pos)
                w_neg = (pos + neg) / (2.0 * neg)
                sample_weight = np.where(y_train_task == 1, w_pos, w_neg).astype(float)
            else:
                sample_weight = None

            model = PLSRegression(n_components=n_components)

            try:
                model.fit(X_train, y_train_task, sample_weight=sample_weight)
            except TypeError:
                model.fit(X_train, y_train_task)

            pred = model.predict(X_val).ravel()
            probas = np.clip(pred, 0.0, 1.0).astype(np.float32)

            oof_pred[val_idx, task_idx] = probas
            auc_scores[task].append(_safe_auc(y_val_task, probas))
            fold_models[task] = model

        models_per_fold.append(fold_models)

    return models_per_fold, auc_scores, oof_pred


# -----------------------------
# CatBoost
# -----------------------------
def train_catboost_cv(
    X: np.ndarray,
    y: np.ndarray,
    label_cols: List[str],
    n_splits: int = 5,
    random_state: int = 42,
    n_iterations: int = 1500,
    learning_rate: float = 0.05,
    depth: int = 8,
    early_stopping_rounds: int = 100,
    **cb_params: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], np.ndarray]:
    if not HAS_CATBOOST:
        raise ImportError("CatBoost is not installed. Please install catboost to use this function.")

    skf, stratify_label = _cv_split_indices(y, n_splits=n_splits, random_state=random_state)

    n_samples = X.shape[0]
    n_tasks = len(label_cols)
    oof_pred = _init_oof(n_samples, n_tasks)

    auc_scores: Dict[str, List[float]] = {t: [] for t in label_cols}
    models_per_fold: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, stratify_label)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_models: Dict[str, Any] = {}
        for task_idx, task in enumerate(label_cols):
            y_train_task = y_train[:, task_idx].astype(int)
            y_val_task = y_val[:, task_idx].astype(int)

            cw = _compute_class_weight_sklearn(y_train_task)
            cb_cw = _catboost_class_weights_from_sklearn(cw)

            model = CatBoostClassifier(
                iterations=n_iterations,
                learning_rate=learning_rate,
                depth=depth,
                loss_function="Logloss",
                eval_metric="AUC",
                verbose=False,
                random_seed=random_state + fold_idx,
                class_weights=cb_cw,
                od_type="Iter",
                od_wait=early_stopping_rounds,
                **cb_params,
            )
            model.fit(X_train, y_train_task, eval_set=(X_val, y_val_task), use_best_model=True)

            probas = model.predict_proba(X_val)[:, 1].astype(np.float32)
            oof_pred[val_idx, task_idx] = probas
            auc_scores[task].append(_safe_auc(y_val_task, probas))
            fold_models[task] = model

        models_per_fold.append(fold_models)

    return models_per_fold, auc_scores, oof_pred


# -----------------------------
# XGBoost
# -----------------------------
def train_xgboost_cv(
    X: np.ndarray,
    y: np.ndarray,
    label_cols: List[str],
    n_splits: int = 5,
    random_state: int = 42,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    **xgb_params: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], np.ndarray]:
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is not installed. Please install xgboost to use this function.")

    skf, stratify_label = _cv_split_indices(y, n_splits=n_splits, random_state=random_state)

    n_samples = X.shape[0]
    n_tasks = len(label_cols)
    oof_pred = _init_oof(n_samples, n_tasks)

    auc_scores: Dict[str, List[float]] = {t: [] for t in label_cols}
    models_per_fold: List[Dict[str, Any]] = []

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": random_state,
    }
    base_params.update(xgb_params)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, stratify_label)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_models: Dict[str, Any] = {}
        for task_idx, task in enumerate(label_cols):
            y_train_task = y_train[:, task_idx].astype(int)
            y_val_task = y_val[:, task_idx].astype(int)

            spw = _compute_scale_pos_weight(y_train_task)
            params = dict(base_params)
            if spw is not None:
                params["scale_pos_weight"] = spw

            dtrain = xgb.DMatrix(X_train, label=y_train_task)
            dval = xgb.DMatrix(X_val, label=y_val_task)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, "valid")],
                verbose_eval=False,
                early_stopping_rounds=early_stopping_rounds,
            )

            probas = model.predict(dval).astype(np.float32)
            oof_pred[val_idx, task_idx] = probas
            auc_scores[task].append(_safe_auc(y_val_task, probas))
            fold_models[task] = model

        models_per_fold.append(fold_models)

    return models_per_fold, auc_scores, oof_pred


# -----------------------------
# LightGBM
# -----------------------------
def train_lightgbm_cv(
    X: np.ndarray,
    y: np.ndarray,
    label_cols: List[str],
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 5000,
    early_stopping_rounds: int = 100,
    **lgb_params: Any,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], np.ndarray]:
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM is not installed. Please install lightgbm to use this function.")

    skf, stratify_label = _cv_split_indices(y, n_splits=n_splits, random_state=random_state)

    n_samples = X.shape[0]
    n_tasks = len(label_cols)
    oof_pred = _init_oof(n_samples, n_tasks)

    auc_scores: Dict[str, List[float]] = {t: [] for t in label_cols}
    models_per_fold: List[Dict[str, Any]] = []

    base_params = {
        "objective": "binary",
        "learning_rate": 0.02,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_estimators": n_estimators,
        "n_jobs": -1,
        "verbose": -1,
    }
    base_params.update(lgb_params)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, stratify_label)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        fold_models: Dict[str, Any] = {}
        for task_idx, task in enumerate(label_cols):
            y_train_task = y_train[:, task_idx].astype(int)
            y_val_task = y_val[:, task_idx].astype(int)

            spw = _compute_scale_pos_weight(y_train_task)
            params = dict(base_params)
            params["random_state"] = random_state + fold_idx
            if spw is not None:
                params["scale_pos_weight"] = spw

            model = LGBMClassifier(**params)
            _fit_lgbm_with_early_stopping(model, X_train, y_train_task, X_val, y_val_task, early_stopping_rounds)

            probas = model.predict_proba(X_val)[:, 1].astype(np.float32)
            oof_pred[val_idx, task_idx] = probas
            auc_scores[task].append(_safe_auc(y_val_task, probas))
            fold_models[task] = model

        models_per_fold.append(fold_models)

    return models_per_fold, auc_scores, oof_pred
