"""Evaluation utilities for EUOS25 models.

This module defines functions to compute common evaluation metrics
for multi-task binary classification and optionally save ROC and PR
curve plots along with confusion matrices.

NEW:
- Find best probability thresholds per task (e.g., maximizing F1)
- Allow evaluation to use per-task thresholds instead of fixed 0.5
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Find the best threshold for a single binary task.

    Parameters
    ----------
    y_true : array (n_samples,)
        Ground truth 0/1 labels.
    y_prob : array (n_samples,)
        Predicted probabilities in [0,1].
    metric : str
        Which metric to maximize. Currently supports: "f1".
    grid : array, optional
        Thresholds to search. If None, uses 0.01..0.99.

    Returns
    -------
    best_t : float
        Threshold that maximizes the chosen metric.
    best_score : float
        Best metric value achieved at best_t.
    """
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    best_t = 0.5
    best_score = -1.0

    # If task has no positives or no negatives, thresholding is meaningless
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5, float("nan")

    for t in grid:
        y_hat = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_hat, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric={metric!r}. Use 'f1'.")
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, float(best_score)


def find_thresholds_per_task(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    label_cols: List[str],
    metric: str = "f1",
) -> Dict[str, float]:
    """Compute best threshold per task using y_true vs y_pred_proba."""
    thresholds: Dict[str, float] = {}
    for t_idx, t_name in enumerate(label_cols):
        t, _ = find_best_threshold(y_true[:, t_idx], y_pred_proba[:, t_idx], metric=metric)
        thresholds[t_name] = t
    return thresholds


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    label_cols: List[str],
    save_dir: Optional[str] = None,
    prefix: str = "",
    thresholds: Optional[Dict[str, float]] = None,
    learn_thresholds: bool = False,
    threshold_metric: str = "f1",
) -> Dict[str, Dict[str, float]]:
    """Compute evaluation metrics and optionally generate diagnostic plots.

    NEW behavior:
    - You can pass a dict `thresholds` like {"TaskA":0.13, ...}
    - Or set learn_thresholds=True to compute thresholds from (y_true, y_pred_proba)
      (useful for quick analysis, but best practice is: learn on OOF).

    Parameters
    ----------
    y_true : ndarray
        Ground truth binary labels of shape (n_samples, n_tasks).
    y_pred_proba : ndarray
        Predicted probabilities of shape (n_samples, n_tasks).
    label_cols : list of str
        Names of the tasks.
    save_dir : str, optional
        Directory in which to save ROC, PR and confusion matrix plots.
    prefix : str, default ""
        Prefix applied to saved filenames.
    thresholds : dict, optional
        Per-task threshold values used to binarize predictions for precision/recall/F1.
        If not provided and learn_thresholds=False, uses 0.5 for all tasks.
    learn_thresholds : bool, default False
        If True, compute per-task thresholds from (y_true, y_pred_proba)
        using `threshold_metric`. (Recommended only on OOF predictions.)
    threshold_metric : str, default "f1"
        Metric optimized when learning thresholds.

    Returns
    -------
    metrics : dict
        Dictionary keyed by task names containing AUC, PR-AUC, accuracy,
        precision, recall and F1 scores, plus the threshold used.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Optionally learn thresholds from these predictions
    if learn_thresholds:
        thresholds = find_thresholds_per_task(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            label_cols=label_cols,
            metric=threshold_metric,
        )

    # If still None: default 0.5 everywhere
    if thresholds is None:
        thresholds = {t: 0.5 for t in label_cols}

    # Save thresholds for record/reuse
    if save_dir:
        thr_df = pd.DataFrame(
            [{"task": t, "threshold": thresholds.get(t, 0.5)} for t in label_cols]
        )
        thr_df.to_csv(os.path.join(save_dir, f"{prefix}thresholds.csv"), index=False)

    metrics: Dict[str, Dict[str, float]] = {}

    for t_idx, t_name in enumerate(label_cols):
        y_true_task = y_true[:, t_idx]
        y_prob_task = y_pred_proba[:, t_idx]

        # AUC / PR-AUC from probabilities (threshold-free metrics)
        try:
            auc = roc_auc_score(y_true_task, y_prob_task)
            pr_auc = average_precision_score(y_true_task, y_prob_task)
        except ValueError:
            auc = float("nan")
            pr_auc = float("nan")

        # Thresholded metrics (depend on chosen threshold)
        thr = float(thresholds.get(t_name, 0.5))
        y_pred_bin = (y_prob_task >= thr).astype(int)

        acc = accuracy_score(y_true_task, y_pred_bin)
        precision = precision_score(y_true_task, y_pred_bin, zero_division=0)
        recall = recall_score(y_true_task, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_task, y_pred_bin, zero_division=0)

        metrics[t_name] = {
            "AUC": auc,
            "PR_AUC": pr_auc,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Threshold": thr,
        }

        # Curves are based on probabilities, not thresholds
        if save_dir:
            # ROC curve
            try:
                fpr, tpr, _ = roc_curve(y_true_task, y_prob_task)
                plt.figure()
                plt.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"ROC Curve for {t_name}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{prefix}roc_{t_name}.png"))
                plt.close()
            except ValueError:
                pass

            # PR curve
            try:
                prec, rec, _ = precision_recall_curve(y_true_task, y_prob_task)
                plt.figure()
                plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.3f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"Precision-Recall Curve for {t_name}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{prefix}pr_{t_name}.png"))
                plt.close()
            except ValueError:
                pass

            # Confusion matrix at chosen threshold
            cm = confusion_matrix(y_true_task, y_pred_bin)
            plt.figure()
            plt.imshow(cm, interpolation="nearest")
            plt.title(f"Confusion Matrix for {t_name} (thr={thr:.2f})")
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ["Negative", "Positive"])
            plt.yticks(tick_marks, ["Negative", "Positive"])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}cm_{t_name}.png"))
            plt.close()

    return metrics
