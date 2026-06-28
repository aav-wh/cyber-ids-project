"""
ids.evaluation.metrics
----------------------
Extended evaluation metrics for the IDS.

Supplements sklearn's standard metrics with IDS-specific measures:
  - False Negative Rate (FNR) — proportion of attacks missed
  - False Positive Rate (FPR) — proportion of benign flows flagged
  - Detection Rate (DR)       — synonym for recall / True Positive Rate
  - Matthews Correlation Coefficient (MCC)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    label_names: list[str] | None = None,
    attack_label: int = 0,
) -> dict:
    """
    Compute the full suite of binary classification metrics for IDS evaluation.

    Parameters
    ----------
    y_true       : true integer labels
    y_pred       : predicted integer labels
    y_proba      : predicted probability of ATTACK (optional, for AUC)
    label_names  : class names for the classification report
    attack_label : integer index of the ATTACK class

    Returns
    -------
    dict with: accuracy, precision, recall, f1, mcc,
               fpr, fnr, detection_rate, auc (if y_proba given),
               report (classification report string)
    """
    cm = confusion_matrix(y_true, y_pred)

    # For binary confusion matrix:
    # attack_label=0 → row 0 = actual ATTACK, row 1 = actual BENIGN
    if cm.shape == (2, 2):
        if attack_label == 0:
            tp = cm[0, 0]   # ATTACK predicted as ATTACK
            fn = cm[0, 1]   # ATTACK predicted as BENIGN
            fp = cm[1, 0]   # BENIGN predicted as ATTACK
            tn = cm[1, 1]   # BENIGN predicted as BENIGN
        else:
            tp = cm[1, 1]
            fn = cm[1, 0]
            fp = cm[0, 1]
            tn = cm[0, 0]

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        dr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Detection Rate = Recall
    else:
        fpr = fnr = dr = None

    metrics = {
        "accuracy":        round(float(accuracy_score(y_true, y_pred)), 6),
        "precision":       round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "recall":          round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "f1":              round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "mcc":             round(float(matthews_corrcoef(y_true, y_pred)), 6),
        "fpr":             round(float(fpr), 6) if fpr is not None else None,
        "fnr":             round(float(fnr), 6) if fnr is not None else None,
        "detection_rate":  round(float(dr), 6)  if dr  is not None else None,
        "report":          classification_report(
                               y_true, y_pred,
                               target_names=label_names or ["ATTACK", "BENIGN"],
                               zero_division=0,
                           ),
    }

    if y_proba is not None:
        try:
            metrics["auc"] = round(float(roc_auc_score(y_true, y_proba)), 6)
        except ValueError:
            metrics["auc"] = None

    return metrics


def metrics_to_dataframe(metrics_dict: dict, model_name: str = "") -> pd.DataFrame:
    """
    Convert a metrics dict to a single-row DataFrame for easy comparison.

    Parameters
    ----------
    metrics_dict : output of compute_full_metrics
    model_name   : label to prefix the row

    Returns
    -------
    pd.DataFrame with one row
    """
    scalar_keys = ["accuracy", "precision", "recall", "f1", "mcc",
                   "fpr", "fnr", "detection_rate", "auc"]
    row = {"model": model_name}
    row.update({k: metrics_dict.get(k) for k in scalar_keys if k in metrics_dict})
    return pd.DataFrame([row])


def compare_models(*metric_dicts: tuple[str, dict]) -> pd.DataFrame:
    """
    Build a comparison table for multiple models.

    Parameters
    ----------
    *metric_dicts : tuples of (model_name, metrics_dict)

    Returns
    -------
    pd.DataFrame with one row per model, sorted by F1 descending
    """
    rows = [metrics_to_dataframe(d, name) for name, d in metric_dicts]
    df = pd.concat(rows, ignore_index=True)
    if "f1" in df.columns:
        df = df.sort_values("f1", ascending=False).reset_index(drop=True)
    return df
