"""
ids.models.threshold
--------------------
Threshold optimisation for the Random Forest classifier.

The default RF threshold (0.5) maximises accuracy but not necessarily
the metric that matters most for IDS (recall or F1).  This module
provides:
  - PR-curve based optimal threshold selection
  - F1-maximising threshold search
  - Bootstrap confidence intervals for the selected threshold
  - Threshold comparison table generation

Used in notebooks/09_pr_optimal_threshold.ipynb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve


def optimal_f1_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    attack_label: int = 0,
    resolution: int = 1000,
) -> tuple[float, float]:
    """
    Find the decision threshold that maximises macro-F1.

    Parameters
    ----------
    y_true       : true integer labels
    y_proba      : predicted probability of the ATTACK class
    attack_label : integer index of the ATTACK class (default 0)
    resolution   : number of threshold candidates to evaluate

    Returns
    -------
    (optimal_threshold, best_f1_score)
    """
    thresholds = np.linspace(0.01, 0.99, resolution)
    best_t = 0.5
    best_f1 = 0.0

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        # Remap: if attack_label==0, proba >= t means ATTACK
        if attack_label == 0:
            preds_mapped = np.where(preds, attack_label, 1)
        else:
            preds_mapped = np.where(preds, 1, attack_label)

        f1 = f1_score(y_true, preds_mapped, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t

    return round(float(best_t), 4), round(float(best_f1), 6)


def pr_curve_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> pd.DataFrame:
    """
    Build a precision/recall/F1 table across all PR-curve thresholds.

    Returns
    -------
    pd.DataFrame with columns: threshold, precision, recall, f1
    sorted by threshold ascending
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns len(thresholds) + 1 precision/recall values
    thresholds = np.append(thresholds, 1.0)

    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls + 1e-12),
        0.0,
    )

    return pd.DataFrame({
        "threshold": thresholds.round(4),
        "precision": precisions.round(4),
        "recall":    recalls.round(4),
        "f1":        f1s.round(4),
    })


def bootstrap_threshold_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_iter: int = 200,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict:
    """
    Bootstrap confidence interval for the F1-optimal threshold.

    Parameters
    ----------
    y_true, y_proba : true labels and predicted probabilities
    n_iter          : number of bootstrap resamples
    alpha           : significance level (0.05 → 95% CI)
    random_state    : seed

    Returns
    -------
    dict: {best_threshold, mean_threshold, ci_lower, ci_upper, best_f1}
    """
    rng = np.random.default_rng(random_state)
    n   = len(y_true)
    thresholds = []

    for _ in range(n_iter):
        idx  = rng.integers(0, n, n)
        yt   = y_true[idx]
        yp   = y_proba[idx]
        t, _ = optimal_f1_threshold(yt, yp)
        thresholds.append(t)

    thresholds = np.array(thresholds)
    ci_lo = float(np.percentile(thresholds, alpha / 2 * 100))
    ci_hi = float(np.percentile(thresholds, (1 - alpha / 2) * 100))

    best_t, best_f1 = optimal_f1_threshold(y_true, y_proba)

    return {
        "best_threshold": best_t,
        "mean_threshold": round(float(thresholds.mean()), 4),
        "ci_lower":       round(ci_lo, 4),
        "ci_upper":       round(ci_hi, 4),
        "best_f1":        best_f1,
        "n_iter":         n_iter,
    }


def threshold_comparison_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Compare a set of named thresholds on key metrics.

    Parameters
    ----------
    y_true, y_proba : labels and probabilities
    thresholds      : list of threshold values to compare;
                      defaults to [0.3, 0.4, 0.5, optimal]

    Returns
    -------
    pd.DataFrame with one row per threshold
    """
    from sklearn.metrics import precision_score, recall_score

    opt_t, _ = optimal_f1_threshold(y_true, y_proba)
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, opt_t]

    rows = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        rows.append({
            "threshold": round(t, 4),
            "precision": round(float(precision_score(y_true, preds, average="macro", zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, preds, average="macro", zero_division=0)), 4),
            "f1":        round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 4),
            "note":      "optimal" if abs(t - opt_t) < 1e-4 else "",
        })

    return pd.DataFrame(rows)
