"""
ids.evaluation.bootstrap
------------------------
Bootstrap confidence intervals for classifier performance metrics.

Bootstrap resampling gives us uncertainty estimates for point metrics
like F1 and AUC without requiring a held-out validation set.

These are reported in notebooks/09_pr_optimal_threshold.ipynb and
notebooks/10_critical_analysis.ipynb.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "f1",
    n_iter: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    y_proba: np.ndarray | None = None,
) -> dict:
    """
    Compute a bootstrap confidence interval for a single metric.

    Parameters
    ----------
    y_true, y_pred : true and predicted integer labels
    metric         : 'f1' | 'precision' | 'recall' | 'auc'
    n_iter         : number of bootstrap resamples
    alpha          : significance level (0.05 → 95% CI)
    random_state   : seed
    y_proba        : predicted probabilities (required for 'auc')

    Returns
    -------
    dict: {metric, point_estimate, mean_bootstrap, ci_lower, ci_upper, std}
    """
    rng = np.random.default_rng(random_state)
    n   = len(y_true)

    fn_map = {
        "f1":        lambda yt, yp, ypr: f1_score(yt, yp, average="macro", zero_division=0),
        "precision": lambda yt, yp, ypr: precision_score(yt, yp, average="macro", zero_division=0),
        "recall":    lambda yt, yp, ypr: recall_score(yt, yp, average="macro", zero_division=0),
        "auc":       lambda yt, yp, ypr: roc_auc_score(yt, ypr) if ypr is not None else float("nan"),
    }

    if metric not in fn_map:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: {list(fn_map)}")

    score_fn = fn_map[metric]

    # Point estimate on the full set
    point = float(score_fn(y_true, y_pred, y_proba))

    # Bootstrap samples
    scores = []
    for _ in range(n_iter):
        idx  = rng.integers(0, n, n)
        yt   = y_true[idx]
        yp   = y_pred[idx]
        ypr  = y_proba[idx] if y_proba is not None else None
        try:
            scores.append(float(score_fn(yt, yp, ypr)))
        except Exception:
            pass  # skip degenerate samples (e.g. only one class)

    scores = np.array(scores)
    ci_lo  = float(np.percentile(scores, alpha / 2 * 100))
    ci_hi  = float(np.percentile(scores, (1 - alpha / 2) * 100))

    return {
        "metric":          metric,
        "point_estimate":  round(point, 6),
        "mean_bootstrap":  round(float(scores.mean()), 6),
        "ci_lower":        round(ci_lo, 6),
        "ci_upper":        round(ci_hi, 6),
        "std":             round(float(scores.std()), 6),
        "n_iter":          n_iter,
        "confidence_level": f"{int((1 - alpha) * 100)}%",
    }


def bootstrap_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_iter: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, dict]:
    """
    Compute bootstrap CIs for all standard metrics at once.

    Returns
    -------
    dict mapping metric name → bootstrap result dict
    """
    metrics = ["f1", "precision", "recall"]
    if y_proba is not None:
        metrics.append("auc")

    return {
        m: bootstrap_metric(
            y_true, y_pred, metric=m,
            n_iter=n_iter, alpha=alpha,
            random_state=random_state,
            y_proba=y_proba,
        )
        for m in metrics
    }
