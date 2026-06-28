"""
ids.evaluation.cross_day
------------------------
Cross-day generalisation evaluation.

Evaluates the model trained on Mon–Wed against each test day individually
to reveal temporal degradation / concept drift patterns across the week.

Used in notebooks/05_cross_day_drift.ipynb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ids.evaluation.metrics import compute_full_metrics


def evaluate_per_day(
    rf_model: RandomForestClassifier,
    scaler: StandardScaler,
    le: LabelEncoder,
    day_splits: list[dict],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate the RF model on each day's data independently.

    Parameters
    ----------
    rf_model    : fitted RandomForestClassifier
    scaler      : fitted StandardScaler (from training data)
    le          : fitted LabelEncoder
    day_splits  : list of {"day": str, "X_test": ndarray, "y_test": ndarray}
                  where y_test contains string labels
    verbose     : print per-day results

    Returns
    -------
    pd.DataFrame with one row per day
    """
    rows = []

    for split in day_splits:
        day    = split["day"]
        X_test = split["X_test"]
        y_str  = split["y_test"]

        # Encode labels
        y_test = le.transform(y_str)

        # Scale using the TRAINING scaler (no re-fit)
        X_scaled = scaler.transform(X_test).astype(np.float32)

        # Predict
        preds  = rf_model.predict(X_scaled)
        probas = rf_model.predict_proba(X_scaled)[:, 0]

        metrics = compute_full_metrics(
            y_test, preds, y_proba=probas,
            label_names=list(le.classes_),
        )

        row = {
            "day":            day,
            "n_flows":        len(y_test),
            "n_attacks":      int((y_test == 0).sum()),
            "attack_pct":     round(float((y_test == 0).mean() * 100), 2),
            "precision":      metrics["precision"],
            "recall":         metrics["recall"],
            "f1":             metrics["f1"],
            "auc":            metrics.get("auc"),
            "detection_rate": metrics["detection_rate"],
            "fpr":            metrics["fpr"],
        }
        rows.append(row)

        if verbose:
            print(f"  {day:<45} F1={metrics['f1']:.4f}  "
                  f"Recall={metrics['recall']:.4f}  "
                  f"FPR={metrics['fpr']:.4f}")

    return pd.DataFrame(rows)


def drift_summary(per_day_df: pd.DataFrame, baseline_day: str) -> pd.DataFrame:
    """
    Compute metric deltas relative to a baseline day.

    Parameters
    ----------
    per_day_df   : output of evaluate_per_day
    baseline_day : name of the day to use as the reference

    Returns
    -------
    pd.DataFrame with additional Δ columns showing degradation from baseline
    """
    df = per_day_df.copy()
    baseline = df[df["day"] == baseline_day].iloc[0]

    for metric in ["f1", "recall", "precision", "fpr"]:
        if metric in df.columns:
            df[f"delta_{metric}"] = (df[metric] - baseline[metric]).round(4)

    return df
