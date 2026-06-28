"""
ids.evaluation.reports
----------------------
Report generation: classification reports, model comparison tables,
and CSV/JSON export helpers.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return sklearn's classification report as a structured DataFrame.

    Parameters
    ----------
    y_true, y_pred : integer label arrays
    label_names    : class names; defaults to ['ATTACK', 'BENIGN']

    Returns
    -------
    pd.DataFrame with columns: class, precision, recall, f1-score, support
    """
    from sklearn.metrics import classification_report as cr
    report = cr(
        y_true, y_pred,
        target_names=label_names or ["ATTACK", "BENIGN"],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for k, v in report.items():
        if isinstance(v, dict):
            rows.append({"class": k, **v})
    return pd.DataFrame(rows)


def save_metrics_csv(
    metrics_list: list[dict],
    save_path: str,
    verbose: bool = True,
) -> None:
    """
    Save a list of metric dicts to a CSV file.

    Parameters
    ----------
    metrics_list : list of metric dicts (e.g. from compute_full_metrics)
    save_path    : output CSV path
    """
    # Drop non-scalar fields (e.g. 'report')
    rows = []
    for m in metrics_list:
        row = {k: v for k, v in m.items() if not isinstance(v, str)}
        rows.append(row)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(save_path, index=False)
    if verbose:
        print(f"Metrics saved to: {save_path}")


def save_metrics_json(
    metrics: dict,
    save_path: str,
    verbose: bool = True,
) -> None:
    """Save a metrics dict as formatted JSON."""
    # Remove non-serialisable fields
    clean = {k: v for k, v in metrics.items()
             if not isinstance(v, (np.ndarray, str)) or k == "model"}
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(clean, f, indent=2)
    if verbose:
        print(f"Metrics JSON saved to: {save_path}")


def model_comparison_report(
    comparisons: list[dict],
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate a formatted model comparison report.

    Parameters
    ----------
    comparisons : list of dicts, each with 'model' key and metric keys
    save_path   : optional CSV path

    Returns
    -------
    Formatted pd.DataFrame sorted by F1 descending
    """
    df = pd.DataFrame(comparisons)
    metric_cols = [c for c in ["precision", "recall", "f1", "auc", "fpr", "fnr", "mcc"]
                   if c in df.columns]

    for col in metric_cols:
        df[col] = df[col].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (float, int)) and x is not None else "—"
        )

    if "f1" in df.columns:
        df = df.sort_values("f1", ascending=False).reset_index(drop=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df.to_csv(save_path, index=False)

    return df
