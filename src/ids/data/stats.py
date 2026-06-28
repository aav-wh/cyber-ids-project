"""
ids.data.stats
--------------
Dataset statistics and EDA helpers.

These functions produce the summary tables and quick sanity-check plots
used in notebooks/01_data_loading.ipynb and notebooks/02_preprocessing.ipynb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def class_distribution(
    df: pd.DataFrame,
    label_col: str = "BinaryLabel",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Return a DataFrame summarising class counts and percentages.

    Parameters
    ----------
    df        : cleaned dataframe with a label column
    label_col : name of the target column

    Returns
    -------
    pd.DataFrame with columns: class, count, pct
    """
    counts = df[label_col].value_counts()
    pcts   = (counts / len(df) * 100).round(2)
    result = pd.DataFrame({
        "class":   counts.index,
        "count":   counts.values,
        "pct":     pcts.values,
    })

    if verbose:
        print(f"\nClass distribution ({label_col}):")
        print(result.to_string(index=False))

    return result


def attack_type_distribution(
    df: pd.DataFrame,
    label_col: str = "Label",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Return per-attack-type counts from the multi-class label column.

    Parameters
    ----------
    df        : cleaned dataframe
    label_col : multi-class label column (e.g. 'Label')

    Returns
    -------
    pd.DataFrame sorted by count descending
    """
    counts = df[label_col].value_counts().reset_index()
    counts.columns = ["attack_type", "count"]
    counts["pct"] = (counts["count"] / len(df) * 100).round(3)

    if verbose:
        print("\nAttack type distribution:")
        print(counts.to_string(index=False))

    return counts


def feature_summary(
    df: pd.DataFrame,
    label_col: str = "BinaryLabel",
) -> pd.DataFrame:
    """
    Descriptive statistics per feature, stratified by class.

    Returns
    -------
    pd.DataFrame: mean/std/min/max for each numeric feature, per class
    """
    drop_cols = [c for c in ["Label", "BinaryLabel"] if c in df.columns]
    feat_cols = [c for c in df.columns if c not in drop_cols]

    groups = df.groupby(label_col)[feat_cols]
    summary = groups.agg(["mean", "std", "min", "max"])
    return summary


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report NaN and Inf counts per column.

    Returns
    -------
    pd.DataFrame with columns: feature, nan_count, inf_count, total_bad, pct_bad
    """
    rows = []
    for col in df.select_dtypes(include=[np.number]).columns:
        nan_count = int(df[col].isna().sum())
        inf_count = int(np.isinf(df[col]).sum())
        total_bad = nan_count + inf_count
        pct_bad   = round(total_bad / len(df) * 100, 4)
        if total_bad > 0:
            rows.append({
                "feature":    col,
                "nan_count":  nan_count,
                "inf_count":  inf_count,
                "total_bad":  total_bad,
                "pct_bad":    pct_bad,
            })

    return pd.DataFrame(rows).sort_values("total_bad", ascending=False)


def per_day_summary(
    dfs: dict[str, pd.DataFrame],
    label_col: str = "BinaryLabel",
) -> pd.DataFrame:
    """
    Build a one-row-per-day summary table.

    Parameters
    ----------
    dfs       : dict mapping day name → cleaned DataFrame
    label_col : binary label column

    Returns
    -------
    pd.DataFrame with rows: day, n_rows, n_attacks, n_benign, attack_pct
    """
    rows = []
    for day, df in dfs.items():
        n_rows    = len(df)
        n_attacks = int((df[label_col] == "ATTACK").sum())
        n_benign  = n_rows - n_attacks
        rows.append({
            "day":        day,
            "n_rows":     n_rows,
            "n_attacks":  n_attacks,
            "n_benign":   n_benign,
            "attack_pct": round(n_attacks / n_rows * 100, 2) if n_rows else 0.0,
        })

    return pd.DataFrame(rows)


def correlation_with_label(
    df: pd.DataFrame,
    label_col: str = "BinaryLabel",
    top_n: int = 20,
) -> pd.Series:
    """
    Pearson correlation between each numeric feature and the binary label.

    The label is encoded as 0=BENIGN, 1=ATTACK for correlation purposes.

    Returns
    -------
    pd.Series of |correlation| values, sorted descending
    """
    df_enc = df.copy()
    df_enc["_target"] = (df_enc[label_col] == "ATTACK").astype(int)
    numeric_cols = df_enc.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "_target"]

    corrs = df_enc[numeric_cols].corrwith(df_enc["_target"]).abs()
    return corrs.sort_values(ascending=False).head(top_n)
