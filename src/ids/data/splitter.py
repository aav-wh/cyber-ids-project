"""
ids.data.splitter
-----------------
Deterministic train/test splitting utilities.

The CICIDS2017 split strategy used in this project is temporal:
  Train: Monday, Tuesday, Wednesday  (earlier in the week)
  Test:  Thursday, Friday            (later in the week)

This mirrors realistic deployment where a model is trained on historical
traffic and evaluated on newer, unseen data — avoiding temporal leakage.

Additional helpers provide stratified random splits for ablation studies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def temporal_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str = "BinaryLabel",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Formalise the pre-loaded temporal split into X/y pairs.

    Parameters
    ----------
    train_df  : dataframe from Monday–Wednesday CSVs (already loaded)
    test_df   : dataframe from Thursday–Friday CSVs
    label_col : binary target column name

    Returns
    -------
    X_train, X_test, y_train, y_test  (all pd.DataFrame/Series)
    """
    drop_cols = [c for c in ["Label", "BinaryLabel"] if c in train_df.columns]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[label_col]
    X_test  = test_df.drop(columns=drop_cols)
    y_test  = test_df[label_col]

    if verbose:
        print(f"Train : {X_train.shape[0]:,} rows | "
              f"Attack={sum(y_train == 'ATTACK'):,}, Benign={sum(y_train == 'BENIGN'):,}")
        print(f"Test  : {X_test.shape[0]:,} rows  | "
              f"Attack={sum(y_test == 'ATTACK'):,}, Benign={sum(y_test == 'BENIGN'):,}")

    return X_train, X_test, y_train, y_test


def stratified_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple:
    """
    Stratified random train/test split preserving class proportions.

    Used for ablation studies and cross-validation experiments.

    Parameters
    ----------
    X, y         : feature matrix and labels
    test_size    : fraction reserved for test (0–1)
    random_state : reproducibility seed

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    if verbose:
        print(f"[StratifiedSplit] train={len(y_tr):,}, test={len(y_te):,}")

    return X_tr, X_te, y_tr, y_te


def cross_day_splits(
    dfs: dict[str, pd.DataFrame],
    label_col: str = "BinaryLabel",
) -> list[dict]:
    """
    Generate per-day evaluation splits for cross-day drift analysis.

    Each element of the returned list represents one test day evaluated
    against the full training set (Mon–Wed).

    Parameters
    ----------
    dfs       : mapping of day-name → cleaned DataFrame
    label_col : binary target column

    Returns
    -------
    list of dicts: [{"day": str, "X_test": ndarray, "y_test": ndarray}, ...]
    """
    from ids.data.loader import TRAIN_FILES

    splits = []
    for day, df in dfs.items():
        drop_cols = [c for c in ["Label", "BinaryLabel"] if c in df.columns]
        splits.append({
            "day":    day,
            "X_test": df.drop(columns=drop_cols).values,
            "y_test": df[label_col].values,
        })

    return splits


def save_split_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    save_dir: str,
) -> None:
    """Persist train/test index arrays so experiments are exactly reproducible."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(save_dir, "test_idx.npy"), test_idx)
