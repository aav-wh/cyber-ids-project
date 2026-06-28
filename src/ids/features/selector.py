"""
ids.features.selector
---------------------
Feature selection utilities: RFE, mutual information, and variance thresholding.

Used in notebooks/12_feature_selection.ipynb to identify the minimal
feature subset that preserves detection performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)


def variance_filter(
    X: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.01,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Remove features with variance below threshold.

    Low-variance features carry almost no information — e.g. a flag
    that is always 0 contributes nothing to classification.

    Parameters
    ----------
    X            : scaled feature matrix [n_samples, n_features]
    feature_names: ordered feature names
    threshold    : features with var < threshold are dropped

    Returns
    -------
    X_filtered, selected_feature_names
    """
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    mask = selector.get_support()
    selected = [f for f, m in zip(feature_names, mask) if m]

    if verbose:
        n_removed = len(feature_names) - len(selected)
        print(f"[VarianceFilter] Removed {n_removed} low-variance features. "
              f"Kept {len(selected)}/{len(feature_names)}.")

    return X_filtered, selected


def mutual_info_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    k: int = 30,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Select top-k features by mutual information with the target.

    MI measures how much knowing a feature reduces uncertainty about the label.
    It captures non-linear relationships that Pearson correlation misses.

    Parameters
    ----------
    X, y          : feature matrix and integer labels
    feature_names : ordered feature names
    k             : number of features to keep

    Returns
    -------
    X_selected, selected_names, mi_scores (full array, descending order)
    """
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=random_state),
        k=k,
    )
    X_selected = selector.fit_transform(X, y)
    mask       = selector.get_support()
    selected   = [f for f, m in zip(feature_names, mask) if m]
    mi_scores  = selector.scores_

    if verbose:
        top5 = sorted(zip(feature_names, mi_scores), key=lambda x: -x[1])[:5]
        print(f"[MutualInfo] Top-5 features: {[f for f, _ in top5]}")

    return X_selected, selected, mi_scores


def rfe_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_features: int = 30,
    step: int = 5,
    rf_kwargs: dict | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Recursive Feature Elimination using a RandomForest estimator.

    RFE iteratively fits the model and prunes the least important feature(s)
    until n_features remain.

    Parameters
    ----------
    X, y          : feature matrix and integer labels
    feature_names : ordered feature names
    n_features    : target number of features to select
    step          : features removed per iteration
    rf_kwargs     : keyword arguments passed to RandomForestClassifier

    Returns
    -------
    X_selected, selected_names, ranking (1 = selected)
    """
    kwargs = {"n_estimators": 50, "random_state": 42, "n_jobs": -1}
    if rf_kwargs:
        kwargs.update(rf_kwargs)

    estimator = RandomForestClassifier(**kwargs)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=step)
    rfe.fit(X, y)

    mask     = rfe.get_support()
    selected = [f for f, m in zip(feature_names, mask) if m]
    X_sel    = rfe.transform(X)

    if verbose:
        print(f"[RFE] Selected {len(selected)} features from {len(feature_names)}.")

    return X_sel, selected, rfe.ranking_


def importance_ranking(
    rf_model: RandomForestClassifier,
    feature_names: list[str],
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of RF feature importances (MDI) sorted descending.

    Parameters
    ----------
    rf_model      : fitted RandomForestClassifier
    feature_names : ordered feature names
    top_n         : if given, return only the top-N features

    Returns
    -------
    pd.DataFrame with columns: rank, feature, importance, cumulative_importance
    """
    imps   = rf_model.feature_importances_
    order  = np.argsort(imps)[::-1]
    df = pd.DataFrame({
        "rank":       range(1, len(order) + 1),
        "feature":    [feature_names[i] for i in order],
        "importance": imps[order].round(6),
    })
    df["cumulative_importance"] = df["importance"].cumsum().round(6)

    if top_n:
        df = df.head(top_n)

    return df.reset_index(drop=True)
