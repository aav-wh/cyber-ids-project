"""
ids.explainability.shap_analysis
---------------------------------
SHAP-based model explainability helpers.

Extracted from notebooks/07_shap_explainability.ipynb.
Provides functions to compute SHAP values and return serialisable
summaries for the dashboard API endpoint.
"""

import os

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier


def compute_shap_values(
    rf_model: RandomForestClassifier,
    X_sample: np.ndarray,
    check_additivity: bool = False,
) -> shap.Explainer:
    """
    Create a TreeExplainer and compute SHAP values for a sample.

    Uses TreeExplainer which is exact and fast for tree-based models — no
    approximation needed. check_additivity=False suppresses the slow
    consistency check which is not needed for production use.

    Parameters
    ----------
    rf_model         : fitted RandomForestClassifier
    X_sample         : scaled feature matrix to explain, shape [n, n_features]
    check_additivity : set True only for debugging (slow)

    Returns
    -------
    shap.Explainer with .shap_values() already computed
    """
    explainer = shap.TreeExplainer(rf_model)
    return explainer, explainer.shap_values(X_sample, check_additivity=check_additivity)


def top_features_by_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    class_index: int = 0,
    top_n: int = 20,
) -> list[dict]:
    """
    Return the top-N features by mean absolute SHAP value.

    Parameters
    ----------
    shap_values   : array of shape [n_samples, n_features] or
                    [n_classes, n_samples, n_features] (multi-output)
    feature_names : ordered feature name list
    class_index   : which class to explain (0 = ATTACK)
    top_n         : number of features to return

    Returns
    -------
    List of dicts: [{"feature": str, "mean_abs_shap": float}, ...]
    sorted descending by importance
    """
    # Handle both 2-D (binary shap_values) and 3-D (multi-class shap_values)
    if isinstance(shap_values, list):
        sv = np.array(shap_values[class_index])
    elif shap_values.ndim == 3:
        sv = shap_values[class_index]
    else:
        sv = shap_values

    mean_abs = np.abs(sv).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:top_n]

    return [
        {
            "feature":       feature_names[i],
            "mean_abs_shap": round(float(mean_abs[i]), 6),
        }
        for i in order
    ]


def shap_summary_for_api(
    rf_model: RandomForestClassifier,
    X_sample: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    class_index: int = 0,
) -> list[dict]:
    """
    One-shot helper used by the dashboard /api/shap endpoint.

    Computes SHAP values for a subsample and returns a serialisable list of
    the top-N most important features.

    Parameters
    ----------
    rf_model      : fitted RF model
    X_sample      : scaled feature matrix (subsample recommended, <= 500 rows)
    feature_names : ordered feature names
    top_n         : number of features to return
    class_index   : 0 = ATTACK, 1 = BENIGN

    Returns
    -------
    List of {"feature": str, "mean_abs_shap": float} dicts
    """
    _, shap_values = compute_shap_values(rf_model, X_sample)
    return top_features_by_shap(shap_values, feature_names, class_index, top_n)
