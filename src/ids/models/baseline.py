"""
ids.models.baseline
-------------------
Baseline classifiers for benchmarking.

These simple models provide a performance floor for comparison.
If our RF/IF ensemble does not significantly outperform them,
something is wrong with the pipeline.

Baselines:
  - DummyClassifier (majority-class / stratified)
  - DecisionTree (shallow, single tree)
  - LogisticRegression (linear boundary)
  - KNeighborsClassifier (lazy, distance-based)
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def evaluate_baseline(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Fit a baseline model and return its evaluation metrics.

    Parameters
    ----------
    name     : display name for the model
    model    : unfitted sklearn estimator
    X_train, y_train, X_test, y_test : splits

    Returns
    -------
    dict with keys: name, precision, recall, f1, auc
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "name":      name,
        "precision": round(float(precision_score(y_test, preds, average="macro", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, preds, average="macro", zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, preds, average="macro", zero_division=0)), 4),
    }

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)[:, 0]  # ATTACK probability
            metrics["auc"] = round(float(roc_auc_score(y_test, proba)), 4)
        except Exception:
            metrics["auc"] = None
    else:
        metrics["auc"] = None

    return metrics


def run_all_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True,
) -> list[dict]:
    """
    Evaluate all predefined baselines and return a comparison list.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : scaled splits with integer labels

    Returns
    -------
    List of metric dicts, one per baseline
    """
    baselines = [
        ("DummyClassifier (majority)",   DummyClassifier(strategy="most_frequent")),
        ("DummyClassifier (stratified)",  DummyClassifier(strategy="stratified", random_state=42)),
        ("DecisionTree (depth=5)",        DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("DecisionTree (depth=10)",       DecisionTreeClassifier(max_depth=10, random_state=42)),
        ("LogisticRegression",            LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)),
        ("KNN (k=5)",                     KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ]

    results = []
    for name, model in baselines:
        result = evaluate_baseline(name, model, X_train, y_train, X_test, y_test)
        results.append(result)
        if verbose:
            print(f"  {name:<40} F1={result['f1']:.4f}  Recall={result['recall']:.4f}")

    return results
