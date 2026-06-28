"""
ids.models.hyperopt
-------------------
Hyperparameter search utilities for Random Forest and Isolation Forest.

Uses sklearn's GridSearchCV and RandomizedSearchCV.
For the project's dataset size (2M+ rows) RandomizedSearch with n_iter=20
is sufficient to find near-optimal hyperparameters without exhaustive search.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


# ── Default search spaces ─────────────────────────────────────────────────────

RF_PARAM_GRID = {
    "n_estimators":     [50, 100, 150, 200, 300],
    "max_depth":        [None, 10, 20, 30],
    "min_samples_split":[2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     ["sqrt", "log2", None],
    "class_weight":     ["balanced", None],
}

IF_CONTAMINATION_GRID = [0.005, 0.01, 0.02, 0.05, 0.10]


def random_search_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_distributions: dict | None = None,
    n_iter: int = 20,
    cv: int = 3,
    scoring: str = "f1_macro",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
) -> RandomizedSearchCV:
    """
    Randomised hyperparameter search for RandomForestClassifier.

    Parameters
    ----------
    X_train, y_train     : scaled training data and integer labels
    param_distributions  : search space dict (defaults to RF_PARAM_GRID)
    n_iter               : number of parameter settings to sample
    cv                   : cross-validation folds
    scoring              : sklearn scoring string
    random_state         : seed

    Returns
    -------
    Fitted RandomizedSearchCV object (access .best_estimator_, .best_params_)
    """
    if param_distributions is None:
        param_distributions = RF_PARAM_GRID

    base_rf = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    cv_strat = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv_strat,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        return_train_score=False,
    )

    search.fit(X_train, y_train)

    if verbose:
        print(f"\nBest params: {search.best_params_}")
        print(f"Best CV {scoring}: {search.best_score_:.4f}")

    return search


def contamination_sweep(
    X_benign: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    contaminations: list[float] | None = None,
    n_estimators: int = 100,
    random_state: int = 42,
) -> list[dict]:
    """
    Evaluate Isolation Forest across a range of contamination values.

    Trains one model per contamination level and returns evaluation metrics.

    Parameters
    ----------
    X_benign       : benign-only training data
    X_test, y_test : test data and integer labels (0=ATTACK, 1=BENIGN)
    contaminations : list of contamination fractions to evaluate

    Returns
    -------
    List of dicts: [{contamination, precision, recall, f1, n_flags}, ...]
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    if contaminations is None:
        contaminations = IF_CONTAMINATION_GRID

    results = []
    for cont in contaminations:
        iso = IsolationForest(
            contamination=cont,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        iso.fit(X_benign)

        # IF: 1=normal(BENIGN), -1=anomaly(ATTACK) → map to 0=ATTACK, 1=BENIGN
        raw_preds  = iso.predict(X_test)
        preds      = np.where(raw_preds == 1, 1, 0)

        results.append({
            "contamination": cont,
            "precision":     round(float(precision_score(y_test, preds, average="macro", zero_division=0)), 4),
            "recall":        round(float(recall_score(y_test, preds, average="macro", zero_division=0)), 4),
            "f1":            round(float(f1_score(y_test, preds, average="macro", zero_division=0)), 4),
            "n_flags":       int((preds == 0).sum()),
        })

    return results
