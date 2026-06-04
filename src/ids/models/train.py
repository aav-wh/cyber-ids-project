"""
ids.models.train
----------------
Training helpers for Random Forest and Isolation Forest.

Extracted from notebooks/03_modelling.ipynb.
Each function fits a model and optionally saves it to the models/ directory.
"""

import os
import time

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier

_DEFAULT_MODELS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "models",
)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 150,
    random_state: int = 42,
    n_jobs: int = -1,
    class_weight: str = "balanced",
    save_path: str | None = None,
    verbose: bool = True,
) -> RandomForestClassifier:
    """
    Fit a RandomForestClassifier on scaled training data.

    class_weight='balanced' is used instead of RandomOverSampler because it
    produces identical decision boundaries while avoiding the ~1 GB memory
    cost of duplicating rows (see notebook 02 for the equivalence proof).

    Parameters
    ----------
    X_train      : scaled training features, shape [n_samples, n_features]
    y_train      : encoded integer labels (0 = ATTACK, 1 = BENIGN)
    n_estimators : number of trees
    random_state : reproducibility seed
    n_jobs       : CPU cores for parallelism (-1 = all)
    class_weight : 'balanced' or dict {class_label: weight}
    save_path    : if given, save the fitted model to this .pkl path

    Returns
    -------
    Fitted RandomForestClassifier
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=class_weight,
    )

    t0 = time.time()
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0

    if verbose:
        print(f"Random Forest trained: {n_estimators} trees, {elapsed:.1f}s")

    if save_path is None:
        os.makedirs(_DEFAULT_MODELS, exist_ok=True)
        save_path = os.path.join(_DEFAULT_MODELS, "random_forest.pkl")

    joblib.dump(rf, save_path)
    if verbose:
        print(f"  Saved -> {save_path}")

    return rf


def train_isolation_forest(
    X_benign: np.ndarray,
    contamination: float = 0.01,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
    save_path: str | None = None,
    verbose: bool = True,
) -> IsolationForest:
    """
    Fit an IsolationForest on BENIGN-only training flows (semi-supervised).

    Because IF is trained exclusively on normal traffic it learns the
    distribution of benign behaviour and flags deviations as anomalies.
    This avoids needing labelled attack samples at training time.

    Parameters
    ----------
    X_benign      : scaled benign-only training flows
    contamination : expected proportion of anomalies in live traffic
    n_estimators  : number of isolation trees
    random_state  : reproducibility seed
    n_jobs        : CPU cores for parallelism
    save_path     : optional explicit output path

    Returns
    -------
    Fitted IsolationForest
    """
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    t0 = time.time()
    iso.fit(X_benign)
    elapsed = time.time() - t0

    if verbose:
        print(f"Isolation Forest trained: contamination={contamination}, {elapsed:.1f}s")

    if save_path is None:
        os.makedirs(_DEFAULT_MODELS, exist_ok=True)
        save_path = os.path.join(_DEFAULT_MODELS, "isolation_forest.pkl")

    joblib.dump(iso, save_path)
    if verbose:
        print(f"  Saved -> {save_path}")

    return iso
