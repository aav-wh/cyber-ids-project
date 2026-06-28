"""
ids.data.augmentation
---------------------
Class-balancing strategies for the training set.

Three strategies are provided:
  1. RandomOverSampler  — duplicate minority rows (fast, exact copy)
  2. SMOTE              — synthesise minority rows (better generalisation)
  3. NearMiss           — under-sample the majority class (saves memory)

Note on strategy choice
-----------------------
For CICIDS2017 the majority class (BENIGN) dominates ~80 % of rows.
Using class_weight='balanced' in RandomForest is mathematically equivalent
to RandomOverSampler and avoids the ~1 GB memory overhead of duplicating rows.
SMOTE is recommended when the dataset is small enough to fit in memory and
better generalisation on the minority class is needed.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import NearMiss


def random_oversample(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str | float = "auto",
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance classes by duplicating minority-class rows.

    Parameters
    ----------
    X, y         : training features and encoded integer labels
    strategy     : sampling_strategy — 'auto' balances to majority count
    random_state : seed for reproducibility

    Returns
    -------
    X_resampled, y_resampled
    """
    ros = RandomOverSampler(sampling_strategy=strategy, random_state=random_state)
    X_res, y_res = ros.fit_resample(X, y)

    if verbose:
        unique, counts = np.unique(y_res, return_counts=True)
        print(f"[RandomOverSampler] class distribution after resampling: "
              f"{dict(zip(unique.tolist(), counts.tolist()))}")

    return X_res, y_res


def smote_oversample(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    strategy: str | float = "auto",
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance classes using SMOTE (Synthetic Minority Oversampling Technique).

    SMOTE generates synthetic minority samples by interpolating between
    existing minority neighbours, producing more diverse training examples
    than simple duplication.

    Parameters
    ----------
    X, y         : training features and encoded integer labels
    k_neighbors  : number of nearest neighbours used by SMOTE
    strategy     : sampling_strategy
    random_state : seed

    Returns
    -------
    X_resampled, y_resampled
    """
    smote = SMOTE(
        k_neighbors=k_neighbors,
        sampling_strategy=strategy,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X, y)

    if verbose:
        unique, counts = np.unique(y_res, return_counts=True)
        print(f"[SMOTE] class distribution after resampling: "
              f"{dict(zip(unique.tolist(), counts.tolist()))}")

    return X_res, y_res


def nearmiss_undersample(
    X: np.ndarray,
    y: np.ndarray,
    version: int = 1,
    strategy: str | float = "auto",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance classes by under-sampling the majority class using NearMiss.

    NearMiss selects majority-class samples that are closest to minority
    samples, keeping the decision boundary informative.

    Parameters
    ----------
    X, y     : training features and encoded integer labels
    version  : NearMiss version (1, 2, or 3)
    strategy : sampling_strategy

    Returns
    -------
    X_resampled, y_resampled
    """
    nm = NearMiss(version=version, sampling_strategy=strategy)
    X_res, y_res = nm.fit_resample(X, y)

    if verbose:
        unique, counts = np.unique(y_res, return_counts=True)
        print(f"[NearMiss v{version}] class distribution after resampling: "
              f"{dict(zip(unique.tolist(), counts.tolist()))}")

    return X_res, y_res


def get_class_weights(y: np.ndarray) -> dict[int, float]:
    """
    Compute inverse-frequency class weights for use in model training.

    Equivalent to sklearn's class_weight='balanced' but returned as an
    explicit dict so it can be inspected or logged.

    Parameters
    ----------
    y : integer label array

    Returns
    -------
    dict mapping class index → weight
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weights = n_samples / (n_classes * counts)
    return dict(zip(classes.tolist(), weights.tolist()))
