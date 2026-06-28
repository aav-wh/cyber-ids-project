"""
ids.utils.validation
---------------------
Generic validation helpers used across the codebase.
"""

from __future__ import annotations

import numpy as np


def is_valid_probability(p: float) -> bool:
    """Return True if p is a valid probability (0 ≤ p ≤ 1)."""
    return isinstance(p, (int, float)) and 0.0 <= float(p) <= 1.0


def assert_shapes_match(X: np.ndarray, y: np.ndarray) -> None:
    """Raise ValueError if X and y have incompatible first dimensions."""
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {y.shape[0]} elements."
        )


def assert_feature_count(X: np.ndarray, expected: int) -> None:
    """Raise ValueError if X does not have the expected number of columns."""
    actual = X.shape[1] if X.ndim == 2 else len(X)
    if actual != expected:
        raise ValueError(
            f"Expected {expected} features, got {actual}."
        )


def validate_labels(y: np.ndarray, valid_labels: set) -> list:
    """
    Return a list of any labels in y that are not in valid_labels.

    Returns an empty list if all labels are valid.
    """
    unique = set(np.unique(y).tolist())
    return list(unique - valid_labels)


def check_no_nan_inf(arr: np.ndarray, name: str = "array") -> None:
    """Raise ValueError if arr contains NaN or Inf values."""
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values.")
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains Inf values.")


def validate_config_keys(config: dict, required: list[str]) -> list[str]:
    """Return list of any required keys missing from config dict."""
    return [k for k in required if k not in config]
