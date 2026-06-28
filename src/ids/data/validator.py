"""
ids.data.validator
------------------
Input validation for raw dataframes and feature vectors.

Used by the API and preprocessing pipeline to catch bad data early
with clear, actionable error messages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ids.exceptions import FeatureValidationError


def validate_feature_vector(
    fv: list | np.ndarray,
    expected_length: int,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> np.ndarray:
    """
    Validate and coerce a single feature vector.

    Parameters
    ----------
    fv              : raw feature values (list or array)
    expected_length : number of features the model expects
    allow_nan       : if False, raises on NaN values
    allow_inf       : if False, raises on ±Inf values

    Returns
    -------
    np.ndarray of float32 with shape (expected_length,)

    Raises
    ------
    FeatureValidationError on any validation failure
    """
    try:
        arr = np.array(fv, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise FeatureValidationError(
            f"Feature vector contains non-numeric values: {exc}"
        ) from exc

    if len(arr) != expected_length:
        raise FeatureValidationError(
            f"Expected {expected_length} features, got {len(arr)}."
        )

    if not allow_nan and np.isnan(arr).any():
        bad_idx = np.where(np.isnan(arr))[0].tolist()
        raise FeatureValidationError(
            f"Feature vector contains NaN at index/indices: {bad_idx[:5]}"
        )

    if not allow_inf and np.isinf(arr).any():
        bad_idx = np.where(np.isinf(arr))[0].tolist()
        raise FeatureValidationError(
            f"Feature vector contains Inf at index/indices: {bad_idx[:5]}"
        )

    return arr.astype(np.float32)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    label_col: str = "Label",
) -> None:
    """
    Validate a raw CICIDS2017 dataframe.

    Parameters
    ----------
    df               : input dataframe
    required_columns : if given, checks all columns are present
    label_col        : name of the label column

    Raises
    ------
    FeatureValidationError if validation fails
    """
    if df.empty:
        raise FeatureValidationError("DataFrame is empty.")

    if required_columns:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise FeatureValidationError(
                f"DataFrame missing {len(missing)} required column(s): {missing[:10]}"
            )

    if label_col not in df.columns:
        raise FeatureValidationError(
            f"Label column '{label_col}' not found. "
            f"Available columns: {list(df.columns[:10])}"
        )


def validate_batch(
    flows: list,
    expected_length: int,
    max_size: int = 1000,
) -> None:
    """
    Validate a batch of feature vectors before inference.

    Parameters
    ----------
    flows           : list of feature arrays
    expected_length : expected number of features per flow
    max_size        : maximum allowed batch size

    Raises
    ------
    FeatureValidationError on any failure
    """
    if not isinstance(flows, list) or len(flows) == 0:
        raise FeatureValidationError("'flows' must be a non-empty list.")

    if len(flows) > max_size:
        raise FeatureValidationError(
            f"Batch size {len(flows)} exceeds maximum allowed ({max_size})."
        )


def check_label_distribution(
    y: np.ndarray | pd.Series,
    min_class_ratio: float = 0.001,
) -> dict[str, float]:
    """
    Warn if any class represents less than min_class_ratio of all labels.

    Returns
    -------
    dict mapping class label → fraction of total
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    dist = {str(k): round(float(v) / total, 6) for k, v in zip(unique, counts)}

    for label, frac in dist.items():
        if frac < min_class_ratio:
            import warnings
            warnings.warn(
                f"Class '{label}' represents only {frac:.4%} of labels — "
                "consider oversampling or adjusting class weights.",
                UserWarning,
                stacklevel=2,
            )

    return dist
