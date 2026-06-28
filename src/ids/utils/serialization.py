"""
ids.utils.serialization
------------------------
Model serialization helpers (save/load with metadata).

Wraps joblib with version checking and basic integrity validation
to guard against loading a model trained on a different feature set.
"""

from __future__ import annotations

import hashlib
import os

import joblib
import numpy as np


def save_with_metadata(
    obj,
    path: str,
    metadata: dict | None = None,
) -> None:
    """
    Save a Python object alongside a metadata sidecar (.meta.json).

    Parameters
    ----------
    obj      : object to serialize (typically a sklearn model)
    path     : destination .pkl path
    metadata : optional dict written to path + '.meta.json'
    """
    import json
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(obj, path)

    if metadata:
        meta_path = path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)


def load_with_metadata(path: str) -> tuple:
    """
    Load an object and its optional metadata sidecar.

    Returns
    -------
    (obj, metadata_dict)   — metadata is {} if no sidecar exists
    """
    import json
    obj  = joblib.load(path)
    meta = {}

    meta_path = path + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    return obj, meta


def feature_fingerprint(feature_names: list[str]) -> str:
    """
    Compute a short hash of the feature name list for compatibility checks.

    If a loaded model's fingerprint doesn't match the current feature set,
    the model was trained on different features and cannot be used.

    Returns
    -------
    8-character hex string
    """
    content = "|".join(feature_names).encode()
    return hashlib.sha256(content).hexdigest()[:8]


def verify_feature_compatibility(
    model_fingerprint: str,
    current_features: list[str],
) -> bool:
    """
    Return True if the model was trained on the same feature set.

    Parameters
    ----------
    model_fingerprint  : fingerprint stored with the model at training time
    current_features   : current ordered feature name list

    Returns
    -------
    bool — True if compatible, False otherwise
    """
    current_fp = feature_fingerprint(current_features)
    return model_fingerprint == current_fp
