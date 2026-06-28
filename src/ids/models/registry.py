"""
ids.models.registry
-------------------
Lightweight model versioning registry.

Tracks trained model artefacts with metadata (training date, dataset hash,
key metrics) so experiments are reproducible and model provenance is auditable.

Models are stored as versioned .pkl files alongside a JSON manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np


_REGISTRY_FILE = "registry.json"


def _load_manifest(registry_dir: str) -> dict:
    path = os.path.join(registry_dir, _REGISTRY_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"models": []}


def _save_manifest(manifest: dict, registry_dir: str) -> None:
    path = os.path.join(registry_dir, _REGISTRY_FILE)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def register_model(
    model,
    name: str,
    metrics: dict,
    registry_dir: str,
    tag: str = "",
    overwrite: bool = False,
) -> str:
    """
    Save a trained model with metadata to the registry.

    Parameters
    ----------
    model        : fitted sklearn model to save
    name         : model name (e.g. 'random_forest', 'isolation_forest')
    metrics      : dict of evaluation metrics to log
    registry_dir : directory for versioned model files
    tag          : optional human label (e.g. 'experiment_smote')
    overwrite    : if True, overwrite an existing entry with the same name+tag

    Returns
    -------
    version string (e.g. '20250604_143021')
    """
    os.makedirs(registry_dir, exist_ok=True)
    manifest = _load_manifest(registry_dir)

    version  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{name}__{version}{('__' + tag) if tag else ''}.pkl"
    filepath = os.path.join(registry_dir, filename)

    joblib.dump(model, filepath)

    entry = {
        "name":         name,
        "version":      version,
        "tag":          tag,
        "filename":     filename,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "metrics":      {k: round(float(v), 6) if isinstance(v, float) else v
                         for k, v in metrics.items()},
    }

    # Remove previous entry with same name+tag if overwrite
    if overwrite:
        manifest["models"] = [
            m for m in manifest["models"]
            if not (m["name"] == name and m["tag"] == tag)
        ]

    manifest["models"].append(entry)
    _save_manifest(manifest, registry_dir)

    return version


def list_models(registry_dir: str) -> list[dict]:
    """Return all registered model entries."""
    return _load_manifest(registry_dir).get("models", [])


def load_best_model(
    name: str,
    registry_dir: str,
    metric: str = "f1",
    higher_is_better: bool = True,
) -> tuple:
    """
    Load the model version with the best value for a given metric.

    Parameters
    ----------
    name             : model name to filter by
    registry_dir     : registry directory
    metric           : metric key to compare
    higher_is_better : True for F1/AUC, False for loss

    Returns
    -------
    (model, entry_dict)
    """
    manifest = _load_manifest(registry_dir)
    candidates = [m for m in manifest["models"] if m["name"] == name
                  and metric in m.get("metrics", {})]

    if not candidates:
        raise FileNotFoundError(
            f"No registered model named '{name}' with metric '{metric}'."
        )

    best = max(candidates, key=lambda m: m["metrics"][metric]) \
           if higher_is_better else \
           min(candidates, key=lambda m: m["metrics"][metric])

    filepath = os.path.join(registry_dir, best["filename"])
    model    = joblib.load(filepath)
    return model, best


def delete_model(version: str, registry_dir: str) -> bool:
    """
    Remove a model version from the registry and delete its .pkl file.

    Returns True if deleted, False if not found.
    """
    manifest = _load_manifest(registry_dir)
    entry    = next((m for m in manifest["models"] if m["version"] == version), None)

    if entry is None:
        return False

    filepath = os.path.join(registry_dir, entry["filename"])
    if os.path.exists(filepath):
        os.remove(filepath)

    manifest["models"] = [m for m in manifest["models"] if m["version"] != version]
    _save_manifest(manifest, registry_dir)
    return True
