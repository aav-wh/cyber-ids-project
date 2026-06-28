"""
ids.utils.io
------------
File IO helpers: safe JSON/YAML loading, directory management,
artefact existence checks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def ensure_dir(path: str | Path) -> str:
    """Create directory (and parents) if it does not exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return str(path)


def safe_json_load(path: str, default=None):
    """
    Load a JSON file, returning default if the file does not exist.

    Parameters
    ----------
    path    : JSON file path
    default : value to return if file is missing or malformed
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def safe_json_save(data, path: str, indent: int = 2) -> None:
    """Save data as formatted JSON, creating parent directories as needed."""
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def check_artefacts(paths: dict[str, str]) -> list[str]:
    """
    Check which artefact files are missing.

    Parameters
    ----------
    paths : dict mapping label → file path

    Returns
    -------
    list of missing labels
    """
    return [label for label, path in paths.items() if not os.path.exists(path)]


def list_csv_files(directory: str) -> list[str]:
    """Return sorted list of CSV file names in a directory."""
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".csv") and not f.startswith(".")
    )


def file_size_mb(path: str) -> float:
    """Return file size in megabytes."""
    return round(os.path.getsize(path) / (1024 ** 2), 2)


def project_root() -> str:
    """Return the project root directory (parent of src/)."""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
