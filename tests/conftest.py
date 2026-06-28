"""
tests/conftest.py
-----------------
Shared pytest fixtures for the full test suite.

Fixtures are session-scoped where possible to minimise setup overhead.
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC  = os.path.join(_ROOT, "src")
for p in [_ROOT, _SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

N_FEATURES = 78
STUB_FEATURE_NAMES = [f"feature_{i}" for i in range(N_FEATURES)]
STUB_FEATURE_NAMES[0] = "Destination Port"
STUB_FEATURE_NAMES[1] = "Flow Duration"
STUB_FEATURE_NAMES[2] = "Total Fwd Packets"


@pytest.fixture(scope="session")
def n_features():
    return N_FEATURES


@pytest.fixture(scope="session")
def feature_names():
    return list(STUB_FEATURE_NAMES)


@pytest.fixture(scope="session")
def zero_vector(feature_names):
    """All-zero feature vector."""
    return [0.0] * len(feature_names)


@pytest.fixture(scope="session")
def random_vector(feature_names):
    """Random feature vector for property-based tests."""
    rng = np.random.default_rng(42)
    return rng.uniform(0, 1, len(feature_names)).tolist()


@pytest.fixture(scope="session")
def stub_rf():
    rf = MagicMock()
    rf.n_estimators = 150
    rf.predict.return_value = np.array([0])
    rf.predict_proba.return_value = np.array([[0.9, 0.1]])
    rf.feature_importances_ = np.ones(N_FEATURES) / N_FEATURES
    rf.classes_ = np.array([0, 1])
    return rf


@pytest.fixture(scope="session")
def stub_iso():
    iso = MagicMock()
    iso.contamination = 0.01
    iso.predict.return_value = np.array([-1])
    iso.decision_function.return_value = np.array([-0.05])
    return iso


@pytest.fixture(scope="session")
def stub_scaler():
    scaler = MagicMock()
    scaler.transform.side_effect = lambda X: X
    return scaler


@pytest.fixture(scope="session")
def stub_le():
    le = MagicMock()
    le.classes_ = np.array(["ATTACK", "BENIGN"])
    le.transform.side_effect = lambda x: np.array(
        [0 if v == "ATTACK" else 1 for v in x]
    )
    le.inverse_transform.side_effect = lambda x: np.array(
        ["ATTACK" if v == 0 else "BENIGN" for v in x]
    )
    return le


@pytest.fixture(scope="session")
def sample_X_y():
    """Small balanced dataset for unit tests (100 rows, 78 features)."""
    rng   = np.random.default_rng(42)
    X     = rng.standard_normal((100, N_FEATURES)).astype(np.float32)
    y     = np.array([0] * 50 + [1] * 50)   # 0=ATTACK, 1=BENIGN
    return X, y
