"""
Integration test: full API request → inference → response cycle.

Uses stub models (no real .pkl files needed) to test the complete
request handling flow through Flask.
"""

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC  = os.path.join(_ROOT, "src")
for p in [_ROOT, _SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

N = 78
FEATURE_NAMES = [f"feature_{i}" for i in range(N)]
FEATURE_NAMES[0] = "Destination Port"


def _build_stubs():
    from unittest.mock import MagicMock
    scaler = MagicMock(); scaler.transform.side_effect = lambda X: X
    le = MagicMock()
    le.classes_ = np.array(["ATTACK", "BENIGN"])
    le.inverse_transform.side_effect = lambda x: np.array(
        ["ATTACK" if v == 0 else "BENIGN" for v in x]
    )
    rf = MagicMock()
    rf.n_estimators = 10
    rf.predict.return_value = np.array([0])
    rf.predict_proba.return_value = np.array([[0.9, 0.1]])
    rf.feature_importances_ = np.ones(N) / N
    iso = MagicMock()
    iso.contamination = 0.01
    iso.predict.return_value = np.array([-1])
    iso.decision_function.return_value = np.array([-0.05])
    return scaler, le, list(FEATURE_NAMES), rf, iso


@pytest.fixture(scope="module")
def api_client():
    stubs = _build_stubs()
    with patch("ids.models.predict.load_models", return_value=stubs):
        import src.api as api_module
        api_module.scaler        = stubs[0]
        api_module.le            = stubs[1]
        api_module.feature_names = stubs[2]
        api_module.rf_model      = stubs[3]
        api_module.iso_model     = stubs[4]
        api_module.app.config["TESTING"] = True
        with api_module.app.test_client() as c:
            yield c


class TestFullAPIFlow:
    def test_health_then_predict(self, api_client):
        h = api_client.get("/health").get_json()
        assert h["status"] == "healthy"

        fv = [0.0] * N
        r  = api_client.post("/predict", json={"features": fv}).get_json()
        assert r["status"] == "ok"

    def test_predict_then_feed(self, api_client):
        fv = [0.0] * N
        api_client.post("/predict", json={"features": fv})
        feed = api_client.get("/api/feed").get_json()
        assert feed["count"] >= 1

    def test_predict_then_stats(self, api_client):
        fv = [0.0] * N
        api_client.post("/predict", json={"features": fv})
        stats = api_client.get("/api/stats").get_json()
        assert stats["total"] >= 1

    def test_batch_then_feed(self, api_client):
        flows = [[0.0] * N] * 5
        r = api_client.post("/predict/batch", json={"flows": flows}).get_json()
        assert r["count"] == 5

    def test_dashboard_loads(self, api_client):
        r = api_client.get("/dashboard")
        assert r.status_code == 200
        assert b"AI-IDS" in r.data
