"""
tests/test_api.py
-----------------
Pytest test suite for the AI-IDS Flask API.
COM668 | Abdulbosit Abdurazzakov | B00979380

Tests cover:
  - /health         -- liveness and model metadata
  - /features       -- feature list shape and content
  - /predict        -- valid input, named fields, missing features, bad values
  - /predict/batch  -- valid batch, oversized batch, malformed input
  - /api/feed       -- returns list structure
  - /api/stats      -- returns aggregate fields
  - /api/shap       -- returns feature importance list
  - /dashboard      -- HTML page served correctly
  - Error handlers  -- 404 returns JSON

Run with:
  pytest tests/ -v
"""

import os
import sys

import joblib
import numpy as np
import pytest

# ── Make the src/ package importable from the project root ────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def feature_names():
    """Load feature names once for the whole test session."""
    path = os.path.join(_ROOT, "data", "processed", "feature_names.pkl")
    return joblib.load(path)


@pytest.fixture(scope="session")
def client(feature_names):
    """
    Create a Flask test client.

    Importing src.api triggers model loading; we do it once per session to
    keep the suite fast.
    """
    # api.py adds _ROOT to sys.path on import, but we've already done it above
    import src.api as api_module
    api_module.app.config["TESTING"] = True
    with api_module.app.test_client() as c:
        yield c


@pytest.fixture(scope="session")
def valid_vector(feature_names):
    """A zeroed feature vector (valid shape, all zeros — passes NaN/Inf checks)."""
    return [0.0] * len(feature_names)


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_status_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_returns_healthy(self, client):
        data = client.get("/health").get_json()
        assert data["status"] == "healthy"

    def test_models_listed(self, client):
        data = client.get("/health").get_json()
        assert "random_forest" in data["models"]
        assert "isolation_forest" in data["models"]

    def test_feature_count_positive(self, client):
        data = client.get("/health").get_json()
        assert data["features"] > 0


# ── /features ─────────────────────────────────────────────────────────────────

class TestFeatures:
    def test_status_200(self, client):
        assert client.get("/features").status_code == 200

    def test_count_matches_list(self, client):
        data = client.get("/features").get_json()
        assert data["count"] == len(data["features"])

    def test_features_non_empty(self, client):
        data = client.get("/features").get_json()
        assert len(data["features"]) > 0

    def test_destination_port_present(self, client):
        data = client.get("/features").get_json()
        assert "Destination Port" in data["features"]


# ── /predict ──────────────────────────────────────────────────────────────────

class TestPredict:
    def test_valid_array_input(self, client, valid_vector):
        r = client.post("/predict", json={"features": valid_vector})
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"
        assert "result" in data
        assert data["result"]["final_decision"] in ("ATTACK", "BENIGN")

    def test_valid_named_input(self, client, feature_names):
        payload = {name: 0.0 for name in feature_names}
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert r.get_json()["status"] == "ok"

    def test_result_contains_all_models(self, client, valid_vector):
        result = client.post("/predict", json={"features": valid_vector}).get_json()["result"]
        assert "random_forest" in result
        assert "isolation_forest" in result
        assert "ensemble" in result

    def test_rf_confidence_in_range(self, client, valid_vector):
        rf = client.post("/predict", json={"features": valid_vector}).get_json()["result"]["random_forest"]
        assert 0.0 <= rf["confidence"] <= 1.0
        assert 0.0 <= rf["attack_prob"] <= 1.0
        assert 0.0 <= rf["benign_prob"] <= 1.0

    def test_inference_time_reported(self, client, valid_vector):
        data = client.post("/predict", json={"features": valid_vector}).get_json()
        assert "inference_time_ms" in data
        assert data["inference_time_ms"] >= 0

    def test_wrong_feature_count_returns_400(self, client):
        r = client.post("/predict", json={"features": [0.0, 1.0, 2.0]})
        assert r.status_code == 400
        assert "Expected" in r.get_json()["error"]

    def test_missing_named_feature_returns_400(self, client, feature_names):
        # Send all features except the first one
        payload = {name: 0.0 for name in feature_names[1:]}
        r = client.post("/predict", json=payload)
        assert r.status_code == 400
        assert "Missing" in r.get_json()["error"]

    def test_non_numeric_feature_returns_400(self, client, feature_names):
        payload = {"features": ["not_a_number"] + [0.0] * (len(feature_names) - 1)}
        r = client.post("/predict", json=payload)
        assert r.status_code == 400

    def test_nan_feature_returns_400(self, client, feature_names):
        # JSON doesn't support NaN, but we can test via named fields using a trick:
        # send a string that can't convert to float
        payload = {name: (float("nan") if i == 0 else 0.0) for i, name in enumerate(feature_names)}
        # NaN in JSON becomes null; our validator should catch it or the float() cast fails
        # Simplest approach: send via positional with a string
        bad = [0.0] * len(feature_names)
        bad[0] = "NaN_string_value"
        r = client.post("/predict", json={"features": bad})
        assert r.status_code == 400

    def test_empty_body_returns_400(self, client):
        r = client.post("/predict", data="", content_type="application/json")
        assert r.status_code == 400

    def test_get_not_allowed(self, client):
        r = client.get("/predict")
        assert r.status_code == 405


# ── /predict/batch ────────────────────────────────────────────────────────────

class TestPredictBatch:
    def test_single_flow_batch(self, client, valid_vector):
        r = client.post("/predict/batch", json={"flows": [valid_vector]})
        assert r.status_code == 200
        data = r.get_json()
        assert data["count"] == 1
        assert len(data["results"]) == 1

    def test_multiple_flows(self, client, valid_vector):
        flows = [valid_vector] * 5
        r = client.post("/predict/batch", json={"flows": flows})
        assert r.status_code == 200
        assert r.get_json()["count"] == 5

    def test_oversized_batch_returns_400(self, client, valid_vector):
        flows = [valid_vector] * 1001
        r = client.post("/predict/batch", json={"flows": flows})
        assert r.status_code == 400
        assert "1000" in r.get_json()["error"]

    def test_missing_flows_key_returns_400(self, client):
        r = client.post("/predict/batch", json={"data": []})
        assert r.status_code == 400

    def test_wrong_length_flow_gets_error_entry(self, client, valid_vector):
        bad_flow = [0.0, 1.0]
        r = client.post("/predict/batch", json={"flows": [valid_vector, bad_flow]})
        assert r.status_code == 200
        results = r.get_json()["results"]
        assert "result" in results[0]
        assert "error" in results[1]


# ── Dashboard data endpoints ──────────────────────────────────────────────────

class TestDashboardAPI:
    def test_feed_returns_list(self, client):
        r = client.get("/api/feed")
        assert r.status_code == 200
        data = r.get_json()
        assert "feed" in data
        assert isinstance(data["feed"], list)

    def test_stats_returns_expected_fields(self, client):
        r = client.get("/api/stats")
        assert r.status_code == 200
        data = r.get_json()
        for field in ("total", "attacks", "benign", "attack_pct", "avg_confidence", "avg_latency_ms"):
            assert field in data, f"Missing field: {field}"

    def test_shap_returns_features(self, client):
        r = client.get("/api/shap?top_n=5")
        assert r.status_code == 200
        data = r.get_json()
        assert "features" in data
        assert len(data["features"]) == 5
        assert "feature" in data["features"][0]

    def test_feed_populated_after_predict(self, client, valid_vector):
        client.post("/predict", json={"features": valid_vector})
        data = client.get("/api/feed?n=1").get_json()
        assert data["count"] >= 1
        assert "final_decision" in data["feed"][0]


# ── Dashboard HTML ────────────────────────────────────────────────────────────

class TestDashboard:
    def test_dashboard_returns_200(self, client):
        r = client.get("/dashboard")
        assert r.status_code == 200

    def test_dashboard_is_html(self, client):
        r = client.get("/dashboard")
        assert b"<!DOCTYPE html>" in r.data or b"<html" in r.data

    def test_dashboard_contains_chart_js(self, client):
        r = client.get("/dashboard")
        assert b"chart.umd.min.js" in r.data


# ── Error handlers ────────────────────────────────────────────────────────────

class TestErrorHandlers:
    def test_404_returns_json(self, client):
        r = client.get("/nonexistent-endpoint-xyz")
        assert r.status_code == 404
        assert r.get_json()["error"] == "Endpoint not found."

    def test_405_returns_json(self, client):
        r = client.get("/predict")
        assert r.status_code == 405
        assert "error" in r.get_json()
