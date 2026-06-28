"""
Unit tests for ids.models.predict.classify_flow
"""

import numpy as np
import pytest

from ids.models.predict import classify_flow


class TestClassifyFlow:
    """Tests using the stub fixtures from conftest.py."""

    def test_returns_dict(self, stub_scaler, stub_le, stub_rf, stub_iso, zero_vector):
        result = classify_flow(zero_vector, stub_scaler, stub_le, stub_rf, stub_iso)
        assert isinstance(result, dict)

    def test_has_required_keys(self, stub_scaler, stub_le, stub_rf, stub_iso, zero_vector):
        result = classify_flow(zero_vector, stub_scaler, stub_le, stub_rf, stub_iso)
        assert "random_forest" in result
        assert "isolation_forest" in result
        assert "ensemble" in result
        assert "final_decision" in result

    def test_final_decision_valid(self, stub_scaler, stub_le, stub_rf, stub_iso, zero_vector):
        result = classify_flow(zero_vector, stub_scaler, stub_le, stub_rf, stub_iso)
        assert result["final_decision"] in ("ATTACK", "BENIGN")

    def test_rf_confidence_in_range(self, stub_scaler, stub_le, stub_rf, stub_iso, zero_vector):
        result = classify_flow(zero_vector, stub_scaler, stub_le, stub_rf, stub_iso)
        rf = result["random_forest"]
        assert 0.0 <= rf["confidence"]  <= 1.0
        assert 0.0 <= rf["attack_prob"] <= 1.0
        assert 0.0 <= rf["benign_prob"] <= 1.0

    def test_probabilities_sum_to_one(self, stub_scaler, stub_le, stub_rf, stub_iso, zero_vector):
        result = classify_flow(zero_vector, stub_scaler, stub_le, stub_rf, stub_iso)
        rf = result["random_forest"]
        assert abs(rf["attack_prob"] + rf["benign_prob"] - 1.0) < 1e-5

    def test_accepts_numpy_array(self, stub_scaler, stub_le, stub_rf, stub_iso, n_features):
        fv = np.zeros(n_features, dtype=np.float32)
        result = classify_flow(fv, stub_scaler, stub_le, stub_rf, stub_iso)
        assert "final_decision" in result

    def test_if_score_is_float(self, stub_scaler, stub_le, stub_rf, stub_iso, zero_vector):
        result = classify_flow(zero_vector, stub_scaler, stub_le, stub_rf, stub_iso)
        assert isinstance(result["isolation_forest"]["anomaly_score"], float)
