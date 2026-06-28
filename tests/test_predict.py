"""
tests/test_predict.py
Unit tests for ids.models.predict module.
COM668 | Abdulbosit Abdurazzakov | B00979380

Tests: output structure, value ranges, OR-rule logic, anomaly score
direction, input format flexibility, load_models error handling,
and integration tests with real saved artefacts.

Run:
  pytest tests/test_predict.py -v
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC  = os.path.join(_ROOT, "src")
for _p in [_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ids.models.predict import classify_flow, load_models

_ARTEFACTS_PRESENT = (
    os.path.exists(os.path.join(_ROOT, "data", "processed", "feature_names.pkl"))
    and os.path.exists(os.path.join(_ROOT, "models", "random_forest.pkl"))
    and os.path.exists(os.path.join(_ROOT, "models", "isolation_forest.pkl"))
)

N_FEATURES = 78


# ── Stub factories ─────────────────────────────────────────────────────────────

def _stub_scaler():
    s = MagicMock()
    s.transform.side_effect = lambda X: np.array(X, dtype=np.float32)
    return s


def _stub_le():
    le = MagicMock()
    le.classes_ = np.array(["ATTACK", "BENIGN"])
    le.inverse_transform.side_effect = lambda x: np.array(
        ["ATTACK" if v == 0 else "BENIGN" for v in x]
    )
    return le


def _stub_rf(pred=0, proba=None):
    rf = MagicMock()
    rf.predict.return_value       = np.array([pred])
    rf.predict_proba.return_value = np.array(
        [proba or ([0.85, 0.15] if pred == 0 else [0.10, 0.90])]
    )
    rf.feature_importances_ = np.ones(N_FEATURES) / N_FEATURES
    return rf


def _stub_iso(pred=1, score=-0.05):
    iso = MagicMock()
    iso.predict.return_value           = np.array([pred])
    iso.decision_function.return_value = np.array([score])
    return iso


def _zero_vector():
    return [0.0] * N_FEATURES


# ── Output structure ───────────────────────────────────────────────────────────

class TestClassifyFlowStructure:
    def test_returns_dict(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(), _stub_rf(), _stub_iso())
        assert isinstance(r, dict)

    def test_top_level_keys_present(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(), _stub_rf(), _stub_iso())
        for key in ("random_forest", "isolation_forest", "ensemble", "final_decision"):
            assert key in r

    def test_rf_subkeys(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(), _stub_rf(), _stub_iso())
        for key in ("prediction", "confidence", "attack_prob", "benign_prob"):
            assert key in r["random_forest"]

    def test_if_subkeys(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(), _stub_rf(), _stub_iso())
        for key in ("prediction", "anomaly_score"):
            assert key in r["isolation_forest"]

    def test_ensemble_subkeys(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(), _stub_rf(), _stub_iso())
        for key in ("prediction", "rule"):
            assert key in r["ensemble"]

    def test_final_decision_is_valid_label(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(), _stub_rf(), _stub_iso())
        assert r["final_decision"] in ("ATTACK", "BENIGN")


# ── Value ranges ───────────────────────────────────────────────────────────────

class TestClassifyFlowValues:
    def test_confidence_in_range(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(), _stub_rf(), _stub_iso())
        assert 0.0 <= r["random_forest"]["confidence"] <= 1.0

    def test_attack_prob_in_range(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(), _stub_rf(), _stub_iso())
        assert 0.0 <= r["random_forest"]["attack_prob"] <= 1.0

    def test_benign_prob_in_range(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(), _stub_rf(), _stub_iso())
        assert 0.0 <= r["random_forest"]["benign_prob"] <= 1.0

    def test_probs_sum_to_one(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(), _stub_rf(), _stub_iso())
        rf = r["random_forest"]
        assert abs(rf["attack_prob"] + rf["benign_prob"] - 1.0) < 1e-4

    def test_anomaly_score_is_float(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(), _stub_rf(), _stub_iso())
        assert isinstance(r["isolation_forest"]["anomaly_score"], float)


# ── Ensemble OR-rule logic ─────────────────────────────────────────────────────

class TestEnsembleLogic:
    def test_both_benign_gives_benign(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(),
                          _stub_rf(pred=1), _stub_iso(pred=1))
        assert r["ensemble"]["prediction"] == "BENIGN"
        assert r["final_decision"]         == "BENIGN"

    def test_rf_attack_gives_attack(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(),
                          _stub_rf(pred=0), _stub_iso(pred=1))
        assert r["ensemble"]["prediction"] == "ATTACK"
        assert r["final_decision"]         == "ATTACK"

    def test_if_attack_gives_attack(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(),
                          _stub_rf(pred=1), _stub_iso(pred=-1))
        assert r["ensemble"]["prediction"] == "ATTACK"
        assert r["final_decision"]         == "ATTACK"

    def test_both_attack_gives_attack(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(),
                          _stub_rf(pred=0), _stub_iso(pred=-1))
        assert r["ensemble"]["prediction"] == "ATTACK"

    def test_final_decision_always_matches_ensemble(self):
        for rf_pred, if_pred in [(0, 1), (1, -1), (0, -1), (1, 1)]:
            r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(),
                              _stub_rf(pred=rf_pred), _stub_iso(pred=if_pred))
            assert r["final_decision"] == r["ensemble"]["prediction"]


# ── Anomaly score direction ────────────────────────────────────────────────────

class TestAnomalyScore:
    def test_attack_has_negative_score(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(),
                          _stub_rf(), _stub_iso(pred=-1, score=-0.15))
        assert r["isolation_forest"]["anomaly_score"] < 0

    def test_benign_has_positive_score(self):
        r = classify_flow(_zero_vector(), _stub_scaler(), _stub_le(),
                          _stub_rf(), _stub_iso(pred=1, score=0.08))
        assert r["isolation_forest"]["anomaly_score"] > 0


# ── Input format flexibility ───────────────────────────────────────────────────

class TestInputFormats:
    def test_accepts_list(self):
        r = classify_flow(list(_zero_vector()), _stub_scaler(), _stub_le(),
                          _stub_rf(), _stub_iso())
        assert "final_decision" in r

    def test_accepts_numpy_array(self):
        r = classify_flow(np.zeros(N_FEATURES), _stub_scaler(), _stub_le(),
                          _stub_rf(), _stub_iso())
        assert "final_decision" in r

    def test_accepts_float64_array(self):
        r = classify_flow(np.zeros(N_FEATURES, dtype=np.float64),
                          _stub_scaler(), _stub_le(), _stub_rf(), _stub_iso())
        assert "final_decision" in r

    def test_scaler_called_exactly_once(self):
        scaler = _stub_scaler()
        classify_flow(_zero_vector(), scaler, _stub_le(), _stub_rf(), _stub_iso())
        scaler.transform.assert_called_once()


# ── load_models error handling ─────────────────────────────────────────────────

class TestLoadModels:
    def test_missing_artefacts_raise_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError, match="Missing artefact"):
                load_models(processed_dir=tmp, models_dir=tmp)

    def test_error_message_names_missing_artefact(self):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                load_models(processed_dir=tmp, models_dir=tmp)
            except FileNotFoundError as e:
                assert "artefact" in str(e).lower()


# ── Integration (real artefacts required) ─────────────────────────────────────

@pytest.mark.skipif(not _ARTEFACTS_PRESENT, reason="Real model artefacts not present")
class TestIntegrationWithRealModels:
    @pytest.fixture(scope="class")
    def real_models(self):
        return load_models()

    def test_load_returns_five_objects(self, real_models):
        assert len(real_models) == 5

    def test_feature_count_is_78(self, real_models):
        _, _, feature_names, _, _ = real_models
        assert len(feature_names) == 78

    def test_classify_zero_vector(self, real_models):
        scaler, le, feature_names, rf, iso = real_models
        r = classify_flow([0.0] * len(feature_names), scaler, le, rf, iso)
        assert r["final_decision"] in ("ATTACK", "BENIGN")

    def test_high_packet_rate_flow_structure_valid(self, real_models):
        # KNOWN LIMITATION: RF recall ~14.9%, so even extreme flows may be
        # classified BENIGN at default threshold. This verifies pipeline
        # structure only. See notebook 09 for threshold-tuned results (97% recall).
        scaler, le, feature_names, rf, iso = real_models
        fv = [0.0] * len(feature_names)
        idx = {name: i for i, name in enumerate(feature_names)}
        if "Flow Packets/s" in idx:
            fv[idx["Flow Packets/s"]] = 1_000_000.0
        if "Total Backward Packets" in idx:
            fv[idx["Total Backward Packets"]] = 0.0
        if "Flow Duration" in idx:
            fv[idx["Flow Duration"]] = 100.0
        r = classify_flow(fv, scaler, le, rf, iso)
        assert r["final_decision"] in ("ATTACK", "BENIGN")
        assert 0.0 <= r["random_forest"]["attack_prob"] <= 1.0

    def test_predict_is_deterministic(self, real_models):
        scaler, le, feature_names, rf, iso = real_models
        fv = [0.0] * len(feature_names)
        r1 = classify_flow(fv, scaler, le, rf, iso)
        r2 = classify_flow(fv, scaler, le, rf, iso)
        assert r1["final_decision"]                    == r2["final_decision"]
        assert r1["random_forest"]["attack_prob"]      == r2["random_forest"]["attack_prob"]
        assert r1["isolation_forest"]["anomaly_score"] == r2["isolation_forest"]["anomaly_score"]
