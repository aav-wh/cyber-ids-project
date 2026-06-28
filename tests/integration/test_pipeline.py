"""
Integration test: preprocessing → training → evaluation pipeline.

Runs on synthetic data (no CICIDS2017 CSVs required) to validate the
full pipeline wiring without large file dependencies.
"""

import os
import tempfile

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ids.features.preprocessing import fit_and_save_artefacts, scale, load_artefacts
from ids.models.train import train_random_forest, train_isolation_forest
from ids.models.predict import classify_flow
from ids.evaluation.metrics import compute_full_metrics


@pytest.fixture(scope="module")
def synthetic_pipeline():
    """End-to-end pipeline on 200-sample synthetic data."""
    rng = np.random.default_rng(42)
    n, f = 200, 20

    X = rng.standard_normal((n, f)).astype(np.float32)
    y = np.array(["ATTACK"] * 100 + ["BENIGN"] * 100)
    feature_names = [f"feat_{i}" for i in range(f)]

    import pandas as pd
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)

    with tempfile.TemporaryDirectory() as tmpdir:
        processed_dir = os.path.join(tmpdir, "processed")
        models_dir    = os.path.join(tmpdir, "models")
        os.makedirs(processed_dir)
        os.makedirs(models_dir)

        scaler, le = fit_and_save_artefacts(
            X_df, y_series, feature_names, processed_dir, verbose=False
        )

        X_scaled = scale(X_df, scaler)
        y_enc    = le.transform(y)

        rf = train_random_forest(
            X_scaled, y_enc,
            n_estimators=10,
            save_path=os.path.join(models_dir, "random_forest.pkl"),
            verbose=False,
        )
        iso = train_isolation_forest(
            X_scaled[y_enc == le.transform(["BENIGN"])[0]],
            contamination=0.1,
            n_estimators=10,
            save_path=os.path.join(models_dir, "isolation_forest.pkl"),
            verbose=False,
        )

        yield {
            "scaler": scaler, "le": le, "rf": rf, "iso": iso,
            "X": X_scaled, "y": y_enc,
            "feature_names": feature_names,
            "processed_dir": processed_dir,
            "models_dir": models_dir,
        }


class TestPipelineIntegration:
    def test_scaler_fitted(self, synthetic_pipeline):
        scaler = synthetic_pipeline["scaler"]
        assert hasattr(scaler, "mean_")

    def test_rf_trained(self, synthetic_pipeline):
        rf = synthetic_pipeline["rf"]
        assert rf.n_estimators == 10

    def test_predictions_valid_labels(self, synthetic_pipeline):
        p = synthetic_pipeline
        preds = p["rf"].predict(p["X"])
        assert set(preds).issubset({0, 1})

    def test_classify_flow_works(self, synthetic_pipeline):
        p   = synthetic_pipeline
        fv  = p["X"][0]
        res = classify_flow(fv, p["scaler"], p["le"], p["rf"], p["iso"])
        assert res["final_decision"] in ("ATTACK", "BENIGN")

    def test_metrics_computed(self, synthetic_pipeline):
        p     = synthetic_pipeline
        preds = p["rf"].predict(p["X"])
        m     = compute_full_metrics(p["y"], preds)
        assert 0 <= m["f1"] <= 1

    def test_artefacts_loadable(self, synthetic_pipeline):
        p = synthetic_pipeline
        scaler2, le2, names2 = load_artefacts(p["processed_dir"])
        assert names2 == p["feature_names"]
