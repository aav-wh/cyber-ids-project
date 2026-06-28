"""
Unit tests for ids.models.evaluate and ids.evaluation.metrics
"""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from ids.models.evaluate import compute_metrics
from ids.evaluation.metrics import compute_full_metrics


def perfect_preds():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    return y_true, y_pred


def all_wrong_preds():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    return y_true, y_pred


class TestComputeMetrics:
    def test_perfect_f1_is_one(self):
        yt, yp = perfect_preds()
        m = compute_metrics(yt, yp)
        assert m["f1"] == pytest.approx(1.0)

    def test_accuracy_field_present(self):
        yt, yp = perfect_preds()
        m = compute_metrics(yt, yp)
        assert "accuracy" in m

    def test_report_is_string(self):
        yt, yp = perfect_preds()
        m = compute_metrics(yt, yp)
        assert isinstance(m["report"], str)

    def test_auc_computed_when_proba_given(self):
        yt   = np.array([0, 0, 1, 1])
        yp   = np.array([0, 0, 1, 1])
        prob = np.array([0.9, 0.8, 0.2, 0.1])
        m    = compute_metrics(yt, yp, y_proba=prob)
        assert "auc" in m
        assert 0.0 <= m["auc"] <= 1.0


class TestComputeFullMetrics:
    def test_has_fpr_fnr(self):
        yt, yp = perfect_preds()
        m = compute_full_metrics(yt, yp)
        assert "fpr" in m
        assert "fnr" in m

    def test_detection_rate_equals_recall(self):
        yt = np.array([0, 0, 1, 1])
        yp = np.array([0, 1, 1, 1])
        m  = compute_full_metrics(yt, yp)
        # detection_rate = TP / (TP + FN) = recall for attack class
        assert m["detection_rate"] is not None
        assert 0.0 <= m["detection_rate"] <= 1.0

    def test_mcc_in_range(self):
        yt, yp = all_wrong_preds()
        m = compute_full_metrics(yt, yp)
        assert -1.0 <= m["mcc"] <= 1.0

    def test_perfect_predictions(self):
        yt, yp = perfect_preds()
        m = compute_full_metrics(yt, yp)
        assert m["fpr"] == pytest.approx(0.0)
        assert m["fnr"] == pytest.approx(0.0)
