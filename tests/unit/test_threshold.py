"""
Unit tests for ids.models.threshold
"""

import numpy as np
import pytest

from ids.models.threshold import optimal_f1_threshold, pr_curve_thresholds
from ids.detection.thresholds import StaticThreshold, AdaptiveThreshold


class TestOptimalF1Threshold:
    def test_threshold_in_range(self):
        yt = np.array([0, 0, 1, 1, 0, 1])
        yp = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.15])
        t, f1 = optimal_f1_threshold(yt, yp)
        assert 0.0 < t < 1.0

    def test_f1_between_zero_and_one(self):
        yt = np.array([0, 1, 0, 1])
        yp = np.array([0.9, 0.1, 0.8, 0.2])
        _, f1 = optimal_f1_threshold(yt, yp)
        assert 0.0 <= f1 <= 1.0


class TestPRCurveThresholds:
    def test_returns_dataframe(self):
        import pandas as pd
        yt = np.array([0, 0, 1, 1])
        yp = np.array([0.9, 0.2, 0.8, 0.1])
        df = pr_curve_thresholds(yt, yp)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        yt = np.array([0, 1, 0, 1])
        yp = np.array([0.8, 0.6, 0.3, 0.9])
        df = pr_curve_thresholds(yt, yp)
        for col in ["threshold", "precision", "recall", "f1"]:
            assert col in df.columns


class TestStaticThreshold:
    def test_classify_attack(self):
        t = StaticThreshold(0.5)
        assert t.classify(0.9) == "ATTACK"

    def test_classify_benign(self):
        t = StaticThreshold(0.5)
        assert t.classify(0.1) == "BENIGN"

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            StaticThreshold(1.5)


class TestAdaptiveThreshold:
    def test_initial_threshold(self):
        t = AdaptiveThreshold(initial_threshold=0.5)
        assert t.value == pytest.approx(0.5)

    def test_classify_returns_string(self):
        t = AdaptiveThreshold()
        result = t.classify(0.8)
        assert result in ("ATTACK", "BENIGN")

    def test_stats_dict(self):
        t = AdaptiveThreshold()
        s = t.stats()
        assert "current_threshold" in s
