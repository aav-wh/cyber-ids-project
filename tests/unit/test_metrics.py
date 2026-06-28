"""
Unit tests for ids.evaluation.metrics — compare_models and metrics_to_dataframe
"""

import numpy as np
import pytest

from ids.evaluation.metrics import metrics_to_dataframe, compare_models, compute_full_metrics


def sample_metrics(f1=0.90, name="Model"):
    m = compute_full_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
    )
    m["model"] = name
    return m


class TestMetricsToDataframe:
    def test_returns_one_row(self):
        df = metrics_to_dataframe(sample_metrics(), "TestModel")
        assert len(df) == 1

    def test_has_model_column(self):
        df = metrics_to_dataframe(sample_metrics(), "RF")
        assert "model" in df.columns

    def test_has_f1_column(self):
        df = metrics_to_dataframe(sample_metrics())
        assert "f1" in df.columns


class TestCompareModels:
    def test_returns_dataframe_with_rows(self):
        m1 = sample_metrics()
        m2 = sample_metrics()
        df = compare_models(("RF", m1), ("IF", m2))
        assert len(df) == 2

    def test_sorted_by_f1(self):
        # Create two models with different F1
        yt  = np.array([0, 0, 1, 1])
        # Perfect predictions
        yp1 = np.array([0, 0, 1, 1])
        # Wrong predictions
        yp2 = np.array([1, 1, 0, 0])
        m1  = compute_full_metrics(yt, yp1)
        m2  = compute_full_metrics(yt, yp2)
        df  = compare_models(("Good", m1), ("Bad", m2))
        assert df.iloc[0]["model"] == "Good"
