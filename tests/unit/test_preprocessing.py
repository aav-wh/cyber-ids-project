"""
Unit tests for ids.features.preprocessing
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler, LabelEncoder

from ids.features.preprocessing import extract_features, scale, oversample


def make_labelled_df(n=100):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((n, 5)),
                      columns=[f"f{i}" for i in range(5)])
    df["Label"]       = ["BENIGN"] * (n // 2) + ["ATTACK"] * (n // 2)
    df["BinaryLabel"] = df["Label"].apply(lambda x: x if x == "BENIGN" else "ATTACK")
    return df


class TestExtractFeatures:
    def test_returns_correct_shapes(self):
        df = make_labelled_df(100)
        X, y, names = extract_features(df)
        assert X.shape == (100, 5)
        assert len(y) == 100
        assert len(names) == 5

    def test_drops_label_columns(self):
        df = make_labelled_df(10)
        X, _, names = extract_features(df)
        assert "Label" not in names
        assert "BinaryLabel" not in names

    def test_feature_names_match_columns(self):
        df = make_labelled_df(10)
        X, _, names = extract_features(df)
        assert names == [f"f{i}" for i in range(5)]


class TestScale:
    def test_output_dtype(self):
        rng    = np.random.default_rng(1)
        X      = rng.standard_normal((50, 10))
        scaler = StandardScaler().fit(X)
        Xs     = scale(X, scaler, dtype=np.float32)
        assert Xs.dtype == np.float32

    def test_shape_preserved(self):
        rng    = np.random.default_rng(2)
        X      = rng.standard_normal((50, 10))
        scaler = StandardScaler().fit(X)
        Xs     = scale(X, scaler)
        assert Xs.shape == X.shape

    def test_mean_approximately_zero(self):
        rng    = np.random.default_rng(3)
        X      = rng.standard_normal((1000, 5)) * 100 + 50
        scaler = StandardScaler().fit(X)
        Xs     = scale(X, scaler)
        assert abs(Xs.mean()) < 0.1


class TestOversample:
    def test_output_balanced(self):
        X = np.random.default_rng(0).standard_normal((100, 5))
        y = np.array([0] * 90 + [1] * 10)
        X_res, y_res = oversample(X, y, verbose=False)
        unique, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_shape_increase(self):
        X = np.zeros((100, 3))
        y = np.array([0] * 80 + [1] * 20)
        X_res, y_res = oversample(X, y, verbose=False)
        assert len(X_res) > len(X)
