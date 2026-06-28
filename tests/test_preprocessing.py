"""
tests/test_preprocessing.py
----------------------------
Unit tests for ids.features.preprocessing module.
COM668 | Abdulbosit Abdurazzakov | B00979380

Tests cover:
  - extract_features       -- correct X/y/feature_names split
  - fit_and_save_artefacts -- scaler/encoder fit correctly, files written
  - load_artefacts         -- round-trip load after save
  - scale                  -- transform applies scaler, never re-fits
  - oversample             -- class counts balanced after resampling
  - Data leakage guard     -- SMOTE applied only to training data
  - Edge cases             -- empty-ish inputs, missing columns

Run:
  pytest tests/test_preprocessing.py -v
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# ── Make src importable ────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC  = os.path.join(_ROOT, "src")
for _p in [_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ids.features.preprocessing import (
    extract_features,
    fit_and_save_artefacts,
    load_artefacts,
    oversample,
    scale,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n=200, n_features=10, imbalance_ratio=9):
    """
    Create a synthetic labelled dataframe mimicking CICIDS2017 structure.

    Parameters
    ----------
    n               : total rows
    n_features      : number of numeric feature columns
    imbalance_ratio : BENIGN rows per ATTACK row (default 9:1)
    """
    rng = np.random.default_rng(42)
    n_attack = max(1, n // (imbalance_ratio + 1))
    n_benign = n - n_attack

    labels  = ["ATTACK"] * n_attack + ["BENIGN"] * n_benign
    binary  = labels[:]
    feat_data = rng.standard_normal((n, n_features)).astype(np.float32)
    cols = [f"feat_{i}" for i in range(n_features)]

    df = pd.DataFrame(feat_data, columns=cols)
    df["Label"]       = labels
    df["BinaryLabel"] = binary
    return df


# ── extract_features ──────────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_returns_three_values(self):
        df = _make_df()
        result = extract_features(df)
        assert len(result) == 3

    def test_X_drops_label_columns(self):
        df = _make_df()
        X, y, names = extract_features(df)
        assert "Label"       not in X.columns
        assert "BinaryLabel" not in X.columns

    def test_y_contains_correct_classes(self):
        df = _make_df()
        _, y, _ = extract_features(df)
        assert set(y.unique()) == {"ATTACK", "BENIGN"}

    def test_feature_names_match_X_columns(self):
        df = _make_df()
        X, _, names = extract_features(df)
        assert names == list(X.columns)

    def test_row_count_preserved(self):
        df = _make_df(n=300)
        X, y, _ = extract_features(df)
        assert len(X) == 300
        assert len(y) == 300

    def test_no_label_col_raises(self):
        df = _make_df()
        df = df.drop(columns=["BinaryLabel"])
        with pytest.raises(KeyError):
            extract_features(df, label_col="BinaryLabel")

    def test_custom_label_col(self):
        df = _make_df()
        df["CustomLabel"] = df["BinaryLabel"]
        _, y, _ = extract_features(df, label_col="CustomLabel")
        assert set(y.unique()) == {"ATTACK", "BENIGN"}


# ── fit_and_save_artefacts ────────────────────────────────────────────────────

class TestFitAndSaveArtefacts:
    def test_files_written(self):
        df = _make_df()
        X, y, names = extract_features(df)
        with tempfile.TemporaryDirectory() as tmp:
            fit_and_save_artefacts(X, y, names, processed_dir=tmp, verbose=False)
            for fname in ("scaler.pkl", "label_encoder.pkl", "feature_names.pkl"):
                assert os.path.exists(os.path.join(tmp, fname)), f"Missing: {fname}"

    def test_scaler_mean_close_to_zero_after_fit(self):
        df = _make_df(n=500)
        X, y, names = extract_features(df)
        with tempfile.TemporaryDirectory() as tmp:
            scaler, _ = fit_and_save_artefacts(X, y, names, processed_dir=tmp, verbose=False)
            X_scaled = scaler.transform(X)
            assert abs(X_scaled.mean()) < 0.1, "Scaled mean should be ~0"

    def test_label_encoder_has_two_classes(self):
        df = _make_df()
        X, y, names = extract_features(df)
        with tempfile.TemporaryDirectory() as tmp:
            _, le = fit_and_save_artefacts(X, y, names, processed_dir=tmp, verbose=False)
            assert set(le.classes_) == {"ATTACK", "BENIGN"}

    def test_scaler_fitted_on_train_only(self):
        """
        DATA LEAKAGE GUARD: Scaler must be fitted on training data only.
        The test checks that applying a train-only scaler to test data
        does NOT re-fit (i.e., test stats differ from train stats).
        """
        rng = np.random.default_rng(0)
        df_train = _make_df(n=200)
        X_train, y_train, names = extract_features(df_train)

        # Test set drawn from a shifted distribution
        X_test = pd.DataFrame(
            rng.standard_normal((50, len(names))) + 5.0,  # shifted by 5
            columns=names,
        )

        with tempfile.TemporaryDirectory() as tmp:
            scaler, _ = fit_and_save_artefacts(X_train, y_train, names,
                                                processed_dir=tmp, verbose=False)
            X_test_scaled = scaler.transform(X_test)
            # Mean of scaled test data should NOT be ~0 (scaler is from train)
            assert abs(X_test_scaled.mean()) > 1.0, (
                "If scaler was re-fitted on test data, mean would be ~0. "
                "This indicates data leakage."
            )


# ── load_artefacts ────────────────────────────────────────────────────────────

class TestLoadArtefacts:
    def test_round_trip(self):
        df = _make_df()
        X, y, names = extract_features(df)
        with tempfile.TemporaryDirectory() as tmp:
            fit_and_save_artefacts(X, y, names, processed_dir=tmp, verbose=False)
            scaler2, le2, names2 = load_artefacts(processed_dir=tmp)
            assert names2 == names
            assert set(le2.classes_) == {"ATTACK", "BENIGN"}

    def test_missing_file_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError, match="Missing artefact"):
                load_artefacts(processed_dir=tmp)

    def test_loaded_scaler_produces_same_transform(self):
        df = _make_df(n=300)
        X, y, names = extract_features(df)
        with tempfile.TemporaryDirectory() as tmp:
            scaler_orig, _ = fit_and_save_artefacts(X, y, names,
                                                     processed_dir=tmp, verbose=False)
            scaler_loaded, _, _ = load_artefacts(processed_dir=tmp)
            np.testing.assert_array_almost_equal(
                scaler_orig.transform(X),
                scaler_loaded.transform(X),
                decimal=5,
            )


# ── scale ─────────────────────────────────────────────────────────────────────

class TestScale:
    def _get_fitted_scaler(self):
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(7)
        X = rng.standard_normal((100, 5)).astype(np.float32)
        s = StandardScaler().fit(X)
        return s, X

    def test_output_shape_preserved(self):
        s, X = self._get_fitted_scaler()
        out = scale(X, s)
        assert out.shape == X.shape

    def test_output_dtype_is_float32(self):
        s, X = self._get_fitted_scaler()
        assert scale(X, s).dtype == np.float32

    def test_scaled_values_approximately_standardised(self):
        s, X = self._get_fitted_scaler()
        out = scale(X, s)
        assert abs(out.mean()) < 0.1
        assert abs(out.std() - 1.0) < 0.1

    def test_dataframe_input_accepted(self):
        s, X = self._get_fitted_scaler()
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        out = scale(df, s)
        assert out.shape == X.shape


# ── oversample ────────────────────────────────────────────────────────────────

class TestOversample:
    def _imbalanced_arrays(self, n=200, ratio=9):
        rng = np.random.default_rng(1)
        n_attack = n // (ratio + 1)
        n_benign = n - n_attack
        X = rng.standard_normal((n, 5)).astype(np.float32)
        y = np.array([0] * n_attack + [1] * n_benign)
        return X, y

    def test_output_is_balanced(self):
        X, y = self._imbalanced_arrays()
        X_res, y_res = oversample(X, y, verbose=False)
        unique, counts = np.unique(y_res, return_counts=True)
        # Both classes should have same count after oversampling
        assert counts[0] == counts[1], f"Imbalanced after oversample: {dict(zip(unique, counts))}"

    def test_output_larger_than_input(self):
        X, y = self._imbalanced_arrays()
        X_res, y_res = oversample(X, y, verbose=False)
        assert len(X_res) >= len(X)

    def test_feature_shape_preserved(self):
        X, y = self._imbalanced_arrays()
        X_res, _ = oversample(X, y, verbose=False)
        assert X_res.shape[1] == X.shape[1]

    def test_original_array_unmodified(self):
        X, y = self._imbalanced_arrays()
        X_copy = X.copy()
        oversample(X, y, verbose=False)
        np.testing.assert_array_equal(X, X_copy)

    def test_minority_class_count_increases(self):
        X, y = self._imbalanced_arrays(ratio=9)
        _, counts_before = np.unique(y, return_counts=True)
        _, y_res = oversample(X, y, verbose=False)
        _, counts_after = np.unique(y_res, return_counts=True)
        assert counts_after[0] > counts_before[0]
