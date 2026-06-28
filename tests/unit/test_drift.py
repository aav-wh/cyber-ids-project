"""
Unit tests for ids.features.drift
"""

import numpy as np
import pytest

from ids.features.drift import (
    mean_shift_report,
    population_stability_index,
    psi_report,
    kl_divergence,
)


def make_ref_live(n=200, shift=0.0):
    rng = np.random.default_rng(42)
    ref  = rng.standard_normal((n, 5))
    live = rng.standard_normal((n, 5)) + shift
    return ref, live


class TestMeanShiftReport:
    def test_no_shift_all_ok(self):
        ref, live = make_ref_live(shift=0.0)
        names = [f"f{i}" for i in range(5)]
        df = mean_shift_report(ref, live, names)
        assert "status" in df.columns

    def test_large_shift_flagged(self):
        rng  = np.random.default_rng(1)
        ref  = rng.standard_normal((200, 3))
        live = rng.standard_normal((200, 3)) + 100.0  # extreme shift
        names = ["a", "b", "c"]
        df = mean_shift_report(ref, live, names)
        assert any(df["status"] == "ALERT")

    def test_returns_all_features(self):
        ref, live = make_ref_live()
        names = [f"f{i}" for i in range(5)]
        df = mean_shift_report(ref, live, names)
        assert len(df) == 5


class TestPSI:
    def test_identical_distributions_zero(self):
        ref = np.random.default_rng(0).standard_normal(500)
        psi = population_stability_index(ref, ref)
        assert psi == pytest.approx(0.0, abs=0.01)

    def test_different_distributions_positive(self):
        rng  = np.random.default_rng(0)
        ref  = rng.standard_normal(500)
        live = rng.standard_normal(500) + 5.0
        psi  = population_stability_index(ref, live)
        assert psi > 0.1

    def test_constant_feature_returns_zero(self):
        ref  = np.zeros(100)
        live = np.zeros(100)
        assert population_stability_index(ref, live) == 0.0


class TestPSIReport:
    def test_returns_all_features(self):
        ref, live = make_ref_live()
        names = [f"f{i}" for i in range(5)]
        df = psi_report(ref, live, names)
        assert len(df) == 5

    def test_has_status_column(self):
        ref, live = make_ref_live()
        names = [f"f{i}" for i in range(5)]
        df = psi_report(ref, live, names)
        assert "status" in df.columns


class TestKLDivergence:
    def test_same_distribution_near_zero(self):
        rng = np.random.default_rng(0)
        p   = rng.standard_normal(1000)
        kl  = kl_divergence(p, p)
        assert kl == pytest.approx(0.0, abs=0.1)

    def test_different_distributions_positive(self):
        rng = np.random.default_rng(0)
        p   = rng.standard_normal(500)
        q   = rng.standard_normal(500) + 10.0
        assert kl_divergence(p, q) > 0
