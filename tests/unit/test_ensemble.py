"""
Unit tests for ids.models.ensemble
"""

import numpy as np
import pytest

from ids.models.ensemble import ensemble_or, ensemble_and, ensemble_weighted_vote, apply_ensemble


ATTACK = 0
BENIGN = 1


class TestEnsembleOR:
    def test_both_attack_returns_attack(self):
        rf = np.array([ATTACK, ATTACK])
        ifo = np.array([ATTACK, ATTACK])
        result = ensemble_or(rf, ifo, attack_label=ATTACK)
        assert (result == ATTACK).all()

    def test_one_attack_returns_attack(self):
        rf  = np.array([BENIGN, ATTACK])
        ifo = np.array([ATTACK, BENIGN])
        result = ensemble_or(rf, ifo, attack_label=ATTACK)
        assert (result == ATTACK).all()

    def test_both_benign_returns_benign(self):
        rf = np.array([BENIGN, BENIGN])
        ifo = np.array([BENIGN, BENIGN])
        result = ensemble_or(rf, ifo, attack_label=ATTACK)
        assert (result == BENIGN).all()


class TestEnsembleAND:
    def test_both_attack_returns_attack(self):
        rf = np.array([ATTACK])
        ifo = np.array([ATTACK])
        assert ensemble_and(rf, ifo, attack_label=ATTACK)[0] == ATTACK

    def test_one_benign_returns_benign(self):
        rf  = np.array([ATTACK])
        ifo = np.array([BENIGN])
        assert ensemble_and(rf, ifo, attack_label=ATTACK)[0] == BENIGN

    def test_both_benign_returns_benign(self):
        rf = np.array([BENIGN])
        ifo = np.array([BENIGN])
        assert ensemble_and(rf, ifo, attack_label=ATTACK)[0] == BENIGN


class TestWeightedVote:
    def test_high_rf_prob_returns_attack(self):
        rf_proba  = np.array([0.95])
        if_scores = np.array([-0.1])
        result    = ensemble_weighted_vote(rf_proba, if_scores, alpha=0.9)
        assert result[0] == ATTACK

    def test_low_rf_prob_returns_benign(self):
        rf_proba  = np.array([0.05])
        if_scores = np.array([0.1])
        result    = ensemble_weighted_vote(rf_proba, if_scores, alpha=0.9)
        assert result[0] == BENIGN


class TestApplyEnsemble:
    def test_or_rule(self):
        rf = np.array([ATTACK])
        ifo = np.array([BENIGN])
        assert apply_ensemble(rf, ifo, rule="or")[0] == ATTACK

    def test_and_rule(self):
        rf = np.array([ATTACK])
        ifo = np.array([BENIGN])
        assert apply_ensemble(rf, ifo, rule="and")[0] == BENIGN

    def test_invalid_rule_raises(self):
        with pytest.raises(ValueError):
            apply_ensemble(np.array([0]), np.array([0]), rule="invalid")
