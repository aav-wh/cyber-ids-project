"""
Unit tests for ids.detection.alerts
"""

import pytest

from ids.detection.alerts import (
    Alert, AlertSeverity, AlertQueue, generate_alert,
    _severity_from_confidence,
)


def make_attack_result(attack_prob=0.95, if_score=-0.1):
    return {
        "final_decision": "ATTACK",
        "random_forest": {
            "prediction":  "ATTACK",
            "confidence":  attack_prob,
            "attack_prob": attack_prob,
            "benign_prob": 1 - attack_prob,
        },
        "isolation_forest": {
            "prediction":    "ATTACK",
            "anomaly_score": if_score,
        },
        "ensemble": {"prediction": "ATTACK"},
    }


def make_benign_result():
    return {
        "final_decision": "BENIGN",
        "random_forest": {
            "prediction": "BENIGN", "confidence": 0.95,
            "attack_prob": 0.05, "benign_prob": 0.95,
        },
        "isolation_forest": {"prediction": "BENIGN", "anomaly_score": 0.1},
        "ensemble": {"prediction": "BENIGN"},
    }


class TestSeverityFromConfidence:
    def test_critical_high_prob_and_anomaly(self):
        sev = _severity_from_confidence(0.96, -0.1)
        assert sev == AlertSeverity.CRITICAL

    def test_high_prob_alone(self):
        sev = _severity_from_confidence(0.90, 0.1)
        assert sev == AlertSeverity.HIGH

    def test_medium_prob(self):
        sev = _severity_from_confidence(0.75, 0.1)
        assert sev == AlertSeverity.MEDIUM

    def test_low_prob(self):
        sev = _severity_from_confidence(0.60, 0.1)
        assert sev == AlertSeverity.LOW


class TestGenerateAlert:
    def test_attack_produces_alert(self):
        result = make_attack_result()
        alert  = generate_alert(result, latency_ms=2.5)
        assert alert is not None
        assert isinstance(alert, Alert)

    def test_benign_returns_none(self):
        result = make_benign_result()
        assert generate_alert(result, latency_ms=1.0) is None

    def test_alert_has_id(self):
        alert = generate_alert(make_attack_result(), latency_ms=1.0)
        assert alert.alert_id.startswith("ALT-")

    def test_alert_to_dict(self):
        alert = generate_alert(make_attack_result(), latency_ms=1.0)
        d     = alert.to_dict()
        assert d["final_decision"] == "ATTACK"
        assert "severity" in d


class TestAlertQueue:
    def test_push_and_retrieve(self):
        q = AlertQueue(maxlen=10)
        a = generate_alert(make_attack_result(), latency_ms=1.0)
        q.push(a)
        assert len(q.recent(5)) == 1

    def test_maxlen_respected(self):
        q = AlertQueue(maxlen=3)
        for _ in range(5):
            q.push(generate_alert(make_attack_result(), latency_ms=1.0))
        assert len(q.recent(10)) == 3

    def test_counts_by_severity(self):
        q = AlertQueue()
        q.push(generate_alert(make_attack_result(attack_prob=0.96, if_score=-0.1), latency_ms=1.0))
        counts = q.counts_by_severity()
        assert "CRITICAL" in counts
