"""
ids.detection.alerts
--------------------
Alert data structures and generation logic.

Alerts are produced whenever the ensemble classifies a flow as ATTACK.
They carry severity, confidence, and relevant feature context for the
security analyst reviewing the dashboard.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class AlertSeverity(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """Represents a single IDS detection alert."""

    alert_id:       str
    timestamp:      float
    severity:       AlertSeverity
    final_decision: str          # 'ATTACK' or 'BENIGN'
    rf_prediction:  str
    rf_confidence:  float
    attack_prob:    float
    if_prediction:  str
    if_score:       float
    latency_ms:     float
    feature_snapshot: dict = field(default_factory=dict)
    notes:           str   = ""

    def to_dict(self) -> dict:
        return {
            "alert_id":       self.alert_id,
            "timestamp":      self.timestamp,
            "ts_human":       time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime(self.timestamp)),
            "severity":       self.severity.value,
            "final_decision": self.final_decision,
            "rf_prediction":  self.rf_prediction,
            "rf_confidence":  self.rf_confidence,
            "attack_prob":    self.attack_prob,
            "if_prediction":  self.if_prediction,
            "if_score":       self.if_score,
            "latency_ms":     self.latency_ms,
            "notes":          self.notes,
        }


def _severity_from_confidence(
    attack_prob: float,
    if_score: float,
) -> AlertSeverity:
    """
    Derive alert severity from RF attack probability and IF anomaly score.

    Heuristic rules:
      CRITICAL — RF > 0.95 AND IF anomaly (score < 0)
      HIGH     — RF > 0.85 OR IF score < -0.05
      MEDIUM   — RF > 0.70
      LOW      — below all thresholds
    """
    if attack_prob > 0.95 and if_score < 0:
        return AlertSeverity.CRITICAL
    if attack_prob > 0.85 or if_score < -0.05:
        return AlertSeverity.HIGH
    if attack_prob > 0.70:
        return AlertSeverity.MEDIUM
    return AlertSeverity.LOW


def generate_alert(
    result: dict,
    latency_ms: float,
    feature_snapshot: dict | None = None,
) -> Alert | None:
    """
    Generate an Alert from a classify_flow result.

    Returns None if the final decision is BENIGN (no alert needed).

    Parameters
    ----------
    result           : dict from ids.models.predict.classify_flow()
    latency_ms       : inference time
    feature_snapshot : optional dict of feature values for the analyst

    Returns
    -------
    Alert or None
    """
    if result["final_decision"] != "ATTACK":
        return None

    rf   = result["random_forest"]
    ifo  = result["isolation_forest"]

    severity = _severity_from_confidence(rf["attack_prob"], ifo["anomaly_score"])

    return Alert(
        alert_id        = f"ALT-{int(time.time() * 1000)}",
        timestamp       = time.time(),
        severity        = severity,
        final_decision  = result["final_decision"],
        rf_prediction   = rf["prediction"],
        rf_confidence   = rf["confidence"],
        attack_prob     = rf["attack_prob"],
        if_prediction   = ifo["prediction"],
        if_score        = ifo["anomaly_score"],
        latency_ms      = latency_ms,
        feature_snapshot= feature_snapshot or {},
    )


class AlertQueue:
    """Thread-safe in-memory alert queue with a fixed maximum length."""

    def __init__(self, maxlen: int = 500) -> None:
        from collections import deque
        from threading import Lock
        self._queue: deque[Alert] = deque(maxlen=maxlen)
        self._lock = Lock()

    def push(self, alert: Alert) -> None:
        with self._lock:
            self._queue.appendleft(alert)

    def recent(self, n: int = 50) -> list[Alert]:
        with self._lock:
            return list(self._queue)[:n]

    def counts_by_severity(self) -> dict[str, int]:
        with self._lock:
            counts: dict[str, int] = {}
            for a in self._queue:
                key = a.severity.value
                counts[key] = counts.get(key, 0) + 1
        return counts

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
