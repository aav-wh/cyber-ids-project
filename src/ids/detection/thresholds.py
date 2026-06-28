"""
ids.detection.thresholds
------------------------
Dynamic threshold management for production deployment.

Supports:
  - Static threshold (set at deployment)
  - Adaptive threshold (adjusted based on recent FPR)
  - Per-hour threshold profiles (lower threshold at night)
"""

from __future__ import annotations

import time
from threading import RLock


class StaticThreshold:
    """Fixed decision threshold — RF attack probability >= threshold → ATTACK."""

    def __init__(self, threshold: float = 0.5) -> None:
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self._threshold = threshold

    @property
    def value(self) -> float:
        return self._threshold

    def classify(self, attack_prob: float) -> str:
        return "ATTACK" if attack_prob >= self._threshold else "BENIGN"


class AdaptiveThreshold:
    """
    Threshold that self-adjusts to maintain a target False Positive Rate.

    If the recent FPR exceeds the target, the threshold is raised.
    If FPR falls well below target, the threshold is lowered to catch more attacks.

    Parameters
    ----------
    initial_threshold : starting threshold value
    target_fpr        : desired false positive rate (e.g. 0.02 = 2%)
    window_size       : number of predictions in the sliding window
    learning_rate     : step size for each threshold adjustment
    min_threshold     : lower bound for the threshold
    max_threshold     : upper bound for the threshold
    """

    def __init__(
        self,
        initial_threshold: float = 0.5,
        target_fpr:        float = 0.02,
        window_size:       int   = 200,
        learning_rate:     float = 0.01,
        min_threshold:     float = 0.20,
        max_threshold:     float = 0.80,
    ) -> None:
        self._threshold    = initial_threshold
        self._target_fpr   = target_fpr
        self._window       = []
        self._window_size  = window_size
        self._lr           = learning_rate
        self._min          = min_threshold
        self._max          = max_threshold
        self._lock         = RLock()

    @property
    def value(self) -> float:
        return self._threshold

    def update(self, attack_prob: float, true_label: str) -> None:
        """
        Record a prediction outcome and adapt the threshold.

        Parameters
        ----------
        attack_prob : RF probability of ATTACK
        true_label  : 'ATTACK' or 'BENIGN' (ground truth, if known)
        """
        with self._lock:
            pred = "ATTACK" if attack_prob >= self._threshold else "BENIGN"
            fp   = (pred == "ATTACK" and true_label == "BENIGN")
            tn   = (pred == "BENIGN" and true_label == "BENIGN")

            self._window.append((fp, tn))
            if len(self._window) > self._window_size:
                self._window.pop(0)

            if len(self._window) >= 20:
                fps   = sum(1 for fp_, tn_ in self._window if fp_)
                tns   = sum(1 for fp_, tn_ in self._window if tn_)
                denom = fps + tns
                fpr   = fps / denom if denom > 0 else 0.0

                if fpr > self._target_fpr:
                    self._threshold = min(self._max, self._threshold + self._lr)
                elif fpr < self._target_fpr * 0.5:
                    self._threshold = max(self._min, self._threshold - self._lr)

    def classify(self, attack_prob: float) -> str:
        with self._lock:
            return "ATTACK" if attack_prob >= self._threshold else "BENIGN"

    def stats(self) -> dict:
        with self._lock:
            return {
                "current_threshold": round(self._threshold, 4),
                "window_size":       len(self._window),
                "target_fpr":        self._target_fpr,
            }


class HourlyThresholdProfile:
    """
    Map hour-of-day (0–23) to a threshold value.

    Example: raise the threshold during low-traffic night hours to
    reduce false positives when anomaly volume is naturally higher.
    """

    def __init__(self, profiles: dict[int, float] | None = None) -> None:
        # Default: lower threshold (more sensitive) during business hours
        self._profiles: dict[int, float] = profiles or {
            h: 0.40 if 8 <= h < 18 else 0.55
            for h in range(24)
        }

    def current_threshold(self) -> float:
        hour = time.localtime().tm_hour
        return self._profiles.get(hour, 0.5)

    def classify(self, attack_prob: float) -> str:
        return "ATTACK" if attack_prob >= self.current_threshold() else "BENIGN"

    def set_hour_threshold(self, hour: int, threshold: float) -> None:
        if not 0 <= hour <= 23:
            raise ValueError(f"hour must be 0–23, got {hour}")
        self._profiles[hour] = threshold
