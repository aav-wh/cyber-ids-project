"""
ids.monitoring.performance_tracker
------------------------------------
Tracks live model performance when labelled feedback is available.

In production, ground-truth labels may arrive with a delay (e.g. from
a SIEM or analyst review). This tracker accumulates those labelled
outcomes and computes rolling metrics to detect performance degradation.
"""

from __future__ import annotations

import time
from collections import deque
from threading import RLock

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class PerformanceTracker:
    """
    Rolling performance tracker for production IDS.

    Parameters
    ----------
    window_size       : number of labelled flows in the rolling window
    degradation_f1    : F1 score below which a degradation alert is raised
    """

    def __init__(
        self,
        window_size:    int   = 1000,
        degradation_f1: float = 0.85,
    ) -> None:
        self._window     = deque(maxlen=window_size)   # (y_true, y_pred) pairs
        self._lock       = RLock()
        self._deg_f1     = degradation_f1
        self._n_alerts   = 0

    def record(self, y_true: int, y_pred: int) -> None:
        """Record one labelled prediction outcome."""
        with self._lock:
            self._window.append((y_true, y_pred))

    def record_batch(self, y_trues: list[int], y_preds: list[int]) -> None:
        with self._lock:
            for yt, yp in zip(y_trues, y_preds):
                self._window.append((yt, yp))

    def current_metrics(self) -> dict:
        """Compute metrics on the current rolling window."""
        with self._lock:
            if len(self._window) < 10:
                return {"status": "insufficient_data", "n": len(self._window)}

            data = list(self._window)

        y_true = np.array([d[0] for d in data])
        y_pred = np.array([d[1] for d in data])

        f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec  = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

        degraded = f1 < self._deg_f1

        return {
            "n":              len(data),
            "f1":             round(f1, 4),
            "precision":      round(prec, 4),
            "recall":         round(rec, 4),
            "degraded":       degraded,
            "degradation_f1": self._deg_f1,
            "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def check_degradation(self) -> bool:
        """Return True if current F1 is below the degradation threshold."""
        metrics = self.current_metrics()
        degraded = metrics.get("degraded", False)
        if degraded:
            self._n_alerts += 1
        return degraded

    @property
    def n_degradation_alerts(self) -> int:
        return self._n_alerts

    def reset(self) -> None:
        with self._lock:
            self._window.clear()
            self._n_alerts = 0
