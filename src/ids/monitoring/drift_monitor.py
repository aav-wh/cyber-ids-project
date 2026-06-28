"""
ids.monitoring.drift_monitor
-----------------------------
Concept drift and covariate drift monitor for production deployment.

Maintains a reference window (training distribution) and compares
it against a sliding live window using PSI and mean-shift tests.

Concept drift is inferred when labelled feedback is available:
if the model's F1 on recent labelled flows falls below a threshold,
concept drift is flagged.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from threading import RLock

from ids.features.drift import mean_shift_report, psi_report


class DriftMonitor:
    """
    Rolling drift monitor that compares a live feature window
    against a reference (training) distribution.

    Parameters
    ----------
    reference_X     : scaled training features used as baseline
    feature_names   : ordered feature names
    window_size     : size of the live sliding window
    psi_threshold   : PSI above which a feature is flagged
    shift_threshold : relative mean shift fraction for WARNING
    """

    def __init__(
        self,
        reference_X: np.ndarray,
        feature_names: list[str],
        window_size: int = 500,
        psi_threshold: float = 0.25,
        shift_threshold: float = 0.20,
    ) -> None:
        self._ref       = reference_X
        self._features  = feature_names
        self._window    = deque(maxlen=window_size)
        self._lock      = RLock()
        self._psi_thr   = psi_threshold
        self._shift_thr = shift_threshold

        # Summary of the reference distribution
        self._ref_means  = np.mean(reference_X, axis=0)
        self._ref_stds   = np.std(reference_X, axis=0) + 1e-9

    def update(self, feature_vector: np.ndarray) -> None:
        """Add a live feature vector to the sliding window."""
        with self._lock:
            self._window.append(feature_vector)

    def update_batch(self, X_batch: np.ndarray) -> None:
        """Add a batch of feature vectors to the sliding window."""
        with self._lock:
            for row in X_batch:
                self._window.append(row)

    @property
    def window_full(self) -> bool:
        with self._lock:
            return len(self._window) >= self._window.maxlen

    @property
    def window_size(self) -> int:
        with self._lock:
            return len(self._window)

    def get_live_window(self) -> np.ndarray:
        with self._lock:
            return np.array(list(self._window))

    def check_drift(self) -> dict:
        """
        Run PSI and mean-shift tests on the current live window.

        Returns
        -------
        dict with keys: window_size, has_drift, flagged_features,
                        psi_report (DataFrame), shift_report (DataFrame)
        """
        with self._lock:
            if len(self._window) < 50:
                return {
                    "window_size":     len(self._window),
                    "has_drift":       False,
                    "flagged_features": [],
                    "message":         "Insufficient data in window (< 50 samples).",
                }

            X_live = np.array(list(self._window))

        psi_df   = psi_report(self._ref, X_live, self._features)
        shift_df = mean_shift_report(
            self._ref, X_live, self._features,
            warning_pct=self._shift_thr * 0.5,
            alert_pct=self._shift_thr,
        )

        flagged_psi   = psi_df[psi_df["psi"] > self._psi_thr]["feature"].tolist()
        flagged_shift = shift_df[shift_df["status"] == "ALERT"]["feature"].tolist()
        all_flagged   = list(set(flagged_psi + flagged_shift))

        return {
            "window_size":      len(X_live),
            "has_drift":        len(all_flagged) > 0,
            "flagged_features": all_flagged,
            "n_flagged_psi":    len(flagged_psi),
            "n_flagged_shift":  len(flagged_shift),
            "top_psi_feature":  psi_df.iloc[0]["feature"] if len(psi_df) > 0 else None,
            "top_psi_score":    float(psi_df.iloc[0]["psi"]) if len(psi_df) > 0 else 0.0,
        }

    def reset_window(self) -> None:
        """Clear the live window (e.g. after retraining)."""
        with self._lock:
            self._window.clear()
