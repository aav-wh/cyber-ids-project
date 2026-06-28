"""
ids.detection.simulator
-----------------------
Live detection simulator for notebook 08 and demo purposes.

Replays CICIDS2017 test flows one-by-one through the full inference
pipeline, printing real-time results and accumulating statistics.
Simulates the experience of running the IDS on live network traffic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SimulationStats:
    """Running statistics accumulated during a simulation run."""
    total_flows:    int   = 0
    total_attacks:  int   = 0
    total_benign:   int   = 0
    true_positives: int   = 0
    false_positives: int  = 0
    true_negatives: int   = 0
    false_negatives: int  = 0
    latencies_ms:   list  = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    def summary(self) -> dict:
        return {
            "total_flows":    self.total_flows,
            "total_attacks":  self.total_attacks,
            "total_benign":   self.total_benign,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "precision":      round(self.precision, 4),
            "recall":         round(self.recall, 4),
            "f1":             round(self.f1, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


def run_simulation(
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
    le,
    rf_model,
    iso_model,
    feature_names: list[str],
    n_flows: int = 200,
    delay_s: float = 0.0,
    verbose: bool = True,
    random_state: int = 42,
) -> SimulationStats:
    """
    Replay n_flows from the test set through the full inference pipeline.

    Parameters
    ----------
    X_test, y_test   : scaled test features and integer labels
    scaler, le       : artefacts for classify_flow
    rf_model, iso_model : trained models
    feature_names    : ordered feature names
    n_flows          : number of flows to simulate
    delay_s          : inter-flow delay in seconds (0 for maximum speed)
    verbose          : print each flow's result
    random_state     : seed for random sampling

    Returns
    -------
    SimulationStats with accumulated metrics
    """
    from ids.models.predict import classify_flow

    rng      = np.random.default_rng(random_state)
    indices  = rng.choice(len(X_test), size=min(n_flows, len(X_test)), replace=False)
    stats    = SimulationStats()
    label_names = list(le.classes_)   # ['ATTACK', 'BENIGN']

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Live Detection Simulation  ({n_flows} flows)")
        print(f"{'='*60}")
        header = f"{'#':>5}  {'True':>8}  {'Pred':>8}  {'RF Conf':>8}  {'IF Score':>9}  {'ms':>6}"
        print(header)
        print("-" * 60)

    for i, idx in enumerate(indices):
        fv         = X_test[idx]
        true_label = label_names[y_test[idx]]

        t0     = time.perf_counter()
        result = classify_flow(fv, scaler, le, rf_model, iso_model)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        pred_label = result["final_decision"]
        rf_conf    = result["random_forest"]["confidence"]
        if_score   = result["isolation_forest"]["anomaly_score"]

        # Update stats
        stats.total_flows  += 1
        stats.latencies_ms.append(elapsed_ms)

        if true_label == "ATTACK":
            stats.total_attacks += 1
            if pred_label == "ATTACK":
                stats.true_positives  += 1
            else:
                stats.false_negatives += 1
        else:
            stats.total_benign += 1
            if pred_label == "BENIGN":
                stats.true_negatives  += 1
            else:
                stats.false_positives += 1

        if verbose:
            marker = "✓" if true_label == pred_label else "✗"
            print(f"{i+1:>5}  {true_label:>8}  {pred_label:>8}  "
                  f"{rf_conf:>8.4f}  {if_score:>9.4f}  {elapsed_ms:>5.1f}ms  {marker}")

        if delay_s > 0:
            time.sleep(delay_s)

    if verbose:
        print("=" * 60)
        s = stats.summary()
        print(f"  Flows: {s['total_flows']} | "
              f"TP={s['true_positives']} FP={s['false_positives']} "
              f"TN={s['true_negatives']} FN={s['false_negatives']}")
        print(f"  Precision={s['precision']:.4f}  "
              f"Recall={s['recall']:.4f}  F1={s['f1']:.4f}")
        print(f"  Avg latency: {s['avg_latency_ms']:.2f}ms")
        print("=" * 60)

    return stats
