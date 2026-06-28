"""
ids.visualization.dashboard_helpers
------------------------------------
Helper functions for the Flask live dashboard.

These produce lightweight JSON-serialisable summaries consumed by the
/api/stats, /api/feed, and /api/shap dashboard endpoints.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def feed_to_attack_rate_series(
    feed: deque,
    window: int = 20,
) -> list[float]:
    """
    Compute a rolling attack-rate time series from the detection feed.

    Returns the last `window` per-point attack indicators (1 or 0)
    suitable for a sparkline chart.

    Parameters
    ----------
    feed   : deque of prediction feed entries (newest-first)
    window : number of recent predictions to include

    Returns
    -------
    list of float (1.0 = ATTACK, 0.0 = BENIGN), chronological order
    """
    entries = list(feed)[:window]
    entries.reverse()  # oldest first for chart
    return [1.0 if e["final_decision"] == "ATTACK" else 0.0
            for e in entries]


def aggregate_attack_types(feed: deque) -> dict[str, int]:
    """
    Count predictions by RF label across the current feed window.

    Returns
    -------
    dict mapping label → count
    """
    counts: dict[str, int] = {}
    for entry in feed:
        label = entry.get("rf_prediction", "UNKNOWN")
        counts[label] = counts.get(label, 0) + 1
    return counts


def compute_confidence_histogram(
    feed: deque,
    n_bins: int = 10,
) -> dict:
    """
    Compute a histogram of RF confidence scores from the feed.

    Returns
    -------
    dict with keys: bins (list of floats), counts (list of ints)
    """
    confidences = [e["rf_confidence"] for e in feed if "rf_confidence" in e]

    if not confidences:
        return {"bins": [], "counts": []}

    counts, edges = np.histogram(confidences, bins=n_bins, range=(0, 1))
    return {
        "bins":   [round(float(e), 2) for e in edges[:-1]],
        "counts": counts.tolist(),
    }


def top_shap_features(
    rf_model,
    feature_names: list[str],
    top_n: int = 15,
) -> list[dict]:
    """
    Return top-N features by RF MDI importance as dashboard-ready JSON.

    This is the fast MDI path used by /api/shap to avoid SHAP overhead
    on every dashboard refresh.

    Returns
    -------
    list of {"feature": str, "importance": float} sorted descending
    """
    importances = rf_model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    return [
        {
            "feature":    feature_names[i],
            "importance": round(float(importances[i]), 6),
        }
        for i in order
    ]


def format_prediction_for_feed(
    result: dict,
    latency_ms: float,
    ts: str,
) -> dict:
    """
    Format a classify_flow result for insertion into the detection feed.

    Parameters
    ----------
    result     : output of classify_flow()
    latency_ms : inference latency in milliseconds
    ts         : timestamp string (e.g. "14:32:05")

    Returns
    -------
    Feed entry dict
    """
    return {
        "ts":             ts,
        "final_decision": result["final_decision"],
        "rf_prediction":  result["random_forest"]["prediction"],
        "rf_confidence":  result["random_forest"]["confidence"],
        "attack_prob":    result["random_forest"]["attack_prob"],
        "if_prediction":  result["isolation_forest"]["prediction"],
        "if_score":       result["isolation_forest"]["anomaly_score"],
        "latency_ms":     latency_ms,
    }
