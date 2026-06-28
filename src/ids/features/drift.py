"""
ids.features.drift
------------------
Covariate (feature) drift detection.

Covariate drift occurs when the statistical distribution of input features
changes between training and deployment without a change in the true
mapping from features to labels.  This is distinct from concept drift
(where the mapping itself changes).

Methods implemented
-------------------
- Mean shift detection (fast, interpretable)
- KL-divergence estimation (distribution-level)
- Population Stability Index (PSI) — standard in financial risk models,
  applied here to network traffic monitoring
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_shift_report(
    X_ref: np.ndarray,
    X_live: np.ndarray,
    feature_names: list[str],
    warning_pct: float = 0.10,
    alert_pct: float   = 0.20,
) -> pd.DataFrame:
    """
    Compare per-feature means between a reference and live window.

    Parameters
    ----------
    X_ref, X_live  : feature matrices [n_samples, n_features]
    feature_names  : ordered feature names
    warning_pct    : relative mean shift that triggers WARNING
    alert_pct      : relative mean shift that triggers ALERT

    Returns
    -------
    pd.DataFrame with columns:
      feature, ref_mean, live_mean, abs_shift, rel_shift, status
    """
    ref_means  = np.mean(X_ref,  axis=0)
    live_means = np.mean(X_live, axis=0)
    abs_shift  = np.abs(live_means - ref_means)
    # Avoid division by zero for features with near-zero reference mean
    rel_shift  = abs_shift / (np.abs(ref_means) + 1e-9)

    def _status(r):
        if r >= alert_pct:   return "ALERT"
        if r >= warning_pct: return "WARNING"
        return "OK"

    rows = [
        {
            "feature":   feature_names[i],
            "ref_mean":  round(float(ref_means[i]),  6),
            "live_mean": round(float(live_means[i]), 6),
            "abs_shift": round(float(abs_shift[i]),  6),
            "rel_shift": round(float(rel_shift[i]),  6),
            "status":    _status(rel_shift[i]),
        }
        for i in range(len(feature_names))
    ]

    return pd.DataFrame(rows).sort_values("rel_shift", ascending=False).reset_index(drop=True)


def population_stability_index(
    ref: np.ndarray,
    live: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute the Population Stability Index (PSI) for a single feature.

    PSI interpretation:
      < 0.10  — No drift (stable)
      0.10–0.25 — Moderate drift (monitor)
      > 0.25  — Significant drift (retrain)

    Parameters
    ----------
    ref, live : 1-D arrays of feature values
    n_bins    : number of histogram bins (computed on ref distribution)
    epsilon   : small value to avoid log(0)

    Returns
    -------
    float — PSI score
    """
    ref_min, ref_max = ref.min(), ref.max()
    if ref_min == ref_max:
        return 0.0  # constant feature — no drift possible

    bins = np.linspace(ref_min, ref_max, n_bins + 1)
    ref_counts,  _ = np.histogram(ref,  bins=bins)
    live_counts, _ = np.histogram(live, bins=bins)

    ref_pct  = ref_counts  / len(ref)  + epsilon
    live_pct = live_counts / len(live) + epsilon

    psi = float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))
    return round(psi, 6)


def psi_report(
    X_ref: np.ndarray,
    X_live: np.ndarray,
    feature_names: list[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute PSI for every feature and return a ranked report.

    Returns
    -------
    pd.DataFrame with columns: feature, psi, status
    """
    rows = []
    for i, name in enumerate(feature_names):
        psi = population_stability_index(X_ref[:, i], X_live[:, i], n_bins)
        if psi > 0.25:
            status = "SIGNIFICANT"
        elif psi > 0.10:
            status = "MODERATE"
        else:
            status = "STABLE"
        rows.append({"feature": name, "psi": psi, "status": status})

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    n_bins: int = 50,
    epsilon: float = 1e-9,
) -> float:
    """
    Estimate KL divergence D(P || Q) for two continuous distributions.

    Uses histogram binning on the combined range.

    Returns
    -------
    float — KL divergence (0 = identical distributions)
    """
    all_vals = np.concatenate([p, q])
    lo, hi   = all_vals.min(), all_vals.max()

    if lo == hi:
        return 0.0

    bins = np.linspace(lo, hi, n_bins + 1)
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)

    p_hist = p_hist + epsilon
    q_hist = q_hist + epsilon

    p_norm = p_hist / p_hist.sum()
    q_norm = q_hist / q_hist.sum()

    kl = float(np.sum(p_norm * np.log(p_norm / q_norm)))
    return round(kl, 6)
