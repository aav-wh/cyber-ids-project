"""
ids.models.ensemble
-------------------
Ensemble combination strategies for the Random Forest + Isolation Forest pair.

Three strategies are available:
  OR     — flag ATTACK if EITHER model predicts ATTACK (maximises recall)
  AND    — flag ATTACK only if BOTH models predict ATTACK (maximises precision)
  VOTING — weighted probability vote (requires a calibration weight α)

The OR rule is the project default because in IDS contexts the cost of
a missed attack (false negative) far outweighs a false alarm (false positive).
"""

from __future__ import annotations

import numpy as np


def ensemble_or(
    rf_preds: np.ndarray,
    if_preds: np.ndarray,
    attack_label: int = 0,
) -> np.ndarray:
    """
    OR rule: ATTACK if either model says ATTACK.

    Maximises recall at the cost of some precision — appropriate for IDS
    where missing an attack is more dangerous than a false alarm.

    Parameters
    ----------
    rf_preds, if_preds : integer label arrays (attack_label = 0, benign = 1)
    attack_label       : integer index representing ATTACK class

    Returns
    -------
    np.ndarray of combined integer predictions
    """
    return np.where(
        (rf_preds == attack_label) | (if_preds == attack_label),
        attack_label,
        1 - attack_label,
    )


def ensemble_and(
    rf_preds: np.ndarray,
    if_preds: np.ndarray,
    attack_label: int = 0,
) -> np.ndarray:
    """
    AND rule: ATTACK only if both models say ATTACK.

    Maximises precision — fewer false alarms, more missed attacks.
    Used as a comparison baseline in notebook 06.

    Parameters
    ----------
    rf_preds, if_preds : integer label arrays
    attack_label       : integer index representing ATTACK class

    Returns
    -------
    np.ndarray of combined integer predictions
    """
    return np.where(
        (rf_preds == attack_label) & (if_preds == attack_label),
        attack_label,
        1 - attack_label,
    )


def ensemble_weighted_vote(
    rf_proba: np.ndarray,
    if_scores: np.ndarray,
    alpha: float = 0.7,
    attack_label: int = 0,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Weighted probability vote combining RF probability and IF anomaly score.

    The IF anomaly score is normalised to [0, 1] (1 = most anomalous) and
    blended with the RF attack probability using weight α.

    final_score = α × P_RF(ATTACK) + (1 − α) × IF_normalised

    Parameters
    ----------
    rf_proba   : RF predicted probability of ATTACK, shape [n_samples]
    if_scores  : IF decision_function output (more negative = more anomalous)
    alpha      : weight for RF contribution (0–1); IF gets (1−α)
    threshold  : score above which the flow is labelled ATTACK

    Returns
    -------
    np.ndarray of integer predictions
    """
    # Normalise IF scores: invert (negative = anomalous) then scale to [0,1]
    if_min, if_max = if_scores.min(), if_scores.max()
    if if_max - if_min < 1e-9:
        if_norm = np.zeros_like(if_scores)
    else:
        # More negative → higher anomaly score → normalised closer to 1
        if_norm = 1.0 - (if_scores - if_min) / (if_max - if_min)

    combined = alpha * rf_proba + (1.0 - alpha) * if_norm
    return np.where(combined >= threshold, attack_label, 1 - attack_label)


def apply_ensemble(
    rf_preds: np.ndarray,
    if_preds: np.ndarray,
    rule: str = "or",
    attack_label: int = 0,
    **kwargs,
) -> np.ndarray:
    """
    Dispatch to the appropriate ensemble rule by name.

    Parameters
    ----------
    rf_preds, if_preds : integer prediction arrays
    rule               : 'or' | 'and' | 'voting'
    attack_label       : integer representing ATTACK class
    **kwargs           : extra arguments forwarded to the rule function

    Returns
    -------
    np.ndarray of combined predictions
    """
    rule = rule.lower()
    if rule == "or":
        return ensemble_or(rf_preds, if_preds, attack_label)
    elif rule == "and":
        return ensemble_and(rf_preds, if_preds, attack_label)
    elif rule in ("voting", "weighted"):
        rf_proba   = kwargs.get("rf_proba",   rf_preds.astype(float))
        if_scores  = kwargs.get("if_scores",  if_preds.astype(float))
        alpha      = kwargs.get("alpha",      0.7)
        threshold  = kwargs.get("threshold",  0.5)
        return ensemble_weighted_vote(rf_proba, if_scores, alpha, attack_label, threshold)
    else:
        raise ValueError(f"Unknown ensemble rule: '{rule}'. Choose 'or', 'and', or 'voting'.")
