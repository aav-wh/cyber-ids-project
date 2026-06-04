"""
ids.models.predict
------------------
Inference helpers: load models + run classify_flow.

This module is the single source of truth for inference logic.
The Flask API imports classify_flow from here rather than re-implementing it.
"""

import os

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

_DEFAULT_PROCESSED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data", "processed",
)
_DEFAULT_MODELS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "models",
)


def load_models(
    processed_dir: str = _DEFAULT_PROCESSED,
    models_dir: str = _DEFAULT_MODELS,
) -> tuple[StandardScaler, LabelEncoder, list[str], RandomForestClassifier, IsolationForest]:
    """
    Load all inference artefacts from disk.

    Raises FileNotFoundError with a clear message if any file is missing —
    the server should fail fast at startup, not mid-request.

    Returns
    -------
    scaler, label_encoder, feature_names, rf_model, iso_model
    """
    required = {
        "scaler":           os.path.join(processed_dir, "scaler.pkl"),
        "label_encoder":    os.path.join(processed_dir, "label_encoder.pkl"),
        "feature_names":    os.path.join(processed_dir, "feature_names.pkl"),
        "random_forest":    os.path.join(models_dir,    "random_forest.pkl"),
        "isolation_forest": os.path.join(models_dir,    "isolation_forest.pkl"),
    }

    for label, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing artefact '{label}' at '{path}'. "
                "Run the training notebooks to generate all required files."
            )

    scaler        = joblib.load(required["scaler"])
    le            = joblib.load(required["label_encoder"])
    feature_names = joblib.load(required["feature_names"])
    rf_model      = joblib.load(required["random_forest"])
    iso_model     = joblib.load(required["isolation_forest"])

    return scaler, le, feature_names, rf_model, iso_model


def classify_flow(
    feature_vector: list[float] | np.ndarray,
    scaler: StandardScaler,
    le: LabelEncoder,
    rf_model: RandomForestClassifier,
    iso_model: IsolationForest,
) -> dict:
    """
    Classify a single network flow using RF, IF, and the OR-rule ensemble.

    Detection strategy
    ------------------
    Random Forest   — supervised, predicts exact class + confidence
    Isolation Forest — semi-supervised, anomaly score from benign-only training
    Ensemble (OR)   — ATTACK if either model flags; maximises recall at the
                      cost of some precision (appropriate for IDS: missing an
                      attack is worse than a false alarm)

    Parameters
    ----------
    feature_vector : raw (unscaled) feature values, length == len(feature_names)
    scaler         : fitted StandardScaler
    le             : fitted LabelEncoder (classes_: ['ATTACK', 'BENIGN'])
    rf_model       : fitted RandomForestClassifier
    iso_model      : fitted IsolationForest

    Returns
    -------
    dict with keys: random_forest, isolation_forest, ensemble, final_decision
    """
    # Reshape to (1, n_features) and scale — MUST use transform(), not fit_transform()
    X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_pred       = rf_model.predict(X_scaled)[0]         # integer class index
    rf_proba      = rf_model.predict_proba(X_scaled)[0]   # [P(ATTACK), P(BENIGN)]
    rf_label      = le.inverse_transform([rf_pred])[0]    # 'ATTACK' or 'BENIGN'
    rf_confidence = float(rf_proba[rf_pred])

    # ── Isolation Forest ──────────────────────────────────────────────────────
    # predict() returns 1 (normal/BENIGN) or -1 (anomaly/ATTACK)
    # decision_function() returns anomaly score: more negative = more anomalous
    if_raw   = iso_model.predict(X_scaled)[0]
    if_score = float(iso_model.decision_function(X_scaled)[0])
    if_pred  = 1 if if_raw == 1 else 0          # remap to 1=BENIGN, 0=ATTACK
    if_label = le.inverse_transform([if_pred])[0]

    # ── Ensemble: OR rule ─────────────────────────────────────────────────────
    ensemble_pred  = 0 if (rf_pred == 0 or if_pred == 0) else 1
    ensemble_label = le.inverse_transform([ensemble_pred])[0]

    # ── ATTACK class index (for consistent probability reporting) ──────────────
    attack_idx = list(le.classes_).index("ATTACK")
    benign_idx = list(le.classes_).index("BENIGN")

    return {
        "random_forest": {
            "prediction":  rf_label,
            "confidence":  round(rf_confidence, 4),
            "attack_prob": round(float(rf_proba[attack_idx]), 4),
            "benign_prob": round(float(rf_proba[benign_idx]), 4),
        },
        "isolation_forest": {
            "prediction":    if_label,
            "anomaly_score": round(if_score, 4),
            "note": "More negative score = more anomalous = more likely ATTACK",
        },
        "ensemble": {
            "prediction": ensemble_label,
            "rule":       "OR — ATTACK if either model flags (maximises recall)",
        },
        "final_decision": ensemble_label,
    }
