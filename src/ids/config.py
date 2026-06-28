"""
ids.config
----------
Centralised configuration for the AI-IDS project.

All tuneable parameters live here so notebooks, scripts, and the API
all read from one source of truth.  Override individual settings by
passing keyword arguments to IDSConfig().

Usage
-----
    from ids.config import CFG
    print(CFG.RF_N_ESTIMATORS)   # 150

    custom = IDSConfig(RF_N_ESTIMATORS=300)
    print(custom.RF_N_ESTIMATORS)  # 300
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field

# Project root — one level above src/
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class IDSConfig:
    # ── Paths ──────────────────────────────────────────────────────────────
    PROJECT_ROOT:    str = _ROOT
    DATA_DIR:        str = field(default_factory=lambda: os.path.join(_ROOT, "data"))
    PROCESSED_DIR:   str = field(default_factory=lambda: os.path.join(_ROOT, "data", "processed"))
    MODELS_DIR:      str = field(default_factory=lambda: os.path.join(_ROOT, "models"))
    RESULTS_DIR:     str = field(default_factory=lambda: os.path.join(_ROOT, "results"))

    # ── Dataset ────────────────────────────────────────────────────────────
    TRAIN_FILES: tuple = (
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
    )
    TEST_FILES: tuple = (
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    )
    LABEL_COL:        str = "Label"
    BINARY_LABEL_COL: str = "BinaryLabel"
    BENIGN_CLASS:     str = "BENIGN"
    ATTACK_CLASS:     str = "ATTACK"

    # ── Random Forest ──────────────────────────────────────────────────────
    RF_N_ESTIMATORS: int   = 150
    RF_RANDOM_STATE: int   = 42
    RF_N_JOBS:       int   = -1
    RF_CLASS_WEIGHT: str   = "balanced"
    RF_MAX_DEPTH:    int | None = None
    RF_MIN_SAMPLES_SPLIT: int  = 2

    # ── Isolation Forest ───────────────────────────────────────────────────
    IF_CONTAMINATION: float = 0.01
    IF_N_ESTIMATORS:  int   = 100
    IF_RANDOM_STATE:  int   = 42
    IF_N_JOBS:        int   = -1

    # ── Ensemble ───────────────────────────────────────────────────────────
    ENSEMBLE_RULE: str = "or"    # "or" | "and" | "voting"

    # ── Preprocessing ──────────────────────────────────────────────────────
    SCALER_DTYPE: str = "float32"

    # ── Threshold optimisation ─────────────────────────────────────────────
    THRESHOLD_METRIC:    str   = "f1"   # "f1" | "recall" | "precision"
    BOOTSTRAP_N_ITER:    int   = 200
    BOOTSTRAP_SEED:      int   = 42

    # ── API ────────────────────────────────────────────────────────────────
    API_HOST:           str   = "0.0.0.0"
    API_PORT:           int   = 5000
    API_DEBUG:          bool  = False
    API_FEED_MAXLEN:    int   = 200
    API_BATCH_LIMIT:    int   = 1000

    # ── Monitoring / drift ─────────────────────────────────────────────────
    DRIFT_WINDOW_SIZE:  int   = 500
    DRIFT_WARNING_PCT:  float = 0.10   # >10% feature mean shift triggers warning
    DRIFT_ALERT_PCT:    float = 0.20   # >20% triggers alert

    # ── Logging ────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    def __post_init__(self):
        """Ensure output directories exist."""
        for d in (self.PROCESSED_DIR, self.MODELS_DIR, self.RESULTS_DIR):
            os.makedirs(d, exist_ok=True)


# Singleton default config used across the codebase
CFG = IDSConfig()
