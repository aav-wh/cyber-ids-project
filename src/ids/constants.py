"""
ids.constants
-------------
Project-wide constants that do NOT vary between runs.

Tuneable hyperparameters belong in ids.config.IDSConfig.
Immutable definitions (attack type mapping, feature groups, etc.) live here.
"""

# ── CICIDS2017 attack type taxonomy ───────────────────────────────────────────

ATTACK_TYPES = (
    "DDoS",
    "PortScan",
    "Bot",
    "Infiltration",
    "Web Attack – Brute Force",
    "Web Attack – XSS",
    "Web Attack – Sql Injection",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "DoS Hulk",
    "DoS GoldenEye",
    "Heartbleed",
)

BENIGN_LABEL = "BENIGN"

# Map multi-class attack labels → binary
BINARY_MAP: dict[str, str] = {
    label: "ATTACK" for label in ATTACK_TYPES
}
BINARY_MAP[BENIGN_LABEL] = BENIGN_LABEL

# ── Class indices used by LabelEncoder(['ATTACK', 'BENIGN']) ──────────────────
ATTACK_IDX = 0
BENIGN_IDX = 1
CLASS_NAMES = ["ATTACK", "BENIGN"]

# ── Feature groups (logical clusters used in analysis notebooks) ──────────────

FEATURE_GROUPS: dict[str, list[str]] = {
    "flow_volume": [
        "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Fwd Packet Length Max", "Fwd Packet Length Min",
        "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min",
        "Bwd Packet Length Mean", "Bwd Packet Length Std",
    ],
    "flow_timing": [
        "Flow Duration", "Flow IAT Mean", "Flow IAT Std",
        "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
        "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
        "Bwd IAT Max", "Bwd IAT Min",
    ],
    "flow_rates": [
        "Flow Bytes/s", "Flow Packets/s",
        "Fwd Packets/s", "Bwd Packets/s",
    ],
    "tcp_flags": [
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
        "CWE Flag Count", "ECE Flag Count",
    ],
    "header": [
        "Destination Port", "Protocol",
        "Fwd Header Length", "Bwd Header Length",
    ],
    "bulk_and_subflow": [
        "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
        "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
        "Subflow Fwd Packets", "Subflow Fwd Bytes",
        "Subflow Bwd Packets", "Subflow Bwd Bytes",
    ],
    "active_idle": [
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    ],
    "window_and_segment": [
        "Init_Win_bytes_forward", "Init_Win_bytes_backward",
        "act_data_pkt_fwd", "min_seg_size_forward",
    ],
}

# ── Model file names ──────────────────────────────────────────────────────────
RF_MODEL_FILE       = "random_forest.pkl"
IF_MODEL_FILE       = "isolation_forest.pkl"
SCALER_FILE         = "scaler.pkl"
LABEL_ENCODER_FILE  = "label_encoder.pkl"
FEATURE_NAMES_FILE  = "feature_names.pkl"

# ── API constants ─────────────────────────────────────────────────────────────
API_VERSION    = "v2"
MAX_BATCH_SIZE = 1000
FEED_MAXLEN    = 200

# ── Evaluation thresholds (used in reports) ───────────────────────────────────
ACCEPTABLE_F1_SCORE     = 0.90
ACCEPTABLE_RECALL       = 0.92
ACCEPTABLE_PRECISION    = 0.85
ACCEPTABLE_AUC          = 0.95
ACCEPTABLE_LATENCY_MS   = 10.0    # p95 single-flow inference

# ── SHAP display ──────────────────────────────────────────────────────────────
SHAP_TOP_N_DEFAULT = 20
