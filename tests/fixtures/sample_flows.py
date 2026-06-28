"""
tests/fixtures/sample_flows.py
-------------------------------
Hard-coded sample network flows for deterministic unit tests.

These are synthetic flows with realistic feature magnitudes,
NOT from the actual CICIDS2017 dataset, so they can be committed
to the repo without data licensing concerns.
"""

from __future__ import annotations

# 78-feature zero vector (safe baseline)
ZERO_FLOW = [0.0] * 78

# Synthetic high-confidence attack flow (SYN-flood-like characteristics)
SYNTHETIC_ATTACK_FLOW = [
    80,       # Destination Port (HTTP)
    100,      # Flow Duration (short)
    1000,     # Total Fwd Packets (high)
    0,        # Total Backward Packets (asymmetric)
    64000,    # Total Length of Fwd Packets
    0,        # Total Length of Bwd Packets
    64.0,     # Fwd Packet Length Max
    64.0,     # Fwd Packet Length Min
    64.0,     # Fwd Packet Length Mean
    0.0,      # Fwd Packet Length Std
    0.0,      # Bwd Packet Length Max
    0.0,      # Bwd Packet Length Min
    0.0,      # Bwd Packet Length Mean
    0.0,      # Bwd Packet Length Std
] + [0.0] * 64  # remaining features zeroed


# Synthetic benign flow (normal HTTPS browsing)
SYNTHETIC_BENIGN_FLOW = [
    443,      # Destination Port (HTTPS)
    500000,   # Flow Duration (normal)
    15,       # Total Fwd Packets
    12,       # Total Backward Packets
    4500,     # Total Length of Fwd Packets
    3600,     # Total Length of Bwd Packets
    300.0,    # Fwd Packet Length Max
    40.0,     # Fwd Packet Length Min
    150.0,    # Fwd Packet Length Mean
    80.0,     # Fwd Packet Length Std
    400.0,    # Bwd Packet Length Max
    50.0,     # Bwd Packet Length Min
    200.0,    # Bwd Packet Length Mean
    100.0,    # Bwd Packet Length Std
] + [0.0] * 64


# Port scan flow characteristics
SYNTHETIC_PORT_SCAN_FLOW = [
    22,       # Destination Port (SSH — common scan target)
    1000,     # Flow Duration (very short)
    1,        # Total Fwd Packets
    0,        # Total Backward Packets
    60,       # Total Length of Fwd Packets
    0,        # Total Length of Bwd Packets
] + [0.0] * 72


def get_sample_batch(n: int = 10) -> list[list[float]]:
    """
    Return a small mixed batch of synthetic attack and benign flows.

    Parameters
    ----------
    n : total flows to return (evenly split ATTACK / BENIGN)

    Returns
    -------
    list of n feature vectors
    """
    import numpy as np
    rng    = np.random.default_rng(42)
    flows  = []
    half   = n // 2

    # Attack-like: short duration, high fwd packets
    for _ in range(half):
        f = ZERO_FLOW.copy()
        f[0] = float(rng.integers(1, 1024))  # well-known port
        f[2] = float(rng.integers(500, 2000))
        f[1] = float(rng.integers(10, 1000))
        flows.append(f)

    # Benign-like: normal HTTPS traffic
    for _ in range(n - half):
        f = ZERO_FLOW.copy()
        f[0] = 443.0
        f[2] = float(rng.integers(5, 30))
        f[1] = float(rng.integers(100000, 1000000))
        flows.append(f)

    return flows
