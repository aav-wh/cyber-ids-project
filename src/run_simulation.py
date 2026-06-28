"""
COM668 Final Year Project — AI-Based Intrusion Detection System
Live Detection Simulation Runner
Author : Abdulbosit Abdurazzakov | B00979380

Replays test-set flows through the inference pipeline in real time,
printing per-flow decisions and a final statistics summary.

Usage
-----
    python src/run_simulation.py
    python src/run_simulation.py --n-flows 500 --delay 0.05
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC  = os.path.join(_ROOT, "src")
for p in [_ROOT, _SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from ids.config import CFG
from ids.models.predict import load_models
from ids.detection.simulator import run_simulation


def parse_args():
    p = argparse.ArgumentParser(description="AI-IDS Live Detection Simulation")
    p.add_argument("--n-flows",      type=int,   default=200, help="Number of flows to simulate")
    p.add_argument("--delay",        type=float, default=0.0, help="Inter-flow delay in seconds")
    p.add_argument("--random-state", type=int,   default=42)
    p.add_argument("--processed-dir", default=CFG.PROCESSED_DIR)
    p.add_argument("--models-dir",    default=CFG.MODELS_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  AI-IDS Live Detection Simulation")
    print("  COM668 | Abdulbosit Abdurazzakov | B00979380")
    print("=" * 60)

    # Load artefacts
    scaler, le, feature_names, rf_model, iso_model = load_models(
        args.processed_dir, args.models_dir
    )

    # Load test data
    y_test_path = os.path.join(args.processed_dir, "y_test.npy")
    X_test_path = os.path.join(args.processed_dir, "X_test.npy")

    if os.path.exists(X_test_path):
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
    else:
        # Build from CSVs on the fly
        from ids.data.loader import load_and_clean
        from ids.features.preprocessing import extract_features, scale
        test_df = load_and_clean(list(CFG.TEST_FILES), CFG.DATA_DIR, verbose=False)
        X_df, y_str, _ = extract_features(test_df)
        X_test = scale(X_df, scaler)
        y_test = le.transform(y_str)

    # Run simulation
    stats = run_simulation(
        X_test, y_test,
        scaler, le, rf_model, iso_model,
        feature_names=feature_names,
        n_flows=args.n_flows,
        delay_s=args.delay,
        verbose=True,
        random_state=args.random_state,
    )

    summary = stats.summary()
    print(f"\nFinal summary:")
    for k, v in summary.items():
        print(f"  {k:<25}: {v}")


if __name__ == "__main__":
    main()
