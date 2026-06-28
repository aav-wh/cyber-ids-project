"""
COM668 Final Year Project — AI-Based Intrusion Detection System
Evaluation Pipeline Script
Author : Abdulbosit Abdurazzakov | B00979380

Loads trained models and artefacts, evaluates on the test set,
and writes a full CSV metrics report to results/.

Usage
-----
    python src/evaluate_pipeline.py
    python src/evaluate_pipeline.py --results-dir /path/to/results
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
import pandas as pd

from ids.config import CFG
from ids.models.predict import load_models
from ids.evaluation.metrics import compute_full_metrics
from ids.models.ensemble import ensemble_or


def parse_args():
    p = argparse.ArgumentParser(description="AI-IDS Evaluation Pipeline")
    p.add_argument("--processed-dir", default=CFG.PROCESSED_DIR)
    p.add_argument("--models-dir",    default=CFG.MODELS_DIR)
    p.add_argument("--results-dir",   default=CFG.RESULTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("=" * 55)
    print("  AI-IDS Evaluation Pipeline")
    print("=" * 55)

    # Load artefacts
    print("\n[1/3] Loading artefacts...")
    scaler, le, feature_names, rf_model, iso_model = load_models(
        args.processed_dir, args.models_dir
    )
    print(f"  Features   : {len(feature_names)}")
    print(f"  RF trees   : {rf_model.n_estimators}")
    print(f"  IF contam  : {iso_model.contamination}")

    # Load test data
    print("\n[2/3] Loading test data...")
    X_test_path = os.path.join(args.processed_dir, "X_test.npy")
    y_test_path = os.path.join(args.processed_dir, "y_test.npy")

    if not os.path.exists(X_test_path):
        print("  X_test.npy not found — re-loading and scaling from CSVs...")
        from ids.data.loader import load_and_clean
        from ids.features.preprocessing import extract_features, scale

        test_df = load_and_clean(list(CFG.TEST_FILES), CFG.DATA_DIR, verbose=True)
        X_df, y_str, _ = extract_features(test_df)
        X_test = scale(X_df, scaler)
        y_test = le.transform(y_str)
        np.save(X_test_path, X_test)
        np.save(y_test_path, y_test)
    else:
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")

    # Evaluate
    print("\n[3/3] Evaluating models...")
    rf_preds  = rf_model.predict(X_test)
    rf_proba  = rf_model.predict_proba(X_test)[:, 0]
    iso_raw   = iso_model.predict(X_test)
    iso_preds = np.where(iso_raw == 1, 1, 0)

    attack_idx = list(le.classes_).index("ATTACK")
    benign_idx = list(le.classes_).index("BENIGN")
    ens_preds  = ensemble_or(rf_preds, iso_preds, attack_label=attack_idx)

    results = []
    for name, preds, proba in [
        ("RandomForest",     rf_preds,  rf_proba),
        ("IsolationForest",  iso_preds, None),
        ("Ensemble (OR)",    ens_preds, rf_proba),
    ]:
        m = compute_full_metrics(
            y_test, preds, y_proba=proba,
            label_names=list(le.classes_),
        )
        results.append({
            "model":     name,
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1":        m["f1"],
            "auc":       m.get("auc"),
            "fpr":       m["fpr"],
            "fnr":       m["fnr"],
            "mcc":       m["mcc"],
        })
        print(f"  {name:<25} F1={m['f1']:.4f}  Recall={m['recall']:.4f}  "
              f"FPR={m['fpr']:.4f}")

    # Save CSV
    save_path = os.path.join(args.results_dir, "evaluation_results.csv")
    pd.DataFrame(results).to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
