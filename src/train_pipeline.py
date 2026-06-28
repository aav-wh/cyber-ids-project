"""
COM668 Final Year Project — AI-Based Intrusion Detection System
Full Training Pipeline Script
Author : Abdulbosit Abdurazzakov | B00979380

Runs the complete ML pipeline end-to-end:
  1. Load and clean CICIDS2017 CSVs
  2. Preprocess and save artefacts
  3. Train RandomForest and IsolationForest
  4. Evaluate on the test set
  5. Save models and print a results summary

Usage
-----
    python src/train_pipeline.py
    python src/train_pipeline.py --data-dir /path/to/data --no-verbose
"""

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC  = os.path.join(_ROOT, "src")
for p in [_ROOT, _SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from ids.config import CFG
from ids.data.loader import load_and_clean
from ids.features.preprocessing import extract_features, fit_and_save_artefacts, scale
from ids.models.train import train_random_forest, train_isolation_forest
from ids.models.evaluate import compute_metrics
from ids.evaluation.metrics import compute_full_metrics


def parse_args():
    p = argparse.ArgumentParser(description="AI-IDS Training Pipeline")
    p.add_argument("--data-dir",      default=CFG.DATA_DIR,      help="Raw CSV directory")
    p.add_argument("--processed-dir", default=CFG.PROCESSED_DIR, help="Artefact output directory")
    p.add_argument("--models-dir",    default=CFG.MODELS_DIR,    help="Model output directory")
    p.add_argument("--n-estimators",  type=int,   default=CFG.RF_N_ESTIMATORS)
    p.add_argument("--contamination", type=float, default=CFG.IF_CONTAMINATION)
    p.add_argument("--no-verbose",    action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    verbose = not args.no_verbose
    t_start = time.time()

    print("=" * 60)
    print("  AI-IDS Full Training Pipeline")
    print("  COM668 | Abdulbosit Abdurazzakov | B00979380")
    print("=" * 60)

    # ── 1. Load data ───────────────────────────────────────────────────────
    print("\n[1/5] Loading training data...")
    train_df = load_and_clean(list(CFG.TRAIN_FILES), args.data_dir, verbose=verbose)

    print("\n[2/5] Loading test data...")
    test_df  = load_and_clean(list(CFG.TEST_FILES),  args.data_dir, verbose=verbose)

    # ── 2. Preprocess ──────────────────────────────────────────────────────
    print("\n[3/5] Preprocessing...")
    X_train_df, y_train_str, feature_names = extract_features(train_df)
    X_test_df,  y_test_str,  _             = extract_features(test_df)

    scaler, le = fit_and_save_artefacts(
        X_train_df, y_train_str, feature_names,
        args.processed_dir, verbose=verbose,
    )

    X_train = scale(X_train_df, scaler)
    X_test  = scale(X_test_df,  scaler)
    y_train = le.transform(y_train_str)
    y_test  = le.transform(y_test_str)

    # Save y arrays
    np.save(os.path.join(args.processed_dir, "y_train_resampled.npy"), y_train)
    np.save(os.path.join(args.processed_dir, "y_test.npy"),            y_test)

    # Benign-only subset for Isolation Forest
    X_train_benign = X_train[y_train == le.transform(["BENIGN"])[0]]

    # ── 3. Train models ────────────────────────────────────────────────────
    print("\n[4/5] Training models...")
    rf_model  = train_random_forest(
        X_train, y_train,
        n_estimators=args.n_estimators,
        save_path=os.path.join(args.models_dir, "random_forest.pkl"),
        verbose=verbose,
    )
    iso_model = train_isolation_forest(
        X_train_benign,
        contamination=args.contamination,
        save_path=os.path.join(args.models_dir, "isolation_forest.pkl"),
        verbose=verbose,
    )

    # ── 4. Evaluate ────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating...")
    rf_preds  = rf_model.predict(X_test)
    rf_proba  = rf_model.predict_proba(X_test)[:, 0]
    rf_metrics = compute_full_metrics(
        y_test, rf_preds, y_proba=rf_proba,
        label_names=list(le.classes_),
    )

    iso_raw   = iso_model.predict(X_test)
    iso_preds = np.where(iso_raw == 1, le.transform(["BENIGN"])[0], le.transform(["ATTACK"])[0])
    iso_metrics = compute_full_metrics(
        y_test, iso_preds,
        label_names=list(le.classes_),
    )

    elapsed = time.time() - t_start

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Complete — Results Summary")
    print("=" * 60)
    print(f"\n  Random Forest:")
    print(f"    Precision : {rf_metrics['precision']:.4f}")
    print(f"    Recall    : {rf_metrics['recall']:.4f}")
    print(f"    F1        : {rf_metrics['f1']:.4f}")
    print(f"    AUC       : {rf_metrics.get('auc', 'N/A')}")
    print(f"\n  Isolation Forest:")
    print(f"    Precision : {iso_metrics['precision']:.4f}")
    print(f"    Recall    : {iso_metrics['recall']:.4f}")
    print(f"    F1        : {iso_metrics['f1']:.4f}")
    print(f"\n  Total pipeline time: {elapsed:.1f}s")
    print(f"  Models saved to   : {args.models_dir}")
    print(f"  Artefacts saved to: {args.processed_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
