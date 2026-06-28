"""
ids.data.pipeline
-----------------
End-to-end data pipeline: load → clean → label → split → scale → save.

This module orchestrates the full preprocessing workflow so it can be
invoked from a single function call in scripts or CI pipelines.
"""

from __future__ import annotations

import os
import numpy as np

from ids.config import CFG
from ids.data.loader import load_and_clean, TRAIN_FILES, TEST_FILES
from ids.data.splitter import temporal_split
from ids.features.preprocessing import (
    extract_features,
    fit_and_save_artefacts,
    scale,
)


def run_data_pipeline(
    data_dir: str | None = None,
    processed_dir: str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Execute the full data pipeline and return all processed arrays.

    Steps
    -----
    1. Load and clean train CSVs (Mon–Wed)
    2. Load and clean test CSVs (Thu–Fri)
    3. Extract X/y from each split
    4. Fit and save scaler + label encoder on train data only
    5. Scale both splits
    6. Save y arrays to processed_dir

    Parameters
    ----------
    data_dir      : directory containing raw CSVs (default: CFG.DATA_DIR)
    processed_dir : output directory for artefacts (default: CFG.PROCESSED_DIR)
    verbose       : print progress

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test,
                    scaler, le, feature_names
    """
    data_dir      = data_dir      or CFG.DATA_DIR
    processed_dir = processed_dir or CFG.PROCESSED_DIR

    if verbose:
        print("=" * 55)
        print("AI-IDS Data Pipeline")
        print("=" * 55)

    # Step 1 & 2 — Load data
    if verbose:
        print("\n[1/5] Loading training data (Mon–Wed)...")
    train_df = load_and_clean(list(CFG.TRAIN_FILES), data_dir, verbose=verbose)

    if verbose:
        print("\n[2/5] Loading test data (Thu–Fri)...")
    test_df = load_and_clean(list(CFG.TEST_FILES), data_dir, verbose=verbose)

    # Step 3 — Extract features
    if verbose:
        print("\n[3/5] Extracting features...")
    X_train_df, y_train, feature_names = extract_features(train_df)
    X_test_df,  y_test,  _             = extract_features(test_df)

    # Step 4 — Fit artefacts on TRAIN only
    if verbose:
        print("\n[4/5] Fitting and saving artefacts (train only)...")
    scaler, le = fit_and_save_artefacts(
        X_train_df, y_train, feature_names, processed_dir, verbose=verbose
    )

    # Step 5 — Scale
    if verbose:
        print("\n[5/5] Scaling feature matrices...")
    X_train = scale(X_train_df, scaler)
    X_test  = scale(X_test_df,  scaler)

    # Save y arrays
    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)
    np.save(os.path.join(processed_dir, "y_train_resampled.npy"), y_train_enc)
    np.save(os.path.join(processed_dir, "y_test.npy"),            y_test_enc)

    if verbose:
        print(f"\nPipeline complete.")
        print(f"  X_train : {X_train.shape},  y_train : {y_train_enc.shape}")
        print(f"  X_test  : {X_test.shape},   y_test  : {y_test_enc.shape}")

    return {
        "X_train":      X_train,
        "X_test":       X_test,
        "y_train":      y_train_enc,
        "y_test":       y_test_enc,
        "scaler":       scaler,
        "le":           le,
        "feature_names": feature_names,
    }
