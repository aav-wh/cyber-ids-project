"""
ids.features.preprocessing
--------------------------
Feature engineering, scaling, label encoding, and class balancing.

Extracted from notebooks/02_preprocessing.ipynb.
The fitted artefacts (scaler, label_encoder, feature_names) are saved to
data/processed/ and loaded by the API at inference time.
"""

import os

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Default output paths (relative to project root)
_DEFAULT_PROCESSED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data", "processed",
)


def extract_features(
    df: pd.DataFrame,
    label_col: str = "BinaryLabel",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Split a dataframe into features (X), binary labels (y), and feature names.

    Drops 'Label' and 'BinaryLabel' columns; everything else is a feature.

    Parameters
    ----------
    df        : cleaned dataframe with 'Label' and 'BinaryLabel' columns
    label_col : name of the binary target column

    Returns
    -------
    X            : pd.DataFrame of numeric features
    y            : pd.Series of string labels ('BENIGN' / 'ATTACK')
    feature_names: list of column names in the same order as X columns
    """
    drop_cols = [c for c in ["Label", "BinaryLabel"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[label_col]
    return X, y, list(X.columns)


def fit_and_save_artefacts(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
    processed_dir: str = _DEFAULT_PROCESSED,
    verbose: bool = True,
) -> tuple[StandardScaler, LabelEncoder]:
    """
    Fit StandardScaler and LabelEncoder on training data, then save all
    artefacts to processed_dir.

    Artefacts saved
    ---------------
    scaler.pkl        — fitted StandardScaler
    label_encoder.pkl — fitted LabelEncoder
    feature_names.pkl — ordered list of feature column names

    IMPORTANT: Fitting must happen on training data ONLY. Test data is
    transformed using the saved scaler — never re-fitted.

    Parameters
    ----------
    X_train       : training features (pd.DataFrame, shape [n_train, n_feat])
    y_train       : training labels (pd.Series of strings)
    feature_names : ordered list of feature column names
    processed_dir : directory to write .pkl artefacts

    Returns
    -------
    scaler, label_encoder
    """
    os.makedirs(processed_dir, exist_ok=True)

    # Fit LabelEncoder on training labels only
    le = LabelEncoder()
    le.fit(y_train)
    if verbose:
        encoding = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"Label encoding: {encoding}")

    # Fit StandardScaler on training features only
    scaler = StandardScaler()
    scaler.fit(X_train)
    if verbose:
        print(f"Scaler fitted on {X_train.shape[0]:,} training rows, {X_train.shape[1]} features.")

    # Persist all three artefacts
    joblib.dump(scaler,        os.path.join(processed_dir, "scaler.pkl"))
    joblib.dump(le,            os.path.join(processed_dir, "label_encoder.pkl"))
    joblib.dump(feature_names, os.path.join(processed_dir, "feature_names.pkl"))

    if verbose:
        print(f"Artefacts saved to: {processed_dir}")

    return scaler, le


def load_artefacts(
    processed_dir: str = _DEFAULT_PROCESSED,
) -> tuple[StandardScaler, LabelEncoder, list[str]]:
    """
    Load previously fitted artefacts from processed_dir.

    Raises FileNotFoundError with a clear message if any file is missing.

    Returns
    -------
    scaler, label_encoder, feature_names
    """
    required = {
        "scaler":        os.path.join(processed_dir, "scaler.pkl"),
        "label_encoder": os.path.join(processed_dir, "label_encoder.pkl"),
        "feature_names": os.path.join(processed_dir, "feature_names.pkl"),
    }
    for label, path in required.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing artefact '{label}' at '{path}'. "
                "Run notebook 02_preprocessing to generate all required files."
            )

    scaler        = joblib.load(required["scaler"])
    le            = joblib.load(required["label_encoder"])
    feature_names = joblib.load(required["feature_names"])
    return scaler, le, feature_names


def scale(
    X: pd.DataFrame | np.ndarray,
    scaler: StandardScaler,
    dtype: type = np.float32,
) -> np.ndarray:
    """
    Apply a fitted scaler. Always use transform(), never fit_transform() at
    inference — the scaler must not be re-fitted on test/live data.

    Parameters
    ----------
    X      : feature matrix (DataFrame or ndarray)
    scaler : fitted StandardScaler
    dtype  : output dtype (float32 halves memory vs float64)

    Returns
    -------
    np.ndarray of scaled values
    """
    return scaler.transform(X).astype(dtype)


def oversample(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "auto",
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply RandomOverSampler to balance the training set.

    Note: For large datasets this duplicates rows in memory (~1 GB extra).
    Using class_weight='balanced' in RandomForest is mathematically equivalent
    without the memory cost — see notebook 03 for discussion.

    Parameters
    ----------
    X, y         : training features and encoded labels
    strategy     : sampling_strategy passed to RandomOverSampler
    random_state : reproducibility seed

    Returns
    -------
    X_resampled, y_resampled
    """
    ros = RandomOverSampler(sampling_strategy=strategy, random_state=random_state)
    X_res, y_res = ros.fit_resample(X, y)

    if verbose:
        unique, counts = np.unique(y_res, return_counts=True)
        print(f"After oversampling: {dict(zip(unique, counts))}")

    return X_res, y_res
