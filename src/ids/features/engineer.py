"""
ids.features.engineer
---------------------
Feature engineering helpers for CICIDS2017 network flow data.

These transformations are applied BEFORE scaling and are fitted on
training data only — test/live data uses the same fitted transformers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add packet and byte ratio features.

    These ratios capture the directional asymmetry of traffic flows,
    which is a strong signal for certain attack types (e.g. DDoS produces
    highly asymmetric fwd/bwd ratios).

    Parameters
    ----------
    df : cleaned dataframe with standard CICIDS2017 columns

    Returns
    -------
    df with new ratio columns appended
    """
    df = df.copy()

    # Fwd/Total packet ratio — close to 1.0 for pure upload floods
    tot_pkts = (df.get("Total Fwd Packets", 0) + df.get("Total Backward Packets", 0))
    df["fwd_pkt_ratio"] = np.where(
        tot_pkts > 0,
        df.get("Total Fwd Packets", 0) / (tot_pkts + 1e-9),
        0.5,
    )

    # Byte efficiency — payload bytes / total packets
    tot_bytes = df.get("Total Length of Fwd Packets", 0) + df.get("Total Length of Bwd Packets", 0)
    df["bytes_per_packet"] = np.where(
        tot_pkts > 0,
        tot_bytes / (tot_pkts + 1e-9),
        0.0,
    )

    # Duration × packet rate (volume proxy)
    if "Flow Duration" in df.columns and "Flow Packets/s" in df.columns:
        df["flow_volume_proxy"] = df["Flow Duration"] * df["Flow Packets/s"]

    return df


def log_transform_skewed(
    df: pd.DataFrame,
    skew_threshold: float = 3.0,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply log1p transform to highly skewed numeric features.

    Network flow features are often heavily right-skewed (e.g. byte counts).
    log1p brings the distribution closer to normal, improving tree split
    quality and distance-based model performance.

    Parameters
    ----------
    df             : input dataframe (numeric columns only transformed)
    skew_threshold : features with |skewness| > threshold are transformed

    Returns
    -------
    transformed df, list of transformed column names
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    transformed  = []

    for col in numeric_cols:
        skewness = df[col].skew()
        if abs(skewness) > skew_threshold and df[col].min() >= 0:
            df[col] = np.log1p(df[col])
            transformed.append(col)

    if verbose:
        print(f"[LogTransform] Applied log1p to {len(transformed)} skewed features.")

    return df, transformed


def clip_outliers(
    df: pd.DataFrame,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Clip extreme outliers at specified percentiles.

    Extreme values (e.g. a single 10-second flow in a 1ms dataset) can
    dominate StandardScaler means and distort the scaled feature space.

    Parameters
    ----------
    df                : dataframe to clip
    lower/upper       : percentile thresholds
    exclude_cols      : columns to leave unchanged

    Returns
    -------
    Clipped dataframe
    """
    df = df.copy()
    exclude = set(exclude_cols or [])
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    for col in numeric_cols:
        lo = np.percentile(df[col].dropna(), lower_percentile)
        hi = np.percentile(df[col].dropna(), upper_percentile)
        df[col] = df[col].clip(lo, hi)

    return df


def encode_port_category(series: pd.Series) -> pd.Series:
    """
    Map destination port to a categorical bucket.

    Well-known ports (0–1023) carry different semantics than ephemeral ports.
    This coarse bucketing adds a human-interpretable signal alongside the raw
    port number.

    Returns
    -------
    pd.Series of int categories:
      0 = well-known (0–1023)
      1 = registered (1024–49151)
      2 = dynamic/private (49152–65535)
    """
    return pd.cut(
        series.clip(0, 65535),
        bins=[-1, 1023, 49151, 65535],
        labels=[0, 1, 2],
    ).astype(int)
