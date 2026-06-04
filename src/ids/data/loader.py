"""
ids.data.loader
---------------
Raw CSV loading and cleaning for the CICIDS2017 dataset.

Extracted from notebooks/01_data_loading.ipynb and notebooks/02_preprocessing.ipynb.
Notebooks import these functions instead of re-implementing them inline.
"""

import os

import numpy as np
import pandas as pd

# ── Default file splits used throughout the project ──────────────────────────

TRAIN_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

TEST_FILES = [
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


def load_csv_files(file_list: list[str], data_dir: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load and concatenate multiple CICIDS2017 CSV files.

    Strips leading/trailing whitespace from column names (a quirk of the
    dataset) so downstream code can reference columns by their clean names.

    Parameters
    ----------
    file_list : list of filename strings (not full paths)
    data_dir  : directory that contains the CSV files
    verbose   : print per-file row counts when True

    Returns
    -------
    pd.DataFrame — concatenated, column-stripped dataframe
    """
    dfs = []
    for fname in file_list:
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if verbose:
            print(f"  Loaded: {fname}  ({df.shape[0]:,} rows)")
        dfs.append(df)
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    if verbose:
        print(f"  -> Combined shape: {combined.shape}")
    return combined


def clean_dataframe(df: pd.DataFrame, name: str = "DataFrame", verbose: bool = True) -> pd.DataFrame:
    """
    Replace ±inf with NaN then drop all rows with any NaN.

    CICIDS2017 contains a small percentage of rows with infinite flow
    statistics (e.g. divide-by-zero in duration calculations). These rows
    are safe to drop — they are <0.1 % of the dataset.

    Parameters
    ----------
    df      : input dataframe (modified in-place)
    name    : label used in log output
    verbose : print before/after counts when True

    Returns
    -------
    Cleaned pd.DataFrame (same object, returned for chaining)
    """
    n_before = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    n_dropped = n_before - len(df)

    if verbose:
        pct = n_dropped / n_before * 100 if n_before > 0 else 0
        print(f"{name}: {n_before:,} rows -> dropped {n_dropped:,} ({pct:.3f}%) -> {len(df):,} rows")

    return df


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'BinaryLabel' column: 'BENIGN' or 'ATTACK'.

    The original CICIDS2017 'Label' column has 15 distinct attack types.
    For binary classification we collapse all attacks into one class.

    Parameters
    ----------
    df : dataframe that must contain a 'Label' column

    Returns
    -------
    Same dataframe with 'BinaryLabel' column added
    """
    df["BinaryLabel"] = df["Label"].apply(
        lambda x: "BENIGN" if x == "BENIGN" else "ATTACK"
    )
    return df


def load_and_clean(
    file_list: list[str],
    data_dir: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convenience wrapper: load → clean → add binary label.

    Parameters
    ----------
    file_list : list of filenames
    data_dir  : directory containing the CSVs
    verbose   : print progress

    Returns
    -------
    Cleaned pd.DataFrame with 'BinaryLabel' column
    """
    df = load_csv_files(file_list, data_dir, verbose=verbose)
    df = clean_dataframe(df, verbose=verbose)
    df = add_binary_label(df)
    return df
