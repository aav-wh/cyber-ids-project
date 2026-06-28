"""
Unit tests for ids.data.loader
"""

import numpy as np
import pandas as pd
import pytest

from ids.data.loader import add_binary_label, clean_dataframe


def make_df(with_inf=False, with_nan=False):
    df = pd.DataFrame({
        "Feature1": [1.0, 2.0, 3.0],
        "Feature2": [0.5, 0.8, 0.2],
        "Label":    ["BENIGN", "DDoS", "BENIGN"],
    })
    if with_inf:
        df.loc[0, "Feature1"] = float("inf")
    if with_nan:
        df.loc[1, "Feature2"] = float("nan")
    return df


class TestCleanDataframe:
    def test_removes_inf(self):
        df = make_df(with_inf=True)
        cleaned = clean_dataframe(df.copy(), verbose=False)
        assert len(cleaned) == 2  # one row dropped

    def test_removes_nan(self):
        df = make_df(with_nan=True)
        cleaned = clean_dataframe(df.copy(), verbose=False)
        assert len(cleaned) == 2

    def test_clean_data_unchanged(self):
        df = make_df()
        cleaned = clean_dataframe(df.copy(), verbose=False)
        assert len(cleaned) == 3

    def test_returns_dataframe(self):
        df = make_df()
        assert isinstance(clean_dataframe(df, verbose=False), pd.DataFrame)


class TestAddBinaryLabel:
    def test_benign_stays_benign(self):
        df = make_df()
        df = add_binary_label(df)
        assert df.loc[df["Label"] == "BENIGN", "BinaryLabel"].iloc[0] == "BENIGN"

    def test_attack_mapped(self):
        df = make_df()
        df = add_binary_label(df)
        assert df.loc[df["Label"] == "DDoS", "BinaryLabel"].iloc[0] == "ATTACK"

    def test_column_added(self):
        df = make_df()
        df = add_binary_label(df)
        assert "BinaryLabel" in df.columns

    def test_only_two_unique_values(self):
        df = pd.DataFrame({
            "Label": ["BENIGN", "DDoS", "PortScan", "Heartbleed", "BENIGN"]
        })
        df = add_binary_label(df)
        assert set(df["BinaryLabel"].unique()) == {"BENIGN", "ATTACK"}
