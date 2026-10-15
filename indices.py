"""
indices.py — Compute spectral indices from raw Sentinel-2 bands.

Supported indices
-----------------
  NDVI  = (B8  - B4)  / (B8  + B4)      Normalized Difference Vegetation Index
  NDMI  = (B8  - B11) / (B8  + B11)     Normalized Difference Moisture Index
  NDRE  = (B8A - B5)  / (B8A + B5)      Normalized Difference Red Edge
  NDYI =  (B3  - B2) / (B3  + B2)
  PSRI  = (B4  - B2)  /  B6             Plant Senescence Reflectance Index

All divisions are protected against zero denominators (result --> NaN).

Public API
----------
compute_indices(df)  -> df   (adds index columns, drops raw band columns)
S2_BAND_COLS         : list of the 10 raw band column names
S2_INDEX_COLS        : list of the computed index column names
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Raw band columns present in every S2 CSV
S2_BAND_COLS: list[str] = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Derived index columns produced by compute_indices()
S2_INDEX_COLS: list[str] = ["NDVI", "NDMI", "NDRE", "PSRI", "NDYI"]

# 10m-only bands
S2_BAND_10M_COLS: list[str] = ["B2", "B3", "B4", "B8"]

# Combined: 10m bands + all indices (used by "bands_10m+indices" mode)
S2_BAND_AND_INDEX_COLS: list[str] = S2_BAND_10M_COLS + S2_INDEX_COLS


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise num/den with zero-denominator --> NaN."""
    return num / den.replace(0, np.nan)


def compute_indices(df: pd.DataFrame, keep_10m_bands: bool = False) -> pd.DataFrame:
    """
    Add NDVI, NDMI, NDRE, NDYI, PSRI columns to df and drop all raw band columns.

    Parameters
    ----------
    df : DataFrame containing at least the columns in S2_BAND_COLS.
    keep_10m_bands : bool, optional
        If True, keep the 10m band columns in the returned DataFrame.

    Returns
    -------
    df with index columns added and S2_BAND_COLS removed.
    """

    df = df.copy()

    df["NDVI"] = _safe_ratio(df["B8"]  - df["B4"],  df["B8"]  + df["B4"])
    df["NDMI"] = _safe_ratio(df["B8"]  - df["B11"], df["B8"]  + df["B11"])
    df["NDRE"] = _safe_ratio(df["B8A"] - df["B5"],  df["B8A"] + df["B5"])
    df["NDYI"] = _safe_ratio(df["B3"]  - df['B2'],  df["B3"]  + df['B2'])
    df["PSRI"] = _safe_ratio(df["B4"]  - df["B2"],  df["B6"])
    # df["S2REP"] = 705 + 35 * _safe_ratio(
    #     ((df["B7"] + df["B4"]) / 2) - df["B5"],
    #     df["B6"] - df["B5"]
    # )


    cols_to_drop = S2_BAND_COLS if not keep_10m_bands else [c for c in S2_BAND_COLS if c not in S2_BAND_10M_COLS]
    return df.drop(columns=cols_to_drop)
