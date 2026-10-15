"""
ndvi_postfilter.py — Splits parcels predicted as 'Other vegetation' into
cultivated / non_cultivated using S2 NDVI. Called by predict.py

Rule: max of monthly median NDVI over NDVI_FILTER_START_MONTH..NDVI_FILTER_END_MONTH
(clipped to the months available in the predict year) > NDVI_FILTER_THRESHOLD
--> cultivated, otherwise non_cultivated.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from config import (S2_DIR, MONTH_ORDER, NDVI_FILTER_START_MONTH,
                    NDVI_FILTER_END_MONTH, NDVI_FILTER_THRESHOLD)
from utils import _FILE_RE
from common_preproc import load_long


def build_ndvi_table(year: int) -> pd.DataFrame | None:
    """Monthly-median NDVI per parcel for `year`. None if there are no S2 files."""
    has_s2 = any((m := _FILE_RE.search(p.stem)) and int(m.group(1)) == year
                 for p in S2_DIR.glob("S2_*.csv"))
    if not has_s2:
        print(f"[ndvi_filter] not possible: no S2 data found for {year}.")
        return None
    long_s2 = load_long("S2", [year], unlabeled_years=(year,))
    return (long_s2.groupby(["id_parcel", "agri_year", "month_tag"])["NDVI"].median()
                   .reset_index()
                   .pivot(index=["id_parcel", "agri_year"], columns="month_tag", values="NDVI")
                   .add_prefix("s2_NDVI_median_").reset_index())


def add_veg_subclass(out: pd.DataFrame, ndvi: pd.DataFrame) -> pd.DataFrame:
    """Adds ndvi_max_window, ndvi_max_month, veg_subclass to the prediction table."""
    start, end = NDVI_FILTER_START_MONTH, NDVI_FILTER_END_MONTH
    window = MONTH_ORDER[MONTH_ORDER.index(start): MONTH_ORDER.index(end) + 1]
    avail = [f"s2_NDVI_median_{m}" for m in window
             if f"s2_NDVI_median_{m}" in ndvi.columns and ndvi[f"s2_NDVI_median_{m}"].notna().any()]
    if f"s2_NDVI_median_{start}" not in avail:
        print(f"[ndvi_filter] not possible: no S2 NDVI for {start} in the predict year "
              f"(season too short for window {start}-{end}).")
        return out

    print(f"[ndvi_filter] window {start}-{end} | months used: {[c[-3:] for c in avail]} "
          f"| threshold {NDVI_FILTER_THRESHOLD}")
    out = out.copy()
    vals = out[["id_parcel", "agri_year"]].merge(
        ndvi[["id_parcel", "agri_year"] + avail], on=["id_parcel", "agri_year"], how="left")[avail]
    has_ndvi = vals.notna().any(axis=1)
    out["ndvi_max_window"] = vals.max(axis=1).values
    out["ndvi_max_month"] = vals.fillna(-np.inf).idxmax(axis=1).str[-3:].where(has_ndvi).values

    pc, mx = out["pred_class"], out["ndvi_max_window"]
    sub = pd.Series("unclassified", index=out.index)
    sub[pc == 0] = "winter_cereal"
    sub[pc == 1] = "rapeseed"
    sub[(pc == 2) & mx.isna()] = "unknown"
    sub[(pc == 2) & (mx > NDVI_FILTER_THRESHOLD)] = "cultivated"
    sub[(pc == 2) & (mx <= NDVI_FILTER_THRESHOLD)] = "non_cultivated"
    out["veg_subclass"] = sub
    print(sub.value_counts().to_string())
    return out