"""
ml_preproc.py — Per-file aggregation (S1 / S2) into monthly statistics,
                 wide-format stacking, and leakage-column removal.
                 ML wide-feature-matrix pipeline ONLY.

Shared low-level primitives (class assignment, angle normalization,
spike/dip filters) live in common_preproc.py.

Two-step aggregation strategy
──────────────────────────────
  Step 1 : median within each (parcel, date) (deduplication)
  Step 2 : AGGREGATION_FUNCS of those daily medians --> monthly stats.
"""
from __future__ import annotations
import re
import pandas as pd
import numpy as np

from config import MONTH_ORDER, AGG_FUNCS

def aggregate_monthly(long: pd.DataFrame, sensor_prefix: str, feature_cols: list[str],
                       orbit_tag: str | None = None) -> pd.DataFrame:
    key = ["id_parcel", "agri_year", "month_tag"]
    grp = long.groupby(key)[feature_cols].agg(AGG_FUNCS)
    if orbit_tag is None:
        grp.columns = [f"{sensor_prefix}_{col}_{agg}" for col, agg in grp.columns]
    else:
        grp.columns = [f"{sensor_prefix}_{col}_{orbit_tag}_{agg}" for col, agg in grp.columns]
    grp = grp.reset_index()

    meta = (long.groupby(["id_parcel", "agri_year"])[["label", "code_cultu", "code_group"]]
                .first().reset_index())
    grp = grp.merge(meta, on=["id_parcel", "agri_year"], how="left")
    return grp

# -------------------------------------------------------------
# WIDE-FORMAT STACKING
# -------------------------------------------------------------

def stack_to_wide(
    monthly_frames: list[pd.DataFrame],
    id_cols: list[str],
    label_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Pivot long monthly data to wide format.
    Columns become  <feature>_<month_tag>  (e.g. s2_NDVI_mean_m09).
    """
    label_cols = label_cols or []
    long = pd.concat(monthly_frames, ignore_index=True)

    labels = None
    if label_cols:
        labels = long.groupby(id_cols)[label_cols].first().reset_index()

    feat_cols = [
        c for c in long.columns
        if c not in id_cols + ["month_tag"] + label_cols
        and pd.api.types.is_numeric_dtype(long[c])
    ]

    wide = long.drop(columns=label_cols, errors="ignore").pivot_table(
        index=id_cols, columns="month_tag", values=feat_cols, aggfunc="median"
    )
    wide.columns = [f"{feat}_{month}" for feat, month in wide.columns]
    wide = wide.reset_index()

    if labels is not None:
        wide = wide.merge(labels, on=id_cols, how="left")
    return wide

# -------------------------------------------------------------
# LEAKAGE COLUMN REMOVAL
# -------------------------------------------------------------

def drop_leakage_columns(features: pd.DataFrame) -> pd.DataFrame:
    leakage_cols = ["code_cultu", "code_group"]
    dropped = [c for c in leakage_cols if c in features.columns]
    if dropped:
        print(f"  [leakage] Dropping columns: {dropped}")
    return features.drop(columns=dropped, errors="ignore")


# -------------------------------------------------------------
# TEMPORAL INTERPOLATION (replaces zero-fill + missing flags)
# -------------------------------------------------------------

def _interp_rows(a: np.ndarray) -> np.ndarray:
    """Row-wise linear interpolation, edges filled with nearest valid value."""
    n, m = a.shape
    idx   = np.arange(m)
    valid = ~np.isnan(a)
    last  = np.maximum.accumulate(np.where(valid, idx, -1), axis=1)
    nxt   = np.minimum.accumulate(np.where(valid, idx, m)[:, ::-1], axis=1)[:, ::-1]
    rows  = np.arange(n)[:, None]
    lo, hi = np.clip(last, 0, m - 1), np.clip(nxt, 0, m - 1)
    vlo, vhi = a[rows, lo], a[rows, hi]
    denom = hi - lo
    frac  = np.where(denom > 0, (idx - lo) / np.where(denom > 0, denom, 1), 0.0)
    out   = vlo + (vhi - vlo) * frac
    out   = np.where(last == -1, vhi, out)   # leading NaNs
    out   = np.where(nxt == m, vlo, out)     # trailing NaNs
    return out

def interpolate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (sensor, index) pair, linearly interpolate NaNs across
    months using the wide-format row structure.

    Operates entirely per-row (per parcel×year), so there is no
    cross-parcel or train/test leakage.


    Example columns handled: s2_B4_median_m04, s2_B4_median_m05 …
    """

    df = df.copy()
    pattern = re.compile(r"^(s[12]_.+)_(m\d{2})$")  # keep month tag as string

    base_groups: dict[str, list[tuple[int, str]]] = {}
    for col in df.columns:
        m = pattern.match(col)
        if m:
            base, month_tag = m.group(1), m.group(2)
            # rank by position in the agricultural calendar, not by integer value
            rank = MONTH_ORDER.index(month_tag) if month_tag in MONTH_ORDER else 99
            base_groups.setdefault(base, []).append((rank, col))

    for base, month_cols in base_groups.items():
        sorted_cols = [col for _, col in sorted(month_cols)]
        df[sorted_cols] = _interp_rows(df[sorted_cols].to_numpy(dtype=float))
        

    return df

_SENSOR_COL_RE = re.compile(r"^(s[12])_.+_(m\d{2})$")


def _month_of(col: str) -> str:
    return col.rsplit("_", 1)[1]

def drop_unavailable_months(features: pd.DataFrame, predict_year: int) -> pd.DataFrame:
    """Drop every sensor column that is entirely NaN in the predict year (for all years)."""
    pred = features[features["agri_year"] == predict_year]
    empty = {c for c in features.columns if _SENSOR_COL_RE.match(c) and pred[c].isna().all()}

    for p in ("s1", "s2"):
        cols = [c for c in features.columns
                if (m := _SENSOR_COL_RE.match(c)) and m.group(1) == p]
        if not cols:
            continue
        kept    = sorted({_month_of(c) for c in cols if c not in empty}, key=MONTH_ORDER.index)
        dropped = sorted({_month_of(c) for c in cols if c in empty},     key=MONTH_ORDER.index)
        print(f"  [window] {p.upper()} {predict_year}: using {kept if kept else 'NO months'}"
              + (f" | dropped (no data in {predict_year}): {dropped}" if dropped else ""))
    return features.drop(columns=list(empty))


def count_valid_months(features: pd.DataFrame, cols: list[str], prefix: str) -> pd.Series:
    """Per parcel: number of months with at least one real value among `cols` of sensor
    `prefix` ('s1'/'s2'). Call BEFORE interpolation."""
    by_month: dict[str, list[str]] = {}
    for c in cols:
        m = _SENSOR_COL_RE.match(c)
        if m and m.group(1) == prefix:
            by_month.setdefault(m.group(2), []).append(c)
    n = pd.Series(0, index=features.index)
    for mcols in by_month.values():
        n += features[mcols].notna().any(axis=1).astype(int)
    return n