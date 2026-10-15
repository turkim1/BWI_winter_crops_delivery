"""
it_harmonic.py — Harmonic fit and fixed-grid resampling for S1 and S2 series.

For each (parcel, year, channel):
    y(t) = a0 + Σ_k [ a_k·cos(2πkωt) + b_k·sin(2πkωt) ]
    ω = 1 / season_length_days

Evaluated on a regular grid every IT_HARMONIC_STEP_DAYS days over the
full Sep 1 → Jun 30 season --> tensor shape (N, C, T_grid), no padding,
no NaN from missing obs.

NaN observations (e.g. cloudy S2 pixels) are simply excluded from the
lstsq fit — no special handling needed.

"""
from __future__ import annotations
import os
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from config import IT_HARMONIC_STEP_DAYS, IT_HARMONIC_N_PER_SEASON

_SEP = (9, 1)
_JUN = (6, 30)

N_JOBS = max(1, os.cpu_count() - 2)

# ─────────────────────────────────────────────────────────────
# SEASON GRID
# ─────────────────────────────────────────────────────────────

def season_grid(agri_year: int, step: int = IT_HARMONIC_STEP_DAYS) -> np.ndarray:
    """Day-offsets from Sep-01(year-1) at *step*-day intervals through Jun-30."""
    t0  = pd.Timestamp(agri_year - 1, *_SEP)
    t1  = pd.Timestamp(agri_year, *_JUN)
    end = (t1 - t0).days
    return np.arange(0, end + 1, step, dtype=float)


def grid_size(step: int = IT_HARMONIC_STEP_DAYS) -> int:
    """Fixed T (representative year 2020)."""
    return len(season_grid(2020, step))


# ─────────────────────────────────────────────────────────────
# HARMONIC FIT
# ─────────────────────────────────────────────────────────────

def _design_matrix(t: np.ndarray, omega: float, n_harmonics: int) -> np.ndarray:
    cols = [np.ones_like(t)]
    for k in range(1, n_harmonics + 1):
        cols += [np.cos(2 * np.pi * k * omega * t),
                 np.sin(2 * np.pi * k * omega * t)]
    return np.column_stack(cols)


def fit_harmonic(xm: np.ndarray, ym: np.ndarray,
                 n_harmonics: int,
                 season_length: float) -> "callable | None":
    mask = np.isfinite(xm) & np.isfinite(ym)
    xm, ym = xm[mask], ym[mask]
    if len(xm) < 2 * n_harmonics + 2:
        return None
    try:
        omega = 1.0 / season_length
        D     = _design_matrix(xm, omega, n_harmonics)
        theta, *_ = np.linalg.lstsq(D, ym, rcond=None)
        return lambda t: _design_matrix(np.asarray(t), omega, n_harmonics) @ theta
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# CUT-OFF OFFSET
# ─────────────────────────────────────────────────────────────

def cutoff_offset(agri_year: int, last_month: str) -> int:
    """Days from Sep-01 to the last day of `last_month` in season `agri_year`."""
    mm = int(last_month[1:])
    yr = agri_year - 1 if mm >= 9 else agri_year
    end = pd.Timestamp(yr, mm, 1) + pd.offsets.MonthEnd(0)
    return (end - pd.Timestamp(agri_year - 1, *_SEP)).days

# ─────────────────────────────────────────────────────────────
# SINGLE SERIES RESAMPLING
# ─────────────────────────────────────────────────────────────

def resample_series(dates, values, agri_year, step, n_harmonics_per_season, n_grid=None):
    t0            = pd.Timestamp(agri_year - 1, *_SEP)
    t1            = pd.Timestamp(agri_year, *_JUN)
    season_length = float((t1 - t0).days)
    grid          = season_grid(agri_year, step) if n_grid is None else np.arange(n_grid) * float(step)
    T, C          = len(grid), values.shape[1]
    out           = np.full((C, T), np.nan, dtype=np.float32)

    xm = np.array([(d - t0).days for d in dates], dtype=float)

    for c in range(C):
        fn = fit_harmonic(xm, values[:, c].astype(float),
                          n_harmonics_per_season, season_length)
        if fn is not None:
            out[c] = fn(grid).astype(np.float32)
    return out



# ─────────────────────────────────────────────────────────────
# BATCH TENSOR BUILDER
# ─────────────────────────────────────────────────────────────
def _build_one(key, grp, feat_cols, date_col, step, n_harmonics_per_season,
               last_month=None, n_grid=None):
    agri_year, id_parcel = key
    grp = grp.sort_values(date_col)
    lab = grp["label"].iloc[0]
    label = int(lab) if pd.notna(lab) else -1          # -1 = unlabeled (predict year)
    if last_month is not None:                          # drop observations after the cutoff
        t0 = pd.Timestamp(agri_year - 1, *_SEP)
        days = (pd.DatetimeIndex(grp[date_col].values) - t0).days
        grp = grp[days <= cutoff_offset(agri_year, last_month)]
    dates = pd.DatetimeIndex(grp[date_col].values)
    vals  = grp[feat_cols].values.astype(np.float32)
    n_obs = int(np.isfinite(vals).any(axis=1).sum()) if len(vals) else 0
    resampled = resample_series(dates, vals, agri_year, step, n_harmonics_per_season, n_grid)
    return resampled, label, id_parcel, agri_year, n_obs


def build_harmonic_tensor(long, feat_cols, date_col="date",
                           step=IT_HARMONIC_STEP_DAYS,
                           n_harmonics_per_season=IT_HARMONIC_N_PER_SEASON,
                           last_month=None):
    if last_month is None:
        T_grid = grid_size(step)
    else:
        T_grid = int(cutoff_offset(int(long["agri_year"].max()), last_month) // step) + 1
    C = len(feat_cols)
    groups = list(long.groupby(["agri_year", "id_parcel"]))
    N = len(groups)
    print(f"\n  [harmonic] {N:,} series × {C} channels --> T={T_grid} "
          f"(step={step}d, harmonics/season={n_harmonics_per_season}, "
          f"season=Sep-{last_month or 'Jun'})")

    results = Parallel(n_jobs=N_JOBS, verbose=3)(
        delayed(_build_one)(key, grp, feat_cols, date_col, step,
                            n_harmonics_per_season, last_month, T_grid)
        for key, grp in groups
    )

    X   = np.stack([r[0] for r in results]).astype(np.float32)
    y   = np.array([r[1] for r in results], dtype=np.int64)
    ids = pd.DataFrame({
        "id_parcel": [r[2] for r in results],
        "agri_year": [r[3] for r in results],
        "label":     y,
        "n_obs":     [r[4] for r in results],
    })

    n_fail = int(np.all(np.isnan(X), axis=(1, 2)).sum())
    print(f"  [harmonic] All-NaN series : {n_fail:,}/{N:,} ({100 * n_fail / max(N, 1):.1f}%)")
    nan_channels = np.isnan(X).any(axis=2)
    print(f"  Parcels with ≥1 NaN channel : {nan_channels.any(axis=1).sum():,} / {N:,}")
    print("  NaN rate per channel:")
    for j, col in enumerate(feat_cols):
        print(f"    {col:20s}: {nan_channels[:, j].mean() * 100:.2f}%")
    return X, y, ids