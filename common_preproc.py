"""
common_preproc.py — Low-level preprocessing primitives shared by both
pipelines:
  - the classical ML pipeline  (ml_preproc.py     -> ml_dataloader.py)
  - the InceptionTime pipeline (it_dataloader.py)

Everything here operates on a single per-file DataFrame and is agnostic
to whether the caller wants a wide feature matrix or a harmonic tensor.
Changing anything in this file affects BOTH pipelines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import os
from joblib import Parallel, delayed
from config import EXCLUDED_CODE_GROUPS, CLOUD_THRESHOLD, S2_FEATURE_COLS, S2_MODE, \
                    S1_SPIKE_FILTER, S1_SPIKE_THRESHOLD_DB, S2_NDVI_DIP_FILTER, S2_NDVI_DIP_THRESHOLD
from indices import compute_indices
from utils import parse_filename

from config import WINTER_CROPS, CZH_CROPS
from utils import _FILE_RE, to_db

spike_dip_verbose = False  # set False to suppress spike/dip filter printouts

N_JOBS = max(1, os.cpu_count() - 2)

# -------------------------------------------------------------
# CLASS LABEL
# -------------------------------------------------------------

def assign_class(code: str) -> int:
    if code in WINTER_CROPS:
        return 0
    if code in CZH_CROPS:
        return 1
    return 2


# -------------------------------------------------------------
# DATE COLUMN LOOKUP
# -------------------------------------------------------------

def _date_col(df: pd.DataFrame) -> str | None:
    """Return the name of the acquisition-date column, or None."""
    return next((c for c in ["date", "DATE", "acquisition_date"] if c in df.columns), None)


# -------------------------------------------------------------
# S1 INCIDENCE-ANGLE NORMALIZATION
# -------------------------------------------------------------

def compute_ref_angle(s1_files: list[Path], train_years: list[int],
                      sample_files: int = 6) -> float:
    """
    Compute the mean incidence angle from a sample of S1 training files.
    Uses only train_years files to avoid leakage from the test year.
    """
    train_files = [
        p for p in s1_files
        if int(_FILE_RE.search(p.stem).group(1)) in train_years
    ]
    if sample_files:
        idx = np.linspace(0, len(train_files) - 1,
                          min(sample_files, len(train_files)), dtype=int)
        train_files = [train_files[i] for i in idx]

    angles = []
    for p in train_files:
        df = pd.read_csv(p, usecols=["angle"], low_memory=False)
        angles.append(df["angle"].dropna().values)

    ref = float(np.mean(np.concatenate(angles)))
    print(f"  [ref_angle] Computed from training data: {ref:.2f}°")
    return ref


def normalize1(mat: pd.Series, angle: pd.Series, ref_angle: float) -> np.ndarray:
    """
    Normalize backscatter to a reference incidence angle.
        sigma_norm = sigma * cos²(ref) / cos²(angle)
    """
    ref2   = np.cos(ref_angle * np.pi / 180) ** 2
    angle2 = np.cos(angle.values * np.pi / 180) ** 2
    return mat.values * (ref2 / angle2)


# -------------------------------------------------------------
# OUTLIER / DIP FILTERS
# -------------------------------------------------------------

def remove_s1_spikes(df: pd.DataFrame, numeric_cols: list[str],
                      month_tag: str, agri_year: int,
                      threshold_db: float = 4.0,
                      verbose: bool = spike_dip_verbose) -> pd.DataFrame:
    """
    For each parcel, NaN out acquisition t(i) if:
        |dB(t(i)) - dB(t(i-1))| > threshold AND
        |dB(t(i+1)) - dB(t(i))| > threshold
    Operates on linear values: converts to dB for comparison only.
    """
    dc = _date_col(df)
    if dc is None:
        print("  [spike filter] No date column found — skipping.")
        return df

    df = df.copy()
    df[dc] = pd.to_datetime(df[dc])
    df = df.sort_values(["id_parcel", dc])

    n_parcels      = df["id_parcel"].nunique()
    n_acquisitions = df.shape[0]

    for col in numeric_cols:
        if col not in df.columns:
            continue

        db = pd.Series(to_db(df[col].values), index=df.index)
        diff_prev = db.groupby(df["id_parcel"]).diff()
        diff_next = db.groupby(df["id_parcel"]).diff(-1)

        spike_mask = (diff_prev.abs() > threshold_db) & (diff_next.abs() > threshold_db)
        n_spikes   = spike_mask.sum()
        n_parcels_affected = df.loc[spike_mask, "id_parcel"].nunique()

        if verbose:
            print(
                f"  [spike filter] {agri_year} {month_tag} | {col} | "
                f"threshold={threshold_db} dB | "
                f"{n_spikes:,} / {n_acquisitions:,} acquisitions NaNed "
                f"({100*n_spikes/n_acquisitions:.1f}%) "
                f"across {n_parcels_affected:,} / {n_parcels:,} parcels"
            )

        df.loc[spike_mask, col] = np.nan

    return df


def remove_s2_ndvi_dips(df: pd.DataFrame, feature_cols: list[str],
                          month_tag: str, agri_year: int,
                          threshold: float = 0.5,
                          verbose: bool = spike_dip_verbose) -> pd.DataFrame:
    """
    For each parcel, NaN out acquisition t(i) across ALL indices if:
        (NDVI(t(i-1)) - NDVI(t(i))) > threshold  AND
        (NDVI(t(i+1)) - NDVI(t(i))) > threshold
    """
    if "NDVI" not in df.columns:
        print("  [ndvi dip filter] NDVI column not found — skipping.")
        return df

    dc = _date_col(df)
    if dc is None:
        print("  [ndvi dip filter] No date column found — skipping.")
        return df

    df = df.copy()
    df[dc] = pd.to_datetime(df[dc])
    df = df.sort_values(["id_parcel", dc])

    n_acquisitions = df.shape[0]
    n_parcels      = df["id_parcel"].nunique()

    diff_prev = df.groupby("id_parcel")["NDVI"].diff()
    diff_next = df.groupby("id_parcel")["NDVI"].diff(-1)

    dip_mask = (diff_prev < -threshold) & (diff_next < -threshold)

    n_dips             = dip_mask.sum()
    n_parcels_affected = df.loc[dip_mask, "id_parcel"].nunique()

    if verbose:
        print(
            f"  [ndvi dip filter] {agri_year} {month_tag} | "
            f"threshold={threshold} | "
            f"{n_dips:,} / {n_acquisitions:,} acquisitions NaNed "
            f"({100 * n_dips / max(n_acquisitions, 1):.1f}%) "
            f"across {n_parcels_affected:,} / {n_parcels:,} parcels"
        )

    cols_to_nan = [c for c in feature_cols if c in df.columns]
    df.loc[dip_mask, cols_to_nan] = np.nan

    return df

def load_sensor_file(
    path: Path,
    base_sensor: str,                 # "S1" or "S2"
    all_years: list[int],
    ref_angle: float | None = None,
    orbit_filter: str | None = None,  # "ASCENDING" | "DESCENDING" | None
    unlabeled_years=()
) -> pd.DataFrame | None:
    """
    Canonical per-file loader — single source of truth for both ML and IT branches.
    Fixed order: read -> sensor row filter -> EXCLUDED_CODE_GROUPS -> normalize/indices
    -> spike/dip filter -> label -> dedupe to one row per (id_parcel, date).
    Returns long df: id_parcel, date, agri_year, month_tag, <feature cols>, label, code_cultu, code_group
    """
    agri_year, month_tag = parse_filename(path)
    if agri_year not in all_years:
        return None

    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return None

    if base_sensor == "S2":
        df = df[df["MSK_CLDPRB"] <= CLOUD_THRESHOLD]
    if df.empty:
        return None

    if orbit_filter is not None:
        if "orbit_direction" not in df.columns:
            return None
        df = df[df["orbit_direction"].str.upper() == orbit_filter]
    if df.empty:
        return None

    labeled = agri_year not in unlabeled_years and "code_cultu" in df.columns
    if labeled and "code_group" in df.columns:
        df = df[~df["code_group"].isin(EXCLUDED_CODE_GROUPS)]
    if df.empty:
        return None

    if base_sensor == "S1":
        for col in ("VV", "VH"):
            if col in df.columns and "angle" in df.columns:
                df[col] = normalize1(df[col], df["angle"], ref_angle)
        if {"VV", "VH"}.issubset(df.columns):
            df["VV_VH_ratio"] = df["VV"] / df["VH"].replace(0, np.nan)
        numeric_cols = [c for c in ["VV", "VH", "VV_VH_ratio"] if c in df.columns]

        if S1_SPIKE_FILTER:
            df = remove_s1_spikes(df, numeric_cols, month_tag=month_tag,
                                    agri_year=agri_year, threshold_db=S1_SPIKE_THRESHOLD_DB)
        feature_cols = numeric_cols

    else:  # S2
        if {"B8", "B4"}.issubset(df.columns):
            df["NDVI"] = (df["B8"] - df["B4"]) / (df["B8"] + df["B4"]).replace(0, np.nan)
        if S2_MODE == "indices":
            df = compute_indices(df, keep_10m_bands=False)
        elif S2_MODE == "bands_10m+indices":
            df = compute_indices(df, keep_10m_bands=True)

        if S2_NDVI_DIP_FILTER:
            df = remove_s2_ndvi_dips(df, S2_FEATURE_COLS, month_tag=month_tag,
                                       agri_year=agri_year, threshold=S2_NDVI_DIP_THRESHOLD)
        # NDVI always retained here — consumers decide whether to use it as a feature
        feature_cols = S2_FEATURE_COLS + (["NDVI"] if "NDVI" not in S2_FEATURE_COLS else [])

    if labeled:
        df["label"] = df["code_cultu"].apply(assign_class)
    else:
        df["label"] = np.nan
        df["code_cultu"] = np.nan
        df["code_group"] = np.nan

    dc = _date_col(df)
    if dc is None:
        return None
    df[dc] = pd.to_datetime(df[dc])
    daily = df.groupby(["id_parcel", dc])[feature_cols].median().reset_index()

    meta = df.groupby("id_parcel")[["label", "code_cultu", "code_group"]].first().reset_index()
    daily = daily.merge(meta, on="id_parcel", how="left")
    daily["agri_year"] = agri_year
    daily["month_tag"] = month_tag
    return daily.rename(columns={dc: "date"})


def load_long(sensor: str, all_years: list[int], ref_angle: float | None = None, unlabeled_years=()) -> pd.DataFrame:
    """Glob files for `sensor`, load in parallel with load_sensor_file, concat, sort."""
    from config import S1_DIR, S2_DIR
    base = "S1" if sensor.startswith("S1") else "S2"
    files = sorted((S1_DIR if base == "S1" else S2_DIR).glob(f"{base}_*.csv"))
    orbit_filter = {"S1_ASC": "ASCENDING", "S1_DESC": "DESCENDING"}.get(sensor)

    print(f"  Loading {sensor} CSVs in parallel …")
    frames = Parallel(n_jobs=N_JOBS)(
        delayed(load_sensor_file)(p, base, all_years, ref_angle, orbit_filter, unlabeled_years) for p in files
    )
    frames = [f for f in frames if f is not None]
    if not frames:
        raise ValueError(
            f"No usable rows for sensor '{sensor}'. Check: files exist for the requested years, "
            f"'orbit_direction' column present (for S1_ASC/S1_DESC), CLOUD_THRESHOLD, "
            f"EXCLUDED_CODE_GROUPS."
        )
    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values(["agri_year", "id_parcel", "date"]).reset_index(drop=True)
    print(f"  {sensor} long df : {long.shape}  |  parcels: {long['id_parcel'].nunique():,}  |  "
          f"dates: {long['date'].min().date()} → {long['date'].max().date()}")
    return long