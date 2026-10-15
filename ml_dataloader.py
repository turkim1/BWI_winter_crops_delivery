"""
ml_dataloader.py — Build (or load from cache) the wide feature matrix.

Public API
----------
build_features(split_orbits, sensors, unlabeled_years) -> tuple[pd.DataFrame, list[int], int, list[int], pd.DataFrame]
    Returns (features_df, all_years, test_year, train_years, meta_df).
    meta_df : DataFrame(id_parcel, agri_year, code_cultu) — never used for training,
              kept for post-classification diagnostics (e.g. NDVI activity filter).

select_sensor_columns(all_feat_cols, sensor) -> list[str]
    Filters feature column names to those matching the chosen sensor.
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from config import (
    S1_DIR, S2_DIR, CACHE_DIR, SITE,
    S2_MODE, CLOUD_THRESHOLD, AGG_FUNCS,
    S2_FEATURE_COLS, S1_NUMERIC_COLS,
    REF_ANGLE, EXCLUDED_CODE_GROUPS, CLASS_NAMES,
    S1_SPIKE_FILTER, S1_SPIKE_THRESHOLD_DB,
    S2_NDVI_DIP_FILTER, S2_NDVI_DIP_THRESHOLD, WINTER_CROPS, CZH_CROPS
)
from utils import detect_years, validate_input_files
from common_preproc import compute_ref_angle, load_long
from ml_preproc import aggregate_monthly, stack_to_wide, drop_leakage_columns


# -------------------------------------------------------------
# CACHE KEY
# -------------------------------------------------------------

def _cache_key(all_years: list[int], split_orbits: bool, train_years: list[int],
               sensors: str, unlabeled_years: list[int], files: list[str]) -> str:
    payload = {
        "ALL_YEARS":             all_years,
        "SENSORS":               sensors,
        "S2_MODE":               S2_MODE,
        "CLOUD_THRESHOLD":       CLOUD_THRESHOLD,
        "AGG_FUNCS":             AGG_FUNCS,
        "S2_FEATURE_COLS":       S2_FEATURE_COLS,
        "S1_NUMERIC_COLS":       S1_NUMERIC_COLS,
        "EXCLUDED_CODE_GROUPS":  sorted(EXCLUDED_CODE_GROUPS),
        "WINTER_CROPS":          sorted(WINTER_CROPS),
        "CZH_CROPS":             sorted(CZH_CROPS),
        "S1_DIR":                str(S1_DIR),
        "S2_DIR":                str(S2_DIR),
        "S1_SPIKE_FILTER":       S1_SPIKE_FILTER,
        "S1_SPIKE_THRESHOLD_DB": S1_SPIKE_THRESHOLD_DB,
        "S2_NDVI_DIP_FILTER":    S2_NDVI_DIP_FILTER,
        "S2_NDVI_DIP_THRESHOLD": S2_NDVI_DIP_THRESHOLD,
        "SPLIT_ORBITS":          split_orbits,
        "REF_ANGLE":             REF_ANGLE,
        "TRAIN_YEARS":           sorted(train_years),
        "UNLABELED":             sorted(unlabeled_years),
        "MATRIX_STAGE":          "raw_nan",            # the cached matrix is no longer interpolated
        "FILES":                 files,                # new monthly files must invalidate the cache
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


# -------------------------------------------------------------
# SENSOR COLUMN SELECTION
# -------------------------------------------------------------

def select_sensor_columns(all_feat_cols: list[str], sensor: str) -> list[str]:

    if sensor == "S2":
        selected = [c for c in all_feat_cols if c.startswith("s2_")]

    elif sensor == "S1":
        selected = [
            c for c in all_feat_cols
            if c.startswith("s1_") and "_asc_" not in c and "_desc_" not in c
        ]

    elif sensor == "S1_ASC":
        selected = [c for c in all_feat_cols if c.startswith("s1_") and "_asc_" in c]

    elif sensor == "S1_DESC":
        selected = [c for c in all_feat_cols if c.startswith("s1_") and "_desc_" in c]

    elif sensor == "S1_SPLIT_ORBIT":
        selected = [
            c for c in all_feat_cols
            if c.startswith("s1_") and ("_asc_" in c or "_desc_" in c)
        ]

    elif sensor in ("both", "both_split_orbit"):
        selected = list(all_feat_cols)

    else:
        raise ValueError(f"Unknown sensor '{sensor}'.")

    if S2_MODE == "bands" or S2_MODE == "bands_10m" :
        selected = [c for c in selected if "_NDVI_" not in c]

    if not selected:
        raise ValueError(
            f"No columns found for sensor '{sensor}'. "
            f"Available columns (first 10): {all_feat_cols[:10]}"
        )
    
    print(f"  [sensor={sensor}] Using {len(selected)} / {len(all_feat_cols)} feature columns.")

    return selected

# -------------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------------

def build_features(
    split_orbits: bool = False,
    sensors: str = "both",            # "S1" | "S2" | "both"
    unlabeled_years: tuple = ()
) -> tuple[pd.DataFrame, list[int], int, list[int], pd.DataFrame | None]:

    if sensors not in ("S1", "S2", "both"):
        raise ValueError(f"sensors must be 'S1', 'S2' or 'both', got {sensors!r}")
    use_s1 = sensors in ("S1", "both")
    use_s2 = sensors in ("S2", "both")

    s1_files = sorted(S1_DIR.glob("S1_*.csv")) if use_s1 else []
    s2_files = sorted(S2_DIR.glob("S2_*.csv")) if use_s2 else []
    print(f"S2 files : {len(s2_files)}  |  S1 files : {len(s1_files)}  |  sensors : {sensors}")

    all_years, test_year, train_years = detect_years(s1_files, s2_files, sensors)
    validate_input_files(s1_files, s2_files, all_years, sensors)

    files = [(p.name, p.stat().st_size) for p in s1_files + s2_files]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    orbit_tag_str = "split" if split_orbits else "all"
    feature_cache = CACHE_DIR / (
        f"features_{_cache_key(all_years, split_orbits, train_years, sensors, unlabeled_years, files)}"
        f"_{SITE}_{orbit_tag_str}.parquet"
    )
    meta_cache = feature_cache.with_name(feature_cache.stem + "_meta.parquet")

    if feature_cache.exists():
        print(f"[cache] Loading feature matrix from {feature_cache}")
        features = pd.read_parquet(feature_cache)
        meta_df  = pd.read_parquet(meta_cache) if meta_cache.exists() else None
        if meta_df is None:
            print("[cache] WARNING : meta cache not found — code_cultu unavailable.")
        return features, all_years, test_year, train_years, meta_df

    ref_angle = (REF_ANGLE if REF_ANGLE is not None
                 else compute_ref_angle(s1_files, train_years)) if use_s1 else None
    id_cols = ["id_parcel", "agri_year"]
    label_cols = ["label", "code_cultu", "code_group"]

    # ── S2 wide ───────────────────────────────────────────────────────
    feat_s2 = None
    if use_s2:
        long_s2 = load_long("S2", all_years, unlabeled_years=unlabeled_years)
        monthly_s2 = [aggregate_monthly(long_s2[long_s2["month_tag"] == m], "s2", S2_FEATURE_COLS)
                      for m in long_s2["month_tag"].unique()]
        feat_s2 = stack_to_wide(monthly_s2, id_cols, label_cols=label_cols)
        print(f"  S2 wide : {feat_s2.shape}")

    # ── S1 wide ───────────────────────────────────────────────────────
    feat_s1 = None
    if use_s1:
        s1_cols = S1_NUMERIC_COLS[:-1] + ["VV_VH_ratio"]
        if split_orbits:
            long_asc  = load_long("S1_ASC", all_years, ref_angle, unlabeled_years=unlabeled_years)
            long_desc = load_long("S1_DESC", all_years, ref_angle, unlabeled_years=unlabeled_years)
            monthly_s1 = (
                [aggregate_monthly(long_asc[long_asc["month_tag"] == m], "s1", s1_cols, "asc")
                 for m in long_asc["month_tag"].unique()] +
                [aggregate_monthly(long_desc[long_desc["month_tag"] == m], "s1", s1_cols, "desc")
                 for m in long_desc["month_tag"].unique()]
            )
        else:
            long_s1 = load_long("S1", all_years, ref_angle, unlabeled_years=unlabeled_years)
            monthly_s1 = [aggregate_monthly(long_s1[long_s1["month_tag"] == m], "s1", s1_cols)
                          for m in long_s1["month_tag"].unique()]

        feat_s1 = stack_to_wide(monthly_s1, id_cols, label_cols=label_cols)
        if use_s2:   # labels come from S2 when both are present; S1-only keeps its own
            feat_s1 = feat_s1.drop(columns=label_cols, errors="ignore")
        print(f"  S1 wide : {feat_s1.shape}")

    # ── Merge ─────────────────────────────────────────────────────────
    if feat_s1 is not None and feat_s2 is not None:
        print("Merging sensors …")
        features = feat_s2.merge(feat_s1, on=id_cols, how="outer")
    else:
        features = feat_s2 if feat_s2 is not None else feat_s1
    print(f"  Shape   : {features.shape}")

    print("\n  Class distribution:")
    print(features["label"].value_counts().rename(index=CLASS_NAMES).to_string())

    meta_df  = features[["id_parcel", "agri_year", "code_cultu"]].copy()
    features = drop_leakage_columns(features)

    features.to_parquet(feature_cache, index=False)
    meta_df.to_parquet(meta_cache, index=False)
    print(f"  [cache] Saved --> {feature_cache}")
    print(f"  [cache] Meta  --> {meta_cache}")

    return features, all_years, test_year, train_years, meta_df