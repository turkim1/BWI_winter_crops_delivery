"""
it_dataloader.py — Generic harmonic tensor builder for S1 and S2.


Public API
----------
build_tensor(sensor, all_years, train_years) -> dict
    "X"         : (N, C, T_grid) float32
    "y"         : (N,)           int64
    "ids"       : DataFrame(id_parcel, agri_year, label)
    "all_years" : list[int]
    "feat_names": list[str]

Single sensors   : "S1", "S1_ASC", "S1_DESC", "S2"
                   → loaded from raw CSVs, harmonically resampled, cached on disk.
Composite sensors: "both", "S1_SPLIT_ORBIT", "both_split_orbit"
                   → recursive build_tensor calls + in-memory merge, NOT cached
                   (the expensive single tensors are already cached individually).

"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from plots import plot_harmonic_timeseries
from config import (
    S1_DIR, S2_DIR, OUTPUT_DIR, CACHE_DIR, SITE,
    S1_SPIKE_FILTER, S1_SPIKE_THRESHOLD_DB,
    CLOUD_THRESHOLD,
    S2_FEATURE_COLS, S2_MODE,
    S2_NDVI_DIP_FILTER, S2_NDVI_DIP_THRESHOLD,
    EXCLUDED_CODE_GROUPS,
    IT_HARMONIC_STEP_DAYS, IT_HARMONIC_N_PER_SEASON, REF_ANGLE, WINTER_CROPS, CZH_CROPS
)
from utils import validate_input_files
from common_preproc import compute_ref_angle
from it_preproc import S1_TS_COLS, _s1_base
from common_preproc import load_long
from it_harmonic import build_harmonic_tensor

_SINGLE_SENSORS    = frozenset({"S1", "S1_ASC", "S1_DESC", "S2"})
_COMPOSITE_SENSORS = frozenset({"both", "S1_SPLIT_ORBIT", "both_split_orbit"})


# ─────────────────────────────────────────────────────────────
# CACHE KEY  (single sensors only)
# ─────────────────────────────────────────────────────────────
def _cache_key_long(sensor, all_years, train_years, unlabeled_years=()):
    d, pat = (S1_DIR, "S1_*.csv") if _s1_base(sensor) == "S1" else (S2_DIR, "S2_*.csv")
    payload = {
        "sensor":          sensor,
        "all_years":       all_years,
        "excluded_groups": sorted(EXCLUDED_CODE_GROUPS),
        "winter_crops":    sorted(WINTER_CROPS),
        "czh_crops":       sorted(CZH_CROPS),
        "ref_angle":       REF_ANGLE,
        "unlabeled":       sorted(unlabeled_years),
        "files":           [(p.name, p.stat().st_size) for p in sorted(d.glob(pat))],
    }
    if _s1_base(sensor) == "S1":
        payload.update({
            "s1_ts_cols":         S1_TS_COLS,
            "spike_filter":       S1_SPIKE_FILTER,
            "spike_threshold_db": S1_SPIKE_THRESHOLD_DB,
            "s1_dir":             str(S1_DIR),
            "train_years":        sorted(train_years),
        })
    else:
        payload.update({
            "feat_cols":       S2_FEATURE_COLS,
            "s2_mode":         S2_MODE,
            "cloud_threshold": CLOUD_THRESHOLD,
            "ndvi_dip_filter": S2_NDVI_DIP_FILTER,
            "ndvi_dip_thresh": S2_NDVI_DIP_THRESHOLD,
            "s2_dir":          str(S2_DIR),
        })
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def _cache_key_tensor(sensor, all_years, train_years):
    payload = {
        "long_key":      _cache_key_long(sensor, all_years, train_years),
        "harmonic_step": IT_HARMONIC_STEP_DAYS,
        "harmonic_n":    IT_HARMONIC_N_PER_SEASON,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]

# ─────────────────────────────────────────────────────────────
# TENSOR MERGE  (composite sensors)
# ─────────────────────────────────────────────────────────────

def _merge_tensors(d1: dict, d2: dict,
                   prefix: tuple[str, str] | None = None) -> dict:
    """
    Inner-join two tensor dicts on (agri_year, id_parcel) and concatenate
    their channels (axis=1).
    """
    key = ["agri_year", "id_parcel"]

    ids1 = d1["ids"].reset_index(drop=True).reset_index().rename(columns={"index": "idx1"})
    ids2 = d2["ids"].reset_index(drop=True).reset_index().rename(columns={"index": "idx2"})

    merged = (ids1[key + ["label", "idx1"]]
              .merge(ids2[key + ["idx2"]], on=key, how="inner")
              .reset_index(drop=True))

    idx1, idx2 = merged["idx1"].values, merged["idx2"].values

    if prefix is None:
        feat_names = d1["feat_names"] + d2["feat_names"]
    else:
        p1, p2 = prefix
        feat_names = ([f"{f}_{p1}" for f in d1["feat_names"]] +
                      [f"{f}_{p2}" for f in d2["feat_names"]])

    return {
        "X":          np.concatenate([d1["X"][idx1], d2["X"][idx2]], axis=1),
        "y":          d1["y"][idx1],
        "ids":        merged[key + ["label"]].reset_index(drop=True),
        "all_years":  d1["all_years"],
        "feat_names": feat_names,
    }


# ─────────────────────────────────────────────────────────────
# BUILD TENSOR  (single sensors)
# ─────────────────────────────────────────────────────────────
def get_long(sensor: str, all_years: list[int], train_years: list[int],
             unlabeled_years=()) -> pd.DataFrame:
    """Cached long DataFrame for ONE single sensor (S1, S1_ASC, S1_DESC, S2)."""
    if sensor not in _SINGLE_SENSORS:
        raise ValueError(f"Unknown single sensor: {sensor!r}")
    s1_files = sorted(S1_DIR.glob("S1_*.csv")) if _s1_base(sensor) == "S1" else []
    s2_files = sorted(S2_DIR.glob("S2_*.csv")) if sensor == "S2" else []
    validate_input_files(s1_files, s2_files, all_years, _s1_base(sensor))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lk = _cache_key_long(sensor, all_years, train_years, unlabeled_years)
    cache_long = CACHE_DIR / f"{sensor.lower()}_long_{lk}_{SITE}.parquet"
    if cache_long.exists():
        print(f"  [cache] Loading long df  {cache_long.name}")
        return pd.read_parquet(cache_long)

    ref_angle = (
        (REF_ANGLE if REF_ANGLE is not None else compute_ref_angle(s1_files, train_years))
        if _s1_base(sensor) == "S1" else None
    )
    long = load_long(sensor, all_years, ref_angle, unlabeled_years)
    long.to_parquet(cache_long, index=False)
    print(f"  [cache] Saved long df → {cache_long}")
    return long

def build_tensor(sensor: str, all_years: list[int], train_years: list[int]) -> dict:
    """
    Build (or load from cache) a harmonic tensor dict for *sensor*, using
    the full Sep–Jun season.

    train_years is used ONLY to compute the S1 reference angle (from
    training data only, to avoid leaking test-year acquisitions into the
    normalization). The returned tensor itself still covers all_years —
    splitting into train/test happens downstream.
    """
    if sensor not in _SINGLE_SENSORS | _COMPOSITE_SENSORS:
        raise ValueError(f"Unknown sensor: {sensor!r}. "
                         f"Valid: {sorted(_SINGLE_SENSORS | _COMPOSITE_SENSORS)}")

    if sensor == "both":
        return _merge_tensors(
            build_tensor("S1", all_years, train_years),
            build_tensor("S2", all_years, train_years),
        )

    if sensor == "S1_SPLIT_ORBIT":
        return _merge_tensors(
            build_tensor("S1_ASC",  all_years, train_years),
            build_tensor("S1_DESC", all_years, train_years),
            prefix=("asc", "desc"),
        )

    if sensor == "both_split_orbit":
        return _merge_tensors(
            build_tensor("S1_SPLIT_ORBIT", all_years, train_years),
            build_tensor("S2",             all_years, train_years),
        )

    # ── Single sensors ────────────────────────────────────────────────
    long = get_long(sensor, all_years, train_years)
    feat_names = S1_TS_COLS if _s1_base(sensor) == "S1" else S2_FEATURE_COLS

    feat_cols = [c for c in feat_names if c in long.columns]

    ck        = _cache_key_tensor(sensor, all_years, train_years)
    cache_X   = CACHE_DIR / f"{sensor.lower()}_harm_X_{ck}_{SITE}.npy"
    cache_ids = CACHE_DIR / f"{sensor.lower()}_harm_ids_{ck}_{SITE}.parquet"

    if cache_X.exists() and cache_ids.exists():
        print(f"  [cache] Loading tensor  {cache_X.name}")
        X      = np.load(cache_X)
        ids_df = pd.read_parquet(cache_ids)
        y      = ids_df["label"].values.astype(np.int64)
        print(f"  X: {X.shape}  y: {y.shape}")
        return {"X": X, "y": y, "ids": ids_df, "all_years": all_years, "feat_names": feat_cols}

    for year in all_years:
        plot_harmonic_timeseries(
            long, sensor, feat_cols,
            agri_year=year,
            plot_dir=OUTPUT_DIR / "plots_harmonic",
        )

    X, y, ids_df = build_harmonic_tensor(long, feat_cols)

    tmp_X   = cache_X.with_suffix(".tmp.npy")
    tmp_ids = cache_ids.with_suffix(".tmp.parquet")
    np.save(tmp_X, X)
    ids_df.to_parquet(tmp_ids, index=False)
    tmp_X.rename(cache_X)
    tmp_ids.rename(cache_ids)
    print(f"  [cache] Saved tensor → {cache_X}")

    return {"X": X, "y": y, "ids": ids_df, "all_years": all_years, "feat_names": feat_cols}