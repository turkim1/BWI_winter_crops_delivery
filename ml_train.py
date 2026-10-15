"""
ml_train.py — Entry point.

Loop behaviour (default)
------------------------
For each sensor mode in SENSOR_MODES (e.g. ["S1", "S2", "both"]):
For leave-one-year-out, change the last line to main(single=False)

Usage
-----
  python ml_train.py               # full LOYO loop for all sensor modes
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import SENSOR_MODES, S2_MODE, OUTPUT_DIR, CLASS_NAMES
from ml_dataloader import build_features, select_sensor_columns
from ml_preproc import interpolate_monthly
from plots import (
    plot_class_distribution,
    plot_ndvi_timeseries,
    plot_VV_VH_timeseries,
)
from ml_model import train_and_evaluate
from ml_metrics_logger import log_trial
from utils import required_data

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# DIAGNOSTIC CHECKS
# -------------------------------------------------------------

def _check_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_parcels = set(train_df["id_parcel"])
    test_parcels  = set(test_df["id_parcel"])
    overlap       = train_parcels & test_parcels
    pct           = 100 * len(overlap) / max(len(test_parcels), 1)
    print(
        f"\n[leakage check] Parcel overlap: {len(overlap):,} / {len(test_parcels):,} "
        f"({pct:.1f}% of test parcels seen in training)"
    )


# -------------------------------------------------------------
# SINGLE RUN
# -------------------------------------------------------------

def run_one(
    features: pd.DataFrame,
    test_year: int,
    train_years: list[int],
    sensor: str,
    sensor_run_dir,          # pathlib.Path
) -> None:
    """Execute one complete train / test split end-to-end."""
    features = interpolate_monthly(features)
    run_tag  = f"train_{'_'.join(str(y) for y in train_years)}_test_{test_year}"
    run_dir  = sensor_run_dir / run_tag
    plot_dir = run_dir / "plots"
    cm_dir   = sensor_run_dir / "confusion_matrices"

    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Time-series diagnostic plots ────────────
    if sensor in ("S2", "both"):
        plot_ndvi_timeseries(features, train_years, test_year, plot_dir=plot_dir)
    if sensor in ("S1", "both"):
        plot_VV_VH_timeseries(features, train_years, test_year, plot_dir=plot_dir)

    # ── Feature column selection ─────────────────────────────────────
    skip_cols     = {"id_parcel", "agri_year", "label", "code_cultu", "code_group", "orbit_direction"} # not features - orbit_direction is only for filtering asc desc
    all_feat_cols = [c for c in features.columns if c not in skip_cols]
    feat_cols     = select_sensor_columns(all_feat_cols, sensor)

    train_df = features[features["agri_year"].isin(train_years)]
    test_df  = features[features["agri_year"] == test_year]

    # ── Diagnostic checks ────────────────────────────────────────────
    _check_leakage(train_df, test_df)

    # ── Build numpy arrays ───────────────────────────────────────────
    X_train = train_df[feat_cols].values.astype(float)
    y_train = train_df["label"].values
    X_test = test_df[feat_cols].values.astype(float)
    y_test = test_df["label"].values

    # ── Residual NaN fallback ────────────────────────────────────────
    train_col_medians = np.nanmedian(X_train, axis=0)
    n_nan_train = np.isnan(X_train).sum()
    n_nan_test  = np.isnan(X_test).sum()
    if n_nan_train or n_nan_test:
        print(
            f"  [imputation] Residual NaNs — train: {n_nan_train}, test: {n_nan_test}. "
            f"Filling with train-set column medians."
        )
        nan_mask_train          = np.isnan(X_train)
        X_train[nan_mask_train] = np.take(train_col_medians, np.where(nan_mask_train)[1])
        nan_mask_test           = np.isnan(X_test)
        X_test[nan_mask_test]   = np.take(train_col_medians, np.where(nan_mask_test)[1])

    # ── Standardize features ─────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(
        f"\nTrain : {len(X_train):,}  |  Test : {len(X_test):,}"
        f"  |  Features : {X_train.shape[1]}"
    )


    _, y_pred = train_and_evaluate(
        X_train, y_train,
        X_test,  y_test,
        feat_cols,
        test_year,
        run_dir=run_dir,
        plot_dir=plot_dir,
        cm_dir=cm_dir,
        train_years=train_years,
    )

    log_trial(
        output_dir     = OUTPUT_DIR,
        sensor         = sensor,
        test_year      = test_year,
        train_years    = train_years,
        n_train        = len(X_train),
        n_test         = len(X_test),
        y_test         = y_test,
        y_pred         = y_pred,
        class_names    = CLASS_NAMES,
    )

    # ── Save predictions for optional standalone NDVI filtering ───────
    predictions = pd.DataFrame({
        "id_parcel": test_df["id_parcel"].values,
        "agri_year": test_df["agri_year"].values,
        "y_true":    y_test,
        "y_pred":    y_pred,
    })
    pred_path = run_dir / f"predictions_{test_year}.csv"
    predictions.to_csv(pred_path, index=False)
    print(f"  [predictions] Saved --> {pred_path}")

# -------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------
# Two cache groups:
#   "base"  → s1_* (no orbit tag) + s2_*
#   "split" → s1_*_asc_* + s1_*_desc_* + s2_*
_SENSOR_BUILD_PARAMS: dict[str, tuple[str | None, bool, str | None]] = {
    "S1":               (None, False, None),  # cache A
    "S2":               (None, False, None),  # cache A
    "both":             (None, False, None),  # cache A
    "S1_ASC":           (None, True,  None),  # cache B 
    "S1_DESC":          (None, True,  None),  # cache B
    "S1_SPLIT_ORBIT":   (None, True,  None),  # cache B
    "both_split_orbit": (None, True,  None),  # cache B
}

def main(single: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for s in SENSOR_MODES:
        if s not in _SENSOR_BUILD_PARAMS:
            raise ValueError(f"Sensor '{s}' unknown. Valid: {list(_SENSOR_BUILD_PARAMS)}")

    # Build only the caches the configured modes need, each with only the sensors it needs
    caches: dict[bool, tuple[pd.DataFrame, list[int]]] = {}
    for split in (False, True):
        modes = [s for s in SENSOR_MODES if _SENSOR_BUILD_PARAMS[s][1] == split]
        if not modes:
            continue
        need = required_data(modes)
        print(f"\n[cache {'B' if split else 'A'}] Building {'split-orbit' if split else 'base'} "
              f"feature matrix (sensors: {need}) …")
        feats, yrs, _, _, _ = build_features(split_orbits=split, sensors=need)
        caches[split] = (feats.dropna(subset=["label"]), yrs)

    first_feats, first_years = next(iter(caches.values()))
    plot_class_distribution(first_feats, first_years, plot_dir=OUTPUT_DIR / "plots")

    for sensor in SENSOR_MODES:
        _, split_orbits, _ = _SENSOR_BUILD_PARAMS[sensor]
        features, all_years = caches[split_orbits]

        s2_label     = S2_MODE if sensor in ("S2", "both", "both_split_orbit") else ""
        sensor_label = f"{sensor}_{s2_label}" if s2_label else sensor

        print("\n" + "#" * 60)
        print(f"  SENSOR MODE : {sensor}  (S2={S2_MODE if s2_label else 'n/a'})")
        print("#" * 60)

        sensor_run_dir = OUTPUT_DIR / sensor_label
        sensor_run_dir.mkdir(parents=True, exist_ok=True)

        if single:
            train_years = all_years[:-1]
            test_year   = all_years[-1]
            print(f"\nSINGLE RUN | train={train_years} | test={test_year}")
            run_one(features, test_year, train_years, sensor, sensor_run_dir)
        else:
            n_runs = len(all_years)
            for i, test_year in enumerate(all_years):
                train_years = [y for y in all_years if y != test_year]
                print("\n" + "-" * 60)
                print(f"RUN {i + 1}/{n_runs} [{sensor}] | train={train_years} | test={test_year}")
                print("-" * 60)
                run_one(features, test_year, train_years, sensor, sensor_run_dir)


if __name__ == "__main__":
    main(single=True) # TEST ONLY ON SINGLE LATEST YEAR # Change to main(single=False) for full LOYO loop