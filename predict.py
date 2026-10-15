"""
predict.py — Train on all labeled years, predict the newest year (unlabeled).

Usage:  python predict.py
Set MODEL_NAME ("xgboost" | "inceptiontime"), SENSOR_MODES and paths in config.py.
The predict year and the months used are inferred from the data.
Output: one CSV per sensor mode, one row per parcel found in the predict-year files.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

from config import (S1_DIR, S2_DIR, OUTPUT_DIR, CLASS_NAMES, MONTH_ORDER, MODEL_NAME,
                    SENSOR_MODES, NDVI_POSTFILTER, S2_FEATURE_COLS)
from utils import _FILE_RE, required_data, detect_years
from ndvi_postfilter import build_ndvi_table, add_veg_subclass

warnings.filterwarnings("ignore")

_VALID = {"S1", "S2", "both", "S1_ASC", "S1_DESC", "S1_SPLIT_ORBIT", "both_split_orbit"}
_SPLIT = {"S1_ASC", "S1_DESC", "S1_SPLIT_ORBIT", "both_split_orbit"}
_SKIP  = {"id_parcel", "agri_year", "label", "code_cultu", "code_group", "orbit_direction"}
_IT_BLOCKS = {
    "S1": ["S1"], "S2": ["S2"], "both": ["S1", "S2"],
    "S1_ASC": ["S1_ASC"], "S1_DESC": ["S1_DESC"],
    "S1_SPLIT_ORBIT": ["S1_ASC", "S1_DESC"],
    "both_split_orbit": ["S1_ASC", "S1_DESC", "S2"],
}


# ── helpers ──────────────────────────────────────────────────────
def _years(files):
    return {int(m.group(1)) for p in files if (m := _FILE_RE.search(p.stem))}


def _uses(mode):
    return (mode.startswith("S1") or mode.startswith("both"),
            mode == "S2" or mode.startswith("both"))


def _window(cols, prefix):
    months = {c.rsplit("_", 1)[1] for c in cols if c.startswith(prefix + "_")}
    ordered = [m for m in MONTH_ORDER if m in months]
    return f"{ordered[0]}-{ordered[-1]}" if ordered else ""


def _check_config(s1_files, s2_files) -> int:
    """Validate config against the data. Returns the predict year."""
    if MODEL_NAME not in ("xgboost", "inceptiontime"):
        raise ValueError(f"MODEL_NAME must be 'xgboost' or 'inceptiontime', got {MODEL_NAME!r}")
    bad = [m for m in SENSOR_MODES if m not in _VALID]
    if bad:
        raise ValueError(f"Unknown sensor mode(s) {bad}. Valid: {sorted(_VALID)}")
    s1_y, s2_y = _years(s1_files), _years(s2_files)
    need = required_data(SENSOR_MODES)
    years = (s1_y if need in ("S1", "both") else set()) | (s2_y if need in ("S2", "both") else set())
    if not years:
        raise ValueError("No files found for the sensors required by SENSOR_MODES. "
                         f"Check S1_DIR / S2_DIR in config.py ({S1_DIR}, {S2_DIR}).")
    year = max(years)
    for mode in SENSOR_MODES:
        u1, u2 = _uses(mode)
        if u1 and year not in s1_y:
            raise ValueError(f"Mode '{mode}' needs S1 but there are no S1 files for {year} in {S1_DIR}.")
        if u2 and year not in s2_y:
            raise ValueError(f"Mode '{mode}' needs S2 but there are no S2 files for {year} in {S2_DIR}.")
        if mode in _SPLIT and u1:
            f = next(p for p in s1_files if (m := _FILE_RE.search(p.stem)) and int(m.group(1)) == year)
            if "orbit_direction" not in pd.read_csv(f, nrows=0).columns:
                raise ValueError(f"Mode '{mode}' needs an 'orbit_direction' column in the S1 files.")
    return year


def _all_parcels(year, mode) -> pd.DataFrame:
    """Every parcel in the raw predict-year files of the sensors this mode uses."""
    u1, u2 = _uses(mode)
    ids = []
    for used, folder, pre in ((u1, S1_DIR, "S1"), (u2, S2_DIR, "S2")):
        if used:
            for p in folder.glob(f"{pre}_*.csv"):
                m = _FILE_RE.search(p.stem)
                if m and int(m.group(1)) == year:
                    ids.append(pd.read_csv(p, usecols=["id_parcel"])["id_parcel"])
    return pd.DataFrame({"id_parcel": pd.concat(ids).unique()}).assign(agri_year=year)


# ── XGBoost branch ───────────────────────────────────────────────
def run_ml(mode, features, year):
    from sklearn.preprocessing import StandardScaler
    from ml_dataloader import select_sensor_columns
    from ml_preproc import drop_unavailable_months, count_valid_months, interpolate_monthly
    from ml_model import load_params, _get_clf, _fit

    u1, u2 = _uses(mode)
    feats = drop_unavailable_months(features, year)                 # months first
    cols = select_sensor_columns([c for c in feats.columns if c not in _SKIP], mode)
    counts = {p: count_valid_months(feats, cols, p)                 # count before filling
              for p, used in (("s1", u1), ("s2", u2)) if used}
    feats = interpolate_monthly(feats)                              # then fill (as in the report)
    for p, n in counts.items():
        feats[f"n_valid_months_{p}"] = n

    train_df = feats[(feats["agri_year"] != year) & feats["label"].notna()]
    pred_df = feats[feats["agri_year"] == year].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("No labeled training parcels found (need a labeled year before the predict year).")
    print(f"[predict] {mode}: train years {sorted(int(y) for y in train_df['agri_year'].unique())} -> predict {year}")

    X_tr = train_df[cols].values.astype(float)
    y_tr = train_df["label"].values.astype(int)
    X_pr = pred_df[cols].values.astype(float)
    med = np.nanmedian(X_tr, axis=0)                                # residual NaNs only
    for X in (X_tr, X_pr):
        r, c = np.where(np.isnan(X))
        X[r, c] = med[c]
    scaler = StandardScaler()
    X_tr, X_pr = scaler.fit_transform(X_tr), scaler.transform(X_pr)

    clf = _fit(_get_clf(load_params()), X_tr, y_tr)
    proba = clf.predict_proba(X_pr)

    reason = pd.Series("", index=pred_df.index)
    extras = pd.DataFrame(index=pred_df.index)
    for p, used in (("s1", u1), ("s2", u2)):
        if used:
            reason[pred_df[f"n_valid_months_{p}"] == 0] += f"no valid {p.upper()} month; "
            extras[f"n_valid_months_{p}"] = pred_df[f"n_valid_months_{p}"].values
            extras[f"window_{p}"] = _window(cols, p)
    return pred_df[["id_parcel", "agri_year"]], reason, proba, clf.classes_, extras


# ── InceptionTime branch ─────────────────────────────────────────
def run_it(mode, year, all_years):
    from it_dataloader import get_long
    from it_preproc import S1_TS_COLS, _s1_base
    from it_harmonic import build_harmonic_tensor
    from it_model_inception import fit_predict_inception

    train_years = [y for y in all_years if y != year]
    blocks = _IT_BLOCKS[mode]
    longs = {b: get_long(b, all_years, train_years, (year,)) for b in blocks}

    # Cutoff = last month with data in the predict year (shortest across sensors)
    lasts = []
    for b, lg in longs.items():
        months = set(lg.loc[lg["agri_year"] == year, "month_tag"])
        if not months:
            raise ValueError(f"No usable {b} observations in {year} after filtering.")
        lasts.append(max(months, key=MONTH_ORDER.index))
    cutoff = min(lasts, key=MONTH_ORDER.index)
    print(f"[predict] {mode}: season window m09-{cutoff}")

    tensors = {}
    for b, lg in longs.items():
        names = S1_TS_COLS if _s1_base(b) == "S1" else S2_FEATURE_COLS
        cols = [c for c in names if c in lg.columns]
        X, _, ids = build_harmonic_tensor(lg, cols, last_month=cutoff)
        tensors[b] = (X, ids)

    # Align sensor blocks on (agri_year, id_parcel); a missing block stays NaN (= unusable)
    key = ["agri_year", "id_parcel"]
    keys = pd.concat([ids[key] for _, ids in tensors.values()]).drop_duplicates().reset_index(drop=True)
    n = len(keys)
    label = np.full(n, -1)
    Xs, usable, nobs = [], {}, {}
    for b, (X, ids) in tensors.items():
        pos = keys.merge(ids.assign(i=np.arange(len(ids))), on=key, how="left")
        i = pos["i"].to_numpy()
        have = ~np.isnan(i)
        Xb = np.full((n,) + X.shape[1:], np.nan, dtype=np.float32)
        Xb[have] = X[i[have].astype(int)]
        Xs.append(Xb)
        usable[b] = have & ~np.isnan(Xb).any(axis=(1, 2))       # a failed fit leaves NaN
        nobs[b] = pos["n_obs"].fillna(0).astype(int).to_numpy()
        label = np.where(label < 0, pos["label"].fillna(-1).astype(int).to_numpy(), label)
    X = np.concatenate(Xs, axis=1)

    ok = np.all([usable[b] for b in blocks], axis=0)
    is_pred = (keys["agri_year"] == year).to_numpy()
    tr = ~is_pred & ok & (label >= 0)
    if not tr.any():
        raise ValueError("No usable labeled training parcels found.")
    y_tr = label[tr]
    if set(np.unique(y_tr)) != set(CLASS_NAMES):
        raise ValueError(f"Training data must contain all classes {sorted(CLASS_NAMES)}; "
                         f"found {sorted(set(np.unique(y_tr)))}.")
    print(f"[predict] {mode}: {int(tr.sum()):,} usable training parcels "
          f"(train years {train_years}) | predict-year usable: {int(ok[is_pred].sum()):,} / {int(is_pred.sum()):,}")

    proba = np.full((int(is_pred.sum()), len(CLASS_NAMES)), np.nan)
    pr_ok = ok[is_pred]
    if pr_ok.any():
        proba[pr_ok] = fit_predict_inception(X[tr], y_tr, X[is_pred][pr_ok],
                                             OUTPUT_DIR / "InceptionTime" / mode)

    ids_out = keys[is_pred][["id_parcel", "agri_year"]].reset_index(drop=True)
    reason = pd.Series("", index=ids_out.index)
    extras = pd.DataFrame({"window": f"m09-{cutoff}"}, index=ids_out.index)
    for b in blocks:
        reason[~usable[b][is_pred]] += f"insufficient observations ({b}); "
        extras[f"n_valid_obs_{b.lower()}"] = nobs[b][is_pred]
    return ids_out, reason, proba, np.arange(len(CLASS_NAMES)), extras


# ── shared output ────────────────────────────────────────────────
def finish(mode, year, parcels, ids, reason, proba, classes, extras, ndvi):
    ok = (reason.values == "")
    p = np.nan_to_num(proba)
    out = ids.reset_index(drop=True).copy()
    out["classified"] = ok
    out["reason"] = reason.str.strip("; ").values
    out["pred_class"] = np.where(ok, classes[p.argmax(1)], np.nan)
    out["pred_name"] = out["pred_class"].map(CLASS_NAMES)
    out["confidence"] = np.where(ok, p.max(1), np.nan)
    for i, cls in enumerate(classes):
        out[f"proba_{CLASS_NAMES[cls].lower().replace(' ', '_')}"] = np.where(ok, p[:, i], np.nan)
    out = pd.concat([out, extras.reset_index(drop=True)], axis=1)

    # Parcels present in the raw files but with no usable observation at all
    seen = parcels.merge(out[["id_parcel", "agri_year"]], how="left", indicator=True)
    lost = seen.loc[seen["_merge"] == "left_only", ["id_parcel", "agri_year"]]
    if len(lost):
        lost = lost.assign(classified=False, reason="no valid observation after filtering")
        out = pd.concat([out, lost], ignore_index=True)

    if NDVI_POSTFILTER and ndvi is not None:
        out = add_veg_subclass(out, ndvi)

    path = OUTPUT_DIR / f"predictions_{year}_{MODEL_NAME}_{mode}.csv"
    out.to_csv(path, index=False)
    n_ok = int(out["classified"].sum())
    print(f"[predict] {n_ok:,} classified / {len(out):,} parcels "
          f"({100 * n_ok / max(len(out), 1):.1f}%) --> {path}")
    print(out.loc[out["classified"], "pred_name"].value_counts().to_string())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    s1_files = sorted(S1_DIR.glob("S1_*.csv"))
    s2_files = sorted(S2_DIR.glob("S2_*.csv"))
    year = _check_config(s1_files, s2_files)
    print(f"\n[predict] model: {MODEL_NAME} | year to predict: {year} | modes: {SENSOR_MODES}")

    ndvi = build_ndvi_table(year) if NDVI_POSTFILTER else None

    matrices, all_years = {}, None
    if MODEL_NAME == "xgboost":
        from ml_dataloader import build_features
        for split in (False, True):
            modes = [m for m in SENSOR_MODES if (m in _SPLIT) == split]
            if modes:
                matrices[split] = build_features(split_orbits=split, sensors=required_data(modes),
                                                 unlabeled_years=(year,))[0]
    else:
        all_years, test_year, _ = detect_years(s1_files, s2_files, required_data(SENSOR_MODES))
        assert test_year == year

    for mode in SENSOR_MODES:
        print("\n" + "#" * 60 + f"\n  SENSOR MODE : {mode}\n" + "#" * 60)
        parcels = _all_parcels(year, mode)
        if MODEL_NAME == "xgboost":
            parts = run_ml(mode, matrices[mode in _SPLIT], year)
        else:
            parts = run_it(mode, year, all_years)
        finish(mode, year, parcels, *parts, ndvi)


if __name__ == "__main__":
    main()