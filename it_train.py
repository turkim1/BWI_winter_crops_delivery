"""
it_train.py — InceptionTime entry point.

Trains once per sensor mode in SENSOR_MODES, on every labeled year except
the most recent, and evaluates on the held-out most recent year — same
held-out-year convention as ml_train.py.

Before running: edit SENSOR_MODES in config.py to match the data you
have (S1 only, S2 only, or both).

Usage
-----
  python it_train.py
"""
from __future__ import annotations

import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED, OUTPUT_DIR, SENSOR_MODES
from utils import detect_years, required_data
from it_dataloader import build_tensor
from it_model_inception import train_and_evaluate_inception

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────
# SINGLE RUN
# ─────────────────────────────────────────────────────────────

def run_one(data, test_year, train_years, run_dir, sensor):
    cm_dir = run_dir.parent / "confusion_matrices"
    ids = data["ids"]

    test_mask  = (ids["agri_year"] == test_year).values
    train_mask = ids["agri_year"].isin(train_years).values

    X_all_tr, y_all_tr = data["X"][train_mask], data["y"][train_mask]
    X_test,   y_test   = data["X"][test_mask],  data["y"][test_mask]

    X_train, X_val, y_train, y_val = train_test_split(
        X_all_tr, y_all_tr, test_size=0.20, stratify=y_all_tr, random_state=RANDOM_SEED,
    )

    model, y_pred = train_and_evaluate_inception(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        feat_names=data["feat_names"],
        test_year=test_year, run_dir=run_dir,
        train_years=train_years, sensor=sensor, cm_dir=cm_dir,
    )

    # ── Save predictions for optional standalone NDVI filtering ───────
    test_ids = ids.loc[test_mask, ["id_parcel", "agri_year"]].reset_index(drop=True)
    predictions = test_ids.assign(y_true=y_test, y_pred=y_pred)
    pred_path = run_dir / f"predictions_{test_year}.csv"
    predictions.to_csv(pred_path, index=False)
    print(f"  [predictions] Saved --> {pred_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from config import S1_DIR, S2_DIR
    s1_files = sorted(S1_DIR.glob("S1_*.csv"))
    s2_files = sorted(S2_DIR.glob("S2_*.csv"))
    all_years, test_year, train_years = detect_years(s1_files, s2_files, required_data(SENSOR_MODES))
    print(f"Detected years : {all_years}  |  train={train_years}  |  test={test_year}")

    for sensor in SENSOR_MODES:
        print("\n" + "-" * 60)
        print(f"  InceptionTime — sensor : {sensor}")
        print("-" * 60)

        data = build_tensor(sensor, all_years, train_years)
        run_dir = (OUTPUT_DIR / "InceptionTime" / sensor /
                   f"train_{'_'.join(str(y) for y in train_years)}_test_{test_year}")

        run_one(data, test_year, train_years, run_dir, sensor=sensor)


if __name__ == "__main__":
    main()