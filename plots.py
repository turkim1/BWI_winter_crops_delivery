"""
plots.py — All diagnostic and result plots.

Every public function accepts an optional `plot_dir` keyword argument.
When provided the figure is saved there; when omitted it is only shown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

from config import CLASS_NAMES, AGG_FUNCS, IT_HARMONIC_STEP_DAYS, IT_HARMONIC_N_PER_SEASON, RANDOM_SEED
from utils import to_db

_PALETTE = {
    "Winter crops":     "steelblue",
    "CZH":              "tomato",
    "Other vegetation": "seagreen",
}

_MONTH_ORDER = ["m09","m10","m11","m12","m01","m02","m03","m04","m05","m06"]


# -------------------------------------------------------------
# INTERNAL HELPERS
# -------------------------------------------------------------

def _save(plot_dir: Path | None, filename: str) -> None:
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        path = plot_dir / filename
        plt.savefig(path, dpi=150)
        print(f"  [plot] Saved --> {path}")
    plt.close()



# -------------------------------------------------------------
# CLASS DISTRIBUTION
# -------------------------------------------------------------

def plot_class_distribution(
    features: pd.DataFrame,
    all_years: list[int],
    plot_dir: Path | None = None,
) -> None:
    _, axes = plt.subplots(1, len(all_years) + 1, figsize=(14, 4))

    for ax, yr in zip(axes, all_years):
        counts = (features[features["agri_year"] == yr]["label"]
                  .value_counts().rename(index=CLASS_NAMES))
        counts.plot.bar(ax=ax, color=[_PALETTE[n] for n in counts.index], edgecolor="k")
        ax.set_title(f"Year {yr}")
        ax.set_ylabel("Parcels")
        ax.tick_params(axis="x", rotation=25)

    counts_all = features["label"].value_counts().rename(index=CLASS_NAMES)
    counts_all.plot.bar(ax=axes[-1], color=[_PALETTE[n] for n in counts_all.index], edgecolor="k")
    axes[-1].set_title("All years")
    axes[-1].tick_params(axis="x", rotation=25)

    plt.suptitle("Class distribution per year", fontweight="bold")
    plt.tight_layout()
    _save(plot_dir, "class_distribution.png")
# -------------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------------

def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    test_year: int,
    plot_dir: Path | None = None,
) -> None:
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    _, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (normalised) — test year {test_year}")
    plt.tight_layout()
    _save(plot_dir, f"confusion_matrix_{test_year}.png")


# -------------------------------------------------------------
# FEATURE IMPORTANCES
# -------------------------------------------------------------

def plot_feature_importances(
    test_year: int,
    importances: pd.Series,
    top_n: int = 25,
    plot_dir: Path | None = None,
) -> None:
    top = importances.nlargest(top_n)
    _, ax = plt.subplots(figsize=(9, 6))
    top[::-1].plot.barh(ax=ax, color="steelblue", edgecolor="k")
    ax.set_title(f"Top {top_n} feature importances \n {test_year}")
    ax.set_xlabel("Mean decrease in impurity")
    plt.tight_layout()
    _save(plot_dir, f"feature_importances_{test_year}.png")


# -------------------------------------------------------------
# FEATURE CORRELATION HEATMAP
# -------------------------------------------------------------

def plot_feature_correlation(
    X_train: np.ndarray,
    feat_cols: list[str],
    importances: pd.Series,
    top_n: int = 20,
    plot_dir: Path | None = None,
) -> None:
    top_cols = importances.nlargest(top_n).index.tolist()
    idx      = [feat_cols.index(c) for c in top_cols]
    df_top   = pd.DataFrame(X_train[:, idx], columns=top_cols)

    corr = df_top.corr()

    # Lower-triangle mask — but keep the diagonal so edge labels get a cell
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    _, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        xticklabels=top_cols, yticklabels=top_cols,
        linewidths=0.4, linecolor="white",
        ax=ax,
    )
    ax.set_title(f"Feature correlation — top {top_n} by importance")
    # Rotate so long names don't overlap
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=8)
    plt.tight_layout()
    _save(plot_dir, "feature_correlation.png")


# -------------------------------------------------------------
# S1 BACKSCATTER TIME SERIES  (VV and VH as separate plots, in dB)
# -------------------------------------------------------------

def _backscatter_timeseries(
    features: pd.DataFrame,
    train_years: list[int],
    test_year: int,
    pol: str,                   # "VV" or "VH"
    plot_dir: Path | None,
) -> None:
    """Internal helper — one polarisation at a time."""
    month_cols = [m for m in _MONTH_ORDER if f"s1_{pol}_{AGG_FUNCS[0]}_{m}" in features.columns]
    if not month_cols:
        print(f"  [plot] No S1 {pol} columns found — skipping.")
        return

    def _collect(years, label_suffix):
        rows = []
        for yr in years:
            sub = features[features["agri_year"] == yr]
            for cls_id, cls_name in CLASS_NAMES.items():
                cls_sub = sub[sub["label"] == cls_id]
                for m in month_cols:
                    col = f"s1_{pol}_{AGG_FUNCS[0]}_{m}"
                    if col in cls_sub.columns:
                        rows.append({
                            "month": m,
                            "class": cls_name,
                            "val":   cls_sub[col].mean(skipna=True),
                            "split": label_suffix,
                        })
        return rows

    rows = _collect(train_years, "train") + _collect([test_year], "test")
    df   = pd.DataFrame(rows)
    df["month"] = pd.Categorical(df["month"], categories=month_cols, ordered=True)
    df   = df.sort_values("month")

    # Average train years into one line per class
    df_train = (df[df["split"] == "train"]
                  .groupby(["month", "class"])["val"].mean().reset_index())
    df_test  =  df[df["split"] == "test"].copy()

    # Convert to dB for display only
    df_train["val_db"] = to_db(df_train["val"])
    df_test["val_db"]  = to_db(df_test["val"])

    _, ax = plt.subplots(figsize=(10, 5))
    for cls_name in CLASS_NAMES.values():
        c  = _PALETTE[cls_name]
        tr = df_train[df_train["class"] == cls_name]
        te = df_test[df_test["class"]   == cls_name]
        ax.plot(tr["month"], tr["val_db"], marker="o",  color=c,
                linestyle="-",  label=f"{cls_name} (train avg)")
        ax.plot(te["month"], te["val_db"], marker="s",  color=c,
                linestyle="--", label=f"{cls_name} (test {test_year})")

    ax.set_title(f"Mean {pol} time series by class [dB] — test year {test_year}")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Mean {pol} (dB)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(plot_dir, f"s1_{pol.lower()}_timeseries_{test_year}.png")


def plot_VV_VH_timeseries(
    features: pd.DataFrame,
    train_years: list[int],
    test_year: int,
    plot_dir: Path | None = None,
) -> None:
    """Produce one plot for VV and one for VH (both in dB)."""
    _backscatter_timeseries(features, train_years, test_year, "VV", plot_dir)
    _backscatter_timeseries(features, train_years, test_year, "VH", plot_dir)


# -------------------------------------------------------------
# S2 INDEX TIME SERIES  (one plot per index)
# -------------------------------------------------------------

def plot_ndvi_timeseries(
    features: pd.DataFrame,
    train_years: list[int],
    test_year: int,
    plot_dir: Path | None = None,
) -> None:
    """Plot a time series for every S2 index present, overlaying test year."""
    available_indices = {
        idx for idx in ["NDVI", "NDMI", "NDRE", "PSRI", "NDYI"] # add / remove indices
        + [f"B{b}" for b in ["2","3","4","5","6","7","8","8A","11","12"]]
        if any(f"s2_{idx}_{AGG_FUNCS[0]}_{m}" in features.columns for m in _MONTH_ORDER)
    }

    if not available_indices:
        print("  [plot] No S2 feature columns found — skipping S2 time-series plots.")
        return

    for index_name in sorted(available_indices):
        month_cols = [
            m for m in _MONTH_ORDER
            if f"s2_{index_name}_{AGG_FUNCS[0]}_{m}" in features.columns
        ]
        if not month_cols:
            continue

        def _collect(years, split_label):
            rows = []
            for yr in years:
                sub = features[features["agri_year"] == yr]
                for cls_id, cls_name in CLASS_NAMES.items():
                    cls_sub = sub[sub["label"] == cls_id]
                    for m in month_cols:
                        col = f"s2_{index_name}_{AGG_FUNCS[0]}_{m}"
                        if col in cls_sub.columns:
                            rows.append({
                                "month": m,
                                "class": cls_name,
                                "val":   cls_sub[col].mean(skipna=True),
                                "split": split_label,
                            })
            return rows

        rows = _collect(train_years, "train") + _collect([test_year], "test")
        if not rows:
            continue

        df = pd.DataFrame(rows)
        df["month"] = pd.Categorical(df["month"], categories=month_cols, ordered=True)
        df = df.sort_values("month")

        df_train = df[df["split"] == "train"].groupby(["month","class"])["val"].mean().reset_index()
        df_test  = df[df["split"] == "test"].copy()

        _, ax = plt.subplots(figsize=(10, 5))
        for cls_name in CLASS_NAMES.values():
            c  = _PALETTE[cls_name]
            tr = df_train[df_train["class"] == cls_name]
            te = df_test[df_test["class"]   == cls_name]
            ax.plot(tr["month"], tr["val"], marker="o",  color=c,
                    linestyle="-",  label=f"{cls_name} (train avg)")
            ax.plot(te["month"], te["val"], marker="s",  color=c,
                    linestyle="--", label=f"{cls_name} (test {test_year})")

        ax.set_title(f"Mean {index_name} time series — test year {test_year}")
        ax.set_xlabel("Month")
        ax.set_ylabel(f"Mean {index_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(plot_dir, f"s2_{index_name.lower()}_timeseries.png")


def plot_harmonic_timeseries(
    long: pd.DataFrame,
    sensor: str,
    feat_cols: list[str],
    agri_year: int,
    plot_dir: Path,
    n_parcels: int = 3,
    step: int = IT_HARMONIC_STEP_DAYS,
    n_harmonics_per_season: int = IT_HARMONIC_N_PER_SEASON,
    random_state: int = RANDOM_SEED,
) -> None:
    """
    For each class, pick one representative parcel in *agri_year* and plot
    for each channel: original acquisition points, continuous harmonic fit,
    and resampled grid points.

    One figure per class, saved to
    plot_dir/harmonic_<sensor>_<year>_class<label>.png
    """
    from it_harmonic import fit_harmonic, season_grid

    plot_dir.mkdir(parents=True, exist_ok=True)

    year_df = long[long["agri_year"] == agri_year]

    t0         = pd.Timestamp(agri_year - 1, 9, 1)
    t1         = pd.Timestamp(agri_year,     6, 30)
    season_length = float((t1 - t0).days)
    grid_off   = season_grid(agri_year, step)
    grid_dates = [t0 + pd.Timedelta(days=int(d)) for d in grid_off]
    dense_off  = np.linspace(0, (t1 - t0).days, 500)
    dense_dates= [t0 + pd.Timedelta(days=int(d)) for d in dense_off]

    n_channels = len(feat_cols)

    for label_id, class_name in CLASS_NAMES.items():
        cls_df  = year_df[year_df["label"] == label_id]
        parcels = cls_df["id_parcel"].unique()

        if len(parcels) == 0:
            print(f"  [plot] No parcels for class '{class_name}' in {agri_year} — skipping.")
            continue

        # Pick the parcel with the most observations (most data for a clean fit),
        # with a random tiebreak so it isn't always the same parcel
        obs_counts = cls_df.groupby("id_parcel")["date"].count()
        max_obs    = obs_counts.max()
        candidates = obs_counts[obs_counts == max_obs].index.tolist()
        rng_tb     = np.random.default_rng(random_state + label_id)
        pid        = rng_tb.choice(candidates)

        grp      = cls_df[cls_df["id_parcel"] == pid].sort_values("date")
        xm       = np.array([(d - t0).days
                             for d in pd.DatetimeIndex(grp["date"].values)],
                            dtype=float)

        n_harm   = n_harmonics_per_season

        color    = _PALETTE.get(class_name, "gray")

        fig, axes = plt.subplots(n_channels, 1,
                                 figsize=(12, 3 * n_channels),
                                 sharex=True)
        if n_channels == 1:
            axes = [axes]

        fig.suptitle(
            f"{sensor} — class: {class_name}  |  year {agri_year}  |  parcel {pid}",
            fontsize=11, y=1.01,
        )

        for ax, col in zip(axes, feat_cols):
            raw = pd.Series(grp[col].values.astype(float))
            ym = to_db(raw, nan_on_nonpositive=False) if sensor.startswith("S1") else raw.values

            ax.scatter(grp["date"].values, ym,
                       s=18, alpha=0.7, zorder=3, color=color,
                       label="Observations")

            fn = fit_harmonic(xm, ym, n_harm, season_length)
            if fn is not None:
                ax.plot(dense_dates, fn(dense_off),
                        linewidth=1.5, alpha=0.85, zorder=2, color=color,
                        label=f"Harmonic fit (n={n_harm})")
                ax.scatter(grid_dates, fn(grid_off),
                           marker="x", s=40, linewidths=1.8, color=color,
                           zorder=4, label=f"Resampled (Δ={step}d)")
            else:
                ax.text(0.5, 0.5, "fit failed — too few valid obs",
                        transform=ax.transAxes, ha="center", color="red")

            ax.set_ylabel(col, fontsize=14)

            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="upper right")

        axes[-1].set_xlabel("Date")
        fig.autofmt_xdate(rotation=30)
        plt.tight_layout()

        out = plot_dir / f"harmonic_{sensor}_{agri_year}_class{label_id}_{class_name.replace(' ', '_')}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [plot] Harmonic timeseries --> {out}")


def plot_training_curves(history: dict, test_year: int, plot_dir: Path) -> None:
    """Loss and accuracy curves over epochs — InceptionTime training."""
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss — test year {test_year}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_acc"], color="tab:green", label="Val accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Validation accuracy — test year {test_year}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = plot_dir / f"training_curves_{test_year}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [plot] Training curves --> {path}")