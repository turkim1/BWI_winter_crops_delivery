"""
ml_model.py — model training, and evaluation.

----------
load_params() -> dict
    Returns BEST_PARAMS_XGB from config (with random_state).

train_and_evaluate(X_train, y_train, X_test, y_test, feat_cols_aug,
                   test_year, run_dir, plot_dir, cm_dir, train_years) -> (classifier, y_pred)
    
    Trains the final model, prints the classification report, saves
    artefacts, and returns the fitted classifier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import dump
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    BEST_PARAMS_XGB,
    RANDOM_SEED,
    CLASS_NAMES,
    MODEL_NAME,
    )
from plots import plot_confusion_matrix, plot_feature_importances, plot_feature_correlation


# -------------------------------------------------------------
# VALID PARAM KEYS PER MODEL
# -------------------------------------------------------------
_XGB_KEYS = {
    "n_estimators", "max_depth", "learning_rate", "subsample",
    "colsample_bytree", "min_child_weight", "gamma", "reg_alpha",
    "reg_lambda", "objective", "tree_method", "random_state"
}

# -------------------------------------------------------------
# MODEL FACTORY
# -------------------------------------------------------------

def _get_clf(params: dict) -> XGBClassifier:
    """Instantiate the correct classifier, keeping only valid keys."""
    if MODEL_NAME == "xgboost":
        return XGBClassifier(**{k: v for k, v in params.items() if k in _XGB_KEYS})

# -------------------------------------------------------------
# SHARED FIT HELPER
# -------------------------------------------------------------

def _fit(clf, X_tr: np.ndarray, y_tr: np.ndarray):
    """Fit any classifier with balanced sample weights."""
    clf.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
    return clf

# -------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------

def load_params() -> dict:
    """
    Return hyperparameters for the final model.
    BEST_PARAMS_XGB must always be set in config.py.
    """
    if MODEL_NAME == "xgboost":
        if BEST_PARAMS_XGB is None:
            raise ValueError("BEST_PARAMS_XGB must be set in config.py.")
        return {**BEST_PARAMS_XGB, "random_state": RANDOM_SEED}
    raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME!r}")


# -------------------------------------------------------------
# TRAINING & EVALUATION
# -------------------------------------------------------------

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feat_cols_aug: list[str],
    test_year: int,
    run_dir: Path,
    plot_dir: Path,
    cm_dir: Path,
    train_years: list[int],
) -> tuple[XGBClassifier, np.ndarray]:
    """
    1. Load hyperparameters.
    2. Fit the final model on the full training set.
    3. Evaluate on the test set and print the classification report.
    4. Save the model and plots.
    """
    best = load_params()
    
    print(f"\nTraining {MODEL_NAME} on all {len(feat_cols_aug)} features …")
    clf = _fit(_get_clf(best), X_train, y_train)

    importances = pd.Series(clf.feature_importances_, index=feat_cols_aug)
    print("\nTop 20 most important features:")
    print(importances.nlargest(20).to_string())

    feat_cols_final = feat_cols_aug
    X_train_eval, X_test_eval = X_train, X_test

    # ── Save model ───────────────────────────────────────────────────
    model_path = run_dir / f"{MODEL_NAME}_model_test_{test_year}.joblib"
    dump(clf, model_path)
    print(f"  [model] Saved --> {model_path}")

    # ── Evaluate ─────────────────────────────────────────────────────
    y_pred = clf.predict(X_test_eval)
    target_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]

    print(f"\n{'='*55}")
    print(f"  RESULTS — model : {MODEL_NAME} | test year : {test_year}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index  =[f"True  {n}" for n in target_names],
        columns=[f"Pred  {n}" for n in target_names],
    )
    print("Confusion matrix (raw counts):")
    print(cm_df.to_string())

    # ── Plots ────────────────────────────────────────────────────────
    plot_confusion_matrix(y_test, y_pred, target_names, test_year, plot_dir=cm_dir)
    plot_feature_importances(test_year, importances, plot_dir=plot_dir)
    plot_feature_correlation(X_train_eval, feat_cols_final, importances, plot_dir=plot_dir)

    return clf, y_pred