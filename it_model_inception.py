"""
it_model_inception.py — InceptionTime architecture, training loop, and evaluation.

Public API
----------
train_and_evaluate_inception(X_train, y_train, X_test, y_test,
                             feat_names, test_year, run_dir, train_years)
"""
from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
from pathlib import Path

from ml_metrics_logger import log_trial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


from config import (
    RANDOM_SEED, CLASS_NAMES,
    IT_NB_FILTERS, IT_DEPTH, IT_KERNEL_SIZES, IT_BOTTLENECK_DIM,
    IT_N_EPOCHS, IT_BATCH_SIZE, IT_LR, IT_WEIGHT_DECAY,
    IT_EARLY_STOPPING_PATIENCE, IT_EARLY_STOPPING_MIN_DELTA,
    IT_LR_SCHEDULER, IT_LR_PATIENCE, IT_LR_FACTOR, IT_LR_MIN, OUTPUT_DIR
)
from plots import plot_confusion_matrix, plot_training_curves

# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────
# ARCHITECTURE
# ─────────────────────────────────────────────────────────────

class _InceptionModule(nn.Module):
    def __init__(self, in_channels: int, nb_filters: int,
                 kernel_sizes: list[int], bottleneck_channels: int):
        super().__init__()
        if in_channels > bottleneck_channels:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels,
                                        kernel_size=1, bias=False)
            conv_in = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            conv_in = in_channels

        self.convs = nn.ModuleList([
            nn.Conv1d(conv_in, nb_filters, kernel_size=k,
                      padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False),
        )
        self.bn   = nn.BatchNorm1d(nb_filters * (len(kernel_sizes) + 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        z        = self.bottleneck(x)
        branches = [conv(z) for conv in self.convs] + [self.maxpool_branch(x)]
        return self.relu(self.bn(torch.cat(branches, dim=1)))


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x, residual):
        return self.relu(x + self.shortcut(residual))


class InceptionTime(nn.Module):
    def __init__(self, in_channels: int, n_classes: int,
                 nb_filters: int = IT_NB_FILTERS, depth: int = IT_DEPTH,
                 kernel_sizes: list[int] | None = None,
                 bottleneck_channels: int = IT_BOTTLENECK_DIM):
        super().__init__()
        kernel_sizes  = kernel_sizes or IT_KERNEL_SIZES
        out_channels  = nb_filters * (len(kernel_sizes) + 1)
        self.depth    = depth

        self.inception_blocks = nn.ModuleList()
        self.residual_blocks  = nn.ModuleList()

        current_channels  = in_channels
        residual_channels = in_channels

        for d in range(depth):
            self.inception_blocks.append(
                _InceptionModule(current_channels, nb_filters,
                                 kernel_sizes, bottleneck_channels)
            )
            if (d + 1) % 3 == 0:
                self.residual_blocks.append(
                    _ResidualBlock(residual_channels, out_channels)
                )
                residual_channels = out_channels
            else:
                self.residual_blocks.append(None)
            current_channels = out_channels

        self.gap        = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(out_channels, n_classes)

    def forward(self, x):
        residual = x
        for d in range(self.depth):
            if d % 3 == 0:
                residual = x
            x = self.inception_blocks[d](x)
            if self.residual_blocks[d] is not None:
                x = self.residual_blocks[d](x, residual)
        return self.classifier(self.gap(x).squeeze(-1))


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────

def impute_tensor(X_train, X_val, X_test):
    medians = np.nanmedian(X_train, axis=0)

    def _fill(X):
        X = X.copy()
        idx_n, idx_c, idx_t = np.where(np.isnan(X))
        X[idx_n, idx_c, idx_t] = medians[idx_c, idx_t]
        return X

    return _fill(X_train), _fill(X_val), _fill(X_test)

def normalize_tensor(X_train: np.ndarray,
                     X_val:np.ndarray,
                     X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score per (channel, timestep), statistics from training set only."""
    mean = np.mean(X_train, axis=0, keepdims=True)
    std  = np.std (X_train, axis=0, keepdims=True)
    return ((X_train - mean) / (std + 1e-8),
            (X_val   - mean) / (std + 1e-8),
            (X_test  - mean) / (std + 1e-8))

# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def _run_epoch_train(model, loader, criterion, optimizer, scaler,
                     device, n_samples: int) -> float:
    model.train()
    running = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            loss = criterion(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * len(xb)
    return running / n_samples


def _run_epoch_val(model, loader, criterion, device,
                   n_samples: int) -> tuple[float, np.ndarray]:
    model.eval()
    running = 0.0
    preds   = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(xb)
                loss   = criterion(logits, yb)
            running += loss.item() * len(xb)
            preds.append(logits.argmax(dim=1).cpu())
    return running / n_samples, torch.cat(preds).numpy()


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def train_and_evaluate_inception(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    feat_names: list[str],
    test_year:  int,
    run_dir:    Path,
    train_years: list[int],
    sensor: str,
    cm_dir: Path | None = None,
) -> InceptionTime:
    """
    Impute --> normalise --> train InceptionTime --> evaluate.
    Saves model weights, training curves, and confusion matrix.
    Returns the fitted model.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = run_dir / "plots"
    cm_dir   = cm_dir or run_dir / "confusion_matrices"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"  X_train : {X_train.shape}  X_test : {X_test.shape}")

    # ── Imputation & normalisation ───────────────────────────
    X_train, X_val, X_test = impute_tensor(X_train, X_val, X_test)
    X_train, X_val, X_test = normalize_tensor(X_train, X_val, X_test)

    n_nan = np.isnan(X_train).sum() + np.isnan(X_test).sum()
    if n_nan:
        print(f"  [warning] {n_nan:,} NaNs remain after imputation")

    # ── DataLoaders ──────────────────────────────────────────
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(Xt, yt), batch_size=IT_BATCH_SIZE,
                              shuffle=True, num_workers=0,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(TensorDataset(Xv, yv), batch_size=IT_BATCH_SIZE,
                              shuffle=False, num_workers=0,
                              pin_memory=(device.type == "cuda"))

    # ── Class weights ────────────────────────────────────────
    classes = np.unique(y_train)
    cw      = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)

    # ── Model, loss, optimiser ───────────────────────────────
    model = InceptionTime(in_channels=X_train.shape[1],
                               n_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))


    # ── Training loop ────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    t0      = time.time()

    print("\nTraining InceptionTime" )

    optimizer = torch.optim.Adam(model.parameters(), lr=IT_LR, weight_decay=IT_WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-7)
    scheduler = (
        ReduceLROnPlateau(optimizer, mode="min", patience=IT_LR_PATIENCE,
                        factor=IT_LR_FACTOR, min_lr=IT_LR_MIN)
        if IT_LR_SCHEDULER else None
    )

    # ── Early stopping state ──────────────────────────────────────
    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    for epoch in range(1, IT_N_EPOCHS + 1):
        t_ep = time.time()

        train_loss          = _run_epoch_train(model, train_loader, criterion,
                                               optimizer, scaler, device, len(X_train))
        val_loss, y_pred_val = _run_epoch_val(model, val_loader, criterion,
                                             device, len(X_val))
        val_acc              = (y_pred_val == y_val).mean()

        if scheduler is not None:
            scheduler.step(val_loss)

        # ── Early stopping check ──────────────────────────────────
        improved = val_loss < best_val_loss - IT_EARLY_STOPPING_MIN_DELTA
        if improved:
            best_val_loss  = val_loss
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= IT_EARLY_STOPPING_PATIENCE:
                print(f"  [early stop] epoch {epoch}  best_val_loss={best_val_loss:.4f}")
                break


        
        history["train_loss"].append(train_loss)
        history["val_loss"]  .append(val_loss)
        history["val_acc"]   .append(val_acc)

        print(f"  Epoch {epoch:03d}/{IT_N_EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.4f}  ({time.time()-t_ep:.1f}s)")

        if device.type == "cuda":
            torch.cuda.empty_cache()
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  [restore] Best weights restored (val_loss={best_val_loss:.4f})")

    print(f"\n  Total training time: {time.time()-t0:.1f}s")

    # ── Recalcul des prédictions sur le test set avec les meilleurs poids ──
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                        torch.tensor(y_test,  dtype=torch.long)),
                            batch_size=IT_BATCH_SIZE, shuffle=False)
    _, y_pred = _run_epoch_val(model, test_loader, criterion, device, len(X_test))

    # ── Evaluation ───────────────────────────────────────────
    target_names = [CLASS_NAMES[c] for c in classes]

    print(f"\n{'='*55}")
    print(f"  RESULTS — InceptionTime | test year : {test_year}")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred,
                                 labels=classes, target_names=target_names))

    cm    = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm,
                         index  =[f"True  {n}" for n in target_names],
                         columns=[f"Pred  {n}" for n in target_names])
    print("Confusion matrix (raw counts):")
    print(cm_df.to_string())

    log_trial(
        output_dir     = OUTPUT_DIR / "InceptionTime",  # ← was run_dir
        sensor         = sensor,
        test_year      = test_year,
        train_years    = train_years,
        n_train        = len(X_train),
        n_test         = len(X_test),
        y_test         = y_test,
        y_pred         = y_pred,
        class_names    = CLASS_NAMES,
    )

    # ── Plots ────────────────────────────────────────────────
    plot_training_curves(history, test_year, plot_dir)
    plot_confusion_matrix(y_test, y_pred, target_names, test_year, plot_dir=cm_dir)

    # ── Save model ───────────────────────────────────────────
    model_path = run_dir / f"inception_model_test_{test_year}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  [model] Saved --> {model_path}")

    return model, y_pred

def fit_predict_inception(X_train, y_train, X_pred, run_dir=None) -> np.ndarray:
    """Train on X_train (random stratified 20% validation for early stopping / LR schedule)
    and return class probabilities (m, n_classes) for X_pred. Same settings as the report."""
    from sklearn.model_selection import train_test_split

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_SEED)
    X_tr, X_val, X_pred = impute_tensor(X_tr, X_val, X_pred)
    X_tr, X_val, X_pred = normalize_tensor(X_tr, X_val, X_pred)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    print(f"[device] {device} | train {X_tr.shape} | val {X_val.shape} | predict {X_pred.shape}")

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                            torch.tensor(y_tr, dtype=torch.long)),
                              batch_size=IT_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.long)),
                            batch_size=IT_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

    classes = np.unique(y_tr)
    cw = compute_class_weight("balanced", classes=classes, y=y_tr)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    model = InceptionTime(in_channels=X_tr.shape[1], n_classes=len(classes)).to(device)
    scaler = torch.amp.GradScaler(enabled=pin)
    optimizer = torch.optim.Adam(model.parameters(), lr=IT_LR, weight_decay=IT_WEIGHT_DECAY,
                                 betas=(0.9, 0.999), eps=1e-7)
    scheduler = (ReduceLROnPlateau(optimizer, mode="min", patience=IT_LR_PATIENCE,
                                   factor=IT_LR_FACTOR, min_lr=IT_LR_MIN)
                 if IT_LR_SCHEDULER else None)

    best_val_loss, best_state, patience_count = float("inf"), None, 0
    for epoch in range(1, IT_N_EPOCHS + 1):
        train_loss = _run_epoch_train(model, train_loader, criterion, optimizer,
                                      scaler, device, len(X_tr))
        val_loss, y_pred_val = _run_epoch_val(model, val_loader, criterion, device, len(X_val))
        if scheduler is not None:
            scheduler.step(val_loss)
        print(f"  Epoch {epoch:03d}/{IT_N_EPOCHS}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={(y_pred_val == y_val).mean():.4f}")
        if val_loss < best_val_loss - IT_EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= IT_EARLY_STOPPING_PATIENCE:
                print(f"  [early stop] epoch {epoch}  best_val_loss={best_val_loss:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_pred), 256):
            xb = torch.tensor(X_pred[i:i + 256], dtype=torch.float32).to(device)
            probs.append(torch.softmax(model(xb).float(), dim=1).cpu().numpy())

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), run_dir / "inception_model_predict.pt")
    return np.concatenate(probs)