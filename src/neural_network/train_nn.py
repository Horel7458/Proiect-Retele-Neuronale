import os
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def find_project_root(start_dir: str) -> str:
    """Find project root by searching upward for a folder named 'data'."""
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, "data")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start_dir)
        cur = parent


PROJECT_ROOT = find_project_root(os.path.dirname(os.path.abspath(__file__)))

# =========================
# PATHS
# =========================
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "nn_dataset.csv")

OUT_MODEL = os.path.join(PROJECT_ROOT, "data", "processed", "model.pth")
OUT_SCALER = os.path.join(PROJECT_ROOT, "data", "processed", "nn_scaler.json")
OUT_PLOT = os.path.join(PROJECT_ROOT, "data", "processed", "train_curve.png")

SEED = 42

# Training summary (short comments):
# - reads data/processed/nn_dataset.csv
# - builds X (3 features) and y (label_risk)
# - splits into train/val/test with fixed seed
# - fits min-max scaler on train only
# - trains a small MLP regressor (sigmoid output)
# - saves model.pth and nn_scaler.json into data/processed/
# - optionally saves a training curve plot

# Notes:
# - this is a tiny dataset, so training is fast on CPU
# - we keep hyperparams simple (baseline is best here)


# =========================
# Utils
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_columns(df: pd.DataFrame, required: List[str]):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing column(s) in nn_dataset.csv: {missing}\n"
            f"Existing columns: {list(df.columns)}\n"
            f"Tip: verifica denumirile coloanelor din dataset."
        )


def pick_first_existing(df: pd.DataFrame, options: List[str], name_for_error: str) -> str:
    for c in options:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find required column for {name_for_error}. Tried: {options}\n"
        f"Existing columns: {list(df.columns)}"
    )


def train_val_test_split_idx(n: int, val_ratio=0.15, test_ratio=0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def minmax_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    return mn, mx


def minmax_transform(X: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    denom = (mx - mn)
    denom[denom == 0] = 1.0
    return (X - mn) / denom


# =========================
# Dataset
# =========================
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================
# Model
# =========================
class RiskMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Main
# =========================
def main():
    set_seed(SEED)

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] Missing: {CSV_PATH}")
        return

    # Load final dataset built by src/preprocessing/dataset_builder.py
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    # --- map column names (your CSV uses: accidente_intersectie, accidente_vehicul, scor_judet, label_risk)
    col_acc_i = pick_first_existing(df, ["acc_intersection", "accidente_intersectie"], "intersection accidents")
    col_acc_v = pick_first_existing(df, ["acc_vehicle", "accidente_vehicul"], "vehicle accidents")
    col_county = pick_first_existing(df, ["county_score", "scor_judet"], "county score")
    col_label = pick_first_existing(df, ["label", "label_risk", "risk", "target", "y"], "label")

    feature_cols = [col_acc_i, col_acc_v, col_county]

    ensure_columns(df, feature_cols + [col_label])

    # Numeric cleanup:
    # - coerce invalid strings to NaN
    # - drop incomplete rows (keeps training stable)
    for c in feature_cols + [col_label]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols + [col_label]).reset_index(drop=True)

    X = df[feature_cols].astype(float).values
    y = df[col_label].astype(float).values

    # Split indices (seeded) so results are reproducible.
    # Ratios: 70/15/15 (train/val/test)
    n = len(df)
    train_idx, val_idx, test_idx = train_val_test_split_idx(n, val_ratio=0.15, test_ratio=0.15)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Fit scaler only on train, then apply to val/test.
    # This avoids leaking info from val/test into train normalization.
    mn, mx = minmax_fit(X_train)
    X_train_n = minmax_transform(X_train, mn, mx)
    X_val_n = minmax_transform(X_val, mn, mx)
    X_test_n = minmax_transform(X_test, mn, mx)

    # save scaler in the same format your UI loader expects
    scaler = {
        "feature_cols": ["acc_intersection", "acc_vehicle", "county_score"],
        "min": {
            "acc_intersection": float(mn[0]),
            "acc_vehicle": float(mn[1]),
            "county_score": float(mn[2]),
        },
        "max": {
            "acc_intersection": float(mx[0]),
            "acc_vehicle": float(mx[1]),
            "county_score": float(mx[2]),
        }
    }

    os.makedirs(os.path.dirname(OUT_SCALER), exist_ok=True)
    with open(OUT_SCALER, "w", encoding="utf-8") as f:
        json.dump(scaler, f, indent=2)

    print(f"[INFO] Using CSV: {CSV_PATH}")
    print(f"[INFO] Features: {feature_cols}  | Label: {col_label}")
    print(f"[INFO] Train/Val/Test sizes: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
    print(f"[OK] Saved scaler: {OUT_SCALER}")

    # data loaders
    train_ds = TabDataset(X_train_n, y_train)
    val_ds = TabDataset(X_val_n, y_val)
    test_ds = TabDataset(X_test_n, y_test)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # Model/training:
    # - small MLP for tabular features
    # - MSE loss (regression)
    # - Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model = RiskMLP(in_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_val = float("inf")
    best_state = None

    train_curve = []
    val_curve = []

    # Training loop (kept simple, no early stopping needed on this dataset)
    EPOCHS = 120
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        # val
        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                va_losses.append(loss.item())

        tr = float(np.mean(tr_losses)) if tr_losses else 0.0
        va = float(np.mean(va_losses)) if va_losses else 0.0
        train_curve.append(tr)
        val_curve.append(va)

        if va < best_val:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch:>3}/{EPOCHS} | train MSE={tr:.6f} | val MSE={va:.6f}")

    # test
    model.load_state_dict(best_state)
    model.eval()
    te_losses = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            te_losses.append(loss.item())

    test_mse = float(np.mean(te_losses)) if te_losses else 0.0

    print(f"[RESULT] Best val MSE: {best_val:.6f}")
    print(f"[RESULT] Test MSE: {test_mse:.6f}")

    # save checkpoint in a format your UI loader supports (model_state + feature_cols)
    ckpt = {
        "model_state": model.state_dict(),
        "feature_cols": ["acc_intersection", "acc_vehicle", "county_score"]
    }

    torch.save(ckpt, OUT_MODEL)
    print(f"[OK] Saved model: {OUT_MODEL}")

    # plot
    if plt is not None:
        try:
            plt.figure()
            plt.plot(train_curve, label="train")
            plt.plot(val_curve, label="val")
            plt.xlabel("epoch")
            plt.ylabel("MSE")
            plt.legend()
            plt.tight_layout()
            plt.savefig(OUT_PLOT, dpi=150)
            print(f"[OK] Saved plot: {OUT_PLOT}")
        except Exception as e:
            print(f"[WARN] Plot failed: {e}")
    else:
        print("[INFO] matplotlib not available -> skipping plot.")


if __name__ == "__main__":
    main()
