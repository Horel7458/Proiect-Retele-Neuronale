import os
import json
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

OUT_CSV = os.path.join(OUT_DIR, "learning_curve.csv")
OUT_PNG = os.path.join(OUT_DIR, "learning_curve.png")


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# Dataset / Model
# =========================
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RiskMLP(nn.Module):
    def __init__(self, in_dim: int = 3):
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
# Helpers
# =========================
def pick_first_existing(df: pd.DataFrame, options: List[str], name_for_error: str) -> str:
    for c in options:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find required column for {name_for_error}. Tried: {options}\n"
        f"Existing columns: {list(df.columns)}"
    )


def split_indices(n: int, val_ratio=0.15, test_ratio=0.15, seed=42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return train_idx, val_idx, test_idx


def fit_minmax(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mn = X_train.min(axis=0)
    mx = X_train.max(axis=0)
    # avoid division by zero
    mx = np.where(mx - mn == 0, mn + 1.0, mx)
    return mn, mx


def transform_minmax(X: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    return (X - mn) / (mx - mn)


@torch.no_grad()
def mse_on_loader(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    crit = nn.MSELoss(reduction="sum")
    total = 0.0
    n = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = model(Xb)
        loss = crit(pred, yb)
        total += float(loss.item())
        n += int(yb.shape[0])
    return total / max(1, n)


def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = 120, lr: float = 0.003, batch_size: int = 64,
                seed: int = 42) -> Tuple[float, float]:
    """
    Returns: (train_mse, val_mse) at the end using best val checkpoint.
    """
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RiskMLP(in_dim=X_train.shape[1]).to(device)

    train_loader = DataLoader(TabDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TabDataset(X_val, y_val), batch_size=256, shuffle=False)

    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

        # checkpoint by val MSE
        val_mse = mse_on_loader(model, val_loader, device)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_mse = mse_on_loader(model, DataLoader(TabDataset(X_train, y_train), batch_size=256, shuffle=False), device)
    val_mse = mse_on_loader(model, val_loader, device)
    return float(train_mse), float(val_mse)


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing dataset: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    # Your dataset columns (from earlier): accidente_intersectie, accidente_vehicul, scor_judet, label_risk
    col_acc_i = pick_first_existing(df, ["acc_intersection", "accidente_intersectie"], "intersection accidents")
    col_acc_v = pick_first_existing(df, ["acc_vehicle", "accidente_vehicul"], "vehicle accidents")
    col_county = pick_first_existing(df, ["county_score", "scor_judet"], "county score")
    col_label = pick_first_existing(df, ["label", "label_risk", "label_risk", "risk", "target", "y"], "label")

    # numeric
    for c in [col_acc_i, col_acc_v, col_county, col_label]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[col_acc_i, col_acc_v, col_county, col_label]).reset_index(drop=True)

    X_all = df[[col_acc_i, col_acc_v, col_county]].astype(float).values
    y_all = df[col_label].astype(float).values

    train_idx, val_idx, _ = split_indices(len(df), val_ratio=0.15, test_ratio=0.15, seed=42)

    X_train_full = X_all[train_idx]
    y_train_full = y_all[train_idx]
    X_val = X_all[val_idx]
    y_val = y_all[val_idx]

    # Fractions of training set
    fracs = [0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
    results = []

    for frac in fracs:
        n_sub = max(10, int(len(X_train_full) * frac))
        X_sub = X_train_full[:n_sub]
        y_sub = y_train_full[:n_sub]

        # Fit scaler on current training subset (important!)
        mn, mx = fit_minmax(X_sub)
        X_sub_n = transform_minmax(X_sub, mn, mx)
        X_val_n = transform_minmax(X_val, mn, mx)

        train_mse, val_mse = train_model(
            X_sub_n, y_sub,
            X_val_n, y_val,
            epochs=120,
            lr=0.003,
            batch_size=64,
            seed=42
        )

        results.append({
            "train_fraction": frac,
            "train_size": n_sub,
            "train_mse": train_mse,
            "val_mse": val_mse
        })

        print(f"[LC] frac={frac:.2f} n={n_sub:>4} | train_mse={train_mse:.6f} | val_mse={val_mse:.6f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved: {OUT_CSV}")

    if plt is not None:
        plt.figure()
        plt.plot(out_df["train_size"], out_df["train_mse"], marker="o", label="train MSE")
        plt.plot(out_df["train_size"], out_df["val_mse"], marker="o", label="val MSE")
        plt.xlabel("Training set size")
        plt.ylabel("MSE")
        plt.title("Learning Curve (MSE vs Training Size)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=160)
        plt.close()
        print(f"[OK] Saved: {OUT_PNG}")
    else:
        print("[INFO] matplotlib not available -> plot skipped.")

    print("\nDONE. Open learning_curve.png and learning_curve.csv in data/processed.\n")


if __name__ == "__main__":
    main()
