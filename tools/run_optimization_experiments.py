import os
import time
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


CSV_PATH = os.path.join("data", "processed", "nn_dataset.csv")
OUT_CSV = os.path.join("data", "processed", "optimization_experiments.csv")

SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15


@dataclass(frozen=True)
class Experiment:
    name: str
    hidden1: int
    hidden2: int
    dropout: float
    lr: float
    batch_size: int
    epochs: int
    weight_decay: float = 0.0


class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RiskMLP(nn.Module):
    def __init__(self, in_dim: int, h1: int, h2: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, h1), nn.ReLU()]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(h1, h2), nn.ReLU()]
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(h2, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_val_test_split_idx(n: int, val_ratio=0.15, test_ratio=0.15):
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return train_idx, val_idx, test_idx


def minmax_fit(X: np.ndarray):
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    return mn, mx


def minmax_transform(X: np.ndarray, mn: np.ndarray, mx: np.ndarray):
    denom = (mx - mn)
    denom[denom == 0] = 1.0
    return (X - mn) / denom


@torch.no_grad()
def mse_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    crit = nn.MSELoss(reduction="sum")
    total = 0.0
    n = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = model(Xb)
        total += float(crit(pred, yb).item())
        n += int(yb.shape[0])
    return total / max(1, n)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    return model(xt).detach().cpu().numpy().reshape(-1)


def run_experiment(exp: Experiment, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RiskMLP(in_dim=X_train.shape[1], h1=exp.hidden1, h2=exp.hidden2, dropout=exp.dropout).to(device)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=exp.lr, weight_decay=exp.weight_decay)

    train_loader = DataLoader(TabDataset(X_train, y_train), batch_size=exp.batch_size, shuffle=True)
    val_loader = DataLoader(TabDataset(X_val, y_val), batch_size=256, shuffle=False)

    best_val = float("inf")
    best_state = None

    t0 = time.perf_counter()
    for _ in range(exp.epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()

        val_mse = mse_on_loader(model, val_loader, device)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    t1 = time.perf_counter()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics
    pred_train = predict(model, X_train, device)
    pred_val = predict(model, X_val, device)
    pred_test = predict(model, X_test, device)

    def pack(y_true, y_pred):
        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        return mse, mae, rmse, r2

    tr_mse, tr_mae, tr_rmse, tr_r2 = pack(y_train, pred_train)
    va_mse, va_mae, va_rmse, va_r2 = pack(y_val, pred_val)
    te_mse, te_mae, te_rmse, te_r2 = pack(y_test, pred_test)

    return {
        "experiment": exp.name,
        "arch": f"{X_train.shape[1]}-{exp.hidden1}-{exp.hidden2}-1",
        "dropout": exp.dropout,
        "lr": exp.lr,
        "batch_size": exp.batch_size,
        "epochs": exp.epochs,
        "weight_decay": exp.weight_decay,
        "time_sec": float(t1 - t0),
        "train_mse": tr_mse,
        "val_mse": va_mse,
        "test_mse": te_mse,
        "train_mae": tr_mae,
        "val_mae": va_mae,
        "test_mae": te_mae,
        "test_r2": te_r2,
    }


def main():
    set_seed(SEED)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing dataset: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    feature_cols = ["accidente_intersectie", "accidente_vehicul", "scor_judet"]
    label_col = "label_risk"

    for c in feature_cols + [label_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)

    X = df[feature_cols].astype(float).values
    y = df[label_col].astype(float).values

    train_idx, val_idx, test_idx = train_val_test_split_idx(len(df), VAL_RATIO, TEST_RATIO)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Normalize based on train
    mn, mx = minmax_fit(X_train)
    X_train_n = minmax_transform(X_train, mn, mx)
    X_val_n = minmax_transform(X_val, mn, mx)
    X_test_n = minmax_transform(X_test, mn, mx)

    # Experiments: baseline + 4 variations
    exps = [
        Experiment(name="Baseline", hidden1=16, hidden2=8, dropout=0.0, lr=0.003, batch_size=64, epochs=120, weight_decay=0.0),
        Experiment(name="Exp1_lr_0.001", hidden1=16, hidden2=8, dropout=0.0, lr=0.001, batch_size=64, epochs=120, weight_decay=0.0),
        Experiment(name="Exp2_lr_0.01", hidden1=16, hidden2=8, dropout=0.0, lr=0.01, batch_size=64, epochs=120, weight_decay=0.0),
        Experiment(name="Exp3_bigger_net", hidden1=32, hidden2=16, dropout=0.0, lr=0.003, batch_size=64, epochs=120, weight_decay=0.0),
        Experiment(name="Exp4_dropout_0.2", hidden1=16, hidden2=8, dropout=0.2, lr=0.003, batch_size=64, epochs=120, weight_decay=0.0),
    ]

    rows = []
    for exp in exps:
        print(f"[RUN] {exp.name} ...")
        set_seed(SEED)
        rows.append(run_experiment(exp, X_train_n, y_train, X_val_n, y_val, X_test_n, y_test))
        print(
            f"  test_mse={rows[-1]['test_mse']:.8f} test_mae={rows[-1]['test_mae']:.6f} test_r2={rows[-1]['test_r2']:.6f} time={rows[-1]['time_sec']:.2f}s"
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("test_mse", ascending=True).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved: {OUT_CSV}")
    print("\nTop results:\n", out[["experiment", "arch", "dropout", "lr", "batch_size", "test_mse", "test_mae", "test_r2"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
