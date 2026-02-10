import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

# ------------------------------
# Notes (short, human comments)
# ------------------------------
# This script evaluates the trained regression model on a test split.
# Output:
# - MSE / MAE / RMSE / R2 for train/val/test
# - binned confusion matrix (LOW/MEDIUM/HIGH) for interpretation
# - top 5 absolute errors with context columns
#
# Assumptions:
# - dataset exists at data/processed/nn_dataset.csv
# - model exists at data/processed/model.pth
# - features are the 3 numeric columns used in training
# - split ratios match training (70/15/15) with seed 42
#
# Why confusion matrix in a regression project:
# - evaluator wants an easy-to-read error summary
# - we discretize score using fixed thresholds
# - matrix shows if predictions cross category boundaries


CSV_PATH = os.path.join("data", "processed", "nn_dataset.csv")
MODEL_PATH = os.path.join("data", "processed", "model.pth")

SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15


class RiskMLP(nn.Module):
    def __init__(self, in_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int = 42):
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


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0


def risk_cat(r: float) -> str:
    if r < 0.40:
        return "LOW"
    if r < 0.70:
        return "MEDIUM"
    return "HIGH"


def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


@torch.no_grad()
def predict(model: nn.Module, Xn: np.ndarray) -> np.ndarray:
    xt = torch.tensor(Xn, dtype=torch.float32)
    out = model(xt).cpu().numpy().reshape(-1)
    return out


def main():
    set_seed(SEED)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing dataset: {CSV_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    # Load dataset and clean numeric fields.
    # We evaluate on the same split strategy as training.
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

    mn, mx = minmax_fit(X_train)
    X_train_n = minmax_transform(X_train, mn, mx)
    X_val_n = minmax_transform(X_val, mn, mx)
    X_test_n = minmax_transform(X_test, mn, mx)

    # Load trained model weights (saved by src/neural_network/train_nn.py)
    state = torch.load(MODEL_PATH, map_location="cpu")
    model = RiskMLP(in_dim=3)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.eval()

    pred_train = predict(model, X_train_n)
    pred_val = predict(model, X_val_n)
    pred_test = predict(model, X_test_n)

    m_train = metrics(y_train, pred_train)
    m_val = metrics(y_val, pred_val)
    m_test = metrics(y_test, pred_test)

    # Confusion matrix (binned):
    # We turn regression score into 3 categories for easy interpretation.
    true_cat = np.array([risk_cat(v) for v in y_test])
    pred_cat = np.array([risk_cat(v) for v in pred_test])
    classes = ["LOW", "MEDIUM", "HIGH"]
    cm = pd.crosstab(pd.Series(true_cat, name="true"), pd.Series(pred_cat, name="pred"), dropna=False)
    cm = cm.reindex(index=classes, columns=classes, fill_value=0)

    # Top errors help explain where the model deviates most.
    abs_err = np.abs(y_test - pred_test)
    top_idx_local = np.argsort(-abs_err)[:5]
    rows = df.iloc[test_idx[top_idx_local]].copy()

    out = rows[["plate", "county_code", "intersection", "interval_label", "time_range"]].copy()
    out["y_true"] = y_test[top_idx_local]
    out["y_pred"] = pred_test[top_idx_local]
    out["abs_err"] = abs_err[top_idx_local]
    out["true_cat"] = [risk_cat(v) for v in y_test[top_idx_local]]
    out["pred_cat"] = [risk_cat(v) for v in pred_test[top_idx_local]]

    print("sizes:", len(train_idx), len(val_idx), len(test_idx))
    print("train", m_train)
    print("val", m_val)
    print("test", m_test)
    print("\nconfusion (binned):\n", cm)
    print("\nTop5 abs errors:\n", out.to_string(index=False))


if __name__ == "__main__":
    main()
