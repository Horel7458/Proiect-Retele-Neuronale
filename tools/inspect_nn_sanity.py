import json
import os
from typing import Sequence

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise SystemExit(f"Torch not available: {e}")


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(ROOT, "data", "processed", "nn_dataset.csv")
MODEL_PATH = os.path.join(ROOT, "data", "processed", "model.pth")
SCALER_PATH = os.path.join(ROOT, "data", "processed", "nn_scaler.json")


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


def pick_col(df: pd.DataFrame, options: Sequence[str]) -> str:
    for c in options:
        if c in df.columns:
            return c
    raise KeyError(f"Missing columns. Tried: {options}. Have: {list(df.columns)}")


def load_scaler() -> dict:
    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_features(X: np.ndarray, scaler: dict) -> np.ndarray:
    cols = scaler.get("feature_cols", ["acc_intersection", "acc_vehicle", "county_score"])
    mins = scaler.get("min", {})
    maxs = scaler.get("max", {})

    mn = np.array([float(mins.get(c, 0.0)) for c in cols], dtype=float)
    mx = np.array([float(maxs.get(c, 1.0)) for c in cols], dtype=float)
    denom = (mx - mn)
    denom[denom == 0] = 1.0
    return (X - mn) / denom


@torch.no_grad()
def predict(model: nn.Module, Xn: np.ndarray) -> np.ndarray:
    xt = torch.tensor(Xn, dtype=torch.float32)
    return model(xt).cpu().numpy().reshape(-1)


def load_model(in_dim: int) -> nn.Module:
    state = torch.load(MODEL_PATH, map_location="cpu")
    model = RiskMLP(in_dim=in_dim)
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"], strict=True)
    else:
        model.load_state_dict(state, strict=True)
    model.eval()
    return model


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def main() -> None:
    for p in (DATASET_PATH, MODEL_PATH, SCALER_PATH):
        if not os.path.exists(p):
            raise SystemExit(f"Missing required file: {p}")

    scaler = load_scaler()
    feat_cols = scaler.get("feature_cols", ["acc_intersection", "acc_vehicle", "county_score"])
    print("scaler.feature_cols:", feat_cols)

    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    # Dataset typically uses Romanian column names
    c_i = pick_col(df, ["accidente_intersectie", "acc_intersection"])  # intersection accidents
    c_v = pick_col(df, ["accidente_vehicul", "acc_vehicle"])          # vehicle accidents
    c_c = pick_col(df, ["scor_judet", "county_score"])               # county score
    c_y = pick_col(df, ["label_risk", "label", "risk", "target", "y"])

    for c in (c_i, c_v, c_c, c_y):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[c_i, c_v, c_c, c_y]).reset_index(drop=True)

    X = df[[c_i, c_v, c_c]].astype(float).values
    y = df[c_y].astype(float).to_numpy(dtype=float)

    Xn = normalize_features(X, scaler)

    # Out-of-range check (extrapolation)
    below0 = float(np.mean(Xn < 0.0))
    above1 = float(np.mean(Xn > 1.0))
    print(f"normalized out-of-range: <0: {below0*100:.2f}%  >1: {above1*100:.2f}%")

    model = load_model(in_dim=len(feat_cols))
    yp = predict(model, Xn)

    print("pred stats: n=", len(yp), "min=", float(yp.min()), "mean=", float(yp.mean()), "max=", float(yp.max()))
    print("label stats: n=", len(y), "min=", float(np.min(y)), "mean=", float(np.mean(y)), "max=", float(np.max(y)))

    print("mse(all):", mse(y, yp))

    # Quick spot-check: print 8 samples with (features, y, yhat)
    idx = np.linspace(0, len(df) - 1, num=min(8, len(df)), dtype=int)
    print("\nSample rows:")
    for i in idx:
        xi = X[i]
        print(f"  X=[{xi[0]:.1f}, {xi[1]:.1f}, {xi[2]:.3f}]  y={y[i]:.3f}  yhat={yp[i]:.3f}")


if __name__ == "__main__":
    main()
