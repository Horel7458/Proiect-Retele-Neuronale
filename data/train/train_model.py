import os
import json
import math
import random
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------- CONFIG ----------------
SEED = 42
EPOCHS = 120
BATCH_SIZE = 64
LR = 0.001
# --------------------------------------


def find_project_root(start_path: str) -> str:
    cur = os.path.abspath(start_path)
    while True:
        if os.path.isdir(os.path.join(cur, "data")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start_path)
        cur = parent


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RiskDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim=3):
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


def minmax_fit(train_df, cols):
    mins = train_df[cols].min().to_dict()
    maxs = train_df[cols].max().to_dict()
    for c in cols:
        if float(maxs[c]) == float(mins[c]):
            maxs[c] = float(mins[c]) + 1.0
    return mins, maxs


def minmax_transform(df, cols, mins, maxs):
    out = df.copy()
    for c in cols:
        out[c] = (out[c] - mins[c]) / (maxs[c] - mins[c])
    return out


def split_df(df, train=0.7, val=0.15, test=0.15):
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train)
    n_val = int(n * val)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    return train_df, val_df, test_df


def main():
    set_seed(SEED)

    # --- robust paths ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(script_dir)

    csv_path = os.path.join(project_root, "data", "processed", "nn_dataset_labeled.csv")
    out_model = os.path.join(project_root, "data", "processed", "model.pth")
    out_scaler = os.path.join(project_root, "data", "processed", "nn_scaler.json")

    if not os.path.isfile(csv_path):
        print("[ERROR] Missing:", csv_path)
        print("TIP: Make sure nn_dataset.csv exists in data/processed/")
        return

    df = pd.read_csv(csv_path)

    feature_cols = ["accidente_intersectie", "accidente_vehicul", "scor_judet"]
    label_col = "label_risk"

    df = df.dropna(subset=feature_cols + [label_col]).copy()

    train_df, val_df, test_df = split_df(df)

    mins, maxs = minmax_fit(train_df, feature_cols)
    train_df = minmax_transform(train_df, feature_cols, mins, maxs)
    val_df = minmax_transform(val_df, feature_cols, mins, maxs)
    test_df = minmax_transform(test_df, feature_cols, mins, maxs)

    os.makedirs(os.path.dirname(out_scaler), exist_ok=True)
    with open(out_scaler, "w", encoding="utf-8") as f:
        json.dump({"mins": mins, "maxs": maxs, "feature_cols": feature_cols}, f, indent=2)

    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[label_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[label_col].values

    train_loader = DataLoader(RiskDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RiskDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(in_dim=len(feature_cols)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = math.inf
    best_state = None

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Using CSV: {csv_path}")
    print(f"[INFO] Train/Val/Test sizes: {len(train_df)} / {len(val_df)} / {len(test_df)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * Xb.size(0)

        train_loss /= len(train_df)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * Xb.size(0)
        val_loss /= len(val_df)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | train MSE={train_loss:.6f} | val MSE={val_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    with torch.no_grad():
        pred_test = model(X_test_t)
        test_mse = criterion(pred_test, y_test_t).item()

    print(f"[RESULT] Best val MSE: {best_val:.6f}")
    print(f"[RESULT] Test MSE: {test_mse:.6f}")

    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    torch.save({"model_state": best_state, "feature_cols": feature_cols}, out_model)

    print("[OK] Saved model:", out_model)
    print("[OK] Saved scaler:", out_scaler)


if __name__ == "__main__":
    main()
