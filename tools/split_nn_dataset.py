import os

import numpy as np
import pandas as pd


def find_project_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, "data")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return os.path.abspath(start_dir)
        cur = parent


def train_val_test_split_idx(n: int, val_ratio: float = 0.15, test_ratio: float = 0.15):
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return train_idx, val_idx, test_idx


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = find_project_root(script_dir)

    src_csv = os.path.join(project_root, "data", "processed", "nn_dataset.csv")
    if not os.path.isfile(src_csv):
        raise FileNotFoundError(f"Missing source dataset: {src_csv}")

    df = pd.read_csv(src_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    feature_cols = ["accidente_intersectie", "accidente_vehicul", "scor_judet"]
    label_col = "label_risk"

    for c in feature_cols + [label_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in {src_csv}. Existing: {list(df.columns)}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)

    np.random.seed(42)
    train_idx, val_idx, test_idx = train_val_test_split_idx(len(df), val_ratio=0.15, test_ratio=0.15)

    out_train_dir = os.path.join(project_root, "data", "train")
    out_val_dir = os.path.join(project_root, "data", "validation")
    out_test_dir = os.path.join(project_root, "data", "test")
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_val_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    train_csv = os.path.join(out_train_dir, "nn_dataset_train.csv")
    val_csv = os.path.join(out_val_dir, "nn_dataset_val.csv")
    test_csv = os.path.join(out_test_dir, "nn_dataset_test.csv")

    df.iloc[train_idx].to_csv(train_csv, index=False, encoding="utf-8")
    df.iloc[val_idx].to_csv(val_csv, index=False, encoding="utf-8")
    df.iloc[test_idx].to_csv(test_csv, index=False, encoding="utf-8")

    print("[OK] Wrote splits:")
    print(" -", train_csv, "rows=", len(train_idx))
    print(" -", val_csv, "rows=", len(val_idx))
    print(" -", test_csv, "rows=", len(test_idx))


if __name__ == "__main__":
    main()
